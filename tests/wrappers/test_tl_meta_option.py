from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, override

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium import spaces
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionPipelineConfigReader
from hrl_tl.wrappers.low_level_policies.base import (
    LowLevelPolicy,
    LowLevelPolicyBuffer,
)
from hrl_tl.wrappers.tl_meta_option import (
    TLMetaOptionPrimitiveStepTimeLimitWrapper,
    TLMetaOptionWrapper,
)
from hrl_tl.wrappers.utils.spec_rep import SpecRep


class DummyEnv(gym.Env[np.ndarray, int]):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.step_count = 0

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    @override
    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        obs = np.array(
            [0.1 * self.step_count, 0.1 * self.step_count], dtype=np.float32
        )
        reward = 1.0
        return obs, reward, False, False, {}

    @override
    def render(self) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.uint8)


class DummySpecRep(SpecRep[int]):
    def __init__(self) -> None:
        super().__init__(predicate_names=["a", "b"], num_clauses=1)
        self._action_space = spaces.Discrete(5)

    @property
    @override
    def action_space(self) -> spaces.Space[int]:
        return self._action_space

    @override
    def weights2tl(self, tl_weights: Any) -> str:
        mapping = {
            0: "F(a)",
            1: "F(b)",
            2: "0",
            3: "invalid_spec",
            4: "terminated_spec",
        }
        return mapping.get(int(tl_weights), "")

    @override
    def weights2ltl(self, tl_weights: Any) -> str:
        return self.weights2tl(tl_weights)

    @override
    def action2policy_args(self, action: Any) -> dict[str, Any]:
        return {"target": int(action)}


class DummyLowLevelPolicy(LowLevelPolicy[Any, Any, np.ndarray, int]):
    buffer_class = LowLevelPolicyBuffer

    @override
    def define_policy(self, policy_args: Any) -> Any:
        return None

    @property
    @override
    def is_aut_terminated(self) -> bool:
        return self.tl_spec == "terminated_spec"

    @override
    def update_env(
        self,
        current_env: gym.Env[Any, Any],
        obs: np.ndarray,
        info: dict[str, Any],
        tl_wrapper_args: dict[str, Any],
    ) -> None:
        pass

    @override
    def predict(
        self,
        current_env: gym.Env[Any, Any],
        obs: np.ndarray,
        info: dict[str, Any],
        tl_wrapper_args: dict[str, Any],
        excluded_obs_keys: list[str] | None = None,
    ) -> tuple[int, bool, bool]:
        self.policy_step += 1
        ll_terminated = self.policy_step >= 3
        return 1, ll_terminated, False

    @override
    def act(
        self,
        obs: Any,
        info: dict[str, Any] | None = None,
        current_env: gym.Env[Any, Any] | None = None,
        tl_wrapper_args: dict[str, Any] | None = None,
    ) -> int:
        return 1

    @override
    def delete_policy(self) -> None:
        pass


class TLMetaOptionWrapperTest(parameterized.TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.formulae_path = Path(self.temp_dir.name) / "formulae.json"
        data = {"specifications": ["F(a)", "F(b)", "terminated_spec"]}
        with self.formulae_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        self.env = DummyEnv()
        self.spec_rep = DummySpecRep()
        self.wrapper = TLMetaOptionWrapper(
            env=self.env,
            spec_rep=self.spec_rep,
            low_level_policy_class=DummyLowLevelPolicy,
            all_formulae_file_path=str(self.formulae_path),
            reward_type="smdp",
            gamma=0.9,
        )

    @override
    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        super().tearDown()

    def test_obs_space_has_no_step_count(self) -> None:
        self.assertEqual(
            self.wrapper.observation_space, self.env.observation_space
        )
        obs, _ = self.wrapper.reset()
        self.assertNotIn("step_count", obs)

    @parameterized.named_parameters(
        ("zero_formula", 2),
        ("unregistered_formula", 3),
        ("terminated_automaton", 4),
    )
    def test_invalid_or_terminated_spec_triggers_random_primitive_action(
        self, action: int
    ) -> None:
        self.wrapper.reset()
        _, _, _, _, info = self.wrapper.step(np.array(action, dtype=np.int64))
        self.assertTrue(info["invalid_option"])
        self.assertEqual(info["meta_option_steps"], 1)

    def test_macro_option_smdp_discounted_reward(self) -> None:
        self.wrapper.reset()
        _, reward, _, _, info = self.wrapper.step(np.array(0, dtype=np.int64))
        self.assertFalse(info["invalid_option"])
        self.assertEqual(info["meta_option_steps"], 3)
        self.assertAlmostEqual(float(reward), 2.71, places=5)

    def test_primitive_step_time_limit_truncation(self) -> None:
        composite = TLMetaOptionPrimitiveStepTimeLimitWrapper(
            env=self.env,
            max_episode_steps=5,
            spec_rep=self.spec_rep,
            low_level_policy_class=DummyLowLevelPolicy,
            all_formulae_file_path=str(self.formulae_path),
        )
        composite.reset()
        _, _, _, truncated1, _ = composite.step(np.array(0, dtype=np.int64))
        self.assertFalse(truncated1)

        _, _, _, truncated2, info2 = composite.step(np.array(1, dtype=np.int64))
        self.assertTrue(truncated2)
        self.assertTrue(info2.get("TimeLimit.truncated", False))

    def test_record_render_frames_captures_all_primitive_steps(self) -> None:
        self.wrapper.reset()
        _, _, _, _, info_default = self.wrapper.step(
            np.array(0, dtype=np.int64)
        )
        self.assertNotIn("render_frames", info_default)

        self.wrapper.set_record_render_frames(True)
        _, _, _, _, info = self.wrapper.step(np.array(0, dtype=np.int64))
        self.assertIn("render_frames", info)
        self.assertEqual(len(info["render_frames"]), 3)


class TLMetaOptionPipelineConfigReaderTest(parameterized.TestCase):
    def test_model_config_reader_resolves_primitive_step_ppo(self) -> None:
        reader = TLMetaOptionPipelineConfigReader.model_construct(
            config_dir="configs/zone",
            model_config_file="rl/ppo_dict_512_20M.yaml",
        )
        model_reader = reader._to_model_config_reader()
        self.assertEqual(
            model_reader.algo_config.algorithm,
            "sb3_hrl.option.policies.primitive_step_ppo.PrimitiveStepPPO",
        )
        algo_config = model_reader.algo_config.to_config()
        self.assertIs(algo_config.algorithm, PrimitiveStepPPO)


if __name__ == "__main__":
    absltest.main()
