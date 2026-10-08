"""Unit tests for SpecConditionedTLWrapper."""

from __future__ import annotations

from typing import Any, override

import gymnasium as gym
import numpy as np
from absl.testing import absltest
from gym_tl_tools import BaseVarValueInfoGenerator, Predicate, RewardConfig
from gymnasium import spaces

from hrl_tl.wrappers.spec_cond import SpecConditionedTLWrapper


class DummyGenerator(
    BaseVarValueInfoGenerator[dict[str, np.ndarray], np.ndarray]
):
    """Test variable value generator returning dummy predicate values."""

    def __init__(self, d_y: float = 1.0, d_r: float = 1.0) -> None:
        self.d_y = d_y
        self.d_r = d_r

    def get_var_values(
        self,
        env: (
            gym.Env[dict[str, np.ndarray], np.ndarray]
            | gym.Wrapper[
                dict[str, np.ndarray],
                np.ndarray,
                dict[str, np.ndarray],
                np.ndarray,
            ]
            | None
        ),
        obs: dict[str, np.ndarray],
        info: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del env, obs, info
        return {"d_y": self.d_y, "d_r": self.d_r}


class DummyDictEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Test environment with Dict observation space."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-10.0, high=10.0, shape=(5,), dtype=np.float32
                ),
            }
        )

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        return {"observation": np.zeros(5, dtype=np.float32)}, {}

    @override
    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        return (
            {"observation": np.zeros(5, dtype=np.float32)},
            0.0,
            False,
            False,
            {},
        )


class DummyBoxEnv(gym.Env[np.ndarray, np.ndarray]):
    """Test environment with Box observation space."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    @override
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


class SpecConditionedTLWrapperTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sample_predicates = [
            Predicate(name="psi_y", formula="d_y < 0.05"),
            Predicate(name="psi_r", formula="d_r < 0.05"),
        ]
        self.sample_specifications = [
            "F psi_y",
            "F psi_r",
            "!psi_r U psi_y",
        ]

    def test_wrapper_initialization(self) -> None:
        env = DummyDictEnv()
        gen = DummyGenerator()
        wrapper = SpecConditionedTLWrapper(
            env=env,
            specifications=self.sample_specifications,
            atomic_predicates=self.sample_predicates,
            var_value_info_generator=gen,
            step_penalty=0.01,
            early_termination=True,
            spec_rep="one_hot",
        )

        self.assertLen(wrapper.automatons, 3)
        self.assertIn("goal_spec", wrapper.observation_space.spaces)
        self.assertIsInstance(
            wrapper.observation_space.spaces["goal_spec"], spaces.MultiDiscrete
        )
        self.assertLen(wrapper.spec_encodings, 3)
        np.testing.assert_array_equal(wrapper.spec_encodings[0], [1, 0, 0])
        np.testing.assert_array_equal(wrapper.spec_encodings[1], [0, 1, 0])
        np.testing.assert_array_equal(wrapper.spec_encodings[2], [0, 0, 1])

    def test_wrapper_index_representation(self) -> None:
        env = DummyDictEnv()
        gen = DummyGenerator()
        wrapper = SpecConditionedTLWrapper(
            env=env,
            specifications=self.sample_specifications,
            atomic_predicates=self.sample_predicates,
            var_value_info_generator=gen,
            spec_rep="index",
        )

        self.assertIsInstance(
            wrapper.observation_space.spaces["goal_spec"], spaces.Discrete
        )
        self.assertEqual(wrapper.observation_space.spaces["goal_spec"].n, 3)
        self.assertEqual(int(wrapper.spec_encodings[1]), 1)

    def test_wrapper_box_env_observation_space(self) -> None:
        env = DummyBoxEnv()
        gen = DummyGenerator()
        wrapper = SpecConditionedTLWrapper(
            env=env,
            specifications=self.sample_specifications,
            atomic_predicates=self.sample_predicates,
            var_value_info_generator=gen,
        )

        self.assertIsInstance(wrapper.observation_space, spaces.Dict)
        self.assertIn("obs", wrapper.observation_space.spaces)
        self.assertIn("goal_spec", wrapper.observation_space.spaces)

    def test_wrapper_invalid_inputs(self) -> None:
        env = DummyDictEnv()
        gen = DummyGenerator()

        with self.assertRaisesRegex(
            ValueError, "specifications list must not be empty"
        ):
            SpecConditionedTLWrapper(
                env=env,
                specifications=[],
                atomic_predicates=self.sample_predicates,
                var_value_info_generator=gen,
            )

        with self.assertRaisesRegex(
            ValueError, "atomic_predicates list must not be empty"
        ):
            SpecConditionedTLWrapper(
                env=env,
                specifications=self.sample_specifications,
                atomic_predicates=[],
                var_value_info_generator=gen,
            )

        with self.assertRaisesRegex(TypeError, "reward_config must be"):
            SpecConditionedTLWrapper(
                env=env,
                specifications=self.sample_specifications,
                atomic_predicates=self.sample_predicates,
                var_value_info_generator=gen,
                reward_config="invalid",  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "Unsupported spec_rep"):
            SpecConditionedTLWrapper(
                env=env,
                specifications=self.sample_specifications,
                atomic_predicates=self.sample_predicates,
                var_value_info_generator=gen,
                spec_rep="binary",  # type: ignore[arg-type]
            )

    def test_reset_sampling_and_options(self) -> None:
        env = DummyDictEnv()
        gen = DummyGenerator()
        wrapper = SpecConditionedTLWrapper(
            env=env,
            specifications=self.sample_specifications,
            atomic_predicates=self.sample_predicates,
            var_value_info_generator=gen,
        )

        # Reset with specific spec_idx
        obs, info = wrapper.reset(options={"spec_idx": 1})
        self.assertEqual(info["spec_idx"], 1)
        self.assertEqual(info["spec"], "F psi_r")
        self.assertFalse(info["success"])
        self.assertEqual(info["is_success"], 0.0)
        np.testing.assert_array_equal(obs["goal_spec"], [0, 1, 0])

        # Reset with specific tl_spec
        obs, info = wrapper.reset(options={"tl_spec": "!psi_r U psi_y"})
        self.assertEqual(info["spec_idx"], 2)
        self.assertEqual(info["spec"], "!psi_r U psi_y")
        np.testing.assert_array_equal(obs["goal_spec"], [0, 0, 1])

        # Reset out of bounds
        with self.assertRaisesRegex(ValueError, "out of range"):
            wrapper.reset(options={"spec_idx": 10})

        with self.assertRaisesRegex(
            ValueError, "not in candidate specifications"
        ):
            wrapper.reset(options={"tl_spec": "F psi_unknown"})

        # Random reset
        obs, info = wrapper.reset(seed=42)
        self.assertGreaterEqual(info["spec_idx"], 0)
        self.assertLess(info["spec_idx"], 3)
        self.assertIn("goal_spec", obs)

    def test_step_and_termination(self) -> None:
        env = DummyDictEnv()
        gen = DummyGenerator(d_y=0.01, d_r=1.0)
        reward_config = RewardConfig(
            terminal_state_reward=1.0,
            state_trans_reward_scale=0.5,
            dense_reward=False,
        )
        wrapper = SpecConditionedTLWrapper(
            env=env,
            specifications=self.sample_specifications,
            atomic_predicates=self.sample_predicates,
            var_value_info_generator=gen,
            reward_config=reward_config,
            step_penalty=0.05,
            early_termination=True,
        )

        # Formula 0 is "F psi_y". Since d_y = 0.01 < 0.05, it satisfies psi_y.
        obs, info = wrapper.reset(options={"spec_idx": 0})
        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        self.assertIn("goal_spec", obs)
        self.assertTrue(info["success"])
        self.assertEqual(info["is_success"], 1.0)
        self.assertEqual(info["status"], "goal")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 0.02 - 0.05, places=5)


if __name__ == "__main__":
    absltest.main()
