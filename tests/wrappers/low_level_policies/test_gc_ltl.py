from __future__ import annotations

from typing import Any
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from absl.testing import absltest, parameterized
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from torch import Tensor

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.envs.fetch.action import FetchMultiDiscreteAction
from hrl_tl.wrappers.gc_ltl import Predicate
from hrl_tl.wrappers.low_level_policies.gc_ltl import (
    GCLTLCompositePolicy,
    GCLTLCompositePolicyConfigReader,
)


class FakePolicyNetwork:
    def __init__(self, action_dims: list[int]) -> None:
        self._action_dims = action_dims

    def obs_to_tensor(
        self, obs: dict[str, np.ndarray] | np.ndarray
    ) -> tuple[dict[str, Tensor] | Tensor, bool]:
        if isinstance(obs, dict):
            return {
                k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)
                for k, v in obs.items()
            }, False
        return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), False

    def get_distribution(
        self, obs: dict[str, Tensor] | Tensor
    ) -> MultiCategoricalDistribution:
        dist = MultiCategoricalDistribution(self._action_dims)
        logits_list = [
            torch.zeros(1, dim, dtype=torch.float32)
            for dim in self._action_dims
        ]
        logits_list[0][0, 4] = 10.0
        logits_list[1][0, 2] = 10.0
        logits_list[2][0, 2] = 10.0
        dist.proba_distribution(torch.cat(logits_list, dim=-1))
        return dist

    def predict_values(self, obs: dict[str, Tensor] | Tensor) -> Tensor:
        return torch.tensor([[0.99]], dtype=torch.float32)


class FakePPO(PPO):
    def __init__(self, action_dims: list[int] | None = None) -> None:
        self.policy: Any = FakePolicyNetwork(action_dims or [5, 5, 5])

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        return np.array([4, 2, 2], dtype=np.int64), None


class GCLTLTest(parameterized.TestCase):
    def test_strict_action_mode_validation_in_config_reader(self) -> None:
        reader = GCLTLCompositePolicyConfigReader(
            model_path="dummy_path.zip",
            discrete_state_space=False,
            action_mode_cls=None,
        )
        fake_model = FakePPO()
        self.enter_context(
            mock.patch.object(
                PPO,
                "load",
                autospec=True,
                spec_set=True,
                return_value=fake_model,
            )
        )
        with self.assertRaises(ValueError) as ctx:
            reader.to_config()
        self.assertIn("action_mode_cls must be specified", str(ctx.exception))

    def test_config_reader_instantiates_fetch_action_mode(self) -> None:
        reader = GCLTLCompositePolicyConfigReader(
            model_path="dummy_path.zip",
            discrete_state_space=False,
            action_mode_cls="hrl_tl.envs.fetch.action.FetchMultiDiscreteAction",
            action_mode_args={"num_bins": 5},
        )
        fake_model = FakePPO()
        self.enter_context(
            mock.patch.object(
                PPO,
                "load",
                autospec=True,
                spec_set=True,
                return_value=fake_model,
            )
        )
        config = reader.to_config()
        self.assertIsInstance(config.action_mode, FetchMultiDiscreteAction)

    def test_gcltl_fetch_action_composition_prediction(self) -> None:
        env = gym.make("hrl_tl/FetchReachAvoid-v0")
        obs, _ = env.reset(seed=42)
        env.close()
        obs["aut_state"] = np.zeros(2, dtype=np.int64)

        fake_model = FakePPO(action_dims=[5, 5, 5])
        policy = GCLTLCompositePolicy(
            tl_spec="F psi_y & G !psi_r",
            predicates=[
                Predicate(name="psi_y", formula="d_y < 0.03"),
                Predicate(name="psi_r", formula="d_r < 0.03"),
            ],
            model=fake_model,
            switch_threshold=0.5,
            action_sum_coeff=0.5,
            threshold_type="value",
            discrete_state_space=False,
            action_mode=FetchMultiDiscreteAction(num_bins=5),
        )

        action, state = policy.predict(obs)
        self.assertIsNone(state)
        self.assertEqual(action.shape, (3,))


if __name__ == "__main__":
    absltest.main()
