from __future__ import annotations

from typing import override
from unittest import mock

import numpy as np
import torch as th
from absl.testing import absltest, parameterized
from gymnasium import spaces
from rl_pipeline.sb3 import SB3PipelineConfig, SB3ReplicatePipelineConfig
from sb3_hrl import ALLO

from hrl_tl.wrappers.option import intrinsic_rewards as intrinsic_rewards_module
from hrl_tl.wrappers.option.intrinsic_rewards import (
    AlloIntrinsicReward,
    build_observation_batch,
    observation_to_flat_array,
)


class IntrinsicRewardsTest(parameterized.TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.zone_obs_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-0.5, high=11.5, shape=(2,), dtype=np.float64
                ),
                "agent_vel": spaces.Box(
                    low=-17.0, high=17.0, shape=(2,), dtype=np.float64
                ),
                "wall_dist": spaces.Box(
                    low=0.0, high=17.0, shape=(4,), dtype=np.float64
                ),
                "yellow_dist": spaces.Box(
                    low=-12.0, high=12.0, shape=(3, 2), dtype=np.float64
                ),
                "red_dist": spaces.Box(
                    low=-12.0, high=12.0, shape=(5, 2), dtype=np.float64
                ),
                "white_dist": spaces.Box(
                    low=-12.0, high=12.0, shape=(1, 2), dtype=np.float64
                ),
                "black_dist": spaces.Box(
                    low=-12.0, high=12.0, shape=(2, 2), dtype=np.float64
                ),
                "zone_visits": spaces.MultiDiscrete([2, 2, 2, 2]),
            }
        )

        self.sample_obs = {
            "agent_pos": np.array([1.0, 2.0], dtype=np.float64),
            "agent_vel": np.array([0.5, -0.5], dtype=np.float64),
            "wall_dist": np.zeros(4, dtype=np.float64),
            "yellow_dist": np.zeros((3, 2), dtype=np.float64),
            "red_dist": np.zeros((5, 2), dtype=np.float64),
            "white_dist": np.zeros((1, 2), dtype=np.float64),
            "black_dist": np.zeros((2, 2), dtype=np.float64),
            "zone_visits": np.array([0, 1, 0, 0], dtype=np.int64),
        }

    def test_observation_to_flat_array_without_space_produces_raw_dimension(
        self,
    ) -> None:
        flat_without_space = observation_to_flat_array(self.sample_obs)
        self.assertEqual(flat_without_space.shape, (1, 34))
        np.testing.assert_allclose(
            flat_without_space[0, -4:], [0.0, 1.0, 0.0, 0.0]
        )

    def test_observation_to_flat_array_with_space_encodes_multidiscrete(
        self,
    ) -> None:
        flat_with_space = observation_to_flat_array(
            self.sample_obs, observation_space=self.zone_obs_space
        )
        self.assertEqual(flat_with_space.shape, (1, 38))
        np.testing.assert_allclose(
            flat_with_space[0, -8:],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        )

    def test_observation_to_flat_array_batched_with_space(self) -> None:
        batched_obs = {
            k: np.stack([v, v], axis=0) for k, v in self.sample_obs.items()
        }
        flat_batched = observation_to_flat_array(
            batched_obs, observation_space=self.zone_obs_space
        )
        self.assertEqual(flat_batched.shape, (2, 38))
        np.testing.assert_allclose(flat_batched[0], flat_batched[1])

    def test_observation_to_flat_array_box_space(self) -> None:
        box_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 3), dtype=np.float32
        )
        box_sample = np.ones((2, 3), dtype=np.float32)
        flat = observation_to_flat_array(
            box_sample, observation_space=box_space
        )
        self.assertEqual(flat.shape, (1, 6))
        np.testing.assert_allclose(flat, np.ones((1, 6), dtype=np.float32))

    def test_build_observation_batch_shape(self) -> None:
        next_obs = {k: v.copy() for k, v in self.sample_obs.items()}
        obs_tensor = build_observation_batch(
            [self.sample_obs, next_obs],
            observation_space=self.zone_obs_space,
        )
        self.assertEqual(obs_tensor.shape, (2, 38))
        self.assertEqual(obs_tensor.dtype, th.float32)

    @parameterized.named_parameters(
        ("forward_reward", False, 1.0),
        ("reverse_reward", True, -1.0),
    )
    def test_allo_intrinsic_reward_step_computation(
        self, reverse_reward: bool, expected_reward: float
    ) -> None:
        mock_model = mock.create_autospec(ALLO, instance=True, spec_set=False)
        mock_model.observation_space = self.zone_obs_space
        mock_model.device = th.device("cpu")
        feature_net = th.nn.Linear(38, 8, bias=False)
        with th.no_grad():
            feature_net.weight.zero_()
            feature_net.weight[0, 0] = 1.0
        mock_model.feature_net = feature_net

        # SB3ReplicatePipeline assigns instance attributes in __init__, so
        # spec_set=False is required to attach mock configs dynamically.
        with mock.patch.object(
            intrinsic_rewards_module,
            "SB3ReplicatePipeline",
            autospec=True,
            spec_set=False,
        ) as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_ind_config = mock.create_autospec(
                SB3PipelineConfig, instance=True, spec_set=False
            )
            mock_pipeline.ind_pipeline_configs = [mock_ind_config]
            mock_pipeline.load_model.return_value = mock_model

            with mock.patch.object(
                intrinsic_rewards_module,
                "SB3ReplicatePipelineConfigReader",
                autospec=True,
                spec_set=True,
            ) as mock_reader_cls:
                mock_rep_config = mock.create_autospec(
                    SB3ReplicatePipelineConfig, instance=True, spec_set=False
                )
                mock_rep_config.ind_pipeline_configs = [mock_ind_config]
                mock_reader_cls.from_yaml.return_value.to_config.return_value = mock_rep_config

                reward_fn = AlloIntrinsicReward(
                    pipeline_config_path="dummy_pipeline.yaml",
                    eig_idx=0,
                    reverse_reward=reverse_reward,
                )

                next_obs = {k: v.copy() for k, v in self.sample_obs.items()}
                next_obs["agent_pos"] = np.array([2.0, 2.0], dtype=np.float64)

                val = reward_fn.intrinsic_reward(
                    self.sample_obs,
                    0,
                    next_obs,
                    0.0,
                    False,
                )
                self.assertIsInstance(val, float)
                self.assertAlmostEqual(val, expected_reward, places=4)


if __name__ == "__main__":
    absltest.main()
