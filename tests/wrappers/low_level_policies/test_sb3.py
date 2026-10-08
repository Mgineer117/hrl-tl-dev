from __future__ import annotations

from unittest import mock

from absl.testing import absltest, parameterized
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.wrappers.low_level_policies.sb3 import (
    SB3Subpolicy,
    SB3SubpolicyBuffer,
    SB3SubpolicyConfig,
    SB3SubpolicyConfigReader,
)


class SB3SubpolicyTest(parameterized.TestCase):
    def test_config_reader_device_settings(self) -> None:
        reader = SB3SubpolicyConfigReader(
            config_path="configs/fr_cont/train/baseline/9.c.a_subpolicy_rep0.yaml",
            device="cuda:1",
            available_devices=["cuda:0", "cuda:1", "cuda:2"],
        )
        config = reader.to_config()
        self.assertEqual(
            config.config_path,
            "configs/fr_cont/train/baseline/9.c.a_subpolicy_rep0.yaml",
        )
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(
            config.available_devices, ["cuda:0", "cuda:1", "cuda:2"]
        )

    def test_config_reader_flag_settings(self) -> None:
        reader = SB3SubpolicyConfigReader(
            retrain_model=True,
            verbose=True,
            auto_train=True,
        )
        config = reader.to_config()
        self.assertTrue(config.retrain_model)
        self.assertTrue(config.verbose)
        self.assertTrue(config.auto_train)

    def test_buffer_caches_policy(self) -> None:
        config = SB3SubpolicyConfig(
            config_path="configs/fr_cont/train/baseline/9.c.a_subpolicy_rep0.yaml",
            device="cpu",
        )
        buffer = SB3SubpolicyBuffer(policy_args=config)
        mock_policy = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )
        tl_spec = "Fpsi_ld & G!psi_lv"

        buffer.policy_cache[tl_spec] = mock_policy
        self.assertIn(tl_spec, buffer.policy_cache)
        self.assertEqual(buffer.policy_cache[tl_spec], mock_policy)

    def test_buffer_preserves_cache_on_reset(self) -> None:
        config = SB3SubpolicyConfig(
            config_path="configs/fr_cont/train/baseline/9.c.a_subpolicy_rep0.yaml",
            device="cpu",
        )
        buffer = SB3SubpolicyBuffer(policy_args=config)
        mock_policy = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )
        tl_spec = "Fpsi_ld & G!psi_lv"
        buffer.policy_cache[tl_spec] = mock_policy

        buffer.at_reset()
        self.assertIn(tl_spec, buffer.policy_cache)
        self.assertEqual(buffer.policy_cache[tl_spec], mock_policy)

    def test_define_policy_retrieves_from_cache(self) -> None:
        config = SB3SubpolicyConfig(
            config_path="configs/fr_cont/train/baseline/9.c.a_subpolicy_rep0.yaml",
            device="cpu",
            retrain_model=False,
        )
        buffer = SB3SubpolicyBuffer(policy_args=config)
        mock_policy = mock.create_autospec(
            BaseAlgorithm, instance=True, spec_set=True
        )
        mock_env = mock.create_autospec(Env, instance=True, spec_set=True)
        tl_spec = "Fpsi_ld & G!psi_lv"
        buffer.policy_cache[tl_spec] = mock_policy

        subpolicy = SB3Subpolicy(
            env=mock_env,
            tl_spec=tl_spec,
            max_policy_steps=50,
            policy_args=config,
            buffer=buffer,
        )
        self.assertEqual(subpolicy.policy, mock_policy)

    def test_subpolicy_missing_model_raises_when_auto_train_disabled(
        self,
    ) -> None:
        config = SB3SubpolicyConfig(
            config_path="configs/fetch/train/ablation/12.a.a_subpolicy_rep0.yaml",
            device="cpu",
            retrain_model=False,
            auto_train=False,
        )
        mock_env = mock.create_autospec(Env, instance=True, spec_set=True)
        with self.assertRaises(FileNotFoundError):
            SB3Subpolicy(
                env=mock_env,
                tl_spec="Fpsi_w & G!psi_r",
                max_policy_steps=50,
                policy_args=config,
            )


if __name__ == "__main__":
    absltest.main()
