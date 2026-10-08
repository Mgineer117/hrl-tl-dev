"""Unit tests for SpecCondSubpolicy and its configuration readers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from absl.testing import absltest
from gymnasium import Env
from hrl_tl.wrappers.low_level_policies.spec_cond import (
    SpecCondSubpolicy,
    SpecCondSubpolicyConfig,
    SpecCondSubpolicyConfigReader,
)
from stable_baselines3.common.base_class import BaseAlgorithm


class SpecCondSubpolicyTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mock_env = mock.MagicMock(spec=Env)
        self.mock_model = mock.create_autospec(BaseAlgorithm, instance=True)
        self.mock_model.predict.return_value = (
            np.array([0.5, -0.5], dtype=np.float32),
            None,
        )
        self.specifications = [
            "F psi_y",
            "F psi_r",
            "!psi_r U psi_y",
        ]

    def test_define_policy_one_hot(self) -> None:
        config = SpecCondSubpolicyConfig(
            model=self.mock_model,
            specifications=self.specifications,
            spec_rep="one_hot",
            dict_spec_key="goal_spec",
            deterministic=True,
        )
        policy = SpecCondSubpolicy(
            env=self.mock_env,
            tl_spec="F psi_r",
            max_policy_steps=40,
            policy_args=config,
        )

        self.assertIs(policy.policy, self.mock_model)
        self.assertEqual(policy.spec_idx, 1)
        np.testing.assert_array_equal(policy.spec_encoding, [0, 1, 0])

    def test_define_policy_index(self) -> None:
        config = SpecCondSubpolicyConfig(
            model=self.mock_model,
            specifications=self.specifications,
            spec_rep="index",
        )
        policy = SpecCondSubpolicy(
            env=self.mock_env,
            tl_spec="!psi_r U psi_y",
            max_policy_steps=40,
            policy_args=config,
        )

        self.assertEqual(policy.spec_idx, 2)
        self.assertEqual(int(policy.spec_encoding), 2)

    def test_define_policy_invalid_spec(self) -> None:
        config = SpecCondSubpolicyConfig(
            model=self.mock_model,
            specifications=self.specifications,
        )
        with self.assertRaisesRegex(ValueError, "not in candidate list"):
            SpecCondSubpolicy(
                env=self.mock_env,
                tl_spec="F unknown_predicate",
                max_policy_steps=40,
                policy_args=config,
            )

    def test_act_strips_aut_state_and_injects_spec(self) -> None:
        config = SpecCondSubpolicyConfig(
            model=self.mock_model,
            specifications=self.specifications,
            dict_spec_key="goal_spec",
            deterministic=True,
        )
        policy = SpecCondSubpolicy(
            env=self.mock_env,
            tl_spec="F psi_y",
            max_policy_steps=40,
            policy_args=config,
        )

        obs = {
            "observation": np.array([1.0, 2.0], dtype=np.float32),
            "aut_state": 3,
        }

        action = policy.act(obs)

        np.testing.assert_array_equal(
            action, np.array([0.5, -0.5], dtype=np.float32)
        )
        self.mock_model.predict.assert_called_once()
        passed_obs, kwargs = self.mock_model.predict.call_args
        called_dict = passed_obs[0]
        self.assertNotIn("aut_state", called_dict)
        self.assertIn("goal_spec", called_dict)
        np.testing.assert_array_equal(called_dict["goal_spec"], [1, 0, 0])
        self.assertTrue(kwargs.get("deterministic"))

    def test_act_non_dict_obs(self) -> None:
        config = SpecCondSubpolicyConfig(
            model=self.mock_model,
            specifications=self.specifications,
            dict_spec_key="goal_spec",
        )
        policy = SpecCondSubpolicy(
            env=self.mock_env,
            tl_spec="F psi_r",
            max_policy_steps=40,
            policy_args=config,
        )

        obs_array = np.array([1.0, 2.0], dtype=np.float32)
        action = policy.act(obs_array)

        np.testing.assert_array_equal(
            action, np.array([0.5, -0.5], dtype=np.float32)
        )
        called_dict = self.mock_model.predict.call_args[0][0]
        self.assertIn("obs", called_dict)
        self.assertIn("goal_spec", called_dict)
        np.testing.assert_array_equal(called_dict["goal_spec"], [0, 1, 0])

    def test_config_reader_file_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_model_file = tmp_path / "model.zip"
            fake_model_file.write_text("dummy_content", encoding="utf-8")

            fake_formulae_file = tmp_path / "formulae.json"
            fake_formulae_file.write_text(
                json.dumps({"specifications": ["F psi_y"]}), encoding="utf-8"
            )

            # Invalid algo
            reader = SpecCondSubpolicyConfigReader(
                model_path=str(fake_model_file),
                all_formulae_file_path=str(fake_formulae_file),
                algo="UNSUPPORTED_ALGO",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported algorithm"):
                reader.to_config()

            # Nonexistent model path
            reader_no_model = SpecCondSubpolicyConfigReader(
                model_path=str(tmp_path / "nonexistent.zip"),
                all_formulae_file_path=str(fake_formulae_file),
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "Model checkpoint not found"
            ):
                reader_no_model.to_config()

            # Nonexistent formulae path
            reader_no_formula = SpecCondSubpolicyConfigReader(
                model_path=str(fake_model_file),
                all_formulae_file_path=str(tmp_path / "missing.json"),
            )
            with (
                self.assertRaisesRegex(
                    FileNotFoundError, "Formulae file not found"
                ),
                mock.patch("sb3_soft.SDSAC.load") as mock_load,
            ):
                mock_load.return_value = self.mock_model
                reader_no_formula.to_config()

            # Invalid json without 'specifications' key
            bad_json_file = tmp_path / "bad.json"
            bad_json_file.write_text(
                json.dumps({"wrong_key": []}), encoding="utf-8"
            )
            with mock.patch("sb3_soft.SDSAC.load") as mock_load:
                mock_load.return_value = self.mock_model
                reader_bad_json = SpecCondSubpolicyConfigReader(
                    model_path=str(fake_model_file),
                    all_formulae_file_path=str(bad_json_file),
                    algo="SDSAC",
                )
                with self.assertRaisesRegex(
                    KeyError, "Key 'specifications' missing"
                ):
                    reader_bad_json.to_config()

            # Successful load
            with mock.patch("sb3_soft.SDSAC.load") as mock_load:
                mock_load.return_value = self.mock_model
                valid_reader = SpecCondSubpolicyConfigReader(
                    model_path=str(fake_model_file),
                    all_formulae_file_path=str(fake_formulae_file),
                    algo="SDSAC",
                )
                valid_config = valid_reader.to_config()
                self.assertEqual(valid_config.specifications, ["F psi_y"])
                self.assertIs(valid_config.model, self.mock_model)


if __name__ == "__main__":
    absltest.main()
