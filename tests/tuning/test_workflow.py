from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import optuna
import yaml
from absl.testing import absltest
from rl_pipeline.sb3 import (
    SB3OptunaConfig,
    SB3OptunaParamConfig,
    SB3Pipeline,
)

from hrl_tl.tuning.workflow import (
    TuningWorkflowConfig,
    build_optuna_config,
    export_replicate_yaml,
    run_tuning_stage,
)


class TuningWorkflowTest(absltest.TestCase):
    def test_build_optuna_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_dir = Path(tmp_dir) / "db"
            config = TuningWorkflowConfig(
                study_name="test_study",
                db_filename="test.db",
                db_dir=db_dir,
                total_timesteps=1000,
                direction="minimize",
                metric="loss/total",
                n_trials=10,
                n_startup_trials=2,
                n_evaluations=3,
                n_eval_episodes=50,
                n_jobs=1,
                tune_params=[
                    SB3OptunaParamConfig(
                        name="lr",
                        target="algo_kwargs.learning_rate",
                        suggest_type="float",
                        low=1e-4,
                        high=1e-3,
                    )
                ],
            )
            optuna_cfg = build_optuna_config(config)

            self.assertEqual(optuna_cfg.study_name, "test_study")
            self.assertEqual(optuna_cfg.direction, "minimize")
            self.assertEqual(optuna_cfg.metric, "loss/total")
            self.assertEqual(optuna_cfg.n_trials, 10)
            self.assertEqual(optuna_cfg.total_timesteps, 1000)
            self.assertIsNone(optuna_cfg.eval_freq)
            self.assertIsNotNone(optuna_cfg.storage_url)
            assert optuna_cfg.storage_url is not None
            self.assertTrue(optuna_cfg.storage_url.startswith("sqlite:///"))
            self.assertTrue(db_dir.exists())

            config_with_eval_freq = config.model_copy(update={"eval_freq": 200})
            optuna_cfg2 = build_optuna_config(config_with_eval_freq)
            self.assertEqual(optuna_cfg2.eval_freq, 200)

    def test_export_replicate_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_yaml = tmp_path / "base.yaml"
            target_yaml = tmp_path / "replicate.yaml"

            base_content = {
                "device": 0,
                "experiment_id": "exp1",
                "algo_kwargs": {"learning_rate": 0.001, "batch_size": 256},
                "wrapper_kwargs": {"max_steps": 20},
            }
            with source_yaml.open("w", encoding="utf-8") as f:
                yaml.safe_dump(base_content, f)

            best_params = {
                "batch_size": 1024,
                "wrapper_kwargs.max_steps": 50,
                "switch_threshold_min": -25.0,
                "switch_threshold_max": 10.0,
            }
            tune_params = [
                SB3OptunaParamConfig(
                    name="switch_threshold_min",
                    target="wrapper_kwargs.spec_rep_args.args.switch_threshold.range.0",
                    suggest_type="float",
                    low=-30.0,
                    high=-5.0,
                ),
                SB3OptunaParamConfig(
                    name="switch_threshold_max",
                    target="wrapper_kwargs.spec_rep_args.args.switch_threshold.range.1",
                    suggest_type="float",
                    low=0.0,
                    high=15.0,
                ),
            ]

            export_replicate_yaml(
                source_yaml_path=source_yaml,
                target_yaml_path=target_yaml,
                best_params=best_params,
                tune_params=tune_params,
                num_replicates=5,
                replicate_signature="rep_{rep_id}",
            )

            self.assertTrue(target_yaml.exists())
            with target_yaml.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)

            self.assertEqual(loaded["replicate_config"]["num_replicates"], 5)
            self.assertEqual(
                loaded["replicate_config"]["replicate_signature"],
                "rep_{rep_id}",
            )
            single = loaded["single_pipeline_config"]
            self.assertEqual(single["algo_kwargs"]["batch_size"], 1024)
            self.assertEqual(single["wrapper_kwargs"]["max_steps"], 50)
            self.assertEqual(
                single["wrapper_kwargs"]["spec_rep_args"]["args"][
                    "switch_threshold"
                ]["range"],
                [-25.0, 10.0],
            )

    def test_tuning_configs_validity(self) -> None:
        configs_dir = Path("configs/tuning")
        yaml_files = list(configs_dir.glob("*.yaml"))
        self.assertGreaterEqual(len(yaml_files), 13)

        for yaml_path in yaml_files:
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            config = TuningWorkflowConfig(**data)
            self.assertNotEmpty(config.study_name)
            self.assertNotEmpty(config.tune_params)
            self.assertIn(config.direction, ("maximize", "minimize"))
            if config.metric == "mean_reward":
                self.assertEqual(config.n_eval_episodes, 100)

    def test_run_tuning_stage_with_strict_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            export_path = tmp_path / "best.yaml"

            mock_pipeline = mock.create_autospec(
                SB3Pipeline, instance=True, spec_set=True
            )
            mock_study = mock.create_autospec(
                optuna.Study, instance=True, spec_set=True
            )
            mock_trial = mock.create_autospec(
                optuna.trial.FrozenTrial, instance=True, spec_set=True
            )
            mock_trial.number = 3
            type(mock_study).study_name = mock.PropertyMock(
                return_value="mock_study"
            )
            type(mock_study).best_value = mock.PropertyMock(return_value=42.0)
            type(mock_study).best_trial = mock.PropertyMock(
                return_value=mock_trial
            )
            type(mock_study).best_params = mock.PropertyMock(
                return_value={"learning_rate": 0.0005}
            )
            mock_pipeline.optimize.return_value = mock_study

            optuna_cfg = SB3OptunaConfig(
                storage_url=f"sqlite:///{tmp_path / 'mock.db'}",
                study_name="mock_study",
                n_trials=1,
                total_timesteps=10,
                tune_params=[],
            )

            study = run_tuning_stage(
                pipeline=mock_pipeline,
                optuna_config=optuna_cfg,
                export_yaml_path=export_path,
            )

            mock_pipeline.optimize.assert_called_once_with(
                optuna_config=optuna_cfg
            )
            mock_pipeline.export_best_params.assert_called_once_with(
                mock_study, out_path=export_path
            )
            self.assertEqual(study, mock_study)


if __name__ == "__main__":
    absltest.main()
