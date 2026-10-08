"""Tests validating all Ablation S2 (Learned LTL) configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml
from absl.testing import absltest, parameterized
from rl_pipeline.sb3 import SB3ReplicatePipelineConfigReader

from hrl_tl.config.meta_option import TLMetaOptionPipelineConfigReader
from hrl_tl.config.wrapper import (
    SpecConditionedPipelineConfigReader,
    SpecConditionedWrapperConfigReader,
)


class FetchSpecCondConfigsTest(parameterized.TestCase):
    def test_spec_cond_wrapper_config_parses(self) -> None:
        wrapper_path = Path(
            "configs/fetch/spec_cond/wrapper/spec_cond_dense_tuned.yaml"
        )
        self.assertTrue(wrapper_path.exists())
        reader = SpecConditionedWrapperConfigReader.from_yaml(str(wrapper_path))
        self.assertEqual(reader.dict_spec_key, "goal_spec")
        self.assertEqual(reader.spec_rep, "one_hot")
        self.assertTrue(reader.early_termination)
        self.assertLen(reader.atomic_predicates, 3)

        # Check candidate specifications file exists
        spec_path = Path(reader.spec_file_path)
        self.assertTrue(spec_path.exists())
        cfg = reader.to_config()
        self.assertLen(cfg.wrapper_kwargs["specifications"], 9)

    def test_subpolicy_tuning_pipeline_config_parses(self) -> None:
        cfg_path = Path(
            "configs/fetch/spec_cond/pr/15.a.a_tune_spec_primitive.yaml"
        )
        self.assertTrue(cfg_path.exists())
        reader = SB3ReplicatePipelineConfigReader[
            SpecConditionedPipelineConfigReader
        ].from_yaml(str(cfg_path))
        self.assertEqual(reader.replicate_config.num_replicates, 1)
        self.assertEqual(reader.replicate_config.replicate_start_id, 0)
        self.assertEqual(reader.single_pipeline_config.experiment_id, "15.a.a")
        self.assertEqual(
            reader.single_pipeline_config.wrapper_config_file,
            "spec_cond/wrapper/spec_cond_dense_tuned.yaml",
        )
        self.assertEqual(
            reader.single_pipeline_config.model_config_file,
            "spec_cond/rl/sdsac_dict_512_10M_tuned.yaml",
        )

    def test_subpolicy_replicate_pipeline_config_parses(self) -> None:
        cfg_path = Path(
            "configs/fetch/spec_cond/pr/15.a.b_train_spec_primitive_rep.yaml"
        )
        self.assertTrue(cfg_path.exists())
        reader = SB3ReplicatePipelineConfigReader[
            SpecConditionedPipelineConfigReader
        ].from_yaml(str(cfg_path))
        self.assertEqual(reader.replicate_config.num_replicates, 5)
        self.assertEqual(reader.replicate_config.replicate_start_id, 0)
        self.assertEqual(
            reader.replicate_config.replicate_signature, "rep_{rep_id}"
        )
        self.assertEqual(reader.single_pipeline_config.experiment_id, "15.a.b")

    def test_meta_tuning_configs_parse(self) -> None:
        wrapper_path = Path(
            "configs/fetch/hrl/15.b/15.b.a_spec_cond_wrapper.yaml"
        )
        self.assertTrue(wrapper_path.exists())
        with wrapper_path.open("r", encoding="utf-8") as f:
            wrapper_dict = yaml.safe_load(f)
        self.assertEqual(
            wrapper_dict["low_level_policy_class"],
            "hrl_tl.wrappers.low_level_policies.spec_cond.SpecCondSubpolicy",
        )
        self.assertEqual(
            wrapper_dict["low_level_policy_args"]["dict_spec_key"], "goal_spec"
        )

        train_path = Path("configs/fetch/train/15.b/15.b.a_tune_meta.yaml")
        self.assertTrue(train_path.exists())
        reader = SB3ReplicatePipelineConfigReader[
            TLMetaOptionPipelineConfigReader
        ].from_yaml(str(train_path))
        self.assertEqual(reader.replicate_config.num_replicates, 1)
        self.assertEqual(reader.replicate_config.replicate_start_id, 0)
        self.assertEqual(reader.single_pipeline_config.experiment_id, "15.b.a")
        self.assertEqual(
            reader.single_pipeline_config.wrapper_config_file,
            "hrl/15.b/15.b.a_spec_cond_wrapper.yaml",
        )

    @parameterized.parameters(0, 1, 2, 3, 4)
    def test_meta_replicate_configs_parse(self, rep_id: int) -> None:
        wrapper_path = Path(
            f"configs/fetch/hrl/15.b/15.b.b_spec_cond_wrapper_rep{rep_id}.yaml"
        )
        self.assertTrue(wrapper_path.exists())
        with wrapper_path.open("r", encoding="utf-8") as f:
            wrapper_dict = yaml.safe_load(f)
        self.assertIn(
            f"rep_{rep_id}/best_model.zip",
            wrapper_dict["low_level_policy_args"]["model_path"],
        )

        train_path = Path(
            f"configs/fetch/train/15.b/15.b.b_meta_rep{rep_id}.yaml"
        )
        self.assertTrue(train_path.exists())
        reader = SB3ReplicatePipelineConfigReader[
            TLMetaOptionPipelineConfigReader
        ].from_yaml(str(train_path))
        self.assertEqual(reader.replicate_config.num_replicates, 1)
        self.assertEqual(reader.replicate_config.replicate_start_id, 0)
        self.assertEqual(
            reader.replicate_config.replicate_signature, f"rep_{rep_id}"
        )
        self.assertEqual(reader.single_pipeline_config.experiment_id, "15.b.b")
        self.assertEqual(
            reader.single_pipeline_config.wrapper_config_file,
            f"hrl/15.b/15.b.b_spec_cond_wrapper_rep{rep_id}.yaml",
        )

    def test_optuna_tuning_configs_syntax(self) -> None:
        sub_tune_path = Path("configs/tuning/spec_cond_sub_fetch.yaml")
        self.assertTrue(sub_tune_path.exists())
        with sub_tune_path.open("r", encoding="utf-8") as f:
            sub_tune_data = yaml.safe_load(f)
        self.assertEqual(sub_tune_data["study_name"], "spec_cond_sub_fetch")
        self.assertEqual(sub_tune_data["n_trials"], 50)
        self.assertIn("tune_params", sub_tune_data)

        hl_tune_path = Path("configs/tuning/spec_cond_hl_fetch.yaml")
        self.assertTrue(hl_tune_path.exists())
        with hl_tune_path.open("r", encoding="utf-8") as f:
            hl_tune_data = yaml.safe_load(f)
        self.assertEqual(hl_tune_data["study_name"], "spec_cond_hl_fetch")
        self.assertEqual(hl_tune_data["n_trials"], 20)
        self.assertIn("tune_params", hl_tune_data)


if __name__ == "__main__":
    absltest.main()
