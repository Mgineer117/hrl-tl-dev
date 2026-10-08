from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from absl.testing import absltest
from rl_pipeline.sb3 import SB3ReplicatePipelineConfigReader
from sb3_soft import SDSAC

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.config.meta_option import TLMetaOptionPipelineConfigReader
from hrl_tl.config.wrapper import (
    AlloMetaPipelineConfigReader,
    TLSB3PipelineConfigReader,
)


class FetchConfigsTest(absltest.TestCase):
    def test_allo_meta_config_parses(self) -> None:
        path_7la = Path("configs/fetch/train/allo/7.l.a_allo_meta_rep0.yaml")
        reader_7la = SB3ReplicatePipelineConfigReader[
            AlloMetaPipelineConfigReader
        ].from_yaml(str(path_7la))
        self.assertEqual(
            reader_7la.single_pipeline_config.env_config_file,
            "env/fetch_reach_avoid_250.yaml",
        )

    def test_ablation_subpolicy_pipeline_configs(self) -> None:
        base_path = Path(
            "configs/fetch/train/ablation/12.a.a_subpolicy_rep0.yaml"
        )
        self.assertTrue(base_path.exists())
        reader = TLSB3PipelineConfigReader.from_yaml(str(base_path))
        pipe_cfg = reader.to_config()
        self.assertEqual(pipe_cfg.algo_config.algorithm, SDSAC)
        self.assertEqual(
            pipe_cfg.algo_config.algo_kwargs.get("policy"), "MultiInputPolicy"
        )
        self.assertIsNotNone(pipe_cfg.wrapper_config)

    def test_ablation_meta_pipeline_config(self) -> None:
        base_path = Path("configs/fetch/train/ablation/12.b.a_meta_rep0.yaml")
        self.assertTrue(base_path.exists())
        reader = SB3ReplicatePipelineConfigReader[
            TLMetaOptionPipelineConfigReader
        ].from_yaml(str(base_path))
        single_cfg = reader.single_pipeline_config
        self.assertIn("ablation", str(single_cfg.wrapper_config_file))

    def test_m1_fixed_lambda1_pipeline_configs(self) -> None:
        base_path = Path("configs/fetch/train/13.a/13.a.a_meta_rep0.yaml")
        self.assertTrue(base_path.exists())
        reader = SB3ReplicatePipelineConfigReader[
            TLMetaOptionPipelineConfigReader
        ].from_yaml(str(base_path))
        single_cfg = reader.single_pipeline_config
        self.assertIn("13.a", str(single_cfg.wrapper_config_file))
        pipe_cfg = single_cfg.to_config()
        self.assertIsNotNone(pipe_cfg.wrapper_config)
        self.assertEqual(
            pipe_cfg.wrapper_config.wrapper_kwargs.get("max_low_level_policy_steps"),
            40,
        )

    def test_parameter_omission_pipeline_configs(self) -> None:
        """Verifies that 14.a, 14.b, and 14.c pipeline configs parse and resolve correctly."""
        for exp_id in ("14.a", "14.b", "14.c"):
            base_path = Path(f"configs/fetch/train/{exp_id}/{exp_id}.a_meta_rep0.yaml")
            self.assertTrue(base_path.exists())
            reader = SB3ReplicatePipelineConfigReader[
                TLMetaOptionPipelineConfigReader
            ].from_yaml(str(base_path))
            single_cfg = reader.single_pipeline_config
            self.assertIn(exp_id, str(single_cfg.wrapper_config_file))
            pipe_cfg = single_cfg.to_config()
            self.assertIsNotNone(pipe_cfg.wrapper_config)
            self.assertEqual(
                pipe_cfg.wrapper_config.wrapper_kwargs.get("max_low_level_policy_steps"),
                40,
            )


if __name__ == "__main__":
    absltest.main()
