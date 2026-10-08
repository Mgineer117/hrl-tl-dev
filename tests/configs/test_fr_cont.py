from __future__ import annotations

from absl.testing import absltest
from rl_pipeline.sb3 import SB3ReplicatePipelineConfigReader

from hrl_tl.config.wrapper import (
    TLHighLevelPipelineConfigReader,
    TLSB3PipelineConfigReader,
)


class FrContConfigsTest(absltest.TestCase):
    def test_9_c_a_meta_config_loading(self) -> None:
        config_path = "configs/fr_cont/train/baseline/9.c.a_meta_rep0.yaml"
        replicate_config = (
            SB3ReplicatePipelineConfigReader[TLHighLevelPipelineConfigReader]
            .from_yaml(config_path)
            .to_config()
        )
        self.assertLen(replicate_config.ind_pipeline_configs, 1)
        ind_cfg = replicate_config.ind_pipeline_configs[0]
        self.assertEqual(ind_cfg.experiment_id, "9.c.a")
        assert ind_cfg.vec_config is not None
        self.assertEqual(ind_cfg.vec_config.n_envs, 5)

    def test_9_c_a_subpolicy_config_loading(self) -> None:
        config_path = "configs/fr_cont/train/baseline/9.c.a_subpolicy_rep0.yaml"
        subpolicy_reader = TLSB3PipelineConfigReader.from_yaml(config_path)
        subpolicy_reader.tl_spec = "Fpsi_ld & G!psi_lv"
        subpolicy_config = subpolicy_reader.to_config()
        self.assertEqual(subpolicy_config.experiment_id, "9.c.a")
        assert subpolicy_config.vec_config is not None
        self.assertEqual(subpolicy_config.vec_config.n_envs, 50)


if __name__ == "__main__":
    absltest.main()
