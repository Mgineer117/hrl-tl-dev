import contgrid
from rl_pipeline.sb3 import (
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)

from hrl_tl.config.wrapper import TLHighLevelPipelineConfigReader

if __name__ == "__main__":
    exp_config_path: str = "configs/fr_cont/train/5.b/5.b.b_hrl_train_rep4.yaml"
    retrain_model: bool = False

    exp_config: SB3ReplicatePipelineConfig = (
        SB3ReplicatePipelineConfigReader[TLHighLevelPipelineConfigReader]
        .from_yaml(exp_config_path)
        .to_config()
    )
    exp_config.ind_pipeline_configs[0].retrain_model = retrain_model

    pipeline = SB3ReplicatePipeline(config=exp_config, verbose=True)
    pipeline.train_on_unsaved_model()

    models = pipeline.load_models("best")
    pipeline.record_replays(models, verbose=True)

    # pipeline.evaluate(n_eval_episodes=200, checkpoint="best")
