import gym_multigrid
from rl_pipeline.sb3 import (
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)

from hrl_tl.config.wrapper import TLHighLevelPipelineConfigReader

if __name__ == "__main__":
    exp_config_path: str = "configs/fourroom/sm/hrl/3.c.a_hrl_sm.yaml"

    exp_config: SB3ReplicatePipelineConfig = (
        SB3ReplicatePipelineConfigReader[TLHighLevelPipelineConfigReader]
        .from_yaml(exp_config_path)
        .to_config()
    )

    pipeline = SB3ReplicatePipeline(config=exp_config, verbose=True)
    pipeline.train_on_unsaved_model()
    pipeline.evaluate(n_eval_episodes=100, checkpoint="best")
