import contgrid
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig

from hrl_tl.config.wrapper import GCLTLPipelineConfigReader

if __name__ == "__main__":
    pipeline_config_path: str = (
        "configs/fr_cont/cpc/pr_soft/3.k.l_train_primitive.yaml"
    )

    gpu_id: int = 3
    retrain_model: bool = True

    config_reader = GCLTLPipelineConfigReader.from_yaml(pipeline_config_path)

    # Override GPU ID and retrain_model from command line args
    config_reader.device = gpu_id
    config_reader.retrain_model = retrain_model

    pipeline_config: SB3PipelineConfig = config_reader.to_config()
    pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

    # Train the model on the unsaved model
    pipeline.optimize()
