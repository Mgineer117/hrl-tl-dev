import contgrid
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig

from hrl_tl.config.wrapper import FixedGCLTLPipelineConfigReader

if __name__ == "__main__":
    pipeline_config_path: str = (
        "configs/fr_cont/cpc/pr/3.f.c_train_primitive.yaml"
    )

    gpu_id: int = 1
    retrain_model: bool = False

    config_reader = FixedGCLTLPipelineConfigReader.from_yaml(
        pipeline_config_path
    )

    # Override GPU ID and retrain_model from command line args
    config_reader.device = gpu_id
    config_reader.retrain_model = retrain_model

    pipeline_config: SB3PipelineConfig = config_reader.to_config()
    pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

    # Train the model on the unsaved model
    pipeline.train_on_unsaved_model()

    # Save the animation of the trained agent
    model = pipeline.load_model("best")
    pipeline.record_replay(model, verbose=True)

    # Evaluate the trained agent
    # eval_result = pipeline.evaluate(
    #     n_eval_episodes=100,
    #     checkpoint="best",
    #     eval_file_name="best_model_eval.yaml",
    # )
