import gym_multigrid

from hrl_tl.config.train import HiroTrainingConfig
from hrl_tl.pipeline import HiroPipeline

if __name__ == "__main__":
    # Example usage of HiroPipeline
    config_path: str = "configs/fourroom/train/hiro/train_hiro_4.a.c.yaml"
    hiro_pipeline: HiroPipeline[HiroTrainingConfig] = HiroPipeline[
        HiroTrainingConfig
    ].from_yaml(config_path)

    model = hiro_pipeline.train_on_unsaved_model()
    hiro_pipeline.record_replay(model, verbose=False)
    eval_result = hiro_pipeline.evaluate(n_eval_episodes=100)
