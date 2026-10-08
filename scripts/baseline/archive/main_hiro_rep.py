import gym_multigrid
import tqdm

from hrl_tl.config.train import HiroTrainingConfig
from hrl_tl.pipeline import HiroPipeline

if __name__ == "__main__":
    # Example usage of HiroPipeline
    config_path: str = "configs/fourroom/train/hiro/train_hiro_4.a.c.yaml"
    num_replicates: int = 10
    for replicate_idx in tqdm.tqdm(range(num_replicates), desc="Replicates"):
        replicate_dir: str = f"rep_{replicate_idx}"
        hiro_pipeline: HiroPipeline[HiroTrainingConfig] = HiroPipeline[
            HiroTrainingConfig
        ].from_yaml(config_path, replicate_dir=replicate_dir)

        model = hiro_pipeline.train_on_unsaved_model()
        hiro_pipeline.record_replay(model, verbose=False)
        eval_result = hiro_pipeline.evaluate(n_eval_episodes=100)
