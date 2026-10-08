import gym_multigrid
import tqdm

from hrl_tl.config.train import SB3BaseTrainingConfig
from hrl_tl.pipeline import SB3Pipeline

if __name__ == "__main__":
    training_config_path: str = (
        "configs/fourroom/train/2.b/test_tensor_ppo_2.b.q.yaml"
    )
    num_replicate: int = 10

    for rep_idx in tqdm.tqdm(range(num_replicate), desc="Replicates"):
        pipeline: SB3Pipeline[SB3BaseTrainingConfig] = SB3Pipeline[
            SB3BaseTrainingConfig
        ].from_yaml(training_config_path, f"rep_{rep_idx}")

        model = pipeline.train_on_unsaved_model()

        # Generate a video of the evaluation
        pipeline.record_replay(model)

        # Evaluate the model
        eval_result = pipeline.evaluate(
            n_eval_episodes=100, deterministic=False, checkpoint="best"
        )
