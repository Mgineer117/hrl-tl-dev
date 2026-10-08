import gym_multigrid

from hrl_tl.config.train import SB3BaseTrainingConfig
from hrl_tl.pipeline import SB3Pipeline

if __name__ == "__main__":
    training_config_path: str = (
        "configs/fourroom/train/test_pos_obs_ppo_2.b.h.yaml"
    )
    pipeline: SB3Pipeline[SB3BaseTrainingConfig] = SB3Pipeline[
        SB3BaseTrainingConfig
    ].from_yaml(training_config_path)

    model = pipeline.train_on_unsaved_model()

    # Generate a video of the evaluation
    pipeline.record_replay(model)

    # Evaluate the model
    eval_result = pipeline.evaluate(n_eval_episodes=100, deterministic=False)
