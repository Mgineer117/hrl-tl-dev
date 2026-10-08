import json
import os
from typing import Any

import gym_multigrid
import tqdm
import yaml

from hrl_tl.config import SB3LowLevelTrainingConfig
from hrl_tl.pipeline import SB3Pipeline

if __name__ == "__main__":
    spec_file_path: str = "assets/formulae/fourroom/all_primitives.json"
    # training_config_path: str = "configs/fourroom/train/pretrain_tl_ppo.yaml"
    training_config_path: str = "configs/fourroom/train/3.b/3.b.f_pretrain_primitives_dqn_tensor_2_obs.yaml"

    process_id: int = 0
    total_processes: int = 1
    total_gpus: int = 3
    gpu_id: int = 2  # int(process_id % total_gpus)
    retrain_model: bool = False
    eval_only: bool = False

    print(f"Process ID: {process_id}, GPU ID: {gpu_id}")

    with open(spec_file_path, "r") as f:
        spec_data: dict[str, Any] = json.load(f)

    # Assign specs for this process with even distribution
    total_specs = len(spec_data["specifications"])
    base_specs_per_process = total_specs // total_processes
    extra_specs = total_specs % total_processes

    # Calculate start and end indices for even distribution
    if process_id < extra_specs:
        # First 'extra_specs' processes get one additional spec
        start_index = process_id * (base_specs_per_process + 1)
        end_index = start_index + base_specs_per_process + 1
    else:
        # Remaining processes get base number of specs
        start_index = (
            extra_specs * (base_specs_per_process + 1)
            + (process_id - extra_specs) * base_specs_per_process
        )
        end_index = start_index + base_specs_per_process

    tl_specs: list[str] = spec_data["specifications"][start_index:end_index]

    for tl_spec in tqdm.tqdm(tl_specs, desc="Processing specs"):
        with open(training_config_path, "r") as f:
            training_config_dict: dict[str, Any] = yaml.safe_load(f)
            training_config_dict.update(
                {
                    "tl_spec": tl_spec,
                    "gpu_id": gpu_id,
                    "retrain_model": retrain_model,
                }
            )
            training_config = SB3LowLevelTrainingConfig(**training_config_dict)

        pipeline: SB3Pipeline[SB3LowLevelTrainingConfig] = SB3Pipeline(
            training_config
        )

        if not os.path.exists(training_config.model_save_path):
            if eval_only:
                print(
                    f"Model {training_config.model_full_name} does not exist, skipping evaluation..."
                )
                print(f"- Path: {training_config.model_save_path}")
                continue
            else:
                print(
                    f"Model {training_config.model_full_name} does not exist, training..."
                )
                model = pipeline.train()
        elif training_config.retrain_model:
            print(
                f"Model {training_config.model_full_name} already exists, but retraining..."
            )
            model = pipeline.train()
        else:
            print(
                f"Model {training_config.model_full_name} already exists, loading..."
            )
            model = pipeline.load_model()

        # Save the animation
        pipeline.record_replay(model)

        # Evaluate the model
        eval_result = pipeline.evaluate(n_eval_episodes=100, checkpoint=model)
