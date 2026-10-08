import json
import os
from typing import Any

import contgrid
import tqdm
import yaml
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig

from hrl_tl.config.wrapper import TLSB3PipelineConfigReader

if __name__ == "__main__":
    training_config_path: str = "configs/fr_cont/train/3.g/3.g.b_ll_train.yaml"
    spec_file_path: str = "assets/formulae/fourroom/all_formulae_1_cla_1_max_pred_core_undone.json"

    process_id: int = 0
    total_processes: int = 6
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
    # tl_specs = [
    #     "Fpsi_ld & G!psi_lv",
    # ]

    for tl_spec in tqdm.tqdm(tl_specs, desc="Processing specs"):
        training_config_reader = TLSB3PipelineConfigReader.from_yaml(
            training_config_path
        )
        training_config_reader.tl_spec = tl_spec
        training_config_reader.pipeline_config.device = f"cuda:{gpu_id}"
        training_config_reader.pipeline_config.retrain_model = retrain_model
        training_config: SB3PipelineConfig = training_config_reader.to_config()

        pipeline: SB3Pipeline = SB3Pipeline(
            config=training_config, verbose=True
        )

        if not eval_only:
            # Train the model
            pipeline.train_on_unsaved_model()

        model = pipeline.load_model("best")
        # Save the animation
        pipeline.record_replay(model, verbose=True)

        # Evaluate the model
        eval_result = pipeline.evaluate(
            n_eval_episodes=100,
            checkpoint="best",
            eval_file_name="best_model_eval.yaml",
        )
