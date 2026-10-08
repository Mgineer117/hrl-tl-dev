import tqdm
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)

from hrl_tl.config.spec import AvailableSpecs
from hrl_tl.config.wrapper import TLSB3PipelineConfigReader

if __name__ == "__main__":
    spec_path: str = "assets/formulae/fourroom/sm/all_primitives.yaml"
    # training_config_path: str = "configs/fourroom/train/pretrain_tl_ppo.yaml"
    training_config_path: str = (
        "configs/fourroom/sm/primitives/3.b/3.b.h_primitives.yaml"
    )

    process_id: int = 4
    total_processes: int = 5
    total_gpus: int = 1
    gpu_id: int = 2  # int(process_id % total_gpus)
    retrain_model: bool = False
    eval_only: bool = False

    print(f"Process ID: {process_id}, GPU ID: {gpu_id}")

    pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        training_config_path
    )

    pipeline_config_reader.device = gpu_id
    pipeline_config_reader.retrain_model = retrain_model

    avail_specs: AvailableSpecs = AvailableSpecs.from_yaml(spec_path)

    # Assign specs for this process with even distribution
    total_specs = len(avail_specs.specifications)
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

    tl_specs: list[str] = avail_specs.specifications[start_index:end_index]

    for tl_spec in tqdm.tqdm(tl_specs, desc="Processing specs"):
        print(f"Processing TL spec: {tl_spec}")

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=pipeline_config_reader
        ).to_config()

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

        pipeline.train_on_unsaved_model()
        model = pipeline.load_model("best")

        # Save the animation
        pipeline.record_replay(model)

        # Evaluate the model
        eval_result = pipeline.evaluate(
            n_eval_episodes=100,
            checkpoint="best",
            eval_file_name="best_model_eval.yaml",
        )
