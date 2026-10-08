import contgrid
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from stable_baselines3 import PPO

from hrl_tl.config.spec import AvailableSpecs
from hrl_tl.config.wrapper import (
    GCLTLWrapperConfigReader,
    TLSB3PipelineConfigReader,
)
from hrl_tl.envs.tl_fourroom import ContRoomsObsVarValueInfoGenerator
from hrl_tl.wrappers.low_level_policies.gc_ltl import GCLTLCompositePolicy

if __name__ == "__main__":
    model_path: str = "out/fr_cont/gcltl/pr/3.e.o/gcltl_5.0M/best_model.zip"
    spec_path: str = "assets/formulae/fourroom/gc_ltl/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    gcltlt_config_path: str = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_penalty_0.4.yaml"
    )
    eval_pipeline_config_path: str = (
        "configs/fr_cont/gcltl/primitive/3.e.o_train_primitive_eval.yaml"
    )

    var_info_generator = ContRoomsObsVarValueInfoGenerator()

    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        eval_pipeline_config_path
    )
    gcltlt_config_reader = GCLTLWrapperConfigReader.from_yaml(
        gcltlt_config_path
    )

    avail_specs: AvailableSpecs = AvailableSpecs.from_yaml(spec_path)
    model = PPO.load(model_path)

    for tl_spec in avail_specs.specifications:
        print(f"Evaluating TL spec: {tl_spec}")
        policy: GCLTLCompositePolicy = GCLTLCompositePolicy(
            tl_spec=tl_spec,
            predicates=gcltlt_config_reader.predicates,
            model=model,
            switch_threshold=-0.5,
            action_sum_coeff=0.5,
            threshold_type="robustness",
            discrete_state_space=False,
            var_value_info_generator=var_info_generator,
            verbose=True,
        )

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=eval_pipeline_config_reader
        ).to_config()

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

        pipeline.record_replay(policy, verbose=True)
        # eval_stats = pipeline.evaluate(
        #     n_eval_episodes=100,
        #     deterministic=True,
        #     checkpoint=policy,
        # )
