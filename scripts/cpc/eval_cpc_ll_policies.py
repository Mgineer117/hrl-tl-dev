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
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)

if __name__ == "__main__":
    model_path: str = "out/fr_cont/cpc/pr/3.f.f/cpc_40.0M/best_model.zip"
    spec_path: str = "assets/formulae/fourroom/gc_ltl/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    gcltlt_config_path: str = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_dense_oh_0.35.yaml"
    )
    eval_pipeline_config_path: str = (
        "configs/fr_cont/cpc/pr/3.f.f_train_primitive_eval_cpc.yaml"
    )

    # lambda_config = LambdaConfig(L_gain=5, k_steepness=0.2, eps_margin=1.5)
    lambda_config = LambdaConfig(L_gain=7, k_steepness=0.2, eps_margin=1.5)

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
        policy: CPCCompositePolicy = CPCCompositePolicy(
            tl_spec=tl_spec,
            predicates=gcltlt_config_reader.predicates,
            model=model,
            var_value_info_generator=var_info_generator,
            lambda_config=lambda_config,
            goal_rep="one_hot",
            verbose=True,
        )

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=eval_pipeline_config_reader
        ).to_config()

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

        pipeline.record_replay(policy, verbose=True)
        # eval_stats = pipeline.evaluate(
        #     n_eval_episodes=200, deterministic=False, checkpoint=policy, env="vec"
        # )
