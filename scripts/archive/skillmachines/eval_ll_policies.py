from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)

from hrl_tl.config.spec import AvailableSpecs
from hrl_tl.config.wrapper import TLSB3PipelineConfigReader
from hrl_tl.wrappers.low_level_policies.sm import (
    CompositeLowLevelPolicyConfig,
    CompositeLowLevelPolicyConfigReader,
    DQNSkillMachinePolicy,
    PrimitivePipelineMapConfigReader,
    SkillMachinesLowLevelPolicyBuffer,
)

if __name__ == "__main__":
    ll_config_reader = CompositeLowLevelPolicyConfigReader(
        composite_policy_class="hrl_tl.wrappers.low_level_policies.sm.PPOCompositePolicy",
        primitive_pipeline_map_config=PrimitivePipelineMapConfigReader(
            prim_spec_map_config_file="configs/fourroom/sm/primitives/primitives_small.yaml",
            pipeline_config_file="configs/fourroom/sm/primitives/3.b/3.b.h_primitives_eval.yaml",
        ),
    )
    ll_config: CompositeLowLevelPolicyConfig = ll_config_reader.to_config()

    spec_path: str = (
        "assets/formulae/fourroom/sm/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    )

    buffer = SkillMachinesLowLevelPolicyBuffer(ll_config)

    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        ll_config_reader.primitive_pipeline_map_config.pipeline_config_file
    )
    eval_pipeline_config_reader.save_config.models_dir += "/eval"
    avail_specs: AvailableSpecs = AvailableSpecs.from_yaml(spec_path)

    for tl_spec in avail_specs.specifications:
        print(f"Evaluating TL spec: {tl_spec}")

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=eval_pipeline_config_reader
        ).to_config()

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

        assert isinstance(
            ll_config.composite_policy_class, type(DQNSkillMachinePolicy)
        )

        policy: DQNSkillMachinePolicy = ll_config.composite_policy_class(
            tl_spec=tl_spec,
            primitives=buffer.primitives,
            primitive_model_map=buffer.primitive_model_map,
        )
        pipeline.record_replay(policy)
        # eval_stats = pipeline.evaluate(
        #     n_eval_episodes=100,
        #     deterministic=True,
        #     checkpoint=policy,
        # )
