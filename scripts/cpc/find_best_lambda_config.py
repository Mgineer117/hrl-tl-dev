import json
from itertools import product

import contgrid
import numpy as np
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from stable_baselines3 import PPO
from tqdm import tqdm

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


def evaluate_lambda_config(
    L_gain: float,
    k_steepness: float,
    eps_margin: float,
    model,
    var_info_generator,
    eval_pipeline_config_reader,
    gcltlt_config_reader,
    avail_specs,
) -> dict:
    """
    Evaluate a single lambda configuration across all specifications.
    """
    lambda_config = LambdaConfig(
        L_gain=L_gain, k_steepness=k_steepness, eps_margin=eps_margin
    )

    success_rates: list[float] = []
    failure_rates: list[float] = []

    for tl_spec in avail_specs.specifications:
        policy: CPCCompositePolicy = CPCCompositePolicy(
            tl_spec=tl_spec,
            predicates=gcltlt_config_reader.predicates,
            model=model,
            var_value_info_generator=var_info_generator,
            lambda_config=lambda_config,
            for_eval=True,
            goal_rep="one_hot",
            verbose=False,
        )

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=eval_pipeline_config_reader
        ).to_config()

        pipeline = SB3Pipeline(config=pipeline_config, verbose=False)

        eval_stats = pipeline.evaluate(
            n_eval_episodes=200,
            deterministic=False,
            checkpoint=policy,
            save_to_file=False,
            env="single",
        )

        # Extract success rate if available
        if eval_stats.success_rate is not None:
            success_rates.append(eval_stats.success_rate)

        if eval_stats.failure_rate is not None:
            failure_rates.append(eval_stats.failure_rate)

    # Calculate statistics
    avg_success_rate = float(np.mean(success_rates)) if success_rates else 0.0

    result = {
        "L_gain": L_gain,
        "k_steepness": k_steepness,
        "eps_margin": eps_margin,
        "avg_success_rate": avg_success_rate,
        "std_success_rate": float(np.std(success_rates))
        if success_rates
        else 0.0,
        "min_success_rate": float(np.min(success_rates))
        if success_rates
        else 0.0,
        "max_success_rate": float(np.max(success_rates))
        if success_rates
        else 0.0,
        "avg_failure_rate": float(np.mean(failure_rates))
        if failure_rates
        else 0.0,
        "std_failure_rate": float(np.std(failure_rates))
        if failure_rates
        else 0.0,
        "min_failure_rate": float(np.min(failure_rates))
        if failure_rates
        else 0.0,
        "max_failure_rate": float(np.max(failure_rates))
        if failure_rates
        else 0.0,
        "num_specs": len(success_rates),
    }

    return result


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_path: str = "out/fr_cont/cpc/pr/3.f.d/cpc_40.0M/best_model.zip"
    spec_path: str = "assets/formulae/fourroom/gc_ltl/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    gcltlt_config_path: str = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_dense_0.35.yaml"
    )
    eval_pipeline_config_path: str = (
        "configs/fr_cont/cpc/pr/3.f.d_train_primitive_eval_cpc.yaml"
    )
    output_file = "lambda_config_search_results_hl.json"

    # Initialize resources once
    print("Loading model and configuration files...")
    var_info_generator = ContRoomsObsVarValueInfoGenerator()
    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        eval_pipeline_config_path
    )
    gcltlt_config_reader = GCLTLWrapperConfigReader.from_yaml(
        gcltlt_config_path
    )
    avail_specs = AvailableSpecs.from_yaml(spec_path)
    model = PPO.load(model_path, device="cuda:0")
    print("Resources loaded successfully.\n")

    # Define search space for lambda_config parameters
    L_gain_values = [3.0, 5.0, 7.0, 9.0]
    k_steepness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    eps_margin_values = [0.3, 0.5, 0.7, 1.0]

    # Generate all parameter combinations
    param_combinations = list(
        product(L_gain_values, k_steepness_values, eps_margin_values)
    )
    total_combinations = len(param_combinations)

    print(
        f"Starting grid search with {total_combinations} parameter combinations"
    )
    print(f"Using sequential processing")
    print(f"{'=' * 80}\n")

    # Store results for each parameter combination
    results: list[dict] = []
    best_avg_success_rate = 0.0
    best_lambda_config = None

    # Sequential grid search with progress tracking
    try:
        with tqdm(
            total=total_combinations, desc="Grid Search Progress"
        ) as pbar:
            for L_gain, k_steepness, eps_margin in param_combinations:
                result = evaluate_lambda_config(
                    L_gain=L_gain,
                    k_steepness=k_steepness,
                    eps_margin=eps_margin,
                    model=model,
                    var_info_generator=var_info_generator,
                    eval_pipeline_config_reader=eval_pipeline_config_reader,
                    gcltlt_config_reader=gcltlt_config_reader,
                    avail_specs=avail_specs,
                )
                results.append(result)

                # Update best configuration
                if result["avg_success_rate"] > best_avg_success_rate:
                    best_avg_success_rate = result["avg_success_rate"]
                    best_lambda_config = LambdaConfig(
                        L_gain=result["L_gain"],
                        k_steepness=result["k_steepness"],
                        eps_margin=result["eps_margin"],
                    )
                    tqdm.write(
                        f"\n*** NEW BEST: L_gain={result['L_gain']}, "
                        f"k_steepness={result['k_steepness']}, "
                        f"eps_margin={result['eps_margin']}, "
                        f"avg_success_rate={result['avg_success_rate']:.4f}, "
                        f"avg_failure_rate={result['avg_failure_rate']:.4f} ***"
                    )

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "best_sr": f"{best_avg_success_rate:.4f}",
                        "current_sr": f"{result['avg_success_rate']:.4f}",
                        "current_fr": f"{result['avg_failure_rate']:.4f}",
                    }
                )

    except KeyboardInterrupt:
        print("\n\nGrid search interrupted by user (Ctrl+C)")
        print("Saving partial results...")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        raise

    # Print final results
    if not results:
        print("\n\nNo results collected. Exiting.")
        exit(0)

    # Get best result from collected results
    sorted_results = sorted(
        results, key=lambda x: x["avg_success_rate"], reverse=True
    )
    if sorted_results:
        best_result = sorted_results[0]
        best_lambda_config = LambdaConfig(
            L_gain=best_result["L_gain"],
            k_steepness=best_result["k_steepness"],
            eps_margin=best_result["eps_margin"],
        )
        best_avg_success_rate = best_result["avg_success_rate"]

    assert best_lambda_config is not None
    print(f"\n\n{'=' * 80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nBest Lambda Configuration:")
    print(f"  L_gain: {best_lambda_config.L_gain}")
    print(f"  k_steepness: {best_lambda_config.k_steepness}")
    print(f"  eps_margin: {best_lambda_config.eps_margin}")
    print(f"  Average Success Rate: {best_avg_success_rate:.4f}")

    # Print top 5 configurations
    print(f"\n\nTop 5 Configurations:")
    sorted_results = sorted(
        results, key=lambda x: x["avg_success_rate"], reverse=True
    )
    for i, result in enumerate(sorted_results[:5]):
        print(
            f"\n{i + 1}. L_gain={result['L_gain']}, k_steepness={result['k_steepness']}, eps_margin={result['eps_margin']}"
        )
        print(
            f"   Avg Success Rate: {result['avg_success_rate']:.4f} ± {result['std_success_rate']:.4f}"
        )
        print(
            f"   Range: [{result['min_success_rate']:.4f}, {result['max_success_rate']:.4f}]"
        )
        print(
            f"   Avg Failure Rate: {result['avg_failure_rate']:.4f} ± {result['std_failure_rate']:.4f}"
        )
        print(
            f"   Range: [{result['min_failure_rate']:.4f}, {result['max_failure_rate']:.4f}]"
        )

    # Save results to file (including partial results if interrupted)
    output_data = {
        "best_config": {
            "L_gain": best_lambda_config.L_gain,
            "k_steepness": best_lambda_config.k_steepness,
            "eps_margin": best_lambda_config.eps_margin,
            "avg_success_rate": best_avg_success_rate,
        },
        "all_results": sorted_results,
        "total_evaluated": len(results),
        "total_planned": total_combinations,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
    print(f"Evaluated {len(results)}/{total_combinations} configurations")
