import json
import os

from gym_tl_tools import replace_special_characters

if __name__ == "__main__":
    spec_file_path: str = (
        "assets/formulae/fourroom/all_formulae_1_cla_1_max_pred_core.json"
    )
    model_save_dir: str = "out/fr_cont/ltl_ll/3.g.b"
    model_name_pref: str = "ll_policy_fr_cont"
    final_model_filename: str = "final_model_3.g.b_{tl_spec}.zip"
    # spec_file_path: str = "assets/formulae/fourroom/all_primitives.json"
    # model_save_dir: str = "out/fourroom/primitives/pos/3.b.c"
    # model_name_pref: str = "fourroom_tl_prim_dqn_pos_2_obs"

    done_specs: list[str] = []
    undone_specs: list[str] = []
    done_spec_ids: list[int] = []
    undone_spec_ids: list[int] = []
    with open(spec_file_path, "r") as f:
        data = json.load(f)
        tl_specs: list[str] = data["specifications"]
        predicates: list[str] = data["predicates"]
        num_specs: int = data["num_specifications"]

    for spec_id, tl_spec in enumerate(tl_specs):
        replaced_spec: str = replace_special_characters(tl_spec)
        model_name: str = model_name_pref + "_" + replaced_spec
        model_save_path: str = os.path.join(
            model_save_dir,
            model_name,
            final_model_filename.format(tl_spec=replaced_spec),
        )
        if not os.path.exists(model_save_path):
            undone_spec_ids.append(spec_id)
            print(
                f"Model for spec {spec_id} ({tl_spec}) not found at {model_save_path}"
            )
            undone_specs.append(tl_spec)
        else:
            done_spec_ids.append(spec_id)
            done_specs.append(tl_spec)

    if undone_spec_ids:
        print(
            f"Models for the following specs are not run: {', '.join(map(str, undone_spec_ids))}"
        )
        print("Total unrun models:", len(undone_spec_ids))
    else:
        print("All models have been run successfully.")

    # Save the results to a JSON file
    done_spec_file_path: str = spec_file_path.replace(".json", "_done.json")
    undone_spec_file_path: str = spec_file_path.replace(".json", "_undone.json")

    with open(done_spec_file_path, "w") as f:
        json.dump(
            {
                "predicates": predicates,
                "num_specifications": len(done_specs),
                "specifications": done_specs,
            },
            f,
            indent=4,
        )

    with open(undone_spec_file_path, "w") as f:
        json.dump(
            {
                "predicates": predicates,
                "num_specifications": len(undone_specs),
                "specifications": undone_specs,
            },
            f,
            indent=4,
        )
