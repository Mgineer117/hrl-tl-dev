import json
import os
from typing import Any

from hrl_tl.wrappers.utils import generate_all_reach_avoid_specifications

if __name__ == "__main__":
    predicates: list[str] = [
        "psi_ld",
        "psi_bd",
        "psi_td",
        "psi_rd",
        "psi_gl",
        "psi_lv",
        "psi_hl",
    ]
    num_clauses: int = 1
    max_num_predicates: int = 1
    num_processes: int = 64
    specification_save_path: str = f"assets/formulae/fourroom/all_formulae_{num_clauses}_cla_{max_num_predicates}_max_pred_ra.json"

    print(f"Generating all specifications for {len(predicates)} predicates...")
    specifications: list[str] = generate_all_reach_avoid_specifications(
        predicates, num_processes, num_clauses, max_num_predicates
    )

    print(f"Saving specifications to {specification_save_path}...")
    print(f"Number of specifications generated: {len(specifications)}")
    saved_data: dict[str, Any] = {
        "predicates": predicates,
        "num_specifications": len(specifications),
        "specifications": specifications,
    }
    os.makedirs(os.path.dirname(specification_save_path), exist_ok=True)
    with open(specification_save_path, "w") as f:
        json.dump(saved_data, f, indent=4)
