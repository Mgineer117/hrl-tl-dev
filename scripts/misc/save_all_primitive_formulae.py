import json
import os
from typing import Any

import ruspot as spot

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
    num_processes: int = 128
    specification_save_path: str = (
        "assets/formulae/fourroom/all_primitives.json"
    )

    print(f"Generating all specifications for {len(predicates)} predicates...")
    specifications: list[str] = []
    for predicate in predicates:
        specifications.append(f"F{predicate}")

        other_predicates: list[str] = [p for p in predicates if p != predicate]
        constraint: str = f"F({' | '.join(other_predicates)}) & G!{predicate}"
        simplified_constraint: str = (
            f"{spot.simplify(spot.Formula(constraint))}"
        )
        specifications.append(simplified_constraint)

    # Sort specifications to ensure consistent order
    specifications.sort()

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
