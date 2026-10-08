import os

import gym_multigrid
import gymnasium as gym
import numpy as np
from gym_tl_tools import (
    Predicate,
    TLObservationReward,
    replace_special_characters,
)
from PIL import Image

from hrl_tl.envs.tl_fourroom import var_value_info_generator

if __name__ == "__main__":
    spec_id: int = 0

    tl_specs: list[str] = [
        "F(psi_td) & G(!psi_lv)",
        "F(psi_ld) & G(!psi_hl)",
        "F(psi_gl)",
    ]
    tl_spec: str = tl_specs[spec_id]
    predicates: list[Predicate] = [
        Predicate(name="psi_ld", formula="d_ld < 0.5"),
        Predicate(name="psi_bd", formula="d_bd < 0.5"),
        Predicate(name="psi_td", formula="d_td < 0.5"),
        Predicate(name="psi_rd", formula="d_rd < 0.5"),
        Predicate(name="psi_gl", formula="d_gl < 0.5"),
        Predicate(name="psi_lv", formula="d_lv < 0.5"),
        Predicate(name="psi_hl", formula="d_hl < 0.5"),
    ]

    model_name: str = "maze_tl_ppo_test_rand_" + replace_special_characters(
        tl_spec
    )
    max_episode_steps: int = 100

    env_kwargs = {
        "max_episode_steps": max_episode_steps,
        "spawn_type": spec_id,
        "random_init_pos": False,  # Disable random init pos for consistent rendering
        "layout_config": {
            "field_map": [
                "#############",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#     #     #",
                "## ####     #",
                "#     ### ###",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#############",
            ],
            "spawn_configs": [
                {
                    "agent": (3, 3),
                    "goal": {"pos": (9, 4), "reward": 1.0},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0},
                        {"pos": (9, 2), "reward": 0},
                        {"pos": (11, 1), "reward": 0},
                        {"pos": (5, 3), "reward": 0},
                        {"pos": (3, 5), "reward": 0},
                        {"pos": (3, 2), "reward": 0},
                        {"pos": (5, 9), "reward": -1},
                        {"pos": (3, 8), "reward": -1},
                        {"pos": (2, 11), "reward": -1},
                        {"pos": (10, 8), "reward": -1},
                        {"pos": (8, 9), "reward": -1},
                        {"pos": (7, 11), "reward": -1},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0},
                        {"pos": (10, 5), "reward": 0},
                        {"pos": (8, 6), "reward": 0},
                        {"pos": (4, 4), "reward": -1},
                        {"pos": (2, 3), "reward": -1},
                        {"pos": (1, 1), "reward": -1},
                        {"pos": (2, 7), "reward": 0},
                        {"pos": (1, 9), "reward": 0},
                        {"pos": (4, 10), "reward": 0},
                        {"pos": (7, 8), "reward": -1},
                        {"pos": (9, 10), "reward": -1},
                        {"pos": (11, 11), "reward": -1},
                    ],
                },
            ],
        },
    }
    wrapper_kwargs = {
        "tl_spec": tl_spec,
        "atomic_predicates": predicates,
        "var_value_info_generator": var_value_info_generator,
        "reward_config": {
            "terminal_state_reward": 0,
            "state_trans_reward_scale": 10,
            "dense_reward": True,
            "dense_reward_scale": 0.01,
        },
        "early_termination": True,
    }

    # Create environment
    env = gym.make(
        "multigrid-rooms-v0",
        render_mode="rgb_array",
        **env_kwargs,
    )
    env = TLObservationReward(
        env,
        **wrapper_kwargs,
    )

    # Reset environment to get initial observation
    obs, _ = env.reset()

    # Render the initial state
    initial_frame = env.render()

    # Ensure we have a valid frame
    if initial_frame is None:
        print("Error: Failed to render the environment")
        env.close()
        exit(1)

    # Convert to numpy array if it's not already
    if not isinstance(initial_frame, np.ndarray):
        initial_frame = np.array(initial_frame)

    # Save as PNG
    output_dir = "out/plots/"
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{model_name}_initial.png")

    # Convert numpy array to PIL Image and save
    image = Image.fromarray(initial_frame.astype(np.uint8))
    image.save(image_path)

    env.close()
    print(f"Initial maze rendering saved to: {image_path}")
