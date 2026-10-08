import os

import contgrid
import gym_multigrid
import gymnasium as gym
import numpy as np
import yaml
from PIL import Image
from rl_pipeline.gymnasium import GymEnvConfig, GymEnvLoader, MakeEnvConfig

if __name__ == "__main__":
    model_name: str = "nr"
    yaml_file_name: str = "configs/nr/env/full_pos_ang_8_fix_ag_gauss_obj_std_no_term_500_0.35.yaml"

    with open(yaml_file_name, "r") as f:
        make_dict = yaml.safe_load(f)

    make_config: MakeEnvConfig = MakeEnvConfig(**make_dict)
    gym_config = GymEnvConfig(make_env_config=make_config)
    env_loader = GymEnvLoader(gym_config)

    # Create environment
    env = env_loader.env()

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
