import os

import gym_multigrid
import gymnasium as gym
import numpy as np
import yaml
from PIL import Image

from hrl_tl.config.env import EnvMakeConfig

if __name__ == "__main__":
    spec_id: int = 0

    yaml_file_name: str = "configs/fourroom/env/hard/full_tensor_fix_ag_fix_gl_2_fix_obs_no_term.yaml"

    model_name: str = os.path.basename(yaml_file_name).replace(".yaml", "")

    with open(yaml_file_name, "r") as f:
        make_dict = yaml.safe_load(f)

    make_config: EnvMakeConfig = EnvMakeConfig(**make_dict)

    # Create environment
    env = gym.make(**make_config.model_dump(context={"flatten": True}))

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
