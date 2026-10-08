import yaml
from contgrid.envs import RoomsEnv, RoomsScenario, RoomsScenarioConfig
from stable_baselines3 import PPO

env_config_path: str = "configs/fr_cont/env/eval/full_pos_disc_11_slow_fix_ag_fix_gl_2_fix_obs_no_term_250.yaml"
model_path: str = "out/fr_cont/gcltl/pr/3.e.i/gcltl_10.0M/best_model.zip"
model = PPO.load(model_path)
