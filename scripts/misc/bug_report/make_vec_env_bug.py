from gymnasium.wrappers import ClipReward
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    env = make_vec_env(
        "CartPole-v1",
        n_envs=10,
        seed=42,
        monitor_dir="monitor",
        vec_env_cls=SubprocVecEnv,
        wrapper_class=ClipReward,
        wrapper_kwargs={"min_reward": 0.3, "max_reward": 0.9},
    )
    model = PPO("MlpPolicy", env, batch_size=10, n_steps=100, verbose=1)
    model.learn(total_timesteps=100)

    monitor_csv_path: str = "monitor/0.monitor.csv"
    with open(monitor_csv_path, "r") as f:
        for line in f:
            print(line.strip())

# The output is the following:
# #{"t_start": 1753248007.4024522, "env_id": "CartPole-v1"}
# r,l,t
# 44.0,44,2.169437
# 25.0,25,2.239698
# 20.0,20,2.295889
