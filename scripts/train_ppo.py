import gymnasium as gym
import matplotlib.pyplot as plt
import os
import numpy as np
from stable_baselines3 import PPO

def train_agent(env_name, timesteps, save_path="", render_mode=None):
    """Train a simple PPO policy with an MLP, available simply via StableBaselines3!"""
    # env = gym.make(env_name)
    env = gym.make(env_name, render_mode=render_mode)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    if save_path != "":
        model.save(save_path)
    env.close()
    return model

if __name__ == "__main__":
    timesteps = 1000000

    if f'humanoid_{timesteps}.zip' not in os.listdir("../models/basic/"):
        print(f"no model found, training new model for {timesteps} timesteps")
        humanoid_model = train_agent('Humanoid-v5', timesteps, render_mode=None) #train humanoid model
        humanoid_model.save(f"../models/basic/humanoid_{timesteps}")
    else:
        print(f"model found, loading cached model!")
        humanoid_model = PPO.load(f"../models/basic/humanoid_{timesteps}.zip")
