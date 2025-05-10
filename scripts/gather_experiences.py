import gymnasium as gym
import numpy as np
import os

def collect_rollout_data(env_name: str, out_dir: str, n_timesteps: int=10000, print_n_episodes: int=1000):
    """Simulates `n_timesteps` in the `env_name` environment, saving observations, rewards, actions, and done to a quadruplet of .npy files at `out_dir`."""
    env = gym.make(env_name, render_mode='rgb_array')
    obs, info = env.reset()
    observations, rewards, actions, done = [], [] , [], []
    episode_count = 0

    for timestep in range(n_timesteps):  # Run for n_timesteps or until the episode ends
        action = env.action_space.sample() #select random action
        obs, reward, terminated, truncated, info = env.step(action) #execute and get results
        observations.append(obs) #save observation
        rewards.append(reward) #save reward
        actions.append(action) #save action
        done.append(terminated or truncated) #save timestep of each episode's boundary
        if terminated or truncated: #check for game over, if so reset env
            episode_count+=1
            if episode_count % print_n_episodes == 0: print(f"finished {episode_count} episodes") #provide update on training
            observation, info = env.reset()
        env.close()
    np_obs, np_rewards, np_actions, np_done = np.array(observations), np.array(rewards), np.array(actions), np.array(done)
    print(f"observations has shape: {np_obs.shape}\trewards has shape: {np_rewards.shape}\tactions has shape: {np_actions.shape}\tdone has shape: {np_done.shape}")
    os.mkdir(f'{out_dir}/{env_name}_{n_timesteps}')
    np.save(f'{out_dir}/{env_name}_{n_timesteps}/observations.npy', np_obs) #load with: new_obs = np.load("../data/processed/Humanoid-v5_10000_rollout_observations.npy")
    np.save(f'{out_dir}/{env_name}_{n_timesteps}/rewards.npy', np_rewards)
    np.save(f'{out_dir}/{env_name}_{n_timesteps}/actions.npy', np_actions)
    np.save(f'{out_dir}/{env_name}_{n_timesteps}/done.npy', np_done)
    return np_obs, np_rewards, np_actions, np_done

if __name__ == "__main__":
    out_dir, n_timesteps, print_n_episodes = "../data/processed/", 1000000, 1000
    collect_rollout_data('Humanoid-v5', out_dir, n_timesteps, print_n_episodes)