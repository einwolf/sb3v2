import os

import gymnasium as gym
import ale_py

from stable_baselines3.common.env_checker import check_env

gym.register_envs(ale_py)

environment_name = "ALE/Breakout-v5"

def env_test():
    """
    Test environment with random actions
    """
    env = gym.make(environment_name, render_mode="rgb_array")
    check_env(env, warn=True)

    action_test = env.action_space.sample()
    print(f"{action_test=}")

    obs_test = env.observation_space.sample()
    print(f"{obs_test=}")

    episodes = 1
    for episode in range(1, episodes+1):
        obs, info = env.reset()
        done = False
        step = 0
        score = 0 
        
        while not done:
            frame = env.render()
            action = env.action_space.sample()
            n_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step += 1
            score += reward

            print(f"Step:{step} Score:{score}")

        print(f"Episode:{episode} Final score:{score}")

    env.close()


if __name__ == "__main__":
    env_test()
