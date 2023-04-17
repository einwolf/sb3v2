import argparse
import os

import gymnasium as gym

from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

environment_name = "ALE/Breakout-v5"

log_path = "tensorboard_logs"
a2c_model_path = os.path.join("saved_models", "a2c_model_breakout")


def parse_cmd_line():
    parser = argparse.ArgumentParser(description="Training phase")

    parser.add_argument("--total_timesteps", required=True, type=int,
                    help="Maximum number of training time steps")
    parser.add_argument("--reward_threshold", required=False, type=int, default=100,
                    help="Stop when reward reaches this value")
    parser.add_argument("--load_model", required=False, type=Path, default=False,
                    help="Continue training from this model file")
    parser.add_argument("--save_model", required=False, type=Path, default=False,
                    help="Save model to this file at end of training")

    args = parser.parse_args()
    return args


def make_output_dirs():
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(a2c_model_path, exist_ok=True)


def train_multi():
    """
    Train model using make_atari_env() and vectorized stacking
    """
    # Parse command line
    args = parse_cmd_line()

    print(f"{args.total_timesteps=}")
    print(f"{args.reward_threshold=}")
    print(f"{args.load_model=}")
    print(f"{args.save_model=}")
    print(f"{log_path=}")

    # env = gym.make('ALE/Breakout-v5',
    #     obs_type='rgb',                   # ram | rgb | grayscale
    #     frameskip=4,                      # frame skip
    #     mode=None,                        # game mode, see Machado et al. 2018
    #     difficulty=None,                  # game difficulty, see Machado et al. 2018
    #     repeat_action_probability=0.25,   # Sticky action probability
    #     full_action_space=False,          # Use all actions
    #     render_mode=None                  # None | human | rgb_array
    # )

    # Initialize
    make_output_dirs()

    env = make_atari_env(environment_name, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
    eval_callback = EvalCallback(env, 
                                callback_on_new_best=stop_callback, 
                                eval_freq=1000, 
                                best_model_save_path=a2c_model_path, 
                                verbose=1)

    # Train
    if args.load_model:
        print(f"Load {args.load_model}")
        model.load(path=args.load_model, env=env)

    # The training and eval env mismatch is normal
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    if args.save_model:
        print(f"Save {args.save_model}")
        model.save(path=args.save_model)


if __name__ == "__main__":
    train_multi()
