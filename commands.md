# Commmand examples

```bash
rm -rfv tensorboard_logs/* saved_models/* video/*

export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=3
```

## Terminology

Rollout is one observe-action-reward cycle. Length varies.

## breakout a2c

```bash
tensorboard --bind_all --logdir tensorboard_logs/
# view http://localhost:6006/

python3 breakout/env_test.py
python3 breakout/train.py --total_timesteps 40000 --reward_threshold 10 --save_model saved_models/breakout_a2c/last_train.zip
python3 breakout/train.py --total_timesteps 40000 --reward_threshold 10 --load_model saved_models/breakout_a2c/best_model.zip --save_model saved_models/breakout_a2c/last_train.zip
# python3 breakout/eval_policy.py --load_model saved_models/breakout_eval_a2c/best_model.zip
python3 breakout/eval.py --load_model saved_models/breakout_a2c/best_model.zip

python3 breakout/train_multi.py --total_timesteps 40000 --reward_threshold 10 --save_model saved_models/breakout_multi_a2c/last_train.zip
python3 breakout/train_multi.py --total_timesteps 40000 --reward_threshold 10 --load_model saved_models/breakout_multi_a2c/best_model.zip --save_model saved_models/breakout_multi_a2c/last_train.zip
# python3 breakout/eval_policy_multi.py --load_model saved_models/breakout_multi_a2c/best_model.zip
python3 breakout/eval_multi.py --load_model saved_models/breakout_multi_a2c/best_model.zip
```

## lander dqn

```bash
tensorboard --bind_all --logdir tensorboard_logs/
# view http://localhost:6006/

python3 lander/env_test.py
python3 lander/train.py --total_timesteps 40000 --reward_threshold 100 --save_model saved_models/lander_dqn/last_train.zip
python3 lander/train.py --total_timesteps 40000 --reward_threshold 100 --load_model saved_models/lander_dqn/best_model.zip --save_model saved_models/lander_dqn/last_train.zip
# python3 lander/eval_policy.py --load_model saved_models/lander_eval_dqn/best_model.zip
python3 lander/eval.py --load_model saved_models/lander_dqn/best_model.zip

# SubprocVecEnv
# python3 lander/train_multi.py --total_timesteps 40000 --reward_threshold 100 --save_model saved_models/lander_multi_dqn/last_train.zip
# python3 lander/train_multi.py --total_timesteps 40000 --reward_threshold 100 --load_model saved_models/lander_multi_dqn/best_model.zip --save_model saved_models/lander_multi_dqn/last_train.zip
# python3 lander/eval_policy_multi.py --load_model saved_models/lander_multi_dqn/best_model.zip
# python3 lander/eval_multi.py --load_model saved_models/lander_multi_dqn/best_model.zip
```
