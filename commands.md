# Commmand examples

```bash
tensorboard --bind_all --logdir tensorboard_logs/A2C_1/
# view http://localhost:6006/

python3 breakout/env_test.py
python3 breakout/train.py --total_timesteps 40000 --reward_threshold 10 --save_model saved_models/a2c_model_breakout/last_train.zip
# python3 breakout/eval_policy.py --load_model saved_models/a2c_model_breakout/best_model.zip
python3 breakout/eval.py --load_model saved_models/a2c_model_breakout/best_model.zip

python3 breakout/train_multi.py --total_timesteps 40000 --reward_threshold 10 --save_model saved_models/a2c_model_breakout/last_train.zip
python3 breakout/eval_policy_multi.py --load_model saved_models/a2c_model_breakout/best_model.zip
python3 breakout/eval_multi.py --load_model saved_models/a2c_model_breakout/best_model.zip
```
