# Notes

## pytorch 2.0

pip install torch
installs a cuda 11.7 version.

pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
is needed for cuda 11.8 version.

## gymnasium

Rollout is collecting training experiences... may be 1 rollout is 1 epoch.

## ALE Vectorized

The input is the Atari screen rgb_array.

The make_atari_env() vectorized stacks 4 games in a 2x2 square and plays them
all at once. The rgb_array is 2x2 screens then and the model only works on 2x2.

Uses old gym 0.21 env.reset() and step() return values.
Call render() inline also.
