# Notes

pip install works
pdm has problem with box2d not properly using swig as build dependency
poetry has trouble with pytorch in external download

## pip

```bash
# use with venv
pip install -U pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium[atari,accept-rom-license,box2d]==0.28.1
pip install stable_baselines3
pip install tensorboard
```

## pdm

```bash
# pdm 2.7.4
pdm add https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl
# Takes 3 min
pdm add https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl
pdm add https://download.pytorch.org/whl/cu118/torchaudio-2.0.2%2Bcu118-cp310-cp310-linux_x86_64.whl
pdm add gymnasium[atari,accept-rom-license,box2d]==0.28.1
# build error with swig
pdm add stable_baselines3
pdm add tensorboard
```

## poetry 1.1.4

```bash
# pytorch from pypi
# gives wrong cuda 11.7 version error
poetry add torch tensorboard
poetry add gymnasium==^0.28.1 -E atari -E accept-rom-license -E box2d
poetry add stable_baselines sb3-contrib
poetry install
```

## pytorch 2.0 and poetry 1.5

poetry add torch runs for 400+ seconds

pip install torch
installs a cuda 11.7 version.

pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
is needed for cuda 11.8 version.

```bash
poetry source add --priority supplemental pytorch-cu118 https://download.pytorch.org/whl/cu118
poetry add torch --source pytorch-cu118
```

## Adding pytorch on poetry 1.5

Had to manually specify pytorch in toml. It seems to download way too many versions for dependency check.
Doesn't finish install after 5 minutes.

```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "sb3v2"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
torch = {version = "2.0.1", source = "pytorch_cu118"}

[[tool.poetry.source]]
name = "PyPI"
priority = "primary"

[[tool.poetry.source]]
name = "pytorch_cu118"
url = "https://download.pytorch.org/whl/cu118"
priority = "supplemental"
```

This one uses explicit urls. Need to match python version and os type.
Dependency check still runs for a long time.

```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "sb3v2"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
torch = { url = "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl" }
torchaudio = { url = "https://download.pytorch.org/whl/cu118/torchaudio-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl" }
torchvision = { url = "https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl" }
```

# poetry explicit url

```bash
# poetry 1.1.4
poetry add https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl
# fails to add
```

```bash
# poetry 1.5.1
poetry add https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl
# doesn't resolve dependencies after 500s
```

## gymnasium

Rollout is collecting training experiences... may be 1 rollout is 1 epoch.

## ALE Vectorized

The input is the Atari screen rgb_array.

The make_atari_env() vectorized stacks 4 games in a 2x2 square and plays them
all at once. The rgb_array is 2x2 screens then and the model only works on 2x2.

Uses old gym 0.21 env.reset() and step() return values.
Call render() inline also.
DummyVecEnv uses old gym api also.

## Lander

Apparently stable_baselines3 always uses the vectorized api even for non-graphical
(rgb_array) training environments. It forces a DummyVenEnv.
This also means it uses the old gym 0.21 calling style.
