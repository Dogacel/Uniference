# Jetson Orin Nano Setup

This document explains how to prepare Jetson Orin Nano to be used for running distirbuted models and simulation.

## Initial Setup

You should use a real device (not VM) running Ubuntu 22.04 to install Ubuntu on Jetson Orin Nano using NVIDIA SDK Manager. For more details, follow [NVIDIA's documentation website](https://www.jetson-ai-lab.com/initial_setup_jon_sdkm.html).

## Setting up OS

As we have more than 1 orin, we choose to give them hostnames `orin-X` where X denotes an increasing id of orin, for example 0, 1, 2.

```shell
sudo apt update
sudo hostname orin-X
```

For accessing Jetson cluster from a remote location, we recommend using tailscale.

```shell
curl -fsSL https://tailscale.com/install.sh | sh
sudo vim /etc/sysctl.conf
# Make sure you update the following fields
#net.ipv4.ip_forward=1
#net.ipv6.conf.all.forwarding=1
sudo sysctl -p
sudo tailscale up --advertise-routes=192.168.1.0/24 --advertise-exit-node
```

Later, clone the simulation suite.

```shell
git config --global credential.helper store
git clone https://github.com/Dogacel/uniference.git
# Those steps are only needed when working with a private repository.
# Recommended to use GitHub fine-grained read-only API token.

cd uniference

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync --all-extras

uvx --from huggingface_hub huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --include "original/*" \
  --local-dir checkpoints/Llama-3.2-1B-Instruct

make sanity
```

For installing Llama 3, you need access from Huggingface. If you face issues with getting access to Llama, you can use Llama models from [unsloth AI](https://huggingface.co/unsloth).

## Setting up CUDA

To enable CUDA support for running models on Jetson, you need to install cuSPARSELt and cuDSS along with the right PyTorch version.

```shell
export DEVICE=cuda

sudo apt install nvidia-jetpack

export CUDA_VERSION=12.6

wget https://raw.githubusercontent.com/pytorch/pytorch/a11a66ef320938cd0fd72b44b2b572b06937e100/.ci/docker/common/install_cusparselt.sh
sudo -E bash ./install_cusparselt.sh

# Trick to make cudss installation work
export CUDA_VERSION=12.4
wget -qO install_cudss.sh https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cudss.sh
sudo -E bash ./install_cudss.sh

unset CUDA_VERSION
```

Also make sure you modify `pyproject.toml` to replace torch constraint with the jetson-ai-lab fork,

```toml
"torch @ https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226",
```