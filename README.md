# Edge Sim Suite

Simulation suite for running transformer models on edge devices in distributed settings.

![Simulation Suite Overview](https://i.imgur.com/MVGREdV.png)

## Quickstart

We recommend running _Llama 3_ models. To get started, download the LLama model weights into checkpoints directory.

```shell

mkdir checkpoints
uvx --from huggingface_hub huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --include "original/*" \
  --local-dir checkpoints/Llama-3.2-1B-Instruct

```

A pre-determined set of scenarios and programs are listed under `scenarios` folder. You can use the `run.sh` helper script to run a scenario to get started.

```shell

uv sync --extra torch --extra analysis
./run.sh models.llama3.scripts.failover_simple_scenario

```

## Codebase

This project is initially forked from [llama-models](https://github.com/meta-llama/llama-models/). Therefore inherits some utilities and folders from the upstream llama project.

- **event-visualization**: A web page for replaying a recorded simulation.
- **models**: Keeps transformer models that are run in simulations.
  - Files `checkpoint.py`, `datatypes.py`, `quantize_impls.py` and `tokenizer_utils.py` are taken from the upstream `llama-models` repository. They are common dependencies used by different llama models.
  - Currently Llama models are preferred to be modified for distributed settings, however there is no limitation on the type of models can be used.
- **programs**: Stores programs, that can be loaded into devices to be run by them. Each program should use some `models`.
- **scenarios**: Contains scenarios, which aim to imitate the real-life like scenarios for testing the model performance. Scenarios should use some `programs`.
- **simsuite**: Source code for the simulation software that is used to emulate a multi-device real-life like scenario.

## Adding new models

You should add your models that implement new types of distributed inference algorithms under `models` directory.

Easiest way to get started is to inspect the `llama3_ha` model, which emulates KV-Cache and generated token synchronization between two devices.

Inside your model, you can access the simulated device and world by passing it via the program.

## Writing new programs

In order to run your model on a device, you should write a wrapper program around it. Programs control how your device should interact with the model.

To get started, inspect the `ping_pong_program`. This program waits until all devices have generated a token until it proceeds to generate the next token.

## Creating new scenarios

Here are some key elements while creating scenarios, which describe the world where each device exists, their capabilities and how they connect to each other.

- **World**: Describes a simulation world, where devices live in.
- **Device**: TODO 
- **Chan**: TODO

## Running simulations

Currently only way to run a simulation is to run it on your local device. You can do this by using the `./run.sh` utility.

```shell
./run.sh scenarios.synchronize_scenario --device_count=2 --prompt="Don't say anything else, just count from 1 to 10."
```

## Experiment Setup

### Jetson Nano Orin DevKit

Use the following commands to setup the benchmark.

```shell
sudo hostname orin-X

curl -fsSL https://tailscale.com/install.sh | sh
sudo vim /etc/sysctl.conf
#net.ipv4.ip_forward=1
#net.ipv6.conf.all.forwarding=1
sudo sysctl -p
sudo tailscale up --advertise-routes=192.168.1.0/24 --advertise-exit-node

git config --global credential.helper store
git clone https://github.com/Dogacel/edge-llm-benchmark.git
# Use generated read-only API token

cd edge-llm-benchmark
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync --all-extras

uvx --from huggingface_hub huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --include "original/*" \
  --local-dir checkpoints/Llama-3.2-1B-Instruct

make sanity
```

For CUDA support,

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

# Modify pyproject.toml to replace torch constraint with this,
# "torch @ https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226",
```

#### Tensor Parallelism

To run tensor parallel example on multiple devices, run the following for each device.

Leader device,

```shell
# Note the local ip of the leader, usually in form 192.168.x.x or 10.0.0.x
ip -4 addr show scope global | grep inet

export MASTER_ADDR=192.168.1.14
export MASTER_PORT=25001
export GLOO_SOCKET_IFNAME=enP8p1s0

# Set simulation backend to pytorch
export WORLD_BACKEND=pytorch

export RANK=0

# Update world size if needed
export WORLD_SIZE=2

# Jetson Nano doesn't work with nccl
export DIST_BACKEND=gloo
```

Follower devices,

```shell
# Replace master addr with the result found above.
export MASTER_ADDR=192.168.1.14
export MASTER_PORT=25001
export GLOO_SOCKET_IFNAME=enP8p1s0

# Set simulation backend to pytorch
export WORLD_BACKEND=pytorch

# Update rank for each device
export RANK=1

# Update world size if needed
export WORLD_SIZE=2

# Jetson Nano doesn't work with nccl
export DIST_BACKEND=gloo
```

## Network Simulation

Run the following command in both networks to learn about the network parameters.

```shell
uv run benchmarks/network.py --mode=all_gather
```

### Injecting Latency and Limiting Bandwidth

*WIP:* Toxiproxy doesn't work with GLOO backend because it only uses 25001 for randezvous.

```shell
wget https://github.com/Shopify/toxiproxy/releases/download/v2.7.0/toxiproxy-server-linux-arm64 -O toxiproxy-server
wget https://github.com/Shopify/toxiproxy/releases/download/v2.7.0/toxiproxy-cli-linux-arm64 -O toxiproxy-cli
chmod +x toxiproxy-server toxiproxy-cli

# On a separate session
./toxiproxy-server

./toxiproxy-cli create -l 0.0.0.0:5000 -u 192.168.1.14:25001 mylink

./toxiproxy-cli toxic add -t latency -a latency=10 -u mylink
./toxiproxy-cli toxic add -t latency -a latency=10 -d mylink
./toxiproxy-cli toxic add -t bandwidth -a rate=12500 -u mylink
./toxiproxy-cli toxic add -t bandwidth -a rate=12500 -d mylink

# On server
iperf3 -s -p 25001

# On client
iperf3 -c localhost -p 5000 -t 30

sudo apt install hping3
sudo hping3 -S -p 5000 -c 5 127.0.0.1
```