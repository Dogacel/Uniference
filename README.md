# Edge Sim Suite

Simulation suite for running transformer models on edge devices in distributed settings.

![Simulation Suite Overview](https://i.imgur.com/MVGREdV.png)

## Additional Resources

- [Jetson Setup](docs/jetson_setup.md)
- [Modifying Network Conditions](docs/modifying_network_conditions.md)

## Quickstart

You need `uv` to run the models.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --all-extras
```

We recommend running _Llama 3_ models. To get started, download the LLama model weights into checkpoints directory.

```shell

mkdir checkpoints
uvx --from huggingface_hub huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --include "original/*" \
  --local-dir checkpoints/Llama-3.2-1B-Instruct

```

You can use the `Makefile` to make a sanity run.

```shell
make sanity # CPU only
make sanity_cuda # If you have a CUDA device
make sanity_mps # If you have an MPS device
```

## Codebase

This project is initially forked from [llama-models](https://github.com/meta-llama/llama-models/). Therefore inherits some utilities and folders from the upstream llama project.

- **models**: Keeps transformer models that are run in simulations.
  - Files `checkpoint.py`, `datatypes.py`, `quantize_impls.py` and `tokenizer_utils.py` are taken from the upstream `llama-models` repository. They are common dependencies used by different llama models.
  - Currently Llama models are preferred to be modified for distributed settings, however there is no limitation on the type of models can be used.
- **programs**: Stores programs, that can be loaded into devices to be run by them. Each program should use some `models`.
- **benchmarks**: Consists programs to run benchmarks of certain algorithms such as `TP` or `Voltage`.
- **notebooks**: Consists Jupyter notebooks that are used to evaluate algorithms or the simulation software itself throughout its development.
- **scenarios**: (DEPRECATED), consists some basic scenarios for testing distributed algorithms.
- **simsuite**: Source code for the simulation software that is used to emulate a multi-device real-life like scenario.

## Adding new models

You should add your models that implement new types of distributed inference algorithms under `models` directory.

Easiest way to get started is to inspect the `llama3_ha` model, which emulates KV-Cache and generated token synchronization between two devices.

Inside your model, you can access the simulated device and world by passing it via the program.

## Writing new programs

In order to run your model on a device, you should write a wrapper program around it. Programs control how your device should interact with the model.

To get started, inspect the `ping_pong_program`. This program waits until all devices have generated a token until it proceeds to generate the next token.

## Creating new benchmarks

TODO: Take a look at ... for example.

## Deployment

You can directly deploy the models on devices by setting a few environment variables. First you need the ip address of the leader and it should be accessible from the followers.

```shell
# Note the local ip of the leader, usually in form 192.168.x.x or 10.0.0.x
ip -4 addr show scope global | grep inet
```

Later, save this value and set the following variables for each device, including the leader.

```shell
export MASTER_ADDR=192.168.2.103
export MASTER_PORT=25001
# Use the ifsocket you use to connect devices to each other.
export GLOO_SOCKET_IFNAME=enP8p1s0
# Set simulation backend to pytorch
export WORLD_BACKEND=pytorch
```

Based on your deployment size, you should set the world_size and rank. Make sure each device has a distinct rank.

```shell
export WORLD_SIZE=2
export RANK=0
```

Later, you will pick-up a pytorch.distributed backend. It can take three values: `nccl`, `gloo` and `mpi`. For most deployments, we recommend using `gloo`. Only HPC clusters with inifiniband connection and proper NVIDIA drivers support `nccl`.

```shell
export DIST_BACKEND=gloo
```

## Conducting Experiments

This section explains how to experiment different algorithms with the simulation software,

### Measuring Network Parameters 

To accurately model our world, we need to run some benchmarks to figure out our Network parameters. Run the following command in all devices to learn about the current network's parameters.

```shell
uv run benchmarks/network.py --mode=all_gather
```

For proper evaluation of your network conditions, you can use the `all_gather` Notebook under `notebooks` folder.

### Adjusting the World Parameters

You can create the network with two parameters, _latency_ and _bandwidth_. 

```python
world.network(
    NetworkArgs(
        devices=devices,
        network_params=[latency, 1/bandwidth],
    )
)
```

If you need to modify network in-between the experiments, you can directly access the `world.networks` object.

```python
world.networks[0].network_params = [new_latency, 1/new_bandwidth]
```

### Creating a "Parameter Search Grid Space"

You might want to test your algorithm with different inputs, device speeds, network conditions etc. To do that, create a parameter grid:

```python
text_lengths = [128, 256, 512, 1024]
speed = [10 * Mbps, 100 * Mbps, 1 * Gbps]
latency = [1 * ms, 5 * ms, 20 * ms]
repeats = 100

# Make sure world is created only once
world = setup_world(...)

# Generate Cartesian product
combinations = list(itertools.product(text_lengths, speed, latency, range(repeats)))

for combo in combinations:
    sequence_length, speed, latency, repeat_idx = combo

    # Update world
    world.networks[0].network_params = [latency, 1/speed]

    world.run()
```

You can also access world devices and speed them up or slow them down.

```python
for device in world.devices:
    device.spec.speed_scale = 1.3
```

### Running experiments

To run experiments of multiple devices on a single device, 

```shell
export MASTER_ADDR=localhost
export MASTER_PORT=25001
export WORLD_SIZE=1
export RANK=0

export DEVICE=mps # Change it to cpu/cuda based on your hardware
uv run python benchmarks/real_tp.py --device_count=4
```

### Debugging and Tracing

During those runs, you can enable the debug mode `--debug_run` to enable PyTorch profiler to collect traces. Later you can use the provided trace merger tool to generate a single chrome trace and load it with [Perfetto](https://perfetto.dev/).

![Trace](https://i.imgur.com/jA7Pqus.png)

```shell
uv run simsuite/trace_merger.py results/trace_voltage_improv_4.json --event_log_file=event_log_1761062693.474991.jsonl
```

### Collecting results

The run timing results will be collected under `./results/run_report.json`. You can take a look at some example notebooks to figure out how to interpret those results.

## Running non-Llama models

Simulator is also capable of running other models than Llama, such as CLIP.

```shell
uv run python -m scenarios.clip_perf_scenario  --device_count=1 --backend="simulation"
```