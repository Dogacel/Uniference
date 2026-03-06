# Uniference 

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Last Updated](https://img.shields.io/github/last-commit/Dogacel/Uniference)

A discrete-event simulation (DES) framework designed for developing, benchmarking, and deploying distributed AI models within a unified environment without access to multiple GPUs and devices.

![Simulation Suite Overview](https://i.imgur.com/WfWIOy8.png)

## Why?

Developing distributed inference algorithms typically requires access to multiple GPUs or physical devices, making experimentation expensive and hard to reproduce. Most existing studies rely on ad-hoc testbeds or proprietary infrastructure, and there's no standardized way to compare results across different setups.
Uniference lets you prototype, benchmark, and validate distributed inference strategies on a single machine using discrete-event simulation. The same code can run in simulation and on real hardware with no changes, so you can iterate fast locally and deploy when ready.

All experiments in our paper, including 8-device tensor parallelism for LLama 3.1 8B, were developed and validated on a single M4 Pro MacBook Pro with 48 GB RAM with up to 95% to 98% accuracy compared to real deployments.

## At a Glance

Uniference provides drop-in replacements for common parallel layers. You can directly swap them into your model and simulate across multiple devices on a single machine while still allowing you to deploy on real-instances with no code changes.

```python
from simsuite.components import (
    ColumnParallelLinearSim,
    RowParallelLinearSim,
    VocabParallelEmbeddingSim,
)

# Replace nn.Linear and nn.Embedding with their parallel counterparts
self.tok_embeddings = VocabParallelEmbeddingSim(vocab_size, dim)
self.output = ColumnParallelLinearSim(dim, vocab_size, bias=False)
```

Then simulate distributed inference with no multiple devices or GPUs required, see [Quickstart](#quickstart) to get started.

## Quickstart

We recommend `uv` to run the models.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --all-extras
```

We recommend running _Llama 3_ models. To get started, download the LLama model weights into checkpoints directory from huggingface `meta-llama` or `unsloth` if you lack access.

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

To make sure all baseline algorithms work, you can run the `run_sanity_tests.py` program, which will ensure all standard algorithms work.

```shell
❯ uv run run_sanity_tests.py

Edge-LLM Benchmark Sanity Tests
Device: cpu
Output: /Users/dogac/code/edge-llm-benchmark/sanity_outputs
============================================================

▶ Running: multi_clip
  Command: DEVICE=cpu uv run benchmarks/multi_clip.py
  ✓ PASSED in 10.08s

▶ Running: real_pp
  Command: DEVICE=cpu uv run benchmarks/real_pp.py --text-lengths 128 --max-tokens 8 --repeats 1
  ✓ PASSED in 32.61s

▶ Running: real_tp_pp_poisson
  Command: DEVICE=cpu uv run benchmarks/real_tp_pp_poisson.py --text-lengths 128 --max-tokens 8 --repeats 1 --duration 1 --rate 1
  ✓ PASSED in 55.74s

▶ Running: real_tp_pp
  Command: DEVICE=cpu uv run benchmarks/real_tp_pp.py --text-lengths 128 --max-tokens 8 --repeats 1
  ✓ PASSED in 49.35s

▶ Running: real_tp
  Command: DEVICE=cpu uv run benchmarks/real_tp.py --text-lengths 128 --max-tokens 8 --repeats 1
  ✓ PASSED in 35.76s

▶ Running: real_voltage
  Command: DEVICE=cpu uv run benchmarks/real_voltage.py --device_count=2 --model_type="voltage" --text_lengths="[100]" --speed='[1000000000]' --latency='[0.005]' --repeats=1
  ✓ PASSED in 27.34s

▶ Running: real_voltage_improv
  Command: DEVICE=cpu uv run benchmarks/real_voltage.py --device_count=2 --model_type="voltage_improv" --text_lengths="[100]" --speed='[1000000000]' --latency='[0.005]' --repeats=1
  ✓ PASSED in 22.76s

Results saved to: sanity_outputs/sanity_run_20260305_234646

============================================================
SANITY TEST SUMMARY
============================================================

Results:
  Total:    7
  Passed:   7
  Failed:   0
  Duration: 233.64s

Passed Tests:
  ✓ multi_clip (10.08s)
  ✓ real_pp (32.61s)
  ✓ real_tp_pp_poisson (55.74s)
  ✓ real_tp_pp (49.35s)
  ✓ real_tp (35.76s)
  ✓ real_voltage (27.34s)
  ✓ real_voltage_improv (22.76s)

============================================================
ALL TESTS PASSED
============================================================
```

## Codebase

This project heavily uses models available in [llama-models](https://github.com/meta-llama/llama-models/). Therefore inherits some utilities and folders.

- **models**: Keeps transformer models that are run in simulations.
- **programs**: Stores programs, that can be loaded into devices to be run by them. Each program should use some `models`.
- **benchmarks**: Consists of programs to run benchmarks of certain algorithms such as `PP`, `TP` or `Voltage`.
- **notebooks**: Consists of Jupyter notebooks that are used to evaluate algorithms or the simulation software itself throughout its development.
- **simsuite**: Source of the simulation engine that is used to emulate multiple devices using conservative DES.

## Adding new models

You should add your models that implement new types of distributed inference algorithms under `models` directory. To get started,  inspect the `llama3_tp` model, which uses the provided row, column and vocabulary parallel helpers to implement tensor parallelism.

## Writing new programs

In order to run your model on a device, you should write a wrapper program around it. Programs control how your device should interact with the model. To get started, inspect the `tensor_parallel_program`.

## Creating new benchmarks

Benchmarks allow you to test your program under different network conditions and different number of devices. To get started, check out `real_tp.py` for benchmarking the tensor-parallel program.

## Deployment

You can directly deploy the models on devices by setting a few environment variables. First you need the ip address of the leader and it should be accessible from the followers.

```shell
# Note the local ip of the leader, usually in form 192.168.x.x or 10.0.0.x
ip -4 addr show scope global | grep inet

export MASTER_ADDR=192.168.1.14
export MASTER_PORT=25001
export WORLD_BACKEND=pytorch
export DEVICE=cuda
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

### Measuring Network Parameters 

To accurately model our world, we need to run some benchmarks to figure out our Network parameters. Run the following command in all devices to learn about the current network's parameters.

```shell
uv run benchmarks/network.py --mode=all_gather
```

For proper evaluation of your network conditions, you can use the `all_gather.ipynb` Notebook under `notebooks` folder.

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

## Additional Resources

For reproducibility of our experiments, we shared additional documents on how we setup Jetsons and modified our networks.

- [Jetson Setup](docs/jetson_setup.md)
- [Modifying Network Conditions](docs/modifying_network_conditions.md)

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=Dogacel/Uniference&type=date&legend=top-left)](https://www.star-history.com/?repos=Dogacel%2FUniference&type=date&legend=top-left)

## Citation

If you find Uniference useful in your research, please cite our paper:

```bibtex
@inproceedings{uniference2026,
  title={UNIFERENCE: A Discrete Event Simulation Framework for Developing Distributed AI Models},
  author={},
  booktitle={},
  year={2026}
}
```