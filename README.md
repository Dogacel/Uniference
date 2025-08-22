# Edge Sim Suite

Simulation suite for running transformer models on edge devices in distributed settings.

## Quickstart

We recommend running _Llama 3_ models. To get started, download the LLama model weights into checkpoints directory.

```shell

mkdir checkpoints
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir checkpoints/Llama-3.2-1B-Instruct

```

A pre-determined set of scenarios and programs are listed under `scenarios` folder. You can use the `run.sh` helper script to run a scenario to get started.

```shell

uv sync --extra torch
./run.sh models.llama3.scripts.failover_simple_scenario

```

## Creating new scenarios

TODO: Document how to create new scenarios

## Writing new programs

TODO: Describe how to write new programs

## Modifying models

TODO: Describe how to modify models to 

## Running simulations

TODO: Describe how to run simulations