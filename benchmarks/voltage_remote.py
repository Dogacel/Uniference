import fire
import itertools
import torch

from commons import load_prompt, get_prompt_sequence_first_n, setup_world
from programs.voltage_program import VoltageProgram

from simsuite.units import Gbps, Mbps, ms
import asyncio
import fire
import numpy as np

from scipy.stats import qmc

from commons import load_prompt, get_prompt_sequence_first_n, run_once
from programs.yield_perf_program import YieldPerfProgram
from simsuite.client import Client
from simsuite.server import BackgroundServer
from simsuite.world import World
from simsuite.device import DeviceArgs, DeviceSpec
from simsuite.network import NetworkArgs
from models.datatypes import RawMessage




def main(
    output_file: str = "results/run_report.json",
    prompt_file: str = "checkpoints/prompt_5000.txt",
    text_lengths: list[int] = [128],
    speed: list[float] = [1 * Gbps],
    latency: list[float] = [1 * ms],
    repeats: int = 3,
    max_seq_len: int = 8192,
    device_count: int = 1,
    model_type: str = "voltage",
    mode: str = "server",
    debug_run: bool = False,
) -> int:
    prompt = load_prompt(prompt_file)
    world = World(output_file=output_file, mode=mode)
    world.set_runtime_params({"max_seq_len": max_seq_len})

    combinations = list(itertools.product(text_lengths, speed, latency, range(repeats)))

    print(f"Going to run {len(combinations)} combinations")

    def server():
        world.background_server = BackgroundServer()
        world.background_server.start()

        # Wait until device is connected
        world.background_server.wait_for_clients(device_count)

        print("Clients connected. Setting up remote devices.")
        devices = []
        for client_id in world.background_server.CLIENTS:
            reader, writer = world.background_server.CLIENTS[client_id]

            device = world.remote_device(
                world.background_server.loop,
                reader,
                writer,
                client_id,
            )
            devices.append(device)

        world.network(
            NetworkArgs(
                devices=devices,
                network_params=[0.0005, 1.02e-08],
            )
        )

        # world.run(warmup=True)

        for combo in combinations:
            sequence_length, speed, latency, repeat_idx = combo

            print(
                f"Sequence length: {sequence_length}, tokens, speed: {speed}, latency: {latency}, Repeat index: {repeat_idx}"
            )

            sub_prompt = prompt[:sequence_length]  # get_prompt_sequence_first_n(prompt, sequence_length)
            world.set_runtime_params(
                world.runtime_params
                | {
                    "prompt_length": len(sub_prompt),
                    "device_count": device_count,
                    "max_seq_len": sequence_length,
                    "max_tokens": 1,
                    "network_bandwidth": speed,
                    "network_latency": latency,
                }
            )

            world.networks[0].network_params = [latency, 1 / speed]

            world.run()

            world.destroy()

    def client():
        import os

        name = "device_client_" + os.environ.get("MASTER_PORT")
        me = world.device(
            DeviceArgs(spec=DeviceSpec(), client=True, name=name),
            program=VoltageProgram(
                ckpt_dir="./checkpoints/Llama-3.2-1B-Instruct/original",
                temperature=0.0,
                top_p=1.0,
                max_seq_len=8192,
                max_tokens=1,
                max_batch_size=1,
                input=[],
            ),
        )

        # Connect to server
        client = Client(name, me)

        idx = 0

        def client_pre_run(device):
            nonlocal combinations, idx
            sequence_length, speed, latency, repeat_idx = combinations[idx]
            idx += 1

            device.name = name
            device.client = False
            if name == "device_client_25002":
                print("Device", device.name, "is client, sending input")
                device.client = True
                sub_prompt = prompt[:sequence_length]
                device.program.input = [sub_prompt]
            device.program.max_tokens = 1

        client.pre_run = client_pre_run

        asyncio.run(client.connect_with_retries())

    if mode == "server":
        server()
    elif mode == "client":
        client()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return 0


if __name__ == "__main__":
    fire.Fire(main)
