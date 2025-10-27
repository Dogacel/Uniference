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


def main(
    device_counts=[1, 2, 4],
    sample_count: int = 1000,
    max_seq_len: int = 2000,
    max_yield_prob: float = 1.0,
    output_file: str = "results/run_report.json",
    mode: str = "server",
) -> int:
    # 2D Latin hypercube sampling
    bounds_2d = np.array(
        [
            [99, max_seq_len],  # Sequence length
            [0.999, max_yield_prob],  # Yield probability
        ]
    )
    d = bounds_2d.shape[0]

    sampler2d = qmc.LatinHypercube(d, rng=np.random.default_rng(42))
    x_2d = sampler2d.random(sample_count)
    x_2d = qmc.scale(x_2d, bounds_2d[:, 0], bounds_2d[:, 1])

    # Convert the first dimension to integers by flooring them for X_2d
    x_2d[:, 0] = np.floor(x_2d[:, 0])

    prompt = load_prompt("checkpoints/prompt_5000.txt")
    world = World(output_file=output_file)
    world.set_runtime_params({"max_seq_len": max_seq_len})

    def server():
        world.background_server = BackgroundServer()
        world.background_server.start()

        for device_count in device_counts:
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
                )
                devices.append(device)

            world.network(
                NetworkArgs(
                    devices=devices,
                    network_params=[0.0005, 1.02e-08],
                )
            )

            world.run(warmup=True)

            for x in x_2d:
                sequence_length = int(x[0])
                yield_probability = float(x[1])
                print(f"Sequence length: {sequence_length}, Yield probability: {yield_probability}")

                sub_prompt = get_prompt_sequence_first_n(prompt, sequence_length)

                run_once(
                    prompt=sub_prompt,
                    max_tokens=1,
                    yield_probability=yield_probability,
                    world=world,
                )
            world.destroy()

    def client():
        import os 
        name = "device_client_" + os.environ.get("MASTER_PORT")
        me = world.device(
            DeviceArgs(spec=DeviceSpec(), client=True, name=name),
            program=YieldPerfProgram(
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
