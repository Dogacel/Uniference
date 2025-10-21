import itertools
import sys
import torch

from typing import Optional, Sequence

from commons import load_prompt, get_prompt_sequence_first_n, setup_world
from programs.voltage_program import VoltageProgram
from models.datatypes import RawMessage
import argparse

from simsuite.units import Gbps, Mbps, ms


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argv[1:] if argv else sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", nargs="?", default="run_report.json", help="Output file name")
    parser.add_argument("--device_count", type=int, default=1, help="Number of devices")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--model_type", type=str, default="voltage", help="Type of voltage model to use")
    parser.add_argument("--debug_run", action="store_true", help="Enable debug run mode")
    parsed_args = parser.parse_args(args)

    device_count = parsed_args.device_count
    output_file = "results/" + parsed_args.output_file
    batch_size = parsed_args.batch_size
    prompt = load_prompt("checkpoints/prompt_5000.txt")

    text_lengths = [1024, 2048] # [64, 128, 256]
    speed = [10 * Mbps, 100 * Mbps, 1 * Gbps]
    latency = [1 * ms, 5 * ms, 20 * ms]
    repeats = 10

    world = setup_world(
        device_count=device_count,
        seq_len=8192,
        output_file=output_file,
        program=lambda **kwargs: VoltageProgram(**kwargs),
        batch_size=batch_size,
        program_kwargs={"model_type": parsed_args.model_type},
        world_kwargs={"debug_run": parsed_args.debug_run},
    )

    # Generate Cartesian product
    combinations = list(itertools.product(text_lengths, speed, latency, range(repeats)))

    print(f"Going to run {len(combinations)} combinations")

    for combo in combinations:
        sequence_length, speed, latency, repeat_idx = combo

        print(f"Sequence length: {sequence_length}, tokens, speed: {speed}, latency: {latency}, Repeat index: {repeat_idx}")

        sub_prompt = get_prompt_sequence_first_n(prompt, sequence_length)
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


        world.networks[0].network_params = [latency, 1/speed]

        client = True
        for device in world.devices:
            device.client = client
            if world.backend == "pytorch":
                device.client = torch.distributed.get_rank() == 0
                client = device.client
            if client:
                print("Device", device.name, "is client, sending input")
                device.program.input = [sub_prompt]
            client = False  # Only first device is client
            device.program.max_tokens = 1

        print("Start run...")
        world.run()

    world.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
