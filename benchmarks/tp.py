import itertools
import sys

from typing import Optional, Sequence

from commons import load_prompt, get_prompt_sequence_first_n, setup_world
from programs.tensor_parallel_program import TensorParallelProgram
from models.datatypes import RawMessage
from simsuite.units import Gbps, Mbps


def main(argv: Optional[Sequence[str]] = None) -> int:
    device_counts = [1, 2, 4]  # , 8, 16, 32]
    text_lengths = [256, 2048]
    speeds = [1 * Gbps, 100 * Mbps, 10 * Mbps, 1 * Mbps]
    latencies = [1, 10, 50, 100]  # in ms
    repeats = 10

    prompt = load_prompt("checkpoints/prompt_5000.txt")

    args = argv[1:] if argv else sys.argv[1:]
    output_file = "results/" + args[0] if args and len(args) > 0 else "results/run_report.json"

    for device_count in device_counts:
        world = setup_world(
            device_count=device_count,
            seq_len=8192,
            output_file=output_file,
            program=lambda **kwargs: TensorParallelProgram(**kwargs),
        )

        # Generate Cartesian product
        combinations = list(itertools.product(text_lengths, speeds, latencies, range(repeats)))

        print(f"Going to run {len(combinations)} combinations")

        for combo in combinations:
            sequence_length, speed, latency, repeat_idx = combo

            print(f"Sequence length: {sequence_length}, Speed: {speed}, Latency: {latency}, Repeat index: {repeat_idx}")

            sub_prompt = get_prompt_sequence_first_n(prompt, sequence_length)
            world.set_runtime_params(
                world.runtime_params
                | {
                    "prompt_length": len(sub_prompt),
                    "max_tokens": 1,
                    "device_count": device_count,
                    "network_bandwidth": speed,
                    "network_latency": latency,
                }
            )

            world.networks[0].bandwidth = speed
            world.networks[0].latency = latency

            for device in world.devices:
                device.send("input", [RawMessage(role="user", content=sub_prompt)], "starting_input")

            world.run()

        world.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
