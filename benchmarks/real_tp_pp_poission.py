import itertools
import os
import sys

from typing import Optional, Sequence

from commons import load_prompt, get_prompt_sequence_first_n, setup_world
from programs.pipeline_tensor_parallel_poisson_program import PipelineTensorParallelPoissonProgram
from models.datatypes import RawMessage
import argparse
import numpy as np
import time

from simsuite.world import World

def generate_poisson_messages(
    world: World,
    duration: float = 30.0,  # seconds
    rate: float = 10.0,      # messages per second (lambda)
    seed: int = 42,
    prompt: str = "",
    min_prompt_length: int = 8,
    max_prompt_length: int = 128,
    t: float = 0.0,
):
    random = np.random.RandomState(42)

    # Generate inter-arrival times (exponential distribution)
    messages = []
    
    end_time = t + duration

    while t < end_time:
        # Time until next message
        inter_arrival = random.exponential(1.0 / rate)
        t += inter_arrival

        if t < end_time:
            messages.append({
                "timestamp": t,
                "id": len(messages),
                "content": get_prompt_sequence_first_n(prompt, random.randint(min_prompt_length, max_prompt_length)),
            })

    return messages

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argv[1:] if argv else sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_count", type=int, default=4, help="Number of devices")
    parser.add_argument("output_file", nargs="?", default="run_report.json", help="Output file name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--pp_size", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--debug_run", action="store_true", help="Enable debug run")
    parsed_args = parser.parse_args(args)

    device_count = parsed_args.device_count
    pp_size = parsed_args.pp_size
    output_file = "results/" + parsed_args.output_file
    batch_size = parsed_args.batch_size
    prompt = load_prompt("checkpoints/prompt_5000.txt")

    text_lengths = [256] # [8, 16, 32, 64, 128, 256, 512]
    max_tokens = [1] # [1, 4, 8, 16, 32, 64]
    repeats = 1

    world = setup_world(
        device_count=device_count,
        pp_size=pp_size,
        seq_len=8192,
        output_file=output_file,
        program=lambda **kwargs: PipelineTensorParallelPoissonProgram(**kwargs),
        batch_size=batch_size,
        world_kwargs={"debug_run": parsed_args.debug_run}
    )

    # Generate Cartesian product
    combinations = list(itertools.product(text_lengths, max_tokens, range(repeats)))

    print(f"Going to run {len(combinations)} combinations")

    for combo in combinations:
        sequence_length, max_tokens, repeat_idx = combo

        print(f"Sequence length: {sequence_length}, tokens, max_tokens: {max_tokens} Repeat index: {repeat_idx}")

        sub_prompt = get_prompt_sequence_first_n(prompt, sequence_length)
        world.set_runtime_params(
            world.runtime_params
            | {
                "prompt_length": len(sub_prompt),
                "device_count": device_count,
                "max_seq_len": sequence_length,
                "max_tokens": max_tokens,
            }
        )

        for device in world.devices:
            device.program.max_tokens = max_tokens

            world.inputs = generate_poisson_messages(
                world,
                duration=3.0,
                rate=5.0,
                prompt=prompt,
                min_prompt_length=8,
                max_prompt_length=sequence_length,
            )

        world.run()

    world.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
