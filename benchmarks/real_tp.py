import itertools
import sys

from typing import Optional, Sequence

from commons import load_prompt, get_prompt_sequence_first_n, setup_world
from programs.tensor_parallel_program import TensorParallelProgram
from models.datatypes import RawMessage
import argparse


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argv[1:] if argv else sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_count", type=int, default=1, help="Number of devices")
    parser.add_argument("output_file", nargs="?", default="run_report.json", help="Output file name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parsed_args = parser.parse_args(args)

    device_count = parsed_args.device_count
    output_file = "results/" + parsed_args.output_file
    batch_size = parsed_args.batch_size
    prompt = load_prompt("checkpoints/prompt_5000.txt")

    text_lengths = [8, 16, 32, 64, 128, 256, 512]
    max_tokens = [1, 4, 8, 16, 32, 64]
    repeats = 3

    world = setup_world(
        device_count=device_count,
        seq_len=8192,
        output_file=output_file,
        program=lambda **kwargs: TensorParallelProgram(**kwargs),
        batch_size=batch_size,
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
            device.send("input", [RawMessage(role="user", content=sub_prompt)], "starting_input")

        world.run()

    world.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
