import itertools
import sys

from typing import Optional, Sequence

from commons import load_prompt, get_prompt_sequence_first_n, setup_world, run_once
from programs.yield_perf_program import YieldPerfProgram


def main(argv: Optional[Sequence[str]] = None) -> int:
    device_counts = [1, 2, 4]  # , 8, 16, 32]
    yield_probs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    text_lengths = [16, 64, 256]
    repeats = 20

    prompt = load_prompt("checkpoints/prompt_5000.txt")

    args = argv[1:] if argv else sys.argv[1:]
    output_file = "results/" + args[0] if args and len(args) > 0 else "results/run_report.json"

    for device_count in device_counts:
        world = setup_world(
            device_count=device_count,
            seq_len=8192,
            output_file=output_file,
            program=lambda **kwargs: YieldPerfProgram(input=[prompt], **kwargs),
            pp_size=1,
        )

        combinations = list(itertools.product(yield_probs, text_lengths, range(repeats)))

        for combo in combinations:
            yield_probability, text_length, _ = combo
            for yield_probability in yield_probs:
                tokens_to_generate = text_length
                print(f"Tokens to generate: {tokens_to_generate}, Yield probability: {yield_probability}")

                sub_prompt = get_prompt_sequence_first_n(prompt, 10)
                run_once(
                    prompt=sub_prompt,
                    max_tokens=tokens_to_generate,
                    yield_probability=yield_probability,
                    world=world,
                )
        world.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
