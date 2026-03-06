import numpy as np
import sys

from scipy.stats import qmc
from typing import Optional, Sequence

from commons import load_prompt, get_prompt_sequence_first_n, setup_world, run_once
from programs.yield_perf_program import YieldPerfProgram


def main(argv: Optional[Sequence[str]] = None) -> int:
    device_counts = [1, 2, 4]  # , 8, 16, 32]

    # 2D Latin hypercube sampling
    bounds_2d = np.array(
        [
            [0.0, 1000.0],  # Sequence length
            [0.0, 1.0],  # Yield probability
        ]
    )
    d = bounds_2d.shape[0]
    sample_count = 1000

    sampler2d = qmc.LatinHypercube(d)
    x_2d = sampler2d.random(sample_count)
    x_2d = qmc.scale(x_2d, bounds_2d[:, 0], bounds_2d[:, 1])

    # Convert the first dimension to integers by flooring them for X_2d
    x_2d[:, 0] = np.floor(x_2d[:, 0])

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

        for x in x_2d:
            tokens_to_generate = int(x[0])
            yield_probability = float(x[1])
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
