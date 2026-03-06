import fire
import numpy as np

from scipy.stats import qmc

from commons import load_prompt, get_prompt_sequence_first_n, setup_world, run_once
from programs.yield_perf_program import YieldPerfProgram


def main(
    device_counts=[1, 2, 4],
    sample_count: int = 1000,
    max_seq_len: int = 2000,
    max_yield_prob: float = 1.0,
    output_file: str = "results/run_report.json",
) -> int:
    # 2D Latin hypercube sampling
    bounds_2d = np.array(
        [
            [0.0, max_seq_len],  # Sequence length
            [0.0, max_yield_prob],  # Yield probability
        ]
    )
    d = bounds_2d.shape[0]

    sampler2d = qmc.LatinHypercube(d)
    x_2d = sampler2d.random(sample_count)
    x_2d = qmc.scale(x_2d, bounds_2d[:, 0], bounds_2d[:, 1])

    # Convert the first dimension to integers by flooring them for X_2d
    x_2d[:, 0] = np.floor(x_2d[:, 0])

    prompt = load_prompt("checkpoints/prompt_5000.txt")

    for device_count in device_counts:
        world = setup_world(
            device_count=device_count,
            seq_len=8192,
            output_file=output_file,
            program=lambda **kwargs: YieldPerfProgram(input=[prompt], **kwargs),
            pp_size=1,
        )

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

    return 0


if __name__ == "__main__":
    fire.Fire(main)
