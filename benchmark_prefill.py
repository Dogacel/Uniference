import io
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import subprocess
import sys
from imgcat import imgcat
from scipy.stats import qmc
from typing import List, Optional, Sequence


def load_prompt(prompt_file: str) -> str:
    if not prompt_file:
        return ""
    with open(prompt_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_prompt_sequence_first_n(prompt: str, n: int) -> str:
    if n <= 0 or not prompt:
        return ""
    pos = -1
    for _ in range(n):
        pos = prompt.find(",", pos + 1)
        if pos == -1:
            return prompt
    return prompt[: pos + 1]


def run_once(
    scenario: str,
    device_count: int,
    prompt: str,
    seq_len: int,
    max_tokens: int,
    yield_probability: float,
    output_file: str,
) -> int:
    args: List[str] = [
        "./run.sh",
        scenario,
        "--debug_run=False",
        f"--prompt={prompt}",
        f"--max_seq_len={seq_len}",
        "--temperature=0.0",
        "--top_p=1.0",
        f"--max_tokens={max_tokens}",
        f"--output_file={output_file}",
        f"--yield_probability={yield_probability}",
    ]

    if device_count == 0:
        args.append("--performance_mode")
        device_count = 1

    args.append(f"--device_count={device_count}")

    preexec = os.setsid if os.name != "nt" else None

    p = subprocess.Popen(
        args,
        preexec_fn=preexec,
        env=os.environ,
    )

    try:
        return p.wait()
    except KeyboardInterrupt:
        print("Stopping child...")
        # Kill the whole process group
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        sys.exit(0)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main(argv: Optional[Sequence[str]] = None) -> int:
    device_counts = [1, 2, 4]  # , 8, 16, 32]

    # 2D Latin hypercube sampling
    bounds_2d = np.array(
        [
            [0.0, 2000.0],  # Sequence length
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

    # Visualize the 2D samples
    plt.figure(figsize=(7, 5))
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=x_2d[:, 1], cmap="viridis", s=18, alpha=0.8, edgecolor="white", linewidth=0.3)
    plt.xlabel("Sequence length")
    plt.ylabel("Probability of yielding on layer")
    plt.title("Latin hypercube samples in 2D parameter space")
    plt.colorbar(label="Yield Ratio")


    prompt = load_prompt("checkpoints/prompt_5000.txt")

    args = argv[1:] if argv else sys.argv[1:]
    output_file = "results/" + args[0] if args and len(args) > 0 else "results/run_report.json"

    plt.savefig(output_file.replace(".json", "_lhs.png"), format="png", dpi=200, bbox_inches="tight")

    try:
        # Warmup
        for _ in range(3):
            rc = run_once(
                scenario="scenarios.yield_perf_scenario",
                device_count=1,
                prompt="Hello, world!",
                seq_len=256,
                max_tokens=5,
                yield_probability=0.5,
                output_file="/tmp/warmup.json",
            )

        for device_count in device_counts:
            for x in x_2d:
                sequence_length = int(x[0])
                yield_probability = float(x[1])
                print(f"Sequence length: {sequence_length}, Yield probability: {yield_probability}")

                sub_prompt = get_prompt_sequence_first_n(prompt, sequence_length)
                seq_len = len(sub_prompt)
                rc = run_once(
                    scenario="scenarios.yield_perf_scenario",
                    device_count=device_count,
                    prompt=sub_prompt,
                    seq_len=8192,
                    max_tokens=1,
                    yield_probability=yield_probability,
                    output_file=output_file,
                )

                if rc != 0:
                    print(
                        f"Command failed with exit code {rc}",
                        file=sys.stderr,
                    )
                    return rc
        return 0
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
