import matplotlib.pyplot as plt
import os
import signal
import subprocess
import sys
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
    yield_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    text_lengths = [16, 64, 256]
    repeats = 25

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
            for r in range(repeats):
                for text_length in text_lengths:
                    for yield_probability in yield_probs:
                        tokens_to_generate = text_length
                        print(f"Tokens to generate: {tokens_to_generate}, Yield probability: {yield_probability}")

                        sub_prompt = get_prompt_sequence_first_n(prompt, 10)
                        seq_len = len(sub_prompt)
                        rc = run_once(
                            scenario="scenarios.yield_perf_scenario",
                            device_count=device_count,
                            prompt=sub_prompt,
                            seq_len=8192,
                            max_tokens=tokens_to_generate,
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
