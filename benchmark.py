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
    output_file: str,
) -> int:
    args: List[str] = [
        "./run.sh",
        scenario,
        "--debug_run=False",
        f"--prompt={prompt}",
        f"--max_seq_len={seq_len}",
        f"--max_tokens={max_tokens}",
        f"--output_file={output_file}",
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
    device_counts = [0, 1, 2, 4, 8] #, 16, 32]
    prompt = load_prompt("checkpoints/prompt_5000.txt")
    prompts = [get_prompt_sequence_first_n(prompt, n) for n in [10, 200, 1_000]]
    repeats = 100

    args = argv[1:] if argv else sys.argv[1:]
    output_file = "results/" + args[0] if args and len(args) > 0 else "results/run_report.json"

    try:
        # Warmup
        for _ in range(3):
            rc = run_once(
                scenario="scenarios.concurrent_scenario",
                device_count=1,
                prompt=prompts[0],
                seq_len=256,
                max_tokens=5,
                output_file="/tmp/warmup.json",
            )

        for device_count in device_counts:
            for _ in range(repeats):
                for prompt in prompts:
                    seq_len = len(prompt)
                    rc = run_once(
                        scenario="scenarios.concurrent_scenario",
                        device_count=device_count,
                        prompt=prompt,
                        seq_len=seq_len,
                        max_tokens=seq_len // 10,
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
