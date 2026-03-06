#!/usr/bin/env python3
"""
Executes all sanity tests, captures outputs, and reports results.
Usage: python run_sanity_tests.py [--output-dir OUTPUT_DIR] [--device DEVICE]
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    name: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration: float
    
    @property
    def passed(self) -> bool:
        return self.returncode == 0


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @classmethod
    def disable(cls):
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


def get_sanity_tests(device: str = "cpu") -> list[tuple[str, str]]:
    """
    Returns list of (test_name, command) tuples for sanity tests.
    These mirror the sanity_all target in the Makefile.
    """
    return [
        ("multi_clip", f"DEVICE={device} uv run benchmarks/multi_clip.py"),
        ("real_pp", f"DEVICE={device} uv run benchmarks/real_pp.py --text-lengths 128 --max-tokens 8 --repeats 1"),
        ("real_tp_pp_poisson", f"DEVICE={device} uv run benchmarks/real_tp_pp_poisson.py --text-lengths 128 --max-tokens 8 --repeats 1 --duration 1 --rate 1"),
        ("real_tp_pp", f"DEVICE={device} uv run benchmarks/real_tp_pp.py --text-lengths 128 --max-tokens 8 --repeats 1"),
        ("real_tp", f"DEVICE={device} uv run benchmarks/real_tp.py --text-lengths 128 --max-tokens 8 --repeats 1"),
        ("real_voltage", f"DEVICE={device} uv run benchmarks/real_voltage.py --device_count=2 --model_type=\"voltage\" --text_lengths=\"[100]\" --speed='[1000000000]' --latency='[0.005]' --repeats=1"),
        ("real_voltage_improv", f"DEVICE={device} uv run benchmarks/real_voltage.py --device_count=2 --model_type=\"voltage_improv\" --text_lengths=\"[100]\" --speed='[1000000000]' --latency='[0.005]' --repeats=1"),
    ]


def run_test(name: str, command: str, env: Optional[dict] = None) -> TestResult:
    """Run a single test and capture its output."""
    print(f"{Colors.CYAN}▶ Running:{Colors.ENDC} {name}")
    print(f"  {Colors.BLUE}Command:{Colors.ENDC} {command}")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    # Add common environment variables from Makefile
    merged_env.setdefault("DEBUG", "0")
    merged_env.setdefault("RANK", "0")
    merged_env.setdefault("WORLD_SIZE", "1")
    merged_env.setdefault("MASTER_PORT", "29500")
    merged_env.setdefault("MASTER_ADDR", "localhost")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=merged_env,
            cwd=Path(__file__).parent,
        )
        duration = time.time() - start_time
        
        return TestResult(
            name=name,
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            name=name,
            command=command,
            returncode=-1,
            stdout="",
            stderr=str(e),
            duration=duration,
        )


def print_result_summary(result: TestResult) -> None:
    """Print a summary of a single test result."""
    if result.passed:
        status = f"{Colors.GREEN}✓ PASSED{Colors.ENDC}"
    else:
        status = f"{Colors.RED}✗ FAILED (exit code: {result.returncode}){Colors.ENDC}"
    
    print(f"  {status} in {result.duration:.2f}s")
    
    if not result.passed:
        # Show last few lines of stderr/stdout on failure
        print(f"\n  {Colors.RED}--- Error Output ---{Colors.ENDC}")
        error_output = result.stderr.strip() or result.stdout.strip()
        if error_output:
            lines = error_output.split('\n')
            # Show last 15 lines
            for line in lines[-15:]:
                print(f"  {Colors.YELLOW}{line}{Colors.ENDC}")
        print(f"  {Colors.RED}--- End Error Output ---{Colors.ENDC}\n")


def save_results(results: list[TestResult], output_dir: Path) -> Path:
    """Save all test results to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"sanity_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual test outputs
    for result in results:
        test_dir = run_dir / result.name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        (test_dir / "stdout.txt").write_text(result.stdout)
        (test_dir / "stderr.txt").write_text(result.stderr)
        (test_dir / "info.txt").write_text(
            f"Command: {result.command}\n"
            f"Return Code: {result.returncode}\n"
            f"Duration: {result.duration:.2f}s\n"
            f"Status: {'PASSED' if result.passed else 'FAILED'}\n"
        )
    
    # Save summary report
    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Sanity Test Run - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        total_duration = sum(r.duration for r in results)
        
        f.write(f"Total Tests: {len(results)}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total Duration: {total_duration:.2f}s\n\n")
        
        f.write("Individual Results:\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            status = "PASSED" if result.passed else f"FAILED (exit: {result.returncode})"
            f.write(f"\n{result.name}:\n")
            f.write(f"  Status: {status}\n")
            f.write(f"  Duration: {result.duration:.2f}s\n")
            f.write(f"  Command: {result.command}\n")
    
    return run_dir


def print_final_summary(results: list[TestResult]) -> None:
    """Print final summary of all test results."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    total_duration = sum(r.duration for r in results)
    
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}SANITY TEST SUMMARY{Colors.ENDC}")
    print("=" * 60)
    
    print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
    print(f"  Total:    {len(results)}")
    print(f"  {Colors.GREEN}Passed:   {len(passed)}{Colors.ENDC}")
    print(f"  {Colors.RED}Failed:   {len(failed)}{Colors.ENDC}")
    print(f"  Duration: {total_duration:.2f}s")
    
    if passed:
        print(f"\n{Colors.GREEN}Passed Tests:{Colors.ENDC}")
        for r in passed:
            print(f"  ✓ {r.name} ({r.duration:.2f}s)")
    
    if failed:
        print(f"\n{Colors.RED}Failed Tests:{Colors.ENDC}")
        for r in failed:
            print(f"  ✗ {r.name} (exit: {r.returncode}, {r.duration:.2f}s)")
    
    print("\n" + "=" * 60)
    
    if failed:
        print(f"{Colors.RED}{Colors.BOLD}SOME TESTS FAILED{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED{Colors.ENDC}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run sanity tests for edge-llm-benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sanity_tests.py                    # Run with CPU device
  python run_sanity_tests.py --device cuda      # Run with CUDA
  python run_sanity_tests.py --device mps       # Run with MPS (Apple Silicon)
  python run_sanity_tests.py -o ./test_outputs  # Custom output directory
  python run_sanity_tests.py --no-color         # Disable colored output
        """,
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("sanity_outputs"),
        help="Directory to save test outputs (default: sanity_outputs)",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run tests on (default: cpu)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--tests",
        type=str,
        nargs="*",
        help="Run only specific tests (by name)",
    )
    
    args = parser.parse_args()
    
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}Edge-LLM Benchmark Sanity Tests{Colors.ENDC}")
    print(f"{Colors.CYAN}Device: {args.device}{Colors.ENDC}")
    print(f"{Colors.CYAN}Output: {args.output_dir.absolute()}{Colors.ENDC}")
    print("=" * 60 + "\n")
    
    tests = get_sanity_tests(args.device)
    
    # Filter tests if specific ones requested
    if args.tests:
        tests = [(name, cmd) for name, cmd in tests if name in args.tests]
        if not tests:
            print(f"{Colors.RED}No matching tests found.{Colors.ENDC}")
            print(f"Available tests: {', '.join(t[0] for t in get_sanity_tests())}")
            sys.exit(1)
    
    results: list[TestResult] = []
    
    for name, command in tests:
        result = run_test(name, command)
        results.append(result)
        print_result_summary(result)
        print()
    
    # Save results
    run_dir = save_results(results, args.output_dir)
    print(f"{Colors.BLUE}Results saved to: {run_dir}{Colors.ENDC}")
    
    # Print final summary
    print_final_summary(results)
    
    # Exit with non-zero if any tests failed
    failed_count = sum(1 for r in results if not r.passed)
    sys.exit(failed_count)


if __name__ == "__main__":
    main()
