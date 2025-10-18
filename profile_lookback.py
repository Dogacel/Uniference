import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

#!/usr/bin/env python3
"""
profile_lookback.py

Read a JSON file that contains an array of objects. Each object must have a "log_file" field
(pointing to a file that contains rows with "action" and "duration"). For every row in each log
file where action == "transmit_end" and duration > 1, sum the durations, add that sum to the
object's "total_transmit_duration", and decrement its "transmit_count" by the number of matched rows.

Usage:
    python profile_lookback.py input.json [output.json]
If output.json is omitted, a file named input.json.updated will be created next to input.json.
"""

def parse_log_file(path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        text_stripped = text.strip()
        # Try to parse as a JSON array/object first
        if not text_stripped:
                return []
        try:
                loaded = json.loads(text_stripped)
                if isinstance(loaded, list):
                        return loaded
                if isinstance(loaded, dict):
                        # single object -> treat as one-row list
                        return [loaded]
        except Exception:
                pass
        # Fallback to JSON lines
        rows = []
        for line in text.splitlines():
                line = line.strip()
                if not line:
                        continue
                try:
                        rows.append(json.loads(line))
                except Exception:
                        # ignore unparsable lines
                        continue
        return rows


def sum_transmit_durations(rows: List[Dict[str, Any]]) -> Tuple[float, int]:
        total = 0.0
        count = 0
        for r in rows:
                try:
                        if r.get("action") != "transmit_end":
                                continue
                        d = r.get("duration")
                        if d is None:
                                continue
                        # accept numeric or numeric strings
                        if isinstance(d, str):
                                d = float(d) if d.strip() != "" else None
                        if not isinstance(d, (int, float)):
                                continue
                        if d > 0:
                                total += float(d)
                                count += 1
                except Exception:
                        continue
        return total, count


def process_profile_array(profiles: List[Dict[str, Any]], base_dir: Path) -> None:
        for entry in profiles:
                log_file_field = entry.get("log_file")
                if not log_file_field:
                        continue
                log_path = Path(log_file_field)
                if not log_path.is_absolute():
                        log_path = (base_dir / log_path).resolve()
                if not log_path.exists():
                        # skip missing logs
                        print(f"Warning: log file not found: {log_path}")
                        continue
                rows = parse_log_file(log_path)
                s, c = sum_transmit_durations(rows)
                # print(s, c)
                if s == 0 and c == 0:
                        continue
                # update numeric totals safely
                entry["total_transmit_duration"] = s
                entry["transmit_count"] = c
                # avoid negative counts
                if entry["transmit_count"] < 0:
                        entry["transmit_count"] = 0


def main():
        p = argparse.ArgumentParser(description="Update profiles with transmit durations from log files.")
        p.add_argument("input", help="Input JSON file containing an array of profile objects.")
        p.add_argument("output", nargs="?", help="Output file to write the updated array. Defaults to input + .updated")
        args = p.parse_args()

        input_path = Path(args.input)
        if not input_path.exists():
                raise SystemExit(f"Input file not found: {input_path}")

        out_path = Path(args.output) if args.output else input_path.with_suffix(input_path.suffix + ".updated")
        base_dir = Path()

        with input_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, list):
                raise SystemExit("Input JSON must be an array of objects.")

        process_profile_array(data, base_dir)

        with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Wrote updated profiles to: {out_path}")


if __name__ == "__main__":
        main()