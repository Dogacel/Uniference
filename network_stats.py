import json
import sys


def human_readable_bytes(num_bytes):
    for unit in ["bytes", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def sum_transmit_end_durations(jsonl_files):
    total_duration = 0.0
    transmit_end_count = 0
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("action") == "transmit_end" and "duration" in record:
                        total_duration += record["duration"]
                        transmit_end_count += 1
                except json.JSONDecodeError:
                    continue
    return total_duration


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <jsonl_file1> [<jsonl_file2> ...]")
        sys.exit(1)
    files = sys.argv[1:]
    total = sum_transmit_end_durations(files)
    print(f"Total transmit duration: {total:.6f} seconds")

    # Calculate total size for transmit_start and count
    total_size = 0
    transmit_start_count = 0
    for file_path in files:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("action") == "transmit_start" and "size" in record:
                        total_size += record["size"]
                        transmit_start_count += 1
                except json.JSONDecodeError:
                    continue
    print(f"Total transmit size: {human_readable_bytes(total_size)}")
    print(f"Number of transmits: {transmit_start_count}")
