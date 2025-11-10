import json
import os
from time import time as _time
from typing import Any


class WorldEventLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event: dict):
        self.events.append({"timestamp": _time()} | event)

    def transmit_stats(self) -> dict:
        total_duration = 0.0
        total_size = 0
        transmit_count = 0

        for record in self.events:
            action = record.get("action")
            
            if action == "transmit_end":
                duration = record.get("duration")
                if duration is not None and duration > 0:
                    total_duration += duration
                    transmit_count += 1
            elif action == "transmit_start":
                size = record.get("size")
                if size is not None:
                    total_size += size

        return {
            "total_transmit_duration": total_duration,
            "total_transmit_size": total_size,
            "transmit_count": transmit_count,
        }

    def dump_events(self) -> str:
        now = _time()
        fname = f"profile_out/event_log_{now}.jsonl"
        with open(fname, "w") as f:
            for event in self.events:
                json.dump(event, f)
                f.write("\n")

        return fname

    def report_run(
        self,
        time: float,
        output_file: str,
        params: dict[str, Any],
    ):
        report_path = output_file
        runs = []
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                try:
                    runs = json.load(f)
                except json.JSONDecodeError:
                    runs = []
        run_entry = {"timestamp": _time(), "time": time}
        run_entry.update(params)
        runs.append(run_entry)
        with open(report_path, "w") as f:
            json.dump(runs, f, indent=2)
            f.write("\n")
