import json
import os
from typing import Any


class WorldEventLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event: dict):
        self.events.append(event)

    def dump_events(self):
        with open("profile_out/event_log.jsonl", "w") as f:
            for event in self.events:
                json.dump(event, f)
                f.write("\n")

    def report_run(self, time: float, params: dict[str, Any]):
        report_path = "results/run_report.json"
        runs = []
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                try:
                    runs = json.load(f)
                except json.JSONDecodeError:
                    runs = []
        run_entry = {"time": time}
        run_entry.update(params)
        runs.append(run_entry)
        with open(report_path, "w") as f:
            json.dump(runs, f, indent=2)
            f.write("\n")
