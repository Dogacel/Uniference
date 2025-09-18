import json


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
