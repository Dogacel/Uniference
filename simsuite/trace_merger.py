import json
import sys
import pathlib


def load_trace(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Support both {"traceEvents":[...]} and plain list formats
    if isinstance(data, dict):
        events = data.get("traceEvents", data)
    else:
        events = data
    return list(events), data


def read_custom_events(path):
    """
    Reads the events from the jsonl file and parses them to the correct format.
    """
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Group data by "device" key
    from collections import defaultdict

    grouped = defaultdict(list)
    for e in data:
        if "device" in e and "action" in e and e["action"] in {"send"}:
            grouped[e.get("device")].append(e)
    return grouped


def merge_traces(paths, out_path, mode="concat", group_prefix=None, normalize_logical_clock=False, custom_events=None):
    """
    mode = 'concat' -> append traces one after another in time
           'align'  -> align all traces so each starts at the same ts as the first trace
    group_prefix: if set and mode=='concat', all events get pid=0 and process_name is set.
    normalize_logical_clock: if True, shift all event times in each trace so min(ts) aligns with logical_clock in the root json.
    custom_events: list of custom events to insert into the merged trace (e.g. from event_logger)
    """
    merged = []
    next_pid_base = 0

    # Track global min ts for 'align'
    first_ts = None

    # If group_prefix is set and mode is concat, force all pids to 0 and add process_name metadata
    force_pid = mode == "concat" and group_prefix is not None
    process_pid = 0 if force_pid else None
    if force_pid:
        # Insert process_name metadata event at the start
        merged.append({"ph": "M", "name": "process_name", "pid": process_pid, "args": {"name": group_prefix}})

    for _, p in enumerate(paths):
        events, raw = load_trace(p)
        events = [
            e
            for e in events
            if not (
                isinstance(e, dict)
                and e.get("name") in {"Iteration Start: PyTorch Profiler", "Record Window End", "PyTorch Profiler (0)"}
            )
        ]

        # Find min ts in this trace
        ts_min = min((e.get("ts", 0) for e in events if isinstance(e, dict)), default=0)
        # If normalization is enabled and logical_clock is present, shift all event times
        logical_clock = None
        if normalize_logical_clock and isinstance(raw, dict) and "logical_clock" in raw:
            logical_clock = raw["logical_clock"] * 1_000_000  # assuming logical_clock is in seconds, convert to us
        if logical_clock is not None:
            ts_shift = logical_clock - ts_min
        else:
            ts_shift = 0

        if first_ts is None:
            first_ts = ts_min + ts_shift

        # Remap pid/tid to avoid collisions
        # Collect seen pids/tids in this file
        pids = {e.get("pid") for e in events if isinstance(e, dict) and "pid" in e}
        pid_map = {pid: pid + next_pid_base for pid in pids if isinstance(pid, int)}

        # Apply remaps + time offset
        for e in events:
            if not isinstance(e, dict):  # skip non-dict entries just in case
                continue
            e = e.copy()
            if "ts" in e:
                e["ts"] = e["ts"] + ts_shift
            if force_pid:
                e["pid"] = process_pid
            elif "pid" in e and isinstance(e["pid"], int):
                e["pid"] = pid_map.get(e["pid"], e["pid"])
            # tid is left as-is
            merged.append(e)

        # Bump pid base so the next trace’s pids won’t collide
        if pids and not force_pid:
            next_pid_base += (max(pid_map.values()) - min(pid_map.values()) + 1) if pid_map else 10000

    # Map custom events to this file's pid space and insert
    """
    Format:
    {
        "ph": "X",
        "cat": "user_annotation",
        "name": "Attention.calculate_xq",
        "pid": 0,
        "tid": 7358461,
        "ts": 169775.849609375,
        "dur": 4104.982,
        "args": {
            "External id": 2199,
            "Record function id": 0,
            "Ev Idx": 150
        }
    },
    """
    if custom_events:
        for e in custom_events:
            event = {
                "ph": "X",
                "cat": "user_annotation",
                "name": e.get("action", "custom_event"),
                "pid": process_pid,
                "tid": "network",
                "ts": e.get("time", 0) * 1_000_000,  # assuming time is in seconds, convert to us
                "dur": e.get("arrive_at", 0) * 1_000_000 - e.get("time", 0) * 1_000_000,  # in us
                "args": {
                    "External id": e.get("id", 0),
                },
            }
            merged.append(event)

    # Filter out unwanted profiler events
    out = {"traceEvents": merged}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)


def group_by_prefix(file_list, delimiter="_run."):
    """Group files by prefix before the first delimiter."""
    from collections import defaultdict

    groups = defaultdict(list)
    for f in file_list:
        name = pathlib.Path(f).stem
        prefix = name.split(delimiter)[0]
        groups[prefix].append(f)

    return groups


if __name__ == "__main__":
    # Usage:
    #   python trace_merger.py out.json folder_path [--normalize-logical-clock]
    #   python trace_merger.py out.json mode file1.json file2.json ... [--normalize-logical-clock]
    # If folder_path is given, will group by prefix, merge each group with 'concat', then merge all with 'align'.
    normalize_logical_clock = False
    args = sys.argv[1:]
    if "--normalize-logical-clock" in args:
        normalize_logical_clock = True
        args.remove("--normalize-logical-clock")

    if len(args) == 2 and pathlib.Path(args[1]).is_dir():
        out = args[0]
        folder = pathlib.Path(args[1])
        custom_events = read_custom_events(folder / "event_log.jsonl")
        all_files = sorted(str(p) for p in folder.glob("*.pt.trace.json"))
        if not all_files:
            print(f"No .json files found in {folder}")
            sys.exit(1)
        groups = group_by_prefix(all_files)
        group_outputs = []
        temp_dir = folder / "_tmp_merge"
        temp_dir.mkdir(exist_ok=True)
        for prefix, files in groups.items():
            group_out = temp_dir / f"{prefix}_concat.json"
            merge_traces(
                files,
                group_out,
                mode="concat",
                group_prefix=prefix,
                normalize_logical_clock=normalize_logical_clock,
                custom_events=custom_events.get(prefix, []),
            )
            group_outputs.append(str(group_out))
        # Now align all group outputs
        merge_traces(group_outputs, out, mode="align", normalize_logical_clock=normalize_logical_clock)
        print(f"Merged {len(all_files)} traces from {len(groups)} groups -> {out} (group-concat, then align)")
    else:
        # Fallback to original CLI
        out = args[0]
        mode = args[1] if len(args) > 2 else "concat"
        files = args[2:] if len(args) > 2 else args[1:]
        merge_traces(files, out, mode, normalize_logical_clock=normalize_logical_clock)
        print(f"Merged {len(files)} traces -> {out} (mode={mode})")
