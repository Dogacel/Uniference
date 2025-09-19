import os
import time
import functools
import torch
import json
from pathlib import Path
from typing import Optional
import inspect


class TorchProfiler:
    GLOBAL: "TorchProfiler | None" = None

    """
    Lightweight utility to profile time and collect ops + input shapes.
    - Supports manual spans via `record_function`.
    - Exports:
        - overall wall time
        - op-level stats with input shapes (CPU/GPU times)
        - optional Chrome trace (for TensorBoard/Perfetto)
    """

    def __init__(
        self,
        out_dir: str = "profile_out",
        trace_name: str = "run",
        use_cuda: bool = os.environ["DEVICE"] == "cuda" if "DEVICE" in os.environ else torch.cuda.is_available(),
        include_stack: bool = False,
        id: str = "0",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trace_name = trace_name
        self.use_cuda = use_cuda
        self.include_stack = include_stack
        self.prof: Optional[torch.profiler.profile] = None
        self.t0 = 0.0
        self.wall_ms = None
        self.id = id

        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._ctx = torch.profiler.profile(
            activities=activities,
            record_shapes=True,  # <-- collects input sizes
            with_flops=True,
            profile_memory=False,
            with_stack=self.include_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.out_dir), worker_name=self.trace_name),
        )

    def __enter__(self):
        TorchProfiler.GLOBAL = self
        self.t0 = time.perf_counter()
        self.prof = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self.prof is not None

        # Make sure all device work is finished before stopping timers.
        if self.use_cuda:
            torch.cuda.synchronize()
        self.wall_ms = (time.perf_counter() - self.t0) * 1000.0

        self._ctx.__exit__(exc_type, exc, tb)

        # Save summaries
        self._export()
        TorchProfiler.GLOBAL = None

    def record(self, name: str):
        """Manual annotation span."""
        return torch.profiler.record_function(name)

    def _export(self):
        assert self.prof is not None

        # Optional: per-event raw log (can be large). Comment out if not needed.
        raw_events = []
        for e in self.prof.events():
            raw_events.append(
                {
                    "name": e.name,
                    "cpu_time_us": getattr(e, "cpu_time", 0.0),
                    "cuda_time_us": getattr(e, "cuda_time", 0.0),
                    "input_shapes": getattr(e, "input_shapes", None),
                    "args": getattr(e, "args", {}),  # extra info (e.g. from @profiled)
                    "is_async": getattr(e, "is_async", False),
                    "scope": getattr(e, "scope", None),
                    "fwd_thread_id": getattr(e, "fwd_thread_id", None),
                }
            )

        with open(self.out_dir / f"{self.trace_name}_{self.id}_events.json", "w") as f:
            json.dump(raw_events, f, indent=2)


def profiled(
    name: str | None = None,
    *,
    cuda: bool | None = None,
):
    """
    Decorator that records a function call inside the *current* TorchProfiler.
    If no TorchProfiler is active, it still works as a plain timer+record_function.
    """

    def deco(fn):
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            prof = TorchProfiler.GLOBAL  # current active profiler, if any
            use_cuda = torch.cuda.is_available() if cuda is None else cuda

            # Map args/kwargs to parameter names
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Always mark as a record_function (shows up in trace/timeline)
            with torch.profiler.record_function(label):
                out = fn(*args, **kwargs)

            if use_cuda:
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Optionally log per-call timing into the profiler object
            if prof is not None:
                if not hasattr(prof, "_decorator_logs"):
                    prof._decorator_logs = []
                log_entry = {"name": label, "wall_time_ms": dt_ms}
                prof._decorator_logs.append(log_entry)

            return out

        return wrapped

    return deco
