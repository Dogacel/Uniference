from __future__ import annotations
import os

from fairscale.nn.model_parallel.initialize import destroy_model_parallel
import torch
import gc

from time import perf_counter
from typing import Literal, Optional, Callable, Any
from greenlet import greenlet
from torch.distributed import destroy_process_group

from simsuite.chan import Chan
from simsuite.device import Device, DeviceArgs, DeviceState
from simsuite.event_logger import WorldEventLogger
from simsuite.network import Network, NetworkArgs
from simsuite.profiler import TorchProfiler
from simsuite.common import dprint


class PreparedEvent:
    event_type: str
    condition: Callable[["World"], bool]
    callback: Callable[[], None]

    def hook(self, event_type: str, callback: Callable[[], None]) -> "PreparedEvent":
        self.event_type = event_type
        self.callback = callback
        return self


class Program:
    def __init__(self):
        pass

    def initialize(self, me: Device) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        raise NotImplementedError


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


class World:
    devices: list[Device]
    networks: list[Network]
    chans: list[Chan]
    event_logger: WorldEventLogger
    events: list[PreparedEvent]

    device_states: dict[Device, DeviceState]
    max_time: float
    _runq: list[tuple[Device, greenlet]] = []  # round-robin queue of runnable greenlets
    performance_mode: bool
    debug_run: bool
    output_file: str

    backend: Literal["simulation", "pytorch"]

    def __init__(self, **kwargs) -> None:
        self.devices = []
        self.networks = []
        self.chans = []
        self.event_logger = WorldEventLogger()
        self.events = []
        self.device_states = {}
        self.max_time = 0.0
        self.performance_mode = kwargs.get("performance_mode", False)
        self.debug_run = kwargs.get("debug_run", False)
        self.output_file = kwargs.get("output_file", "results/run_report.json")
        self.device_type = get_device()
        self.backend = kwargs.get("backend", "simulation")

        print("Using backend:", self.backend)

    def device(self, deviceArgs: DeviceArgs, program: Program):
        device = Device(deviceArgs, program, self)
        self.devices.append(device)
        self.device_states[device] = DeviceState()
        device.state = self.device_states[device]

        self.event_logger.log_event({"device": device.name, "action": "created"})
        return device

    def network(self, networkArgs: NetworkArgs):
        network = Network(networkArgs)
        network.world = self
        self.networks.append(network)
        return network

    def xyield(self, device: Optional[Device], event_type: str, data: Optional[Any] = None):
        g = greenlet.getcurrent().parent
        if g is not None and not self.performance_mode:
            if self.device_type == "cuda":
                torch.cuda.synchronize()

            if device is not None:
                device.state.sync_clock()

            g.switch()

            if device is not None:
                device.state.last_run_time = perf_counter()

    def set_runtime_params(self, params: dict[str, Any]):
        self.runtime_params = params

    def chan(self, tag: str) -> "Chan":
        for chan in self.chans:
            if chan.name == tag:
                return chan
        new_chan = Chan(tag, self)
        self.chans.append(new_chan)
        self.event_logger.log_event({"chan": tag, "action": "created"})
        return new_chan

    @torch.inference_mode()
    def run(self, debug_run: bool = False, warmup: bool = False):
        self._run(debug_run=debug_run, warmup=warmup)

        # Post-run cleanup
        self.events = []
        self.event_logger.events = []

        for state in self.device_states.values():
            state.clock = 0.0
            state.last_run_time = perf_counter()
            state.dependency = None

        for device in self.devices:
            device.terminated = False

        for network in self.networks:
            network.internal_clock = 0.0
            network.transmits = []

        for chan in self.chans:
            chan.listeners = []
            # chan.subscribers = []

        self._runq = []
        self.max_time = 0.0

        gc.collect()

    def _run(self, debug_run: bool, warmup: bool):
        # The devices calling calls such as chan("foo").receive() should create dependencies on other channels.
        # Those dependencies will be modelled as graphs, each step will check if the dependencies of each request
        # is satisfied or not. This graph traversal also makes sure there are no deadlocks.
        #
        # Example, (. means idle, - means running)
        # device1: |------------ chan.send(x)) ............ |chan.receive() ----------- end)
        # device2: .............|-------------------------- chan.send(x)) ............ |chan.receive() ---------- end)

        def device_run_wrapper(device: Device):
            device.state.last_run_time = perf_counter()
            device.run(warmup=warmup)
            if self.device_type == "cuda":
                torch.cuda.synchronize()
            device.state.sync_clock()

        for device in self.devices:
            device.initialize()
            self._runq.append(
                (
                    device,
                    greenlet(lambda: device_run_wrapper(device)),
                )
            )

        id = 0
        yield_count = 0
        deadlock_checks = 0

        # Event loop
        while self._runq:
            for event in self.events:
                if event.condition(self):
                    event.callback()
                    self.events.remove(event)
            device, g = self._runq.pop(0)
            state = self.device_states[device]

            if device.terminated:
                print(f"Device {device.name} is terminated after {state.clock} seconds")
                continue

            # Simulate network

            # A network simulation can only run up to the point where the next transmit completes
            # or the next device becomes runnable. Because otherwise, the bandwidth sharing simulation won't be correct.
            run_until = min(
                (
                    device.state.clock
                    for device in self.devices
                    if not device.terminated and device.state.dependency is None
                ),
                default=float("inf"),
            )

            # Step all networks.
            for network in self.networks:
                t, transmit = network.step(
                    max_time=run_until,
                )

                # No in-flight transmits
                if t == -1:
                    deadlock_checks += 1
                    if deadlock_checks > len(self.devices) * 2:
                        if all(d.state.dependency is not None for d in self.devices if not d.terminated):
                            for d in self.devices:
                                if not d.terminated and d.state.dependency is not None:
                                    dprint(f"Deadlock detected: device {d.name} is waiting on {d.state.dependency}")
                            raise RuntimeError("Deadlock detected: all devices are waiting but no network activity")
                    continue

                deadlock_checks = 0
                # Move time forward if needed
                self.max_time = max(self.max_time, t)

            # Only simulate the device that has the lowest clock time and is runnable.
            # If the device is at max_time, it can only run if every other device is also at max_time.
            other_device_times = [
                d.state.clock for d in self.devices if d != device and not d.terminated and d.state.dependency is None
            ]
            if state.clock == self.max_time and not all(t == self.max_time for t in other_device_times):
                dprint(f"Device {device.name} at max_time {self.max_time} but others are not: {other_device_times}")
                self._runq.append((device, g))
                continue

            dependency_completed = hasattr(state.dependency, "completed") and state.dependency.completed()
            if dependency_completed:
                state.clock = max(state.clock, state.dependency.end_time)

            # Device is runnable if no network dependency exists or the dependency is completed.
            if state.dependency is None or dependency_completed:
                self.event_logger.log_event({"device": device.name, "action": "running", "time": state.clock})

                state.last_run_time = perf_counter()

                if self.debug_run and not device.state.warmup:
                    with TorchProfiler(
                        out_dir="profile_out",
                        trace_name=f"{device.name}_run",
                        id=str(id),
                        report=self.debug_run,
                    ) as p:
                        torch.autograd._add_metadata_json("logical_clock", str(state.clock))
                        with p.record("device_run"):
                            g.switch()
                            yield_count += 1
                            id += 1
                else:
                    g.switch()
                    yield_count += 1

                self.event_logger.log_event({"device": device.name, "action": "idle", "time": state.clock})

                self.max_time = max(self.max_time, state.clock)
            else:
                dprint(f"Device {device.name} yielding on dependency {state.dependency}")

            if not g.dead:
                self._runq.append((device, g))
            else:
                print(f"Device {device.name} is terminated after {state.clock} seconds")
                self.event_logger.log_event({"device": device.name, "action": "finished", "time": state.clock})
                device.terminated = True

        if not warmup:
            self.event_logger.dump_events()
            self.event_logger.report_run(
                time=self.max_time,
                output_file=self.output_file,
                params=self.runtime_params | {"yield_count": yield_count},
            )

    def destroy(self):
        if self.device_type == "cuda":
            destroy_process_group()
            destroy_model_parallel()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
