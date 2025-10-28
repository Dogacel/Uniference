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
from simsuite.device import Device, DeviceArgs, DeviceState, RemoteDevice, DeviceSpec
from simsuite.event_logger import WorldEventLogger
from simsuite.network import Network, NetworkArgs
from simsuite.profiler import TorchProfiler
from simsuite.common import dprint
from simsuite.pytorch_chan import PytorchChan
from simsuite.remote_chan import RemoteChan


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

    backend: Literal["simulation", "pytorch", "remote"]

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

        self.backend = kwargs.get("backend", os.getenv("WORLD_BACKEND", "simulation"))
        self.mode = kwargs.get("mode", "server")

        print("Using backend:", self.backend)

    def device(self, deviceArgs: DeviceArgs, program: Program):
        device = Device(deviceArgs, program, self)
        self.devices.append(device)
        self.device_states[device] = DeviceState(device)
        device.state = self.device_states[device]

        self.event_logger.log_event({"device": device.name, "action": "created"})
        return device

    def remote_device(self, loop, reader, writer, client_id: str):
        device = RemoteDevice(DeviceArgs(spec=DeviceSpec(), client=True, name=client_id), self, loop, reader, writer)
        self.devices.append(device)
        self.device_states[device] = DeviceState(device)
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
        if self.backend == "pytorch":
            new_chan = PytorchChan(tag, self)
        elif self.backend == "remote" and self.mode == "client":
            new_chan = RemoteChan(tag, self)
        else:
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

        for device in self.devices:
            if device.remote:
                device.run(warmup=warmup)

        id = 0
        yield_count = 0
        deadlock_checks = 0

        self.event_logger.log_event({"action": "simulation_start", "time": 0})

        # Event loop
        while self._runq:
            for event in self.events:
                if event.condition(self):
                    event.callback()
                    self.events.remove(event)
            device, g = self._runq.pop(0)
            state = self.device_states[device]

            dprint("Simulating device:", device.name, "at time", state.clock)

            if device.terminated:
                print(f"Device {device.name} is terminated after {state.clock} seconds")
                continue

            # Check if we are in a deadlock.
            # Worst case, only a single device is runnable, so we won't do anything for len(devices)-1 checks.
            if deadlock_checks > len(self.devices) + sum([len(x.transmits) for x in self.networks]):
                if all(d.state.dependency is not None for d in self.devices if not d.terminated):
                    for d in self.devices:
                        if not d.terminated and d.state.dependency is not None:
                            dprint(f"Deadlock detected: device {d.name} is waiting on {d.state.dependency}")
                    raise RuntimeError("Deadlock detected: all devices are waiting but no network activity")

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
                    continue

                # Reset deadlock counter
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
                dprint(f"Device {device.name} dependency completed: {state.dependency}")
                millis_took = (state.dependency.end_time - state.clock) * 1000
                if millis_took > 100:
                    breakpoint()
                state.clock = max(state.clock, state.dependency.end_time)
                if device.remote:
                   world.networks[0].complete_transmit(state.dependency, state.clock)

            # Device is runnable if no network dependency exists or the dependency is completed.
            if state.dependency is None or dependency_completed:
                # Reset deadlock counter
                deadlock_checks = 0
                self.event_logger.log_event({"device": device.name, "action": "running", "time": state.clock})

                state.last_run_time = perf_counter()

                if self.debug_run and not device.state.warmup and not device.remote:
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
                elif device.remote:
                    print("Running remote device:", device.name)
                    # breakpoint()
                    device.wc.run_continue(state.dependency.data if state.dependency is not None else None)
                    remote_state = device.sync_remote_state()

                    if remote_state.to_send is not None:
                        print(f"Device {device.name} sending data to remote target")
                        data = remote_state.to_send["data"]
                        target = remote_state.to_send["target"]
                        target = [d for d in self.devices if d.name == target][0]
                        size = remote_state.to_send["size"]
                        time = remote_state.to_send["time"]
                        transmit_id = remote_state.to_send["id"]

                        network.transmit(data, size=size, world_time=time, id=transmit_id, source=device, target=target)

                    if remote_state.dependency is not None:
                        print(f"Device {device.name} receiving data from remote dependency: {remote_state.dependency}")
                        device.state.dependency = self.networks[0].search_transmit(remote_state.dependency)

                    yield_count += 1
                else:
                    g.switch()
                    yield_count += 1

                self.event_logger.log_event({"device": device.name, "action": "idle", "time": state.clock})

                self.max_time = max(self.max_time, state.clock)
            else:
                deadlock_checks += 1
                dprint(f"Device {device.name} yielding on dependency {state.dependency}")

            if not g.dead:
                self._runq.append((device, g))
            else:
                print(f"Device {device.name} is terminated after {state.clock} seconds")
                self.event_logger.log_event({"device": device.name, "action": "finished", "time": state.clock})
                device.terminated = True

        self.event_logger.log_event({"action": "simulation_end", "time": self.max_time})

        if not warmup:
            log_file_name = self.event_logger.dump_events()
            transmit_stats = self.event_logger.transmit_stats()
            self.event_logger.report_run(
                time=self.max_time,
                output_file=self.output_file,
                params=self.runtime_params
                | transmit_stats
                | {
                    "yield_count": yield_count,
                    "log_file": log_file_name,
                },
            )

    def destroy(self):
        if self.backend == "remote":
            for device in self.devices:
                if device.remote:
                    device.wc.close()
            self.background_server.stop()

        if self.device_type == "cuda":
            destroy_process_group()
            destroy_model_parallel()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
