from __future__ import annotations
import os

import simpy
import torch

from time import perf_counter
from typing import Literal, Optional, Callable, Any
from greenlet import greenlet
from torch.distributed import destroy_process_group

from simsuite.chan import Chan
from simsuite.dependency import Dependency
from simsuite.device import Device, DeviceArgs, DeviceState
from simsuite.event_logger import WorldEventLogger
from simsuite.network import Network, NetworkArgs
from simsuite.profiler import TorchProfiler
from simsuite.units import ms

from ns.packet.sink import PacketSink
from ns.port.wire import Wire
from ns.switch.switch import SimplePacketSwitch

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
    _runq = []  # round-robin queue of runnable greenlets
    performance_mode: bool
    debug_run: bool
    output_file: str

    backend: Literal['simulation', 'pytorch']

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

        self.simpy_env = simpy.Environment()
        self.router = SimplePacketSwitch(
            self.simpy_env,
            nports=1,
            port_rate=10_000_000,
            buffer_size=100,
            debug=False,
        )


    def device(self, deviceArgs: DeviceArgs, program: Program):
        device = Device(deviceArgs, program, self)
        self.devices.append(device)
        self.device_states[device] = DeviceState()
        device.state = self.device_states[device]

        device.wire_up = Wire(self.simpy_env, deviceArgs.spec.inherent_latency)
        device.wire_down = Wire(self.simpy_env, deviceArgs.spec.inherent_latency)

        device.sink = PacketSink(self.simpy_env, rec_flow_ids=True)

        device.wire_up.out = self.router
        device.wire_down.out = device.sink

        self.event_logger.log_event({"device": device.name, "action": "created"})
        return device

    def network(self, networkArgs: NetworkArgs):
        network = Network(networkArgs)
        self.networks.append(network)
        return network

    def latency_between(self, device: Device, other_device: Device, transfer_size: float = 0) -> float:
        if device == other_device:
            return 0.0
        connections = [conn for conn in self.networks if device in conn.devices and other_device in conn.devices]
        if connections:
            return min(conn.latency for conn in connections) + (
                transfer_size / max(conn.bandwidth for conn in connections)
            )
        raise ValueError("No network connection found")

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

    def reset_clock(self):
        for state in self.device_states.values():
            state.clock = 0.0
            state.last_run_time = perf_counter()
        self.max_time = 0.0

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

    def print_stats(self, start_time: float) -> None:
        end_time = perf_counter()
        print("\n")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        for chan in self.chans:
            print(f"{chan.name}.total_transferred_bytes: {chan.total_transferred_bytes / 1_000_000:.2f} MB")
            print(f"{chan.name}.total_transferred_count: {chan.total_transferred_count}")
            print(
                f"{chan.name} bandwith used: {chan.total_transferred_bytes / 1_000_000 / (end_time - start_time):.2f} MB/s"
            )

            chan.reset_counters()

    def after_time(self, duration: float) -> PreparedEvent:
        event = PreparedEvent()
        event.condition = lambda world: world.max_time >= duration
        self.events.append(event)
        return event

    def add_dependency(self, device: Device, dependency: Dependency):
        self.device_states[device].dependencies.append(dependency)

    def reset_timers(self):
        for state in self.device_states.values():
            state.clock = 0.0
            state.last_run_time = perf_counter()
        self.max_time = 0.0

    @torch.inference_mode()
    def run(self):
        self._run()

    def _run(self):
        # The devices calling calls such as chan("foo").receive() should create dependencies on other channels.
        # Those dependencies will be modelled as graphs, each step will check if the dependencies of each request
        # is satisfied or not. This graph traversal also makes sure there are no deadlocks.
        #
        # Example, (. means idle, - means running)
        # device1: |------------ chan.send(x)) ............ |chan.receive() ----------- end)
        # device2: .............|-------------------------- chan.send(x)) ............ |chan.receive() ---------- end)

        def device_run_wrapper(device: Device):
            device.state.last_run_time = perf_counter()
            device.run()
            device.state.sync_clock()

        for device in self.devices:
            device.initialize()
            self._runq.append(
                (
                    device,
                    greenlet(lambda: device_run_wrapper(device)),
                )
            )

        deadlock_graph: list[Device] = list()
        id = 0
        yield_count = 0

        # Calculate yield normalization factor
        start_time = perf_counter()
        end_time = perf_counter()

        def gr_perf():
            nonlocal start_time, end_time
            end_time = perf_counter()
            g = greenlet.getcurrent().parent
            g.switch()
            start_time = perf_counter()

        gr = greenlet(gr_perf)
        gr.switch()
        gr.switch()

        print(f"Yield overhead: {(end_time - start_time) * 1_000_000:.2f} us")

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

            if len(deadlock_graph) == sum(1 if not d.terminated else 0 for d in self.devices):
                state.clock += 1 * ms
                self.max_time = max(self.max_time, state.clock)

            # If we are at max_time, but not all devices are terminated and have their dependencies met,
            if state.clock == self.max_time and not all(
                state.clock == self.max_time
                if not device.terminated and all(dep.condition() for dep in state.dependencies)
                else True
                for device, state in self.device_states.items()
            ):
                self._runq.append((device, g))
                continue

            dependencies = state.dependencies

            if all(dep.condition() for dep in dependencies):
                dependency_times = [dep.time() for dep in dependencies]
                dependency_times = [time if time is not None else 0 for time in dependency_times]
                if dependency_times:
                    state.clock = max(state.clock, max(dependency_times))

                dependencies.clear()
                if device in deadlock_graph:
                    deadlock_graph.remove(device)

                self.event_logger.log_event({"device": device.name, "action": "running", "time": state.clock})

                state.last_run_time = perf_counter()

                if self.debug_run and not device.state.warmup:
                    with TorchProfiler(
                        out_dir="profile_out",
                        trace_name=f"{device.name}_run",
                        id=str(id),
                        report=self.debug_run,
                    ) as P:
                        torch.autograd._add_metadata_json("logical_clock", str(state.clock))
                        with P.record("device_run"):
                            g.switch()
                            # state.sync_clock()
                            yield_count += 1
                else:
                    g.switch()
                    # state.sync_clock()
                    yield_count += 1

                id += 1

                self.event_logger.log_event({"device": device.name, "action": "idle", "time": state.clock})
                self.max_time = max(self.max_time, state.clock)
            else:
                if device not in deadlock_graph:
                    deadlock_graph.append(device)

            if not g.dead:
                self._runq.append((device, g))
            else:
                print(f"Device {device.name} is terminated after {state.clock} seconds")
                self.event_logger.log_event({"device": device.name, "action": "finished", "time": state.clock})
                device.terminated = True

        if self.debug_run:
            self.event_logger.dump_events()

        self.event_logger.report_run(
            time=self.max_time,
            output_file=self.output_file,
            params=self.runtime_params | {"yield_count": yield_count},
        )

        if self.device_type == "cuda":
            destroy_process_group()
