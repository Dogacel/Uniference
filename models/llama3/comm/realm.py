from time import perf_counter
from typing import Optional
from dataclasses import dataclass, field
from typing import Callable
from copy import deepcopy
from typing import Any
from torch import Tensor
from greenlet import greenlet

import threading
import json

import torch


## Units

Kbps = 1000
Mbps = 1000 * Kbps
Gbps = 1000 * Mbps

KB = 1000
MB = 1000 * KB
GB = 1000 * MB

s = 1
ms = s / 1000
us = ms / 1000

FLOPs = 1
MFLOPs = 1000 * FLOPs
GFLOPs = 1000 * MFLOPs
TFLOPs = 1000 * GFLOPs


@dataclass
class DeviceSpec:
    flops: float
    mem: float
    max_bandwidth: float
    inherent_latency: float


@dataclass
class DeviceArgs:
    spec: DeviceSpec
    client: bool
    name: str


@dataclass
class EventLog:
    actor: str
    timestamp: float
    event: str


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

    def initialize(self, realm: "Realm") -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError


class Device:
    def __init__(self, args: DeviceArgs, program: Program, world: "World"):
        self.spec = args.spec
        self.client = args.client
        self.name = args.name
        self.program = program
        self.world = world
        self.terminated = False

    def initialize(self):
        print(f"Initializing device {self.name}")
        self.program.initialize(Realm(world=self.world, me=self))

    def run(self):
        self.program.run()

    def terminate(self):
        self.world.event_logger.log_event(
            {"device": self.name, "action": "terminated", "time": self.world.device_states[self].clock}
        )
        self.terminated = True

    def send(self, chan: str, data: Any):
        clock = self.world.device_states[self].clock
        self.world.chan(chan).send(clock, data)


@dataclass
class NetworkArgs:
    devices: list[Device]
    bandwidth: float
    latency: float


class Network:
    def __init__(self, args: NetworkArgs):
        self.devices = args.devices
        self.bandwidth = args.bandwidth
        self.latency = args.latency


class Dependency:
    def __init__(self, condition: Callable[[], bool], time: Callable[[], Optional[float]]):
        self.condition = condition
        self.time = time


@dataclass
class DeviceState:
    """
    A list of function that returns whether the device is ready or not.
    The device is ready if all dependencies return True.
    """

    dependencies: list[Dependency] = field(default_factory=list)

    """
    Current time perception for the device.
    """
    clock: float = 0.0


class WorldEventLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event: dict):
        self.events.append(event)

    def dump_events(self):
        with open("event_log.jsonl", "w") as f:
            for event in self.events:
                json.dump(event, f)
                f.write("\n")


class World:
    devices: list[Device]
    networks: list[Network]
    chans: list["Chan"]
    event_logger: WorldEventLogger
    events: list[PreparedEvent]

    device_states: dict[Device, DeviceState]
    max_time: float
    _runq = []  # round-robin queue of runnable greenlets
    performance_mode: bool

    def __init__(self) -> None:
        self.devices = []
        self.networks = []
        self.chans = []
        self.event_logger = WorldEventLogger()
        self.events = []
        self.device_states = {}
        self.max_time = 0.0
        self.performance_mode = False

    def device(self, deviceArgs: DeviceArgs, program: Program):
        device = Device(deviceArgs, program, self)
        device.initialize()
        self.devices.append(device)
        return device

    def network(self, networkArgs: NetworkArgs):
        network = Network(networkArgs)
        self.networks.append(network)
        return network

    def xyield(self, event_type: str, data: Optional[Any] = None):
        """
        Yield an event.
        """
        # print(f"Yielding event: {event_type} with data: {data}")
        g = greenlet.getcurrent().parent
        if g is not None and not self.performance_mode:
            g.switch()

    def chan(self, tag: str) -> "Chan":
        for chan in self.chans:
            if chan.name == tag:
                return chan
        new_chan = Chan(tag, self)
        self.chans.append(new_chan)
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

    @torch.inference_mode()
    def run(self):
        # The devices calling calls such as chan("foo").receive() should create dependencies on other channels.
        # Those dependencies will be modelled as graphs, each step will check if the dependencies of each request
        # is satisfied or not. This graph traversal also makes sure there are no deadlocks.
        #
        # Example, (. means idle, - means running)
        # device1: |------------ chan.send(x)) ............ |chan.receive() ----------- end)
        # device2: .............|-------------------------- chan.send(x)) ............ |chan.receive() ---------- end)

        for device in self.devices:
            self._runq.append(
                (
                    device,
                    greenlet(lambda: device.run()),
                )
            )
            self.device_states[device] = DeviceState()
            self.event_logger.log_event({"device": device.name, "action": "created"})

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

            # Skip if device is ahead of time in the simulation
            if state.clock == self.max_time and not all(
                state.clock == self.max_time
                if not device.terminated and all(dep.condition() for dep in state.dependencies)
                else True
                for device, state in self.device_states.items()
            ):
                self._runq.append((device, g))
                continue

            if device in self.devices:
                dependencies = state.dependencies
                # Check for satisfied dependencies
                if all(dep.condition() for dep in dependencies):
                    dependency_times = [dep.time() for dep in dependencies]
                    dependency_times = [time if time is not None else 0 for time in dependency_times]
                    if dependency_times:
                        state.clock = max(state.clock, max(dependency_times))

                    dependencies.clear()

                    self.event_logger.log_event({"device": device.name, "action": "running", "time": state.clock})
                    start_time = perf_counter()
                    g.switch()
                    end_time = perf_counter()
                    state.clock += end_time - start_time
                    self.event_logger.log_event({"device": device.name, "action": "idle", "time": state.clock})
                    self.max_time = max(self.max_time, state.clock)
            else:
                self.event_logger.log_event({"device": device.name, "action": "running", "time": state.clock})
                start_time = perf_counter()
                g.switch()
                end_time = perf_counter()
                state.clock += end_time - start_time
                self.event_logger.log_event({"device": device.name, "action": "idle", "time": state.clock})
                self.max_time = max(self.max_time, state.clock)

            if not g.dead:
                self._runq.append((device, g))
            else:
                print(f"Device {device.name} is terminated after {state.clock} seconds")
                self.event_logger.log_event({"device": device.name, "action": "finished", "time": state.clock})
                device.terminated = True

        self.event_logger.dump_events()


@dataclass
class Realm:
    world: World
    me: Device


@dataclass
class ChanItem:
    data: Any
    time: float


class Chan:
    name: str
    listeners: list[Callable[[Any], None]]
    total_transferred_bytes: int
    total_transferred_count: int
    queue: list[ChanItem]

    def __init__(self, name: str, world: World):
        self.name = name
        self._lock = threading.Lock()
        self.listeners = []
        self.total_transferred_bytes = 0
        self.total_transferred_count = 0
        self.queue = []
        self.world = world

    def reset_counters(self):
        with self._lock:
            self.total_transferred_bytes = 0
            self.total_transferred_count = 0

    def add_listener(self, listener: Callable[[Any], None]):
        self.listeners.append(listener)

    def receive(self, me: Device) -> Any:
        self.world.add_dependency(
            me,
            Dependency(
                condition=lambda: len(self.queue) > 0, time=lambda: self.queue[0].time if self.queue else None
            ),
        )
        self.world.xyield(f"chan {self.name} receive()")
        data = self.queue.pop(0)
        return data.data

    def send(self, clock: float, data: Any):
        with self._lock:
            self.queue.append(
                ChanItem(
                    data=deepcopy(data),
                    time=clock,
                )
            )

            for listener in self.listeners:
                listener(data)

            if isinstance(data, SyncKVCache):
                self.total_transferred_bytes += (
                    data.xk.numel() * data.xk.element_size()
                    + data.xv.numel() * data.xv.element_size()
                    + data.layer_id.bit_length() // 8
                    + data.start_pos.bit_length() // 8
                )
                self.total_transferred_count += 1

            if isinstance(data, SyncGen):
                self.total_transferred_bytes += (
                    data.next_token.numel() * data.next_token.element_size() + data.pos.bit_length() // 8
                )
                self.total_transferred_count += 1

        self.world.xyield(f"chan {self.name} send()")


@dataclass
class SyncKVCache:
    layer_id: int
    start_pos: int
    xk: Tensor
    xv: Tensor


@dataclass
class SyncGen:
    pos: int
    next_token: Tensor
