from time import perf_counter
from typing import Optional
from dataclasses import dataclass
from typing import Callable
from copy import deepcopy
from typing import Any
from torch import Tensor
from greenlet import greenlet

import threading


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
    callback: Callable[[Any], None]

    def hook(self, event_type: str, callback: Callable[[Any], None]):
        self.event_type = event_type
        self.callback = callback


class Program:
    def __init__(self):
        pass

    def runnable(self, realm: "Realm") -> None:
        raise NotImplementedError


class Device:
    def __init__(self, args: DeviceArgs, program: Program, world: "World"):
        self.spec = args.spec
        self.client = args.client
        self.name = args.name
        self.program = program
        self.world = world
        self.terminated = False

    def step(self):
        """
        Step should do a unit step of interaction and yield control back to the scenario.
        The unit step of interaction might include long-running tasks but it can't depend on
        other events or tasks.
        """
        self.program.runnable(Realm(world=self.world, me=self))

    def terminate(self):
        self.terminated = True


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


class World:
    devices: list[Device]
    networks: list[Network]
    clock: float
    chans: list["Chan"]

    device_dependencies: dict[Device, list[Callable[[], bool]]]
    _runq = []  # round-robin queue of runnable greenlets

    def __init__(self) -> None:
        self.devices = []
        self.networks = []
        self.clock = 0.0
        self.chans = []
        self.device_dependencies = {}

    def device(self, deviceArgs: DeviceArgs, program: Program):
        device = Device(deviceArgs, program, self)
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
        print(f"Yielding event: {event_type} with data: {data}")
        g = greenlet.getcurrent().parent
        if g is not None:
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
        return PreparedEvent()

    def add_dependency(self, device: Device, dependency: Callable[[], bool]):
        if device not in self.device_dependencies:
            self.device_dependencies[device] = []
        self.device_dependencies[device].append(dependency)

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
                    greenlet(lambda: device.step()),
                )
            )

        print(self._runq)

        while self._runq:
            device, g = self._runq.pop(0)

            if device.terminated:
                continue

            if device in self.device_dependencies:
                dependencies = self.device_dependencies[device]
                # Check for satisfied dependencies
                if all(dep() for dep in dependencies):
                    dependencies.clear()
                    # print(f"All dependencies are met for {device.name}, running the greenlet.")
                    g.switch()
                else:
                    # print(f"Dependencies not met for {device.name}, skipping the greenlet.")
                    pass
            else:
                # print(f"No dependencies found for {device.name}, running the greenlet.")
                g.switch()

            if not g.dead:
                self._runq.append((device, g))


@dataclass
class Realm:
    world: World
    me: Device


# Maybe a script that determines the network conditions when installed on two separate devices?


class Chan:
    name: str
    listeners: list[Callable[[Any], None]]
    total_transferred_bytes: int
    total_transferred_count: int
    queue: list[Any]

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
        self.world.add_dependency(me, lambda: len(self.queue) > 0)
        self.world.xyield(f"chan {self.name} receive()")
        data = self.queue.pop(0)
        return data

    def send(self, data):
        with self._lock:
            self.queue.append(deepcopy(data))

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
