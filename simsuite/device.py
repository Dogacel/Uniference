from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, TYPE_CHECKING, Optional
from simsuite.server import WebClient

import asyncio

if TYPE_CHECKING:
    from simsuite.world import Program, World
    from simsuite.network import Transmit


@dataclass
class DeviceSpec:
    speed_scale: float = 1.0


@dataclass
class DeviceArgs:
    spec: DeviceSpec
    client: bool
    name: str


@dataclass
class DeviceState:
    device: "Device"
    dependency: Optional[Transmit | str] = None

    """
    Current time perception for the device.
    """
    clock: float = 0.0

    """
    Last time the device was run. This information is used to move clock forward.
    """
    last_run_time: float = 0.0

    """
    Whether the device is in warmup mode. In warmup mode, the profiler is not used.
    """
    warmup: bool = False

    def sync_clock(self) -> float:
        now = perf_counter()
        self.clock += (now - self.last_run_time) * self.device.spec.speed_scale
        self.last_run_time = now
        return self.clock


class Device:
    def __init__(self, args: DeviceArgs, program: Program, world: "World"):
        self.spec = args.spec
        self.client = args.client
        self.name = args.name
        self.program = program
        self.world = world
        self.terminated = False
        self.state: "DeviceState"
        self.initialized = False
        self.remote = False

    def initialize(self):
        if self.initialized:
            return
        print(f"Initializing device {self.name}")
        self.program.initialize(self)
        self.initialized = True

    def run(self, warmup: bool = False):
        if warmup:
            self.program.warmup()
        else:
            self.program.run()

    def terminate(self):
        self.world.event_logger.log_event(
            {"device": self.name, "action": "terminated", "time": self.world.device_states[self].clock}
        )
        self.terminated = True

    def send(self, chan: str, data: Any, transmit_id: str, force_time: Optional[float] = None):
        self.world.chan(chan).send(self, data, transmit_id, target=self, force_time=force_time)


class RemoteDevice(Device):
    def __init__(self, args: DeviceArgs, world: "World", loop, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        super().__init__(args, None, world)
        self.remote = True
        self.wc = WebClient(loop, reader, writer)

    def initialize(self):
        self.wc.initialize()
        self.initialized = True

    def run(self, warmup: bool = False):
        self.wc.run(warmup)

    def sync_remote_state(self):
        state_msg = self.wc.get_state()

        self.state.clock = state_msg.clock
        self.terminated = state_msg.terminated
