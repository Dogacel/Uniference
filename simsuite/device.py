from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from simsuite.dependency import Dependency
    from simsuite.world import Program, World


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
        self.clock += (now - self.last_run_time)
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

    def initialize(self):
        print(f"Initializing device {self.name}")
        self.program.initialize(self)

    def run(self):
        self.program.run()

    def terminate(self):
        self.world.event_logger.log_event(
            {"device": self.name, "action": "terminated", "time": self.world.device_states[self].clock}
        )
        self.terminated = True

    def send(self, chan: str, data: Any):
        clock = self.world.device_states[self].clock if self in self.world.device_states else self.world.max_time
        self.world.chan(chan).send(clock, data, self)

    def latency_to(self, other: "Device") -> float:
        return self.world.latency_between(self, other)
