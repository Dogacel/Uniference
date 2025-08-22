from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsuite.device import Device


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
