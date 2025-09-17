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
    full_duplex: bool = True


@dataclass
class Transmit:
    size: int
    start_time: float
    transferred_so_far: float


class Network:
    transmits: list[Transmit]

    def __init__(self, args: NetworkArgs):
        self.devices = args.devices
        self.bandwidth = args.bandwidth
        self.latency = args.latency
        self.transmits = []
        self.full_duplex = args.full_duplex

    def transmit(self, size: int, world_time: float):
        """
        When a network transmit starts, we re-estimate arrival times for all
        in-flight transmits, and then schedule the new transmit.
        """

        # Transfer starts after RTT (latency)
        self.transmits.append(Transmit(size, world_time + self.latency, 0.0))

    def step(self, max_time: float) -> float:
        """
        Step the network simulation forward by one tick. This updates the
        amount of data transferred for each in-flight transmit, and removes
        any transmits that have completed.

        Returns the time when the next transmit will complete, or -1 if there are
        no in-flight transmits.
        """

        def true_bandwidth() -> float:
            if self.full_duplex:
                return self.bandwidth
            else:
                return self.bandwidth / max(1, len(self.transmits))

        def end_time(transmit: Transmit) -> float:
            rem_bytes = transmit.size - transmit.transferred_so_far

            # For half-duplex networks, we divide bandwidth by the number of in-flight transmits

            return transmit.start_time + (rem_bytes / true_bandwidth())

        first_to_end = min(self.transmits, key=end_time)

        if not first_to_end:
            return -1

        first_end_time = end_time(first_to_end)

        if first_end_time > max_time:
            # No transmits will complete in this step, so just update all the
            # in-flight transmits and return max_time
            for transmit in self.transmits:
                time_delta = max_time - transmit.start_time

                transmit.transferred_so_far += time_delta * true_bandwidth()

                return max_time

        # One or more transmits will complete in this step. Update all the
        # in-flight transmits to the time when the first one completes, then
        # remove it from the list of in-flight transmits.

        for transmit in self.transmits:
            time_delta = first_end_time - transmit.start_time

            transmit.transferred_so_far += time_delta * true_bandwidth()

        self.transmits.remove(first_to_end)

        return first_end_time
