from __future__ import annotations

import sys

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Any
from simsuite.common import dprint

if TYPE_CHECKING:
    from simsuite.device import Device
    from simsuite.world import World


@dataclass
class NetworkArgs:
    devices: list[Device]
    bandwidth: float
    latency: float
    full_duplex: bool = True


@dataclass
class Transmit:
    data: Any
    size: float
    start_time: float
    transferred_so_far: float
    id: str
    internal_id: int
    end_time: Optional[float] = None

    def completed(self) -> bool:
        return self.end_time is not None

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Transmit) and self.id == value.id and self.start_time == value.start_time

    def __repr__(self) -> str:
        return (
            f"Transmit(id={self.id!r}, size={self.size}, start_time={self.start_time}, "
            f"transferred_so_far={self.transferred_so_far})"
        )


class Network:
    transmits: list[Transmit]
    internal_clock: float
    world: World

    def __init__(self, args: NetworkArgs):
        self.devices = args.devices
        self.bandwidth = args.bandwidth
        self.latency = args.latency
        self.transmits = []
        self.full_duplex = args.full_duplex
        self.internal_clock = 0.0
        self.internal_id_counter = 0

    def transmit(self, data: Any, size: float, world_time: float, id: str):
        """
        When a network transmit starts, we re-estimate arrival times for all
        in-flight transmits, and then schedule the new transmit.
        """

        # Transfer starts after RTT (latency)
        transmit = Transmit(data, size, world_time + self.latency, 0.0, id, self.internal_id_counter)
        self.internal_id_counter += 1
        self.transmits.append(transmit)
        self.world.event_logger.log_event(
            {
                "time": world_time,
                "action": "transmit_start",
                "id": id,
                "internal_id": transmit.internal_id,
                "size": size / 8,
            }
        )

        for device in self.world.devices:
            if device.state.dependency == id:
                device.state.dependency = transmit

    def complete_transmit(self, transmit: Transmit, device_time: float) -> Optional[Transmit]:
        """
        Complete a transmit, removing it from the
        list of in-flight transmits and returning it.
        """

        if transmit in self.transmits:
            self.transmits.remove(transmit)
            return transmit
        return None

    def search_transmit(self, id: str) -> Transmit | str:
        matches = [transmit for transmit in self.transmits if transmit.id == id]
        if matches:
            return min(matches, key=lambda t: t.start_time)
        return id

    def step(self, max_time: float) -> Tuple[float, Optional[Transmit]]:
        """
        Step the network simulation forward by one tick. This updates the
        amount of data transferred for each in-flight transmit, and removes
        any transmits that have completed.

        Returns the time when the next transmit will complete, or -1 if there are
        no in-flight transmits.
        """

        # Skip if we are already at or beyond max_time.
        if self.internal_clock >= max_time:
            dprint(f"Network already at or beyond max_time {max_time} with internal clock {self.internal_clock}.")
            return (self.internal_clock, None)

        # Helper to compute true bandwidth based on full/half duplex setting.
        def true_bandwidth(device: Device) -> float:
            if self.full_duplex:
                # Full-duplex: always use full bandwidth
                return self.bandwidth
            else:
                # Half-duplex: divide bandwidth by number of in-flight transmits
                on_going_transmits = [
                    t for t in self.transmits if t.start_time <= self.internal_clock and not t.completed()
                ]
                return self.bandwidth / max(1, len(on_going_transmits))

        # Helper to compute when a transmit will complete.
        # This assumes there will be no changes in bandwidth until the transmit completes.
        def end_time(transmit: Optional[Transmit]) -> float:
            if not transmit:
                return float("inf")
            rem_bytes = transmit.size - transmit.transferred_so_far
            rem_time = max(sys.float_info.epsilon, rem_bytes / true_bandwidth())
            return self.internal_clock + rem_time

        dprint("Current transmits in network:")
        for t in self.transmits:
            dprint(f" --> {t}")
        dprint("==============================")

        # Find the next transmit to complete.
        available_transmits = [t for t in self.transmits if not t.completed()]
        first_to_end = min(available_transmits, key=end_time, default=None)
        # Find the time when the next transmit will end assuming there will be no changes in bandwidth.
        first_end_time = end_time(first_to_end)

        # Find the time when the next transmit will start.
        first_start_time = min(
            (t.start_time for t in self.transmits if t.start_time > self.internal_clock), default=float("inf")
        )

        if first_start_time == float("inf") and first_end_time == float("inf"):
            # No in-flight transmits, don't need to move time.
            dprint("No in-flight transmits.")
            return (-1, None)

        dprint(f"Network step at time {self.internal_clock}, simulate until: {max_time}:")
        dprint(f"  Next transmit to start at {first_start_time}, next to end at {first_end_time}.")

        # If next transmit to start is before some transmit to end, we can only
        # move time forward until the next transmit starts.
        if first_end_time > first_start_time:
            dprint(f"Moving time forward to next transmit start at {first_start_time} or the max_time {max_time}.")
            time_delta = min(max_time, first_start_time) - self.internal_clock
            if time_delta < 0:
                raise ValueError("Time delta is negative, something went wrong.")

            for transmit in self.transmits:
                # If the transmit hasn't started yet, skip it.
                if transmit.start_time > self.internal_clock + time_delta or transmit.completed():
                    continue
                transmit.transferred_so_far += time_delta * true_bandwidth()
                if transmit.transferred_so_far > transmit.size + 1:
                    breakpoint()

            self.internal_clock += time_delta
            return (self.internal_clock, None)

        # A transmit will end before the next transmit starts.
        else:
            time_delta = min(max_time, first_end_time) - self.internal_clock
            dprint(f"Moving time forward to next transmit end at {first_end_time} or the max_time {max_time}.")
            dprint(f"  Time delta: {time_delta}")
            if time_delta < 0:
                raise ValueError("Time delta is negative, something went wrong.")

            for transmit in self.transmits:
                # If the transmit hasn't started yet, skip it.
                if transmit.start_time > self.internal_clock + time_delta or transmit.completed():
                    continue

                # Simulate events happened between [internal_clock, max_time]
                transmit.transferred_so_far += time_delta * true_bandwidth()
                if transmit.transferred_so_far > transmit.size + 1:
                    breakpoint()

            self.internal_clock += time_delta

            transmit = None
            if self.internal_clock >= first_end_time and first_to_end is not None:
                dprint(f"  Transmit {first_to_end.id} completed at time {self.internal_clock}.")
                transmit = first_to_end
                # Make sure the first to end is actually done, no rounding errors
                first_to_end.transferred_so_far = first_to_end.size
                first_to_end.end_time = self.internal_clock

                self.world.event_logger.log_event(
                    {
                        "time": first_to_end.end_time,
                        "action": "transmit_end",
                        "id": transmit.id,
                        "internal_id": transmit.internal_id,
                        "duration": self.latency - transmit.start_time + first_to_end.end_time,
                    }
                )

            return (self.internal_clock, transmit)
