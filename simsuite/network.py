from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from simsuite.common import dprint
from typing import TYPE_CHECKING, Optional, Tuple, Any

if TYPE_CHECKING:
    from simsuite.device import Device
    from simsuite.world import World


@dataclass
class NetworkArgs:
    devices: list[Device]
    network_params: list[float]
    full_duplex: bool = True


@dataclass
class Transmit:
    data: Any
    size: float
    start_time: float
    transferred_so_far: float
    id: str
    internal_id: int
    source_device: Device
    target_device: Device
    internal_clock: float = 0.0
    end_time: Optional[float] = None

    def completed(self) -> bool:
        return self.end_time is not None

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Transmit) and self.id == value.id and self.start_time == value.start_time

    def __repr__(self) -> str:
        return (
            f"Transmit(id={self.id!r}, size={self.size}, start_time={self.start_time}, "
            f"transferred_so_far={self.transferred_so_far}), end_time={self.end_time}"
        )


class Network:
    transmits: list[Transmit]
    internal_clock: float
    world: World

    def __init__(self, args: NetworkArgs):
        self.devices = args.devices
        self.transmits = []
        self.full_duplex = args.full_duplex
        self.internal_clock = 0.0
        self.internal_id_counter = 0
        self.network_params = args.network_params

    def transmit(self, data: Any, size: float, world_time: float, id: str, source: Device, target: Device):
        """
        When a network transmit starts, we re-estimate arrival times for all
        in-flight transmits, and then schedule the new transmit.
        """

        transmit = Transmit(
            data, size, world_time, 0.0, id, self.internal_id_counter, source_device=source, target_device=target
        )
        self.internal_id_counter += 1
        self.transmits.append(transmit)
        self.world.event_logger.log_event(
            {
                "time": world_time,
                "action": "transmit_start",
                "id": id,
                "internal_id": transmit.internal_id,
                "source_device": source.name,
                "target_device": target.name,
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

        def active_transmits(transmit: Transmit) -> int:
            # If start_time == internal_clock, we consider it already started.
            # TODO: Maybe max(1, outgoing from source, incoming to target).
            return len(
                [
                    t
                    for t in self.transmits
                    if not t.completed()
                    and t.start_time <= self.internal_clock
                    and t.source_device == transmit.source_device
                ]
            )

        # Helper to compute true bandwidth based on full/half duplex setting.
        def true_bandwidth(transmit: Transmit, delta_time: float) -> float:
            transferred = bytes_transferred_in_window(
                transmit.internal_clock,
                delta_time,
                active_transmits(transmit),
                *self.network_params,
            )

            return transferred * 8

        # Helper to compute when a transmit will complete.
        # This assumes there will be no changes in bandwidth until the transmit completes.
        def end_time(transmit: Optional[Transmit]) -> float:
            if not transmit:
                return float("inf")

            if transmit.size == transmit.transferred_so_far:
                return self.internal_clock + 0.00000000000001

            duration_for_transmit = duration_for(
                transmit.internal_clock,
                (transmit.size - transmit.transferred_so_far) / 8,
                active_transmits(transmit),
                *self.network_params,
            )

            # print(f"Duration for transmit calculation: {duration_for_transmit}")

            if duration_for_transmit < 0:
                duration_for_transmit = 0

            return self.internal_clock + duration_for_transmit

        dprint("Current transmits in network:")
        for t in self.transmits:
            dprint(f" --> {t}")
        dprint("==============================")

        # Find the next transmit to complete.
        # If start_time == internal_clock, we consider it already started.
        # Don't filter out entities that are .completed(), because they need to be picked up by the device
        # and removed from the network. As the same device can schedule new transmits, moving time forward
        # would cause device to send messages from the past that arrive to the future, thus having longer
        # than expected durations.
        available_transmits = [
            t
            for t in self.transmits
            # This line behaves weird for async_ops, so try changing != to == while running async benchmark
            if t.start_time <= self.internal_clock and ((not t.completed()) or t.target_device.state.dependency != t)
        ]
        # print(f"Available transmits: {available_transmits}")
        first_to_end = min(available_transmits, key=end_time, default=None)

        if first_to_end is not None and first_to_end.completed():
            # All available transmits are already completed, we should wait until the device picks it up.
            dprint(
                f"Device {first_to_end.target_device.name} transmit {first_to_end.id} already completed, waiting for device to pick it up."
            )
            return (-1, None)

        # Find the time when the next transmit b_voltagewill end optimistically assuming there will be no changes in bandwidth.
        # print(f"First to end: {first_to_end}")
        first_end_time = end_time(first_to_end)
        # print(f"First end time: {first_end_time}")

        # Find the time when the next transmit will start.
        # If start_time == internal_clock, we consider it already started.
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
                if transmit.start_time > self.internal_clock or transmit.completed():
                    continue
                transmit.transferred_so_far = min(
                    transmit.transferred_so_far + true_bandwidth(transmit, time_delta),
                    transmit.size,
                )
                transmit.internal_clock += time_delta

                if transmit.transferred_so_far > transmit.size + 1:
                    raise ValueError("Transmit transferred more than its size, something went wrong.")

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
                if transmit.start_time > self.internal_clock or transmit.completed():
                    continue

                # Simulate events happened between [internal_clock, max_time]
                transmit.transferred_so_far = min(
                    transmit.transferred_so_far + true_bandwidth(transmit, time_delta), transmit.size
                )
                transmit.internal_clock += time_delta

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
                        "duration": first_to_end.end_time - first_to_end.start_time,
                        "source_device": transmit.source_device.name,
                        "target_device": transmit.target_device.name,
                    }
                )

            return (self.internal_clock, transmit)


def duration_for(t0, S, active_transmits, alpha, beta):
    """
    How long (seconds) from t0 to finish S additional bytes.

    alpha: latency (seconds)
    beta: 1/bandwidth (seconds/byte)
    """

    if S <= 0:
        return 0

    # If time has already passed latency, we can use full bandwidth.
    return max(0, alpha - t0) + S * (beta * max(active_transmits, 1))


def bytes_transferred_in_window(t0, dt, active_transmits, alpha, beta):
    """
    Bytes transferred between times [t0, t0+dt].

    alpha: latency (seconds)
    beta: 1/bandwidth (seconds/byte)
    """
    # Make sure we only consider time outside of latency.
    non_latency_time = max(0, dt - max(0, alpha - t0))

    return non_latency_time / (beta * max(active_transmits, 1))
