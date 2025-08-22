from __future__ import annotations

import threading
import uuid

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING ,Optional
from torch import Tensor
from simsuite.dependency import Dependency

if TYPE_CHECKING:
    from simsuite.world import World
    from simsuite.device import Device


@dataclass
class ChanItem:
    data: Any
    time: float
    id: str


class Chan:
    name: str
    listeners: list[Callable[[Any], None]]
    total_transferred_bytes: int
    total_transferred_count: int
    queue: list[ChanItem]
    subscribers: list[Device]
    sync_points: list[int]

    def __init__(self, name: str, world: World):
        self.name = name
        self._lock = threading.Lock()
        self.listeners = []
        self.total_transferred_bytes = 0
        self.total_transferred_count = 0
        self.queue = []
        self.subscribers = []
        self.sync_points = []
        self.world = world

    def reset_counters(self):
        with self._lock:
            self.total_transferred_bytes = 0
            self.total_transferred_count = 0

    def add_listener(self, listener: Callable[[Any], None]):
        self.listeners.append(listener)

    def subscribe(self, device: Device):
        self.subscribers.append(device)

    def unsubscribe(self, device: Device):
        self.subscribers.remove(device)

    def synchronize(self, me: Device):
        max_latency = max(self.world.latency_between(me, sub) for sub in self.subscribers)
        item = ChanItem(data=len(self.sync_points), time=me.state.clock + max_latency, id=str(uuid.uuid4()))
        self.queue.append(item)
        self.world.add_dependency(
            me,
            Dependency(
                condition=lambda: item.data in self.sync_points
                or len(self.queue) == len(self.subscribers)
                and all(me.state.clock >= item.time for item in self.queue),
                time=lambda: item.time,
            ),
        )
        self.world.event_logger.log_event(
            {
                "chan": self.name,
                "action": "synchronize",
                "device": me.name,
                "time": me.state.clock,
                "arrive_at": item.time,
                "id": item.id,
            }
        )
        self.world.xyield(f"chan {self.name} synchronize()")
        self.world.event_logger.log_event(
            {
                "chan": self.name,
                "action": "desynchronize",
                "device": me.name,
                "time": me.state.clock,
                "id": item.id,
            }
        )
        self.sync_points.append(item.data)
        self.queue.remove(item)

    def receive(self, me: Device) -> Any:
        self.world.add_dependency(
            me,
            Dependency(
                condition=lambda: len(self.queue) > 0,
                time=lambda: self.queue[0].time if self.queue else None,
            ),
        )
        self.world.xyield(f"chan {self.name} receive()")
        self.world.event_logger.log_event(
            {
                "chan": self.name,
                "action": "receive",
                "device": me.name,
                "time": me.state.clock,
                "arrive_at": self.queue[0].time,
                "id": self.queue[0].id,
            }
        )
        data = self.queue.pop(0)
        return data.data

    def send(self, clock: float, data: Any, me: Optional[Device] = None):
        with self._lock:
            # TODO: Send should direclty route to some device, otherwise it is impossible to calculate latency.
            if not me or not self.subscribers:
                latency = 0
            else:
                latency = self.world.latency_between(me, self.subscribers[0]) if me else 0

            item = ChanItem(data=deepcopy(data), time=clock + latency, id=str(uuid.uuid4()))
            self.queue.append(item)

            self.world.event_logger.log_event(
                {
                    "chan": self.name,
                    "action": "send",
                    "device": me.name if me else "user",
                    "time": clock,
                    "arrive_at": clock + latency,
                    "id": item.id,
                }
            )

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

        # TODO: evaluating listeners here might cause time skew
        for listener in self.listeners:
            listener(data)


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
