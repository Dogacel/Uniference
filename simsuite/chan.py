from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING, Optional
from torch import Tensor
import torch.distributed as dist
from simsuite.common import dprint

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
    subscribers: list[Device]

    def __init__(self, name: str, world: World):
        self.name = name
        self.listeners = []
        self.subscribers = []
        self.world = world

    def subscribe(self, device: Device):
        self.subscribers.append(device)

    def unsubscribe(self, device: Device):
        self.subscribers.remove(device)

    def rank(self, device: Device) -> int:
        return self.subscribers.index(device) if device in self.subscribers else -1

    def size(self) -> int:
        return len(self.subscribers)

    def broadcast(self, me: Device, data: Any, id: str, source: int, force_send: bool = False) -> Any:
        """
        Broadcasts data to all subscribers, including self.
        """
        if source != self.rank(me) and not force_send:
            result = self.receive(me, f"broadcast_{id}_{me.name}")
            return result
        else:
            for device in self.subscribers:
                if device != me:
                    self.send(me, data, f"broadcast_{id}_{device.name}", target=device)
            return data

    def all_gather(self, me: Device, my_share: Any, id: str) -> list[Any]:
        """
        Implements a ring all-gather algorithm over the network.
        """
        my_order = self.rank(me)
        rounds = len(self.subscribers) - 1

        items = [None for _ in range(len(self.subscribers))]
        items[my_order] = my_share

        for i in range(rounds):
            sender_id = (my_order - i - 1) % len(self.subscribers)
            receiver_id = (my_order + i + 1) % len(self.subscribers)

            receiver = self.subscribers[receiver_id]
            sender = self.subscribers[sender_id]

            self.send(
                me,
                my_share,
                f"all_gather_{id}_{me.name}_{receiver.name}_{i}",
                target=self.subscribers[receiver_id],
            )
            new_share = self.receive(me, f"all_gather_{id}_{sender.name}_{me.name}_{i}")

            items[sender_id] = new_share.to(my_share.device)

        return items

    def all_gather_async(self, me: Device, my_share: Any, id: str):
        my_order = self.rank(me)
        rounds = len(self.subscribers) - 1

        def send():
            for i in range(rounds):
                receiver_id = (my_order + i + 1) % len(self.subscribers)
                receiver = self.subscribers[receiver_id]

                self.send(
                    me,
                    my_share,
                    f"all_gather_{id}_{me.name}_{receiver.name}_{i}",
                    target=self.subscribers[receiver_id],
                )

        def receive():
            items = [None for _ in range(len(self.subscribers))]
            items[my_order] = my_share

            for i in range(rounds):
                sender_id = (my_order - i - 1) % len(self.subscribers)
                sender = self.subscribers[sender_id]

                new_share = self.receive(me, f"all_gather_{id}_{sender.name}_{me.name}_{i}")
                items[sender_id] = new_share.to(my_share.device)

            return items

        return send, receive

    def all_reduce(
        self, me: Device, my_value: Any, id: str, reduce_fn: Callable[[Any, Any], Any] = lambda x, y: sum(x, y)
    ) -> Any:
        """
        Ring all-reduce over the last axis of `my_value`.
        Each rank starts by sending chunk (rank-1).
        After reduce-scatter, rank r holds reduced chunk r.
        """
        r = self.rank(me)
        P = len(self.subscribers)
        rounds = P - 1

        # ---- Split last axis into P contiguous chunks ----
        L = my_value.shape[-1]
        if L % P != 0:
            # handle uneven splits robustly
            chunks = np.array_split(my_value, P, axis=-1)
            # stack into a new "chunk axis" = -2 (just before last)
            partial = np.stack(chunks, axis=-2)  # shape: [..., P, chunk_len_r]
        else:
            chunk = L // P
            # reshape to expose a chunk axis of length P
            partial = my_value.reshape(*my_value.shape[:-1], P, chunk)  # shape: [..., P, chunk]

        left = (r - 1) % P  # neighbor we receive from
        right = (r + 1) % P  # neighbor we send to

        # ---- Reduce-scatter (start by sending chunk r-1) ----
        for i in range(rounds):
            send_idx = (r - 1 - i) % P  # r-1, r-2, ..., r-(P-1)
            recv_idx = (r - 2 - i) % P  # r-2, r-3, ..., r-P

            # Send our current chunk
            self.send(
                me,
                partial[..., send_idx, :],
                f"all_reduce_reduce_{id}_{me.name}_{self.subscribers[right].name}_{i}",
                target=self.subscribers[right],
            )

            # Receive neighbor's chunk and reduce into our recv slot
            incoming = self.receive(me, f"all_reduce_reduce_{id}_{self.subscribers[left].name}_{me.name}_{i}")
            partial[..., recv_idx, :] = reduce_fn(partial[..., recv_idx, :], incoming)

        # After RS, rank r owns reduced chunk r on the chunk axis
        my_share = partial[..., r, :]

        # ---- All-gather (circulate reduced chunks) ----
        gathered = self.all_gather(me, my_share, f"all_reduce_gather_{id}")
        # `gathered` should be ordered by rank 0..P-1; concatenate along last axis
        result = torch.cat(gathered, axis=-1)
        return result

    def send(self, me: Device, data: Any, transmit_id: str, target: Device, force_time: Optional[float] = None):
        # TODO: Currently single network is supported
        network = me.world.networks[0]

        if force_time is not None:
            time = force_time
        else:
            time = me.state.sync_clock()

        if isinstance(data, Tensor):
            size = data.numel() * data.element_size() * 8
        elif hasattr(data, "nbytes"):
            size = data.nbytes * 8
        else:
            size = 1
            dprint("Data size estimation not implemented for type " + str(type(data)))

        assert size > 0

        dprint(f"[{me.state.clock}] Chan {self.name} send() size={size} time={time} id={transmit_id}")
        network.transmit(data, size=size, world_time=time, id=transmit_id, source=me, target=target)
        self.world.xyield(me, f"chan {self.name} send()")

    def receive(self, me: Device, transmit_id: str) -> Any:
        # TODO: Currently single network is supported
        network = me.world.networks[0]
        me.state.sync_clock()
        me.state.dependency = network.search_transmit(transmit_id)

        dprint(f"[{me.state.clock}] Chan {self.name} receive() waiting for id={transmit_id}")
        self.world.xyield(me, f"chan {self.name} receive()")

        assert me.state.dependency is not None and not isinstance(me.state.dependency, str)

        result = me.state.dependency.data
        network.complete_transmit(me.state.dependency, me.state.sync_clock())
        me.state.dependency = None

        return result
