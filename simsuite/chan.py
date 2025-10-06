from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING, Optional
from torch import Tensor
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

    def broadcast(self, me: Device, data: Any, id: str, force_send: bool = False) -> Any:
        """
        Broadcasts data to all subscribers, including self.
        """
        if data is None and not force_send:
            result = self.receive(me, f"broadcast_{id}_{me.name}")
            return result
        else:
            for device in self.subscribers:
                if device != me:
                    self.send(me, data, f"broadcast_{id}_{device.name}")
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

            self.send(me, my_share, f"all_gather_{id}_{me.name}_{receiver.name}_{i}")
            new_share = self.receive(me, f"all_gather_{id}_{sender.name}_{me.name}_{i}")

            items[sender_id] = new_share

        return items

    def send(self, me: Device, data: Any, transmit_id: str, force_time: Optional[float] = None):
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
            print("Data size estimation not implemented for type " + str(type(data)))

        assert size > 0

        dprint(f"[{me.state.clock}] Chan {self.name} send() size={size} time={time} id={transmit_id}")
        network.transmit(data, size=size, world_time=time, id=transmit_id)
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
        network.complete_transmit(me.state.dependency)
        me.state.dependency = None

        return result
