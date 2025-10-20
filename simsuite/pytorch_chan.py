from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING, Optional
from torch import Tensor
from torch import distributed as dist
from simsuite.common import dprint
from torch.distributed import (
    broadcast,
    all_gather,
    all_reduce,
    send as torch_send,
    recv as torch_recv,
    broadcast_object_list,
    send_object_list,
    recv_object_list,
)

if TYPE_CHECKING:
    from simsuite.world import World
    from simsuite.device import Device


@dataclass
class ChanItem:
    data: Any
    time: float
    id: str


class PytorchChan:
    name: str

    def __init__(self, name: str, world: World):
        self.name = name

    def subscribe(self, device: Device):
        pass

    def unsubscribe(self, device: Device):
        pass

    def rank(self, device: Device) -> int:
        return dist.get_rank()

    def size(self) -> int:
        return dist.get_world_size()

    def broadcast(self, me: Device, data: Any, id: str, source: int, force_send: bool = False) -> Any:
        if data == None:
            data = [None]

        broadcast_object_list(data, src=source)
        return data

    def all_gather(self, me: Device, my_share: Tensor, id: str) -> list[Tensor]:
        tensor_list = [torch.zeros_like(my_share) for i in range(torch.distributed.get_world_size())]
        all_gather(tensor_list, my_share)
        return tensor_list

    def all_gather_async(self, me: Device, my_share: Tensor, id: str):
        tensor_list = [torch.zeros_like(my_share) for _ in range(torch.distributed.get_world_size())]
        work = None

        def send():
            nonlocal work
            nonlocal tensor_list

            work = all_gather(tensor_list, my_share, async_op=True)

        def receive():
            nonlocal work
            nonlocal tensor_list

            work.wait()
            return tensor_list

        return send, receive

    def all_reduce(
        self, me: Device, my_value: Any, id: str, reduce_fn: Callable[[Any, Any], Any] = lambda x, y: sum(x, y)
    ) -> Any:
        all_reduce(my_value)
        return my_value

    def send(self, me: Device, data: Any, transmit_id: str, target: Device, force_time: Optional[float] = None):
        # TODO: Make sure target rank is set. Currently assuming dst=0
        print("Sending data via torch.send")
        send_object_list(data, dst=0)

    def receive(self, me: Device, transmit_id: str) -> Any:
        obj_list = []
        recv_object_list(obj_list, src=0)
        return obj_list
