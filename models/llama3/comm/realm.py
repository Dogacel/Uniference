
from typing import Optional
from dataclasses import dataclass
from typing import Callable
import time
from sympy.printing.pytorch import torch
from copy import deepcopy
from typing import Any
from torch import Tensor

import threading

class Device():
    tag: str

    def __init__(self, tag: str):
        self.tag = tag

class Chan():
    listeners: list[Callable[[Any], None]]
    
    def __init__(self):
        self.listeners = []

    def send(self, data):
        pass

    def receive(self) -> Any:
        pass

    def add_listener(self, listener: Callable[[Any], None]):
        self.listeners.append(listener)

    def start_worker(self):
        # Run a background worker that polls every second
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while True:
            data = self.receive()
            while data is not None:
                for listener in self.listeners:
                    listener(data)
                data = self.receive()
            time.sleep(0.05)

class Realm():
    def __init__(self):
        pass

    def me(self) -> Device:
        raise NotImplementedError("Subclasses should implement this method")

    def device(self, tag: str) -> Device:
        raise NotImplementedError("Subclasses should implement this method")

    def chan(self, tag: str) -> Chan:
        raise NotImplementedError("Subclasses should implement this method")

class InMemoryChan(Chan):
    queue: list[Any]
    _lock: threading.Lock
    total_transferred_bytes: int
    total_transferred_count: int

    def __init__(self):
        super().__init__()
        self.queue = []
        self._lock = threading.Lock()

        self.total_transferred_bytes = 0
        self.total_transferred_count = 0


    def receive(self) -> Optional[Any]:
        with self._lock:
            if not self.queue:
                return None
            data = self.queue.pop(0)
            return data

    def reset_counters(self):
        with self._lock:
            self.total_transferred_bytes = 0
            self.total_transferred_count = 0

    def send(self, data):
        with self._lock:
            self.queue.append(deepcopy(data))

            if isinstance(data, SyncKVCache):
                self.total_transferred_bytes += data.xk.numel() * data.xk.element_size() + data.xv.numel() * data.xv.element_size() + data.layer_id.bit_length() // 8  + data.start_pos.bit_length() // 8
                self.total_transferred_count += 1

            if isinstance(data, SyncGen):
                self.total_transferred_bytes += data.next_token.numel() * data.next_token.element_size() + data.pos.bit_length() // 8
                self.total_transferred_count += 1

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