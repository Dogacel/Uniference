from typing import Callable, Optional


class Dependency:
    node_ids: list[str]

    def __init__(self, condition: Callable[[], bool], time: Callable[[], Optional[float]]):
        self.condition = condition
        self.time = time


class DependencyTree:
    def __init__(self):
        self.dependencies = {}

        self.nodes = []

    def add(self, self_id: str, target_id: str):
        """
        Creates a dependency between self and target.

        I.e. self is a device and runs "all-gather" with round "i", this unique operation
        creates a dependency between "device" and "all-gather-i". However, "all-gather-i"
        also has dependencies on all devices that are participating in the all-gather.

        (device-1) --> (complete-all-gather-1) --> (send-all-gather-1-device-1) --> (network)
                                |
                                |
                                +----------------> (send-all-gather-1-device-2) --> (network)
                                |
                                |
                                +----------------> (send-all-gather-1-device-3) --> (network)
        """

        if self_id not in self.dependencies:
            self.dependencies[self_id] = []

        self.dependencies[self_id].append(target_id)

        return

    def complete_dependency(self, self_id: str, target_id: str):
        """
        Marks a dependency as complete.
        """
        if self_id in self.dependencies:
            self.dependencies[self_id].remove(target_id)
            if len(self.dependencies[self_id]) == 0:
                del self.dependencies[self_id]
