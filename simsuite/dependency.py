from typing import Callable, Optional


class Dependency:
    def __init__(self, condition: Callable[[], bool], time: Callable[[], Optional[float]]):
        self.condition = condition
        self.time = time
