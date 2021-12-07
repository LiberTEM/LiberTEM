import time
from types import TracebackType
from typing import ContextManager, Optional, Type


class Timing(ContextManager):
    t0: float
    t1: float

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        self.t1 = time.perf_counter()

    @property
    def delta(self) -> float:
        return self.t1 - self.t0
