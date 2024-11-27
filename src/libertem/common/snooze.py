import time
import weakref
from enum import Enum, auto
import functools
import threading
import contextlib
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from libertem.common.executor import JobExecutor


class SnoozeMessage(Enum):
    SNOOZE = auto()
    UNSNOOZE_START = auto()
    UNSNOOZE_DONE = auto()
    UPDATE_ACTIVITY = auto()


class SnoozeManager:
    def __init__(
        self,
        *,
        up: Callable[[], None],
        down: Callable[[], None],
        timeout: float,  # seconds
    ):
        if timeout <= 0:
            raise ValueError("Must supply a positive snooze timeout")
        self.scale_up = weakref.WeakMethod(up)
        self.scale_down = weakref.WeakMethod(down)
        self._keep_alive = 0
        self._last_activity = time.monotonic()
        self._is_snoozing = False
        self._snooze_lock = threading.Lock()
        self._snooze_timeout = timeout
        self._snooze_check_interval = min(
            30.0,
            self._snooze_timeout and (self._snooze_timeout * 0.1) or 30.0,
        )
        self._snooze_task = threading.Thread(
            target=self._snooze_check_task,
            daemon=True,
        )
        self._snooze_task.start()

    def _update_last_activity(self):
        self._last_activity = time.monotonic()

    @contextlib.contextmanager
    def in_use(self):
        self._update_last_activity()
        self._keep_alive += 1
        try:
            yield
        finally:
            self._keep_alive -= 1
            self._keep_alive = max(0, self._keep_alive)
            self._update_last_activity()

    def snooze(self):
        if self._keep_alive > 0 or self._snooze_task is None:
            return
        with self._snooze_lock:
            scale_down = self.scale_down()
            if scale_down is not None:
                scale_down()
            self._is_snoozing = True

    def unsnooze(self):
        if not self._is_snoozing:
            return
        with self._snooze_lock:
            scale_up = self.scale_up()
            if scale_up is not None:
                scale_up()
            self._is_snoozing = False

    def _snooze_check_task(self):
        """
        Periodically check if we need to snooze the executor
        """
        while True:
            time.sleep(self._snooze_check_interval)
            if self.scale_down() is None:
                break
            if self._is_snoozing or self._keep_alive > 0:
                continue
            since_last_activity = time.monotonic() - self._last_activity
            if since_last_activity > self._snooze_timeout:
                self.snooze()


def keep_alive(fn):

    @functools.wraps(fn)
    def wrapped(self: 'JobExecutor', *args, **kwargs):
        manager = self.snooze_manager
        if manager is not None:
            manager.unsnooze()
            with manager.in_use():
                return fn(self, *args, **kwargs)
        else:
            return fn(self, *args, **kwargs)

    return wrapped
