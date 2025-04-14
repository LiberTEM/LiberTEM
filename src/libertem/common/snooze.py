import time
import weakref
from enum import Enum
import functools
import threading
import contextlib
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from libertem.common.executor import JobExecutor
    from libertem.common.subscriptions import SubscriptionManager


class SnoozeMessage(Enum):
    SNOOZE = "snooze"
    UNSNOOZE_START = "unsnooze_start"
    UNSNOOZE_DONE = "unsnooze_done"
    UPDATE_ACTIVITY = "update_activity"


class SnoozeManager:
    def __init__(
        self,
        *,
        up: Callable[[], None],
        down: Callable[[], None],
        timeout: float,  # seconds
        subscriptions: 'SubscriptionManager',
    ):
        if timeout <= 0:
            raise ValueError("Must supply a positive snooze timeout")
        self.scale_up = weakref.WeakMethod(up)
        self.scale_down = weakref.WeakMethod(down)
        self.subscriptions = weakref.ref(subscriptions)
        self.keep_alive = 0
        self.last_activity = time.monotonic()
        self.is_snoozing = False
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
        self.last_activity = time.monotonic()

    @contextlib.contextmanager
    def in_use(self):
        self._update_last_activity()
        self.keep_alive += 1
        try:
            yield
        finally:
            self.keep_alive -= 1
            self.keep_alive = max(0, self.keep_alive)
            self._update_last_activity()

    def snooze(self):
        if self.keep_alive > 0 or self._snooze_task is None:
            return
        with self._snooze_lock:
            scale_down = self.scale_down()
            if scale_down is not None:
                subs = self.subscriptions()
                if subs is not None:
                    subs.send(SnoozeMessage.SNOOZE, {})
                scale_down()
            self.is_snoozing = True

    def unsnooze(self):
        if not self.is_snoozing:
            return
        with self._snooze_lock:
            scale_up = self.scale_up()
            if scale_up is not None:
                subs = self.subscriptions()
                if subs is not None:
                    subs.send(SnoozeMessage.UNSNOOZE_START, {})
                scale_up()
                if subs is not None:
                    subs.send(SnoozeMessage.UNSNOOZE_DONE, {})
            self.is_snoozing = False

    def _snooze_check_task(self):
        """
        Periodically check if we need to snooze the executor
        """
        while True:
            time.sleep(self._snooze_check_interval)
            if self.scale_down() is None:
                break
            if self.is_snoozing or self.keep_alive > 0:
                continue
            since_last_activity = time.monotonic() - self.last_activity
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
