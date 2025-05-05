import time
import weakref
from enum import Enum
import functools
import threading
import contextlib
from typing import Callable, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from libertem.common.executor import JobExecutor
    from libertem.common.subscriptions import SubscriptionManager


class SnoozeMessage(Enum):
    SNOOZE = "snooze"
    UNSNOOZE_START = "unsnooze_start"
    UNSNOOZE_DONE = "unsnooze_done"
    UPDATE_ACTIVITY = "update_activity"


class SnoozeManager:
    '''
    Monitor an executor for activity, and call its scale down
    method to free resources if inactive for longer than a
    specified timeout

    Executors can decorate methods with the :function:`~libertem.common.snooze.keep_alive`
    decorator in order to prevent snoozing during execution,
    and to trigger automatic unsnoozing when these methods are called.

    Parameters
    ----------

    up : Callable[[], None]
        Method to call to scale up the executor
    down : Callable[[], None]
        Method to call to scale down the executor
    timeout : float
        The inactivity period before triggering snoozing (seconds)
    subscriptions : Optional[SubscriptionManager]
        An instance of SubscriptionManager used to notify
        when snooze / unsnooze events are happening.
    '''
    def __init__(
        self,
        *,
        up: Callable[[], None],
        down: Callable[[], None],
        timeout: float,  # seconds
        subscriptions: Optional['SubscriptionManager'] = None,
    ):
        if timeout <= 0:
            raise ValueError("Must supply a positive snooze timeout")
        self.scale_up = weakref.WeakMethod(up)
        self.scale_down = weakref.WeakMethod(down)
        if subscriptions is not None:
            self.subscriptions = weakref.ref(subscriptions)
        else:
            self.subscriptions = lambda: None
        self.keep_alive = 0
        self.last_activity: float
        self._update_last_activity()
        self.is_snoozing = False
        self._snooze_lock = threading.Lock()
        self._snooze_timeout = timeout
        self._snooze_check_interval = min(
            30.0,
            self._snooze_timeout and (self._snooze_timeout * 0.1) or 30.0,
        )
        # sentinel value, used during shutdown to stop the snooze task
        self._snooze_task_continue = threading.Event()
        self._snooze_task_continue.set()
        self._snooze_task = threading.Thread(
            target=self._snooze_check_task,
            daemon=True,
        )
        self._snooze_task.start()

    def _update_last_activity(self):
        self.last_activity = time.monotonic()
        subs = self.subscriptions()
        if subs is not None:
            subs.send(SnoozeMessage.UPDATE_ACTIVITY, {})

    @contextlib.contextmanager
    def in_use(self):
        self._update_last_activity()
        self.keep_alive += 1
        try:
            yield
        finally:
            self.keep_alive = max(0, self.keep_alive - 1)
            self._update_last_activity()

    def snooze(self):
        if self.keep_alive > 0:
            # always no-op if something is using the executor
            return
        with self._snooze_lock:
            if self.is_snoozing:
                # Wait for lock to check this as there could be a
                # slow unsnooze action in progress which would leave things
                # in a bad state if we no-op too early
                return
            scale_down = self.scale_down()
            if scale_down is None:
                # must be destroying the executor, do nothing
                return
            subs = self.subscriptions()
            if subs is not None:
                subs.send(SnoozeMessage.SNOOZE, {})
            # Set the is_snoozing flag early, in case of a long call to scale_down()
            self.is_snoozing = True
            scale_down()

    def unsnooze(self):
        with self.in_use():
            with self._snooze_lock:
                if not self.is_snoozing:
                    # Wait for lock to check this as there could be a
                    # slow snooze action in progress which would leave things
                    # in a bad state if we no-op too early
                    return
                scale_up = self.scale_up()
                if scale_up is None:
                    # must be destroying the executor, do nothing
                    return
                subs = self.subscriptions()
                if subs is not None:
                    subs.send(SnoozeMessage.UNSNOOZE_START, {})
                scale_up()
                # Unset the is_snoozing flag only *after* we complete scale_up()
                self.is_snoozing = False
                if subs is not None:
                    subs.send(SnoozeMessage.UNSNOOZE_DONE, {})

    def close(self):
        if self._snooze_task is not None:
            self._snooze_task_continue.clear()

    def _snooze_check_task(self):
        """
        Periodically check if we need to snooze the executor
        """
        while self._snooze_task_continue.is_set():
            time.sleep(self._snooze_check_interval)
            if self.scale_down() is None:
                # Executor is likely being torn down
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


def keep_alive_context(fn):

    @contextlib.contextmanager
    def wrapped(self: 'JobExecutor', *args, **kwargs):
        manager = self.snooze_manager
        if manager is not None:
            manager.unsnooze()
            with manager.in_use():
                with fn(self, *args, **kwargs) as mgr:
                    yield mgr
        else:
            with fn(self, *args, **kwargs) as mgr:
                yield mgr

    return wrapped
