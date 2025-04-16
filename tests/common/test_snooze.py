import time
import pytest
import threading
from typing import Callable

from libertem.common.snooze import SnoozeManager, keep_alive, SnoozeMessage
from libertem.common.subscriptions import SubscriptionManager


class MockSnoozeExecutor:
    def __init__(self, timeout: int, updown_timeout: float = 0):
        self._subscriptions = SubscriptionManager()
        self.snooze_manager = SnoozeManager(
            up=self.up,
            down=self.down,
            timeout=timeout,
            subscriptions=self._subscriptions,
        )
        self.num_up = 0
        self.num_down = 0
        self.updown_timeout = updown_timeout
        self.transitions_in_progress = 0

    def subscribe(self, topic: str, callback: Callable[[str, dict], None]) -> str:
        return self._subscriptions.subscribe(topic, callback)

    def unsubscribe(self, key: str) -> bool:
        return self._subscriptions.unsubscribe(key)

    @keep_alive
    def run_task(self, timeout=0.):
        assert self.snooze_manager.keep_alive > 0
        if timeout > 0.:
            time.sleep(timeout)
        return 42

    def up(self):
        assert self.transitions_in_progress == 0
        self.transitions_in_progress += 1
        if self.updown_timeout > 0.:
            time.sleep(self.updown_timeout)
        self.num_up += 1
        self.transitions_in_progress -= 1

    def down(self):
        assert self.transitions_in_progress == 0
        self.transitions_in_progress += 1
        assert self.snooze_manager.keep_alive == 0
        if self.updown_timeout > 0.:
            time.sleep(self.updown_timeout)
        self.num_down += 1
        self.transitions_in_progress -= 1


def test_timer():
    snooze_time = 0.2
    executor = MockSnoozeExecutor(snooze_time)
    assert not executor.snooze_manager.is_snoozing
    time.sleep(snooze_time * 2)
    assert executor.snooze_manager.is_snoozing
    assert executor.num_down == 1
    assert executor.num_up == 0
    executor.snooze_manager.unsnooze()
    assert not executor.snooze_manager.is_snoozing
    assert executor.num_down == 1
    assert executor.num_up == 1


def test_keep_alive():
    snooze_time = 0.2
    executor = MockSnoozeExecutor(snooze_time)
    assert not executor.snooze_manager.is_snoozing
    assert executor.snooze_manager.keep_alive == 0
    executor.run_task(timeout=snooze_time * 2)
    assert not executor.snooze_manager.is_snoozing
    assert executor.snooze_manager.keep_alive == 0
    assert executor.num_down == 0
    assert executor.num_up == 0


def test_job_prevents_snooze():
    snooze_time = 0.2
    executor = MockSnoozeExecutor(snooze_time)
    th = threading.Thread(target=executor.run_task, kwargs={'timeout': snooze_time * 2})
    th.start()
    assert not executor.snooze_manager.is_snoozing
    assert executor.snooze_manager.keep_alive == 1
    executor.snooze_manager.snooze()
    assert not executor.snooze_manager.is_snoozing
    th.join()


def test_lock_prevents_transitions():
    executor = MockSnoozeExecutor(10_000., updown_timeout=0.1)
    assert not executor.snooze_manager.is_snoozing
    with executor.snooze_manager._snooze_lock:
        # already holding the lock so new thread will wait
        th = threading.Thread(target=executor.snooze_manager.snooze)
        th.start()
        time.sleep(0.05)
        assert executor.num_down == 0
        assert executor.num_up == 0

    th.join()
    assert executor.num_down == 1
    assert executor.num_up == 0

    executor.snooze_manager.unsnooze()
    assert not executor.snooze_manager.is_snoozing
    assert executor.num_down == 1
    assert executor.num_up == 1

    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=executor.snooze_manager.snooze))
        threads[-1].start()
    _ = tuple(th.join() for th in threads)
    assert executor.snooze_manager.is_snoozing
    # extra calls should short-circuit if already snoozing
    assert executor.num_down == 2
    assert executor.num_up == 1

    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=executor.snooze_manager.unsnooze))
        threads[-1].start()
    _ = tuple(th.join() for th in threads)
    assert not executor.snooze_manager.is_snoozing
    # extra calls should short-circuit if already snoozing
    assert executor.num_down == 2
    assert executor.num_up == 2


def test_bad_snooze_timeouts():
    with pytest.raises(ValueError):
        MockSnoozeExecutor(0.)
    with pytest.raises(ValueError):
        MockSnoozeExecutor(-10.)


def test_messages():
    messages_received = []

    def got_message(topic, msg_dict):
        messages_received.append((topic, msg_dict))

    executor = MockSnoozeExecutor(0.1)
    executor.subscribe(SnoozeMessage.SNOOZE, got_message)
    executor.subscribe(SnoozeMessage.UNSNOOZE_START, got_message)
    executor.subscribe(SnoozeMessage.UNSNOOZE_DONE, got_message)
    activity_key = executor.subscribe(SnoozeMessage.UPDATE_ACTIVITY, got_message)
    task_time = 0.1
    executor.run_task(task_time)
    time.sleep(0.15)

    assert messages_received[0][0] == SnoozeMessage.UPDATE_ACTIVITY
    assert executor.snooze_manager.is_snoozing
    assert messages_received[-1][0] == SnoozeMessage.SNOOZE
    task_start = messages_received[0][1]['timestamp']
    task_end = messages_received[-1][1]['timestamp']
    assert (task_end - task_start) >= 0.99 * task_time  # account for jitter

    assert executor.unsubscribe(activity_key)
    executor.snooze_manager.unsnooze()
    messages_received.clear()
    time.sleep(0.15)
    executor.run_task()
    topics = tuple(m[0] for m in messages_received)
    assert SnoozeMessage.UPDATE_ACTIVITY not in topics
    assert topics[-1] == SnoozeMessage.UNSNOOZE_DONE
