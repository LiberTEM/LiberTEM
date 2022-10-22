import threading
from typing import TYPE_CHECKING, Iterable, Dict, Callable, List, Optional, Tuple
import time
from collections import deque
from libertem.common.executor import WorkerQueueEmpty


if TYPE_CHECKING:
    import numpy as np
    from libertem.common.executor import WorkerQueue, TaskCommHandler
    from libertem.udf.base import UDFTask
    from libertem.io.dataset.base.tiling import DataTile
    from libertem.io.dataset.base.partition import Partition


class CommsDispatcher:
    """
    Monitors a :code:`WorkerQueue` in a background thread
    and launches callbacks in response to messages recieved
    Callbacks are registered as a dictionary of subscriptions
        {topic_name: [callback, ...]]}
    and are called in order they were recieved in the same thread
    that is doing the monitoring. The callbacks should be lightweight
    in order to not build up too many messages in the queue

    The functionality of this class mirrors Dask's structured logs
    feature, which has a similar message => topic => callback
    model running in the client event loop
    """
    def __init__(self, queue: 'WorkerQueue', subscriptions: Dict[str, List[Callable]]):
        self._message_q = queue
        self._subscriptions = subscriptions
        self._thread = None

    def __enter__(self, *args, **kwargs):
        if self._thread is not None:
            raise RuntimeError('Cannot re-enter CommsDispatcher')
        self._thread = threading.Thread(target=self.monitor_queue)
        self._thread.start()

    def __exit__(self, *args, **kwargs):
        if self._thread is None:
            return
        self._message_q.put(('STOP', {}))
        self._thread.join()
        self._thread = None
        # Drain queue just in case
        while True:
            try:
                with self._message_q.get(block=False) as _:
                    ...
            except WorkerQueueEmpty:
                break

    def monitor_queue(self):
        """
        Monitor the queue for messages
        If there are no subscribers this should drain
        messages from the queue as fast as they are recieved
        """
        while True:
            with self._message_q.get(block=True) as ((topic, msg), _):
                if topic == 'STOP':
                    break
                else:
                    try:
                        for callback in self._subscriptions[topic]:
                            callback(topic, msg)
                    except KeyError:
                        pass


class ProgressManager:
    def __init__(self, tasks: Iterable['UDFTask']):
        self._task_max = {t.partition.get_ident(): t.task_frames
                          for t in tasks}
        self._counters = {k: 0 for k in self._task_max.keys()}
        total_frames = sum(self._task_max.values())
        # For converting tiles to pseudo-frames
        self._sig_size = tasks[0].partition.shape.sig.size
        # Counters for part display
        self._num_complete = 0
        self._num_in_progress = 0
        self._num_total = len(self._counters)
        # Create the bar object
        self._bar = self.make_bar(total_frames)

    def make_bar(self, maxval):
        from tqdm.auto import tqdm
        return tqdm(desc=self.get_description(),
                    total=maxval,
                    leave=True)

    def get_description(self):
        if self._num_in_progress:
            return (f'Partitions {self._num_complete}({self._num_in_progress})'
                    f'/{self._num_total}, Frames')
        else:
            return (f'Partitions {self._num_complete}'
                    f'/{self._num_total}, Frames')

    def update_description(self):
        self._bar.set_description(self.get_description())

    def close(self):
        self._bar.close()

    def connect(self, comms: 'TaskCommHandler'):
        comms.subscribe('partition_start', self.handle_start_task)
        comms.subscribe('partition_complete', self.handle_end_task)
        comms.subscribe('tile_complete', self.handle_tile_update)

    def handle_start_task(self, topic, message):
        if topic != 'partition_start':
            raise RuntimeError('Unrecognized topic')
        self._num_in_progress += 1
        self.update_description()

    def handle_end_task(self, topic, message):
        if topic != 'partition_complete':
            raise RuntimeError('Unrecognized topic')
        t_id = message['ident']
        remain = self._task_max[t_id] - int(self._counters[t_id])
        if remain:
            self._bar.update(remain)
            self._counters[t_id] = self._task_max[t_id]
        self._num_complete += 1
        self._num_in_progress -= 1
        self.update_description()

    def handle_tile_update(self, topic, message):
        if topic != 'tile_complete':
            raise RuntimeError('Unrecognized topic')
        t_id = message['ident']
        elements = message['elements']
        pframes = elements / self._sig_size
        if int(pframes):
            self._bar.update(int(pframes))
        self._counters[t_id] += pframes


class PartitionProgressTracker:
    def __init__(
                self,
                partition: 'Partition',
                roi: Optional['np.ndarray'],
                history_length: int = 5,
                threshold_part_time: float = 2.,
                min_message_interval: float = 1.,
            ):
        self._ident = partition.get_ident()
        try:
            self._worker_context = partition._worker_context
        except AttributeError:
            self._worker_context = None

        self._elements_complete = 0
        self._last_message = time.time()
        self._min_message_interval = min_message_interval
        # Size of data in partition (accounting for ROI)
        nel = partition.get_frame_count(roi) * partition.meta.shape.sig.size
        self._threshold_rate = nel / threshold_part_time
        self._history = deque(tuple(), history_length)

    def signal_start(self):
        if self._worker_context is None:
            return
        self._worker_context.signal(
            self._ident,
            'partition_start',
            {},
        )

    def signal_tile_complete(self, tile: 'DataTile'):
        if self._worker_context is None:
            return

        send, elements = self.should_send_progress(tile.size)
        if send:
            self._worker_context.signal(
                self._ident,
                'tile_complete',
                {'elements': elements},
            )

    def signal_complete(self):
        if self._worker_context is None:
            return
        self._worker_context.signal(
            self._ident,
            'partition_complete',
            {},
        )

    def should_send_progress(self, elements: int) -> Tuple[bool, int]:
        current_t = time.time()
        current_len = len(self._history)

        if current_len:
            previous_t = self._history.pop()
            this_rate = elements / (current_t - previous_t)
            self._history.append(this_rate)
            avg_rate = sum(self._history) / current_len

        self._history.append(current_t)
        self._elements_complete += elements

        if current_len:
            part_is_slow = avg_rate < self._threshold_rate
            time_since_last_m = current_t - self._last_message
            not_rate_limited = time_since_last_m > self._min_message_interval

            if part_is_slow and not_rate_limited:
                elements = self._elements_complete
                self._elements_complete = 0
                self._last_message = current_t
                return True, elements

        return False, 0
