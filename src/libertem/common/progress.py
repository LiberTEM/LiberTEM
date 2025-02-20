import threading
from typing import TYPE_CHECKING, Callable, Any, NamedTuple, Optional
from collections.abc import Iterable
import time

from libertem.common.executor import WorkerQueueEmpty


if TYPE_CHECKING:
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
    def __init__(self, queue: 'WorkerQueue', subscriptions: dict[str, list[Callable]]):
        self._message_q = queue
        self._subscriptions = subscriptions
        self._thread = None

    def __enter__(self, *args, **kwargs):
        if self._thread is not None:
            raise RuntimeError('Cannot re-enter CommsDispatcher')
        self._thread = threading.Thread(
            target=self.monitor_queue,
            name="CommsDispatcher",
        )
        self._thread.daemon = True
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
        Monitor the queue for messages. This runs as a background thread of the
        main process, and forwards messages from the message queue to all
        subscribers via the registered callback functions.

        If there are no subscribers this should drain
        messages from the queue as fast as they are recieved
        """
        while True:
            with self._message_q.get(block=True) as ((topic, msg), _):
                if topic == 'STOP':
                    break
                try:
                    for callback in self._subscriptions[topic]:
                        callback(topic, msg)
                except KeyError:
                    pass


class ProgressState(NamedTuple):
    """
    Container for progress state, used to communicate
    from ProgressManager to ProgressReporter
    """
    #: float: Number of frames processed
    num_frames_complete: float
    #: int: Total number of frames to process
    num_frames_total: int
    #: int: Number of partitions completed
    num_part_complete: int
    #: int: Number of partitions in-progress
    num_part_in_progress: int
    #: int: Total number of partitions
    num_part_total: int
    #: str: A unique string identifier for the job associated
    #: with this progress message
    progress_id: str


class ProgressReporter:
    """
    Interface for progress bar display / updating

    This class will receive :class:`ProgressState`
    instances to notify it about the start, progression
    and end of a job submitted to the :code:`UDFRunner`.
    The implementation should be adapted to display or
    log the progress as required for the use case.

    It is possible that multiple jobs are submitted
    to a single executor at the same time and therefore
    the implementation should ensure that concurrent instances
    of the class display correctly, or that the same instance
    of the class can handle updates from multiple threads
    concurrently. Each :class:`ProgressState` message contains a field
    :code:`progress_id` which is unique to each job, and therefore
    the implementation can use this to distinguish updates from
    multiple sources.
    """
    def __init__(self):
        raise NotImplementedError()

    def start(self, state: ProgressState):
        """
        Signal the creation of a new job with the expected
        number of partitions and frames, and unique progress_id string.
        """
        raise NotImplementedError()

    def update(self, state: ProgressState):
        """
        Signal an intermediate update to the progress of a job
        """
        raise NotImplementedError()

    def end(self, state: ProgressState):
        """
        Signal the end of a given job

        This method will always be called and any updates
        recieved after this message should be ignored.
        """
        raise NotImplementedError()


class TQDMProgressReporter(ProgressReporter):
    """
    Progress bar display via tqdm

    Supports concurrent usage of multiple instances of
    this class, but does not handle multi-threaded use
    of the same instance to report multiple jobs.
    """
    def __init__(self):
        self._bar = None
        # Integers used to check if bar description should be changed
        # integers because faster than strcmp, updated in _should_update_description
        self._desc_key = (-1, -1, -1)

    def start(self, state: ProgressState):
        from tqdm.auto import tqdm
        self._bar = tqdm(desc=self._get_description(state),
                         total=state.num_frames_total,
                         leave=True)

    def update(self, state: ProgressState):
        return self._update(state, clip=True, refresh=False)

    def _update(self, state: ProgressState, *, clip: bool, refresh: bool):
        if state.num_frames_total != self._bar.total:
            # Should never happen but handle just in case
            self._bar.total = state.num_frames_total
            self._bar.refresh()
        increment = self._get_increment(state, clip=clip)
        if increment > 0:
            self._bar.update(increment)
        if self._should_update_description(state):
            self._bar.set_description(self._get_description(state))
        if refresh:
            self._bar.refresh()

    def end(self, state: ProgressState):
        self._update(state, clip=True, refresh=True)
        self._bar.close()

    def _should_update_description(self, state: ProgressState) -> bool:
        """
        Check the state to see if the elements used by self._get_description
        have changed, and if so update our record (self._desc_key) and return True
        """
        new_desc_key = (
            state.num_part_complete,
            state.num_part_in_progress,
            state.num_part_total
        )
        should_update = new_desc_key != self._desc_key
        if should_update:
            self._desc_key = new_desc_key
        return should_update

    @staticmethod
    def _get_description(state: ProgressState) -> str:
        """
        Get the most recent description string for the
        bar, including partition information
        If we know that partitions are in progress
        include this in parentheses after n_completed
        """

        if state.num_part_in_progress:
            return (f'Partitions {state.num_part_complete}({state.num_part_in_progress})'
                    f'/{state.num_part_total}, Frames')
        else:
            return (f'Partitions {state.num_part_complete}'
                    f'/{state.num_part_total}, Frames')

    def _get_increment(self, state: ProgressState, clip: bool = True):
        """
        Get the increment to apply to the progress bar based on current state
        and the state of the bar itself (bar.n is the total as-tracked by tqdm)

        Assumes self._bar.total has first been updated
        to state.num_frames_total if this is necessary
        """
        increment = int(state.num_frames_complete) - self._bar.n
        if clip:
            max_update = self._bar.total - self._bar.n
        else:
            max_update = increment + 1
        return max(0, min(increment, max_update))


class ProgressManager:
    """
    Handle updating of a progress reporter for a set of :code:`UDFTasks`, to be
    completed in any order. By default constructs a
    :code:`TQDMProgressReporter`, if no instance is passed in.

    The bar displays as such:

        Partitions: n_complete(n_in_progress) / n_total, ...\
            Frames: [XXXXX..] frames_completed / total_frames ...

    When processing tile stacks, stacks are treated as frames
    as such: (pseudo_frames = tile.size // sig_size)

    The bar will render in a Jupyter notebook as a JS widget
    automatically via tqdm.auto
    """
    def __init__(
        self,
        tasks: Iterable['UDFTask'],
        progress_id: str,
        reporter: Optional[ProgressReporter] = None,
    ):
        if not tasks:
            raise ValueError('Cannot display progress for empty tasks')
        self._progress_id = progress_id
        # the number of whole frames we expect each task to process
        self._task_max = {t.partition.get_ident(): t.task_frames
                          for t in tasks}
        # _counters is our record of progress on a task,
        # values are floating whole frames processed
        # as in tile mode we can process part of a frame
        self._counters = {k: 0. for k in self._task_max.keys()}
        self._total_frames = sum(self._task_max.values())
        # For converting tiles to pseudo-frames
        self._sig_size = tasks[0].partition.shape.sig.size
        # Counters for part display
        self._complete = set()
        self._in_progress = set()
        self._num_total = len(self._counters)
        if reporter is None:
            reporter = TQDMProgressReporter()
        elif not isinstance(reporter, ProgressReporter):
            # If not a ProgressReporter instance,
            # instantiate as if it has a bare __init__
            # Useful to be able to inject an instance
            # of ProgressReporter in case we need to setup
            # or access the reporter somehow (e.g. for testing)
            reporter = reporter()
        assert isinstance(reporter, ProgressReporter)
        self.reporter = reporter
        reporter.start(self.state)

    @property
    def state(self) -> ProgressState:
        return ProgressState(
            sum(self._counters.values()),
            self._total_frames,
            len(self._complete),
            len(self._in_progress),
            self._num_total,
            self._progress_id,
        )

    def finalize_task(self, task: 'UDFTask'):
        """
        When a task completes and we recieve its results on
        the main node, this is called to update the partition
        progress counters and frame counter in case we didn't
        recieve a complete history of the partition yet
        """
        topic = 'partition_complete'
        ident = task.partition.get_ident()
        message = {'ident': task.partition.get_ident()}
        if ident in self._task_max:
            self.handle_end_task(topic, message)

    def close(self):
        self.reporter.end(self.state)

    def connect(self, comms: 'TaskCommHandler'):
        """
        Register the callbacks on this class with the TaskCommHandler
        which will be dispatching messages recieved from the tasks
        """
        comms.subscribe('partition_start', self.handle_start_task)
        comms.subscribe('partition_complete', self.handle_end_task)
        comms.subscribe('tile_complete', self.handle_tile_update)

    def handle_start_task(self, topic: str, message: dict[str, Any]):
        """
        Increment the num_in_progress counter

        # NOTE An extension to this would be to track
        the identities of partitions in progress / completed
        for a richer display / more accurate accounting
        """
        if topic != 'partition_start':
            raise RuntimeError('Unrecognized topic')
        t_id = message['ident']
        if t_id not in self._complete:
            # if not complete handles case task was completed
            # before we can process its start message,
            # completion is signalled in the main thread
            # while start messages are processed in the background
            self._in_progress.add(t_id)
        self.reporter.update(self.state)

    def handle_end_task(self, topic: str, message: dict[str, Any]):
        """
        Increment the counter for the task to the max value
        and update the various counters / description
        """
        if topic != 'partition_complete':
            raise RuntimeError('Unrecognized topic')
        t_id = message['ident']
        remain = self._task_max[t_id] - int(self._counters[t_id])
        if remain:
            self._counters[t_id] = self._task_max[t_id]
        self._in_progress.discard(t_id)
        self._complete.add(t_id)
        self.reporter.update(self.state)

    def handle_tile_update(self, topic: str, message: dict[str, Any]):
        """
        Update the frame progress counter for the task
        and push the increment to the progress reporter

        Tile stacks are converted to pseudo-frames via the sig_size
        """
        if topic != 'tile_complete':
            raise RuntimeError('Unrecognized topic')
        t_id = message['ident']
        if self._counters[t_id] >= self._task_max[t_id]:
            return
        elements = message['elements']
        pframes = elements / self._sig_size
        self._counters[t_id] += pframes
        self.reporter.update(self.state)


class PartitionTrackerNoOp:
    """
    A no-op class matching the PartitionProgressTracker interface
    Used when progress == False to avoid any additional overhead
    """
    def signal_start(self, *args, **kwargs):
        ...

    def signal_tile_complete(self, *args, **kwargs):
        ...

    def signal_complete(self, *args, **kwargs):
        ...


def get_time():
    # Exists for testing / mocking
    return time.time()


class PartitionProgressTracker(PartitionTrackerNoOp):
    """
    Tracks the tile processing speed of a Partition and
    dispatches messages via the worker_context.signal() method
    under certain conditions

    Parameters
    ----------
    partition : Partition
        The partition to track progress for
    min_message_interval : float, optional
        The minumum time between messages, by default 1 second.
    """
    def __init__(
        self,
        partition: 'Partition',
        min_message_interval: float = 1.,
    ):
        self._ident = partition.get_ident()
        try:
            self._worker_context = partition._worker_context
        except AttributeError:
            self._worker_context = None

        # Counters to track / rate-limit messages
        self._elements_complete = 0
        self._last_message_t = None
        self._min_message_interval = min_message_interval

    def signal_start(self):
        """
        Signal that the partition has begun processing
        """
        if self._worker_context is None:
            return
        self._worker_context.signal(
            self._ident,
            'partition_start',
            {},
        )

    def signal_tile_complete(self, tile: 'DataTile'):
        """
        Register that tile.size more elements have been processed
        and if certain condition are met, send a signal
        """
        if self._worker_context is None:
            return

        send_elements = self.should_send_progress(tile.size)
        if send_elements:
            self._worker_context.signal(
                self._ident,
                'tile_complete',
                {'elements': send_elements},
            )

    def signal_complete(self):
        """
        Signal that the partition has completed processing

        This is not currently called as partition completion
        is registered on the main node as a fallback
        """
        if self._worker_context is None:
            return
        self._worker_context.signal(
            self._ident,
            'partition_complete',
            {},
        )

    def should_send_progress(self, elements: int) -> int:
        """
        Given the number elements of data that have been processed since
        the last message was sent, decide if a signal should be sent to the
        main node about the partition progress
        """
        current_t = get_time()
        self._elements_complete += elements

        if self._last_message_t is None:
            # Never send a message for the first tile stack
            # as this might have warmup overheads associated
            # Include the first elements in the history, however,
            # to give a better accounting. The first tile stack
            # is essentially treated as 'free'.
            self._last_message_t = current_t
            return 0

        time_since_last_m = current_t - self._last_message_t
        not_rate_limited = time_since_last_m > self._min_message_interval

        if not_rate_limited:
            completed_elements = self._elements_complete
            self._elements_complete = 0
            self._last_message_t = current_t
            return completed_elements

        return 0
