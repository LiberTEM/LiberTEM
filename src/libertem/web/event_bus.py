import asyncio
import queue
import logging
import concurrent.futures

log = logging.getLogger(__name__)


class MessagePump:
    def __init__(self, event_bus, event_registry):
        self._event_bus = event_bus
        self._event_registry = event_registry

    async def run(self):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            loop = asyncio.get_running_loop()
            while True:
                try:
                    msg = await loop.run_in_executor(pool, self._event_bus.get)
                    log.debug("MessagePump.run: got msg: %s", msg)
                    await self._event_registry.broadcast_event(msg)
                except queue.Empty:
                    pass


class EventBus:
    """
    An internal event bus used in the web API for forwarding messages from any
    thread to all websocket connections.

    Do not use for latecy-sensitive messages.
    """

    def __init__(self):
        # unlimited size so we can put items without waiting
        self._queue = queue.Queue()

    def send(self, msg):
        log.debug("EventBus.send: got msg: %s", msg)
        self._queue.put(msg)

    def get(self, timeout=1.0):
        # We need to block here to not busy-wait. Only run in a sync context
        # where you can affort to block, or in a dedicated thread.
        return self._queue.get(block=True, timeout=timeout)
