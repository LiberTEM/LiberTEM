import uuid
import time
from typing import Callable, Any, overload


class SubscriptionManager:
    """
    A simple topic -> callback dispatcher, used by an executor
    to send messages to subscribers about events.
    """
    def __init__(self):
        # Mapping of topic to {key: callback}
        self._subs: dict[str, dict[str, Callable[[str, dict], None]]] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(subs={tuple(self._subs.keys())})"

    @overload
    def subscribe(
        self, topic: tuple[str, ...], callback: Callable[[str, dict], None]
    ) -> tuple[str, ...]:
        ...

    def subscribe(self, topic: str, callback: Callable[[str, dict], None]) -> str:
        """
        Subscribe to one-or-more topics with the provided callback.
        Returns a string UUID used to unsubscribe from the topic.

        The callback will receive :code:`(topic_str, {'timestamp': float, ...})`
        for each event sent.

        The callback is run synchronously so must run efficiently.
        """
        if isinstance(topic, tuple):
            return tuple(self.subscribe(t, callback) for t in topic)
        key = str(uuid.uuid4())
        try:
            self._subs[topic][key] = callback
        except KeyError:
            self._subs[topic] = {key: callback}
        return key

    def unsubscribe(self, key: str) -> bool:
        """
        Unsubscribe from a topic registered under the key, return
        a bool indicating if the key was found and unregistered
        """
        for registered in self._subs.values():
            try:
                _ = registered.pop(key)
                return True
            except KeyError:
                continue
        return False

    def send(self, topic: str, msg_dict: dict[str, Any]):
        if "timestamp" not in msg_dict:
            msg_dict["timestamp"] = time.time()
        for callback in self._subs.get(topic, {}).values():
            callback(topic, msg_dict)
