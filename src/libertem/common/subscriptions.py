import uuid
import time
from typing import Callable, Any, overload


class SubscriptionManager:
    def __init__(self):
        # Mapping of topic to {key: callback}
        self._subs: dict[str, dict[str, Callable[[str, dict], None]]] = {}

    @overload
    def subscribe(
        self, topic: tuple[str, ...], callback: Callable[[str, dict], None]
    ) -> tuple[str, ...]:
        ...

    def subscribe(self, topic: str, callback: Callable[[str, dict], None]) -> str:
        if isinstance(topic, tuple):
            return tuple(self.subscribe(t, callback) for t in topic)
        key = str(uuid.uuid4())
        try:
            self._subs[topic][key] = callback
        except KeyError:
            self._subs[topic] = {key: callback}
        return key

    def unsubscribe(self, key: str) -> bool:
        for registered in self._subs.values():
            try:
                _ = registered.pop(key)
                return True
            except KeyError:
                return False

    def send(self, topic: str, msg_dict: dict[str, Any]):
        if "timestamp" not in msg_dict:
            msg_dict["timestamp"] = time.time()
        for callback in self._subs.get(topic, {}).values():
            callback(topic, msg_dict)
