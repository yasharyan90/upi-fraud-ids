"""Event bus abstraction (Section III-B): in-memory publish/subscribe by default,
Apache Kafka when a bootstrap server is configured (kafka-python, optional)."""
from __future__ import annotations

import json
import queue
from collections import defaultdict


class InMemoryBus:
    def __init__(self):
        self._topics: dict[str, queue.Queue] = defaultdict(queue.Queue)

    def publish(self, topic: str, event: dict) -> None:
        self._topics[topic].put(json.dumps(event, default=str))

    def consume(self, topic: str, timeout: float = 1.0):
        """Generator yielding events until the topic stays empty for `timeout` seconds."""
        q = self._topics[topic]
        while True:
            try:
                yield json.loads(q.get(timeout=timeout))
            except queue.Empty:
                return

    def size(self, topic: str) -> int:
        return self._topics[topic].qsize()


class KafkaBus:
    def __init__(self, bootstrap_servers: str, group_id: str = "upi-ids"):
        from kafka import KafkaConsumer, KafkaProducer  # optional dependency

        self._producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                                       value_serializer=lambda v: json.dumps(v, default=str).encode())
        self._bootstrap, self._group = bootstrap_servers, group_id
        self._KafkaConsumer = KafkaConsumer

    def publish(self, topic: str, event: dict) -> None:
        self._producer.send(topic, event)

    def consume(self, topic: str, timeout: float = 1.0):
        consumer = self._KafkaConsumer(topic, bootstrap_servers=self._bootstrap, group_id=self._group,
                                       value_deserializer=lambda v: json.loads(v.decode()),
                                       consumer_timeout_ms=int(timeout * 1000), auto_offset_reset="earliest")
        for msg in consumer:
            yield msg.value

    def size(self, topic: str) -> int:  # pragma: no cover
        return -1


def make_bus(bootstrap_servers: str | None = None):
    if bootstrap_servers:
        try:
            return KafkaBus(bootstrap_servers)
        except Exception:  # pragma: no cover
            pass
    return InMemoryBus()
