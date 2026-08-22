"""
In-memory message transport for consensus simulations.

Provides a simple point-to-point async transport so the Paxos / EPaxos /
Chain Replication modules can run multiple node instances in a single
process without real networking. Messages may optionally be dropped to
simulate packet loss / partitions for testing.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class Envelope:
    src: str
    dst: str
    msg: Any


class LocalTransport:
    """Registry-based transport that delivers messages to registered handlers."""

    def __init__(self, drop_rate: float = 0.0, clock: Callable[[], float] = None):
        self._handlers: Dict[str, Callable[[str, Any], Any]] = {}
        self.drop_rate = drop_rate
        self._lock = asyncio.Lock()
        # Optional injected clock, unused by default but kept for parity with
        # the Raft node's injectable-clock testing strategy.
        self._clock = clock

    def register(self, node_id: str, handler: Callable[[str, Any], Any]) -> None:
        self._handlers[node_id] = handler

    def unregister(self, node_id: str) -> None:
        self._handlers.pop(node_id, None)

    def node_ids(self) -> list:
        return list(self._handlers.keys())

    async def send(self, src: str, dst: str, msg: Any) -> None:
        handler = self._handlers.get(dst)
        if handler is None:
            return
        if self.drop_rate > 0 and random.random() < self.drop_rate:
            return  # simulate lost message
        if asyncio.iscoroutinefunction(handler):
            await handler(src, msg)
        else:
            handler(src, msg)

    async def broadcast(self, src: str, recipients: list, msg: Any) -> None:
        for dst in recipients:
            if dst != src:
                await self.send(src, dst, msg)


__all__ = ["LocalTransport", "Envelope"]
