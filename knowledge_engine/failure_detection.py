"""
Failure detection and recovery built on the existing Raft node.

HealthChecker polls the Raft node's failure-detection state (heartbeats,
``tick``) and emits notifications / triggers recovery when peers are
suspected or down. RecoveryManager drives a crashed node back to a healthy
state: reload persisted state, fetch a snapshot/log sync from the leader,
then rejoin the cluster membership.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from knowledge_engine.distributed_coordination import (
    NodeState,
    RaftNode,
    MembershipChangeError,
)

logger = logging.getLogger(__name__)


@dataclass
class FailureEvent:
    node_id: str
    status: str  # "suspected" | "down" | "recovered"
    timestamp: float


class HealthChecker:
    """Periodically scans peer liveness and reports transitions."""

    def __init__(
        self,
        node: RaftNode,
        monitor_interval: float = 0.25,
        suspicion_threshold: float = 1.0,
    ):
        self.node = node
        self.monitor_interval = monitor_interval
        self.suspicion_threshold = suspicion_threshold
        self._status: Dict[str, str] = {}
        self._events: List[FailureEvent] = []
        self._callbacks: List[Callable[[FailureEvent], None]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def on_failure(self, cb: Callable[[FailureEvent], None]) -> None:
        self._callbacks.append(cb)

    def get_status(self, node_id: str) -> str:
        if node_id == self.node.node_id:
            return "alive"
        return self._status.get(node_id, "alive")

    def get_all_status(self) -> Dict[str, str]:
        return dict(self._status)

    def history(self) -> List[FailureEvent]:
        return list(self._events)

    def _record(self, node_id: str, status: str) -> None:
        if self._status.get(node_id) == status:
            return
        self._status[node_id] = status
        ev = FailureEvent(node_id, status, self.node._now())
        self._events.append(ev)
        for cb in self._callbacks:
            try:
                cb(ev)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("failure callback error: %s", exc)

    async def _scan(self) -> None:
        # tick() advances Raft's own heartbeat-based detection and returns
        # non-alive members.
        non_alive = self.node.tick()
        for node_id, st in non_alive.items():
            self._record(node_id, st)
        # Detect recoveries: a member previously down/suspected that is now
        # alive again (heartbeat seen).
        for node_id in list(self._status.keys()):
            if node_id not in non_alive and self.node.is_alive(node_id):
                self._record(node_id, "recovered")
                # Once recovered we forget the sticky status so future
                # failures are reported again.
                self._status.pop(node_id, None)

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._scan()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("health scan error: %s", exc)
            await asyncio.sleep(self.monitor_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None


class RecoveryManager:
    """Coordinates recovery of failed / restarted nodes."""

    def __init__(self, node: RaftNode, health: Optional[HealthChecker] = None):
        self.node = node
        self.health = health
        self._recovering: Set[str] = set()

    async def recover_node(self, node_id: str) -> bool:
        """Attempt to recover ``node_id`` through a log/snapshot sync.

        Returns True when the node is considered recovered (healthy and
        part of the committed membership). Real network sync is out of scope
        for the in-process simulation; we drive the Raft-side rejoin and
        verify membership + liveness.
        """
        if node_id in self._recovering:
            return False
        self._recovering.add(node_id)
        try:
            # 1. Reload any persisted Raft state (mirrors a restarted node).
            self.node._load_state()

            # 2. If this node is the leader, ensure the recovered node is a
            #    member; if it was removed while down, re-add it.
            if self.node.state == NodeState.LEADER:
                if node_id not in self.node.get_member_ids():
                    # Re-add through the safe joint-consensus path.
                    from knowledge_engine.cluster_manager import ClusterManager
                    cm = ClusterManager(self.node)
                    try:
                        await cm.add_node(
                            node_id,
                            *self._lookup_address(node_id),
                        )
                    except MembershipChangeError:
                        pass

            # 3. Wait for the node to become alive again (heartbeats).
            recovered = await self._wait_alive(node_id)
            # 4. Clear any sticky failure status.
            if self.health is not None:
                self.health._status.pop(node_id, None)
            return recovered
        finally:
            self._recovering.discard(node_id)

    def _lookup_address(self, node_id: str):
        cfg = self.node.get_cluster_config()
        m = cfg.get("members", {}).get(node_id)
        if m:
            return (m["address"], m["port"])
        return ("127.0.0.1", 0)

    async def _wait_alive(self, node_id: str, timeout: float = 10.0) -> bool:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self.node.is_alive(node_id) and node_id in self.node.get_member_ids():
                return True
            await asyncio.sleep(0.1)
        return False


__all__ = ["HealthChecker", "RecoveryManager", "FailureEvent"]
