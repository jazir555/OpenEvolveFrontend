"""
Cluster membership management built on top of the existing Raft node.

ClusterManager exposes the high-level, spec-described operations
(``add_node`` / ``remove_node``) and orchestrates them through a *two-phase
joint consensus* transition that is replicated via the normal Raft log. This
reuses :class:`knowledge_engine.distributed_coordination.RaftNode` (its
``submit_command`` / ``on_commit`` / ``receive_membership_update`` hooks) so
the existing Raft API stays intact while gaining safe dynamic membership.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from knowledge_engine.distributed_coordination import (
    LogEntryType,
    NodeState,
    RaftNode,
    MembershipChangeError,
    NotLeaderException,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeDescriptor:
    node_id: str
    address: str
    port: int


class JointConsensusManager:
    """Two-phase joint-consensus transition helper.

    A transition proceeds: stable(old) -> joint(old | new) -> stable(new).
    Each phase is recorded as a ``CONFIG_CHANGE`` entry committed through Raft
    so every replica converges on the same configuration.
    """

    def __init__(self, node: RaftNode):
        self.node = node

    def compute_phase(
        self,
        current: set,
        add: Optional[str] = None,
        remove: Optional[str] = None,
    ) -> set:
        if add is not None:
            current = current | {add}
        if remove is not None:
            current = current - {remove}
        return current


class ClusterManager:
    """Leader-driven cluster membership API over an existing Raft node."""

    def __init__(self, node: RaftNode):
        self.node = node
        self.joint = JointConsensusManager(node)
        self._pending: Dict[int, asyncio.Future] = {}
        self._register_commit_hook()

    # ------------------------------------------------------------------
    def _register_commit_hook(self):
        self.node.on_commit(self._on_committed_config)

    def _on_committed_config(self, entry) -> None:
        if entry.entry_type != LogEntryType.CONFIG_CHANGE:
            return
        payload = entry.data
        members = {
            mid: (m["address"], m["port"])
            for mid, m in payload.get("members", {}).items()
        }
        # Apply the committed configuration locally (idempotent for followers;
        # on the leader it simply converges to the same result).
        try:
            self.node.receive_membership_update(members)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to apply membership update: %s", exc)
        future = self._pending.pop(entry.index, None)
        if future and not future.done():
            future.set_result(payload)

    # ------------------------------------------------------------------
    def _require_leader(self) -> None:
        if self.node.state != NodeState.LEADER:
            raise NotLeaderException(
                f"cluster changes must be driven by the leader "
                f"(current={self.node.state.value}, leader={self.node.get_leader()})"
            )

    def _member_map(self, member_ids: set) -> Dict[str, Tuple[str, int]]:
        cfg = self.node.get_cluster_config()
        known = {
            mid: (m["address"], m["port"])
            for mid, m in cfg.get("members", {}).items()
        }
        known[self.node.node_id] = (self.node.address, self.node.port)
        return {mid: known[mid] for mid in member_ids if mid in known}

    async def _propose_config(
        self, member_ids: set, phase: str, timeout: float
    ) -> Dict[str, Any]:
        members = self._member_map(member_ids)
        entry = await self.node.submit_command(
            LogEntryType.CONFIG_CHANGE,
            {"phase": phase, "members": {
                mid: {"address": a, "port": p} for mid, (a, p) in members.items()
            }},
        )
        if entry is None:
            raise NotLeaderException("lost leadership while proposing config")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[entry.index] = fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(entry.index, None)
            raise MembershipChangeError("config change did not commit in time")

    # ------------------------------------------------------------------
    async def add_node(
        self,
        node_id: str,
        address: str,
        port: int,
        timeout: float = 10.0,
    ) -> None:
        """Add a node via two-phase joint consensus. Leader-only."""
        self._require_leader()
        if node_id == self.node.node_id:
            raise MembershipChangeError("cannot add self")
        current = self.node.get_member_ids()
        if node_id in current:
            raise MembershipChangeError(f"{node_id} is already a member")

        # Phase 1: joint configuration (old + new).
        joint = self.joint.compute_phase(current, add=node_id)
        joint = joint | {self.node.node_id}
        await self._propose_config(joint, "joint_consensus", timeout)
        # (Follower-side application happens via the commit hook above.)

        # Phase 2: stable configuration (new only).
        await self._propose_config(joint, "stable", timeout)
        logger.info("leader committed addition of %s", node_id)

    async def remove_node(self, node_id: str, timeout: float = 10.0) -> None:
        """Remove a node via two-phase joint consensus. Leader-only."""
        self._require_leader()
        if node_id == self.node.node_id:
            raise MembershipChangeError("cannot remove self")
        current = self.node.get_member_ids()
        if node_id not in current:
            raise MembershipChangeError(f"{node_id} is not a member")

        # Phase 1: joint configuration (old - new).
        joint = self.joint.compute_phase(current, remove=node_id)
        joint = joint | {self.node.node_id}
        await self._propose_config(joint, "joint_consensus", timeout)

        # Phase 2: stable configuration without the removed node.
        stable = joint - {node_id}
        await self._propose_config(stable, "stable", timeout)
        logger.info("leader committed removal of %s", node_id)

    def get_members(self) -> List[NodeDescriptor]:
        cfg = self.node.get_cluster_config()
        out = []
        for mid, m in cfg.get("members", {}).items():
            out.append(NodeDescriptor(mid, m["address"], m["port"]))
        return out


__all__ = ["ClusterManager", "JointConsensusManager", "NodeDescriptor"]
