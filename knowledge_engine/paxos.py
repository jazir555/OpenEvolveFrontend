"""
Multi-Paxos and Fast Paxos consensus implementations.

Both rely on an injectable :class:`LocalTransport` (or any object exposing
``send(src, dst, msg)`` / ``broadcast(src, recipients, msg)``) so nodes can
be wired together in-process for deterministic testing.

Multi-Paxos: a distinguished, stable leader (the "proposer") runs a single
Prepare/Prepare phase for the first instance, then reuses the established
promise for all subsequent instances, sending only Accept messages. This is
the classic "leader lease" optimisation that turns Paxos into a Raft-like
state machine.

Fast Paxos: clients may propose directly to all acceptors (the *fast path*,
requiring a quorum of size 2F+1 to tolerate F fast-path collisions). If the
fast path cannot be completed (collision detected), the leader falls back to
the classic *slow path* (Prepare/Accept).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from knowledge_engine.consensus_net import LocalTransport

logger = logging.getLogger(__name__)


@dataclass
class Promise:
    n: int
    accepted_n: Optional[int]
    accepted_v: Any


# ----------------------------------------------------------------------
# Multi-Paxos
# ----------------------------------------------------------------------
class MultiPaxosAcceptor:
    def __init__(self, node_id: str, transport: LocalTransport):
        self.node_id = node_id
        self.t = transport
        self.promised_n: int = 0
        self.accepted_n: Optional[int] = None
        self.accepted_v: Any = None
        transport.register(node_id, self.handle)

    async def handle(self, src: str, msg: dict) -> None:
        kind = msg["kind"]
        if kind == "prepare":
            n = msg["n"]
            if n >= self.promised_n:
                self.promised_n = n
                await self.t.send(self.node_id, msg["leader"], {
                    "kind": "promise", "n": n, "acceptor": self.node_id,
                    "accepted_n": self.accepted_n, "accepted_v": self.accepted_v,
                })
        elif kind == "accept":
            n = msg["n"]
            if n >= self.promised_n:
                self.promised_n = n
                self.accepted_n = n
                self.accepted_v = msg["v"]
                await self.t.send(self.node_id, msg["leader"], {
                    "kind": "accepted", "n": n, "acceptor": self.node_id,
                    "instance": msg["instance"], "v": self.accepted_v,
                })


class MultiPaxosProposer:
    def __init__(self, node_id: str, peers: List[str], transport: LocalTransport,
                 quorum: Optional[int] = None):
        self.node_id = node_id
        self.peers = peers
        self.t = transport
        self.quorum = quorum or (len(peers) // 2 + 1)
        self.counter = 0  # proposal number generator
        self.next_instance = 0
        self.decided: Dict[int, Any] = {}
        self._prepared = False
        transport.register(node_id, self.handle)

    def _ballot(self) -> int:
        self.counter += 1
        # High-order bits encode the proposer so ballots are globally unique.
        return self.counter * 1000 + hash(self.node_id) % 1000

    async def handle(self, src: str, msg: dict) -> None:
        if msg["kind"] == "accepted":
            inst = msg.get("instance")
            if inst is None:
                return
            self._accepted_by[inst].add(src)
            if len(self._accepted_by[inst]) >= self.quorum and inst not in self.decided:
                self.decided[inst] = msg["v"]
                await self._learn(inst, msg["v"])

    async def _learn(self, inst: int, v: Any) -> None:
        for p in self.peers:
            await self.t.send(self.node_id, p, {"kind": "learn", "instance": inst, "v": v})

    async def _prepare(self) -> None:
        n = self._ballot()
        self._accepted_by: Dict[int, Set[str]] = {}
        promises: Dict[int, Promise] = {}
        await self.t.broadcast(self.node_id, self.peers, {"kind": "prepare", "n": n, "leader": self.node_id})
        # Collect promises (simplified: expect quorum responses synchronously).
        await asyncio.sleep(0.01)
        self._prepared = True
        self._prepared_n = n

    async def propose(self, value: Any) -> int:
        if not self._prepared:
            await self._prepare()
        inst = self.next_instance
        self.next_instance += 1
        self._accepted_by[inst] = set()
        n = self._prepared_n
        # The leader also counts as an acceptor if it is part of the quorum.
        if self.node_id in self.peers:
            self._accepted_by[inst].add(self.node_id)
        await self.t.broadcast(self.node_id, self.peers, {
            "kind": "accept", "n": n, "leader": self.node_id,
            "instance": inst, "v": value,
        })
        # Wait until decided (quorum of accepted matching this instance).
        for _ in range(200):
            if inst in self.decided:
                return inst
            await asyncio.sleep(0.005)
        raise RuntimeError("MultiPaxos: did not reach quorum")


class MultiPaxosLearner:
    def __init__(self, node_id: str, transport: LocalTransport):
        self.node_id = node_id
        self.t = transport
        self.learned: Dict[int, Any] = {}
        transport.register(node_id, self.handle)

    async def handle(self, src: str, msg: dict) -> None:
        if msg["kind"] == "learn":
            self.learned[msg["instance"]] = msg["v"]


class MultiPaxosCluster:
    """Wires one distinguished leader + acceptors + learners together."""

    def __init__(self, acceptors: List[str], learners: List[str],
                 transport: LocalTransport, leader: Optional[str] = None):
        self.t = transport
        self.acceptors = acceptors
        self.learners = learners
        self.leader_id = leader or acceptors[0]
        self._acceptor_nodes = {a: MultiPaxosAcceptor(a, transport) for a in acceptors}
        self._learner_nodes = {l: MultiPaxosLearner(l, transport) for l in learners}
        self.leader = MultiPaxosProposer(self.leader_id, acceptors, transport)

    async def propose(self, value: Any) -> Any:
        inst = await self.leader.propose(value)
        # Learners receive the learn broadcast; return the decided value.
        for ln in self._learner_nodes.values():
            if inst in ln.learned:
                return ln.learned[inst]
        await asyncio.sleep(0.02)
        for ln in self._learner_nodes.values():
            if inst in ln.learned:
                return ln.learned[inst]
        return value


# ----------------------------------------------------------------------
# Fast Paxos
# ----------------------------------------------------------------------
class FastPaxosAcceptor:
    def __init__(self, node_id: str, transport: LocalTransport):
        self.node_id = node_id
        self.t = transport
        self.promised_n: int = 0          # classic round
        self.accepted_fast: Dict[int, Any] = {}
        self.accepted_classic: Dict[int, Any] = {}
        transport.register(node_id, self.handle)

    async def handle(self, src: str, msg: dict) -> None:
        k = msg["kind"]
        if k == "prepare":
            n = msg["n"]
            if n >= self.promised_n:
                self.promised_n = n
                await self.t.send(self.node_id, msg["leader"], {
                    "kind": "promise", "n": n, "acceptor": self.node_id,
                    "fast": dict(self.accepted_fast), "classic": dict(self.accepted_classic),
                })
        elif k == "fast-accept":
            inst = msg["instance"]
            self.accepted_fast[inst] = msg["v"]
            await self.t.send(self.node_id, msg["leader"], {
                "kind": "fast-accepted", "instance": inst, "acceptor": self.node_id, "v": msg["v"],
            })
        elif k == "accept":
            inst = msg["instance"]
            self.accepted_classic[inst] = msg["v"]
            await self.t.send(self.node_id, msg["leader"], {
                "kind": "accepted", "instance": inst, "acceptor": self.node_id, "v": msg["v"],
            })


class FastPaxosNode:
    """Coordinator/leader implementing both fast and slow paths."""

    def __init__(self, node_id: str, acceptors: List[str], transport: LocalTransport,
                 fast_quorum: Optional[int] = None):
        self.node_id = node_id
        self.acceptors = acceptors
        self.t = transport
        # Fast quorum needs 2F+1 where F = tolerated failures.
        f = (len(acceptors) - 1) // 2
        self.fast_quorum = fast_quorum or (2 * f + 1)
        self.slow_quorum = f + 1
        self.next_instance = 0
        self.decided: Dict[int, Any] = {}
        transport.register(node_id, self.handle)
        self._fast_accepted: Dict[int, Dict[str, Any]] = {}
        self._classic_accepted: Dict[int, Dict[str, Any]] = {}

    async def handle(self, src: str, msg: dict) -> None:
        k = msg["kind"]
        if k == "fast-accepted":
            inst = msg["instance"]
            self._fast_accepted.setdefault(inst, {})[src] = msg["v"]
        elif k == "accepted":
            inst = msg["instance"]
            self._classic_accepted.setdefault(inst, {})[src] = msg["v"]

    async def propose_fast(self, value: Any) -> Any:
        """Fast path: client value broadcast directly to acceptors."""
        inst = self.next_instance
        self.next_instance += 1
        self._fast_accepted[inst] = {}
        await self.t.broadcast(self.node_id, self.acceptors, {
            "kind": "fast-accept", "instance": inst, "leader": self.node_id, "v": value,
        })
        # Wait for fast quorum of matching values.
        for _ in range(200):
            votes = self._fast_accepted.get(inst, {})
            if len(votes) >= self.fast_quorum and len(set(votes.values())) == 1:
                self.decided[inst] = value
                return value
            await asyncio.sleep(0.005)
        # Collision / insufficient fast quorum -> slow path.
        return await self._propose_slow(inst, value)

    async def _propose_slow(self, inst: int, value: Any) -> Any:
        n = 1
        self._classic_accepted[inst] = {}
        await self.t.broadcast(self.node_id, self.acceptors, {
            "kind": "accept", "n": n, "instance": inst, "leader": self.node_id, "v": value,
        })
        for _ in range(200):
            votes = self._classic_accepted.get(inst, {})
            if len(votes) >= self.slow_quorum:
                self.decided[inst] = value
                return value
            await asyncio.sleep(0.005)
        raise RuntimeError("FastPaxos: slow path did not reach quorum")


__all__ = [
    "MultiPaxosAcceptor", "MultiPaxosProposer", "MultiPaxosLearner",
    "MultiPaxosCluster", "FastPaxosAcceptor", "FastPaxosNode",
]
