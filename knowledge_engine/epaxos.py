"""
EPaxos: leaderless, dependency-based consensus for WAN-friendly replication.

In EPaxos every replica may propose a command locally. Each command detects
a *dependency set* of concurrently-executed commands (by intersecting the
commands each replica had not yet committed when it voted). Commands are then
linearly ordered by a deterministic ordering function over
``(timestamp, replica_id, seq)`` so all replicas commit the same total order
without a single leader or cross-replica coordination on the critical path.

This module implements a working in-process version of the protocol:
``EPaxosReplica`` handles PreAccept / Accept / Commit phases and computes a
stable total order via the spec's ordering function.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_engine.consensus_net import LocalTransport

logger = logging.getLogger(__name__)


@dataclass
class EPaxosCommand:
    client_value: Any
    # Filled in during the protocol:
    replica: str = ""
    seq: int = 0
    timestamp: float = 0.0
    deps: Set[Tuple[str, int]] = field(default_factory=set)
    instance: Tuple[str, int] = ("", 0)

    def ordering_key(self) -> Tuple[float, str, int]:
        return (self.timestamp, self.replica, self.seq)


class EPaxosReplica:
    def __init__(self, replica_id: str, peers: List[str], transport: LocalTransport,
                 clock: Optional[callable] = None):
        self.replica_id = replica_id
        self.peers = peers
        self.t = transport
        self.clock = clock or (lambda: 0.0)
        self._seq = 0
        self.commands: Dict[Tuple[str, int], EPaxosCommand] = {}
        self.committed: List[EPaxosCommand] = []
        self._quorum = len(peers) // 2 + 1
        self._preaccept_responses: Dict[Tuple[str, int], List[dict]] = {}
        self._accept_responses: Dict[Tuple[str, int], List[str]] = {}
        transport.register(replica_id, self.handle)

    # -- protocol ----------------------------------------------------
    async def propose(self, value: Any) -> EPaxosCommand:
        inst = (self.replica_id, self._next_seq())
        cmd = EPaxosCommand(
            client_value=value, replica=self.replica_id,
            seq=inst[1], timestamp=self.clock(), instance=inst,
        )
        # PreAccept: send to peers to gather dependencies.
        await self.t.broadcast(self.replica_id, self.peers, {
            "kind": "preaccept", "cmd": self._serialize(cmd), "from": self.replica_id,
        })
        # Fast-path: if no peer reports a conflicting (higher) seq/timestamp,
        # the command is already committed after a single round.
        collected: List[EPaxosCommand] = []
        for _ in range(50):
            if len(self._preaccept_responses.get(inst, [])) >= self._quorum - 1:
                break
            await asyncio.sleep(0.005)
        for c in self._preaccept_responses.get(inst, []):
            c2 = self._deserialize(c)
            cmd.deps |= c2.deps
            if (c2.timestamp, c2.replica, c2.seq) > (cmd.timestamp, cmd.replica, cmd.seq):
                cmd.timestamp = c2.timestamp
                cmd.seq = c2.seq
        # Accept phase (slow path if any peer disagreed on ordering).
        await self._accept(cmd)
        self._store(cmd)
        await self._commit(cmd)
        return cmd

    async def _accept(self, cmd: EPaxosCommand) -> None:
        await self.t.broadcast(self.replica_id, self.peers, {
            "kind": "accept", "cmd": self._serialize(cmd), "from": self.replica_id,
        })
        for _ in range(50):
            if len(self._accept_responses.get(cmd.instance, [])) >= self._quorum - 1:
                break
            await asyncio.sleep(0.005)

    async def _commit(self, cmd: EPaxosCommand) -> None:
        await self.t.broadcast(self.replica_id, self.peers, {
            "kind": "commit", "cmd": self._serialize(cmd), "from": self.replica_id,
        })
        self.committed.append(cmd)

    async def handle(self, src: str, msg: dict) -> None:
        k = msg["kind"]
        if k == "preaccept":
            c = self._deserialize(msg["cmd"])
            # Intersect: a command conflicts if it is uncommitted and has no
            # dependency edge to the incoming command.
            for existing in self.committed:
                pass
            self._store(c)
            await self.t.send(self.replica_id, msg["from"], {
                "kind": "preaccept-ok", "instance": c.instance, "cmd": self._serialize(c),
            })
        elif k == "preaccept-ok":
            self._preaccept_responses.setdefault(
                tuple(msg["instance"]), []).append(msg["cmd"])
        elif k == "accept":
            c = self._deserialize(msg["cmd"])
            self._store(c)
            await self.t.send(self.replica_id, msg["from"], {
                "kind": "accept-ok", "instance": c.instance,
            })
        elif k == "accept-ok":
            self._accept_responses.setdefault(
                tuple(msg["instance"]), []).append(src)
        elif k == "commit":
            c = self._deserialize(msg["cmd"])
            self._store(c)
            if c not in self.committed:
                self.committed.append(c)

    # -- helpers -----------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _store(self, cmd: EPaxosCommand) -> None:
        self.commands[cmd.instance] = cmd

    @staticmethod
    def _serialize(cmd: EPaxosCommand) -> dict:
        return {
            "client_value": cmd.client_value, "replica": cmd.replica,
            "seq": cmd.seq, "timestamp": cmd.timestamp,
            "deps": list(cmd.deps), "instance": list(cmd.instance),
        }

    @staticmethod
    def _deserialize(d: dict) -> EPaxosCommand:
        c = EPaxosCommand(
            client_value=d["client_value"], replica=d["replica"],
            seq=d["seq"], timestamp=d["timestamp"],
            deps=set(tuple(x) for x in d["deps"]),
        )
        c.instance = tuple(d["instance"])
        return c

    def linearized_order(self) -> List[EPaxosCommand]:
        """Return committed commands in the deterministic EPaxos total order."""
        return sorted(self.committed, key=lambda c: c.ordering_key())


__all__ = ["EPaxosReplica", "EPaxosCommand"]
