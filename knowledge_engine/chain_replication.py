"""
Chain Replication: a linear, head-to-tail replication protocol.

Unlike Raft/Paxos, Chain Replication has no leader election. A totally
ordered *chain* of nodes is established. Writes are forwarded from the
client to the **head**; each node applies the operation and forwards it down
the chain. The **tail** is the only node that serves reads and the only node
that acknowledges a write as committed (after the update has propagated
through every node). This yields strong consistency with a single commit
point while keeping the head free to accept concurrent writes.

This module implements ``ChainNode`` (head/middle/tail behaviour) and
``ChainReplication`` (chain construction, membership changes, and
client-facing ``put`` / ``get``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from knowledge_engine.consensus_net import LocalTransport

logger = logging.getLogger(__name__)


@dataclass
class ChainMessage:
    kind: str  # "write" | "write-ack" | "read" | "read-ack"
    key: str
    value: Any = None
    sender: str = ""
    request_id: str = ""


class ChainNode:
    def __init__(self, node_id: str, transport: LocalTransport,
                 store: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.t = transport
        self.state: Dict[str, Any] = store if store is not None else {}
        self.successor: Optional[str] = None
        self.predecessor: Optional[str] = None
        self.is_head = False
        self.is_tail = False
        self._pending: Dict[str, asyncio.Future] = {}
        transport.register(node_id, self.handle)

    def configure(self, predecessor: Optional[str], successor: Optional[str]) -> None:
        self.predecessor = predecessor
        self.successor = successor
        self.is_head = predecessor is None
        self.is_tail = successor is None

    async def handle(self, src: str, msg: dict) -> None:
        m = ChainMessage(**msg)
        if m.kind == "write":
            # Apply locally, then propagate down the chain.
            self.state[m.key] = m.value
            if self.successor is not None:
                await self.t.send(self.node_id, self.successor, msg)
            else:
                # We are the tail: commit and ack back up the chain.
                await self._ack(m)
        elif m.kind == "write-ack":
            if self.predecessor is not None:
                await self.t.send(self.node_id, self.predecessor, msg)
            fut = self._pending.pop(m.request_id, None)
            if fut and not fut.done():
                fut.set_result(True)
        elif m.kind == "read":
            # Reads are served only at the tail for strong consistency;
            # intermediate nodes forward the request down the chain.
            if self.is_tail:
                await self.t.send(self.node_id, m.sender, {
                    "kind": "read-ack", "key": m.key,
                    "value": self.state.get(m.key), "request_id": m.request_id,
                })
            else:
                await self.t.send(self.node_id, self.successor, msg)
        elif m.kind == "read-ack":
            if self.predecessor is not None:
                await self.t.send(self.node_id, self.predecessor, msg)
            fut = self._pending.pop(m.request_id, None)
            if fut and not fut.done():
                fut.set_result(m.value)

    async def _ack(self, m: ChainMessage) -> None:
        await self.t.send(self.node_id, m.sender, {
            "kind": "write-ack", "key": m.key, "value": m.value,
            "request_id": m.request_id,
        })

    # -- client-facing helpers (used by the head) ----------------------
    async def put(self, key: str, value: Any, request_id: str) -> bool:
        if not self.is_head:
            raise RuntimeError("put must be issued to the chain head")
        fut = asyncio.get_event_loop().create_future()
        self._pending[request_id] = fut
        await self.t.send(self.node_id, self.successor, {
            "kind": "write", "key": key, "value": value,
            "sender": self.node_id, "request_id": request_id,
        })
        return await asyncio.wait_for(fut, timeout=5.0)

    async def get(self, key: str, request_id: str) -> Any:
        if not self.is_head:
            raise RuntimeError("get must be issued to the chain head")
        fut = asyncio.get_event_loop().create_future()
        self._pending[request_id] = fut
        await self.t.send(self.node_id, self.successor, {
            "kind": "read", "key": key, "sender": self.node_id,
            "request_id": request_id,
        })
        return await asyncio.wait_for(fut, timeout=5.0)


class ChainReplication:
    """Builds and manages a chain of nodes."""

    def __init__(self, node_ids: List[str], transport: LocalTransport):
        if len(node_ids) < 1:
            raise ValueError("chain requires at least one node")
        self.node_ids = list(node_ids)
        self.t = transport
        self.nodes: Dict[str, ChainNode] = {
            nid: ChainNode(nid, transport) for nid in node_ids
        }
        self._wire()

    def _wire(self) -> None:
        ids = self.node_ids
        for i, nid in enumerate(ids):
            pred = ids[i - 1] if i > 0 else None
            succ = ids[i + 1] if i < len(ids) - 1 else None
            self.nodes[nid].configure(pred, succ)

    @property
    def head(self) -> ChainNode:
        return self.nodes[self.node_ids[0]]

    @property
    def tail(self) -> ChainNode:
        return self.nodes[self.node_ids[-1]]

    def add_node(self, new_id: str) -> None:
        """Append a new node at the tail (becomes new tail)."""
        if new_id in self.nodes:
            return
        old_tail = self.node_ids[-1]
        self.nodes[new_id] = ChainNode(new_id, self.t)
        self.node_ids.append(new_id)
        # Re-wire so the new node is inserted after the old tail.
        self.nodes[old_tail].successor = new_id
        self.nodes[old_tail].is_tail = False
        self.nodes[new_id].configure(predecessor=old_tail, successor=None)

    def remove_node(self, node_id: str) -> None:
        """Remove a node, repairing the chain links around it."""
        if node_id not in self.nodes or len(self.node_ids) == 1:
            return
        idx = self.node_ids.index(node_id)
        pred = self.node_ids[idx - 1] if idx > 0 else None
        succ = self.node_ids[idx + 1] if idx < len(self.node_ids) - 1 else None
        if pred:
            self.nodes[pred].successor = succ
            self.nodes[pred].is_tail = succ is None
        if succ:
            self.nodes[succ].predecessor = pred
            self.nodes[succ].is_head = pred is None
        self.nodes.pop(node_id)
        self.node_ids.remove(node_id)

    async def put(self, key: str, value: Any) -> bool:
        return await self.head.put(key, value, f"w-{key}")

    async def get(self, key: str) -> Any:
        return await self.head.get(key, f"r-{key}")


__all__ = ["ChainNode", "ChainReplication", "ChainMessage"]
