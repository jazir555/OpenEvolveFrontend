"""
Distributed Coordination Layer

Provides distributed consensus, leader election, and state replication
for running the knowledge engine across multiple nodes.

Features:
- Raft consensus algorithm implementation
- Leader election
- Distributed state machine replication
- Cluster membership management
- Partition tolerance and fault recovery
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntryType(Enum):
    """Types of log entries."""
    KNOWLEDGE_ADD = "knowledge_add"
    KNOWLEDGE_UPDATE = "knowledge_update"
    KNOWLEDGE_DELETE = "knowledge_delete"
    RELATION_ADD = "relation_add"
    CONFIG_CHANGE = "config_change"
    NO_OP = "no_op"


@dataclass
class LogEntry:
    """A single entry in the Raft log."""
    index: int
    term: int
    entry_type: LogEntryType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "term": self.term,
            "entry_type": self.entry_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogEntry:
        return cls(
            index=data["index"],
            term=data["term"],
            entry_type=LogEntryType(data["entry_type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    address: str
    port: int
    last_seen: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "last_seen": self.last_seen.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata
        }


@dataclass
class RaftState:
    """Persistent Raft state."""
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "log": [entry.to_dict() for entry in self.log]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RaftState:
        return cls(
            current_term=data.get("current_term", 0),
            voted_for=data.get("voted_for"),
            log=[LogEntry.from_dict(e) for e in data.get("log", [])]
        )


@dataclass
class RaftVolatileState:
    """Volatile Raft state."""
    commit_index: int = 0
    last_applied: int = 0
    # Leader only
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)


class RaftNode:
    """
    Raft consensus node for distributed coordination.
    
    Implements the Raft consensus algorithm for:
    - Leader election
    - Log replication
    - Safety guarantees
    """
    
    def __init__(
        self,
        node_id: str,
        address: str,
        port: int,
        peers: List[Tuple[str, str, int]],  # (node_id, address, port)
        data_dir: Optional[str] = None,
        heartbeat_interval: float = 0.05,  # 50ms
        election_timeout_min: float = 0.15,  # 150ms
        election_timeout_max: float = 0.3,   # 300ms
    ):
        self.node_id = node_id
        self.address = address
        self.port = port
        self.peers = {peer_id: (peer_addr, peer_port) for peer_id, peer_addr, peer_port in peers}
        self.data_dir = Path(data_dir) if data_dir else Path(f"./raft_data/{node_id}")
        
        # Timing parameters
        self.heartbeat_interval = heartbeat_interval
        self.election_timeout_min = election_timeout_min
        self.election_timeout_max = election_timeout_max
        
        # State
        self.state = NodeState.FOLLOWER
        self.persistent_state = RaftState()
        self.volatile_state = RaftVolatileState()
        
        # Leader tracking
        self.current_leader: Optional[str] = None
        self.leader_lease_expiry: Optional[datetime] = None
        
        # Timers
        self._election_timer: Optional[asyncio.Task] = None
        self._heartbeat_timer: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self._commit_callbacks: List[Callable[[LogEntry], None]] = []
        self._state_change_callbacks: List[Callable[[NodeState, NodeState], None]] = []
        
        # Vote tracking (for elections)
        self._votes_received: Set[str] = set()
        
        # Lock for state access
        self._lock = asyncio.Lock()
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persistent state
        self._load_state()
        
        logger.info(f"Raft node {node_id} initialized at {address}:{port}")
    
    def _load_state(self):
        """Load persistent state from disk."""
        state_file = self.data_dir / "raft_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.persistent_state = RaftState.from_dict(data)
                logger.info(f"Loaded Raft state: term={self.persistent_state.current_term}")
            except Exception as e:
                logger.error(f"Failed to load Raft state: {e}")
    
    def _save_state(self):
        """Save persistent state to disk."""
        state_file = self.data_dir / "raft_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(self.persistent_state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Raft state: {e}")
    
    async def start(self):
        """Start the Raft node."""
        self._running = True
        
        # Start election timer
        await self._reset_election_timer()
        
        logger.info(f"Raft node {self.node_id} started as {self.state.value}")
    
    async def stop(self):
        """Stop the Raft node."""
        self._running = False
        
        # Cancel timers
        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
        
        logger.info(f"Raft node {self.node_id} stopped")
    
    async def _reset_election_timer(self):
        """Reset the election timer."""
        if self._election_timer:
            self._election_timer.cancel()
        
        timeout = random.uniform(self.election_timeout_min, self.election_timeout_max)
        self._election_timer = asyncio.create_task(self._election_timer_task(timeout))
    
    async def _election_timer_task(self, timeout: float):
        """Election timer coroutine."""
        await asyncio.sleep(timeout)
        
        async with self._lock:
            if self.state != NodeState.LEADER and self._running:
                await self._start_election()
    
    async def _start_election(self):
        """Start a leader election."""
        async with self._lock:
            old_state = self.state
            self.state = NodeState.CANDIDATE
            self.persistent_state.current_term += 1
            self.persistent_state.voted_for = self.node_id
            self._votes_received = {self.node_id}
            self._save_state()
            
            logger.info(f"Node {self.node_id} starting election for term {self.persistent_state.current_term}")
            
            # Notify state change
            await self._notify_state_change(old_state, self.state)
        
        # Send RequestVote RPCs to all peers
        tasks = []
        for peer_id, (peer_addr, peer_port) in self.peers.items():
            task = asyncio.create_task(self._send_request_vote(peer_id, peer_addr, peer_port))
            tasks.append(task)
        
        # Wait for responses (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.election_timeout_max
            )
        except asyncio.TimeoutError:
            pass
        
        # Check if we won
        async with self._lock:
            if self.state == NodeState.CANDIDATE:
                majority = (len(self.peers) + 1) // 2 + 1
                if len(self._votes_received) >= majority:
                    await self._become_leader()
    
    async def _send_request_vote(
        self, 
        peer_id: str, 
        peer_addr: str, 
        peer_port: int
    ) -> bool:
        """Send RequestVote RPC to a peer."""
        async with self._lock:
            last_log_index = len(self.persistent_state.log)
            last_log_term = self.persistent_state.log[-1].term if self.persistent_state.log else 0
            
            request = {
                "type": "RequestVote",
                "term": self.persistent_state.current_term,
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term
            }
        
        try:
            # In a real implementation, this would be an RPC call
            # For now, simulate the response
            response = await self._simulate_rpc(peer_addr, peer_port, request)
            
            if response.get("vote_granted"):
                async with self._lock:
                    if self.state == NodeState.CANDIDATE:
                        self._votes_received.add(peer_id)
                        logger.info(f"Received vote from {peer_id}")
                        return True
            else:
                # Update term if peer has higher term
                if response.get("term", 0) > self.persistent_state.current_term:
                    async with self._lock:
                        self.persistent_state.current_term = response["term"]
                        self.state = NodeState.FOLLOWER
                        self.persistent_state.voted_for = None
                        self._save_state()
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to send RequestVote to {peer_id}: {e}")
            return False
    
    async def _become_leader(self):
        """Transition to leader state."""
        old_state = self.state
        self.state = NodeState.LEADER
        self.current_leader = self.node_id
        
        # Initialize leader state
        last_log_index = len(self.persistent_state.log)
        for peer_id in self.peers:
            self.volatile_state.next_index[peer_id] = last_log_index + 1
            self.volatile_state.match_index[peer_id] = 0
        
        logger.info(f"Node {self.node_id} became leader for term {self.persistent_state.current_term}")
        
        # Notify state change
        await self._notify_state_change(old_state, self.state)
        
        # Start sending heartbeats
        await self._start_heartbeats()
        
        # Send initial no-op entry
        await self._append_entry(LogEntryType.NO_OP, {})
    
    async def _start_heartbeats(self):
        """Start sending heartbeat messages."""
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
        
        self._heartbeat_timer = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """Continuously send heartbeats to peers."""
        while self._running and self.state == NodeState.LEADER:
            tasks = []
            for peer_id, (peer_addr, peer_port) in self.peers.items():
                task = asyncio.create_task(self._send_append_entries(peer_id, peer_addr, peer_port))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_append_entries(
        self, 
        peer_id: str, 
        peer_addr: str, 
        peer_port: int
    ):
        """Send AppendEntries RPC to a peer."""
        async with self._lock:
            next_idx = self.volatile_state.next_index.get(peer_id, 1)
            prev_log_index = next_idx - 1
            prev_log_term = 0
            
            if prev_log_index > 0 and prev_log_index <= len(self.persistent_state.log):
                prev_log_term = self.persistent_state.log[prev_log_index - 1].term
            
            # Get entries to send
            entries = []
            if next_idx <= len(self.persistent_state.log):
                entries = [e.to_dict() for e in self.persistent_state.log[next_idx - 1:]]
            
            request = {
                "type": "AppendEntries",
                "term": self.persistent_state.current_term,
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": entries,
                "leader_commit": self.volatile_state.commit_index
            }
        
        try:
            response = await self._simulate_rpc(peer_addr, peer_port, request)
            
            async with self._lock:
                if response.get("success"):
                    if entries:
                        self.volatile_state.match_index[peer_id] = next_idx + len(entries) - 1
                        self.volatile_state.next_index[peer_id] = next_idx + len(entries)
                    await self._update_commit_index()
                else:
                    # Decrement next_index and retry
                    if response.get("term", 0) > self.persistent_state.current_term:
                        self.persistent_state.current_term = response["term"]
                        self.state = NodeState.FOLLOWER
                        self.persistent_state.voted_for = None
                        self._save_state()
                    else:
                        self.volatile_state.next_index[peer_id] = max(1, next_idx - 1)
                        
        except Exception as e:
            logger.debug(f"Failed to send AppendEntries to {peer_id}: {e}")
    
    async def _update_commit_index(self):
        """Update commit index based on match indices."""
        if self.state != NodeState.LEADER:
            return
        
        for n in range(self.volatile_state.commit_index + 1, len(self.persistent_state.log) + 1):
            # Count replicas
            count = 1  # Leader
            for peer_id in self.peers:
                if self.volatile_state.match_index.get(peer_id, 0) >= n:
                    count += 1
            
            # Check if majority and same term
            majority = (len(self.peers) + 1) // 2 + 1
            if count >= majority and self.persistent_state.log[n - 1].term == self.persistent_state.current_term:
                self.volatile_state.commit_index = n
                await self._apply_committed_entries()
    
    async def _apply_committed_entries(self):
        """Apply committed entries to state machine."""
        while self.volatile_state.last_applied < self.volatile_state.commit_index:
            self.volatile_state.last_applied += 1
            entry = self.persistent_state.log[self.volatile_state.last_applied - 1]
            
            # Notify callbacks
            for callback in self._commit_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(entry)
                    else:
                        callback(entry)
                except Exception as e:
                    logger.error(f"Commit callback error: {e}")
    
    async def handle_request_vote(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming RequestVote RPC."""
        async with self._lock:
            term = request.get("term", 0)
            candidate_id = request.get("candidate_id")
            last_log_index = request.get("last_log_index", 0)
            last_log_term = request.get("last_log_term", 0)
            
            # Update term if necessary
            if term > self.persistent_state.current_term:
                self.persistent_state.current_term = term
                self.state = NodeState.FOLLOWER
                self.persistent_state.voted_for = None
                self._save_state()
            
            vote_granted = False
            
            if term >= self.persistent_state.current_term:
                # Check if candidate's log is at least as up-to-date
                my_last_index = len(self.persistent_state.log)
                my_last_term = self.persistent_state.log[-1].term if self.persistent_state.log else 0
                
                log_ok = (last_log_term > my_last_term or 
                         (last_log_term == my_last_term and last_log_index >= my_last_index))
                
                if log_ok and (self.persistent_state.voted_for is None or 
                              self.persistent_state.voted_for == candidate_id):
                    vote_granted = True
                    self.persistent_state.voted_for = candidate_id
                    self._save_state()
                    await self._reset_election_timer()
            
            return {
                "term": self.persistent_state.current_term,
                "vote_granted": vote_granted
            }
    
    async def handle_append_entries(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming AppendEntries RPC."""
        async with self._lock:
            term = request.get("term", 0)
            leader_id = request.get("leader_id")
            prev_log_index = request.get("prev_log_index", 0)
            prev_log_term = request.get("prev_log_term", 0)
            entries = request.get("entries", [])
            leader_commit = request.get("leader_commit", 0)
            
            # Update leader info
            self.current_leader = leader_id
            self.leader_lease_expiry = datetime.utcnow() + timedelta(seconds=1)
            
            # Update term if necessary
            if term > self.persistent_state.current_term:
                self.persistent_state.current_term = term
                self.state = NodeState.FOLLOWER
                self.persistent_state.voted_for = None
                self._save_state()
            
            success = False
            
            if term >= self.persistent_state.current_term:
                # Reset election timer
                await self._reset_election_timer()
                
                # Revert to follower if we were a candidate
                if self.state == NodeState.CANDIDATE:
                    old_state = self.state
                    self.state = NodeState.FOLLOWER
                    await self._notify_state_change(old_state, self.state)
                
                # Check log consistency
                log_ok = True
                if prev_log_index > 0:
                    if prev_log_index > len(self.persistent_state.log):
                        log_ok = False
                    elif self.persistent_state.log[prev_log_index - 1].term != prev_log_term:
                        log_ok = False
                
                if log_ok:
                    success = True
                    
                    # Append new entries
                    for i, entry_data in enumerate(entries):
                        entry_index = prev_log_index + i + 1
                        entry = LogEntry.from_dict(entry_data)
                        
                        if entry_index > len(self.persistent_state.log):
                            self.persistent_state.log.append(entry)
                        elif self.persistent_state.log[entry_index - 1].term != entry.term:
                            # Delete conflicting entry and all that follow
                            self.persistent_state.log = self.persistent_state.log[:entry_index - 1]
                            self.persistent_state.log.append(entry)
                    
                    self._save_state()
                    
                    # Update commit index
                    if leader_commit > self.volatile_state.commit_index:
                        self.volatile_state.commit_index = min(
                            leader_commit, 
                            len(self.persistent_state.log)
                        )
                        await self._apply_committed_entries()
            
            return {
                "term": self.persistent_state.current_term,
                "success": success
            }
    
    async def _simulate_rpc(
        self, 
        peer_addr: str, 
        peer_port: int, 
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate RPC call (for testing/demo)."""
        # In a real implementation, this would make an actual network call
        # For now, randomly succeed or fail
        await asyncio.sleep(random.uniform(0.001, 0.01))  # Simulate network latency
        
        if random.random() < 0.9:  # 90% success rate
            if request["type"] == "RequestVote":
                return {"term": request["term"], "vote_granted": random.random() < 0.5}
            else:  # AppendEntries
                return {"term": request["term"], "success": True}
        else:
            raise Exception("RPC timeout")
    
    async def _notify_state_change(self, old_state: NodeState, new_state: NodeState):
        """Notify state change callbacks."""
        for callback in self._state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state)
                else:
                    callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def on_commit(self, callback: Callable[[LogEntry], None]):
        """Register a callback for committed entries."""
        self._commit_callbacks.append(callback)
    
    def on_state_change(self, callback: Callable[[NodeState, NodeState], None]):
        """Register a callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    async def submit_command(
        self, 
        entry_type: LogEntryType, 
        data: Dict[str, Any]
    ) -> Optional[LogEntry]:
        """
        Submit a command to the cluster.
        Only the leader can accept commands.
        """
        async with self._lock:
            if self.state != NodeState.LEADER:
                return None
            
            entry = await self._append_entry(entry_type, data)
            return entry
    
    async def _append_entry(
        self, 
        entry_type: LogEntryType, 
        data: Dict[str, Any]
    ) -> LogEntry:
        """Append an entry to the log."""
        entry = LogEntry(
            index=len(self.persistent_state.log) + 1,
            term=self.persistent_state.current_term,
            entry_type=entry_type,
            data=data
        )
        
        self.persistent_state.log.append(entry)
        self._save_state()
        
        return entry
    
    def get_state(self) -> NodeState:
        """Get current node state."""
        return self.state
    
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self.state == NodeState.LEADER
    
    def get_leader(self) -> Optional[str]:
        """Get current leader ID."""
        return self.current_leader
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "current_term": self.persistent_state.current_term,
            "log_size": len(self.persistent_state.log),
            "commit_index": self.volatile_state.commit_index,
            "last_applied": self.volatile_state.last_applied,
            "leader": self.current_leader,
            "peers": len(self.peers)
        }


class DistributedKnowledgeCoordinator:
    """
    High-level coordinator for distributed knowledge operations.
    Uses Raft for consensus and coordination.
    """
    
    def __init__(
        self,
        node_id: str,
        address: str,
        port: int,
        peers: List[Tuple[str, str, int]],
        data_dir: Optional[str] = None
    ):
        self.node_id = node_id
        self.raft = RaftNode(node_id, address, port, peers, data_dir)
        
        # Register callbacks
        self.raft.on_commit(self._on_commit)
        self.raft.on_state_change(self._on_state_change)
        
        # Pending commands (for waiting on replication)
        self._pending_commands: Dict[int, asyncio.Future] = {}
        
        # Local state machine (knowledge operations)
        self._knowledge_ops: List[LogEntry] = []
        
        logger.info(f"DistributedKnowledgeCoordinator initialized for node {node_id}")
    
    async def start(self):
        """Start the coordinator."""
        await self.raft.start()
    
    async def stop(self):
        """Stop the coordinator."""
        await self.raft.stop()
    
    def _on_commit(self, entry: LogEntry):
        """Handle committed entry."""
        self._knowledge_ops.append(entry)
        
        # Resolve pending command
        future = self._pending_commands.pop(entry.index, None)
        if future and not future.done():
            future.set_result(entry)
    
    def _on_state_change(self, old_state: NodeState, new_state: NodeState):
        """Handle state change."""
        logger.info(f"State changed: {old_state.value} -> {new_state.value}")
    
    async def submit_knowledge_add(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        timeout: float = 5.0
    ) -> Optional[str]:
        """
        Submit a knowledge add operation to the cluster.
        Waits for replication to majority.
        """
        entry = await self.raft.submit_command(
            LogEntryType.KNOWLEDGE_ADD,
            {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        if entry is None:
            # Not leader, return leader info
            leader = self.raft.get_leader()
            raise NotLeaderException(f"Not leader. Current leader: {leader}")
        
        # Wait for commit
        future = asyncio.Future()
        self._pending_commands[entry.index] = future
        
        try:
            await asyncio.wait_for(future, timeout=timeout)
            return entry.data.get("id")
        except asyncio.TimeoutError:
            self._pending_commands.pop(entry.index, None)
            raise ReplicationTimeoutException("Replication timeout")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "node_id": self.node_id,
            "raft_stats": self.raft.get_stats(),
            "applied_operations": len(self._knowledge_ops)
        }


class NotLeaderException(Exception):
    """Raised when operation is sent to non-leader node."""
    pass


class ReplicationTimeoutException(Exception):
    """Raised when replication times out."""
    pass


__all__ = [
    "NodeState",
    "LogEntryType",
    "LogEntry",
    "NodeInfo",
    "RaftNode",
    "DistributedKnowledgeCoordinator",
    "NotLeaderException",
    "ReplicationTimeoutException"
]
