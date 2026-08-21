# Distributed Coordination and Consensus Specification

> **STATUS: partially implemented.** February-2026 draft spec for Raft/consensus coordination. A real implementation exists at `knowledge_engine/distributed_coordination.py` (`RaftNode`, leader election, log replication). The broader spec (cluster membership, failure detection) is design-only.
> **Last reconciled: 2026-08-20**

## Document Information
- **Version**: 1.0
- **Date**: February 1, 2026
- **Status**: Draft
- **Authors**: OpenEvolve Team

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Consensus Algorithms](#consensus-algorithms)
4. [Raft Implementation](#raft-implementation)
5. [Leader Election](#leader-election)
6. [Log Replication](#log-replication)
7. [Cluster Management](#cluster-management)
8. [Membership Management](#membership-management)
9. [Failure Detection](#failure-detection)
10. [Performance](#performance)
11. [Security](#security)
12. [Monitoring](#monitoring)

## Overview

### Purpose
This document specifies the distributed coordination and consensus architecture for the OpenEvolve-Knowledge Engine ecosystem. It defines how multiple nodes coordinate to maintain consistent state, elect leaders, and replicate logs across the distributed system.

### Goals
- Implement reliable consensus for distributed state management
- Provide leader election and failover mechanisms
- Ensure log replication and consistency
- Support dynamic cluster membership
- Maintain high availability and fault tolerance
- Enable scalable distributed coordination

### Non-Goals
- Specifying internal implementation of individual consensus components
- Defining specific business logic of distributed applications
- Detailing UI components or user interfaces

## Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   OpenEvolve    │    │  Distributed         │    │  Coordination   │
│                 │    │  Coordination        │    │  Infrastructure  │
│  • Controllers  │◄──►│  Layer              │◄──►│  • Consul       │
│  • Evaluators   │    │                     │    │  • etcd         │
│  • Evolution    │    │  • Raft Manager     │    │  • Zookeeper    │
│    Processors   │    │  • Leader Election  │    │  • HashiCorp    │
│  • Databases    │    │  • Log Replicator   │    │    Vault        │
└─────────────────┘    │  • Membership       │    │  • Certificate  │
                       │    Manager          │    │    Authority    │
                       │  • State Machine    │    └─────────────────┘
                       │    Replication      │
                       │  • Cluster          │
                       │    Coordinator       │
                       └──────────────────────┘
                                    ▲
                       ┌──────────────────────┐
                       │  Coordination        │
                       │  Services            │
                       │                     │
                       │  • Service Discovery│
                       │  • Configuration    │
                       │    Management       │
                       │  • Lock Management  │
                       │  • Leader Election  │
                       │  • Health Checking  │
                       └──────────────────────┘
```

### Component Roles
- **Raft Manager**: Manages Raft consensus algorithm
- **Leader Election**: Handles leader election and transitions
- **Log Replicator**: Manages log replication across nodes
- **Membership Manager**: Handles cluster membership changes
- **State Machine**: Applies replicated commands to state
- **Cluster Coordinator**: Coordinates cluster-wide operations

## Consensus Algorithms

### 1. Algorithm Selection
The system supports multiple consensus algorithms:

- **Raft**: For strong consistency and understandability
- **Multi-Paxos**: For high-performance scenarios
- **Fast Paxos**: For reduced round trips
- **EPaxos**: For wide-area networks
- **Chain Replication**: For ordered operations

### 2. Raft Consensus Algorithm
The system implements the Raft consensus algorithm for distributed coordination. Raft is chosen for its simplicity and understandability compared to other consensus algorithms like Paxos.

**Key Properties:**
- **Safety**: Under all circumstances, including network partitions
- **Availability**: System remains available as long as majority of nodes are operational
- **Leader Election**: Ensures at most one leader per term
- **Log Matching**: Ensures logs remain consistent across nodes

### 3. Algorithm States
```python
from enum import Enum

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class ConsensusAlgorithm(Enum):
    RAFT = "raft"
    MULTI_PAXOS = "multi_paxos"
    FAST_PAXOS = "fast_paxos"
    EPAXOS = "epaxos"
    CHAIN_REPLICATION = "chain_replication"
```

### 4. Term Management
```python
class TermManager:
    def __init__(self, config):
        self.current_term = 0
        self.voted_for = None
        self.log = Log(config.log_config)
        self.commit_index = 0
        self.last_applied = 0
        self.peers = config.peers
        self.heartbeat_timeout = config.heartbeat_timeout
        self.election_timeout = config.election_timeout
    
    def increment_term(self):
        self.current_term += 1
        self.voted_for = None
        return self.current_term
    
    def update_term(self, new_term):
        if new_term > self.current_term:
            self.current_term = new_term
            self.voted_for = None
            self.state = NodeState.FOLLOWER
            return True
        return False
```

## Raft Implementation

### 1. Raft Node Structure
```python
import asyncio
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Any

class RaftNode:
    def __init__(self, config):
        self.node_id = config.node_id
        self.address = config.address
        self.peers = config.peers  # List of peer addresses
        self.state = NodeState.FOLLOWER
        self.term_manager = TermManager(config.term_config)
        self.log_replicator = LogReplicator(config.replication_config)
        self.election_manager = ElectionManager(config.election_config)
        self.state_machine = StateMachine(config.state_machine_config)
        self.cluster_manager = ClusterManager(config.cluster_config)
        self.heartbeat_manager = HeartbeatManager(config.heartbeat_config)
        self.metrics_collector = MetricsCollector(config.metrics_config)
        
        # Timers
        self.election_timer = None
        self.heartbeat_timer = None
        
        # Event loop
        self.loop = asyncio.get_event_loop()
    
    async def start(self):
        # Initialize node
        await self.term_manager.load_state()
        await self.log_replicator.initialize()
        await self.state_machine.initialize()
        
        # Start background tasks
        self.election_task = asyncio.create_task(self.run_election_timer())
        self.heartbeat_task = asyncio.create_task(self.run_heartbeat_sender())
        self.replication_task = asyncio.create_task(self.run_log_replication())
    
    async def run_election_timer(self):
        while True:
            # Reset timer when receiving heartbeat
            timeout = self.get_random_election_timeout()
            try:
                await asyncio.wait_for(self.reset_election_event.wait(), timeout=timeout)
                # Timer was reset, continue
                self.reset_election_event.clear()
            except asyncio.TimeoutError:
                # Election timeout occurred
                if self.state == NodeState.FOLLOWER:
                    await self.start_election()
    
    async def start_election(self):
        # Transition to candidate
        self.state = NodeState.CANDIDATE
        current_term = self.term_manager.increment_term()
        self.term_manager.voted_for = self.node_id
        
        # Request votes from peers
        votes_received = 1  # Vote for self
        
        vote_requests = []
        for peer in self.peers:
            vote_request = self.send_request_vote(peer, current_term)
            vote_requests.append(vote_request)
        
        # Wait for responses
        responses = await asyncio.gather(*vote_requests, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, RequestVoteResponse) and response.vote_granted:
                votes_received += 1
        
        # Check if received majority of votes
        if votes_received > len(self.peers) / 2:
            # Become leader
            await self.become_leader()
        else:
            # Remain candidate or become follower
            self.state = NodeState.FOLLOWER
```

### 2. Request Vote RPC
```python
class RequestVoteRPC:
    def __init__(self, config):
        self.node_id = config.node_id
        self.rpc_client = RPCClient(config.rpc_config)
    
    async def send_request_vote(self, peer, term):
        request = {
            "term": term,
            "candidate_id": self.node_id,
            "last_log_index": self.term_manager.log.get_last_index(),
            "last_log_term": self.term_manager.log.get_last_term()
        }
        
        try:
            response = await self.rpc_client.call(
                peer, "request_vote", request
            )
            return RequestVoteResponse.from_dict(response)
        except Exception as e:
            return RequestVoteResponse(
                term=term,
                vote_granted=False,
                error=str(e)
            )
    
    async def handle_request_vote(self, request):
        # Check if term is greater than current term
        if request.term < self.term_manager.current_term:
            return RequestVoteResponse(
                term=self.term_manager.current_term,
                vote_granted=False
            )
        
        # Update term if necessary
        if request.term > self.term_manager.current_term:
            self.term_manager.update_term(request.term)
        
        # Reset election timer
        self.reset_election_timer()
        
        # Check if candidate's log is at least as up-to-date as ours
        is_up_to_date = self.is_log_up_to_date(
            request.last_log_index, request.last_log_term
        )
        
        # Grant vote if not voted yet and candidate's log is up-to-date
        vote_granted = (
            self.term_manager.voted_for is None or
            self.term_manager.voted_for == request.candidate_id
        ) and is_up_to_date
        
        if vote_granted:
            self.term_manager.voted_for = request.candidate_id
        
        return RequestVoteResponse(
            term=self.term_manager.current_term,
            vote_granted=vote_granted
        )
    
    def is_log_up_to_date(self, candidate_last_index, candidate_last_term):
        last_log_term = self.term_manager.log.get_last_term()
        last_log_index = self.term_manager.log.get_last_index()
        
        # If candidate's last log term is greater than ours
        if candidate_last_term > last_log_term:
            return True
        
        # If terms are equal, check index
        if candidate_last_term == last_log_term:
            return candidate_last_index >= last_log_index
        
        return False
```

### 3. Append Entries RPC
```python
class AppendEntriesRPC:
    def __init__(self, config):
        self.node_id = config.node_id
        self.rpc_client = RPCClient(config.rpc_config)
    
    async def send_append_entries(self, peer, entries, prev_log_index, prev_log_term, leader_commit):
        request = {
            "term": self.term_manager.current_term,
            "leader_id": self.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": [entry.to_dict() for entry in entries],
            "leader_commit": leader_commit
        }
        
        try:
            response = await self.rpc_client.call(
                peer, "append_entries", request
            )
            return AppendEntriesResponse.from_dict(response)
        except Exception as e:
            return AppendEntriesResponse(
                term=self.term_manager.current_term,
                success=False,
                error=str(e)
            )
    
    async def handle_append_entries(self, request):
        # Check term
        if request.term < self.term_manager.current_term:
            return AppendEntriesResponse(
                term=self.term_manager.current_term,
                success=False
            )
        
        # Update term if necessary
        if request.term > self.term_manager.current_term:
            self.term_manager.update_term(request.term)
        
        # Reset election timer
        self.reset_election_timer()
        
        # Check if prev log matches
        if not self.log_matches(request.prev_log_index, request.prev_log_term):
            return AppendEntriesResponse(
                term=self.term_manager.current_term,
                success=False
            )
        
        # Append new entries
        for entry in request.entries:
            self.term_manager.log.append_entry(LogEntry.from_dict(entry))
        
        # Update commit index
        if request.leader_commit > self.term_manager.commit_index:
            new_commit_index = min(request.leader_commit, self.term_manager.log.get_last_index())
            self.term_manager.commit_index = new_commit_index
            await self.apply_committed_entries()
        
        return AppendEntriesResponse(
            term=self.term_manager.current_term,
            success=True
        )
    
    def log_matches(self, prev_log_index, prev_log_term):
        if prev_log_index == 0:
            return True
        
        if prev_log_index > self.term_manager.log.get_last_index():
            return False
        
        return self.term_manager.log.get_term(prev_log_index) == prev_log_term
```

## Leader Election

### 1. Election Process
```python
class ElectionManager:
    def __init__(self, config):
        self.node = config.node
        self.timeout_range = config.timeout_range
        self.backoff_factor = config.backoff_factor
        self.max_attempts = config.max_attempts
    
    async def conduct_election(self):
        attempts = 0
        base_timeout = self.timeout_range[0]
        
        while attempts < self.max_attempts:
            # Increment term and vote for self
            current_term = self.node.term_manager.increment_term()
            self.node.term_manager.voted_for = self.node.node_id
            
            # Send vote requests
            votes = await self.request_votes(current_term)
            
            # Check if received majority
            if votes > len(self.node.peers) / 2:
                return True
            
            # Exponential backoff
            base_timeout *= self.backoff_factor
            await asyncio.sleep(base_timeout)
            
            attempts += 1
        
        return False
    
    async def request_votes(self, term):
        votes = 1  # Vote for self
        
        vote_tasks = []
        for peer in self.node.peers:
            task = self.request_vote_from_peer(peer, term)
            vote_tasks.append(task)
        
        responses = await asyncio.gather(*vote_tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, RequestVoteResponse) and response.vote_granted:
                votes += 1
        
        return votes
    
    async def request_vote_from_peer(self, peer, term):
        # Get log info
        last_log_index = self.node.term_manager.log.get_last_index()
        last_log_term = self.node.term_manager.log.get_last_term()
        
        request = {
            "term": term,
            "candidate_id": self.node.node_id,
            "last_log_index": last_log_index,
            "last_log_term": last_log_term
        }
        
        try:
            response = await self.node.rpc_client.call(peer, "request_vote", request)
            return RequestVoteResponse.from_dict(response)
        except Exception as e:
            return RequestVoteResponse(term=term, vote_granted=False, error=str(e))
```

### 2. Leader Transition
```python
class LeaderTransition:
    def __init__(self, config):
        self.node = config.node
        self.transition_manager = TransitionManager(config.transition_config)
    
    async def become_leader(self):
        # Transition to leader state
        self.node.state = NodeState.LEADER
        
        # Initialize leader-specific state
        self.next_index = {peer: self.node.term_manager.log.get_last_index() + 1 for peer in self.node.peers}
        self.match_index = {peer: 0 for peer in self.node.peers}
        
        # Send initial heartbeat
        await self.send_initial_heartbeats()
        
        # Start log replication
        asyncio.create_task(self.replicate_logs())
        
        # Log transition
        await self.transition_manager.log_transition(
            self.node.node_id, "became_leader", self.node.term_manager.current_term
        )
    
    async def send_initial_heartbeats(self):
        # Send empty append entries to all peers
        heartbeat_tasks = []
        for peer in self.node.peers:
            task = self.send_heartbeat(peer)
            heartbeat_tasks.append(task)
        
        await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
    
    async def send_heartbeat(self, peer):
        # Send empty append entries as heartbeat
        prev_log_index = max(0, self.node.term_manager.log.get_last_index())
        prev_log_term = self.node.term_manager.log.get_term(prev_log_index) if prev_log_index > 0 else 0
        
        request = {
            "term": self.node.term_manager.current_term,
            "leader_id": self.node.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": [],
            "leader_commit": self.node.term_manager.commit_index
        }
        
        try:
            response = await self.node.rpc_client.call(peer, "append_entries", request)
            return AppendEntriesResponse.from_dict(response)
        except Exception as e:
            return AppendEntriesResponse(
                term=self.node.term_manager.current_term,
                success=False,
                error=str(e)
            )
```

## Log Replication

### 1. Log Structure
```python
class Log:
    def __init__(self, config):
        self.storage = LogStorage(config.storage_config)
        self.snapshot_manager = SnapshotManager(config.snapshot_config)
        self.compression = CompressionService(config.compression_config)
        self.encryption = EncryptionService(config.encryption_config)
    
    async def append_entry(self, entry):
        # Validate entry
        if not self.validate_entry(entry):
            raise ValueError("Invalid log entry")
        
        # Encrypt if needed
        if self.encryption.enabled:
            entry.command = await self.encryption.encrypt(entry.command)
        
        # Compress if needed
        if self.compression.enabled:
            entry.command = await self.compression.compress(entry.command)
        
        # Store entry
        await self.storage.append(entry)
        
        return entry.index
    
    async def get_entries(self, start_index, end_index=None):
        entries = await self.storage.get_range(start_index, end_index)
        
        # Decompress and decrypt entries
        for entry in entries:
            if self.compression.enabled:
                entry.command = await self.compression.decompress(entry.command)
            
            if self.encryption.enabled:
                entry.command = await self.encryption.decrypt(entry.command)
        
        return entries
    
    def validate_entry(self, entry):
        # Check term consistency
        if entry.term < 0:
            return False
        
        # Check index consistency
        if entry.index <= 0:
            return False
        
        # Check command validity
        if not entry.command or len(entry.command) == 0:
            return False
        
        return True
    
    async def compact(self, snapshot_index):
        # Create snapshot up to snapshot_index
        await self.snapshot_manager.create_snapshot(snapshot_index)
        
        # Remove entries up to snapshot_index
        await self.storage.remove_until(snapshot_index)
```

### 2. Log Replication Process
```python
class LogReplicator:
    def __init__(self, config):
        self.node = config.node
        self.replication_manager = ReplicationManager(config.replication_config)
        self.metrics_collector = MetricsCollector(config.metrics_config)
    
    async def replicate_to_peer(self, peer):
        while self.node.state == NodeState.LEADER:
            # Get next index for this peer
            next_idx = self.node.next_index[peer]
            
            if next_idx <= self.node.term_manager.log.get_last_index():
                # Send entries to peer
                await self.send_entries_to_peer(peer, next_idx)
            else:
                # Send heartbeat
                await self.send_heartbeat_to_peer(peer)
            
            # Wait before next replication attempt
            await asyncio.sleep(config.replication_interval)
    
    async def send_entries_to_peer(self, peer, next_idx):
        # Get entries to send
        prev_log_index = max(0, next_idx - 1)
        prev_log_term = self.node.term_manager.log.get_term(prev_log_index) if prev_log_index > 0 else 0
        
        # Get entries starting from next_idx
        entries = await self.node.term_manager.log.get_entries(next_idx)
        
        request = {
            "term": self.node.term_manager.current_term,
            "leader_id": self.node.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": [entry.to_dict() for entry in entries],
            "leader_commit": self.node.term_manager.commit_index
        }
        
        try:
            response = await self.node.rpc_client.call(peer, "append_entries", request)
            response = AppendEntriesResponse.from_dict(response)
            
            if response.success:
                # Update match index
                self.node.match_index[peer] = next_idx + len(entries) - 1
                self.node.next_index[peer] = self.node.match_index[peer] + 1
                
                # Update commit index
                await self.update_commit_index()
            else:
                # Decrement next index and retry
                self.node.next_index[peer] = max(1, self.node.next_index[peer] - 1)
            
            # Record metrics
            await self.metrics_collector.record_replication_attempt(
                peer, response.success, len(entries)
            )
            
        except Exception as e:
            # Log error and continue
            await self.metrics_collector.record_replication_error(peer, str(e))
    
    async def update_commit_index(self):
        # Update commit index based on match indices
        current_term = self.node.term_manager.current_term
        
        # Get all match indices
        match_indices = list(self.node.match_index.values())
        match_indices.append(self.node.term_manager.log.get_last_index())
        
        # Sort in descending order
        match_indices.sort(reverse=True)
        
        # Find median match index
        majority_index = len(match_indices) // 2
        new_commit_index = match_indices[majority_index]
        
        # Only commit entries from current term
        if new_commit_index > self.node.term_manager.commit_index:
            # Check if entry at new_commit_index is from current term
            entry_term = self.node.term_manager.log.get_term(new_commit_index)
            if entry_term == current_term:
                self.node.term_manager.commit_index = new_commit_index
                await self.apply_committed_entries()
    
    async def apply_committed_entries(self):
        # Apply all entries from last_applied + 1 to commit_index
        start_index = self.node.term_manager.last_applied + 1
        end_index = self.node.term_manager.commit_index
        
        if start_index <= end_index:
            entries = await self.node.term_manager.log.get_entries(start_index, end_index)
            
            for entry in entries:
                await self.node.state_machine.apply(entry.command)
                self.node.term_manager.last_applied = entry.index
```

## Cluster Management

### 1. Cluster Configuration
```python
class ClusterManager:
    def __init__(self, config):
        self.node = config.node
        self.cluster_store = ClusterStore(config.cluster_config)
        self.consensus_client = ConsensusClient(config.consensus_config)
        self.health_checker = HealthChecker(config.health_config)
    
    async def add_node(self, new_node_info):
        # Check if node already exists
        if await self.cluster_store.node_exists(new_node_info.node_id):
            raise ValueError(f"Node {new_node_info.node_id} already exists")
        
        # Verify node health
        if not await self.health_checker.check_node_health(new_node_info):
            raise ValueError(f"Node {new_node_info.node_id} is not healthy")
        
        # Propose membership change through consensus
        membership_change = {
            "type": "add_node",
            "node_info": new_node_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await self.consensus_client.propose_membership_change(membership_change)
        
        if success:
            # Update local membership
            await self.cluster_store.add_node(new_node_info)
            
            # Notify other nodes
            await self.broadcast_membership_update(new_node_info, "added")
        
        return success
    
    async def remove_node(self, node_id):
        # Check if node exists
        if not await self.cluster_store.node_exists(node_id):
            raise ValueError(f"Node {node_id} does not exist")
        
        # Check if node is healthy (for graceful removal)
        node_info = await self.cluster_store.get_node(node_id)
        if await self.health_checker.check_node_health(node_info):
            # Gracefully stop node services
            await self.graceful_node_shutdown(node_id)
        
        # Propose membership change through consensus
        membership_change = {
            "type": "remove_node",
            "node_id": node_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await self.consensus_client.propose_membership_change(membership_change)
        
        if success:
            # Update local membership
            await self.cluster_store.remove_node(node_id)
            
            # Notify other nodes
            await self.broadcast_membership_update({"node_id": node_id}, "removed")
        
        return success
    
    async def broadcast_membership_update(self, node_info, change_type):
        # Broadcast membership update to all nodes
        update_message = {
            "type": "membership_update",
            "node_info": node_info,
            "change_type": change_type,
            "timestamp": datetime.utcnow().isoformat(),
            "sender": self.node.node_id
        }
        
        broadcast_tasks = []
        for peer in self.node.peers:
            task = self.send_membership_update(peer, update_message)
            broadcast_tasks.append(task)
        
        await asyncio.gather(*broadcast_tasks, return_exceptions=True)
    
    async def handle_membership_update(self, update_message):
        # Validate update message
        if not self.validate_membership_update(update_message):
            return False
        
        # Apply membership change locally
        if update_message.change_type == "added":
            await self.cluster_store.add_node(update_message.node_info)
        elif update_message.change_type == "removed":
            await self.cluster_store.remove_node(update_message.node_info.node_id)
        
        # Update node's peer list
        current_peers = await self.cluster_store.get_all_nodes()
        self.node.peers = [node.address for node in current_peers if node.node_id != self.node.node_id]
        
        return True
```

### 2. Joint Consensus for Membership Changes
```python
class JointConsensusManager:
    def __init__(self, config):
        self.node = config.node
        self.old_cluster_config = config.initial_cluster_config
        self.new_cluster_config = config.initial_cluster_config
        self.phase = "stable"  # stable, joint_consensus, transition_complete
    
    async def change_cluster_membership(self, new_nodes, removed_nodes):
        # Enter joint consensus phase
        old_config = self.new_cluster_config.copy()
        new_config = self.calculate_new_config(old_config, new_nodes, removed_nodes)
        
        # Propose joint consensus configuration
        joint_config = {
            "old_config": old_config,
            "new_config": new_config,
            "phase": "joint_consensus"
        }
        
        success = await self.propose_joint_configuration(joint_config)
        
        if success:
            # Wait for joint configuration to be committed
            await self.wait_for_joint_config_commit()
            
            # Propose transition to new configuration
            transition_config = {
                "new_config": new_config,
                "phase": "transition_complete"
            }
            
            success = await self.propose_transition_config(transition_config)
            
            if success:
                # Update local configuration
                self.old_cluster_config = old_config
                self.new_cluster_config = new_config
                self.phase = "stable"
                
                # Update node's peer list
                self.update_peers()
        
        return success
    
    def calculate_new_config(self, old_config, new_nodes, removed_nodes):
        # Calculate new cluster configuration
        new_config = old_config.copy()
        
        # Add new nodes
        for node in new_nodes:
            new_config[node.node_id] = node
        
        # Remove nodes
        for node_id in removed_nodes:
            if node_id in new_config:
                del new_config[node_id]
        
        return new_config
    
    async def propose_joint_configuration(self, joint_config):
        # Propose joint configuration through consensus
        command = {
            "type": "joint_config",
            "config": joint_config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.node.consensus_client.propose_command(command)
    
    async def wait_for_joint_config_commit(self):
        # Wait until joint configuration is committed on majority of both old and new clusters
        timeout = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if joint config is committed on old cluster
            old_cluster_committed = await self.check_config_committed(
                self.old_cluster_config, "joint_consensus"
            )
            
            # Check if joint config is committed on new cluster
            new_cluster_committed = await self.check_config_committed(
                self.new_cluster_config, "joint_consensus"
            )
            
            if old_cluster_committed and new_cluster_committed:
                return True
            
            await asyncio.sleep(0.1)
        
        return False
```

## Failure Detection

### 1. Health Monitoring
```python
class HealthChecker:
    def __init__(self, config):
        self.node = config.node
        self.heartbeat_manager = HeartbeatManager(config.heartbeat_config)
        self.health_checker = HealthChecker(config.health_config)
        self.failure_notifier = FailureNotifier(config.notification_config)
    
    async def monitor_peers(self):
        while True:
            for peer in self.node.peers:
                # Check if heartbeat received recently
                if not await self.heartbeat_manager.received_recent_heartbeat(peer):
                    # Perform health check
                    if not await self.health_checker.check_node_health(peer):
                        # Mark node as failed
                        await self.handle_node_failure(peer)
            
            await asyncio.sleep(config.monitoring_interval)
    
    async def handle_node_failure(self, failed_node):
        # Log failure
        await self.failure_notifier.log_failure(failed_node)
        
        # Check if failed node is leader
        if self.node.state == NodeState.FOLLOWER and self.node.leader_id == failed_node:
            # Transition to follower state
            self.node.state = NodeState.FOLLOWER
            self.node.leader_id = None
        
        # Remove from active peers temporarily
        if failed_node in self.node.peers:
            self.node.failed_peers[failed_node] = datetime.utcnow()
        
        # Attempt to remove from cluster if needed
        await self.attempt_cluster_reconfiguration(failed_node)
    
    async def attempt_cluster_reconfiguration(self, failed_node):
        # Check if cluster still has majority
        active_nodes = len(self.node.peers) - len(self.node.failed_peers)
        total_nodes = len(self.node.peers) + 1  # Include self
        
        if active_nodes < total_nodes / 2:
            # Cluster is in minority, cannot operate normally
            # Wait for nodes to recover or initiate manual intervention
            await self.handle_minority_state()
        else:
            # Cluster can continue operating
            # Optionally remove failed node from cluster
            if await self.should_remove_failed_node(failed_node):
                await self.remove_failed_node_from_cluster(failed_node)
    
    async def should_remove_failed_node(self, failed_node):
        # Determine if failed node should be removed from cluster
        failure_time = self.node.failed_peers.get(failed_node)
        if not failure_time:
            return False
        
        time_since_failure = datetime.utcnow() - failure_time
        return time_since_failure.total_seconds() > config.remove_node_timeout
```

### 2. Recovery Mechanisms
```python
class RecoveryManager:
    def __init__(self, config):
        self.node = config.node
        self.snapshot_manager = SnapshotManager(config.snapshot_config)
        self.log_replicator = LogReplicator(config.replication_config)
        self.state_machine = StateMachine(config.state_machine_config)
    
    async def recover_node(self):
        # Load persisted state
        await self.load_persisted_state()
        
        # Check if node was previously part of cluster
        if await self.was_part_of_cluster():
            # Attempt to rejoin cluster
            await self.attempt_cluster_join()
        else:
            # Node is new, wait for cluster invitation
            await self.wait_for_cluster_invitation()
    
    async def load_persisted_state(self):
        # Load term and vote state
        state = await self.node.persistence.load_state()
        if state:
            self.node.term_manager.current_term = state.get("current_term", 0)
            self.node.term_manager.voted_for = state.get("voted_for", None)
            self.node.commit_index = state.get("commit_index", 0)
            self.node.last_applied = state.get("last_applied", 0)
        
        # Load log
        await self.node.term_manager.log.load_from_storage()
        
        # Load snapshot if available
        snapshot = await self.snapshot_manager.load_latest_snapshot()
        if snapshot:
            await self.state_machine.restore_from_snapshot(snapshot)
            self.node.last_applied = snapshot.last_included_index
    
    async def attempt_cluster_join(self):
        # Try to contact known peers
        for peer in self.node.known_peers:
            try:
                # Request cluster membership information
                cluster_info = await self.request_cluster_info(peer)
                
                if cluster_info:
                    # Join the cluster
                    await self.join_cluster(cluster_info)
                    return True
            except Exception:
                continue
        
        # If unable to join, wait for invitation
        await self.wait_for_cluster_invitation()
    
    async def request_cluster_info(self, peer):
        request = {
            "node_id": self.node.node_id,
            "address": self.node.address,
            "last_known_term": self.node.term_manager.current_term,
            "last_known_index": self.node.term_manager.log.get_last_index()
        }
        
        try:
            response = await self.node.rpc_client.call(peer, "cluster_info", request)
            return ClusterInfo.from_dict(response)
        except Exception:
            return None
    
    async def join_cluster(self, cluster_info):
        # Add cluster nodes to peer list
        self.node.peers = cluster_info.nodes
        self.node.cluster_id = cluster_info.cluster_id
        
        # If log is behind, request snapshot or log entries
        if self.node.term_manager.log.get_last_index() < cluster_info.commit_index:
            await self.request_log_sync(cluster_info.leader)
    
    async def request_log_sync(self, leader):
        # Request snapshot if available
        if await self.snapshot_manager.has_snapshot():
            snapshot = await self.snapshot_manager.get_latest_snapshot()
            await self.request_snapshot_sync(leader, snapshot)
        else:
            # Request log entries
            await self.request_log_entries_sync(leader)
```

## Performance

### 1. Performance Metrics
- **Consensus Latency**: Time to reach consensus on a command
- **Throughput**: Commands processed per second
- **Log Replication Speed**: Entries replicated per second
- **Election Time**: Time to elect a new leader
- **Cluster Recovery Time**: Time to recover from failures

### 2. Performance Targets
- **Consensus Latency**: <10ms for 95% of commands
- **Throughput**: 10,000+ commands/second
- **Log Replication**: 1,000+ entries/second per follower
- **Election Time**: <1 second for leader election
- **Recovery Time**: <30 seconds for node recovery

### 3. Optimization Strategies
- **Pipeline Replication**: Send multiple append entries without waiting for responses
- **Batching**: Group multiple commands in a single RPC
- **Compression**: Compress log entries to reduce network overhead
- **Snapshotting**: Periodically create snapshots to reduce log size
- **Quorum Reads**: Allow reads from followers under certain conditions

### 4. Performance Monitoring
```python
class PerformanceMonitor:
    def __init__(self, config):
        self.metrics_collector = MetricsCollector(config.metrics_config)
        self.performance_analyzer = PerformanceAnalyzer(config.analyzer_config)
        self.scaling_manager = ScalingManager(config.scaling_config)
    
    async def monitor_performance(self):
        while True:
            # Collect metrics
            metrics = await self.collect_performance_metrics()
            
            # Analyze performance
            analysis = await self.performance_analyzer.analyze(metrics)
            
            # Adjust resources if needed
            if analysis.needs_scaling:
                await self.scaling_manager.scale_resources(analysis.recommendation)
            
            # Record metrics
            await self.metrics_collector.record(metrics)
            
            await asyncio.sleep(config.monitoring_interval)
    
    async def collect_performance_metrics(self):
        return {
            "consensus_latency": {
                "p50": self.get_percentile("consensus_time", 50),
                "p95": self.get_percentile("consensus_time", 95),
                "p99": self.get_percentile("consensus_time", 99)
            },
            "throughput": self.get_commands_per_second(),
            "replication_speed": self.get_replication_throughput(),
            "election_time": self.get_election_time(),
            "resource_utilization": {
                "cpu_percent": self.get_cpu_usage(),
                "memory_mb": self.get_memory_usage(),
                "network_io": self.get_network_io()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_percentile(self, metric_name, percentile):
        # Calculate percentile of metric values
        values = self.metrics_collector.get_recent_values(metric_name)
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_commands_per_second(self):
        # Calculate commands processed per second
        recent_commands = self.metrics_collector.get_recent_commands(1000)  # Last 1000ms
        return len(recent_commands) / 1.0  # Per second
```

## Security

### 1. Secure Communication
- **TLS Encryption**: All RPC communication encrypted with TLS
- **Certificate Authentication**: Mutual TLS with certificate validation
- **Message Authentication**: Digital signatures for message integrity
- **Secure Channels**: Isolated communication channels for sensitive data

### 2. Access Control
- **Node Authentication**: Verify node identity before accepting RPCs
- **Authorization**: Control which nodes can participate in consensus
- **Audit Logging**: Log all consensus-related activities
- **Secure Configuration**: Encrypt sensitive configuration data

### 3. Security Measures
```python
SECURITY_MEASURES = {
    "communication": {
        "encryption": "TLS 1.3",
        "certificates": "mutual_authentication_required",
        "signature_algorithm": "RSA-SHA256",
        "key_rotation": "every_90_days"
    },
    "authentication": {
        "node_verification": "required",
        "certificate_validation": "strict",
        "revocation_check": "enabled"
    },
    "authorization": {
        "access_control": "role_based",
        "permission_scopes": ["read", "write", "admin"],
        "cluster_membership": "approved_by_majority"
    },
    "data_protection": {
        "log_encryption": "AES-256-GCM",
        "snapshot_encryption": "AES-256-GCM",
        "key_management": "HSM_based"
    },
    "monitoring": {
        "audit_logging": "mandatory",
        "anomaly_detection": "enabled",
        "security_alerts": "real_time"
    }
}
```

### 4. Secure Node Communication
```python
class SecureRPCClient:
    def __init__(self, config):
        self.tls_context = self.create_tls_context(config.tls_config)
        self.certificate_validator = CertificateValidator(config.cert_config)
        self.message_signer = MessageSigner(config.signing_config)
    
    def create_tls_context(self, tls_config):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_verify_locations(tls_config.ca_cert_path)
        context.load_cert_chain(tls_config.cert_path, tls_config.key_path)
        context.check_hostname = False  # Custom hostname verification
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    
    async def call(self, peer, method, request):
        # Serialize request
        serialized_request = json.dumps(request).encode()
        
        # Sign request
        signature = self.message_signer.sign(serialized_request)
        
        # Create secure connection
        reader, writer = await asyncio.open_connection(
            peer.host, peer.port, ssl=self.tls_context
        )
        
        # Send signed request
        message = {
            "method": method,
            "request": serialized_request.decode(),
            "signature": signature,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        writer.write(json.dumps(message).encode() + b'\n')
        await writer.drain()
        
        # Read response
        response_line = await reader.readline()
        response = json.loads(response_line.decode())
        
        # Verify response signature
        if not self.message_signer.verify(
            response["response"].encode(), 
            response["signature"]
        ):
            raise SecurityError("Invalid response signature")
        
        writer.close()
        await writer.wait_closed()
        
        return json.loads(response["response"])
```

## Monitoring

### 1. Consensus Metrics
```json
{
  "consensus_metrics": {
    "node_id": "string",
    "term": "integer",
    "state": "enum (follower|candidate|leader)",
    "leader_id": "string or null",
    "commit_index": "integer",
    "last_applied": "integer",
    "log_size": "integer",
    "peers_count": "integer",
    "active_peers": "integer",
    "failed_peers": "integer",
    "election_count": "integer",
    "replication_lag": {
      "min": "integer",
      "max": "integer",
      "avg": "integer"
    },
    "consensus_performance": {
      "commands_per_second": "float",
      "consensus_latency_ms": "float",
      "throughput_bps": "float"
    },
    "health_status": {
      "is_healthy": "boolean",
      "last_heartbeat": "ISO 8601 datetime",
      "failure_count": "integer",
      "recovery_attempts": "integer"
    }
  },
  "timestamp": "ISO 8601 datetime",
  "cluster_id": "string"
}
```

### 2. Cluster Health Dashboard
```json
{
  "cluster_health": {
    "cluster_id": "string",
    "nodes": [
      {
        "node_id": "string",
        "address": "string",
        "state": "enum (follower|candidate|leader)",
        "term": "integer",
        "log_index": "integer",
        "commit_index": "integer",
        "health_status": "enum (healthy|unhealthy|recovering)",
        "last_contact": "ISO 8601 datetime",
        "replication_lag": "integer"
      }
    ],
    "leader": "string (node_id)",
    "quorum_size": "integer",
    "active_nodes": "integer",
    "total_nodes": "integer",
    "consistency_level": "enum (strong|eventual|none)",
    "cluster_status": "enum (stable|reconfiguring|degraded|unavailable)",
    "metrics": {
      "avg_consensus_time": "float (ms)",
      "commands_per_second": "float",
      "error_rate": "float (0.0-1.0)",
      "availability": "float (0.0-1.0)"
    }
  }
}
```

### 3. Alerting for Consensus
- **Leader Loss**: Alert when cluster loses leader
- **Split Brain**: Alert when multiple leaders detected
- **Log Divergence**: Alert when logs become inconsistent
- **Election Storm**: Alert when frequent elections occur
- **Performance Degradation**: Alert when consensus performance drops
- **Node Failures**: Alert when nodes become unreachable

### 4. Distributed Tracing
```json
{
  "trace": {
    "trace_id": "string",
    "span_id": "string",
    "parent_span_id": "string or null",
    "operation": "string (request_vote|append_entries|command_apply)",
    "node_id": "string",
    "term": "integer",
    "start_time": "ISO 8601 datetime",
    "end_time": "ISO 8601 datetime",
    "duration_ms": "float",
    "status": "enum (success|failure)",
    "error_message": "string (if failed)",
    "tags": {
      "source_node": "string",
      "target_node": "string",
      "request_type": "string"
    }
  }
}
```

## Appendix

### Glossary
- **Raft**: Consensus algorithm for distributed systems
- **Term**: Logical time period in Raft algorithm
- **Leader**: Node responsible for handling client requests
- **Follower**: Node that receives log entries from leader
- **Candidate**: Node attempting to become leader
- **Log Entry**: Command to be applied to state machine
- **Commit Index**: Highest log entry known to be committed
- **Match Index**: Highest log entry known to be replicated on follower
- **Joint Consensus**: Temporary configuration during membership changes

### References
- Raft Consensus Algorithm Paper
- In Search of an Understandable Consensus Algorithm
- Distributed Systems: Principles and Paradigms
- Consensus: Bridging Theory and Practice

### Change Log
- **v1.0** - Initial specification