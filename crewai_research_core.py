"""
CrewAI Research Core - Implementation of 10 Research Roadmap Features

This module provides the core infrastructure for:
1. Hierarchical Process Support
2. Advanced Delegation Mechanisms
3. Memory-Augmented Research
4. External Tool Orchestration
5. Multi-Modal Support
6. Real-Time Collaboration
7. Research Workflow Templates
8. Automated Literature Search
9. Experiment Tracking
10. Research Report Generation

License: MIT
"""

import json
import logging
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE 1: HIERARCHICAL PROCESS SUPPORT
# =============================================================================

class CrewLevel(Enum):
    """Hierarchy levels for crew management"""
    EXECUTIVE = "executive"  # Strategic decisions
    MANAGER = "manager"      # Coordination and delegation
    LEAD = "lead"           # Team leadership
    WORKER = "worker"       # Task execution
    SPECIALIST = "specialist"  # Domain expert


@dataclass
class HierarchicalTask:
    """Task with hierarchical delegation support"""
    task_id: str
    title: str
    description: str
    level: CrewLevel
    parent_task_id: Optional[str] = None
    sub_tasks: List[str] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None
    status: str = "pending"
    priority: int = 5
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class HierarchicalCrew:
    """
    Hierarchical crew management system with manager-worker delegation.
    
    Implements:
    - Multi-level crew hierarchy
    - Manager agent coordination
    - Task delegation with context
    - Result aggregation from workers
    """
    
    def __init__(
        self,
        name: str = "HierarchicalCrew",
        max_depth: int = 3,
        enable_auto_delegation: bool = True
    ):
        self.name = name
        self.max_depth = max_depth
        self.enable_auto_delegation = enable_auto_delegation
        self.tasks: Dict[str, HierarchicalTask] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.task_tree: Dict[str, List[str]] = {}  # parent -> children
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
    def create_manager_crew(
        self,
        manager_config: Dict[str, Any],
        worker_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a manager-led crew with workers.
        
        Args:
            manager_config: Configuration for manager agent
            worker_configs: List of worker agent configurations
            
        Returns:
            Crew configuration dict
        """
        crew_id = f"crew_{uuid.uuid4().hex[:8]}"
        
        # Register manager
        manager_id = f"manager_{uuid.uuid4().hex[:8]}"
        self.agents[manager_id] = {
            "id": manager_id,
            "role": "manager",
            "level": CrewLevel.MANAGER,
            "config": manager_config,
            "assigned_tasks": [],
            "workers": []
        }
        
        # Register workers
        for worker_config in worker_configs:
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"
            self.agents[worker_id] = {
                "id": worker_id,
                "role": "worker",
                "level": CrewLevel.WORKER,
                "config": worker_config,
                "manager_id": manager_id,
                "assigned_tasks": []
            }
            self.agents[manager_id]["workers"].append(worker_id)
        
        self.logger.info(f"Created manager crew: {crew_id} with 1 manager and {len(worker_configs)} workers")
        
        return {
            "crew_id": crew_id,
            "manager_id": manager_id,
            "worker_ids": self.agents[manager_id]["workers"],
            "level": CrewLevel.MANAGER.value
        }
    
    def delegate_task(
        self,
        task: HierarchicalTask,
        from_agent_id: str,
        to_agent_ids: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate task from one agent to others with context.
        
        Args:
            task: Task to delegate
            from_agent_id: Delegating agent
            to_agent_ids: Target agents
            context: Delegation context
            
        Returns:
            Delegation result
        """
        delegation_id = f"del_{uuid.uuid4().hex[:8]}"
        
        # Update task with delegation info
        task.context["delegation"] = {
            "from": from_agent_id,
            "to": to_agent_ids,
            "delegation_id": delegation_id,
            "context": context or {}
        }
        
        # Create sub-tasks for each target agent
        sub_tasks = []
        for to_agent_id in to_agent_ids:
            sub_task = HierarchicalTask(
                task_id=f"sub_{uuid.uuid4().hex[:8]}",
                title=f"{task.title} [{to_agent_id}]",
                description=task.description,
                level=self._get_agent_level(to_agent_id),
                parent_task_id=task.task_id,
                assigned_agent_id=to_agent_id,
                priority=task.priority
            )
            self.tasks[sub_task.task_id] = sub_task
            sub_tasks.append(sub_task)
            
            # Update agent
            if to_agent_id in self.agents:
                self.agents[to_agent_id]["assigned_tasks"].append(sub_task.task_id)
        
        # Update task tree
        self.task_tree[task.task_id] = [st.task_id for st in sub_tasks]
        
        self.logger.info(f"Delegated task {task.task_id} from {from_agent_id} to {len(to_agent_ids)} agents")
        
        return {
            "delegation_id": delegation_id,
            "task_id": task.task_id,
            "sub_tasks": [st.task_id for st in sub_tasks],
            "status": "delegated"
        }
    
    def aggregate_results(
        self,
        parent_task_id: str,
        aggregation_strategy: str = "consensus"
    ) -> Dict[str, Any]:
        """
        Aggregate results from child tasks.
        
        Args:
            parent_task_id: Parent task to aggregate for
            aggregation_strategy: How to aggregate (consensus, best, merge)
            
        Returns:
            Aggregated result
        """
        if parent_task_id not in self.task_tree:
            return {"error": "No child tasks found"}
        
        child_task_ids = self.task_tree[parent_task_id]
        child_results = []
        
        for child_id in child_task_ids:
            if child_id in self.tasks and self.tasks[child_id].result:
                child_results.append({
                    "task_id": child_id,
                    "agent_id": self.tasks[child_id].assigned_agent_id,
                    "result": self.tasks[child_id].result,
                    "status": self.tasks[child_id].status
                })
        
        # Apply aggregation strategy
        if aggregation_strategy == "consensus":
            aggregated = self._consensus_aggregation(child_results)
        elif aggregation_strategy == "best":
            aggregated = self._best_result_aggregation(child_results)
        elif aggregation_strategy == "merge":
            aggregated = self._merge_aggregation(child_results)
        else:
            aggregated = {"results": child_results}
        
        # Update parent task
        if parent_task_id in self.tasks:
            self.tasks[parent_task_id].result = aggregated
            self.tasks[parent_task_id].status = "completed"
            self.tasks[parent_task_id].completed_at = datetime.now().isoformat()
        
        return {
            "parent_task_id": parent_task_id,
            "aggregation_strategy": aggregation_strategy,
            "child_count": len(child_results),
            "aggregated_result": aggregated
        }
    
    def _get_agent_level(self, agent_id: str) -> CrewLevel:
        """Get hierarchy level of an agent"""
        if agent_id in self.agents:
            return self.agents[agent_id].get("level", CrewLevel.WORKER)
        return CrewLevel.WORKER
    
    def _consensus_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate by finding consensus"""
        if not results:
            return {}
        
        # Simple consensus: most common result
        result_values = [str(r.get("result", "")) for r in results]
        from collections import Counter
        most_common = Counter(result_values).most_common(1)
        
        consensus_value = most_common[0][0] if most_common else result_values[0]
        consensus_count = most_common[0][1] if most_common else 1
        
        return {
            "type": "consensus",
            "value": consensus_value,
            "agreement_count": consensus_count,
            "total_count": len(results),
            "confidence": consensus_count / len(results) if results else 0
        }
    
    def _best_result_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate by selecting best result"""
        if not results:
            return {}
        
        # Select result with highest quality/confidence
        best = max(results, key=lambda r: r.get("result", {}).get("quality", 0.5) if isinstance(r.get("result"), dict) else 0.5)
        
        return {
            "type": "best",
            "value": best.get("result"),
            "selected_from": best.get("agent_id"),
            "all_results_count": len(results)
        }
    
    def _merge_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate by merging all results"""
        merged = {
            "type": "merge",
            "sources": [],
            "combined_data": {}
        }
        
        for r in results:
            merged["sources"].append(r.get("agent_id"))
            result_data = r.get("result", {})
            if isinstance(result_data, dict):
                merged["combined_data"].update(result_data)
        
        return merged
    
    def get_hierarchy_status(self) -> Dict[str, Any]:
        """Get complete hierarchy status"""
        return {
            "crew_name": self.name,
            "max_depth": self.max_depth,
            "total_tasks": len(self.tasks),
            "total_agents": len(self.agents),
            "tasks_by_level": self._count_tasks_by_level(),
            "agent_hierarchy": self._build_agent_hierarchy(),
            "completion_rate": self._calculate_completion_rate()
        }
    
    def _count_tasks_by_level(self) -> Dict[str, int]:
        """Count tasks at each hierarchy level"""
        counts = {}
        for task in self.tasks.values():
            level = task.level.value
            counts[level] = counts.get(level, 0) + 1
        return counts
    
    def _build_agent_hierarchy(self) -> Dict[str, Any]:
        """Build agent hierarchy tree"""
        hierarchy = {"managers": [], "workers": []}
        
        for agent_id, agent in self.agents.items():
            if agent["level"] == CrewLevel.MANAGER:
                hierarchy["managers"].append({
                    "id": agent_id,
                    "workers": agent.get("workers", []),
                    "task_count": len(agent.get("assigned_tasks", []))
                })
            else:
                hierarchy["workers"].append({
                    "id": agent_id,
                    "manager": agent.get("manager_id"),
                    "task_count": len(agent.get("assigned_tasks", []))
                })
        
        return hierarchy
    
    def _calculate_completion_rate(self) -> float:
        """Calculate overall task completion rate"""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        return completed / len(self.tasks)


# =============================================================================
# FEATURE 2: ADVANCED DELEGATION MECHANISMS
# =============================================================================

class DelegationType(Enum):
    """Types of delegation strategies"""
    ROLE_BASED = "role_based"
    SKILL_BASED = "skill_based"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    ESCALATION = "escalation"


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_id: str
    skills: List[str] = field(default_factory=list)
    expertise_domains: List[str] = field(default_factory=list)
    workload: int = 0
    max_workload: int = 5
    role: str = "worker"
    performance_score: float = 0.8
    availability: bool = True


class AdvancedDelegationManager:
    """
    Advanced delegation mechanisms for CrewAI.
    
    Supports:
    - Role-based delegation
    - Skill-based delegation
    - Load-balanced delegation
    - Priority-based delegation
    - Escalation mechanisms
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self.delegation_history: List[Dict[str, Any]] = []
        self.escalation_chains: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, capability: AgentCapability) -> None:
        """Register an agent with the delegation manager"""
        self.agents[capability.agent_id] = capability
        self.performance_metrics[capability.agent_id] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_quality": 0.8,
            "response_time_ms": 0
        }
        self.logger.info(f"Registered agent: {capability.agent_id} with role {capability.role}")
    
    def delegate(
        self,
        task: Dict[str, Any],
        delegation_type: DelegationType = DelegationType.SKILL_BASED,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate task using specified strategy.
        
        Args:
            task: Task to delegate
            delegation_type: Strategy to use
            context: Additional context
            
        Returns:
            Delegation result with selected agent
        """
        if delegation_type == DelegationType.ROLE_BASED:
            selected = self._role_based_delegation(task)
        elif delegation_type == DelegationType.SKILL_BASED:
            selected = self._skill_based_delegation(task)
        elif delegation_type == DelegationType.LOAD_BALANCED:
            selected = self._load_balanced_delegation(task)
        elif delegation_type == DelegationType.PRIORITY_BASED:
            selected = self._priority_based_delegation(task)
        elif delegation_type == DelegationType.ESCALATION:
            selected = self._escalation_delegation(task, context)
        else:
            selected = self._skill_based_delegation(task)
        
        if selected:
            # Update workload
            self.agents[selected].workload += 1
            
            # Record delegation
            self.delegation_history.append({
                "timestamp": datetime.now().isoformat(),
                "task_id": task.get("id"),
                "agent_id": selected,
                "delegation_type": delegation_type.value,
                "context": context
            })
            
            return {
                "success": True,
                "agent_id": selected,
                "delegation_type": delegation_type.value,
                "agent_info": {
                    "role": self.agents[selected].role,
                    "skills": self.agents[selected].skills,
                    "workload": self.agents[selected].workload
                }
            }
        
        return {
            "success": False,
            "error": "No suitable agent found",
            "delegation_type": delegation_type.value
        }
    
    def _role_based_delegation(self, task: Dict[str, Any]) -> Optional[str]:
        """Delegate based on required role"""
        required_role = task.get("required_role", "worker")
        
        candidates = [
            agent_id for agent_id, cap in self.agents.items()
            if cap.role == required_role and cap.availability and cap.workload < cap.max_workload
        ]
        
        if candidates:
            # Select least loaded
            return min(candidates, key=lambda a: self.agents[a].workload)
        return None
    
    def _skill_based_delegation(self, task: Dict[str, Any]) -> Optional[str]:
        """Delegate based on required skills"""
        required_skills = set(task.get("required_skills", []))
        
        best_agent = None
        best_score = 0.0
        
        for agent_id, cap in self.agents.items():
            if not cap.availability or cap.workload >= cap.max_workload:
                continue
            
            agent_skills = set(cap.skills)
            matching_skills = required_skills & agent_skills
            
            if matching_skills:
                score = len(matching_skills) / len(required_skills) if required_skills else 0
                score *= (1 - cap.workload / cap.max_workload)  # Factor in workload
                score *= cap.performance_score  # Factor in performance
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
        
        return best_agent
    
    def _load_balanced_delegation(self, task: Dict[str, Any]) -> Optional[str]:
        """Delegate to least loaded available agent"""
        available = [
            (agent_id, cap) for agent_id, cap in self.agents.items()
            if cap.availability and cap.workload < cap.max_workload
        ]
        
        if available:
            # Sort by workload ratio
            available.sort(key=lambda x: x[1].workload / x[1].max_workload)
            return available[0][0]
        return None
    
    def _priority_based_delegation(self, task: Dict[str, Any]) -> Optional[str]:
        """Delegate based on task priority and agent seniority"""
        priority = task.get("priority", 5)
        
        # High priority tasks go to senior agents
        if priority >= 8:
            candidates = [
                agent_id for agent_id, cap in self.agents.items()
                if cap.role in ["senior", "lead", "manager"] and cap.availability
            ]
        else:
            candidates = [
                agent_id for agent_id, cap in self.agents.items()
                if cap.availability and cap.workload < cap.max_workload
            ]
        
        if candidates:
            # Select by performance score
            return max(candidates, key=lambda a: self.agents[a].performance_score)
        return None
    
    def _escalation_delegation(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Delegate with escalation chain support"""
        previous_agent = context.get("previous_agent") if context else None
        escalation_level = context.get("escalation_level", 0) if context else 0
        
        if previous_agent and previous_agent in self.escalation_chains:
            chain = self.escalation_chains[previous_agent]
            if escalation_level < len(chain):
                next_agent = chain[escalation_level]
                if self.agents.get(next_agent, {}).availability:
                    return next_agent
        
        # Fallback to skill-based
        return self._skill_based_delegation(task)
    
    def set_escalation_chain(self, agent_id: str, chain: List[str]) -> None:
        """Set escalation chain for an agent"""
        self.escalation_chains[agent_id] = chain
        self.logger.info(f"Set escalation chain for {agent_id}: {chain}")
    
    def report_task_completion(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        quality_score: float = 0.8
    ) -> None:
        """Report task completion for performance tracking"""
        if agent_id in self.agents:
            self.agents[agent_id].workload = max(0, self.agents[agent_id].workload - 1)
        
        if agent_id in self.performance_metrics:
            metrics = self.performance_metrics[agent_id]
            if success:
                metrics["tasks_completed"] += 1
            else:
                metrics["tasks_failed"] += 1
            
            # Update average quality
            total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
            metrics["avg_quality"] = (
                (metrics["avg_quality"] * (total_tasks - 1) + quality_score) / total_tasks
            )
    
    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation statistics"""
        return {
            "total_agents": len(self.agents),
            "total_delegations": len(self.delegation_history),
            "agents_by_role": self._count_agents_by_role(),
            "average_workload": self._calculate_average_workload(),
            "performance_summary": self.performance_metrics
        }
    
    def _count_agents_by_role(self) -> Dict[str, int]:
        """Count agents by role"""
        counts = {}
        for cap in self.agents.values():
            counts[cap.role] = counts.get(cap.role, 0) + 1
        return counts
    
    def _calculate_average_workload(self) -> float:
        """Calculate average agent workload"""
        if not self.agents:
            return 0.0
        return sum(cap.workload for cap in self.agents.values()) / len(self.agents)


# =============================================================================
# FEATURE 3: MEMORY-AUGMENTED RESEARCH
# =============================================================================

class MemoryType(Enum):
    """Types of memory in the research system"""
    CONVERSATION = "conversation"  # Dialogue history
    ENTITY = "entity"              # Extracted entities
    CONTEXTUAL = "contextual"      # Context information
    LONG_TERM = "long_term"        # Persistent knowledge
    WORKING = "working"            # Temporary working memory


@dataclass
class MemoryEntry:
    """Single memory entry"""
    entry_id: str
    memory_type: MemoryType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None


class MemoryAugmentedResearch:
    """
    Memory-augmented research system.
    
    Provides:
    - Conversation memory
    - Entity memory
    - Contextual memory
    - Long-term knowledge storage
    - Memory retrieval optimization
    """
    
    def __init__(
        self,
        max_entries_per_type: int = 1000,
        enable_persistence: bool = True,
        storage_dir: str = "./research_memory"
    ):
        self.max_entries = max_entries_per_type
        self.enable_persistence = enable_persistence
        self.storage_dir = Path(storage_dir)
        
        if self.enable_persistence:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory stores by type
        self.memories: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            mt: {} for mt in MemoryType
        }
        
        # Entity index for fast lookup
        self.entity_index: Dict[str, List[str]] = {}
        
        # Semantic index (simplified - would use embeddings in production)
        self.semantic_index: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(__name__)
        
        if self.enable_persistence:
            self._load_memories()
    
    def store(
        self,
        content: Any,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """
        Store content in memory.
        
        Args:
            content: Content to store
            memory_type: Type of memory
            metadata: Additional metadata
            importance: Importance score (0-1)
            
        Returns:
            Entry ID
        """
        entry_id = f"mem_{uuid.uuid4().hex[:12]}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance
        )
        
        self.memories[memory_type][entry_id] = entry
        
        # Update indices
        self._update_indices(entry)
        
        # Manage size
        self._enforce_size_limit(memory_type)
        
        # Persist if enabled
        if self.enable_persistence:
            self._persist_memory(entry)
        
        return entry_id
    
    def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
        min_relevance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Search query
            memory_type: Optional specific memory type
            top_k: Number of results to return
            min_relevance: Minimum relevance score
            
        Returns:
            List of relevant memory entries
        """
        results = []
        
        # Determine which memory types to search
        types_to_search = [memory_type] if memory_type else list(MemoryType)
        
        for mt in types_to_search:
            if mt is None:
                continue
                
            for entry in self.memories[mt].values():
                relevance = self._calculate_relevance(query, entry)
                if relevance >= min_relevance:
                    # Update access stats
                    entry.access_count += 1
                    entry.last_accessed = datetime.now().isoformat()
                    
                    results.append({
                        "entry_id": entry.entry_id,
                        "memory_type": entry.memory_type.value,
                        "content": entry.content,
                        "relevance": relevance,
                        "importance": entry.importance,
                        "timestamp": entry.timestamp,
                        "metadata": entry.metadata
                    })
        
        # Sort by combined score
        results.sort(key=lambda x: x["relevance"] * 0.6 + x["importance"] * 0.4, reverse=True)
        
        return results[:top_k]
    
    def retrieve_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session"""
        entries = [
            entry for entry in self.memories[MemoryType.CONVERSATION].values()
            if entry.metadata.get("session_id") == session_id
        ]
        
        entries.sort(key=lambda e: e.timestamp)
        
        return [
            {
                "entry_id": e.entry_id,
                "content": e.content,
                "timestamp": e.timestamp,
                "role": e.metadata.get("role", "unknown")
            }
            for e in entries[-limit:]
        ]
    
    def retrieve_entities(
        self,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve stored entities"""
        entries = self.memories[MemoryType.ENTITY].values()
        
        if entity_type:
            entries = [e for e in entries if e.metadata.get("entity_type") == entity_type]
        
        return [
            {
                "entry_id": e.entry_id,
                "entity": e.content,
                "type": e.metadata.get("entity_type"),
                "mentions": e.metadata.get("mention_count", 0)
            }
            for e in entries
        ]
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate and compress memories"""
        consolidation_report = {
            "total_before": sum(len(m) for m in self.memories.values()),
            "consolidated": 0,
            "removed": 0
        }
        
        for memory_type in MemoryType:
            memories_list = list(self.memories[memory_type].values())
            
            # Remove low-importance, rarely accessed memories
            to_remove = [
                e.entry_id for e in memories_list
                if e.importance < 0.2 and e.access_count < 2
            ]
            
            for entry_id in to_remove:
                del self.memories[memory_type][entry_id]
                consolidation_report["removed"] += 1
        
        consolidation_report["total_after"] = sum(len(m) for m in self.memories.values())
        
        return consolidation_report
    
    def _update_indices(self, entry: MemoryEntry) -> None:
        """Update search indices for an entry"""
        # Entity extraction (simplified)
        if isinstance(entry.content, str):
            words = entry.content.lower().split()
            for word in words:
                if len(word) > 3:  # Simple filter for meaningful words
                    if word not in self.semantic_index:
                        self.semantic_index[word] = []
                    self.semantic_index[word].append(entry.entry_id)
    
    def _calculate_relevance(self, query: str, entry: MemoryEntry) -> float:
        """Calculate relevance score between query and entry"""
        query_words = set(query.lower().split())
        
        if isinstance(entry.content, str):
            content_words = set(entry.content.lower().split())
            overlap = query_words & content_words
            
            if not query_words:
                return 0.0
            
            return len(overlap) / len(query_words)
        
        # For non-string content, check metadata
        metadata_str = str(entry.metadata).lower()
        metadata_words = set(metadata_str.split())
        overlap = query_words & metadata_words
        
        return len(overlap) / len(query_words) * 0.5 if query_words else 0.0
    
    def _enforce_size_limit(self, memory_type: MemoryType) -> None:
        """Enforce size limit for a memory type"""
        memories = self.memories[memory_type]
        
        if len(memories) > self.max_entries:
            # Remove oldest, least important entries
            sorted_entries = sorted(
                memories.values(),
                key=lambda e: (e.importance, e.timestamp)
            )
            
            to_remove = len(memories) - self.max_entries
            for entry in sorted_entries[:to_remove]:
                del memories[entry.entry_id]
    
    def _persist_memory(self, entry: MemoryEntry) -> None:
        """Persist memory to disk"""
        try:
            file_path = self.storage_dir / f"{entry.entry_id}.json"
            with open(file_path, 'w') as f:
                json.dump({
                    "entry_id": entry.entry_id,
                    "memory_type": entry.memory_type.value,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp,
                    "importance": entry.importance,
                    "access_count": entry.access_count
                }, f, indent=2)
        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to persist memory: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk"""
        try:
            for file_path in self.storage_dir.glob("mem_*.json"):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    entry = MemoryEntry(
                        entry_id=data["entry_id"],
                        memory_type=MemoryType(data["memory_type"]),
                        content=data["content"],
                        metadata=data.get("metadata", {}),
                        timestamp=data["timestamp"],
                        importance=data.get("importance", 0.5),
                        access_count=data.get("access_count", 0)
                    )
                    
                    self.memories[entry.memory_type][entry.entry_id] = entry
                    self._update_indices(entry)
                    
        except (OSError, IOError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load memories: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hierarchical_crew(
    name: str = "ResearchCrew",
    max_depth: int = 3
) -> HierarchicalCrew:
    """Factory function to create hierarchical crew"""
    return HierarchicalCrew(name=name, max_depth=max_depth)


def create_delegation_manager() -> AdvancedDelegationManager:
    """Factory function to create delegation manager"""
    return AdvancedDelegationManager()


def create_memory_system(
    storage_dir: str = "./research_memory"
) -> MemoryAugmentedResearch:
    """Factory function to create memory system"""
    return MemoryAugmentedResearch(storage_dir=storage_dir)
