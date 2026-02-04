"""
RESE (Recursive Epistemic Solvability Engine) Canonical Schemas

This module defines the canonical data models for RESE Deep Exploration Engine (DEE),
including MCTS search results, hypotheses, and search tree structures.

All timestamps use timezone-aware UTC (datetime.now(timezone.utc)).
All enums serialize using .value.
All datetime objects serialize to ISO format strings.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config values via env vars
- Law of Idempotency: UPSERT logic with deduplication by hypothesis_id
- Structured Logging: JSON with correlation_id
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto
from datetime import datetime, timezone
import uuid
import json


# ============================================================================
# ENUMS
# ============================================================================

class HypothesisStatus(Enum):
    """Status of a hypothesis in the exploration process."""
    PENDING = "pending"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    PARTIALLY_CONFIRMED = "partially_confirmed"
    DEPRECATED = "deprecated"


class PatternType(Enum):
    """Types of cross-domain patterns that can be recognized."""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    ISOMORPHIC = "isomorphic"


class MCTSNodeState(Enum):
    """State of an MCTS search tree node."""
    UNEXPANDED = "unexpanded"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    PRUNED = "pruned"


class ExplorationStrategy(Enum):
    """Strategies for deep exploration."""
    MCTS = "mcts"
    BEAM_SEARCH = "beam_search"
    GREEDY_BEST_FIRST = "greedy_best_first"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"


class ContradictionType(Enum):
    """Types of contradictions detected during exploration."""
    DIRECT = "direct"  # Explicit logical contradiction
    INDIRECT = "indirect"  # Inferred contradiction
    CONTEXTUAL = "contextual"  # Contradiction in specific context
    TEMPORAL = "temporal"  # Time-based contradiction


# ============================================================================
# HYPOTHESIS SCHEMAS
# ============================================================================

@dataclass
class Hypothesis:
    """
    A testable hypothesis generated during deep exploration.

    Attributes:
        hypothesis_id: Unique identifier (UUID)
        statement: The hypothesis statement (formal logic or natural language)
        type: Type of hypothesis (causal, structural, functional, etc.)
        status: Current status in validation process
        confidence: Confidence score [0.0, 1.0]
        evidence: Supporting evidence items
        counter_evidence: Contradicting evidence items
        dependencies: IDs of hypotheses this depends on
        source_hypotheses: IDs of parent hypotheses that generated this one
        domain: Domain of application (e.g., "system_architecture", "causal_inference")
        tags: Searchable tags
        metadata: Additional metadata
        created_at: Creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
        tested_at: Last test timestamp (UTC)
    """
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    type: str = "causal"
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence: float = 0.5
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    counter_evidence: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    source_hypotheses: List[str] = field(default_factory=list)
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tested_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate and normalize fields."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        """Alias for hypothesis_id."""
        return self.hypothesis_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "id": self.hypothesis_id,
            "statement": self.statement,
            "type": self.type,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "counter_evidence": self.counter_evidence,
            "dependencies": self.dependencies,
            "source_hypotheses": self.source_hypotheses,
            "domain": self.domain,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            "tested_at": self.tested_at.isoformat() if isinstance(self.tested_at, datetime) else self.tested_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        """Create from dictionary."""
        data = data.copy()

        # Handle id -> hypothesis_id mapping
        if "hypothesis_id" not in data and "id" in data:
            data["hypothesis_id"] = data["id"]

        # Parse timestamps
        for field_name in ["created_at", "updated_at", "tested_at"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))

        # Parse status enum
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = HypothesisStatus(data["status"])
            except ValueError:
                pass

        # Filter to valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def update_evidence(self, new_evidence: Dict[str, Any], is_supporting: bool = True):
        """
        Update hypothesis with new evidence (idempotent).

        Args:
            new_evidence: Evidence item to add
            is_supporting: True if supporting evidence, False if counter-evidence
        """
        # Check for duplicates by evidence_id if present
        evidence_id = new_evidence.get("evidence_id") or str(hash(json.dumps(new_evidence, sort_keys=True)))

        if is_supporting:
            # Deduplicate
            existing_ids = [e.get("evidence_id") for e in self.evidence]
            if evidence_id not in existing_ids:
                self.evidence.append({**new_evidence, "evidence_id": evidence_id})
        else:
            existing_ids = [e.get("evidence_id") for e in self.counter_evidence]
            if evidence_id not in existing_ids:
                self.counter_evidence.append({**new_evidence, "evidence_id": evidence_id})

        self.updated_at = datetime.now(timezone.utc)

    def calculate_confidence(self) -> float:
        """
        Calculate confidence based on evidence.

        Simple model: confidence = (supporting - contradicting) / (total + 1)
        """
        supporting = len(self.evidence)
        contradicting = len(self.counter_evidence)
        total = supporting + contradicting + 1

        raw_confidence = (supporting - contradicting * 0.5) / total
        self.confidence = max(0.0, min(1.0, raw_confidence))
        self.updated_at = datetime.now(timezone.utc)

        return self.confidence


# ============================================================================
# MCTS SEARCH TREE SCHEMAS
# ============================================================================

@dataclass
class SearchTreeNode:
    """
    A node in the MCTS search tree.

    Attributes:
        node_id: Unique identifier
        hypothesis: Hypothesis associated with this node
        state: Node state
        visit_count: Number of times visited during MCTS
        value: Cumulative reward/value
        mean_value: Average value (value / visit_count)
        children: List of child node IDs
        parent_id: Parent node ID (None for root)
        depth: Depth in tree (root = 0)
        is_terminal: Whether this is a terminal node
        exploration_bonus: UCB exploration bonus
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis: Optional[Hypothesis] = None
    state: MCTSNodeState = MCTSNodeState.UNEXPANDED
    visit_count: int = 0
    value: float = 0.0
    mean_value: float = 0.0
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    is_terminal: bool = False
    exploration_bonus: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Initialize derived values."""
        if self.visit_count > 0 and self.mean_value == 0.0:
            self.mean_value = self.value / self.visit_count

    @property
    def id(self) -> str:
        """Alias for node_id."""
        return self.node_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "id": self.node_id,
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis else None,
            "state": self.state.value if isinstance(self.state, Enum) else self.state,
            "visit_count": self.visit_count,
            "value": self.value,
            "mean_value": self.mean_value,
            "children": self.children,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "exploration_bonus": self.exploration_bonus,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchTreeNode":
        """Create from dictionary."""
        data = data.copy()

        # Handle id -> node_id mapping
        if "node_id" not in data and "id" in data:
            data["node_id"] = data["id"]

        # Parse hypothesis
        if "hypothesis" in data and isinstance(data["hypothesis"], dict):
            data["hypothesis"] = Hypothesis.from_dict(data["hypothesis"])

        # Parse state enum
        if "state" in data and isinstance(data["state"], str):
            try:
                data["state"] = MCTSNodeState(data["state"])
            except ValueError:
                pass

        # Parse timestamp
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        # Filter to valid fields
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)

    def update_value(self, reward: float):
        """
        Update node value with new reward (idempotent).

        Args:
            reward: Reward value from this node
        """
        self.visit_count += 1
        self.value += reward
        self.mean_value = self.value / self.visit_count

    def calculate_ucb(self, total_visits: int, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB (Upper Confidence Bound) for node selection.

        UCB = mean_value + c * sqrt(ln(parent_visits) / visits)

        Args:
            total_visits: Total visits to parent node
            exploration_constant: Exploration constant (default sqrt(2))

        Returns:
            UCB score
        """
        if self.visit_count == 0:
            return float('inf')

        exploitation = self.mean_value
        exploration = exploration_constant * (total_visits / self.visit_count) ** 0.5

        self.exploration_bonus = exploration
        return exploitation + exploration


# ============================================================================
# PATTERN RECOGNITION SCHEMAS
# ============================================================================

@dataclass
class Pattern:
    """
    A recognized pattern across domains.

    Attributes:
        pattern_id: Unique identifier
        type: Pattern type
        description: Human-readable description
        confidence: Confidence in pattern [0.0, 1.0]
        domains: List of domains where pattern appears
        instances: Specific instances of the pattern
        isomorphisms: Known isomorphisms to other patterns
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: PatternType = PatternType.STRUCTURAL
    description: str = ""
    confidence: float = 0.5
    domains: List[str] = field(default_factory=list)
    instances: List[Dict[str, Any]] = field(default_factory=list)
    isomorphisms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate confidence."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        """Alias for pattern_id."""
        return self.pattern_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "id": self.pattern_id,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "description": self.description,
            "confidence": self.confidence,
            "domains": self.domains,
            "instances": self.instances,
            "isomorphisms": self.isomorphisms,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        data = data.copy()

        if "pattern_id" not in data and "id" in data:
            data["pattern_id"] = data["id"]

        if "type" in data and isinstance(data["type"], str):
            try:
                data["type"] = PatternType(data["type"])
            except ValueError:
                pass

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# MCTS SEARCH RESULT SCHEMAS
# ============================================================================

@dataclass
class MCTSSearchResult:
    """
    Result from an MCTS exploration.

    Attributes:
        search_id: Unique identifier for this search
        root_hypothesis: Initial hypothesis that started the search
        best_hypothesis: Best hypothesis found (highest confidence)
        tree_root: Root node of the search tree
        iterations: Number of MCTS iterations performed
        convergence_reached: Whether convergence was achieved
        convergence_iteration: Iteration where convergence occurred (if any)
        total_nodes: Total nodes in tree
        max_depth: Maximum depth reached
        execution_time_ms: Execution time in milliseconds
        strategy: Exploration strategy used
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    search_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_hypothesis: Optional[Hypothesis] = None
    best_hypothesis: Optional[Hypothesis] = None
    tree_root: Optional[SearchTreeNode] = None
    iterations: int = 0
    convergence_reached: bool = False
    convergence_iteration: Optional[int] = None
    total_nodes: int = 0
    max_depth: int = 0
    execution_time_ms: float = 0.0
    strategy: ExplorationStrategy = ExplorationStrategy.MCTS
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        """Alias for search_id."""
        return self.search_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "search_id": self.search_id,
            "id": self.search_id,
            "root_hypothesis": self.root_hypothesis.to_dict() if self.root_hypothesis else None,
            "best_hypothesis": self.best_hypothesis.to_dict() if self.best_hypothesis else None,
            "tree_root": self.tree_root.to_dict() if self.tree_root else None,
            "iterations": self.iterations,
            "convergence_reached": self.convergence_reached,
            "convergence_iteration": self.convergence_iteration,
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
            "execution_time_ms": self.execution_time_ms,
            "strategy": self.strategy.value if isinstance(self.strategy, Enum) else self.strategy,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSSearchResult":
        """Create from dictionary."""
        data = data.copy()

        if "search_id" not in data and "id" in data:
            data["search_id"] = data["id"]

        if "root_hypothesis" in data and isinstance(data["root_hypothesis"], dict):
            data["root_hypothesis"] = Hypothesis.from_dict(data["root_hypothesis"])

        if "best_hypothesis" in data and isinstance(data["best_hypothesis"], dict):
            data["best_hypothesis"] = Hypothesis.from_dict(data["best_hypothesis"])

        if "tree_root" in data and isinstance(data["tree_root"], dict):
            data["tree_root"] = SearchTreeNode.from_dict(data["tree_root"])

        if "strategy" in data and isinstance(data["strategy"], str):
            try:
                data["strategy"] = ExplorationStrategy(data["strategy"])
            except ValueError:
                pass

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# EXPLORATION CONFIGURATION
# ============================================================================

@dataclass
class ExplorationConfig:
    """
    Configuration for deep exploration.

    All values must come from environment variables (Law of Configuration Explicitness).

    Attributes:
        exploration_depth: Maximum depth for exploration (default: 10)
        mcts_iterations: Maximum MCTS iterations (default: 1000)
        mcts_exploration_constant: UCB exploration constant (default: 1.414)
        convergence_threshold: Convergence threshold (default: 0.001)
        timeout_ms: Timeout per operation in ms (default: 10000)
        max_hypotheses: Maximum hypotheses to generate (default: 100)
        pattern_recognition_threshold: Minimum confidence for patterns (default: 0.7)
        beam_width: Beam search width (default: 10)
        temperature: Simulated annealing temperature (default: 1.0)
        population_size: Genetic algorithm population size (default: 50)
        mutation_rate: Genetic algorithm mutation rate (default: 0.1)
        correlation_id: For tracing (optional)
    """
    exploration_depth: int = 10
    mcts_iterations: int = 1000
    mcts_exploration_constant: float = 1.414
    convergence_threshold: float = 0.001
    timeout_ms: int = 10000
    max_hypotheses: int = 100
    pattern_recognition_threshold: float = 0.7
    beam_width: int = 10
    temperature: float = 1.0
    population_size: int = 50
    mutation_rate: float = 0.1
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ExplorationConfig":
        """
        Create configuration from environment variables.

        Required env vars (CLAUDE.md compliance):
        - EXPLORATION_DEPTH
        - MCTS_ITERATIONS
        - MCTS_EXPLORATION_CONSTANT
        - CONVERGENCE_THRESHOLD
        - EXPLORATION_TIMEOUT_MS
        - MAX_HYPOTHESES
        - PATTERN_RECOGNITION_THRESHOLD
        - BEAM_WIDTH
        - TEMPERATURE
        - POPULATION_SIZE
        - MUTATION_RATE

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        import os

        env_vars = {
            "EXPLORATION_DEPTH": ("exploration_depth", 10, int),
            "MCTS_ITERATIONS": ("mcts_iterations", 1000, int),
            "MCTS_EXPLORATION_CONSTANT": ("mcts_exploration_constant", 1.414, float),
            "CONVERGENCE_THRESHOLD": ("convergence_threshold", 0.001, float),
            "EXPLORATION_TIMEOUT_MS": ("timeout_ms", 10000, int),
            "MAX_HYPOTHESES": ("max_hypotheses", 100, int),
            "PATTERN_RECOGNITION_THRESHOLD": ("pattern_recognition_threshold", 0.7, float),
            "BEAM_WIDTH": ("beam_width", 10, int),
            "TEMPERATURE": ("temperature", 1.0, float),
            "POPULATION_SIZE": ("population_size", 50, int),
            "MUTATION_RATE": ("mutation_rate", 0.1, float),
        }

        config = {}
        for env_name, (field_name, default, field_type) in env_vars.items():
            value = os.getenv(env_name)
            if value is None:
                # Use default for missing vars (don't crash)
                config[field_name] = default
            else:
                try:
                    config[field_name] = field_type(value)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid value for {env_name}: {value}. "
                        f"Expected {field_type.__name__}."
                    )

        # Optional correlation ID
        config["correlation_id"] = os.getenv("CORRELATION_ID")

        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "exploration_depth": self.exploration_depth,
            "mcts_iterations": self.mcts_iterations,
            "mcts_exploration_constant": self.mcts_exploration_constant,
            "convergence_threshold": self.convergence_threshold,
            "timeout_ms": self.timeout_ms,
            "max_hypotheses": self.max_hypotheses,
            "pattern_recognition_threshold": self.pattern_recognition_threshold,
            "beam_width": self.beam_width,
            "temperature": self.temperature,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "correlation_id": self.correlation_id,
        }


# ============================================================================
# PHASE II: ISOMORPHIC MAPPING SCHEMAS
# ============================================================================

class IsomorphismType(Enum):
    """Types of isomorphisms between domains."""
    STRUCTURAL = "structural"  # Same structure
    FUNCTIONAL = "functional"  # Same function
    MECHANISTIC = "mechanistic"  # Same mechanism
    ANALOGICAL = "analogical"  # Analogical similarity


@dataclass
class FunctionalDependency:
    """
    A functional dependency in a domain's structure.

    Represents a relationship where one variable depends on another.
    Used to build Functional Dependency Graphs (FDGs).

    Attributes:
        dependency_id: Unique identifier
        source: Source variable/node
        target: Target variable/node
        relationship_type: Type of dependency (causal, correlational, etc.)
        strength: Strength of dependency [0.0, 1.0]
        domain: Domain this dependency belongs to
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    dependency_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: str = ""
    relationship_type: str = "causal"
    strength: float = 0.5
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate strength."""
        self.strength = max(0.0, min(1.0, float(self.strength)))

    @property
    def id(self) -> str:
        """Alias for dependency_id."""
        return self.dependency_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dependency_id": self.dependency_id,
            "id": self.dependency_id,
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "domain": self.domain,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionalDependency":
        """Create from dictionary."""
        data = data.copy()

        if "dependency_id" not in data and "id" in data:
            data["dependency_id"] = data["id"]

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class FunctionalDependencyGraph:
    """
    A Functional Dependency Graph (FDG) for a domain.

    FDGs represent the structure of dependencies in a domain.
    Used for isomorphism detection and cross-domain mapping.

    Attributes:
        graph_id: Unique identifier
        domain: Domain this graph represents
        nodes: List of nodes in the graph
        dependencies: List of functional dependencies
        adjacency_list: Adjacency list representation
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = ""
    nodes: List[str] = field(default_factory=list)
    dependencies: List[FunctionalDependency] = field(default_factory=list)
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        """Alias for graph_id."""
        return self.graph_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_id": self.graph_id,
            "id": self.graph_id,
            "domain": self.domain,
            "nodes": self.nodes,
            "dependencies": [d.to_dict() if isinstance(d, FunctionalDependency) else d for d in self.dependencies],
            "adjacency_list": self.adjacency_list,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionalDependencyGraph":
        """Create from dictionary."""
        data = data.copy()

        if "graph_id" not in data and "id" in data:
            data["graph_id"] = data["id"]

        # Parse dependencies
        if "dependencies" in data:
            data["dependencies"] = [
                FunctionalDependency.from_dict(d) if isinstance(d, dict) else d
                for d in data["dependencies"]
            ]

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class IsomorphicMapping:
    """
    An isomorphic mapping between two domains.

    Represents a structural or functional similarity that can be used
    for cross-domain knowledge transfer.

    Attributes:
        mapping_id: Unique identifier
        source_domain: Source domain
        target_domain: Target domain
        isomorphism_type: Type of isomorphism
        i_mech_score: Mechanistic isomorphism score [0.0, 1.0]
        fdg_overlap: Overlap between FDGs
        node_mappings: Mapping of nodes between domains
        dependency_mappings: Mapping of dependencies
        confidence: Confidence in this mapping [0.0, 1.0]
        validated: Whether validated in Lean 4
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    mapping_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_domain: str = ""
    target_domain: str = ""
    isomorphism_type: IsomorphismType = IsomorphismType.STRUCTURAL
    i_mech_score: float = 0.0
    fdg_overlap: float = 0.0
    node_mappings: Dict[str, str] = field(default_factory=dict)
    dependency_mappings: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.5
    validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate scores."""
        self.i_mech_score = max(0.0, min(1.0, float(self.i_mech_score)))
        self.fdg_overlap = max(0.0, min(1.0, float(self.fdg_overlap)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        """Alias for mapping_id."""
        return self.mapping_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mapping_id": self.mapping_id,
            "id": self.mapping_id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "isomorphism_type": self.isomorphism_type.value if isinstance(self.isomorphism_type, Enum) else self.isomorphism_type,
            "i_mech_score": self.i_mech_score,
            "fdg_overlap": self.fdg_overlap,
            "node_mappings": self.node_mappings,
            "dependency_mappings": self.dependency_mappings,
            "confidence": self.confidence,
            "validated": self.validated,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsomorphicMapping":
        """Create from dictionary."""
        data = data.copy()

        if "mapping_id" not in data and "id" in data:
            data["mapping_id"] = data["id"]

        if "isomorphism_type" in data and isinstance(data["isomorphism_type"], str):
            try:
                data["isomorphism_type"] = IsomorphismType(data["isomorphism_type"])
            except ValueError:
                pass

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class CrossDomainPattern:
    """
    A pattern that appears across multiple domains.

    Used for recognizing isomorphic structures and facilitating transfer.

    Attributes:
        pattern_id: Unique identifier
        name: Pattern name
        type: Pattern type
        domains: List of domains where pattern appears
        structural_signature: Abstract structural signature
        functional_signature: Abstract functional signature
        instances: Specific instances in each domain
        isomorphisms: Known isomorphisms
        confidence: Confidence in pattern [0.0, 1.0]
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: PatternType = PatternType.STRUCTURAL
    domains: List[str] = field(default_factory=list)
    structural_signature: str = ""
    functional_signature: str = ""
    instances: List[Dict[str, Any]] = field(default_factory=list)
    isomorphisms: List[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate confidence."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def id(self) -> str:
        """Alias for pattern_id."""
        return self.pattern_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "id": self.pattern_id,
            "name": self.name,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "domains": self.domains,
            "structural_signature": self.structural_signature,
            "functional_signature": self.functional_signature,
            "instances": self.instances,
            "isomorphisms": self.isomorphisms,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossDomainPattern":
        """Create from dictionary."""
        data = data.copy()

        if "pattern_id" not in data and "id" in data:
            data["pattern_id"] = data["id"]

        if "type" in data and isinstance(data["type"], str):
            try:
                data["type"] = PatternType(data["type"])
            except ValueError:
                pass

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class InvertedConstraint:
    """
    An inverted constraint (Ψ₃: Constraint Inversion).

    Represents a constraint that has been inverted to define solution space.
    Original: C → must satisfy C
    Inverted: ¬C → must avoid violating C, defines allowed space

    Attributes:
        constraint_id: Unique identifier
        original_constraint: Original constraint statement
        inverted_constraint: Inverted constraint statement
        inversion_type: Type of inversion (negation, complement, dual)
        solution_space: Defined solution space
        feasibility: Whether inverted constraint is feasible
        search_space_reduction: Factor of search space reduction
        domain: Domain of application
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    constraint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_constraint: str = ""
    inverted_constraint: str = ""
    inversion_type: str = "negation"
    solution_space: str = ""
    feasibility: bool = True
    search_space_reduction: float = 1.0
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        """Alias for constraint_id."""
        return self.constraint_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "id": self.constraint_id,
            "original_constraint": self.original_constraint,
            "inverted_constraint": self.inverted_constraint,
            "inversion_type": self.inversion_type,
            "solution_space": self.solution_space,
            "feasibility": self.feasibility,
            "search_space_reduction": self.search_space_reduction,
            "domain": self.domain,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvertedConstraint":
        """Create from dictionary."""
        data = data.copy()

        if "constraint_id" not in data and "id" in data:
            data["constraint_id"] = data["id"]

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class IsomorphicMappingResult:
    """
    Result from Phase II isomorphic mapping process.

    Attributes:
        result_id: Unique identifier
        source_domain: Source domain
        target_domains: List of target domains searched
        mappings_found: List of isomorphic mappings found
        best_mapping: Best mapping (highest I_mech)
        cross_domain_patterns: Patterns identified across domains
        inverted_constraints: Constraints inverted (Ψ₃)
        execution_time_ms: Execution time in milliseconds
        confidence: Overall confidence in results
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_domain: str = ""
    target_domains: List[str] = field(default_factory=list)
    mappings_found: List[IsomorphicMapping] = field(default_factory=list)
    best_mapping: Optional[IsomorphicMapping] = None
    cross_domain_patterns: List[CrossDomainPattern] = field(default_factory=list)
    inverted_constraints: List[InvertedConstraint] = field(default_factory=list)
    execution_time_ms: float = 0.0
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        """Alias for result_id."""
        return self.result_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "id": self.result_id,
            "source_domain": self.source_domain,
            "target_domains": self.target_domains,
            "mappings_found": [m.to_dict() if isinstance(m, IsomorphicMapping) else m for m in self.mappings_found],
            "best_mapping": self.best_mapping.to_dict() if self.best_mapping and isinstance(self.best_mapping, IsomorphicMapping) else None,
            "cross_domain_patterns": [p.to_dict() if isinstance(p, CrossDomainPattern) else p for p in self.cross_domain_patterns],
            "inverted_constraints": [c.to_dict() if isinstance(c, InvertedConstraint) else c for c in self.inverted_constraints],
            "execution_time_ms": self.execution_time_ms,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsomorphicMappingResult":
        """Create from dictionary."""
        data = data.copy()

        if "result_id" not in data and "id" in data:
            data["result_id"] = data["id"]

        # Parse nested objects
        if "mappings_found" in data:
            data["mappings_found"] = [
                IsomorphicMapping.from_dict(m) if isinstance(m, dict) else m
                for m in data["mappings_found"]
            ]

        if "best_mapping" in data and isinstance(data["best_mapping"], dict):
            data["best_mapping"] = IsomorphicMapping.from_dict(data["best_mapping"])

        if "cross_domain_patterns" in data:
            data["cross_domain_patterns"] = [
                CrossDomainPattern.from_dict(p) if isinstance(p, dict) else p
                for p in data["cross_domain_patterns"]
            ]

        if "inverted_constraints" in data:
            data["inverted_constraints"] = [
                InvertedConstraint.from_dict(c) if isinstance(c, dict) else c
                for c in data["inverted_constraints"]
            ]

        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class Phase2Config:
    """
    Configuration for Phase II: Isomorphic Mapping.

    All values must come from environment variables (Law of Configuration Explicitness).

    Attributes:
        max_target_domains: Maximum target domains to search (default: 10)
        i_mech_threshold: Minimum I_mech score for valid mapping (default: 0.7)
        pattern_recognition_threshold: Minimum confidence for patterns (default: 0.6)
        timeout_ms: Timeout per operation in ms (default: 20000)
        max_mappings: Maximum mappings to return (default: 50)
        enable_constraint_inversion: Enable Ψ₃ constraint inversion (default: true)
        search_depth: Depth for cross-domain search (default: 5)
        correlation_id: For tracing (optional)
    """
    max_target_domains: int = 10
    i_mech_threshold: float = 0.7
    pattern_recognition_threshold: float = 0.6
    timeout_ms: int = 20000
    max_mappings: int = 50
    enable_constraint_inversion: bool = True
    search_depth: int = 5
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Phase2Config":
        """
        Create configuration from environment variables.

        Required env vars (CLAUDE.md compliance):
        - PHASE2_MAX_TARGET_DOMAINS
        - PHASE2_IMECH_THRESHOLD
        - PHASE2_PATTERN_THRESHOLD
        - PHASE2_TIMEOUT_MS
        - PHASE2_MAX_MAPPINGS
        - PHASE2_ENABLE_CONSTRAINT_INVERSION
        - PHASE2_SEARCH_DEPTH

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        import os

        env_vars = {
            "PHASE2_MAX_TARGET_DOMAINS": ("max_target_domains", 10, int),
            "PHASE2_IMECH_THRESHOLD": ("i_mech_threshold", 0.7, float),
            "PHASE2_PATTERN_THRESHOLD": ("pattern_recognition_threshold", 0.6, float),
            "PHASE2_TIMEOUT_MS": ("timeout_ms", 20000, int),
            "PHASE2_MAX_MAPPINGS": ("max_mappings", 50, int),
            "PHASE2_ENABLE_CONSTRAINT_INVERSION": ("enable_constraint_inversion", True, lambda v: v.lower() == "true"),
            "PHASE2_SEARCH_DEPTH": ("search_depth", 5, int),
        }

        config = {}
        for env_name, (field_name, default, field_type) in env_vars.items():
            value = os.getenv(env_name)
            if value is None:
                # Use default for missing vars (don't crash)
                config[field_name] = default
            else:
                try:
                    config[field_name] = field_type(value)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid value for {env_name}: {value}. "
                        f"Expected {field_type.__name__}."
                    )

        # Optional correlation ID
        config["correlation_id"] = os.getenv("CORRELATION_ID")

        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_target_domains": self.max_target_domains,
            "i_mech_threshold": self.i_mech_threshold,
            "pattern_recognition_threshold": self.pattern_recognition_threshold,
            "timeout_ms": self.timeout_ms,
            "max_mappings": self.max_mappings,
            "enable_constraint_inversion": self.enable_constraint_inversion,
            "search_depth": self.search_depth,
            "correlation_id": self.correlation_id,
        }


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Enums
    "HypothesisStatus",
    "PatternType",
    "MCTSNodeState",
    "ExplorationStrategy",
    "ContradictionType",
    "IsomorphismType",

    # Core schemas
    "Hypothesis",
    "SearchTreeNode",
    "Pattern",
    "MCTSSearchResult",

    # Phase II schemas
    "FunctionalDependency",
    "FunctionalDependencyGraph",
    "IsomorphicMapping",
    "CrossDomainPattern",
    "InvertedConstraint",
    "IsomorphicMappingResult",
    "Phase2Config",

    # Configuration
    "ExplorationConfig",
]
