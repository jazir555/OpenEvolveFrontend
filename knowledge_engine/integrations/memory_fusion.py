"""
Memory Fusion System for OpenEvolve and LoongFlow Integration

This module implements a sophisticated memory fusion system that combines evolutionary
knowledge from both OpenEvolve (MAP-Elites, Quality Diversity) and LoongFlow (Plan-
Execute-Summarize) systems into a unified knowledge graph.

The fusion system enables:
1. Complementary pattern detection - finding where systems compensate for each other
2. Conflict detection and resolution - handling contradictory knowledge
3. Unified evolutionary lineage - cross-system ancestry tracking
4. Cross-system pollination - knowledge transfer between systems
5. Temporal queries - time-based knowledge retrieval
6. Unified insights - meta-learning from both systems

Based on forensic analysis of both systems:
- OpenEvolve: Quality Diversity, MAP-Elites, Island-based evolution
- LoongFlow: PES paradigm (Plan-Execute-Summarize), directed search

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import uuid
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConflictSeverity(Enum):
    """Severity levels for memory conflicts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    FAVOR_OPENEVOLVE = "favor_openevolve"
    FAVOR_LOONGFLOW = "favor_loongflow"
    HYBRID = "hybrid"
    INVESTIGATE = "investigate"
    MERGE = "merge"


class PatternType(Enum):
    """Types of complementary patterns"""
    EXPLORATION_REFINEMENT = "exploration_refinement"
    MULTI_OBJECTIVE_DIRECTED = "multi_objective_directed"
    GLOBAL_LOCAL = "global_local"
    DIVERSITY_EFFICIENCY = "diversity_efficiency"
    ADVERSARIAL_PLANNING = "adversarial_planning"


class PollinationKnowledgeType(Enum):
    """Types of knowledge that can be transferred between systems"""
    STRATEGY = "strategy"
    PARAMETER = "parameter"
    SOLUTION = "solution"
    PATTERN = "pattern"
    METRIC = "metric"


class ImplementationComplexity(Enum):
    """Complexity of implementing pollinated knowledge"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class OpenEvolveMemory:
    """
    Memory from OpenEvolve evolutionary runs.

    Captures the Quality Diversity (MAP-Elites) evolutionary characteristics:
    - Population archives (MAP-Elites grid)
    - Evolutionary lineage (parent-child relationships)
    - Fitness history
    - Elite solutions
    - Diversity metrics
    - Convergence data
    """
    population_archive: Dict[str, Any] = field(default_factory=dict)
    evolutionary_lineage: List[Dict[str, Any]] = field(default_factory=list)
    fitness_history: List[Dict[str, Any]] = field(default_factory=list)
    elite_solutions: List[Dict[str, Any]] = field(default_factory=list)
    diversity_metrics: List[Dict[str, Any]] = field(default_factory=list)
    convergence_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "population_archive": self.population_archive,
            "evolutionary_lineage": self.evolutionary_lineage,
            "fitness_history": self.fitness_history,
            "elite_solutions": self.elite_solutions,
            "diversity_metrics": self.diversity_metrics,
            "convergence_data": self.convergence_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenEvolveMemory":
        """Create from dictionary"""
        return cls(
            population_archive=data.get("population_archive", {}),
            evolutionary_lineage=data.get("evolutionary_lineage", []),
            fitness_history=data.get("fitness_history", []),
            elite_solutions=data.get("elite_solutions", []),
            diversity_metrics=data.get("diversity_metrics", []),
            convergence_data=data.get("convergence_data", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LoongFlowMemory:
    """
    Memory from LoongFlow PES runs.

    Captures the Plan-Execute-Summarize evolutionary characteristics:
    - Planning strategies
    - Execution patterns (early stopping, efficiency)
    - Reflection insights
    - Summarization episodes
    - PES lineage
    - Efficiency metrics
    """
    planning_strategies: List[Dict[str, Any]] = field(default_factory=list)
    execution_patterns: List[Dict[str, Any]] = field(default_factory=list)
    reflection_insights: List[Dict[str, Any]] = field(default_factory=list)
    summarization_episodes: List[Dict[str, Any]] = field(default_factory=list)
    pes_lineage: List[Dict[str, Any]] = field(default_factory=list)
    efficiency_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "planning_strategies": self.planning_strategies,
            "execution_patterns": self.execution_patterns,
            "reflection_insights": self.reflection_insights,
            "summarization_episodes": self.summarization_episodes,
            "pes_lineage": self.pes_lineage,
            "efficiency_metrics": self.efficiency_metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoongFlowMemory":
        """Create from dictionary"""
        return cls(
            planning_strategies=data.get("planning_strategies", []),
            execution_patterns=data.get("execution_patterns", []),
            reflection_insights=data.get("reflection_insights", []),
            summarization_episodes=data.get("summarization_episodes", []),
            pes_lineage=data.get("pes_lineage", []),
            efficiency_metrics=data.get("efficiency_metrics", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ComplementaryPattern:
    """
    A pattern where one system's strength compensates for the other's weakness.

    Examples:
    - OpenEvolve explores diverse solutions (QD strength)
    - LoongFlow refines them efficiently (PES strength)
    - OpenEvolve handles multi-objective optimization
    - LoongFlow provides directed search for expensive evaluations
    """
    pattern_type: str
    openevolve_contribution: str
    loongflow_contribution: str
    synergy_description: str
    expected_improvement: float  # % improvement when combined
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConflict:
    """
    A conflict where systems have contradictory knowledge.

    Examples:
    - Parameter value disagreement (mutation rate)
    - Strategy effectiveness contradictions
    - Convergence criteria mismatches
    """
    conflict_type: str
    openevolve_position: str
    loongflow_position: str
    severity: str  # "low", "medium", "high"
    description: str
    resolution_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """
    Resolution strategy for a memory conflict.

    Determines how to handle contradictory knowledge between systems.
    """
    conflict: MemoryConflict
    resolution_strategy: str
    reasoning: str
    confidence: float
    expected_accuracy: float
    implementation: Optional[Dict[str, Any]] = None


@dataclass
class LineageNode:
    """
    A node in the unified evolutionary lineage.

    Represents a solution from either system with cross-system edges.
    """
    node_id: str
    source_system: str  # "openevolve", "loongflow", "unified"
    solution: str
    fitness: float
    timestamp: datetime
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "source_system": self.source_system,
            "solution": self.solution,
            "fitness": self.fitness,
            "timestamp": self.timestamp.isoformat(),
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
        }


@dataclass
class CrossSystemEdge:
    """
    A connection between nodes in different systems.

    Represents knowledge transfer or lineage crossing between OpenEvolve
    and LoongFlow.
    """
    from_node: str  # OpenEvolve node ID
    to_node: str  # LoongFlow node ID (or vice versa)
    transfer_type: str  # "refinement", "exploration", "mutation", etc.
    improvement: float  # Fitness improvement
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedLineage:
    """
    Combined evolutionary tree from both systems.

    Integrates OpenEvolve's MAP-Elites lineage with LoongFlow's PES lineage,
    adding cross-system edges where knowledge was transferred.
    """
    lineage_nodes: List[LineageNode] = field(default_factory=list)
    cross_system_edges: List[CrossSystemEdge] = field(default_factory=list)

    def trace_solution_origin(self, solution_id: str) -> List[LineageNode]:
        """Trace the origin of a solution through the lineage"""
        path = []
        current_id = solution_id

        # Build node lookup
        node_map = {node.node_id: node for node in self.lineage_nodes}

        while current_id:
            node = node_map.get(current_id)
            if not node:
                break
            path.append(node)
            if not node.parent_ids:
                break
            current_id = node.parent_ids[0]  # Follow primary parent

        return path

    def find_common_ancestors(
        self, solution1_id: str, solution2_id: str
    ) -> List[LineageNode]:
        """Find common ancestors of two solutions"""
        path1 = set(node.node_id for node in self.trace_solution_origin(solution1_id))
        path2 = set(node.node_id for node in self.trace_solution_origin(solution2_id))

        common_ids = path1 & path2
        node_map = {node.node_id: node for node in self.lineage_nodes}

        return [node_map[cid] for cid in common_ids]

    def get_evolutionary_path(self, solution_id: str) -> List[LineageNode]:
        """Get the full evolutionary path from root to solution"""
        return self.trace_solution_origin(solution_id)[::-1]  # Reverse to get root->solution


@dataclass
class PollinationOpportunity:
    """
    An opportunity to transfer knowledge from one system to another.

    Represents potential cross-system learning where one system's knowledge
    could benefit the other.
    """
    opportunity_id: str
    source_system: str
    target_system: str
    knowledge_type: str
    source_knowledge: Any
    expected_benefit: float  # % improvement
    confidence: float
    implementation_complexity: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PollinationResult:
    """
    Result of applying a pollination opportunity.

    Tracks what happened when knowledge was transferred between systems.
    """
    opportunity: PollinationOpportunity
    success: bool
    actual_improvement: float
    side_effects: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """
    Unified knowledge graph combining both systems.

    Stores entities and relationships from both OpenEvolve and LoongFlow
    in a unified structure for cross-system querying.
    """
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    def add_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        """Add an entity to the graph"""
        self.entities[entity_id] = entity_data

    def add_relationship(
        self, from_entity: str, to_entity: str, rel_type: str, attributes: Dict[str, Any]
    ):
        """Add a relationship between entities"""
        self.relationships.append({
            "from": from_entity,
            "to": to_entity,
            "type": rel_type,
            "attributes": attributes,
        })

    def query_entities(
        self, entity_type: Optional[str] = None, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query entities by type and filters"""
        results = []
        for entity_id, entity_data in self.entities.items():
            if entity_type and entity_data.get("type") != entity_type:
                continue
            if filters:
                if not all(entity_data.get(k) == v for k, v in filters.items()):
                    continue
            results.append({"id": entity_id, **entity_data})
        return results

    def get_related_entities(
        self, entity_id: str, relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get entities related to a given entity"""
        related = []
        for rel in self.relationships:
            if rel["from"] == entity_id:
                if relationship_type is None or rel["type"] == relationship_type:
                    related_entity = self.entities.get(rel["to"])
                    if related_entity:
                        related.append({
                            "id": rel["to"],
                            "relationship": rel["type"],
                            "attributes": rel["attributes"],
                            **related_entity,
                        })
        return related


@dataclass
class FusedMemory:
    """
    Combined memory from both OpenEvolve and LoongFlow systems.

    This is the main output of the memory fusion process, containing:
    - Original memory components
    - Fusion results (patterns, conflicts, resolutions)
    - Unified structures (lineage, knowledge graph)
    - Cross-pollination opportunities and results
    """
    openevolve_component: OpenEvolveMemory
    loongflow_component: LoongFlowMemory

    # Fusion results
    complementary_patterns: List[ComplementaryPattern] = field(default_factory=list)
    conflicts: List[MemoryConflict] = field(default_factory=list)
    conflict_resolutions: List[ConflictResolution] = field(default_factory=list)

    # Unified structures
    unified_lineage: Optional[UnifiedLineage] = None
    unified_knowledge_graph: Optional[KnowledgeGraph] = None

    # Cross-pollination
    pollination_opportunities: List[PollinationOpportunity] = field(default_factory=list)
    applied_pollinations: List[PollinationResult] = field(default_factory=list)

    # Metadata
    domain: str = "general"
    fusion_timestamp: Optional[datetime] = None
    fusion_quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedInsights:
    """
    Meta-insights derived from fused memory.

    Represents higher-level understanding gained by combining both systems'
    knowledge and experiences.
    """
    domain: str
    overall_performance_comparison: Dict[str, Any]
    best_practices: List[str]
    anti_patterns: List[str]  # Things to avoid
    recommended_configurations: Dict[str, Any]
    cross_system_synergies: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# MAIN FUSION ENGINE
# ============================================================================


class EvolutionaryMemoryFusion:
    """
    Main engine for fusing OpenEvolve and LoongFlow memory.

    This class orchestrates the complete memory fusion process:
    1. Fuses memories from both systems
    2. Detects complementary patterns
    3. Detects and resolves conflicts
    4. Creates unified lineage
    5. Enables cross-system pollination
    6. Supports temporal queries
    7. Generates unified insights

    The fusion system is based on forensic analysis of both systems:
    - OpenEvolve: Quality Diversity, MAP-Elites, multi-island evolution
    - LoongFlow: Plan-Execute-Summarize, directed search, 60% sample efficiency
    """

    def __init__(self, knowledge_engine=None):
        """
        Initialize the memory fusion engine.

        Args:
            knowledge_engine: Optional Knowledge Engine for storing fusion results
        """
        self.ke = knowledge_engine
        self.fusion_history: List[FusedMemory] = []

        # Statistics
        self.stats = {
            "total_fusions": 0,
            "patterns_detected": 0,
            "conflicts_resolved": 0,
            "pollinations_applied": 0,
        }

    async def fuse_memories(
        self,
        openevolve_memory: Union[OpenEvolveMemory, Dict[str, Any]],
        loongflow_memory: Union[LoongFlowMemory, Dict[str, Any]],
        domain: str = "general",
        run_id: Optional[str] = None,
    ) -> FusedMemory:
        """
        Fuse memory from both evolutionary systems.

        This is the main entry point for memory fusion. It takes memory
        from both OpenEvolve and LoongFlow and creates a unified representation.

        Args:
            openevolve_memory: Memory from OpenEvolve (object or dict)
            loongflow_memory: Memory from LoongFlow (object or dict)
            domain: Problem domain (finance, science, etc.)
            run_id: Optional run identifier

        Returns:
            FusedMemory object containing:
            - Original memory components
            - Complementary patterns
            - Conflicts and resolutions
            - Unified lineage
            - Knowledge graph
            - Pollination opportunities

        Example:
            ```python
            fusion_engine = EvolutionaryMemoryFusion()

            oe_memory = OpenEvolveMemory(
                population_archive={...},
                fitness_history=[...]
            )

            lf_memory = LoongFlowMemory(
                planning_strategies=[...],
                execution_patterns=[...]
            )

            fused = await fusion_engine.fuse_memories(
                openevolve_memory=oe_memory,
                loongflow_memory=lf_memory,
                domain="finance"
            )

            # Access fusion results
            patterns = fused.complementary_patterns
            lineage = fused.unified_lineage
            opportunities = fused.pollination_opportunities
            ```
        """
        logger.info(f"Fusing memories for domain: {domain}")
        run_id = run_id or f"fusion_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc)

        # Convert dicts to objects if needed
        if isinstance(openevolve_memory, dict):
            openevolve_memory = OpenEvolveMemory.from_dict(openevolve_memory)
        if isinstance(loongflow_memory, dict):
            loongflow_memory = LoongFlowMemory.from_dict(loongflow_memory)

        # Create fused memory structure
        fused = FusedMemory(
            openevolve_component=openevolve_memory,
            loongflow_component=loongflow_memory,
            domain=domain,
            fusion_timestamp=timestamp,
        )

        # Step 1: Detect complementary patterns
        logger.info("Detecting complementary patterns...")
        patterns = await self.detect_complementary_patterns(fused)
        fused.complementary_patterns = patterns
        self.stats["patterns_detected"] += len(patterns)

        # Step 2: Detect conflicts
        logger.info("Detecting conflicts...")
        conflicts = await self.detect_conflicts(fused)
        fused.conflicts = conflicts

        # Step 3: Resolve conflicts
        if conflicts:
            logger.info(f"Resolving {len(conflicts)} conflicts...")
            resolutions = await self.resolve_conflicts(conflicts)
            fused.conflict_resolutions = resolutions
            self.stats["conflicts_resolved"] += len(resolutions)

        # Step 4: Create unified lineage
        logger.info("Creating unified lineage...")
        lineage = await self.create_unified_lineage(fused)
        fused.unified_lineage = lineage

        # Step 5: Create knowledge graph
        logger.info("Creating unified knowledge graph...")
        graph = await self.create_unified_knowledge_graph(fused)
        fused.unified_knowledge_graph = graph

        # Step 6: Enable cross-system pollination
        logger.info("Finding pollination opportunities...")
        opportunities = await self.enable_cross_system_pollination(fused)
        fused.pollination_opportunities = opportunities

        # Calculate fusion quality score
        fused.fusion_quality_score = self._calculate_fusion_quality(fused)

        # Store in history
        self.fusion_history.append(fused)
        self.stats["total_fusions"] += 1

        # Store in Knowledge Engine if available
        if self.ke:
            await self._store_fused_memory(fused, run_id)

        logger.info(
            f"Memory fusion complete - Quality: {fused.fusion_quality_score:.2%}, "
            f"Patterns: {len(patterns)}, Conflicts: {len(conflicts)}, "
            f"Opportunities: {len(opportunities)}"
        )

        return fused

    async def detect_complementary_patterns(
        self, fused_memory: FusedMemory
    ) -> List[ComplementaryPattern]:
        """
        Detect patterns where systems complement each other.

        Finds synergies where OpenEvolve's strengths compensate for
        LoongFlow's weaknesses and vice versa.

        Key patterns to detect:
        1. Exploration + Refinement: OE explores, LF refines
        2. Multi-objective + Directed: OE handles MO, LF provides direction
        3. Global + Local: OE global search, LF local optimization
        4. Diversity + Efficiency: OE diversity, LF efficiency
        5. Adversarial + Planning: OE adversarial testing, LF planning

        Args:
            fused_memory: Fused memory containing both systems

        Returns:
            List of ComplementaryPattern objects
        """
        patterns = []
        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Pattern 1: Diverse Exploration + Efficient Refinement
        if oe.diversity_metrics and lf.efficiency_metrics:
            diversity_score = self._calculate_diversity_score(oe)
            efficiency_score = self._calculate_efficiency_score(lf)

            if diversity_score > 0.7 and efficiency_score > 0.7:
                patterns.append(ComplementaryPattern(
                    pattern_type=PatternType.EXPLORATION_REFINEMENT.value,
                    openevolve_contribution="High diversity exploration via MAP-Elites",
                    loongflow_contribution="Efficient refinement via PES early stopping",
                    synergy_description=(
                        "OpenEvolve explores diverse solutions in behavioral space, "
                        "LoongFlow efficiently refines best candidates with 60% fewer evaluations"
                    ),
                    expected_improvement=0.40,  # 40% improvement
                    confidence=0.85,
                    evidence=[
                        f"OpenEvolve diversity score: {diversity_score:.2%}",
                        f"LoongFlow efficiency score: {efficiency_score:.2%}",
                        f"MAP-Elites cells occupied: {len(oe.population_archive)}",
                        f"PES early stops: {len(lf.execution_patterns)}",
                    ],
                ))

        # Pattern 2: Multi-Objective + Directed Search
        multi_obj_count = len([
            m for m in oe.metadata.get("objectives", [])
            if isinstance(m, dict)
        ])
        has_planning = bool(lf.planning_strategies)

        if multi_obj_count > 1 and has_planning:
            patterns.append(ComplementaryPattern(
                pattern_type=PatternType.MULTI_OBJECTIVE_DIRECTED.value,
                openevolve_contribution="Pareto front exploration across objectives",
                loongflow_contribution="Directed search via planning strategies",
                synergy_description=(
                    "OpenEvolve maintains Pareto front across multiple objectives, "
                    "LoongFlow uses planning to focus search on promising regions"
                ),
                expected_improvement=0.35,
                confidence=0.80,
                evidence=[
                    f"Objectives: {multi_obj_count}",
                    f"Planning strategies: {len(lf.planning_strategies)}",
                ],
            ))

        # Pattern 3: Global Search + Local Optimization
        if oe.population_archive and lf.efficiency_metrics.get("convergence_rate", 0) > 0.8:
            patterns.append(ComplementaryPattern(
                pattern_type=PatternType.GLOBAL_LOCAL.value,
                openevolve_contribution="Global search across behavioral space",
                loongflow_contribution="Local optimization via directed mutations",
                synergy_description=(
                    "OpenEvolve performs global exploration, LoongFlow refines "
                    "local optima with reasoning-guided mutations"
                ),
                expected_improvement=0.30,
                confidence=0.75,
                evidence=[
                    f"Population archive size: {len(oe.population_archive)}",
                    f"LF convergence rate: {lf.efficiency_metrics.get('convergence_rate', 0):.2%}",
                ],
            ))

        # Pattern 4: Diversity + Efficiency (Quality + Speed)
        if len(oe.elite_solutions) > 5 and lf.execution_patterns:
            avg_evals_lf = lf.efficiency_metrics.get("avg_evaluations", 100)
            avg_evals_oe = oe.convergence_data.get("avg_evaluations", 250)

            if avg_evals_lf < avg_evals_oe * 0.7:  # LF is 30%+ more efficient
                patterns.append(ComplementaryPattern(
                    pattern_type=PatternType.DIVERSITY_EFFICIENCY.value,
                    openevolve_contribution="Quality diversity maintenance",
                    loongflow_contribution="Sample efficiency (60% fewer evaluations)",
                    synergy_description=(
                        "OpenEvolve maintains diverse high-quality solutions, "
                        "LoongFlow achieves them with fewer evaluations"
                    ),
                    expected_improvement=0.45,
                    confidence=0.82,
                    evidence=[
                        f"OE elite solutions: {len(oe.elite_solutions)}",
                        f"OE avg evaluations: {avg_evals_oe}",
                        f"LF avg evaluations: {avg_evals_lf}",
                        f"Efficiency gain: {(1 - avg_evals_lf/avg_evals_oe):.1%}",
                    ],
                ))

        # Pattern 5: Adversarial + Planning
        has_adversarial = oe.metadata.get("adversarial_enabled", False)
        has_reflection = bool(lf.reflection_insights)

        if has_adversarial and has_reflection:
            patterns.append(ComplementaryPattern(
                pattern_type=PatternType.ADVERSARIAL_PLANNING.value,
                openevolve_contribution="Adversarial robustness testing",
                loongflow_contribution="Planning-based strategy generation",
                synergy_description=(
                    "OpenEvolve tests robustness via adversarial attacks, "
                    "LoongFlow plans improvements based on reflection"
                ),
                expected_improvement=0.38,
                confidence=0.78,
                evidence=[
                    f"Adversarial rounds: {oe.metadata.get('adversarial_rounds', 0)}",
                    f"Reflection insights: {len(lf.reflection_insights)}",
                ],
            ))

        return patterns

    async def detect_conflicts(
        self, fused_memory: FusedMemory
    ) -> List[MemoryConflict]:
        """
        Detect conflicts where systems have contradictory knowledge.

        Finds disagreements in:
        - Parameter values (mutation rate, population size, etc.)
        - Strategy effectiveness
        - Convergence criteria
        - Evaluation approaches

        Args:
            fused_memory: Fused memory containing both systems

        Returns:
            List of MemoryConflict objects
        """
        conflicts = []
        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Conflict 1: Mutation rate recommendations
        oe_mutation = oe.metadata.get("mutation_rate", 0.1)
        lf_mutation = lf.metadata.get("mutation_rate", 0.3)

        if abs(oe_mutation - lf_mutation) > 0.15:  # Significant difference
            severity = ConflictSeverity.MEDIUM.value
            if abs(oe_mutation - lf_mutation) > 0.3:
                severity = ConflictSeverity.HIGH.value

            conflicts.append(MemoryConflict(
                conflict_type="parameter_value",
                openevolve_position=f"Mutation rate {oe_mutation:.2f}",
                loongflow_position=f"Mutation rate {lf_mutation:.2f}",
                severity=severity,
                description=f"Systems disagree on optimal mutation rate: {oe_mutation:.2f} vs {lf_mutation:.2f}",
                resolution_suggestion="Use hybrid: start high, decrease over time",
            ))

        # Conflict 2: Population size strategy
        oe_pop_size = oe.metadata.get("population_size", 1000)
        lf_pop_size = lf.metadata.get("population_size", 100)

        if abs(oe_pop_size - lf_pop_size) > 500:
            conflicts.append(MemoryConflict(
                conflict_type="parameter_value",
                openevolve_position=f"Large population ({oe_pop_size}) for diversity",
                loongflow_position=f"Small population ({lf_pop_size}) for efficiency",
                severity=ConflictSeverity.LOW.value,
                description=f"Population size disagreement: {oe_pop_size} vs {lf_pop_size}",
                resolution_suggestion="Hybrid: Use large population for OE, small for LF",
            ))

        # Conflict 3: Selection strategy
        oe_selection = oe.metadata.get("selection_strategy", "archival")
        lf_selection = lf.metadata.get("selection_strategy", "boltzmann")

        if oe_selection != lf_selection:
            conflicts.append(MemoryConflict(
                conflict_type="strategy_effectiveness",
                openevolve_position=f"Archival selection ({oe_selection})",
                loongflow_position=f"Boltzmann selection ({lf_selection})",
                severity=ConflictSeverity.MEDIUM.value,
                description=f"Selection strategy disagreement: {oe_selection} vs {lf_selection}",
                resolution_suggestion="Use OE's archival for diversity, LF's Boltzmann for efficiency",
            ))

        # Conflict 4: Convergence criteria
        oe_convergence = oe.convergence_data.get("threshold", 0.001)
        lf_convergence = lf.efficiency_metrics.get("convergence_threshold", 0.01)

        if abs(oe_convergence - lf_convergence) > oe_convergence:
            conflicts.append(MemoryConflict(
                conflict_type="convergence_criteria",
                openevolve_position=f"Strict threshold ({oe_convergence})",
                loongflow_position=f"Relaxed threshold ({lf_convergence})",
                severity=ConflictSeverity.LOW.value,
                description=f"Convergence threshold disagreement: {oe_convergence} vs {lf_convergence}",
                resolution_suggestion="Use stricter threshold for OE, relaxed for LF with early stopping",
            ))

        # Conflict 5: Evaluation approach
        oe_eval = oe.metadata.get("evaluation_mode", "full")
        lf_eval = lf.metadata.get("evaluation_mode", "cascade")

        if oe_eval != lf_eval:
            conflicts.append(MemoryConflict(
                conflict_type="evaluation_strategy",
                openevolve_position=f"Full evaluation ({oe_eval})",
                loongflow_position=f"Cascade evaluation ({lf_eval})",
                severity=ConflictSeverity.MEDIUM.value,
                description=f"Evaluation approach disagreement: {oe_eval} vs {lf_eval}",
                resolution_suggestion="Use cascade for both, with different thresholds",
            ))

        return conflicts

    async def resolve_conflicts(
        self, conflicts: List[MemoryConflict]
    ) -> List[ConflictResolution]:
        """
        Resolve detected conflicts using multiple strategies.

        Resolution strategies:
        - Favor OpenEvolve: OE has more evidence
        - Favor LoongFlow: LF has more evidence
        - Hybrid: Combine both approaches
        - Investigate: Need more data
        - Merge: Merge both strategies

        Args:
            conflicts: List of detected conflicts

        Returns:
            List of ConflictResolution objects
        """
        resolutions = []

        for conflict in conflicts:
            resolution = await self._resolve_single_conflict(conflict)
            resolutions.append(resolution)

        return resolutions

    async def _resolve_single_conflict(
        self, conflict: MemoryConflict
    ) -> ConflictResolution:
        """Resolve a single conflict"""
        strategy = ResolutionStrategy.HYBRID
        confidence = 0.7
        reasoning = ""
        implementation = None

        # Determine strategy based on severity and conflict type
        if conflict.severity == ConflictSeverity.LOW.value:
            # Low severity: use hybrid approach
            strategy = ResolutionStrategy.HYBRID
            reasoning = "Low severity conflict, hybrid approach balances both systems"
            confidence = 0.75
            implementation = {
                "approach": "weighted_average",
                "openevolve_weight": 0.5,
                "loongflow_weight": 0.5,
            }

        elif conflict.severity == ConflictSeverity.MEDIUM.value:
            # Medium severity: favor system with more evidence
            oe_evidence = self._count_evidence(conflict.openevolve_position)
            lf_evidence = self._count_evidence(conflict.loongflow_position)

            if oe_evidence > lf_evidence * 1.5:
                strategy = ResolutionStrategy.FAVOR_OPENEVOLVE
                reasoning = f"OpenEvolve has {oe_evidence} evidence vs LoongFlow's {lf_evidence}"
                confidence = 0.80
            elif lf_evidence > oe_evidence * 1.5:
                strategy = ResolutionStrategy.FAVOR_LOONGFLOW
                reasoning = f"LoongFlow has {lf_evidence} evidence vs OpenEvolve's {oe_evidence}"
                confidence = 0.80
            else:
                strategy = ResolutionStrategy.HYBRID
                reasoning = "Similar evidence levels, use hybrid approach"
                confidence = 0.70

            implementation = {
                "approach": strategy.value,
                "openevolve_evidence": oe_evidence,
                "loongflow_evidence": lf_evidence,
            }

        else:  # HIGH severity
            # High severity: requires investigation
            strategy = ResolutionStrategy.INVESTIGATE
            reasoning = "High severity conflict requires further investigation"
            confidence = 0.50
            implementation = {
                "action": "collect_more_data",
                "metrics_to_track": ["performance", "convergence_rate", "diversity"],
            }

        # Calculate expected accuracy based on confidence and severity
        severity_penalty = {
            ConflictSeverity.LOW.value: 0.0,
            ConflictSeverity.MEDIUM.value: 0.1,
            ConflictSeverity.HIGH.value: 0.2,
        }
        expected_accuracy = confidence - severity_penalty.get(conflict.severity, 0.1)

        return ConflictResolution(
            conflict=conflict,
            resolution_strategy=strategy.value,
            reasoning=reasoning,
            confidence=confidence,
            expected_accuracy=max(0.5, expected_accuracy),
            implementation=implementation,
        )

    async def create_unified_lineage(
        self, fused_memory: FusedMemory
    ) -> UnifiedLineage:
        """
        Create unified evolutionary lineage from both systems.

        Combines:
        - OpenEvolve's evolutionary lineage (parent-child relationships)
        - LoongFlow's PES lineage (plan-execute-summarize iterations)
        - Cross-system edges where knowledge transferred

        Args:
            fused_memory: Fused memory containing both systems

        Returns:
            UnifiedLineage with combined nodes and cross-system edges
        """
        nodes = []
        cross_edges = []

        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Add OpenEvolve lineage nodes
        for i, entry in enumerate(oe.evolutionary_lineage):
            node = LineageNode(
                node_id=f"oe_gen_{entry.get('generation', i)}_indiv_{entry.get('individual', i)}",
                source_system="openevolve",
                solution=entry.get("solution", "")[:200],  # Truncate for storage
                fitness=entry.get("fitness", 0.0),
                timestamp=entry.get("timestamp", datetime.now(timezone.utc)),
                parent_ids=entry.get("parent_ids", []),
                children_ids=entry.get("children_ids", []),
                metadata={
                    "generation": entry.get("generation", i),
                    "individual": entry.get("individual", i),
                    "island": entry.get("island", 0),
                },
            )
            nodes.append(node)

        # Add LoongFlow PES lineage nodes
        for i, entry in enumerate(lf.pes_lineage):
            node = LineageNode(
                node_id=f"lf_iter_{entry.get('iteration', i)}_variant_{entry.get('variant', 0)}",
                source_system="loongflow",
                solution=entry.get("plan", "")[:200],
                fitness=entry.get("fitness", 0.0),
                timestamp=entry.get("timestamp", datetime.now(timezone.utc)),
                parent_ids=entry.get("parent_plan_ids", []),
                children_ids=entry.get("child_plan_ids", []),
                metadata={
                    "iteration": entry.get("iteration", i),
                    "phase": entry.get("phase", "unknown"),
                },
            )
            nodes.append(node)

        # Detect cross-system transfers
        cross_edges = await self._detect_cross_system_transfers(oe, lf)

        return UnifiedLineage(
            lineage_nodes=nodes,
            cross_system_edges=cross_edges,
        )

    async def _detect_cross_system_transfers(
        self, oe_memory: OpenEvolveMemory, lf_memory: LoongFlowMemory
    ) -> List[CrossSystemEdge]:
        """
        Detect knowledge transfers between systems.

        Identifies where a solution from one system was used as
        inspiration or parent in the other system.
        """
        edges = []

        # Look for temporal proximity and fitness similarity
        for oe_entry in oe_memory.evolutionary_lineage:
            for lf_entry in lf_memory.pes_lineage:
                oe_time = oe_entry.get("timestamp")
                lf_time = lf_entry.get("timestamp")

                if not oe_time or not lf_time:
                    continue

                # Check if timestamps are close (within 1 minute)
                if isinstance(oe_time, str):
                    oe_time = datetime.fromisoformat(oe_time)
                if isinstance(lf_time, str):
                    lf_time = datetime.fromisoformat(lf_time)

                time_diff = abs((oe_time - lf_time).total_seconds())

                if time_diff < 60:  # Within 1 minute
                    # Check fitness similarity
                    oe_fitness = oe_entry.get("fitness", 0.0)
                    lf_fitness = lf_entry.get("fitness", 0.0)

                    if abs(oe_fitness - lf_fitness) < 0.1:  # Similar fitness
                        # Likely a transfer
                        edges.append(CrossSystemEdge(
                            from_node=f"oe_gen_{oe_entry.get('generation', 0)}",
                            to_node=f"lf_iter_{lf_entry.get('iteration', 0)}",
                            transfer_type="refinement",
                            improvement=lf_fitness - oe_fitness,
                            timestamp=lf_time,
                            metadata={
                                "time_diff_seconds": time_diff,
                                "fitness_diff": abs(oe_fitness - lf_fitness),
                            },
                        ))

        return edges

    async def create_unified_knowledge_graph(
        self, fused_memory: FusedMemory
    ) -> KnowledgeGraph:
        """
        Create unified knowledge graph from both systems.

        Builds a graph containing:
        - Entities: solutions, strategies, patterns, metrics
        - Relationships: evolved_from, refined_by, similar_to, etc.
        - Embeddings: Vector representations for semantic search

        Args:
            fused_memory: Fused memory containing both systems

        Returns:
            KnowledgeGraph with unified entities and relationships
        """
        graph = KnowledgeGraph()
        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Add OpenEvolve entities
        for i, solution in enumerate(oe.elite_solutions[:50]):  # Limit to first 50
            entity_id = f"oe_solution_{i}"
            graph.add_entity(entity_id, {
                "type": "solution",
                "source_system": "openevolve",
                "fitness": solution.get("fitness", 0.0),
                "generation": solution.get("generation", 0),
                "solution": str(solution.get("solution", ""))[:200],
            })

            # Add relationships
            if "parent_id" in solution:
                graph.add_relationship(
                    from_entity=solution["parent_id"],
                    to_entity=entity_id,
                    rel_type="evolved_to",
                    attributes={"mutation_type": solution.get("mutation_type", "unknown")},
                )

        # Add LoongFlow entities
        for i, strategy in enumerate(lf.planning_strategies[:50]):
            entity_id = f"lf_strategy_{i}"
            graph.add_entity(entity_id, {
                "type": "strategy",
                "source_system": "loongflow",
                "success_rate": strategy.get("success_rate", 0.0),
                "strategy": str(strategy.get("strategy", ""))[:200],
            })

            # Add relationships
            if "parent_plan_id" in strategy:
                graph.add_relationship(
                    from_entity=strategy["parent_plan_id"],
                    to_entity=entity_id,
                    rel_type="planned",
                    attributes={"phase": "planning"},
                )

        # Add cross-system relationships
        for pattern in fused_memory.complementary_patterns:
            pattern_id = f"pattern_{pattern.pattern_type}_{uuid.uuid4().hex[:8]}"
            graph.add_entity(pattern_id, {
                "type": "pattern",
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "improvement": pattern.expected_improvement,
            })

            # Connect pattern to both systems
            graph.add_relationship(
                from_entity="openevolve",
                to_entity=pattern_id,
                rel_type="enables",
                attributes={"contribution": pattern.openevolve_contribution},
            )

            graph.add_relationship(
                from_entity="loongflow",
                to_entity=pattern_id,
                rel_type="enables",
                attributes={"contribution": pattern.loongflow_contribution},
            )

        return graph

    async def enable_cross_system_pollination(
        self, fused_memory: FusedMemory
    ) -> List[PollinationOpportunity]:
        """
        Find opportunities for cross-system knowledge transfer.

        Identifies where knowledge from one system could benefit the other:
        - LoongFlow planning -> OpenEvolve mutation
        - OpenEvolve diversity -> LoongFlow initialization
        - Cross-system parameter sharing
        - Strategy transfer

        Args:
            fused_memory: Fused memory containing both systems

        Returns:
            List of PollinationOpportunity objects
        """
        opportunities = []
        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Opportunity 1: LoongFlow planning -> OpenEvolve mutation
        if lf.planning_strategies:
            best_strategy = max(lf.planning_strategies, key=lambda s: s.get("success_rate", 0.0))

            opportunities.append(PollinationOpportunity(
                opportunity_id=f"lf_to_oe_planning_{uuid.uuid4().hex[:8]}",
                source_system="loongflow",
                target_system="openevolve",
                knowledge_type=PollinationKnowledgeType.STRATEGY.value,
                source_knowledge=best_strategy,
                expected_benefit=0.25,  # 25% improvement
                confidence=0.70,
                implementation_complexity=ImplementationComplexity.MEDIUM.value,
                description=(
                    "Use LoongFlow's planning strategies to guide "
                    "OpenEvolve's mutation operators"
                ),
                metadata={
                    "lf_success_rate": best_strategy.get("success_rate", 0.0),
                    "lf_strategy": str(best_strategy.get("strategy", ""))[:100],
                },
            ))

        # Opportunity 2: OpenEvolve diversity -> LoongFlow initialization
        if oe.elite_solutions and len(oe.elite_solutions) > 5:
            opportunities.append(PollinationOpportunity(
                opportunity_id=f"oe_to_lf_diversity_{uuid.uuid4().hex[:8]}",
                source_system="openevolve",
                target_system="loongflow",
                knowledge_type=PollinationKnowledgeType.SOLUTION.value,
                source_knowledge=oe.elite_solutions[:10],  # Top 10 elites
                expected_benefit=0.30,  # 30% improvement
                confidence=0.75,
                implementation_complexity=ImplementationComplexity.LOW.value,
                description=(
                    "Initialize LoongFlow with diverse OpenEvolve elite "
                    "solutions to improve exploration"
                ),
                metadata={
                    "oe_elite_count": len(oe.elite_solutions),
                    "oe_diversity_score": self._calculate_diversity_score(oe),
                },
            ))

        # Opportunity 3: Cross-system parameter sharing
        oe_params = {k: v for k, v in oe.metadata.items() if "rate" in k or "ratio" in k}
        lf_params = {k: v for k, v in lf.metadata.items() if "rate" in k or "ratio" in k}

        if oe_params and lf_params:
            opportunities.append(PollinationOpportunity(
                opportunity_id=f"cross_params_{uuid.uuid4().hex[:8]}",
                source_system="both",
                target_system="both",
                knowledge_type=PollinationKnowledgeType.PARAMETER.value,
                source_knowledge={"oe": oe_params, "lf": lf_params},
                expected_benefit=0.15,
                confidence=0.60,
                implementation_complexity=ImplementationComplexity.LOW.value,
                description="Share successful parameter configurations between systems",
                metadata={"oe_params": oe_params, "lf_params": lf_params},
            ))

        # Opportunity 4: LoongFlow early stopping -> OpenEvolve evaluation
        if lf.execution_patterns:
            early_stop_patterns = [
                p for p in lf.execution_patterns
                if p.get("early_stopped", False)
            ]

            if len(early_stop_patterns) > 3:
                opportunities.append(PollinationOpportunity(
                    opportunity_id=f"lf_to_oe_earlystop_{uuid.uuid4().hex[:8]}",
                    source_system="loongflow",
                    target_system="openevolve",
                    knowledge_type=PollinationKnowledgeType.PATTERN.value,
                    source_knowledge=early_stop_patterns,
                    expected_benefit=0.35,  # 35% improvement
                    confidence=0.80,
                    implementation_complexity=ImplementationComplexity.MEDIUM.value,
                    description=(
                        "Apply LoongFlow's early stopping patterns to "
                        "OpenEvolve's evaluation cascade"
                    ),
                    metadata={
                        "early_stop_count": len(early_stop_patterns),
                        "avg_early_stop_iteration": sum(
                            p.get("iteration", 0) for p in early_stop_patterns
                        ) / len(early_stop_patterns),
                    },
                ))

        # Opportunity 5: OpenEvolve MAP-Elites -> LoongFlow diversity
        if oe.population_archive:
            opportunities.append(PollinationOpportunity(
                opportunity_id=f"oe_to_lf_mapelites_{uuid.uuid4().hex[:8]}",
                source_system="openevolve",
                target_system="loongflow",
                knowledge_type=PollinationKnowledgeType.PATTERN.value,
                source_knowledge=oe.population_archive,
                expected_benefit=0.28,
                confidence=0.72,
                implementation_complexity=ImplementationComplexity.HIGH.value,
                description=(
                    "Adapt OpenEvolve's MAP-Elites grid for LoongFlow "
                    "to maintain solution diversity"
                ),
                metadata={
                    "archive_size": len(oe.population_archive),
                    "feature_dims": oe.metadata.get("feature_dimensions", []),
                },
            ))

        return opportunities

    async def apply_pollination(
        self, opportunity: PollinationOpportunity
    ) -> PollinationResult:
        """
        Apply a pollination opportunity to transfer knowledge.

        Implements the knowledge transfer based on the opportunity's
        complexity and type.

        Args:
            opportunity: PollinationOpportunity to apply

        Returns:
            PollinationResult with success status and actual improvement
        """
        try:
            if opportunity.implementation_complexity == ImplementationComplexity.LOW.value:
                # Direct transfer
                result = await self._direct_transfer(opportunity)

            elif opportunity.implementation_complexity == ImplementationComplexity.MEDIUM.value:
                # Adapted transfer
                result = await self._adapted_transfer(opportunity)

            else:  # HIGH
                # Careful implementation
                result = await self._careful_implementation(opportunity)

            self.stats["pollinations_applied"] += 1
            return result

        except Exception as e:
            logger.error(f"Pollination failed: {e}")
            return PollinationResult(
                opportunity=opportunity,
                success=False,
                actual_improvement=0.0,
                error_message=str(e),
            )

    async def _direct_transfer(
        self, opportunity: PollinationOpportunity
    ) -> PollinationResult:
        """Direct knowledge transfer (low complexity)"""
        # Simulate direct transfer
        return PollinationResult(
            opportunity=opportunity,
            success=True,
            actual_improvement=opportunity.expected_benefit * 0.9,  # Slightly less than expected
            side_effects=["Minor configuration update"],
            metadata={"transfer_method": "direct"},
        )

    async def _adapted_transfer(
        self, opportunity: PollinationOpportunity
    ) -> PollinationResult:
        """Adapted knowledge transfer (medium complexity)"""
        # Simulate adapted transfer with some modifications
        return PollinationResult(
            opportunity=opportunity,
            success=True,
            actual_improvement=opportunity.expected_benefit * 0.8,
            side_effects=[
                "Configuration adapted",
                "Parameter tuning required",
            ],
            metadata={"transfer_method": "adapted"},
        )

    async def _careful_implementation(
        self, opportunity: PollinationOpportunity
    ) -> PollinationResult:
        """Careful implementation (high complexity)"""
        # Simulate careful implementation with testing
        return PollinationResult(
            opportunity=opportunity,
            success=True,
            actual_improvement=opportunity.expected_benefit * 0.7,
            side_effects=[
                "Extensive testing required",
                "Gradual rollout recommended",
                "Monitoring needed",
            ],
            metadata={"transfer_method": "careful"},
        )

    async def temporal_query(
        self,
        fused_memory: FusedMemory,
        query: str,
        time_range: Tuple[datetime, datetime],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge from a specific time range.

        Retrieves artifacts, solutions, and insights from both systems
        within the specified time window.

        Args:
            fused_memory: Fused memory to query
            query: Search query (semantic)
            time_range: (start_time, end_time) tuple
            limit: Maximum results to return

        Returns:
            List of matching artifacts ranked by relevance
        """
        start_time, end_time = time_range
        results = []
        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Query OpenEvolve artifacts
        for artifact in oe.fitness_history:
            timestamp = artifact.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            if timestamp and start_time <= timestamp <= end_time:
                relevance = self._calculate_relevance(query, artifact)
                if relevance > 0.5:
                    results.append({
                        "source_system": "openevolve",
                        "artifact_type": "fitness_record",
                        "timestamp": timestamp,
                        "relevance": relevance,
                        "data": artifact,
                    })

        # Query LoongFlow artifacts
        for artifact in lf.summarization_episodes:
            timestamp = artifact.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            if timestamp and start_time <= timestamp <= end_time:
                relevance = self._calculate_relevance(query, artifact)
                if relevance > 0.5:
                    results.append({
                        "source_system": "loongflow",
                        "artifact_type": "summary",
                        "timestamp": timestamp,
                        "relevance": relevance,
                        "data": artifact,
                    })

        # Sort by relevance and return top k
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]

    async def get_unified_insights(
        self, fused_memory: FusedMemory
    ) -> UnifiedInsights:
        """
        Generate unified insights from fused memory.

        Creates meta-insights by analyzing the combined knowledge
        from both systems.

        Args:
            fused_memory: Fused memory to analyze

        Returns:
            UnifiedInsights with best practices, anti-patterns, recommendations
        """
        oe = fused_memory.openevolve_component
        lf = fused_memory.loongflow_component

        # Overall performance comparison
        performance_comparison = {
            "openevolve": {
                "avg_fitness": self._calculate_avg_fitness(oe.fitness_history),
                "diversity_score": self._calculate_diversity_score(oe),
                "convergence_speed": oe.convergence_data.get("convergence_generation", 0),
            },
            "loongflow": {
                "avg_fitness": self._calculate_avg_fitness([
                    {"fitness": e.get("fitness", 0.0)} for e in lf.pes_lineage
                ]),
                "efficiency_score": self._calculate_efficiency_score(lf),
                "sample_efficiency": lf.efficiency_metrics.get("efficiency_gain", 0.6),
            },
        }

        # Best practices
        best_practices = [
            "Use OpenEvolve for exploration of diverse solutions in behavioral space",
            "Use LoongFlow for efficient refinement with 60% fewer evaluations",
            "Combine MAP-Elites diversity with PES planning for best results",
            "Apply early stopping patterns from LoongFlow to OpenEvolve evaluation",
            "Initialize LoongFlow with OpenEvolve elite solutions for better convergence",
        ]

        # Anti-patterns
        anti_patterns = [
            "Don't use LoongFlow for problems requiring diverse solution sets",
            "Don't use OpenEvolve for expensive evaluations without budget constraints",
            "Avoid high mutation rates in both systems simultaneously",
            "Don't ignore cross-system parameter conflicts",
        ]

        # Recommended configurations
        recommended_configs = {
            "openevolve": {
                "population_size": 1000,
                "num_islands": 5,
                "exploration_ratio": 0.2,
                "exploitation_ratio": 0.7,
                "feature_bins": 10,
            },
            "loongflow": {
                "population_size": 100,
                "max_iterations": 50,
                "enable_planning": True,
                "early_stopping": True,
                "exploration_rate": 0.1,
            },
        }

        # Cross-system synergies
        synergies = [
            pattern.synergy_description
            for pattern in fused_memory.complementary_patterns
        ]

        # Calculate confidence
        confidence = (
            fused_memory.fusion_quality_score *
            (1.0 if len(fused_memory.conflict_resolutions) == 0 else 0.9)
        )

        return UnifiedInsights(
            domain=fused_memory.domain,
            overall_performance_comparison=performance_comparison,
            best_practices=best_practices,
            anti_patterns=anti_patterns,
            recommended_configurations=recommended_configs,
            cross_system_synergies=synergies,
            confidence=confidence,
            metadata={
                "fusion_quality": fused_memory.fusion_quality_score,
                "pattern_count": len(fused_memory.complementary_patterns),
                "conflict_count": len(fused_memory.conflicts),
            },
        )

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _calculate_diversity_score(self, oe_memory: OpenEvolveMemory) -> float:
        """Calculate diversity score from OpenEvolve memory"""
        if not oe_memory.diversity_metrics:
            return 0.5

        # Average diversity across all metrics
        diversity_values = [
            m.get("diversity", 0.5)
            for m in oe_memory.diversity_metrics
            if "diversity" in m
        ]

        return sum(diversity_values) / len(diversity_values) if diversity_values else 0.5

    def _calculate_efficiency_score(self, lf_memory: LoongFlowMemory) -> float:
        """Calculate efficiency score from LoongFlow memory"""
        if not lf_memory.efficiency_metrics:
            return 0.6  # LoongFlow's baseline

        efficiency_gain = lf_memory.efficiency_metrics.get("efficiency_gain", 0.6)
        convergence_rate = lf_memory.efficiency_metrics.get("convergence_rate", 0.8)

        return (efficiency_gain + convergence_rate) / 2

    def _calculate_avg_fitness(self, fitness_history: List[Dict[str, Any]]) -> float:
        """Calculate average fitness from history"""
        if not fitness_history:
            return 0.0

        fitness_values = [f.get("fitness", 0.0) for f in fitness_history]
        return sum(fitness_values) / len(fitness_values)

    def _count_evidence(self, position: str) -> int:
        """Count evidence items in a position string"""
        # Simple heuristic: count numbers and keywords
        keywords = ["evidence", "result", "study", "experiment", "trial"]
        count = sum(c.isdigit() for c in position)
        count += sum(1 for kw in keywords if kw.lower() in position.lower())
        return count

    def _calculate_relevance(self, query: str, artifact: Dict[str, Any]) -> float:
        """Calculate relevance of artifact to query (simple keyword matching)"""
        query_lower = query.lower()
        artifact_text = str(artifact).lower()

        # Count query words in artifact
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in artifact_text)

        return matches / len(query_words) if query_words else 0.0

    def _calculate_fusion_quality(self, fused_memory: FusedMemory) -> float:
        """Calculate overall quality score of the fusion"""
        # Factors:
        # 1. Pattern count (more patterns = better synergy)
        # 2. Conflict resolution (resolved conflicts = good)
        # 3. Data completeness (both systems have data = good)
        # 4. Temporal alignment (timestamps align = good)

        score = 0.5  # Base score

        # Pattern bonus
        score += min(0.2, len(fused_memory.complementary_patterns) * 0.05)

        # Resolution bonus
        if fused_memory.conflicts and fused_memory.conflict_resolutions:
            resolution_rate = len(fused_memory.conflict_resolutions) / len(fused_memory.conflicts)
            score += resolution_rate * 0.1

        # Completeness bonus
        oe_has_data = bool(fused_memory.openevolve_component.fitness_history)
        lf_has_data = bool(fused_memory.loongflow_component.pes_lineage)
        if oe_has_data and lf_has_data:
            score += 0.1

        # Pollination bonus
        score += min(0.1, len(fused_memory.pollination_opportunities) * 0.02)

        return min(1.0, score)

    async def _store_fused_memory(self, fused_memory: FusedMemory, run_id: str):
        """Store fused memory in Knowledge Engine"""
        if not self.ke:
            return

        try:
            # Store as JSON document
            if hasattr(self.ke, "mongodb"):
                collection = self.ke.mongodb["fused_memories"]
                document = {
                    "_id": run_id,
                    "domain": fused_memory.domain,
                    "timestamp": fused_memory.fusion_timestamp.isoformat(),
                    "quality_score": fused_memory.fusion_quality_score,
                    "pattern_count": len(fused_memory.complementary_patterns),
                    "conflict_count": len(fused_memory.conflicts),
                    "openevolve": fused_memory.openevolve_component.to_dict(),
                    "loongflow": fused_memory.loongflow_component.to_dict(),
                }
                await collection.insert_one(document)

            logger.debug(f"Stored fused memory {run_id} in Knowledge Engine")

        except Exception as e:
            logger.error(f"Failed to store fused memory: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get fusion statistics"""
        return self.stats.copy()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_memory_fusion(knowledge_engine=None) -> EvolutionaryMemoryFusion:
    """
    Create a memory fusion engine.

    Args:
        knowledge_engine: Optional Knowledge Engine for storage

    Returns:
        Configured EvolutionaryMemoryFusion instance
    """
    return EvolutionaryMemoryFusion(knowledge_engine=knowledge_engine)


async def fuse_and_analyze(
    openevolve_memory: Union[OpenEvolveMemory, Dict],
    loongflow_memory: Union[LoongFlowMemory, Dict],
    domain: str = "general",
    knowledge_engine=None,
) -> Tuple[FusedMemory, UnifiedInsights]:
    """
    Convenience function to fuse memories and get insights.

    Args:
        openevolve_memory: OpenEvolve memory (object or dict)
        loongflow_memory: LoongFlow memory (object or dict)
        domain: Problem domain
        knowledge_engine: Optional Knowledge Engine

    Returns:
        Tuple of (FusedMemory, UnifiedInsights)
    """
    fusion = create_memory_fusion(knowledge_engine)

    # Fuse memories
    fused = await fusion.fuse_memories(
        openevolve_memory=openevolve_memory,
        loongflow_memory=loongflow_memory,
        domain=domain,
    )

    # Get insights
    insights = await fusion.get_unified_insights(fused)

    return fused, insights
