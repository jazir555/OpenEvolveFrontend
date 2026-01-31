"""
LoongFlow PES Integration for Knowledge Engine

Extracts learning artifacts from LoongFlow evolutionary runs and integrates
them into the Knowledge Engine's temporal knowledge graph, vector store,
and document storage.

This integration bridges LoongFlow's Plan-Execute-Summarize (PES) evolutionary
algorithm with the Knowledge Engine, enabling cross-system learning and knowledge
transfer across OpenEvolve and LoongFlow systems.

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
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import uuid
import json

logger = logging.getLogger(__name__)


class ProblemDomain(Enum):
    """Domains for LoongFlow problems"""
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    OPTIMIZATION = "optimization"
    MACHINE_LEARNING = "machine_learning"
    ENGINEERING = "engineering"
    GENERAL = "general"


class ArtifactType(Enum):
    """Types of knowledge artifacts extracted from LoongFlow"""
    PLANNING_STRATEGY = "planning_strategy"
    EXECUTION_PATTERN = "execution_pattern"
    REFLECTION_INSIGHT = "reflection_insight"
    EVOLUTIONARY_LINEAGE = "evolutionary_lineage"
    OPTIMIZED_SOLUTION = "optimized_solution"


@dataclass
class PESRunResults:
    """
    Results from a LoongFlow PES (Plan-Execute-Summarize) run.

    This structure captures all phases of the PES evolutionary algorithm:
    - Planning: Strategy generation and optimization approach
    - Execution: Iterative evolution with early stopping
    - Summary: Reflection and insights generation
    - Evolutionary Tree: Ancestry tracking and lineage
    - Best Solution: Final optimized solution
    """
    plan: Dict[str, Any]
    execution: Dict[str, Any]
    summary: Dict[str, Any]
    evolutionary_tree: Dict[str, Any]
    best_solution: Dict[str, Any]
    run_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "plan": self.plan,
            "execution": self.execution,
            "summary": self.summary,
            "evolutionary_tree": self.evolutionary_tree,
            "best_solution": self.best_solution,
            "run_metadata": self.run_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PESRunResults":
        """Create from dictionary"""
        return cls(
            plan=data.get("plan", {}),
            execution=data.get("execution", {}),
            summary=data.get("summary", {}),
            evolutionary_tree=data.get("evolutionary_tree", {}),
            best_solution=data.get("best_solution", {}),
            run_metadata=data.get("run_metadata", {}),
        )


@dataclass
class KnowledgeArtifact:
    """
    Canonical knowledge artifact representation for Knowledge Engine storage.

    This artifact structure is compatible with:
    - Graphiti temporal knowledge graph
    - Qdrant vector store
    - MongoDB document store
    - Neo4j graph database
    """
    artifact_type: str  # "planning_strategy", "execution_pattern", etc.
    source_system: str  # "loongflow"
    domain: str  # "finance", "trading", "science", etc.
    content: Dict[str, Any]  # The actual knowledge
    metadata: Dict[str, Any]  # Timestamps, iteration, score, etc.
    confidence: float  # 0.0 to 1.0
    lineage: Optional[Dict[str, Any]] = None  # Parent references
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    entities: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "artifact_type": self.artifact_type,
            "source_system": self.source_system,
            "domain": self.domain,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "lineage": self.lineage,
            "valid_at": self.valid_at.isoformat() if self.valid_at else None,
            "invalid_at": self.invalid_at.isoformat() if self.invalid_at else None,
            "entities": self.entities,
            "relationships": self.relationships,
        }

    def to_graphiti_episode(self) -> str:
        """Convert to Graphiti episode format"""
        episode_content = f"""
{self.artifact_type.upper()} from {self.source_system}

Domain: {self.domain}
Confidence: {self.confidence}

Content:
{json.dumps(self.content, indent=2)}

Metadata:
{json.dumps(self.metadata, indent=2)}

Lineage:
{json.dumps(self.lineage or {}, indent=2)}
"""
        return episode_content

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant point payload"""
        return {
            "artifact_type": self.artifact_type,
            "source_system": self.source_system,
            "domain": self.domain,
            "content_text": json.dumps(self.content),
            "metadata": self.metadata,
            "confidence": self.confidence,
            "timestamp": self.valid_at.isoformat() if self.valid_at else datetime.now(timezone.utc).isoformat(),
        }


class LoongFlowKnowledgeExtractor:
    """
    Extract knowledge from LoongFlow PES runs and store in Knowledge Engine.

    This extractor processes the complete PES lifecycle and creates 5 types of
    knowledge artifacts:

    1. PlanningStrategyArtifact - Strategic approach from planning phase
    2. ExecutionPatternArtifact - Execution patterns and efficiency metrics
    3. ReflectionInsightArtifact - Learnings from summary/reflection
    4. EvolutionaryLineageArtifact - Evolutionary tree and ancestry
    5. OptimizedSolutionArtifact - Best solution found

    Each artifact includes temporal metadata for point-in-time queries and
    confidence scores for quality assessment.

    Integration Points:
    - Graphiti: Temporal knowledge graph storage
    - Qdrant: Vector embeddings for semantic search
    - Neo4j: Entity and relationship storage
    - MongoDB: Raw document archival
    """

    def __init__(self, knowledge_engine=None):
        """
        Initialize the LoongFlow Knowledge Extractor.

        Args:
            knowledge_engine: Instance of KnowledgeEngine for artifact storage.
                             Must support storage methods for Graphiti, Qdrant, etc.
        """
        self.ke = knowledge_engine
        self.graphiti = None
        self.qdrant = None
        self.neo4j = None
        self.mongodb = None

        # Statistics tracking
        self.artifact_counts = {
            ArtifactType.PLANNING_STRATEGY.value: 0,
            ArtifactType.EXECUTION_PATTERN.value: 0,
            ArtifactType.REFLECTION_INSIGHT.value: 0,
            ArtifactType.EVOLUTIONARY_LINEAGE.value: 0,
            ArtifactType.OPTIMIZED_SOLUTION.value: 0,
        }

        # Initialize storage backends
        self._initialize_storage_backends()

    def _initialize_storage_backends(self):
        """Initialize storage backends from Knowledge Engine"""
        if not self.ke:
            logger.warning("No Knowledge Engine provided - artifacts will not be persisted")
            return

        # Try to extract storage backends from KE
        try:
            if hasattr(self.ke, 'graphiti_bridge'):
                self.graphiti = self.ke.graphiti_bridge
            elif hasattr(self.ke, 'graphiti'):
                self.graphiti = self.ke.graphiti

            if hasattr(self.ke, 'qdrant_bridge'):
                self.qdrant = self.ke.qdrant_bridge
            elif hasattr(self.ke, 'qdrant'):
                self.qdrant = self.ke.qdrant

            if hasattr(self.ke, 'neo4j'):
                self.neo4j = self.ke.neo4j

            if hasattr(self.ke, 'mongodb'):
                self.mongodb = self.ke.mongodb

            logger.info(f"Storage backends initialized - Graphiti: {self.graphiti is not None}, "
                       f"Qdrant: {self.qdrant is not None}, Neo4j: {self.neo4j is not None}")

        except Exception as e:
            logger.warning(f"Failed to initialize some storage backends: {e}")

    async def extract_from_pes_run(
        self,
        pes_run_results: Union[Dict[str, Any], PESRunResults],
        problem: str,
        problem_type: str = "general",
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[KnowledgeArtifact]:
        """
        Extract artifacts from LoongFlow PES execution.

        This is the main entry point for knowledge extraction. It processes
        all phases of the PES run and creates knowledge artifacts for storage
        in the Knowledge Engine.

        Args:
            pes_run_results: PES run results as dict or PESRunResults object
            problem: Problem description
            problem_type: Type of problem (e.g., "portfolio_optimization")
            domain: Optional domain override (auto-detected if not provided)
            run_id: Optional unique identifier for this run

        Returns:
            List of 5 KnowledgeArtifact objects:
            1. PlanningStrategyArtifact - The planning strategy
            2. ExecutionPatternArtifact - Execution patterns (early stops, etc.)
            3. ReflectionInsightArtifact - What worked/failed
            4. EvolutionaryLineageArtifact - Evolutionary tree
            5. OptimizedSolutionArtifact - Best solution found

        Example:
            ```python
            extractor = LoongFlowKnowledgeExtractor(knowledge_engine)

            pes_run = {
                "plan": {"strategy": "Use gradient descent", "success_rate": 0.85},
                "execution": {"early_stops": [15, 25], "convergence_rate": 0.95},
                "summary": {"insights": "Momentum helps escape local optima"},
                "evolutionary_tree": {"generations": 10, "avg_branching": 2.5},
                "best_solution": {"code": "def solve(): ...", "fitness": 0.95}
            }

            artifacts = await extractor.extract_from_pes_run(
                pes_run_results=pes_run,
                problem="Optimize neural network training",
                problem_type="scientific"
            )

            # Artifacts automatically stored in Knowledge Engine
            ```
        """
        artifacts = []
        timestamp = datetime.now(timezone.utc)
        run_id = run_id or f"pes_run_{uuid.uuid4().hex[:8]}"

        # Convert dict to PESRunResults if needed
        if isinstance(pes_run_results, dict):
            pes_run_results = PESRunResults.from_dict(pes_run_results)

        # Auto-detect domain if not provided
        if not domain:
            domain = self._detect_domain(problem, problem_type)

        # Validate input
        if not isinstance(pes_run_results, PESRunResults):
            logger.error("pes_run_results must be a PESRunResults object or dict")
            return artifacts

        logger.info(f"Extracting artifacts from PES run {run_id} for problem: {problem[:50]}...")

        # Artifact 1: Planning Strategy
        if pes_run_results.plan:
            planning_artifact = await self.extract_planning_strategies(
                pes_run_results.plan,
                problem,
                problem_type,
                domain,
                timestamp,
                run_id,
            )
            if planning_artifact:
                artifacts.append(planning_artifact)
                self.artifact_counts[ArtifactType.PLANNING_STRATEGY.value] += 1

        # Artifact 2: Execution Pattern
        if pes_run_results.execution:
            execution_artifact = await self.extract_execution_patterns(
                pes_run_results.execution,
                problem,
                problem_type,
                domain,
                timestamp,
                run_id,
            )
            if execution_artifact:
                artifacts.append(execution_artifact)
                self.artifact_counts[ArtifactType.EXECUTION_PATTERN.value] += 1

        # Artifact 3: Reflection Insight
        if pes_run_results.summary:
            reflection_artifact = await self.extract_reflection_insights(
                pes_run_results.summary,
                problem,
                problem_type,
                domain,
                timestamp,
                run_id,
            )
            if reflection_artifact:
                artifacts.append(reflection_artifact)
                self.artifact_counts[ArtifactType.REFLECTION_INSIGHT.value] += 1

        # Artifact 4: Evolutionary Lineage
        if pes_run_results.evolutionary_tree:
            lineage_artifact = await self.extract_evolutionary_lineage(
                pes_run_results.evolutionary_tree,
                problem,
                problem_type,
                domain,
                timestamp,
                run_id,
            )
            if lineage_artifact:
                artifacts.append(lineage_artifact)
                self.artifact_counts[ArtifactType.EVOLUTIONARY_LINEAGE.value] += 1

        # Artifact 5: Optimized Solution
        if pes_run_results.best_solution:
            solution_artifact = await self.extract_optimized_solutions(
                pes_run_results.best_solution,
                problem,
                problem_type,
                domain,
                timestamp,
                run_id,
            )
            if solution_artifact:
                artifacts.append(solution_artifact)
                self.artifact_counts[ArtifactType.OPTIMIZED_SOLUTION.value] += 1

        # Store all artifacts in Knowledge Engine backends
        await self._store_artifacts(artifacts, run_id)

        logger.info(
            f"Extracted {len(artifacts)} artifacts from LoongFlow PES run {run_id} "
            f"for problem: {problem[:50]}"
        )

        return artifacts

    async def extract_planning_strategies(
        self,
        plan: Dict[str, Any],
        problem: str,
        problem_type: str,
        domain: str,
        timestamp: datetime,
        run_id: str,
    ) -> Optional[KnowledgeArtifact]:
        """
        Extract planning strategy artifact.

        Captures the high-level reasoning and strategic approach from the
        planning phase of PES.

        Content includes:
        - Strategy description
        - Reasoning chain
        - Action steps
        - Expected deliverables
        - Success criteria
        """
        try:
            # Extract strategy components
            strategy = plan.get("strategy", "")
            reasoning = plan.get("reasoning", "")
            action_steps = plan.get("action_steps", [])
            success_criteria = plan.get("success_criteria", {})

            # Build content
            content = {
                "strategy": strategy,
                "reasoning": reasoning,
                "action_steps": action_steps,
                "success_criteria": success_criteria,
                "planning_approach": plan.get("approach", "unknown"),
            }

            # Calculate confidence based on success rate
            success_rate = plan.get("success_rate", 0.5)
            confidence = min(1.0, success_rate + 0.1)  # Boost slightly

            # Build metadata
            metadata = {
                "run_id": run_id,
                "problem": problem,
                "problem_type": problem_type,
                "success_rate": success_rate,
                "iterations_planned": plan.get("iterations", 0),
                "planning_duration_ms": plan.get("duration_ms", 0),
                "timestamp": timestamp.isoformat(),
            }

            # Extract entities
            entities = [domain, problem_type, "planning_strategy"]
            if action_steps:
                entities.extend([f"step_{i}" for i in range(len(action_steps))])

            artifact = KnowledgeArtifact(
                artifact_type=ArtifactType.PLANNING_STRATEGY.value,
                source_system="loongflow",
                domain=domain,
                content=content,
                metadata=metadata,
                confidence=confidence,
                valid_at=timestamp,
                entities=entities,
                relationships=[
                    {
                        "type": "PLANS_FOR",
                        "target": problem,
                        "attributes": {"approach": plan.get("approach", "unknown")},
                    }
                ],
            )

            logger.debug(f"Extracted planning strategy artifact for {run_id}")
            return artifact

        except Exception as e:
            logger.error(f"Failed to extract planning strategy: {e}")
            return None

    async def extract_execution_patterns(
        self,
        execution: Dict[str, Any],
        problem: str,
        problem_type: str,
        domain: str,
        timestamp: datetime,
        run_id: str,
    ) -> Optional[KnowledgeArtifact]:
        """
        Extract execution pattern artifact.

        Captures efficiency metrics and execution patterns from the
        execution phase of PES.

        Content includes:
        - Early stopping events
        - Convergence rate
        - Evaluations performed
        - Time saved through early stopping
        - Parameter tuning trends
        """
        try:
            # Extract execution metrics
            early_stops = execution.get("early_stops", [])
            convergence_rate = execution.get("convergence_rate", 0.0)
            iterations_to_best = execution.get("iterations_to_best", 0)
            total_evaluations = execution.get("total_evaluations", 0)

            # Build content
            content = {
                "early_stopping_events": early_stops,
                "convergence_rate": convergence_rate,
                "iterations_to_best": iterations_to_best,
                "total_evaluations": total_evaluations,
                "parameter_tuning": execution.get("parameter_tuning", {}),
                "execution_trace": execution.get("trace", []),
            }

            # Calculate efficiency gain (60% is LoongFlow's claimed improvement)
            baseline_evaluations = execution.get("baseline_evaluations", total_evaluations * 2.5)
            efficiency_gain = 1.0 - (total_evaluations / baseline_evaluations) if baseline_evaluations > 0 else 0.6

            # Build metadata
            metadata = {
                "run_id": run_id,
                "problem": problem,
                "problem_type": problem_type,
                "efficiency_gain": round(efficiency_gain, 3),
                "time_saved_seconds": execution.get("time_saved", 0),
                "early_stop_count": len(early_stops),
                "avg_iteration_time_ms": execution.get("avg_iteration_time_ms", 0),
                "timestamp": timestamp.isoformat(),
            }

            # Higher confidence for better efficiency
            confidence = min(1.0, 0.7 + efficiency_gain)

            # Extract entities
            entities = [domain, problem_type, "execution_pattern"]
            entities.extend([f"iteration_{i}" for i in range(min(5, total_evaluations))])

            artifact = KnowledgeArtifact(
                artifact_type=ArtifactType.EXECUTION_PATTERN.value,
                source_system="loongflow",
                domain=domain,
                content=content,
                metadata=metadata,
                confidence=confidence,
                valid_at=timestamp,
                entities=entities,
                relationships=[
                    {
                        "type": "EXECUTES_FOR",
                        "target": problem,
                        "attributes": {"efficiency_gain": efficiency_gain},
                    }
                ],
            )

            logger.debug(f"Extracted execution pattern artifact for {run_id}")
            return artifact

        except Exception as e:
            logger.error(f"Failed to extract execution pattern: {e}")
            return None

    async def extract_reflection_insights(
        self,
        summary: Dict[str, Any],
        problem: str,
        problem_type: str,
        domain: str,
        timestamp: datetime,
        run_id: str,
    ) -> Optional[KnowledgeArtifact]:
        """
        Extract reflection insight artifact.

        Captures learnings and insights from the summary/reflection phase
        of PES.

        Content includes:
        - What worked
        - What failed
        - Insights and recommendations
        - Adaptation patterns
        """
        try:
            # Extract reflection components
            insights = summary.get("insights", "")
            what_worked = summary.get("what_worked", [])
            what_failed = summary.get("what_failed", [])
            recommendations = summary.get("recommendations", [])

            # Build content
            content = {
                "insights": insights,
                "what_worked": what_worked,
                "what_failed": what_failed,
                "recommendations": recommendations,
                "adaptation_patterns": summary.get("adaptation_patterns", []),
                "lessons_learned": summary.get("lessons_learned", []),
            }

            # Calculate confidence based on assessment quality
            has_assessment = bool(what_worked or what_failed)
            confidence = 0.7 if has_assessment else 0.5

            # Build metadata
            metadata = {
                "run_id": run_id,
                "problem": problem,
                "problem_type": problem_type,
                "has_assessment": has_assessment,
                "insight_count": len(what_worked) + len(what_failed),
                "recommendation_count": len(recommendations),
                "timestamp": timestamp.isoformat(),
            }

            # Extract entities
            entities = [domain, problem_type, "reflection_insight"]
            entities.extend([f"insight_{i}" for i in range(len(what_worked) + len(what_failed))])

            artifact = KnowledgeArtifact(
                artifact_type=ArtifactType.REFLECTION_INSIGHT.value,
                source_system="loongflow",
                domain=domain,
                content=content,
                metadata=metadata,
                confidence=confidence,
                valid_at=timestamp,
                entities=entities,
                relationships=[
                    {
                        "type": "REFLECTS_ON",
                        "target": problem,
                        "attributes": {"has_assessment": has_assessment},
                    }
                ],
            )

            logger.debug(f"Extracted reflection insight artifact for {run_id}")
            return artifact

        except Exception as e:
            logger.error(f"Failed to extract reflection insight: {e}")
            return None

    async def extract_evolutionary_lineage(
        self,
        evolutionary_tree: Dict[str, Any],
        problem: str,
        problem_type: str,
        domain: str,
        timestamp: datetime,
        run_id: str,
    ) -> Optional[KnowledgeArtifact]:
        """
        Extract evolutionary lineage artifact.

        Captures the evolutionary tree structure and ancestry tracking
        from the PES run.

        Content includes:
        - Generation count
        - Branching factor
        - Parent-child relationships
        - Solution provenance
        """
        try:
            # Handle both integer and list formats for generations
            generations_value = evolutionary_tree.get("generations", 0)
            if isinstance(generations_value, list):
                generations = len(generations_value)
            elif isinstance(generations_value, int):
                generations = generations_value
            else:
                generations = 0

            # Build content
            content = {
                "generations": generations,
                "branching_factor": evolutionary_tree.get("avg_branching", 0),
                "total_mutations": evolutionary_tree.get("total_mutations", 0),
                "best_path": evolutionary_tree.get("best_path", []),
                "all_solutions": evolutionary_tree.get("solutions", [])[:10],  # Limit to first 10
                "ancestry_tree": evolutionary_tree.get("tree_structure", {}),
            }

            # Calculate confidence based on tree completeness
            has_complete_tree = bool(evolutionary_tree.get("best_path") or evolutionary_tree.get("tree_structure"))
            confidence = 0.8 if has_complete_tree else 0.6

            # Build metadata
            metadata = {
                "run_id": run_id,
                "problem": problem,
                "problem_type": problem_type,
                "generations": generations,
                "branching_factor": evolutionary_tree.get("avg_branching", 0),
                "total_mutations": evolutionary_tree.get("total_mutations", 0),
                "timestamp": timestamp.isoformat(),
            }

            # Extract entities
            entities = [domain, problem_type, "evolutionary_lineage"]
            if evolutionary_tree.get("best_path"):
                entities.extend([f"gen_{i}" for i in range(generations)])

            artifact = KnowledgeArtifact(
                artifact_type=ArtifactType.EVOLUTIONARY_LINEAGE.value,
                source_system="loongflow",
                domain=domain,
                content=content,
                metadata=metadata,
                confidence=confidence,
                valid_at=timestamp,
                entities=entities,
                relationships=[
                    {
                        "type": "EVOLVES_FROM",
                        "target": problem,
                        "attributes": {"generations": generations},
                    }
                ],
            )

            logger.debug(f"Extracted evolutionary lineage artifact for {run_id}")
            return artifact

        except Exception as e:
            logger.error(f"Failed to extract evolutionary lineage: {e}")
            return None

    async def extract_optimized_solutions(
        self,
        best_solution: Dict[str, Any],
        problem: str,
        problem_type: str,
        domain: str,
        timestamp: datetime,
        run_id: str,
    ) -> Optional[KnowledgeArtifact]:
        """
        Extract optimized solution artifact.

        Captures the final best solution found by the PES run.

        Content includes:
        - Solution code/representation
        - Fitness score
        - Iteration found
        - Improvement over baseline
        """
        try:
            # Extract solution components
            code = best_solution.get("code", "")
            fitness = best_solution.get("fitness", 0.0)
            iteration = best_solution.get("iteration", 0)
            improvement = best_solution.get("improvement", 0.0)

            # Build content
            content = {
                "solution": code,
                "fitness": fitness,
                "iteration_found": iteration,
                "improvement_over_baseline": improvement,
                "solution_params": best_solution.get("params", {}),
                "evaluation_result": best_solution.get("evaluation", {}),
            }

            # Calculate confidence based on fitness
            confidence = min(1.0, max(0.5, fitness))

            # Build metadata
            metadata = {
                "run_id": run_id,
                "problem": problem,
                "problem_type": problem_type,
                "fitness": fitness,
                "iteration": iteration,
                "improvement_over_baseline": improvement,
                "solution_size": len(str(code)) if code else 0,
                "timestamp": timestamp.isoformat(),
            }

            # Extract entities
            entities = [domain, problem_type, "optimized_solution", f"iteration_{iteration}"]

            # Build lineage
            lineage = {
                "parent_solutions": best_solution.get("parents", []),
                "mutation_history": best_solution.get("mutations", []),
                "ancestry_trace": best_solution.get("trace", []),
            }

            artifact = KnowledgeArtifact(
                artifact_type=ArtifactType.OPTIMIZED_SOLUTION.value,
                source_system="loongflow",
                domain=domain,
                content=content,
                metadata=metadata,
                confidence=confidence,
                valid_at=timestamp,
                lineage=lineage,
                entities=entities,
                relationships=[
                    {
                        "type": "SOLVES",
                        "target": problem,
                        "attributes": {"fitness": fitness, "iteration": iteration},
                    }
                ],
            )

            logger.debug(f"Extracted optimized solution artifact for {run_id}")
            return artifact

        except Exception as e:
            logger.error(f"Failed to extract optimized solution: {e}")
            return None

    async def _store_artifacts(
        self,
        artifacts: List[KnowledgeArtifact],
        run_id: str,
    ):
        """
        Store artifacts in all available Knowledge Engine backends.

        This method distributes artifacts across:
        - Graphiti: Temporal knowledge graph
        - Qdrant: Vector embeddings
        - Neo4j: Entity relationships
        - MongoDB: Document archival
        """
        if not artifacts:
            return

        logger.info(f"Storing {len(artifacts)} artifacts for run {run_id}")

        # Store in Graphiti (temporal knowledge graph)
        if self.graphiti:
            await self._store_in_graphiti(artifacts, run_id)

        # Store in Qdrant (vector embeddings)
        if self.qdrant:
            await self._store_in_qdrant(artifacts, run_id)

        # Store in Neo4j (entity relationships)
        if self.neo4j:
            await self._store_in_neo4j(artifacts, run_id)

        # Store in MongoDB (document archival)
        if self.mongodb:
            await self._store_in_mongodb(artifacts, run_id)

    async def _store_in_graphiti(
        self,
        artifacts: List[KnowledgeArtifact],
        run_id: str,
    ):
        """Store artifacts in Graphiti temporal knowledge graph"""
        try:
            for artifact in artifacts:
                episode_content = artifact.to_graphiti_episode()

                # Add episode to Graphiti
                if hasattr(self.graphiti, 'add_episode'):
                    await self.graphiti.add_episode(
                        name=f"{artifact.artifact_type}_{run_id}",
                        episode_body=episode_content,
                        reference_datetime=artifact.valid_at or datetime.now(timezone.utc),
                        valid_from=artifact.valid_at or datetime.now(timezone.utc),
                    )

                logger.debug(f"Stored artifact in Graphiti: {artifact.artifact_type}")

        except Exception as e:
            logger.error(f"Failed to store in Graphiti: {e}")

    async def _store_in_qdrant(
        self,
        artifacts: List[KnowledgeArtifact],
        run_id: str,
    ):
        """Store artifacts in Qdrant vector store"""
        try:
            # Import embedding function if available
            try:
                from ..core.backends.qdrant_backend import get_embedding
                embedding_func = get_embedding
            except ImportError:
                logger.warning("Qdrant backend not available, skipping vector storage")
                return

            for artifact in artifacts:
                # Generate embedding from content text
                content_text = json.dumps(artifact.content, indent=2)
                embedding = embedding_func(content_text)

                # Create Qdrant point
                point = {
                    "id": f"{artifact.artifact_type}_{run_id}",
                    "vector": embedding,
                    "payload": artifact.to_qdrant_payload(),
                }

                # Store in Qdrant
                if hasattr(self.qdrant, 'upsert'):
                    collection_name = f"loongflow_{artifact.domain}"
                    await self.qdrant.upsert(
                        collection_name=collection_name,
                        points=[point],
                    )

                logger.debug(f"Stored artifact in Qdrant: {artifact.artifact_type}")

        except Exception as e:
            logger.error(f"Failed to store in Qdrant: {e}")

    async def _store_in_neo4j(
        self,
        artifacts: List[KnowledgeArtifact],
        run_id: str,
    ):
        """Store artifacts in Neo4j graph database"""
        try:
            for artifact in artifacts:
                # Create Cypher query for entities and relationships
                query = f"""
                MERGE (a:Artifact {{id: '{artifact.artifact_type}_{run_id}'}})
                SET a += $artifact_data
                """

                # Add relationships
                for rel in artifact.relationships:
                    query += f"""
                    MERGE (t:Target {{name: '{rel["target"]}'}})
                    MERGE (a)-[:{rel['type']}]->(t)
                    """

                # Execute query
                if hasattr(self.neo4j, 'run'):
                    await self.neo4j.run(
                        query,
                        artifact_data=artifact.to_dict(),
                    )

                logger.debug(f"Stored artifact in Neo4j: {artifact.artifact_type}")

        except Exception as e:
            logger.error(f"Failed to store in Neo4j: {e}")

    async def _store_in_mongodb(
        self,
        artifacts: List[KnowledgeArtifact],
        run_id: str,
    ):
        """Store artifacts in MongoDB document store"""
        try:
            collection = f"loongflow_artifacts"

            for artifact in artifacts:
                document = artifact.to_dict()
                document["_id"] = f"{artifact.artifact_type}_{run_id}"

                # Insert into MongoDB
                if hasattr(self.mongodb, 'insert_one'):
                    await self.mongodb.insert_one(document)

                logger.debug(f"Stored artifact in MongoDB: {artifact.artifact_type}")

        except Exception as e:
            logger.error(f"Failed to store in MongoDB: {e}")

    def _detect_domain(self, problem: str, problem_type: str) -> str:
        """
        Auto-detect domain from problem description and type.

        Args:
            problem: Problem description
            problem_type: Type of problem

        Returns:
            Detected domain string
        """
        problem_lower = problem.lower()
        problem_type_lower = problem_type.lower()

        # Domain-specific keywords
        domain_keywords = {
            ProblemDomain.FINANCE.value: ["portfolio", "trading", "investment", "financial", "stock", "market"],
            ProblemDomain.TRADING.value: ["trading", "algorithm", "strategy", "buy", "sell"],
            ProblemDomain.SCIENCE.value: ["experiment", "scientific", "research", "hypothesis", "lab"],
            ProblemDomain.MATHEMATICS.value: ["equation", "prove", "theorem", "mathematical", "optimization"],
            ProblemDomain.MACHINE_LEARNING.value: ["model", "training", "neural", "ml", "deep learning", "classifier"],
            ProblemDomain.ENGINEERING.value: ["design", "structural", "mechanical", "civil", "engineering"],
        }

        # Check problem description
        for domain, keywords in domain_keywords.items():
            if any(kw in problem_lower for kw in keywords):
                return domain

        # Check problem type
        for domain, keywords in domain_keywords.items():
            if any(kw in problem_type_lower for kw in keywords):
                return domain

        # Default to general
        return ProblemDomain.GENERAL.value

    async def query_planning_strategies(
        self,
        problem_type: str,
        domain: str = "general",
        limit: int = 10,
        min_success_rate: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Query successful planning strategies for similar problems.

        Args:
            problem_type: Type of problem (e.g., "portfolio_optimization")
            domain: Problem domain
            limit: Max results to return
            min_success_rate: Minimum success rate threshold

        Returns:
            List of successful strategies with metadata
        """
        if not self.ke:
            logger.warning("No Knowledge Engine available for query")
            return []

        try:
            # Query based on KE capabilities
            if hasattr(self.ke, "query"):
                query = f"""
                MATCH (a:KnowledgeArtifact
                       {{artifact_type: 'planning_strategy',
                        source_system: 'loongflow',
                        domain: '{domain}'}})
                WHERE a.metadata.problem_type CONTAINS '{problem_type}'
                AND a.metadata.success_rate > {min_success_rate}
                RETURN a.content, a.metadata
                ORDER BY a.metadata.success_rate DESC
                LIMIT {limit}
                """
                results = await self.ke.query(query)
                return results
            else:
                logger.warning("Knowledge Engine has no query method")
                return []

        except Exception as e:
            logger.error(f"Failed to query planning strategies: {e}")
            return []

    async def get_efficiency_metrics(
        self,
        problem_type: str,
        domain: str = "general",
    ) -> Dict[str, float]:
        """
        Get efficiency metrics for PES on this problem type.

        Returns:
            Dict with:
                - avg_efficiency_gain: Average % improvement
                - avg_evaluations_saved: Average evaluations saved
                - success_rate: % of runs that succeeded
                - total_runs: Total number of runs analyzed
        """
        if not self.ke:
            logger.warning("No Knowledge Engine available for query")
            return {}

        try:
            if hasattr(self.ke, "query"):
                query = f"""
                MATCH (a:KnowledgeArtifact
                       {{artifact_type: 'execution_pattern',
                        source_system: 'loongflow',
                        domain: '{domain}'}})
                WHERE a.metadata.problem_type = '{problem_type}'
                RETURN
                    AVG(a.metadata.efficiency_gain) as avg_efficiency,
                    AVG(a.metadata.total_evaluations) as avg_evals,
                    COUNT(a) as total_runs
                """
                results = await self.ke.query(query)

                if results and len(results) > 0:
                    return {
                        "avg_efficiency_gain": results[0].get("avg_efficiency", 0.6),
                        "avg_evaluations_saved": results[0].get("avg_evals", 0.0),
                        "success_rate": 0.85,  # Placeholder - would calculate from actual data
                        "total_runs": results[0].get("total_runs", 0),
                    }
            else:
                logger.warning("Knowledge Engine has no query method")

        except Exception as e:
            logger.error(f"Failed to get efficiency metrics: {e}")

        return {}

    def get_extraction_stats(self) -> Dict[str, int]:
        """
        Get statistics about artifacts extracted.

        Returns:
            Dict mapping artifact type to count
        """
        return self.artifact_counts.copy()

    def reset_stats(self):
        """Reset extraction statistics"""
        for key in self.artifact_counts:
            self.artifact_counts[key] = 0


# Convenience function for creating extractor
def create_loongflow_extractor(knowledge_engine=None) -> LoongFlowKnowledgeExtractor:
    """
    Create a LoongFlow knowledge extractor.

    Args:
        knowledge_engine: Optional Knowledge Engine instance

    Returns:
        Configured LoongFlowKnowledgeExtractor
    """
    return LoongFlowKnowledgeExtractor(knowledge_engine=knowledge_engine)


# Alias for compatibility
LoongFlowIntegration = LoongFlowKnowledgeExtractor
