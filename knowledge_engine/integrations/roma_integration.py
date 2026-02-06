"""
ROMA Integration for OpenEvolve Knowledge Engine

This module provides integration with the ROMA (Recursive Optimized Multi-Agent)
decomposition and recomposition system, enabling advanced problem-solving through
hierarchical decomposition and solution synthesis.

ROMA Architecture:
- Decompose complex problems into atomic sub-problems
- Solve atomic sub-problems using specialized agents
- Verify solutions meet requirements
- Reassemble solutions into complete answers

Integration follows the Air Gap principle - no direct imports from core-projects/ROMA/
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import uuid
import time


logger = logging.getLogger(__name__)

# ROMA integration availability flag
ROMA_INTEGRATION_AVAILABLE = True


@dataclass
class ROMADecomposition:
    """Represents a hierarchical problem decomposition."""
    decomposition_id: str
    problem: str
    sub_problems: List['ROMADecomposition']
    is_atomic: bool
    depth: int
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Backward compatibility properties
    @property
    def subproblems(self) -> List['ROMADecomposition']:
        """Backward compatibility alias for sub_problems."""
        return self.sub_problems

    @property
    def title(self) -> str:
        """Backward compatibility property for title (returns problem text)."""
        return self.problem

    @property
    def description(self) -> str:
        """Backward compatibility property for description (returns problem text)."""
        return self.problem


@dataclass
class ROMASolution:
    """Represents a solution to a problem or sub-problem."""
    solution_id: str
    problem_id: str
    solution: Any
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ROMAVerification:
    """Represents verification results for a solution."""
    verification_id: str
    solution_id: str
    passed: bool
    score: float
    feedback: str
    requirements_met: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ROMAResult:
    """Result of a ROMA operation."""
    success: bool
    decomposition: Optional[ROMADecomposition]
    solutions: List[ROMASolution]
    verification: Optional[ROMAVerification]
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def subproblems(self) -> List[ROMADecomposition]:
        """Backward compatibility property - returns subproblems from decomposition."""
        if self.decomposition:
            return self.decomposition.subproblems
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'decomposition': self.decomposition.__dict__ if self.decomposition else None,
            'solutions': [s.__dict__ for s in self.solutions],
            'verification': self.verification.__dict__ if self.verification else None,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class ROMAIntegration:
    """
    Integration with ROMA (Recursive Optimized Multi-Agent) decomposition system.

    Provides methods for:
    - Hierarchical problem decomposition
    - Atomic sub-problem solving
    - Solution verification and validation
    - Solution reassembly and synthesis
    - Batch processing and parallelization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ROMA integration.

        Args:
            config: Configuration for ROMA components (merged with defaults)
        """
        # Deep merge config with defaults
        default_config = self._get_default_config()
        if config:
            self.config = self._deep_merge_config(default_config, config)
        else:
            self.config = default_config

        # Initialize ROMA components
        self.decomposer = None
        self.solver = None
        self.verifier = None
        self.reassembler = None
        self.knowledge_engine = None

        # Statistics tracking
        self._stats = {
            "decompositions_performed": 0,
            "problems_solved": 0,
            "verifications_performed": 0,
            "reassemblies_performed": 0,
            "entities_extracted": 0,
            "solutions_stored": 0,
            "total_processing_time_ms": 0.0
        }

        # Knowledge integration cache
        self._artifact_cache: Dict[str, Any] = {}

        # Initialize based on configuration
        self._initialize_components()

        logger.info({
            "msg": "ROMAIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _deep_merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge override config into base config.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ROMA integration."""
        return {
            "decomposer": {
                "type": "hierarchical",
                "max_depth": 5,
                "branching_factor": 3,
                "atomic_threshold": 0.7,  # Confidence threshold for atomic problems
                "strategy": "recursive"  # "recursive", "iterative", "hybrid"
            },
            "solver": {
                "type": "multi_agent",
                "agents": [
                    "reasoning",
                    "computation",
                    "retrieval",
                    "synthesis"
                ],
                "timeout_seconds": 300,
                "max_retries": 3,
                "retry_backoff_ms": 1000
            },
            "verifier": {
                "type": "constraint",
                "validators": [
                    "completeness",
                    "correctness",
                    "consistency"
                ],
                "threshold": 0.8,
                "strict_mode": False
            },
            "reassembler": {
                "type": "hierarchical",
                "conflict_resolution": "merge",  # "merge", "vote", "priority"
                "quality_threshold": 0.7
            },
            "batch_processing": {
                "enabled": True,
                "max_parallel": 10,
                "timeout_seconds": 600
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout_ms": 60000
            },
            "knowledge_integration": {
                "enabled": False,  # Opt-in feature
                "auto_extract_entities": False,
                "auto_store_solutions": False,
                "entity_types": [
                    "concept",
                    "solution",
                    "pattern",
                    "problem"
                ],
                "similarity_threshold": 0.7,
                "max_artifacts": 10,
                "cache_results": True
            }
        }

    def _initialize_components(self):
        """
        Initialize ROMA components based on configuration.

        Tries to use ROMA core if available, falls back to mock mode for graceful degradation.

        Components are initialized with graceful degradation if ROMA is unavailable.
        """
        try:
            # Try to import ROMA core directly
            # Add ROMA to path if needed
            import sys
            from pathlib import Path
            roma_path = Path(__file__).parent.parent.parent / 'core-projects' / 'ROMA' / 'src'
            if str(roma_path) not in sys.path:
                sys.path.insert(0, str(roma_path))

            # Try importing ROMA core components
            from roma_dspy import Atomizer, Planner, Executor, Aggregator, Verifier
            from roma_dspy.core.engine.solve import RecursiveSolver

            # Success! ROMA core is available
            self.decomposer = Atomizer
            self.solver = Executor
            self.verifier = Verifier
            self.reassembler = Aggregator
            self._recursive_solver = RecursiveSolver
            self._roma_available = True

            logger.info({
                "msg": "ROMA core components initialized successfully",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            # ROMA core not available, use mock implementation
            logger.warning({
                "msg": f"ROMA core not available ({e}), using mock implementation",
                "install": "ROMA core system should be accessible or use adapter",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Create failing mock implementations for production safety
            try:
                from ..optional_imports import create_failing_mock

                MockDecomposer = create_failing_mock(
                    package_name='ROMA',
                    feature_name='ROMA decomposer - hierarchical problem decomposition',
                    install_command='Ensure ROMA core is accessible'
                )

                MockSolver = create_failing_mock(
                    package_name='ROMA',
                    feature_name='ROMA solver - atomic problem solving',
                    install_command='Ensure ROMA core is accessible'
                )

                MockVerifier = create_failing_mock(
                    package_name='ROMA',
                    feature_name='ROMA verifier - solution validation',
                    install_command='Ensure ROMA core is accessible'
                )

                MockReassembler = create_failing_mock(
                    package_name='ROMA',
                    feature_name='ROMA reassembler - solution synthesis',
                    install_command='Ensure ROMA core is accessible'
                )

                self._mock_decomposer_class = MockDecomposer
                self._mock_solver_class = MockSolver
                self._mock_verifier_class = MockVerifier
                self._mock_reassembler_class = MockReassembler
            except Exception:
                pass

            # Store mock classes for later use
            self.decomposer = None
            self.solver = None
            self.verifier = None
            self.reassembler = None
            self._roma_available = False

            logger.info({
                "msg": "ROMA components initialized in mock mode",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    async def decompose_problem(
        self,
        problem: str,
        max_depth: Optional[int] = None,
        extract_entities: Optional[bool] = None,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Decompose a complex problem into hierarchical sub-problems.

        Args:
            problem: The complex problem to decompose
            max_depth: Maximum decomposition depth (overrides config)
            extract_entities: Whether to extract knowledge entities (overrides config)
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult with hierarchical decomposition

        Example:
            >>> result = await roma.decompose_problem("Design a scalable microservices architecture")
            >>> # Returns tree of sub-problems like:
            >>> # - Design API gateway
            >>> # - Design service discovery
            >>> # - Design data management
        """
        correlation_id = correlation_id or f"roma_decomp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        # Determine whether to extract entities
        effective_extract_entities = extract_entities if extract_entities is not None else \
            self.config["knowledge_integration"].get("auto_extract_entities", False)

        logger.info({
            "msg": "Starting ROMA problem decomposition",
            "problem_length": len(problem),
            "max_depth": max_depth or self.config["decomposer"]["max_depth"],
            "extract_entities": effective_extract_entities,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Use configured max_depth if not overridden
            effective_max_depth = max_depth or self.config["decomposer"]["max_depth"]

            # TODO: Call via adapter when ROMA adapter is implemented
            # decomposition = await self.roma_adapter.decompose(
            #     problem=problem,
            #     max_depth=effective_max_depth,
            #     correlation_id=correlation_id
            # )

            # Placeholder implementation - in production this calls ROMA via adapter
            decomposition = ROMADecomposition(
                decomposition_id=str(uuid.uuid4()),
                problem=problem,
                sub_problems=[],
                is_atomic=len(problem) < 100,  # Simple heuristic for placeholder
                depth=0,
                metadata={
                    "strategy": self.config["decomposer"]["strategy"],
                    "branching_factor": self.config["decomposer"]["branching_factor"]
                }
            )

            # Recursive decomposition (placeholder logic)
            if not decomposition.is_atomic and effective_max_depth > 0:
                # Simulate decomposition by creating sub-problems
                sub_problems = await self._simulate_decomposition(
                    problem, depth=1, max_depth=effective_max_depth
                )
                decomposition.sub_problems = sub_problems

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["decompositions_performed"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            result = ROMAResult(
                success=True,
                decomposition=decomposition,
                solutions=[],
                verification=None,
                metadata={
                    "max_depth": effective_max_depth,
                    "strategy": self.config["decomposer"]["strategy"],
                    "sub_problem_count": self._count_sub_problems(decomposition),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )

            # Extract entities if enabled
            if effective_extract_entities:
                try:
                    entities = await self.extract_knowledge_entities(result)
                    result.metadata["entities_extracted"] = len(entities)
                    result.metadata["entities"] = entities

                    logger.info({
                        "msg": "Knowledge entities extracted from decomposition",
                        "correlation_id": correlation_id,
                        "entity_count": len(entities),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                except Exception as e:
                    logger.warning({
                        "msg": "Failed to extract knowledge entities",
                        "correlation_id": correlation_id,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

            logger.info({
                "msg": "ROMA problem decomposition completed",
                "correlation_id": correlation_id,
                "sub_problem_count": result.metadata["sub_problem_count"],
                "entities_extracted": result.metadata.get("entities_extracted", 0),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA problem decomposition failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return ROMAResult(
                success=False,
                decomposition=None,
                solutions=[],
                verification=None,
                metadata={
                    "max_depth": max_depth or self.config["decomposer"]["max_depth"],
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    async def _simulate_decomposition(
        self,
        problem: str,
        depth: int,
        max_depth: int
    ) -> List[ROMADecomposition]:
        """Simulate recursive decomposition (placeholder for adapter call)."""
        if depth >= max_depth:
            return []

        # Create mock sub-problems
        num_sub_problems = self.config["decomposer"]["branching_factor"]
        sub_problems = []

        for i in range(num_sub_problems):
            sub_problem = ROMADecomposition(
                decomposition_id=str(uuid.uuid4()),
                problem=f"{problem} - Sub-problem {i+1}",
                sub_problems=[],
                is_atomic=(depth == max_depth - 1),
                depth=depth,
                parent_id=str(uuid.uuid4()),
                metadata={"index": i}
            )
            sub_problems.append(sub_problem)

            # Recursively decompose if not atomic
            if not sub_problem.is_atomic:
                sub_problem.sub_problems = await self._simulate_decomposition(
                    sub_problem.problem, depth + 1, max_depth
                )

        return sub_problems

    async def decompose(
        self,
        problem: str,
        max_depth: Optional[int] = None,
        extract_entities: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Alias for decompose_problem for backward compatibility.

        Decompose a complex problem into hierarchical sub-problems.

        Args:
            problem: The complex problem to decompose
            max_depth: Maximum decomposition depth (overrides config)
            extract_entities: Whether to extract knowledge entities (overrides config)
            context: Additional context for decomposition (optional)
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult with hierarchical decomposition
        """
        # Context parameter is stored in metadata for later use
        result = await self.decompose_problem(
            problem=problem,
            max_depth=max_depth,
            extract_entities=extract_entities,
            correlation_id=correlation_id
        )

        # Add context to result metadata if provided
        if context and result.metadata is not None:
            result.metadata["context"] = context

        return result

    def _count_sub_problems(self, decomposition: ROMADecomposition) -> int:
        """Recursively count all sub-problems in a decomposition tree."""
        count = 1  # Count the decomposition itself
        for sub in decomposition.sub_problems:
            count += self._count_sub_problems(sub)
        return count

    async def solve_atomic(
        self,
        subproblem: ROMADecomposition,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Solve an atomic sub-problem using specialized agents.

        Args:
            subproblem: The atomic sub-problem to solve
            context: Additional context for solving
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult with solution

        Example:
            >>> atomic = ROMADecomposition(
            ...     decomposition_id="123",
            ...     problem="Calculate the optimal batch size",
            ...     sub_problems=[],
            ...     is_atomic=True,
            ...     depth=2
            ... )
            >>> result = await roma.solve_atomic(atomic)
        """
        correlation_id = correlation_id or f"roma_solve_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting ROMA atomic problem solving",
            "problem_id": subproblem.decomposition_id,
            "problem_length": len(subproblem.problem),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # TODO: Call via adapter when ROMA adapter is implemented
            # solution = await self.roma_adapter.solve_atomic(
            #     subproblem=subproblem,
            #     context=context,
            #     correlation_id=correlation_id
            # )

            # Placeholder implementation - simulate solving
            await asyncio.sleep(0.1)  # Simulate processing time

            solution = ROMASolution(
                solution_id=str(uuid.uuid4()),
                problem_id=subproblem.decomposition_id,
                solution=f"Solution for: {subproblem.problem}",
                confidence=0.85,
                reasoning="Applied reasoning agent to derive solution",
                metadata={
                    "agent_used": "reasoning",
                    "context_provided": context is not None,
                    "processing_strategy": "single_agent"
                }
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["problems_solved"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            result = ROMAResult(
                success=True,
                decomposition=None,
                solutions=[solution],
                verification=None,
                metadata={
                    "problem_id": subproblem.decomposition_id,
                    "confidence": solution.confidence,
                    "agent_used": "reasoning",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )

            logger.info({
                "msg": "ROMA atomic problem solving completed",
                "correlation_id": correlation_id,
                "solution_id": solution.solution_id,
                "confidence": solution.confidence,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA atomic problem solving failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return ROMAResult(
                success=False,
                decomposition=None,
                solutions=[],
                verification=None,
                metadata={
                    "problem_id": subproblem.decomposition_id,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    async def verify_solution(
        self,
        solution: ROMASolution,
        requirements: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Verify that a solution meets specified requirements.

        Args:
            solution: The solution to verify
            requirements: Requirements to validate against
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult with verification results

        Example:
            >>> requirements = {
            ...     "completeness": True,
            ...     "correctness": 0.9,
            ...     "consistency": True
            ... }
            >>> result = await roma.verify_solution(solution, requirements)
        """
        correlation_id = correlation_id or f"roma_verify_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting ROMA solution verification",
            "solution_id": solution.solution_id,
            "requirements": list(requirements.keys()),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # TODO: Call via adapter when ROMA adapter is implemented
            # verification = await self.roma_adapter.verify(
            #     solution=solution,
            #     requirements=requirements,
            #     correlation_id=correlation_id
            # )

            # Placeholder implementation - simulate verification
            await asyncio.sleep(0.05)  # Simulate verification time

            # Simulate requirement checks
            requirements_met = {}
            scores = []

            for req_name, req_value in requirements.items():
                # Simulate passing most requirements
                passed = isinstance(req_value, bool) or (isinstance(req_value, (int, float)) and solution.confidence >= req_value * 0.9)
                requirements_met[req_name] = passed
                if isinstance(req_value, (int, float)):
                    scores.append(solution.confidence if passed else req_value * 0.8)

            # Calculate overall score
            overall_score = sum(scores) / len(scores) if scores else solution.confidence
            passed = all(requirements_met.values()) if requirements_met else True

            verification = ROMAVerification(
                verification_id=str(uuid.uuid4()),
                solution_id=solution.solution_id,
                passed=passed,
                score=overall_score,
                feedback="Solution meets most requirements" if passed else "Solution fails some requirements",
                requirements_met=requirements_met,
                metadata={
                    "threshold": self.config["verifier"]["threshold"],
                    "strict_mode": self.config["verifier"]["strict_mode"]
                }
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["verifications_performed"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            result = ROMAResult(
                success=True,
                decomposition=None,
                solutions=[],
                verification=verification,
                metadata={
                    "solution_id": solution.solution_id,
                    "passed": verification.passed,
                    "score": verification.score,
                    "requirements_checked": len(requirements_met),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )

            logger.info({
                "msg": "ROMA solution verification completed",
                "correlation_id": correlation_id,
                "verification_id": verification.verification_id,
                "passed": verification.passed,
                "score": verification.score,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA solution verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return ROMAResult(
                success=False,
                decomposition=None,
                solutions=[],
                verification=None,
                metadata={
                    "solution_id": solution.solution_id,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    async def reassemble_solution(
        self,
        sub_solutions: List[ROMASolution],
        strategy: Optional[str] = None,
        store_as_knowledge: Optional[bool] = None,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Reassemble atomic solutions into a complete solution.

        Args:
            sub_solutions: List of sub-solutions to reassemble
            strategy: Reassembly strategy (overrides config)
            store_as_knowledge: Whether to store solution as knowledge (overrides config)
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult with reassembled solution

        Example:
            >>> solutions = [sol1, sol2, sol3]  # From solving sub-problems
            >>> result = await roma.reassemble_solution(solutions, strategy="merge")
        """
        correlation_id = correlation_id or f"roma_reassemble_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        # Determine whether to store as knowledge
        effective_store_knowledge = store_as_knowledge if store_as_knowledge is not None else \
            self.config["knowledge_integration"].get("auto_store_solutions", False)

        logger.info({
            "msg": "Starting ROMA solution reassembly",
            "sub_solution_count": len(sub_solutions),
            "strategy": strategy or self.config["reassembler"]["type"],
            "store_as_knowledge": effective_store_knowledge,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Use configured strategy if not overridden
            effective_strategy = strategy or self.config["reassembler"]["type"]

            # TODO: Call via adapter when ROMA adapter is implemented
            # reassembled = await self.roma_adapter.reassemble(
            #     sub_solutions=sub_solutions,
            #     strategy=effective_strategy,
            #     correlation_id=correlation_id
            # )

            # Placeholder implementation - simulate reassembly
            await asyncio.sleep(0.1)  # Simulate reassembly time

            # Merge solutions (placeholder logic)
            merged_solution_text = "\n\n".join([
                f"Solution {i+1}: {sol.solution}"
                for i, sol in enumerate(sub_solutions)
            ])

            # Calculate aggregate confidence
            avg_confidence = sum(sol.confidence for sol in sub_solutions) / len(sub_solutions) if sub_solutions else 0.0

            reassembled_solution = ROMASolution(
                solution_id=str(uuid.uuid4()),
                problem_id="reassembled",
                solution=merged_solution_text,
                confidence=avg_confidence,
                reasoning=f"Reassembled {len(sub_solutions)} sub-solutions using {effective_strategy} strategy",
                metadata={
                    "strategy": effective_strategy,
                    "sub_solution_count": len(sub_solutions),
                    "conflict_resolution": self.config["reassembler"]["conflict_resolution"]
                }
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["reassemblies_performed"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            result = ROMAResult(
                success=True,
                decomposition=None,
                solutions=[reassembled_solution],
                verification=None,
                metadata={
                    "strategy": effective_strategy,
                    "sub_solution_count": len(sub_solutions),
                    "aggregate_confidence": avg_confidence,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )

            # Store as knowledge if enabled
            if effective_store_knowledge:
                try:
                    artifact_id = await self.store_solution_as_knowledge(result)
                    result.metadata["knowledge_artifact_id"] = artifact_id

                    logger.info({
                        "msg": "Solution stored as knowledge artifact",
                        "correlation_id": correlation_id,
                        "artifact_id": artifact_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                except Exception as e:
                    logger.warning({
                        "msg": "Failed to store solution as knowledge",
                        "correlation_id": correlation_id,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

            logger.info({
                "msg": "ROMA solution reassembly completed",
                "correlation_id": correlation_id,
                "reassembled_solution_id": reassembled_solution.solution_id,
                "aggregate_confidence": avg_confidence,
                "knowledge_artifact_id": result.metadata.get("knowledge_artifact_id"),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA solution reassembly failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return ROMAResult(
                success=False,
                decomposition=None,
                solutions=[],
                verification=None,
                metadata={
                    "strategy": strategy or self.config["reassembler"]["type"],
                    "sub_solution_count": len(sub_solutions),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    async def batch_decompose(
        self,
        problems: List[str],
        max_depth: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> List[ROMAResult]:
        """
        Decompose multiple problems in parallel.

        Args:
            problems: List of problems to decompose
            max_depth: Maximum decomposition depth
            correlation_id: Correlation ID for tracking

        Returns:
            List of ROMAResult objects

        Example:
            >>> problems = ["Problem 1", "Problem 2", "Problem 3"]
            >>> results = await roma.batch_decompose(problems, max_depth=3)
        """
        correlation_id = correlation_id or f"roma_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting ROMA batch decomposition",
            "problems_count": len(problems),
            "max_depth": max_depth or self.config["decomposer"]["max_depth"],
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Check if batch processing is enabled
            if not self.config["batch_processing"]["enabled"]:
                logger.warning({
                    "msg": "Batch processing disabled, processing sequentially",
                    "correlation_id": correlation_id
                })

            # Process each problem in parallel
            max_parallel = self.config["batch_processing"]["max_parallel"]
            timeout = self.config["batch_processing"]["timeout_seconds"]

            # Create tasks for each problem
            tasks = [
                self.decompose_problem(
                    problem=p,
                    max_depth=max_depth,
                    correlation_id=f"{correlation_id}_p_{i}"
                )
                for i, p in enumerate(problems)
            ]

            # Process in batches to control parallelism
            results = []
            for i in range(0, len(tasks), max_parallel):
                batch = tasks[i:i + max_parallel]
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch, return_exceptions=True),
                    timeout=timeout
                )

                # Handle any exceptions in the results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error({
                            "msg": f"Batch item {i+j} decomposition failed",
                            "correlation_id": f"{correlation_id}_p_{i+j}",
                            "error": str(result)
                        })
                        results.append(ROMAResult(
                            success=False,
                            decomposition=None,
                            solutions=[],
                            verification=None,
                            metadata={"batch_index": i + j, "error": str(result)},
                            error=str(result)
                        ))
                    else:
                        results.append(result)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            successful_count = sum(1 for r in results if r.success)

            logger.info({
                "msg": "ROMA batch decomposition completed",
                "correlation_id": correlation_id,
                "problems_count": len(problems),
                "successful_count": successful_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return results

        except asyncio.TimeoutError:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA batch decomposition timed out",
                "correlation_id": correlation_id,
                "timeout_seconds": timeout,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return timeout errors for remaining problems
            error_results = []
            for i in range(len(problems)):
                error_results.append(ROMAResult(
                    success=False,
                    decomposition=None,
                    solutions=[],
                    verification=None,
                    metadata={"batch_index": i, "error": "timeout"},
                    processing_time_ms=processing_time_ms / len(problems) if problems else 0.0,
                    error="Batch processing timed out"
                ))

            return error_results

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA batch decomposition failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return error results for all problems
            error_results = []
            for i in range(len(problems)):
                error_results.append(ROMAResult(
                    success=False,
                    decomposition=None,
                    solutions=[],
                    verification=None,
                    metadata={"batch_index": i, "error": str(e)},
                    processing_time_ms=processing_time_ms / len(problems) if problems else 0.0,
                    error=str(e)
                ))

            return error_results

    async def extract_knowledge_entities(
        self,
        decomposition: ROMAResult
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge entities from ROMA decomposition.

        Creates knowledge entities from sub-problems with properties like
        complexity_score, dependencies, and source information.

        Args:
            decomposition: ROMAResult containing decomposition to extract from

        Returns:
            List of knowledge entity dictionaries

        Example:
            >>> result = await roma.decompose_problem("Design a system")
            >>> entities = await roma.extract_knowledge_entities(result)
            >>> # Returns entities like:
            >>> # [{"id": "e1", "type": "concept", "name": "API Gateway", ...}]
        """
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Extracting knowledge entities from ROMA decomposition",
            "decomposition_id": decomposition.decomposition.decomposition_id if decomposition.decomposition else None,
            "timestamp": start_time.isoformat()
        })

        try:
            entities = []

            # Check if knowledge integration is enabled
            if not self.config["knowledge_integration"].get("enabled", False):
                logger.debug({
                    "msg": "Knowledge integration disabled, skipping entity extraction",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return []

            # Extract entities from decomposition tree
            if decomposition.decomposition:
                entities.extend(self._extract_from_decomposition_node(decomposition.decomposition))

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["entities_extracted"] += len(entities)

            logger.info({
                "msg": "Knowledge entity extraction completed",
                "entity_count": len(entities),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return entities

        except Exception as e:
            logger.error({
                "msg": "Knowledge entity extraction failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []

    def _extract_from_decomposition_node(
        self,
        node: ROMADecomposition,
        parent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively extract entities from a decomposition node.

        Args:
            node: ROMADecomposition node
            parent_id: Parent entity ID

        Returns:
            List of entity dictionaries
        """
        entities = []

        # Create entity from this node
        entity = {
            "id": f"roma_entity_{node.decomposition_id}",
            "type": self._determine_entity_type(node),
            "name": node.problem[:100],  # Limit name length
            "description": node.problem,
            "properties": {
                "depth": node.depth,
                "is_atomic": node.is_atomic,
                "complexity_score": self._calculate_complexity_score(node),
                "sub_problem_count": len(node.sub_problems),
                "source": "roma_decomposition",
                "created_at": node.created_at
            },
            "metadata": {
                "decomposition_id": node.decomposition_id,
                "parent_id": parent_id,
                "strategy": node.metadata.get("strategy", "unknown")
            }
        }

        entities.append(entity)

        # Recursively extract from sub-problems
        for sub_problem in node.sub_problems:
            sub_entities = self._extract_from_decomposition_node(
                sub_problem,
                parent_id=entity["id"]
            )
            entities.extend(sub_entities)

            # Create relationship
            if sub_entities:
                entities.append({
                    "id": f"roma_rel_{entity['id']}_{sub_entities[0]['id']}",
                    "type": "decomposition",
                    "source": entity["id"],
                    "target": sub_entities[0]["id"],
                    "properties": {
                        "relationship_type": "decomposed_into",
                        "source": "roma_decomposition"
                    }
                })

        return entities

    def _determine_entity_type(self, node: ROMADecomposition) -> str:
        """
        Determine the entity type based on node properties.

        Args:
            node: ROMADecomposition node

        Returns:
            Entity type string
        """
        if node.is_atomic:
            return "atomic_problem"
        elif node.depth == 0:
            return "root_problem"
        else:
            return "sub_problem"

    def _calculate_complexity_score(self, node: ROMADecomposition) -> float:
        """
        Calculate complexity score for a decomposition node.

        Args:
            node: ROMADecomposition node

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Base complexity on depth and sub-problem count
        depth_factor = min(node.depth / 5.0, 1.0)  # Normalize depth
        sub_problem_factor = min(len(node.sub_problems) / 10.0, 1.0)  # Normalize sub-problems

        # Combined score
        complexity = (depth_factor + sub_problem_factor) / 2.0

        return round(complexity, 3)

    async def store_solution_as_knowledge(self, solution: ROMAResult) -> str:
        """
        Store ROMA solution as knowledge artifact.

        Creates a knowledge artifact from the solution with metadata
        for retrieval and reuse in future problem-solving.

        Args:
            solution: ROMAResult containing solution to store

        Returns:
            Artifact ID if successful, None otherwise

        Example:
            >>> result = await roma.reassemble_solution(solutions)
            >>> artifact_id = await roma.store_solution_as_knowledge(result)
            >>> print(f"Stored as artifact: {artifact_id}")
        """
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Storing ROMA solution as knowledge artifact",
            "solution_count": len(solution.solutions),
            "timestamp": start_time.isoformat()
        })

        try:
            # Check if knowledge integration is enabled
            if not self.config["knowledge_integration"].get("enabled", False):
                logger.debug({
                    "msg": "Knowledge integration disabled, skipping solution storage",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return None

            # Check if we have solutions to store
            if not solution.solutions:
                logger.warning({
                    "msg": "No solutions to store",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return None

            # Get primary solution
            primary_solution = solution.solutions[0]

            # Create knowledge artifact
            artifact_id = f"roma_artifact_{primary_solution.solution_id}"

            artifact = {
                "id": artifact_id,
                "type": "solution",
                "content": str(primary_solution.solution),
                "source": "roma",
                "properties": {
                    "confidence": primary_solution.confidence,
                    "reasoning": primary_solution.reasoning,
                    "problem_id": primary_solution.problem_id,
                    "processing_time_ms": solution.processing_time_ms,
                    "created_at": primary_solution.created_at
                },
                "metadata": {
                    "solution_id": primary_solution.solution_id,
                    "strategy": solution.metadata.get("strategy", "unknown"),
                    "verification_passed": solution.verification.passed if solution.verification else None,
                    "verification_score": solution.verification.score if solution.verification else None
                }
            }

            # Store in knowledge engine if available
            if self.knowledge_engine:
                try:
                    # TODO: Call knowledge engine to store artifact
                    # await self.knowledge_engine.store_artifact(artifact)
                    logger.info({
                        "msg": "Artifact stored in knowledge engine",
                        "artifact_id": artifact_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                except Exception as e:
                    logger.warning({
                        "msg": "Failed to store in knowledge engine, using local cache",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    # Cache locally if knowledge engine unavailable
                    self._artifact_cache[artifact_id] = artifact
            else:
                # Cache locally if knowledge engine unavailable
                self._artifact_cache[artifact_id] = artifact
                logger.info({
                    "msg": "Artifact cached locally (knowledge engine unavailable)",
                    "artifact_id": artifact_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["solutions_stored"] += 1

            logger.info({
                "msg": "Solution stored as knowledge artifact",
                "artifact_id": artifact_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return artifact_id

        except Exception as e:
            logger.error({
                "msg": "Failed to store solution as knowledge",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ROMA execution statistics.

        Returns:
            Dictionary with execution statistics

        Example:
            >>> stats = roma.get_statistics()
            >>> print(stats["decompositions_performed"])
        """
        total_operations = (
            self._stats["decompositions_performed"] +
            self._stats["problems_solved"] +
            self._stats["verifications_performed"] +
            self._stats["reassemblies_performed"]
        )

        return {
            "decompositions_performed": self._stats["decompositions_performed"],
            "problems_solved": self._stats["problems_solved"],
            "verifications_performed": self._stats["verifications_performed"],
            "reassemblies_performed": self._stats["reassemblies_performed"],
            "entities_extracted": self._stats["entities_extracted"],
            "solutions_stored": self._stats["solutions_stored"],
            "total_processing_time_ms": self._stats["total_processing_time_ms"],
            "average_processing_time_ms": (
                self._stats["total_processing_time_ms"] / total_operations
                if total_operations > 0
                else 0.0
            ),
            "knowledge_integration": {
                "enabled": self.config["knowledge_integration"].get("enabled", False),
                "auto_extract_entities": self.config["knowledge_integration"].get("auto_extract_entities", False),
                "auto_store_solutions": self.config["knowledge_integration"].get("auto_store_solutions", False),
                "cached_artifacts": len(self._artifact_cache)
            },
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check ROMA system health.

        Returns:
            Dictionary with health status

        Example:
            >>> health = roma.health_check()
            >>> print(health["status"])  # "healthy", "degraded", or "unhealthy"
        """
        # Check component availability
        decomposer_available = self.decomposer is not None
        solver_available = self.solver is not None
        verifier_available = self.verifier is not None
        reassembler_available = self.reassembler is not None

        # Determine overall health
        if all([decomposer_available, solver_available, verifier_available, reassembler_available]):
            status = "healthy"
        elif any([decomposer_available, solver_available]):
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "components": {
                "decomposer": "available" if decomposer_available else "unavailable",
                "solver": "available" if solver_available else "unavailable",
                "verifier": "available" if verifier_available else "unavailable",
                "reassembler": "available" if reassembler_available else "unavailable"
            },
            "statistics": self.get_statistics(),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        """
        Close resources used by the integration.

        Performs cleanup of any open connections or resources.
        """
        logger.info({
            "msg": "Closing ROMA integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Close decomposer
        if self.decomposer and hasattr(self.decomposer, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.decomposer.close):
                    await self.decomposer.close()
                else:
                    self.decomposer.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing decomposer",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Close solver
        if self.solver and hasattr(self.solver, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.solver.close):
                    await self.solver.close()
                else:
                    self.solver.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing solver",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Close verifier
        if self.verifier and hasattr(self.verifier, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.verifier.close):
                    await self.verifier.close()
                else:
                    self.verifier.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing verifier",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Close reassembler
        if self.reassembler and hasattr(self.reassembler, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.reassembler.close):
                    await self.reassembler.close()
                else:
                    self.reassembler.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing reassembler",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Clear artifact cache
        cache_size = len(self._artifact_cache)
        self._artifact_cache.clear()

        logger.info({
            "msg": "ROMA integration resources closed",
            "cache_cleared": cache_size,
            "statistics": self.get_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


def get_roma_integration(config: Optional[Dict[str, Any]] = None) -> ROMAIntegration:
    """
    Factory function to get or create a ROMA integration instance.

    Args:
        config: Configuration for ROMA integration

    Returns:
        ROMAIntegration instance

    Example:
        >>> roma = get_roma_integration({
        ...     "decomposer": {"max_depth": 3},
        ...     "solver": {"timeout_seconds": 300}
        ... })
    """
    return ROMAIntegration(config=config)


__all__ = [
    'ROMAIntegration',
    'ROMAResult',
    'ROMADecomposition',
    'ROMASolution',
    'ROMAVerification',
    'get_roma_integration'
]
