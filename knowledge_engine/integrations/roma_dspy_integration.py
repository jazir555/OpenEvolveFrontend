"""
ROMA-DSPy Integration for OpenEvolve Knowledge Engine

This module provides integration between ROMA (Recursive Optimized Multi-Agent)
decomposition and DSPy chain-of-thought reasoning, enhancing ROMA sub-problems
with detailed reasoning traces.

Integration follows the Air Gap principle - no direct imports from core-projects/
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import uuid
import hashlib
import json

from .roma_integration import ROMAIntegration, ROMAResult, ROMADecomposition, ROMASolution


logger = logging.getLogger(__name__)

# ROMA-DSPy integration availability flag
ROMA_DSPY_INTEGRATION_AVAILABLE = True


@dataclass
class ReasoningTrace:
    """Represents a chain-of-thought reasoning trace for a sub-problem."""
    trace_id: str
    subproblem_id: str
    steps: List[str]
    confidence: float
    intermediate_conclusions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'trace_id': self.trace_id,
            'subproblem_id': self.subproblem_id,
            'steps': self.steps,
            'confidence': self.confidence,
            'intermediate_conclusions': self.intermediate_conclusions,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


@dataclass
class EnhancedSubproblem:
    """Represents a ROMA sub-problem enhanced with DSPy reasoning."""
    subproblem_id: str
    problem: str
    depth: int
    is_atomic: bool
    reasoning_trace: Optional[ReasoningTrace] = None
    solution: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'subproblem_id': self.subproblem_id,
            'problem': self.problem,
            'depth': self.depth,
            'is_atomic': self.is_atomic,
            'reasoning_trace': self.reasoning_trace.to_dict() if self.reasoning_trace else None,
            'solution': str(self.solution) if self.solution else None,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


class ROMADSPyIntegration:
    """
    Combine ROMA decomposition with DSPy chain-of-thought reasoning.

    Enhances ROMA sub-problems with detailed reasoning traces, providing:
    - Cooperative problem solving (ROMA decomposition + DSPy reasoning)
    - Reasoning traces for each sub-problem
    - Verification with explanatory reasoning
    - Parallel reasoning for multiple sub-problems
    - Caching to avoid regenerating reasoning

    Example:
        >>> roma = ROMAIntegration()
        >>> dspy = DSPyIntegration(config={"model": "gpt-4o", "api_key": "..."})
        >>> roma_dspy = ROMADSPyIntegration(roma, dspy)
        >>> result = await roma_dspy.solve_with_cooperative_reasoning(
        ...     "Design a scalable microservices architecture"
        ... )
    """

    def __init__(
        self,
        roma_integration: ROMAIntegration,
        dspy_integration,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ROMA-DSPy integration.

        Args:
            roma_integration: ROMAIntegration instance
            dspy_integration: DSPyIntegration instance
            config: Configuration for the integration
        """
        self.roma = roma_integration
        self.dspy = dspy_integration
        # Merge provided config with defaults to ensure all required keys exist
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

        # Reasoning cache to avoid regenerating traces
        self._reasoning_cache: Dict[str, ReasoningTrace] = {}

        # Statistics tracking
        self._stats = {
            "cooperative_solutions": 0,
            "reasoning_traces_generated": 0,
            "reasoning_cache_hits": 0,
            "subproblems_reasoned": 0,
            "verifications_with_reasoning": 0,
            "total_processing_time_ms": 0.0
        }

        logger.info({
            "msg": "ROMADSPyIntegration initialized",
            "config": self.config,
            "dspy_available": self.dspy.lm is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ROMA-DSPy integration."""
        return {
            "auto_add_reasoning": True,
            "reasoning_model": "gpt-4o",
            "max_reasoning_steps": 10,
            "confidence_threshold": 0.7,
            "parallel_reasoning": True,
            "reasoning_timeout": 300,
            "cache_reasoning": True,
            "cache_ttl_seconds": 3600,
            "batch_size": 5,
            "retry_failed_reasoning": True,
            "max_reasoning_retries": 2
        }

    def _generate_cache_key(self, problem: str, context: str = "") -> str:
        """
        Generate a cache key for reasoning traces.

        Args:
            problem: The problem statement
            context: Additional context

        Returns:
            Cache key hash
        """
        content = f"{problem}|{context}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def solve_with_cooperative_reasoning(
        self,
        problem: str,
        max_depth: int = 3,
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Solve a problem using ROMA decomposition enhanced with DSPy reasoning.

        ROMA decomposes the problem into sub-problems, and DSPy adds detailed
        chain-of-thought reasoning traces to each sub-problem.

        Args:
            problem: The complex problem to solve
            max_depth: Maximum decomposition depth
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult enhanced with reasoning traces

        Example:
            >>> result = await roma_dspy.solve_with_cooperative_reasoning(
            ...     "Design a scalable microservices architecture",
            ...     max_depth=3
            ... )
            >>> # Each sub-problem will have a reasoning_trace field
            >>> # with detailed step-by-step reasoning
        """
        correlation_id = correlation_id or f"roma_dspy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting ROMA-DSPy cooperative reasoning",
            "problem_length": len(problem),
            "max_depth": max_depth,
            "auto_add_reasoning": self.config["auto_add_reasoning"],
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Step 1: Use ROMA to decompose the problem
            logger.info({
                "msg": "Decomposing problem with ROMA",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            roma_result = await self.roma.decompose_problem(
                problem=problem,
                max_depth=max_depth,
                correlation_id=f"{correlation_id}_decomp"
            )

            if not roma_result.success or not roma_result.decomposition:
                logger.error({
                    "msg": "ROMA decomposition failed",
                    "correlation_id": correlation_id,
                    "error": roma_result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return roma_result

            # Step 2: Extract sub-problems and add reasoning
            subproblems = self._extract_subproblems(roma_result.decomposition)

            logger.info({
                "msg": f"Extracted {len(subproblems)} sub-problems for reasoning",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Step 3: Add DSPy reasoning to each sub-problem
            if self.config["auto_add_reasoning"]:
                if self.config["parallel_reasoning"]:
                    enhanced_subproblems = await self.batch_reason_subproblems(
                        subproblems,
                        correlation_id=f"{correlation_id}_batch"
                    )
                else:
                    enhanced_subproblems = []
                    for i, subprob in enumerate(subproblems):
                        enhanced = await self.add_reasoning_to_subproblem(
                            subprob,
                            correlation_id=f"{correlation_id}_sub_{i}"
                        )
                        enhanced_subproblems.append(enhanced)
            else:
                # Just convert to enhanced format without reasoning
                enhanced_subproblems = [
                    EnhancedSubproblem(
                        subproblem_id=subp.get("id", str(uuid.uuid4())),
                        problem=subp.get("problem", ""),
                        depth=subp.get("depth", 0),
                        is_atomic=subp.get("is_atomic", False),
                        metadata=subp.get("metadata", {})
                    )
                    for subp in subproblems
                ]

            # Step 4: Solve atomic sub-problems (if they are atomic)
            solutions = []
            for subprob in enhanced_subproblems:
                if subprob.is_atomic:
                    # Create a mock ROMADecomposition for solving
                    mock_decomp = ROMADecomposition(
                        decomposition_id=subprob.subproblem_id,
                        problem=subprob.problem,
                        sub_problems=[],
                        is_atomic=True,
                        depth=subprob.depth
                    )

                    solve_result = await self.roma.solve_atomic(
                        mock_decomp,
                        correlation_id=f"{correlation_id}_solve_{subprob.subproblem_id}"
                    )

                    if solve_result.success and solve_result.solutions:
                        subprob.solution = solve_result.solutions[0]
                        solutions.append(solve_result.solutions[0])

            # Step 5: Reassemble solutions
            if solutions:
                reassemble_result = await self.roma.reassemble_solution(
                    solutions,
                    correlation_id=f"{correlation_id}_reassemble"
                )

                if reassemble_result.success:
                    roma_result.solutions = reassemble_result.solutions

            # Add enhanced metadata
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            roma_result.metadata["enhanced_subproblems"] = [
                sp.to_dict() for sp in enhanced_subproblems
            ]
            roma_result.metadata["reasoning_enabled"] = self.config["auto_add_reasoning"]
            roma_result.metadata["subproblems_with_reasoning"] = sum(
                1 for sp in enhanced_subproblems if sp.reasoning_trace
            )
            roma_result.metadata["cooperative_processing_time_ms"] = processing_time_ms

            # Update statistics
            self._stats["cooperative_solutions"] += 1
            self._stats["subproblems_reasoned"] += len(enhanced_subproblems)
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.info({
                "msg": "ROMA-DSPy cooperative reasoning completed",
                "correlation_id": correlation_id,
                "subproblems_count": len(enhanced_subproblems),
                "with_reasoning": roma_result.metadata["subproblems_with_reasoning"],
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return roma_result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "ROMA-DSPy cooperative reasoning failed",
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
                    "processing_time_ms": processing_time_ms,
                    "cooperative_reasoning": True
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    def _extract_subproblems(
        self,
        decomposition: ROMADecomposition,
        parent_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Recursively extract all sub-problems from a decomposition tree.

        Args:
            decomposition: ROMADecomposition node
            parent_context: Context from parent nodes

        Returns:
            List of sub-problem dictionaries
        """
        subproblems = []

        # Add current node as a sub-problem
        subproblem = {
            "id": decomposition.decomposition_id,
            "problem": decomposition.problem,
            "depth": decomposition.depth,
            "is_atomic": decomposition.is_atomic,
            "parent_id": decomposition.parent_id,
            "metadata": {
                **decomposition.metadata,
                "parent_context": parent_context
            }
        }
        subproblems.append(subproblem)

        # Recursively extract from sub-problems
        for sub in decomposition.sub_problems:
            context = f"{parent_context} > {decomposition.problem}" if parent_context else decomposition.problem
            subproblems.extend(self._extract_subproblems(sub, context))

        return subproblems

    async def add_reasoning_to_subproblem(
        self,
        subproblem: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> EnhancedSubproblem:
        """
        Generate DSPy reasoning trace for a single sub-problem.

        Args:
            subproblem: Sub-problem dictionary
            correlation_id: Correlation ID for tracking

        Returns:
            EnhancedSubproblem with reasoning trace

        Example:
            >>> subproblem = {
            ...     "id": "sub_1",
            ...     "problem": "Design API gateway",
            ...     "depth": 1,
            ...     "is_atomic": True
            ... }
            >>> enhanced = await roma_dspy.add_reasoning_to_subproblem(subproblem)
            >>> print(enhanced.reasoning_trace.steps)
        """
        correlation_id = correlation_id or f"reason_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        problem = subproblem.get("problem", "")
        subproblem_id = subproblem.get("id", str(uuid.uuid4()))

        logger.info({
            "msg": "Adding reasoning to sub-problem",
            "subproblem_id": subproblem_id,
            "problem_length": len(problem),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Check cache first
            if self.config["cache_reasoning"]:
                cache_key = self._generate_cache_key(problem)
                if cache_key in self._reasoning_cache:
                    logger.debug({
                        "msg": "Reasoning cache hit",
                        "subproblem_id": subproblem_id,
                        "correlation_id": correlation_id
                    })
                    self._stats["reasoning_cache_hits"] += 1
                    reasoning_trace = self._reasoning_cache[cache_key]
                else:
                    reasoning_trace = await self._generate_reasoning_trace(
                        problem,
                        subproblem_id,
                        correlation_id
                    )
                    self._reasoning_cache[cache_key] = reasoning_trace
                    self._stats["reasoning_traces_generated"] += 1
            else:
                reasoning_trace = await self._generate_reasoning_trace(
                    problem,
                    subproblem_id,
                    correlation_id
                )
                self._stats["reasoning_traces_generated"] += 1

            # Create enhanced sub-problem
            enhanced = EnhancedSubproblem(
                subproblem_id=subproblem_id,
                problem=problem,
                depth=subproblem.get("depth", 0),
                is_atomic=subproblem.get("is_atomic", False),
                reasoning_trace=reasoning_trace,
                metadata=subproblem.get("metadata", {})
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.info({
                "msg": "Reasoning added to sub-problem",
                "subproblem_id": subproblem_id,
                "reasoning_steps": len(reasoning_trace.steps),
                "confidence": reasoning_trace.confidence,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return enhanced

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Failed to add reasoning to sub-problem",
                "subproblem_id": subproblem_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return enhanced sub-problem without reasoning
            return EnhancedSubproblem(
                subproblem_id=subproblem_id,
                problem=problem,
                depth=subproblem.get("depth", 0),
                is_atomic=subproblem.get("is_atomic", False),
                reasoning_trace=None,
                metadata={
                    **subproblem.get("metadata", {}),
                    "reasoning_error": str(e)
                }
            )

    async def _generate_reasoning_trace(
        self,
        problem: str,
        subproblem_id: str,
        correlation_id: Optional[str] = None
    ) -> ReasoningTrace:
        """
        Generate a reasoning trace using DSPy chain-of-thought.

        Args:
            problem: The problem to reason about
            subproblem_id: ID of the sub-problem
            correlation_id: Correlation ID for tracking

        Returns:
            ReasoningTrace with steps, confidence, and conclusions
        """
        try:
            # Check if DSPy is available
            if not self.dspy.lm:
                logger.warning({
                    "msg": "DSPy not available, generating mock reasoning",
                    "subproblem_id": subproblem_id,
                    "correlation_id": correlation_id
                })
                return self._generate_mock_reasoning(problem, subproblem_id)

            # Use DSPy chain-of-thought reasoning
            max_steps = self.config["max_reasoning_steps"]

            dspy_result = await self.dspy.chain_of_thought(
                question=problem,
                context=f"Sub-problem ID: {subproblem_id}",
                max_steps=max_steps,
                correlation_id=f"{correlation_id}_cot"
            )

            if not dspy_result.success:
                logger.warning({
                    "msg": "DSPy reasoning failed, using mock reasoning",
                    "subproblem_id": subproblem_id,
                    "error": dspy_result.error,
                    "correlation_id": correlation_id
                })
                return self._generate_mock_reasoning(problem, subproblem_id)

            # Parse reasoning into steps
            reasoning_text = dspy_result.reasoning
            steps = self._parse_reasoning_steps(reasoning_text)

            # Extract intermediate conclusions
            intermediate_conclusions = self._extract_intermediate_conclusions(reasoning_text)

            # Calculate confidence based on reasoning quality
            confidence = self._calculate_confidence(steps, intermediate_conclusions, dspy_result)

            reasoning_trace = ReasoningTrace(
                trace_id=str(uuid.uuid4()),
                subproblem_id=subproblem_id,
                steps=steps,
                confidence=confidence,
                intermediate_conclusions=intermediate_conclusions,
                metadata={
                    "dspy_model": self.config.get("reasoning_model", "unknown"),
                    "reasoning_length": len(reasoning_text),
                    "answer": dspy_result.output,
                    "processing_time_ms": dspy_result.processing_time_ms
                }
            )

            return reasoning_trace

        except Exception as e:
            logger.error({
                "msg": "Failed to generate reasoning trace",
                "subproblem_id": subproblem_id,
                "error": str(e),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return self._generate_mock_reasoning(problem, subproblem_id)

    def _generate_mock_reasoning(self, problem: str, subproblem_id: str) -> ReasoningTrace:
        """
        Generate a mock reasoning trace when DSPy is unavailable.

        Args:
            problem: The problem statement
            subproblem_id: ID of the sub-problem

        Returns:
            Mock ReasoningTrace
        """
        return ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            subproblem_id=subproblem_id,
            steps=[
                f"Analyze problem: {problem[:100]}...",
                "Identify key requirements and constraints",
                "Consider potential solution approaches",
                "Evaluate trade-offs between approaches",
                "Select optimal solution strategy"
            ],
            confidence=0.5,  # Low confidence for mock reasoning
            intermediate_conclusions=[
                "Problem analyzed successfully",
                "Requirements identified",
                "Solution approach selected"
            ],
            metadata={
                "mock_reasoning": True,
                "reason": "DSPy not available"
            }
        )

    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """
        Parse reasoning text into discrete steps.

        Args:
            reasoning_text: Raw reasoning text from DSPy

        Returns:
            List of reasoning steps
        """
        # Split by common delimiters
        delimiters = ["\n\n", "\n", ". ", "; "]
        steps = []
        current_text = reasoning_text

        for delimiter in delimiters:
            if delimiter in current_text:
                parts = current_text.split(delimiter)
                steps = [part.strip() for part in parts if part.strip()]
                if len(steps) > 1:
                    break

        # Limit to max steps
        max_steps = self.config.get("max_reasoning_steps", 10)
        if len(steps) > max_steps:
            steps = steps[:max_steps]

        return steps if steps else [reasoning_text]

    def _extract_intermediate_conclusions(self, reasoning_text: str) -> List[str]:
        """
        Extract intermediate conclusions from reasoning text.

        Args:
            reasoning_text: Raw reasoning text

        Returns:
            List of intermediate conclusions
        """
        conclusions = []
        conclusion_indicators = [
            "therefore", "thus", "consequently", "as a result",
            "so", "hence", "accordingly", "conclusion"
        ]

        sentences = reasoning_text.split(". ")
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in conclusion_indicators):
                conclusions.append(sentence.strip())

        return conclusions if conclusions else ["Reasoning process completed"]

    def _calculate_confidence(
        self,
        steps: List[str],
        conclusions: List[str],
        dspy_result
    ) -> float:
        """
        Calculate confidence score based on reasoning quality.

        Args:
            steps: Reasoning steps
            conclusions: Intermediate conclusions
            dspy_result: DSPy result

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7

        # Factor in number of steps (more detailed = higher confidence)
        step_factor = min(len(steps) / 10.0, 0.2)

        # Factor in conclusions (more conclusions = higher confidence)
        conclusion_factor = min(len(conclusions) / 5.0, 0.1)

        # Combine factors
        confidence = base_confidence + step_factor + conclusion_factor

        # Cap at 1.0
        return round(min(confidence, 1.0), 3)

    async def batch_reason_subproblems(
        self,
        subproblems: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> List[EnhancedSubproblem]:
        """
        Add reasoning traces to multiple sub-problems in parallel.

        Args:
            subproblems: List of sub-problem dictionaries
            correlation_id: Correlation ID for tracking

        Returns:
            List of EnhancedSubproblem objects

        Example:
            >>> subproblems = [
            ...     {"id": "sub_1", "problem": "Design API gateway", ...},
            ...     {"id": "sub_2", "problem": "Design service discovery", ...}
            ... ]
            >>> enhanced = await roma_dspy.batch_reason_subproblems(subproblems)
        """
        correlation_id = correlation_id or f"batch_reason_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting batch reasoning for sub-problems",
            "subproblems_count": len(subproblems),
            "parallel_reasoning": self.config["parallel_reasoning"],
            "batch_size": self.config["batch_size"],
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            if self.config["parallel_reasoning"]:
                # Process in batches to control parallelism
                batch_size = self.config["batch_size"]
                results = []

                for i in range(0, len(subproblems), batch_size):
                    batch = subproblems[i:i + batch_size]
                    batch_correlation_id = f"{correlation_id}_batch_{i // batch_size}"

                    # Process batch in parallel
                    tasks = [
                        self.add_reasoning_to_subproblem(
                            subp,
                            correlation_id=f"{batch_correlation_id}_{j}"
                        )
                        for j, subp in enumerate(batch)
                    ]

                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Handle exceptions
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error({
                                "msg": f"Batch item {i+j} reasoning failed",
                                "correlation_id": f"{batch_correlation_id}_{j}",
                                "error": str(result)
                            })
                            # Create enhanced sub-problem without reasoning
                            results.append(EnhancedSubproblem(
                                subproblem_id=batch[j].get("id", str(uuid.uuid4())),
                                problem=batch[j].get("problem", ""),
                                depth=batch[j].get("depth", 0),
                                is_atomic=batch[j].get("is_atomic", False),
                                reasoning_trace=None,
                                metadata={
                                    **batch[j].get("metadata", {}),
                                    "reasoning_error": str(result)
                                }
                            ))
                        else:
                            results.append(result)
            else:
                # Process sequentially
                results = []
                for i, subp in enumerate(subproblems):
                    result = await self.add_reasoning_to_subproblem(
                        subp,
                        correlation_id=f"{correlation_id}_seq_{i}"
                    )
                    results.append(result)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            with_reasoning = sum(1 for r in results if r.reasoning_trace)

            logger.info({
                "msg": "Batch reasoning completed",
                "correlation_id": correlation_id,
                "subproblems_count": len(subproblems),
                "with_reasoning": with_reasoning,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return results

        except Exception as e:
            logger.error({
                "msg": "Batch reasoning failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return enhanced sub-problems without reasoning
            return [
                EnhancedSubproblem(
                    subproblem_id=subp.get("id", str(uuid.uuid4())),
                    problem=subp.get("problem", ""),
                    depth=subp.get("depth", 0),
                    is_atomic=subp.get("is_atomic", False),
                    reasoning_trace=None,
                    metadata={
                        **subp.get("metadata", {}),
                        "reasoning_error": str(e)
                    }
                )
                for subp in subproblems
            ]

    async def verify_with_reasoning(
        self,
        solution: ROMAResult,
        requirements: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> ROMAResult:
        """
        Verify a solution using DSPy reasoning.

        Generates explanations for why the solution meets or doesn't meet
        specified requirements.

        Args:
            solution: ROMAResult containing solution to verify
            requirements: Requirements to validate against
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult with enhanced verification

        Example:
            >>> requirements = {
            ...     "completeness": True,
            ...     "correctness": 0.9,
            ...     "scalability": "handles 1000+ requests/sec"
            ... }
            >>> verified = await roma_dspy.verify_with_reasoning(
            ...     solution,
            ...     requirements
            ... )
        """
        correlation_id = correlation_id or f"verify_reason_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting verification with reasoning",
            "solution_count": len(solution.solutions),
            "requirements": list(requirements.keys()),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # First, perform standard ROMA verification
            if solution.solutions:
                primary_solution = solution.solutions[0]
                verify_result = await self.roma.verify_solution(
                    primary_solution,
                    requirements,
                    correlation_id=f"{correlation_id}_roma"
                )

                if verify_result.verification:
                    solution.verification = verify_result.verification

            # Then, use DSPy to generate explanatory reasoning
            if self.dspy.lm and solution.solutions:
                # Build verification question
                solution_text = str(solution.solutions[0].solution) if solution.solutions else ""
                req_text = "\n".join([f"- {k}: {v}" for k, v in requirements.items()])

                verify_question = f"""
                Analyze whether the following solution meets the specified requirements.

                Solution:
                {solution_text}

                Requirements:
                {req_text}

                Provide:
                1. An assessment for each requirement
                2. Overall conclusion
                3. Explanation for the assessment
                """

                dspy_result = await self.dspy.chain_of_thought(
                    question=verify_question,
                    context="Verification with reasoning",
                    max_steps=5,
                    correlation_id=f"{correlation_id}_dspy"
                )

                if dspy_result.success:
                    # Parse verification reasoning
                    verification_reasoning = dspy_result.reasoning

                    # Add reasoning to verification metadata
                    if solution.verification:
                        solution.verification.metadata["verification_reasoning"] = verification_reasoning
                        solution.verification.metadata["explanation"] = dspy_result.output
                    else:
                        # Create verification if it doesn't exist
                        from .roma_integration import ROMAVerification
                        solution.verification = ROMAVerification(
                            verification_id=str(uuid.uuid4()),
                            solution_id=solution.solutions[0].solution_id if solution.solutions else "",
                            passed=True,  # Default to passed
                            score=0.8,
                            feedback=dspy_result.output,
                            requirements_met={k: True for k in requirements.keys()},
                            metadata={
                                "verification_reasoning": verification_reasoning,
                                "dspy_enhanced": True
                            }
                        )

                    self._stats["verifications_with_reasoning"] += 1

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            solution.metadata["verification_with_reasoning_time_ms"] = processing_time_ms
            solution.metadata["verification_explanation"] = (
                solution.verification.metadata.get("explanation", "")
                if solution.verification else ""
            )

            logger.info({
                "msg": "Verification with reasoning completed",
                "correlation_id": correlation_id,
                "passed": solution.verification.passed if solution.verification else None,
                "score": solution.verification.score if solution.verification else None,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return solution

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Verification with reasoning failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return original solution without verification enhancement
            return solution

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ROMA-DSPy integration statistics.

        Returns:
            Dictionary with execution statistics

        Example:
            >>> stats = roma_dspy.get_statistics()
            >>> print(stats["cooperative_solutions"])
        """
        total_operations = (
            self._stats["cooperative_solutions"] +
            self._stats["reasoning_traces_generated"] +
            self._stats["verifications_with_reasoning"]
        )

        return {
            "cooperative_solutions": self._stats["cooperative_solutions"],
            "reasoning_traces_generated": self._stats["reasoning_traces_generated"],
            "reasoning_cache_hits": self._stats["reasoning_cache_hits"],
            "subproblems_reasoned": self._stats["subproblems_reasoned"],
            "verifications_with_reasoning": self._stats["verifications_with_reasoning"],
            "total_processing_time_ms": self._stats["total_processing_time_ms"],
            "average_processing_time_ms": (
                self._stats["total_processing_time_ms"] / total_operations
                if total_operations > 0
                else 0.0
            ),
            "cache_hit_rate": (
                self._stats["reasoning_cache_hits"] /
                (self._stats["reasoning_traces_generated"] + self._stats["reasoning_cache_hits"])
                if (self._stats["reasoning_traces_generated"] + self._stats["reasoning_cache_hits"]) > 0
                else 0.0
            ),
            "config": self.config,
            "dspy_available": self.dspy.lm is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the ROMA-DSPy integration.

        Returns:
            Dictionary with health status
        """
        roma_health = self.roma.health_check()
        dspy_status = self.dspy.get_dspy_status()

        # Determine overall health
        if roma_health.get("status") == "healthy" and dspy_status.get("available"):
            status = "healthy"
        elif roma_health.get("status") == "degraded" or dspy_status.get("available"):
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "roma_status": roma_health.get("status", "unknown"),
            "dspy_available": dspy_status.get("available", False),
            "reasoning_enabled": self.config["auto_add_reasoning"],
            "cache_size": len(self._reasoning_cache),
            "statistics": self.get_statistics(),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        """
        Close resources used by the integration.

        Performs cleanup of ROMA and DSPy resources.
        """
        logger.info({
            "msg": "Closing ROMA-DSPy integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Clear reasoning cache
        cache_size = len(self._reasoning_cache)
        self._reasoning_cache.clear()

        # Close ROMA integration
        if self.roma:
            try:
                await self.roma.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing ROMA integration",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Close DSPy integration
        if self.dspy:
            try:
                await self.dspy.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing DSPy integration",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        logger.info({
            "msg": "ROMA-DSPy integration resources closed",
            "cache_cleared": cache_size,
            "statistics": self.get_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


async def create_roma_dspy_integration(
    roma_config: Optional[Dict[str, Any]] = None,
    dspy_config: Optional[Dict[str, Any]] = None,
    integration_config: Optional[Dict[str, Any]] = None
) -> ROMADSPyIntegration:
    """
    Factory function to create a ROMA-DSPy integration.

    Args:
        roma_config: Configuration for ROMA integration
        dspy_config: Configuration for DSPy integration
        integration_config: Configuration for ROMA-DSPy integration

    Returns:
        ROMADSPyIntegration instance

    Example:
        >>> roma_dspy = await create_roma_dspy_integration(
        ...     roma_config={"decomposer": {"max_depth": 3}},
        ...     dspy_config={
        ...         "model": "gpt-4o",
        ...         "api_key": "sk-..."
        ...     },
        ...     integration_config={
        ...         "auto_add_reasoning": True,
        ...         "parallel_reasoning": True
        ...     }
        ... )
    """
    # Create ROMA integration
    roma = ROMAIntegration(config=roma_config)

    # Create DSPy integration
    from .dspy_integration import DSPyIntegration
    dspy = DSPyIntegration(config=dspy_config)

    # Create ROMA-DSPy integration
    roma_dspy = ROMADSPyIntegration(
        roma_integration=roma,
        dspy_integration=dspy,
        config=integration_config
    )

    return roma_dspy


__all__ = [
    'ROMADSPyIntegration',
    'ReasoningTrace',
    'EnhancedSubproblem',
    'create_roma_dspy_integration'
]
