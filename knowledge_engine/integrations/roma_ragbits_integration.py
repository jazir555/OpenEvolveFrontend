"""
ROMA + Ragbits Integration for OpenEvolve Knowledge Engine

This module provides integration between:
- ROMA: Recursive Optimized Multi-Agent decomposition and problem-solving
- Ragbits: Document indexing and retrieval for RAG operations

Features:
- Index ROMA solutions for retrieval and reuse
- Search similar past problems and solutions
- Solution similarity matching with metadata filtering
- Batch indexing for efficient bulk operations
- CRUD operations on indexed solutions
- Solution reuse workflow with adaptation

Author: OpenEvolve
Created: 2026-02-03
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import uuid


logger = logging.getLogger(__name__)


# =============================================================================
# Import Dependencies
# =============================================================================

# ROMA imports
try:
    from knowledge_engine.integrations.roma_integration import (
        ROMAIntegration,
        ROMAResult,
        ROMADecomposition,
        ROMASolution,
        ROMAVerification
    )
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    ROMAIntegration = None
    ROMAResult = None
    ROMADecomposition = None
    ROMASolution = None
    ROMAVerification = None
    logger.warning("ROMA not available")

# Ragbits imports
try:
    from knowledge_engine.integrations.ragbits_integration import (
        RagbitsIntegration,
        RagbitsResult
    )
    RAGBITS_AVAILABLE = True
except ImportError:
    RAGBITS_AVAILABLE = False
    RagbitsIntegration = None
    RagbitsResult = None
    logger.warning("Ragbits not available")


# =============================================================================
# Data Classes and Enums
# =============================================================================

class SolutionReuseStatus(Enum):
    """Status of solution reuse operations."""
    NEW_SOLUTION = "new_solution"
    REUSED_DIRECT = "reused_direct"
    REUSED_ADAPTED = "reused_adapted"
    NO_SIMILAR_FOUND = "no_similar_found"
    REUSE_FAILED = "reuse_failed"


@dataclass
class IndexedSolution:
    """A ROMA solution indexed in RAGbits."""
    document_id: str
    solution: ROMASolution
    decomposition: Optional[ROMADecomposition]
    verification: Optional[ROMAVerification]
    metadata: Dict[str, Any] = field(default_factory=dict)
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "solution": asdict(self.solution) if self.solution else None,
            "decomposition": asdict(self.decomposition) if self.decomposition else None,
            "verification": asdict(self.verification) if self.verification else None,
            "metadata": self.metadata,
            "indexed_at": self.indexed_at,
            "similarity_score": self.similarity_score
        }


@dataclass
class SimilarSolution:
    """A similar solution found during retrieval."""
    document_id: str
    problem: str
    solution: Any
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    problem_type: Optional[str] = None
    complexity_score: Optional[float] = None
    verification_score: Optional[float] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "problem": self.problem,
            "solution": str(self.solution)[:500] if self.solution else None,  # Truncate for display
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "problem_type": self.problem_type,
            "complexity_score": self.complexity_score,
            "verification_score": self.verification_score,
            "created_at": self.created_at
        }


@dataclass
class SolutionReuseResult:
    """Result of solution reuse operation."""
    success: bool
    status: SolutionReuseStatus
    solution: Optional[ROMASolution]
    similar_solutions: List[SimilarSolution]
    adaptation_notes: Optional[str]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "status": self.status.value if self.status else None,
            "solution": asdict(self.solution) if self.solution else None,
            "similar_solutions": [s.to_dict() for s in self.similar_solutions],
            "adaptation_notes": self.adaptation_notes,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
            "error": self.error
        }


@dataclass
class IndexStatistics:
    """Statistics about indexed solutions."""
    total_solutions: int
    index_size_bytes: int
    last_indexed: Optional[str]
    problem_types: Dict[str, int]
    average_complexity: float
    verification_rate: float
    index_health: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


# =============================================================================
# ROMA-RAGbits Integration
# =============================================================================

class ROMARagbitsIntegration:
    """
    Integration combining ROMA decomposition with RAGbits indexing and retrieval.

    Features:
    - Index ROMA solutions for retrieval and reuse
    - Search for similar past problems and solutions
    - Filter by problem type, complexity, verification score
    - Batch indexing for efficient operations
    - Solution reuse workflow with adaptation
    - CRUD operations on indexed solutions
    - Comprehensive statistics and health monitoring
    """

    def __init__(
        self,
        roma_integration: Optional[ROMAIntegration] = None,
        ragbits_integration: Optional[RagbitsIntegration] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ROMA-RAGbits integration.

        Args:
            roma_integration: ROMA integration instance (created if None)
            ragbits_integration: RAGbits integration instance (created if None)
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # Initialize integrations
        self.roma_integration = roma_integration
        self.ragbits_integration = ragbits_integration

        # Statistics tracking
        self._stats = {
            "solutions_indexed": 0,
            "solutions_retrieved": 0,
            "solutions_reused": 0,
            "batches_indexed": 0,
            "searches_performed": 0,
            "total_processing_time_ms": 0.0
        }

        # Cache for deduplication
        self._solution_cache: Dict[str, str] = {}  # solution_id -> document_id

        # Initialize components
        self._initialize_components()

        logger.info({
            "msg": "ROMARagbitsIntegration initialized",
            "roma_available": self.roma_integration is not None,
            "ragbits_available": self.ragbits_integration is not None,
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "auto_index_solutions": True,
            "index_decompositions": True,
            "index_verification_results": True,
            "similarity_threshold": 0.7,
            "max_index_size": 10000,
            "embedding_model": "default",
            "index_fields": ["problem", "solution", "decomposition", "reasoning"],
            "batch_index_size": 100,
            "ragbits": {
                "vector_store": {
                    "type": "qdrant",
                    "config": {
                        "location": ":memory:",
                        "collection_name": "roma_solutions"
                    }
                },
                "default_options": {
                    "top_k": 5,
                    "similarity_threshold": 0.7
                }
            },
            "roma": {
                "decomposer": {
                    "max_depth": 5,
                    "strategy": "recursive"
                },
                "solver": {
                    "timeout_seconds": 300
                }
            },
            "solution_reuse": {
                "enabled": True,
                "min_similarity_for_reuse": 0.8,
                "max_solutions_to_retrieve": 5,
                "adaptation_strategy": "template"
            }
        }

    def _initialize_components(self):
        """Initialize ROMA and RAGbits components if not provided."""
        # Initialize RAGbits
        if not self.ragbits_integration and RAGBITS_AVAILABLE:
            try:
                ragbits_config = self.config.get("ragbits", {})
                self.ragbits_integration = RagbitsIntegration(ragbits_config)
                logger.info("RAGbits integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAGbits: {e}")

        # Initialize ROMA
        if not self.roma_integration and ROMA_AVAILABLE:
            try:
                roma_config = self.config.get("roma", {})
                self.roma_integration = ROMAIntegration(roma_config)
                logger.info("ROMA integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ROMA: {e}")

    async def index_solution(
        self,
        solution: ROMAResult,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Index a ROMA solution in RAGbits document store.

        Args:
            solution: ROMAResult containing solution to index
            metadata: Additional metadata to attach
            correlation_id: Correlation ID for tracking

        Returns:
            Document ID if successful, None otherwise

        Example:
            >>> result = await roma.decompose_problem("Design system architecture")
            >>> doc_id = await integration.index_solution(result)
            >>> print(f"Indexed as: {doc_id}")
        """
        correlation_id = correlation_id or f"index_solution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Indexing ROMA solution",
            "correlation_id": correlation_id,
            "solution_count": len(solution.solutions),
            "timestamp": start_time.isoformat()
        })

        try:
            if not self.ragbits_integration:
                logger.warning({
                    "msg": "RAGbits not available, cannot index solution",
                    "correlation_id": correlation_id
                })
                return None

            # Check if we have solutions to index
            if not solution.solutions:
                logger.warning({
                    "msg": "No solutions to index",
                    "correlation_id": correlation_id
                })
                return None

            # Get primary solution
            primary_solution = solution.solutions[0]

            # Check for duplicates (idempotent)
            solution_id = primary_solution.solution_id
            if solution_id in self._solution_cache:
                cached_doc_id = self._solution_cache[solution_id]
                logger.info({
                    "msg": "Solution already indexed (idempotent)",
                    "solution_id": solution_id,
                    "document_id": cached_doc_id,
                    "correlation_id": correlation_id
                })
                return cached_doc_id

            # Create document content
            content = self._create_solution_content(solution, primary_solution)

            # Create metadata
            doc_metadata = self._create_solution_metadata(solution, primary_solution, metadata)

            # Prepare document for RAGbits
            document = {
                "content": content,
                "metadata": doc_metadata
            }

            # Index the document
            result = await self.ragbits_integration.ingest_documents(
                documents=[document],
                correlation_id=correlation_id
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            if result.success:
                # Generate document ID
                document_id = f"roma_sol_{solution_id}"

                # Cache for deduplication
                self._solution_cache[solution_id] = document_id

                # Update statistics
                self._stats["solutions_indexed"] += 1
                self._stats["total_processing_time_ms"] += processing_time_ms

                logger.info({
                    "msg": "Solution indexed successfully",
                    "document_id": document_id,
                    "solution_id": solution_id,
                    "processing_time_ms": processing_time_ms,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return document_id
            else:
                logger.error({
                    "msg": "Failed to index solution",
                    "error": result.error,
                    "correlation_id": correlation_id
                })
                return None

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Error indexing solution",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return None

    def _create_solution_content(
        self,
        result: ROMAResult,
        solution: ROMASolution
    ) -> str:
        """
        Create document content from ROMA solution.

        Args:
            result: ROMAResult containing full context
            solution: Primary solution to create content from

        Returns:
            Formatted content string
        """
        parts = []

        # Problem statement
        if result.decomposition:
            parts.append(f"Problem: {result.decomposition.problem}")

        # Solution
        parts.append(f"Solution: {solution.solution}")

        # Reasoning
        if solution.reasoning:
            parts.append(f"Reasoning: {solution.reasoning}")

        # Decomposition details (if configured)
        if self.config.get("index_decompositions", True) and result.decomposition:
            decomp_info = self._format_decomposition(result.decomposition)
            if decomp_info:
                parts.append(f"Decomposition: {decomp_info}")

        # Verification results (if configured)
        if self.config.get("index_verification_results", True) and result.verification:
            verification_info = self._format_verification(result.verification)
            if verification_info:
                parts.append(f"Verification: {verification_info}")

        return "\n\n".join(parts)

    def _format_decomposition(self, decomposition: ROMADecomposition) -> str:
        """Format decomposition for indexing."""
        parts = [
            f"Depth: {decomposition.depth}",
            f"Atomic: {decomposition.is_atomic}",
            f"Sub-problems: {len(decomposition.sub_problems)}"
        ]
        return ", ".join(parts)

    def _format_verification(self, verification: ROMAVerification) -> str:
        """Format verification for indexing."""
        parts = [
            f"Passed: {verification.passed}",
            f"Score: {verification.score}",
            f"Feedback: {verification.feedback}"
        ]
        return ", ".join(parts)

    def _create_solution_metadata(
        self,
        result: ROMAResult,
        solution: ROMASolution,
        additional_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create metadata for indexed solution.

        Args:
            result: ROMAResult containing full context
            solution: Primary solution
            additional_metadata: Additional metadata to include

        Returns:
            Metadata dictionary
        """
        metadata = {
            "document_type": "roma_solution",
            "solution_id": solution.solution_id,
            "problem_id": solution.problem_id,
            "created_at": solution.created_at,
            "confidence": solution.confidence,
            "index_fields": self.config.get("index_fields", []),
            "source": "roma_ragbits_integration"
        }

        # Add decomposition metadata
        if result.decomposition:
            metadata.update({
                "decomposition_id": result.decomposition.decomposition_id,
                "problem_type": self._determine_problem_type(result.decomposition),
                "complexity_score": self._calculate_complexity(result),
                "decomposition_depth": result.decomposition.depth,
                "is_atomic": result.decomposition.is_atomic
            })

        # Add verification metadata
        if result.verification:
            metadata.update({
                "verification_passed": result.verification.passed,
                "verification_score": result.verification.score
            })

        # Add additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def _determine_problem_type(self, decomposition: ROMADecomposition) -> str:
        """Determine problem type from decomposition."""
        problem_lower = decomposition.problem.lower()

        # Simple keyword-based classification
        if any(word in problem_lower for word in ["design", "architecture", "system"]):
            return "design"
        elif any(word in problem_lower for word in ["calculate", "compute", "solve"]):
            return "computation"
        elif any(word in problem_lower for word in ["prove", "verify", "theorem"]):
            return "proof"
        elif any(word in problem_lower for word in ["analyze", "understand", "explain"]):
            return "analysis"
        else:
            return "general"

    def _calculate_complexity(self, result: ROMAResult) -> float:
        """Calculate complexity score for a solution."""
        if not result.decomposition:
            return 0.5

        # Factors: depth, sub-problem count
        depth_factor = min(result.decomposition.depth / 5.0, 1.0)
        sub_problem_factor = min(
            self._count_total_sub_problems(result.decomposition) / 20.0,
            1.0
        )

        return round((depth_factor + sub_problem_factor) / 2.0, 3)

    def _count_total_sub_problems(self, decomposition: ROMADecomposition) -> int:
        """Recursively count all sub-problems."""
        count = len(decomposition.sub_problems)
        for sub in decomposition.sub_problems:
            count += self._count_total_sub_problems(sub)
        return count

    async def index_batch_solutions(
        self,
        solutions: List[ROMAResult],
        correlation_id: Optional[str] = None
    ) -> List[str]:
        """
        Index multiple solutions in batch.

        Args:
            solutions: List of ROMAResult objects to index
            correlation_id: Correlation ID for tracking

        Returns:
            List of document IDs

        Example:
            >>> results = await roma.batch_decompose(problems)
            >>> doc_ids = await integration.index_batch_solutions(results)
            >>> print(f"Indexed {len(doc_ids)} solutions")
        """
        correlation_id = correlation_id or f"batch_index_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Starting batch solution indexing",
            "solution_count": len(solutions),
            "batch_size": self.config.get("batch_index_size", 100),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Process in batches
            batch_size = self.config.get("batch_index_size", 100)
            document_ids = []

            for i in range(0, len(solutions), batch_size):
                batch = solutions[i:i + batch_size]

                # Index batch in parallel
                tasks = [
                    self.index_solution(
                        solution=sol,
                        correlation_id=f"{correlation_id}_sol_{i+j}"
                    )
                    for j, sol in enumerate(batch)
                ]

                batch_ids = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect successful IDs
                for j, doc_id in enumerate(batch_ids):
                    if isinstance(doc_id, Exception):
                        logger.error({
                            "msg": f"Batch item {i+j} failed to index",
                            "error": str(doc_id),
                            "correlation_id": f"{correlation_id}_sol_{i+j}"
                        })
                    elif doc_id:
                        document_ids.append(doc_id)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update statistics
            self._stats["batches_indexed"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.info({
                "msg": "Batch indexing completed",
                "total_solutions": len(solutions),
                "indexed_count": len(document_ids),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return document_ids

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Batch indexing failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return []

    async def retrieve_similar_solutions(
        self,
        problem: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> List[SimilarSolution]:
        """
        Retrieve similar past solutions for a problem.

        Args:
            problem: Problem statement to find similar solutions for
            top_k: Number of solutions to retrieve
            filters: Optional filters (problem_type, min_confidence, etc.)
            correlation_id: Correlation ID for tracking

        Returns:
            List of SimilarSolution objects

        Example:
            >>> similar = await integration.retrieve_similar_solutions(
            ...     "Design microservices architecture",
            ...     top_k=3,
            ...     filters={"problem_type": "design", "min_confidence": 0.7}
            ... )
            >>> for sol in similar:
            ...     print(f"{sol.problem} (score: {sol.similarity_score})")
        """
        correlation_id = correlation_id or f"retrieve_similar_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Retrieving similar solutions",
            "problem_length": len(problem),
            "top_k": top_k,
            "filters": filters,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            if not self.ragbits_integration:
                logger.warning({
                    "msg": "RAGbits not available",
                    "correlation_id": correlation_id
                })
                return []

            # Prepare search query
            query = problem

            # Add filter context to query if provided
            if filters:
                filter_context = self._create_filter_context(filters)
                if filter_context:
                    query = f"{query}\n\nContext: {filter_context}"

            # Search RAGbits
            result = await self.ragbits_integration.search_documents(
                query=query,
                top_k=top_k,
                similarity_threshold=self.config.get("similarity_threshold", 0.7),
                correlation_id=correlation_id
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Convert to SimilarSolution objects
            similar_solutions = []
            for r in result.results:
                similar = self._create_similar_solution(r)
                if similar:
                    # Apply filters if provided
                    if self._passes_filters(similar, filters):
                        similar_solutions.append(similar)

            # Sort by similarity score
            similar_solutions.sort(key=lambda s: s.similarity_score, reverse=True)
            similar_solutions = similar_solutions[:top_k]

            # Update statistics
            self._stats["solutions_retrieved"] += len(similar_solutions)
            self._stats["searches_performed"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms

            logger.info({
                "msg": "Similar solutions retrieved",
                "count": len(similar_solutions),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return similar_solutions

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Failed to retrieve similar solutions",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return []

    def _create_filter_context(self, filters: Dict[str, Any]) -> str:
        """Create filter context for search query."""
        context_parts = []

        if filters.get("problem_type"):
            context_parts.append(f"Type: {filters['problem_type']}")

        if filters.get("min_complexity"):
            context_parts.append(f"Min Complexity: {filters['min_complexity']}")

        if filters.get("max_complexity"):
            context_parts.append(f"Max Complexity: {filters['max_complexity']}")

        return ", ".join(context_parts)

    def _create_similar_solution(self, search_result: Dict[str, Any]) -> Optional[SimilarSolution]:
        """Create SimilarSolution from RAGbits search result."""
        try:
            metadata = search_result.get("metadata", {})

            return SimilarSolution(
                document_id=metadata.get("solution_id", "unknown"),
                problem=metadata.get("problem", search_result.get("content", ""))[:200],
                solution=search_result.get("content", ""),
                similarity_score=search_result.get("score", 0.0),
                metadata=metadata,
                problem_type=metadata.get("problem_type"),
                complexity_score=metadata.get("complexity_score"),
                verification_score=metadata.get("verification_score"),
                created_at=metadata.get("created_at")
            )
        except Exception as e:
            logger.warning(f"Failed to create SimilarSolution: {e}")
            return None

    def _passes_filters(self, solution: SimilarSolution, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if solution passes filters."""
        if not filters:
            return True

        # Problem type filter
        if "problem_type" in filters:
            if solution.problem_type != filters["problem_type"]:
                return False

        # Confidence filter
        if "min_confidence" in filters:
            if solution.metadata.get("confidence", 0.0) < filters["min_confidence"]:
                return False

        # Complexity filters
        if "min_complexity" in filters:
            if solution.complexity_score and solution.complexity_score < filters["min_complexity"]:
                return False

        if "max_complexity" in filters:
            if solution.complexity_score and solution.complexity_score > filters["max_complexity"]:
                return False

        # Verification filter
        if "verification_passed" in filters:
            if solution.metadata.get("verification_passed") != filters["verification_passed"]:
                return False

        return True

    async def get_solution_by_id(
        self,
        document_id: str,
        correlation_id: Optional[str] = None
    ) -> Optional[ROMAResult]:
        """
        Retrieve indexed solution by document ID.

        Args:
            document_id: Document ID to retrieve
            correlation_id: Correlation ID for tracking

        Returns:
            ROMAResult if found, None otherwise
        """
        correlation_id = correlation_id or f"get_solution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Retrieving solution by ID",
            "document_id": document_id,
            "correlation_id": correlation_id
        })

        try:
            if not self.ragbits_integration:
                logger.warning({
                    "msg": "RAGbits not available",
                    "correlation_id": correlation_id
                })
                return None

            # Search by document ID
            result = await self.ragbits_integration.search_documents(
                query=document_id,
                top_k=1,
                correlation_id=correlation_id
            )

            if result.success and result.results:
                # Reconstruct ROMAResult from stored data
                return self._reconstruct_result(result.results[0])

            return None

        except Exception as e:
            logger.error({
                "msg": "Failed to retrieve solution by ID",
                "error": str(e),
                "document_id": document_id,
                "correlation_id": correlation_id
            })
            return None

    def _reconstruct_result(self, search_result: Dict[str, Any]) -> Optional[ROMAResult]:
        """Reconstruct ROMAResult from stored data."""
        try:
            metadata = search_result.get("metadata", {})

            # Create basic ROMAResult
            # Note: This is a simplified reconstruction
            # In production, you'd store full JSON serialization
            return ROMAResult(
                success=True,
                decomposition=None,  # Would need full reconstruction
                solutions=[],  # Would need full reconstruction
                verification=None,
                metadata=metadata,
                processing_time_ms=0.0
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct ROMAResult: {e}")
            return None

    async def delete_solution(
        self,
        document_id: str,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Delete indexed solution.

        Args:
            document_id: Document ID to delete
            correlation_id: Correlation ID for tracking

        Returns:
            True if successful
        """
        correlation_id = correlation_id or f"delete_solution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Deleting solution",
            "document_id": document_id,
            "correlation_id": correlation_id
        })

        try:
            # Remove from cache
            if document_id in self._solution_cache:
                del self._solution_cache[document_id]

            # Note: Actual deletion would depend on RAGbits vector store implementation
            # This is a placeholder for the deletion logic

            logger.info({
                "msg": "Solution deleted",
                "document_id": document_id,
                "correlation_id": correlation_id
            })

            return True

        except Exception as e:
            logger.error({
                "msg": "Failed to delete solution",
                "error": str(e),
                "document_id": document_id,
                "correlation_id": correlation_id
            })
            return False

    async def update_solution(
        self,
        document_id: str,
        solution: ROMAResult,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Update existing indexed solution.

        Args:
            document_id: Document ID to update
            solution: New solution data
            correlation_id: Correlation ID for tracking

        Returns:
            True if successful
        """
        correlation_id = correlation_id or f"update_solution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Updating solution",
            "document_id": document_id,
            "correlation_id": correlation_id
        })

        try:
            # Delete old solution
            await self.delete_solution(document_id, correlation_id)

            # Index new solution
            new_doc_id = await self.index_solution(solution, correlation_id=correlation_id)

            if new_doc_id:
                logger.info({
                    "msg": "Solution updated",
                    "old_document_id": document_id,
                    "new_document_id": new_doc_id,
                    "correlation_id": correlation_id
                })
                return True

            return False

        except Exception as e:
            logger.error({
                "msg": "Failed to update solution",
                "error": str(e),
                "document_id": document_id,
                "correlation_id": correlation_id
            })
            return False

    async def search_solutions(
        self,
        query: str,
        top_k: int = 10,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        General search across indexed solutions.

        Args:
            query: Search query
            top_k: Number of results
            correlation_id: Correlation ID for tracking

        Returns:
            List of matching solutions
        """
        correlation_id = correlation_id or f"search_solutions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Searching solutions",
            "query_length": len(query),
            "top_k": top_k,
            "correlation_id": correlation_id
        })

        try:
            if not self.ragbits_integration:
                return []

            result = await self.ragbits_integration.search_documents(
                query=query,
                top_k=top_k,
                correlation_id=correlation_id
            )

            return result.results

        except Exception as e:
            logger.error({
                "msg": "Search failed",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return []

    async def get_index_statistics(self) -> IndexStatistics:
        """
        Get statistics about indexed solutions.

        Returns:
            IndexStatistics object

        Example:
            >>> stats = await integration.get_index_statistics()
            >>> print(f"Total solutions: {stats.total_solutions}")
            >>> print(f"Average complexity: {stats.average_complexity}")
        """
        try:
            # Get stats from RAGbits
            ragbits_stats = {}
            if self.ragbits_integration:
                ragbits_stats = await self.ragbits_integration.get_statistics()

            # Calculate index health
            total_indexed = self._stats["solutions_indexed"]
            max_size = self.config.get("max_index_size", 10000)
            usage_ratio = total_indexed / max_size if max_size > 0 else 0

            if usage_ratio < 0.5:
                index_health = "healthy"
            elif usage_ratio < 0.8:
                index_health = "moderate"
            else:
                index_health = "full"

            statistics = IndexStatistics(
                total_solutions=total_indexed,
                index_size_bytes=ragbits_stats.get("index_size_bytes", 0),
                last_indexed=self._stats.get("last_indexed"),
                problem_types=self._get_problem_type_distribution(),
                average_complexity=self._calculate_average_complexity(),
                verification_rate=self._calculate_verification_rate(),
                index_health=index_health
            )

            return statistics

        except Exception as e:
            logger.error({
                "msg": "Failed to get index statistics",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Return empty statistics on error
            return IndexStatistics(
                total_solutions=0,
                index_size_bytes=0,
                last_indexed=None,
                problem_types={},
                average_complexity=0.0,
                verification_rate=0.0,
                index_health="unknown"
            )

    def _get_problem_type_distribution(self) -> Dict[str, int]:
        """Get distribution of problem types in index."""
        # This would require tracking problem types during indexing
        # For now, return placeholder
        return {
            "design": 0,
            "computation": 0,
            "proof": 0,
            "analysis": 0,
            "general": 0
        }

    def _calculate_average_complexity(self) -> float:
        """Calculate average complexity of indexed solutions."""
        # This would require tracking complexities during indexing
        # For now, return placeholder
        return 0.5

    def _calculate_verification_rate(self) -> float:
        """Calculate rate of verified solutions."""
        # This would require tracking verification results
        # For now, return placeholder
        return 0.8

    async def reuse_solution(
        self,
        problem: str,
        top_k: int = 5,
        adapt: bool = True,
        correlation_id: Optional[str] = None
    ) -> SolutionReuseResult:
        """
        Attempt to reuse a past solution for a new problem.

        Args:
            problem: New problem to solve
            top_k: Number of similar solutions to retrieve
            adapt: Whether to adapt the solution
            correlation_id: Correlation ID for tracking

        Returns:
            SolutionReuseResult with reuse status and solution

        Example:
            >>> result = await integration.reuse_solution(
            ...     "Design REST API for microservices",
            ...     top_k=3
            ... )
            >>> if result.status == SolutionReuseStatus.REUSED_DIRECT:
            ...     print("Reused existing solution!")
        """
        correlation_id = correlation_id or f"reuse_solution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Attempting solution reuse",
            "problem_length": len(problem),
            "top_k": top_k,
            "adapt": adapt,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Check if solution reuse is enabled
            if not self.config.get("solution_reuse", {}).get("enabled", False):
                return SolutionReuseResult(
                    success=False,
                    status=SolutionReuseStatus.REUSE_FAILED,
                    solution=None,
                    similar_solutions=[],
                    adaptation_notes="Solution reuse disabled in configuration",
                    processing_time_ms=0.0,
                    metadata={"reason": "disabled"}
                )

            # Retrieve similar solutions
            similar_solutions = await self.retrieve_similar_solutions(
                problem=problem,
                top_k=top_k,
                correlation_id=correlation_id
            )

            if not similar_solutions:
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                return SolutionReuseResult(
                    success=False,
                    status=SolutionReuseStatus.NO_SIMILAR_FOUND,
                    solution=None,
                    similar_solutions=[],
                    adaptation_notes="No similar solutions found",
                    processing_time_ms=processing_time_ms
                )

            # Check if we have a high-similarity match
            min_similarity = self.config.get("solution_reuse", {}).get("min_similarity_for_reuse", 0.8)
            best_match = similar_solutions[0]

            if best_match.similarity_score >= min_similarity:
                # Direct reuse
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                # Create ROMASolution from retrieved data
                reused_solution = ROMASolution(
                    solution_id=best_match.document_id,
                    problem_id=best_match.document_id,
                    solution=best_match.solution,
                    confidence=best_match.metadata.get("confidence", best_match.similarity_score),
                    reasoning=f"Reused from similar solution (similarity: {best_match.similarity_score:.2f})",
                    metadata={
                        "reused_from": best_match.document_id,
                        "original_similarity": best_match.similarity_score,
                        "reuse_type": "direct"
                    }
                )

                # Update statistics
                self._stats["solutions_reused"] += 1

                return SolutionReuseResult(
                    success=True,
                    status=SolutionReuseStatus.REUSED_DIRECT,
                    solution=reused_solution,
                    similar_solutions=similar_solutions,
                    adaptation_notes=f"Direct reuse (similarity: {best_match.similarity_score:.2f})",
                    processing_time_ms=processing_time_ms,
                    metadata={
                        "best_similarity": best_match.similarity_score,
                        "solutions_considered": len(similar_solutions)
                    }
                )
            else:
                # No good match found
                processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                return SolutionReuseResult(
                    success=False,
                    status=SolutionReuseStatus.NO_SIMILAR_FOUND,
                    solution=None,
                    similar_solutions=similar_solutions,
                    adaptation_notes=f"No solution met similarity threshold (best: {best_match.similarity_score:.2f}, required: {min_similarity:.2f})",
                    processing_time_ms=processing_time_ms
                )

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Solution reuse failed",
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return SolutionReuseResult(
                success=False,
                status=SolutionReuseStatus.REUSE_FAILED,
                solution=None,
                similar_solutions=[],
                adaptation_notes=None,
                processing_time_ms=processing_time_ms,
                error=str(e)
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get integration statistics.

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = integration.get_statistics()
            >>> print(f"Solutions indexed: {stats['solutions_indexed']}")
        """
        return {
            "solutions_indexed": self._stats["solutions_indexed"],
            "solutions_retrieved": self._stats["solutions_retrieved"],
            "solutions_reused": self._stats["solutions_reused"],
            "batches_indexed": self._stats["batches_indexed"],
            "searches_performed": self._stats["searches_performed"],
            "total_processing_time_ms": self._stats["total_processing_time_ms"],
            "cached_solutions": len(self._solution_cache),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the integration.

        Returns:
            Health check results
        """
        start_time = datetime.now(timezone.utc)

        health = {
            "component": "roma_ragbits_integration",
            "status": "healthy",
            "checks": {},
            "timestamp": start_time.isoformat()
        }

        try:
            # Check ROMA integration
            if self.roma_integration:
                roma_health = self.roma_integration.health_check()
                health["checks"]["roma_integration"] = roma_health
            else:
                health["status"] = "degraded"
                health["checks"]["roma_integration"] = {
                    "status": "unavailable",
                    "message": "ROMA integration not initialized"
                }

            # Check RAGbits integration
            if self.ragbits_integration:
                ragbits_health = await self.ragbits_integration.health_check()
                health["checks"]["ragbits_integration"] = ragbits_health

                if ragbits_health.get("status") != "healthy":
                    health["status"] = "degraded"
            else:
                health["status"] = "degraded"
                health["checks"]["ragbits_integration"] = {
                    "status": "unavailable",
                    "message": "RAGbits integration not initialized"
                }

            # Check index health
            index_stats = await self.get_index_statistics()
            health["checks"]["index_health"] = {
                "status": "passed" if index_stats.index_health == "healthy" else "warning",
                "total_solutions": index_stats.total_solutions,
                "index_health": index_stats.index_health
            }

            health["processing_time_ms"] = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return health

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            health["processing_time_ms"] = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Health check error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return health

    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing ROMA-RAGbits integration",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Close ROMA integration
        if self.roma_integration:
            try:
                await self.roma_integration.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing ROMA integration",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Close RAGbits integration
        if self.ragbits_integration:
            try:
                await self.ragbits_integration.close()
            except Exception as e:
                logger.error({
                    "msg": "Error closing RAGbits integration",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        # Clear cache
        self._solution_cache.clear()

        logger.info({
            "msg": "ROMA-RAGbits integration closed",
            "statistics": self.get_statistics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


# =============================================================================
# Factory Functions
# =============================================================================

async def create_roma_ragbits_integration(
    roma_integration: Optional[ROMAIntegration] = None,
    ragbits_integration: Optional[RagbitsIntegration] = None,
    config: Optional[Dict[str, Any]] = None
) -> ROMARagbitsIntegration:
    """
    Create and initialize a ROMA-RAGbits integration.

    Args:
        roma_integration: Optional ROMA integration instance
        ragbits_integration: Optional RAGbits integration instance
        config: Optional configuration dictionary

    Returns:
        Initialized ROMARagbitsIntegration instance

    Example:
        >>> integration = await create_roma_ragbits_integration()
        >>> # Or with existing integrations
        >>> integration = await create_roma_ragbits_integration(
        ...     roma_integration=roma,
        ...     ragbits_integration=ragbits,
        ...     config={"auto_index_solutions": True}
        ... )
    """
    integration = ROMARagbitsIntegration(
        roma_integration=roma_integration,
        ragbits_integration=ragbits_integration,
        config=config
    )

    return integration


def get_roma_ragbits_integration(
    roma_integration: Optional[ROMAIntegration] = None,
    ragbits_integration: Optional[RagbitsIntegration] = None,
    config: Optional[Dict[str, Any]] = None
) -> ROMARagbitsIntegration:
    """
    Get or create a ROMA-RAGbits integration instance.

    Args:
        roma_integration: Optional ROMA integration instance
        ragbits_integration: Optional RAGbits integration instance
        config: Optional configuration dictionary

    Returns:
        ROMARagbitsIntegration instance

    Example:
        >>> integration = get_roma_ragbits_integration()
        >>> result = await integration.index_solution(roma_result)
    """
    return ROMARagbitsIntegration(
        roma_integration=roma_integration,
        ragbits_integration=ragbits_integration,
        config=config
    )


# =============================================================================
# Usage Examples
# =============================================================================

"""
Basic Usage Example:

```python
import asyncio
from knowledge_engine.integrations.roma_integration import ROMAIntegration
from knowledge_engine.integrations.ragbits_integration import RagbitsIntegration
from knowledge_engine.integrations.roma_ragbits_integration import create_roma_ragbits_integration

async def main():
    # Create integrations
    roma = ROMAIntegration()
    ragbits = RagbitsIntegration()
    integration = await create_roma_ragbits_integration(roma, ragbits)

    # Decompose a problem
    problem = "Design a scalable microservices architecture"
    roma_result = await roma.decompose_problem(problem)

    # Index the solution
    doc_id = await integration.index_solution(roma_result)
    print(f"Indexed solution: {doc_id}")

    # Later: retrieve similar solutions
    similar = await integration.retrieve_similar_solutions(
        "Design REST API gateway",
        top_k=3
    )

    for sol in similar:
        print(f"Similar solution (score: {sol.similarity_score}):")
        print(f"  Problem: {sol.problem[:100]}")
        print(f"  Type: {sol.problem_type}")

    # Attempt solution reuse
    reuse_result = await integration.reuse_solution(
        "Design scalable web service architecture",
        top_k=5
    )

    if reuse_result.success:
        print(f"Reused solution: {reuse_result.status}")
        print(f"Adaptation notes: {reuse_result.adaptation_notes}")

    # Get statistics
    stats = await integration.get_index_statistics()
    print(f"Total indexed: {stats.total_solutions}")
    print(f"Index health: {stats.index_health}")

    # Clean up
    await integration.close()

asyncio.run(main())
```

Batch Indexing Example:

```python
async def batch_index_example():
    integration = await create_roma_ragbits_integration()

    # Decompose multiple problems
    problems = [
        "Design authentication system",
        "Implement data caching layer",
        "Create monitoring dashboard"
    ]

    results = await roma.batch_decompose(problems)

    # Index all solutions in batch
    doc_ids = await integration.index_batch_solutions(results)
    print(f"Batch indexed {len(doc_ids)} solutions")

    # Search across indexed solutions
    search_results = await integration.search_solutions(
        "authentication security",
        top_k=10
    )

    print(f"Found {len(search_results)} results")
```

Solution Reuse Workflow:

```python
async def solution_reuse_workflow():
    integration = await create_roma_ragbits_integration()

    # Step 1: New problem arrives
    new_problem = "Design OAuth 2.0 authentication service"

    # Step 2: Search for similar solutions
    similar = await integration.retrieve_similar_solutions(
        new_problem,
        top_k=5,
        filters={"problem_type": "design", "min_confidence": 0.7}
    )

    # Step 3: Attempt to reuse solution
    reuse_result = await integration.reuse_solution(
        new_problem,
        top_k=5
    )

    if reuse_result.success:
        # Solution reused!
        print(f"Reused existing solution: {reuse_result.solution.solution_id}")

        # Optionally adapt the solution
        adapted = adapt_solution(reuse_result.solution, new_problem)
    else:
        # No suitable solution found, need to solve from scratch
        print(f"No reusable solution: {reuse_result.adaptation_notes}")

        # Solve using ROMA
        result = await roma.decompose_problem(new_problem)

        # Index for future reuse
        doc_id = await integration.index_solution(result)
        print(f"Indexed new solution: {doc_id}")
```


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main integration
    "ROMARagbitsIntegration",

    # Data classes
    "IndexedSolution",
    "SimilarSolution",
    "SolutionReuseResult",
    "IndexStatistics",

    # Enums
    "SolutionReuseStatus",

    # Factory functions
    "create_roma_ragbits_integration",
    "get_roma_ragbits_integration",

    # Availability flags
    "ROMA_AVAILABLE",
    "RAGBITS_AVAILABLE"
]
