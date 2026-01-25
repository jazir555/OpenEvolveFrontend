"""
Problem Recomposition System

This module focuses on RECOMPOSING solved sub-problems back into integrated solutions.
It handles the assembly process: taking individual sub-solutions and combining them
while detecting and resolving conflicts.

Core Functionality:
- Conflict detection between sub-solutions
- Conflict resolution strategies
- Solution assembly with multiple strategies
- Quality metrics calculation

This module is responsible for the RECOMPOSITION process only.
Final solution validation and delivery are handled by final_solution.py.

PRODUCTION-GRADE FEATURES:
- Multiple assembly strategies (hierarchical, linear, parallel, adaptive)
- Conflict detection (contradiction, overlap, dependency, inconsistency)
- Conflict resolution (priority, merge, LLM-mediated, manual)
- Solution quality metrics
- End-to-end workflow integration
"""

import logging
import json
import re
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import numpy with fallback (for type hints and embedding generation)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

# Use TYPE_CHECKING to avoid numpy reference at runtime for type hints
if TYPE_CHECKING:
    if np is not None:
        NDArray = np.ndarray
    else:
        NDArray = Any
else:
    NDArray = Any

# MIGRATION: Import from sovereign_data_models with fallbacks
try:
    from sovereign_data_models import (
        ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
        ValidationResult, generate_id
    )
except ImportError as e:
    logging.warning(f"Failed to import from sovereign_data_models: {e}")
    ProblemDefinition = SubProblem = DecompositionPlan = SolutionAttempt = None
    ValidationResult = None
    generate_id = lambda prefix="": f"{prefix}_{str(uuid.uuid4())[:8]}" if prefix else str(uuid.uuid4())[:8]

# Create stubs for classes that don't exist in sovereign_data_models
@dataclass
class ComplexityScore:
    """Complexity score for problems."""
    overall_complexity: float
    technical_complexity: float
    domain_complexity: float

@dataclass
class SuccessCriterion:
    """Success criterion for solutions."""
    id: str
    description: str
    metric: str
    threshold: float

@dataclass
class IntegratedSolution:
    """Integrated solution from recomposed sub-solutions."""
    solution_id: str
    decomposition_plan_id: str
    assembled_content: str
    assembly_strategy: str
    sub_solutions: List[str]
    integration_order: List[str]
    conflicts_detected: List[Any]
    conflicts_resolved: List[Any]
    quality_metrics: Any
    validation_results: Any
    metadata: Dict[str, Any]

@dataclass
class Conflict:
    """Conflict between sub-solutions."""
    conflict_id: str
    conflict_type: str
    severity: str
    involved_sub_solutions: List[str]
    description: str
    metadata: Dict[str, Any]

@dataclass
class SolutionQualityMetrics:
    """Quality metrics for solutions."""
    completeness: float
    consistency: float
    correctness: float
    overall_score: float

logger = logging.getLogger(__name__)

# Optional ROMA integration for recomposition
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_recomposition_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_recomposition_config = None

try:
    from roma_mcp_tools import solve_with_roma
    ROMA_RECOMPOSITION_AVAILABLE = True
except ImportError:
    solve_with_roma = None
    ROMA_RECOMPOSITION_AVAILABLE = False

# Optional Hephaestus integration for recomposition tracking
try:
    import requests
    HEPHAESTUS_RECOMPOSITION_AVAILABLE = True
except ImportError:
    requests = None
    HEPHAESTUS_RECOMPOSITION_AVAILABLE = False

# Import OpenEvolveClient for LLM-mediated resolution
try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
except ImportError:
    logger.warning("OpenEvolveClient not found. LLM-mediated conflict resolution will be disabled.")
    OpenEvolveClient = None
    OPENEVOLVE_AVAILABLE = False


# ============================================================================
# ENHANCED CONFLICT DETECTOR
# ============================================================================

class ConflictDetector:
    """
    Advanced conflict detector with ML-based and algorithmic detection capabilities.

    Features:
    - Semantic similarity detection (embedding-based with fallback to Jaccard)
    - Advanced contradiction detection (logical, numerical, temporal)
    - Cross-domain conflict detection
    - Conflict severity scoring (0.0-1.0)
    - Conflict clustering and pattern detection
    - Performance optimizations (caching, parallel processing)

    Configuration:
    - semantic_threshold: Cosine similarity threshold for semantic conflicts (default: 0.75)
    - overlap_threshold: Jaccard similarity threshold for overlap (default: 0.7)
    - enable_advanced_contradictions: Enable ML-based contradiction detection (default: True)
    - enable_cross_domain: Enable cross-domain conflict detection (default: True)
    - enable_clustering: Enable conflict clustering (default: True)
    - use_embeddings: Use sentence transformers for semantic similarity (default: True)
    - cache_embeddings: Cache embeddings for performance (default: True)
    - parallel_detection: Enable parallel conflict detection (default: True)
    """

    def __init__(
        self,
        openevolve_client: Optional['OpenEvolveClient'] = None,
        semantic_threshold: float = 0.75,
        overlap_threshold: float = 0.7,
        enable_advanced_contradictions: bool = True,
        enable_cross_domain: bool = True,
        enable_clustering: bool = True,
        use_embeddings: bool = True,
        cache_embeddings: bool = True,
        parallel_detection: bool = True
    ):
        """
        Initialize enhanced ConflictDetector.

        Args:
            openevolve_client: Optional OpenEvolve client for LLM-based detection
            semantic_threshold: Cosine similarity threshold (0.0-1.0)
            overlap_threshold: Jaccard similarity threshold (0.0-1.0)
            enable_advanced_contradictions: Enable advanced contradiction detection
            enable_cross_domain: Enable cross-domain conflict detection
            enable_clustering: Enable conflict clustering
            use_embeddings: Use sentence transformers if available
            cache_embeddings: Cache embeddings for performance
            parallel_detection: Enable parallel processing
        """
        self.openevolve_client = openevolve_client
        self.semantic_threshold = semantic_threshold
        self.overlap_threshold = overlap_threshold
        self.enable_advanced_contradictions = enable_advanced_contradictions
        self.enable_cross_domain = enable_cross_domain
        self.enable_clustering = enable_clustering
        self.use_embeddings = use_embeddings
        self.cache_embeddings = cache_embeddings
        self.parallel_detection = parallel_detection

        # Initialize embedding model if requested
        self.embedding_model = None
        self._embedding_cache = {} if cache_embeddings else None

        self._init_client()
        self._init_embeddings()

        # Domain keyword mappings for cross-domain detection
        self.domain_keywords = {
            'technical': ['api', 'database', 'server', 'architecture', 'implementation',
                         'code', 'algorithm', 'protocol', 'framework', 'integration'],
            'business': ['revenue', 'cost', 'roi', 'customer', 'market', 'strategy',
                        'budget', 'profit', 'stakeholder', 'requirement'],
            'security': ['authentication', 'authorization', 'encryption', 'vulnerability',
                        'compliance', 'privacy', 'audit', 'security', 'protection'],
            'performance': ['latency', 'throughput', 'scalability', 'optimization',
                          'caching', 'load balancing', 'response time', 'efficiency'],
            'ux': ['user', 'interface', 'experience', 'accessibility', 'usability',
                  'design', 'interaction', 'workflow', 'navigation']
        }

        # Contradiction patterns for advanced detection
        self.contradiction_patterns = {
            'logical': [
                (r'\b(?:should|must|will)\s+(?:not|never)\b', r'\b(?:should|must|will)\b'),
                (r'\b(?:all|every)\b', r'\b(?:none|no|not)\s+(?:any|one)\b'),
                (r'\b(?:always|forever)\b', r'\b(?:never|at\s+no\s+time)\b'),
            ],
            'numerical': [
                (r'(\d+\.?\d*)\s*(?:percent|%|times?)', 'numerical_value'),
            ],
            'temporal': [
                (r'\bbefore\b', r'\bafter\b'),
                (r'\bprior\s+to\b', r'\bpost\b'),
                (r'\bearlier\b', r'\blater\b'),
            ]
        }

        logger.info(f"ConflictDetector initialized with semantic_threshold={semantic_threshold}, "
                   f"advanced_contradictions={enable_advanced_contradictions}, "
                   f"cross_domain={enable_cross_domain}, clustering={enable_clustering}")

    def _init_client(self):
        """Initialize OpenEvolve client if needed."""
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for conflict detection")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to instantiate OpenEvolve client: {e}")
                self.openevolve_client = None

    def _init_embeddings(self):
        """Initialize sentence transformer model if available."""
        if not self.use_embeddings:
            return

        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to load sentence transformer model: {e}. "
                             "Falling back to Jaccard similarity.")
                self.use_embeddings = False
        except ImportError:
            logger.warning("sentence-transformers not installed. "
                         "Use 'pip install sentence-transformers' for semantic similarity.")
            self.use_embeddings = False

    def _get_embedding(self, text: str) -> Optional[NDArray]:
        """
        Get embedding for text with caching.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if embeddings unavailable
        """
        if not self.use_embeddings or not self.embedding_model:
            return None

        # Check cache
        if self.cache_embeddings:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._embedding_cache:
                return self._embedding_cache[text_hash]

        try:
            import numpy as np
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)

            # Cache if enabled
            if self.cache_embeddings:
                self._embedding_cache[text_hash] = embedding

            return embedding
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _calculate_cosine_similarity(self, emb1: NDArray, emb2: NDArray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        try:
            import numpy as np
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def detect_conflicts(
        self,
        sub_solutions: Dict[str, SolutionAttempt],
        dependencies: List[SubProblem]
    ) -> List[Conflict]:
        """
        Detect all conflicts between sub-solutions with enhanced detection.

        Args:
            sub_solutions: Dict mapping sub_problem_id -> SolutionAttempt
            dependencies: List of SubProblem objects with dependency info

        Returns:
            List of detected conflicts with enhanced severity scoring
        """
        logger.info(f"Detecting conflicts among {len(sub_solutions)} sub-solutions")

        conflicts = []

        # Detect different types of conflicts
        if self.parallel_detection:
            conflicts = self._detect_conflicts_parallel(sub_solutions, dependencies)
        else:
            contradictions = self._detect_contradictions(sub_solutions)
            conflicts.extend(contradictions)

            overlaps = self._detect_semantic_overlaps(sub_solutions)
            conflicts.extend(overlaps)

            dependency_violations = self._detect_dependency_violations(sub_solutions, dependencies)
            conflicts.extend(dependency_violations)

            inconsistencies = self._detect_inconsistencies(sub_solutions)
            conflicts.extend(inconsistencies)

            if self.enable_advanced_contradictions:
                advanced = self._detect_advanced_contradictions(sub_solutions)
                conflicts.extend(advanced)

            if self.enable_cross_domain:
                cross_domain = self._detect_cross_domain_conflicts(sub_solutions)
                conflicts.extend(cross_domain)

        # Apply severity scoring to all conflicts
        conflicts = self._apply_severity_scoring(conflicts, sub_solutions)

        # Cluster conflicts if enabled
        if self.enable_clustering:
            conflicts = self._cluster_conflicts(conflicts)

        logger.info(f"Detected {len(conflicts)} total conflicts")
        return conflicts

    def _detect_conflicts_parallel(
        self,
        sub_solutions: Dict[str, SolutionAttempt],
        dependencies: List[SubProblem]
    ) -> List[Conflict]:
        """
        Detect conflicts in parallel for better performance.

        Args:
            sub_solutions: Dict mapping sub_problem_id -> SolutionAttempt
            dependencies: List of SubProblem objects with dependency info

        Returns:
            List of detected conflicts
        """
        conflicts = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._detect_contradictions, sub_solutions): 'contradictions',
                executor.submit(self._detect_semantic_overlaps, sub_solutions): 'overlaps',
                executor.submit(self._detect_dependency_violations, sub_solutions, dependencies): 'dependencies',
                executor.submit(self._detect_inconsistencies, sub_solutions): 'inconsistencies'
            }

            if self.enable_advanced_contradictions:
                futures[executor.submit(self._detect_advanced_contradictions, sub_solutions)] = 'advanced'

            if self.enable_cross_domain:
                futures[executor.submit(self._detect_cross_domain_conflicts, sub_solutions)] = 'cross_domain'

            for future in as_completed(futures):
                conflict_type = futures[future]
                try:
                    detected = future.result()
                    conflicts.extend(detected)
                    logger.debug(f"Parallel detection completed for {conflict_type}: {len(detected)} conflicts")
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    logger.error(f"Error in parallel {conflict_type} detection: {e}")

        return conflicts

    def _detect_contradictions(self, sub_solutions: Dict[str, SolutionAttempt]) -> List[Conflict]:
        """
        Find direct contradictions between sub-solutions.

        Uses semantic analysis to detect statements that directly contradict each other.
        """
        logger.info("Detecting contradictions between sub-solutions")
        contradictions = []

        # Get all pairs of sub-solutions
        solution_ids = list(sub_solutions.keys())
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i + 1:]:
                sol1 = sub_solutions[id1]
                sol2 = sub_solutions[id2]

                # Check for contradiction markers
                if self._has_contradiction_markers(sol1.solution_content, sol2.solution_content):
                    contradiction = Conflict(
                        conflict_id=generate_id("contradiction"),
                        conflict_type="contradiction",
                        severity="high",
                        involved_sub_solutions=[id1, id2],
                        description=f"Potential contradiction detected between solutions {id1} and {id2}",
                        metadata={
                            'solution1_content': sol1.solution_content[:200],
                            'solution2_content': sol2.solution_content[:200]
                        }
                    )
                    contradictions.append(contradiction)
                    logger.warning(f"Contradiction found between {id1} and {id2}")

        return contradictions

    def _has_contradiction_markers(self, content1: str, content2: str) -> bool:
        """
        Check for textual contradiction markers between two content pieces.

        This is a heuristic-based approach. For production use with LLM, this would
        use semantic analysis.
        """
        # Lowercase for comparison
        c1_lower = content1.lower()
        c2_lower = content2.lower()

        # Contradiction keyword pairs
        contradiction_pairs = [
            ('should', 'should not'),
            ('must', 'must not'),
            ('will', 'will not'),
            ('can', 'cannot'),
            ('enable', 'disable'),
            ('include', 'exclude'),
            ('add', 'remove'),
            ('increase', 'decrease'),
            ('always', 'never'),
            ('all', 'none')
        ]

        for pos, neg in contradiction_pairs:
            # Check if content1 has positive and content2 has negative
            if pos in c1_lower and neg in c2_lower:
                return True
            # Check if content1 has negative and content2 has positive
            if neg in c1_lower and pos in c2_lower:
                return True

        return False

    def _detect_semantic_overlaps(self, sub_solutions: Dict[str, SolutionAttempt]) -> List[Conflict]:
        """
        Find overlapping content between sub-solutions using semantic similarity.

        Uses embedding-based cosine similarity with fallback to Jaccard similarity.

        Args:
            sub_solutions: Dict mapping sub_problem_id -> SolutionAttempt

        Returns:
            List of overlap conflicts
        """
        logger.info("Detecting semantic overlaps between sub-solutions")
        overlaps = []

        solution_ids = list(sub_solutions.keys())
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i + 1:]:
                sol1 = sub_solutions[id1]
                sol2 = sub_solutions[id2]

                # Calculate semantic similarity
                similarity = self._calculate_semantic_similarity(
                    sol1.solution_content,
                    sol2.solution_content
                )

                # If similarity is high, flag as overlap
                if similarity > self.overlap_threshold:
                    overlap = Conflict(
                        conflict_id=generate_id("overlap"),
                        conflict_type="overlap",
                        severity="medium",
                        involved_sub_solutions=[id1, id2],
                        description=f"High content overlap ({similarity:.2%}) detected between solutions {id1} and {id2}",
                        metadata={
                            'similarity_score': round(similarity, 4),
                            'detection_method': 'semantic' if self.use_embeddings else 'jaccard'
                        }
                    )
                    overlaps.append(overlap)
                    logger.debug(f"Overlap found between {id1} and {id2} (similarity: {similarity:.2%})")

        return overlaps

    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate semantic similarity between two text contents.

        Uses embedding-based cosine similarity if available, falls back to Jaccard.

        Args:
            content1: First text content
            content2: Second text content

        Returns:
            Similarity score (0.0-1.0)
        """
        # Try embedding-based similarity first
        if self.use_embeddings and self.embedding_model:
            emb1 = self._get_embedding(content1)
            emb2 = self._get_embedding(content2)

            if emb1 is not None and emb2 is not None:
                similarity = self._calculate_cosine_similarity(emb1, emb2)
                logger.debug(f"Semantic similarity (cosine): {similarity:.4f}")
                return similarity

        # Fallback to Jaccard similarity
        similarity = self._calculate_jaccard_similarity(content1, content2)
        logger.debug(f"Semantic similarity (Jaccard fallback): {similarity:.4f}")
        return similarity

    def _calculate_jaccard_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate Jaccard similarity between two text contents.

        Args:
            content1: First text content
            content2: Second text content

        Returns:
            Jaccard similarity score (0.0-1.0)
        """
        # Tokenize into words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _detect_dependency_violations(
        self,
        sub_solutions: Dict[str, SolutionAttempt],
        dependencies: List[SubProblem]
    ) -> List[Conflict]:
        """
        Find unsatisfied dependencies between sub-solutions.

        Checks if dependencies declared in the decomposition plan are respected
        in the actual solutions.
        """
        logger.info("Detecting dependency violations")
        violations = []

        # Build dependency map
        dependency_map = {}
        for sub_problem in dependencies:
            dependency_map[sub_problem.id] = sub_problem.dependencies

        # Check each sub-solution's dependencies
        for solution_id, solution in sub_solutions.items():
            if solution_id not in dependency_map:
                continue

            deps = dependency_map[solution_id]
            for dep_id in deps:
                # Check if dependency is satisfied
                if dep_id not in sub_solutions:
                    violation = Conflict(
                        conflict_id=generate_id("dependency"),
                        conflict_type="dependency",
                        severity="critical",
                        involved_sub_solutions=[solution_id, dep_id],
                        description=f"Dependency violation: {solution_id} depends on {dep_id}, but {dep_id} has no solution",
                        metadata={'missing_dependency': dep_id}
                    )
                    violations.append(violation)
                    logger.error(f"Dependency violation: {solution_id} -> {dep_id}")

        return violations

    def _detect_inconsistencies(self, sub_solutions: Dict[str, SolutionAttempt]) -> List[Conflict]:
        """
        Find semantic inconsistencies between sub-solutions.

        This would use LLM-based semantic analysis in production.
        For now, uses heuristic patterns.
        """
        logger.info("Detecting inconsistencies between sub-solutions")
        inconsistencies = []

        # Check for inconsistent terminology, approaches, or standards
        solution_ids = list(sub_solutions.keys())

        # Extract key terms from each solution
        solution_terms = {}
        for sol_id, solution in sub_solutions.items():
            terms = self._extract_key_terms(solution.solution_content)
            solution_terms[sol_id] = terms

        # Compare terms across solutions
        for i, id1 in enumerate(solution_ids):
            for id2 in solution_ids[i + 1:]:
                terms1 = solution_terms[id1]
                terms2 = solution_terms[id2]

                # Check for conflicting approaches
                if self._has_conflicting_approaches(terms1, terms2):
                    inconsistency = Conflict(
                        conflict_id=generate_id("inconsistency"),
                        conflict_type="inconsistency",
                        severity="medium",
                        involved_sub_solutions=[id1, id2],
                        description=f"Inconsistent approaches detected between solutions {id1} and {id2}",
                        metadata={
                            'solution1_terms': list(terms1)[:10],
                            'solution2_terms': list(terms2)[:10]
                        }
                    )
                    inconsistencies.append(inconsistency)

        return inconsistencies

    def _extract_key_terms(self, content: str) -> Set[str]:
        """Extract key terms from content."""
        # Simple extraction: words that are capitalized or appear frequently
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        return set(words)

    def _has_conflicting_approaches(self, terms1: Set[str], terms2: Set[str]) -> bool:
        """Check if two sets of terms suggest conflicting approaches."""
        # This is a simplified heuristic
        # In production, would use LLM-based semantic analysis

        approach_keywords = {
            'agile': ['sprint', 'scrum', 'kanban', 'iteration'],
            'waterfall': ['phase', 'sequential', ' milestone', 'documentation'],
            'microservices': ['microservice', 'service', 'api', 'distributed'],
            'monolith': ['monolithic', 'single', 'unified', 'integrated']
        }

        approach1 = None
        approach2 = None

        for approach, keywords in approach_keywords.items():
            if any(keyword in ' '.join(terms1).lower() for keyword in keywords):
                approach1 = approach
            if any(keyword in ' '.join(terms2).lower() for keyword in keywords):
                approach2 = approach

        # If solutions use different fundamental approaches, flag as inconsistent
        return approach1 and approach2 and approach1 != approach2


# ============================================================================
# CONFLICT RESOLVER
# ============================================================================

class ConflictResolver:
    """Resolves conflicts between sub-solutions."""

    def __init__(self, openevolve_client: Optional['OpenEvolveClient'] = None):
        """Initialize with optional OpenEvolve client for LLM-mediated resolution."""
        self.openevolve_client = openevolve_client
        self._init_client()

    def _init_client(self):
        """Initialize OpenEvolve client if needed."""
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for conflict resolution")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to instantiate OpenEvolve client: {e}")
                self.openevolve_client = None

    def resolve_conflicts(
        self,
        conflicts: List[Conflict],
        sub_solutions: Dict[str, SolutionAttempt],
        resolution_strategy: str = "priority"
    ) -> List[Conflict]:
        """
        Resolve conflicts using specified strategy.

        Args:
            conflicts: List of conflicts to resolve
            sub_solutions: Dict of sub-solutions
            resolution_strategy: "priority", "merge", "llm", "manual"

        Returns:
            List of resolved conflicts
        """
        logger.info(f"Resolving {len(conflicts)} conflicts using strategy: {resolution_strategy}")

        resolved_conflicts = []

        for conflict in conflicts:
            if resolution_strategy == "priority":
                resolved = self._resolve_by_priority(conflict, sub_solutions)
            elif resolution_strategy == "merge":
                resolved = self._resolve_by_merge(conflict, sub_solutions)
            elif resolution_strategy == "llm":
                resolved = self._resolve_by_llm(conflict, sub_solutions)
            elif resolution_strategy == "manual":
                resolved = self._resolve_manually(conflict, sub_solutions)
            else:
                logger.warning(f"Unknown resolution strategy: {resolution_strategy}")
                resolved = conflict

            resolved_conflicts.append(resolved)

        logger.info(f"Resolved {len(resolved_conflicts)} conflicts")
        return resolved_conflicts

    def _resolve_by_priority(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Conflict:
        """
        Higher priority (earlier in hierarchy) wins.

        Priority is determined by the order in involved_sub_solutions list.
        The first solution has higher priority.
        """
        logger.info(f"Resolving conflict {conflict.conflict_id} by priority")

        if not conflict.involved_sub_solutions:
            conflict.status = "deferred"
            return conflict

        # First solution wins
        winner = conflict.involved_sub_solutions[0]
        resolution = f"Priority-based resolution: Solution {winner} takes precedence"

        conflict.resolution = resolution
        conflict.resolution_strategy = "priority"
        conflict.status = "resolved"

        logger.info(f"Resolved {conflict.conflict_id}: {winner} takes precedence")
        return conflict

    def _resolve_by_merge(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Conflict:
        """
        Intelligently merge conflicting solutions.

        Creates a merged content that combines the best of both solutions.
        """
        logger.info(f"Resolving conflict {conflict.conflict_id} by merge")

        if len(conflict.involved_sub_solutions) < 2:
            return self._resolve_by_priority(conflict, sub_solutions)

        # Get contents
        sol1_id = conflict.involved_sub_solutions[0]
        sol2_id = conflict.involved_sub_solutions[1]

        sol1 = sub_solutions.get(sol1_id)
        sol2 = sub_solutions.get(sol2_id)

        if not sol1 or not sol2:
            return self._resolve_by_priority(conflict, sub_solutions)

        # Simple merge: combine unique content from both
        # In production, would use more sophisticated merging
        merged_content = self._merge_contents(sol1.solution_content, sol2.solution_content)

        # Update the first solution with merged content
        sol1.solution_content = merged_content

        resolution = f"Merged content from {sol1_id} and {sol2_id}"
        conflict.resolution = resolution
        conflict.resolution_strategy = "merge"
        conflict.status = "resolved"

        logger.info(f"Merged solutions for conflict {conflict.conflict_id}")
        return conflict

    def _merge_contents(self, content1: str, content2: str) -> str:
        """
        Merge two content pieces.

        Simple concatenation with deduplication for now.
        In production, would use intelligent paragraph merging.
        """
        # Split into paragraphs
        paras1 = content1.split('\n\n')
        paras2 = content2.split('\n\n')

        # Combine and deduplicate
        all_paras = paras1 + [p for p in paras2 if p not in paras1]

        return '\n\n'.join(all_paras)

    def _resolve_by_llm(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Conflict:
        """
        Use LLM to mediate and resolve conflict.

        This is the most sophisticated resolution method.
        """
        logger.info(f"Resolving conflict {conflict.conflict_id} by LLM mediation")

        if not self.openevolve_client:
            logger.warning("LLM not available, falling back to priority-based resolution")
            return self._resolve_by_priority(conflict, sub_solutions)

        # Build prompt for LLM
        conflict_desc = self._build_conflict_description(conflict, sub_solutions)

        prompt = f"""You are an expert mediator resolving conflicts between solution components.

CONFLICT TO RESOLVE:
{conflict_desc}

TASK:
Analyze this conflict and provide a resolution that:
1. Preserves the best aspects of each solution
2. Maintains consistency and coherence
3. Ensures the final integrated solution is complete and actionable

Provide your resolution in the following format:
RESOLUTION: [Your resolution description]
APPROACH: [priority/merge/compromise/hybrid]
RATIONALE: [Why this resolution works]
"""

        try:
            response = self.openevolve_client.generate_completion(prompt)

            # Parse response
            resolution = self._parse_llm_resolution(response)

            conflict.resolution = resolution['description']
            conflict.resolution_strategy = "llm_mediated"
            conflict.status = "resolved"
            conflict.metadata['llm_resolution_details'] = resolution

            logger.info(f"LLM-mediated resolution for conflict {conflict.conflict_id}")
            return conflict

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"LLM resolution failed: {e}, falling back to priority")
            return self._resolve_by_priority(conflict, sub_solutions)

    def _build_conflict_description(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> str:
        """Build a description of the conflict for LLM analysis."""
        desc = f"Conflict Type: {conflict.conflict_type}\n"
        desc += f"Severity: {conflict.severity}\n"
        desc += f"Description: {conflict.description}\n\n"
        desc += "Involved Solutions:\n"

        for sol_id in conflict.involved_sub_solutions:
            sol = sub_solutions.get(sol_id)
            if sol:
                desc += f"\n{sol_id}:\n{sol.solution_content[:500]}\n"

        return desc

    def _parse_llm_resolution(self, response: str) -> Dict[str, str]:
        """Parse LLM resolution response."""
        resolution = {'description': '', 'approach': '', 'rationale': ''}

        # Simple parsing (can be enhanced)
        if 'RESOLUTION:' in response:
            parts = response.split('RESOLUTION:')[1].split('\n')
            if parts:
                resolution['description'] = parts[0].strip()

        if 'APPROACH:' in response:
            parts = response.split('APPROACH:')[1].split('\n')
            if parts:
                resolution['approach'] = parts[0].strip()

        if 'RATIONALE:' in response:
            parts = response.split('RATIONALE:')[1].split('\n')
            if parts:
                resolution['rationale'] = parts[0].strip()

        return resolution

    def _resolve_manually(
        self,
        conflict: Conflict,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Conflict:
        """
        Flag for human review and resolution.

        Does not automatically resolve, but marks for manual intervention.
        """
        logger.info(f"Flagging conflict {conflict.conflict_id} for manual resolution")

        resolution = "Requires manual human review and resolution"
        conflict.resolution = resolution
        conflict.resolution_strategy = "manual"
        conflict.status = "deferred"

        logger.warning(f"Conflict {conflict.conflict_id} deferred for manual resolution")
        return conflict


# ============================================================================
# SOLUTION ASSEMBLER
# ============================================================================

class SolutionAssembler:
    """Assembles sub-solutions into final integrated solution."""

    def __init__(
        self,
        conflict_detector: Optional[ConflictDetector] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        openevolve_client: Optional['OpenEvolveClient'] = None,
        enable_roma: bool = True,
        roma_max_depth: int = 2,
        roma_execution_mode: str = "recursive",
        roma_provider: Optional[str] = None,
        roma_model: Optional[str] = None,
        hephaestus_api_base: Optional[str] = None,
        hephaestus_api_key: Optional[str] = None,
        hephaestus_workflow_id: Optional[str] = None,
        hephaestus_agent_id: str = "recomposition-system",
    ):
        """Initialize with optional conflict resolver."""
        self.conflict_detector = conflict_detector or ConflictDetector(openevolve_client)
        self.conflict_resolver = conflict_resolver or ConflictResolver(openevolve_client)
        self.openevolve_client = openevolve_client
        self.enable_roma = enable_roma
        self.roma_max_depth = roma_max_depth
        self.roma_execution_mode = roma_execution_mode
        self.roma_provider = roma_provider
        self.roma_model = roma_model
        self.hephaestus_api_base = hephaestus_api_base or os.getenv(
            "HEPHAESTUS_API_BASE",
            "http://localhost:8000",
        )
        self.hephaestus_api_key = hephaestus_api_key or os.getenv("HEPHAESTUS_API_KEY")
        self.hephaestus_workflow_id = hephaestus_workflow_id or os.getenv(
            "HEPHAESTUS_WORKFLOW_ID"
        )
        self.hephaestus_agent_id = hephaestus_agent_id

        # Initialize ROMA-MDAP-MAKER Engine for robust recomposition
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT recomposition preset for standardized high-reliability assembly
                config_roma = get_recomposition_config(
                    mdap_max_samples=50,
                    mdap_min_confidence=0.4
                )
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for SolutionAssembler")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to initialize ROMA engine: {e}")

        logger.info("SolutionAssembler initialized")

    def assemble_solution(
        self,
        decomposition_plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        assembly_strategy: str = "hierarchical"
    ) -> IntegratedSolution:
        """
        Assemble individual sub-solutions into final solution.

        Args:
            decomposition_plan: Original decomposition with dependencies
            sub_solutions: Dict mapping sub_problem_id -> SolutionAttempt
            assembly_strategy: "hierarchical", "linear", "parallel", "adaptive", "roma", "roma_hephaestus"

        Returns:
            IntegratedSolution with final assembled solution
        """
        logger.info(f"Assembling solution using strategy: {assembly_strategy}")

        # Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(
            sub_solutions,
            decomposition_plan.sub_problems
        )

        # Resolve conflicts
        resolved_conflicts = self.conflict_resolver.resolve_conflicts(
            conflicts,
            sub_solutions,
            resolution_strategy="priority"  # Default strategy
        )

        # Execute assembly based on strategy
        if assembly_strategy == "hierarchical":
            assembled_content, integration_order = self._assemble_hierarchical(
                decomposition_plan,
                sub_solutions
            )
        elif assembly_strategy == "linear":
            assembled_content, integration_order = self._assemble_linear(
                decomposition_plan,
                sub_solutions
            )
        elif assembly_strategy == "parallel":
            assembled_content, integration_order = self._assemble_parallel(
                decomposition_plan,
                sub_solutions
            )
        elif assembly_strategy == "adaptive":
            assembled_content, integration_order = self._assemble_adaptive(
                decomposition_plan,
                sub_solutions
            )
        elif assembly_strategy in {"roma", "roma_hephaestus"}:
            assembled_content, integration_order = self._assemble_with_roma(
                decomposition_plan,
                sub_solutions,
                conflicts,
                resolved_conflicts,
                track_in_hephaestus=(assembly_strategy == "roma_hephaestus"),
            )
        else:
            logger.warning(f"Unknown assembly strategy: {assembly_strategy}, using hierarchical")
            assembled_content, integration_order = self._assemble_hierarchical(
                decomposition_plan,
                sub_solutions
            )

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            assembled_content,
            sub_solutions,
            conflicts
        )

        # Create integrated solution
        integrated_solution = IntegratedSolution(
            solution_id=generate_id("solution"),
            decomposition_plan_id=decomposition_plan.id,
            assembled_content=assembled_content,
            assembly_strategy=assembly_strategy,
            sub_solutions=sub_solutions,
            integration_order=integration_order,
            conflicts_detected=conflicts,
            conflicts_resolved=resolved_conflicts,
            quality_metrics=quality_metrics,
            validation_results=[],  # Will be populated by SolutionValidator
            metadata={
                'assembly_timestamp': datetime.now().isoformat(),
                'num_sub_solutions': len(sub_solutions),
                'num_conflicts': len(conflicts),
                'num_resolved': len([c for c in resolved_conflicts if c.status == 'resolved']),
                'roma_recomposition': assembly_strategy in {"roma", "roma_hephaestus"},
                'hephaestus_tracking': assembly_strategy == "roma_hephaestus",
            }
        )

        logger.info(f"Solution assembly complete: {integrated_solution.solution_id}")
        return integrated_solution

    def _assemble_hierarchical(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Tuple[str, List[str]]:
        """
        Assemble following dependency hierarchy.

        Uses topological sort to determine integration order.
        """
        logger.info("Assembling using hierarchical strategy")

        # Build dependency graph
        graph = self._build_dependency_graph(plan.sub_problems)

        # Topological sort
        integration_order = self._topological_sort(graph, sub_solutions)

        # Assemble in order
        assembled_parts = []
        for sol_id in integration_order:
            if sol_id in sub_solutions:
                sol = sub_solutions[sol_id]
                assembled_parts.append(f"## {sol_id}\n\n{sol.solution_content}")

        assembled_content = '\n\n'.join(assembled_parts)

        logger.info(f"Hierarchical assembly complete: {len(integration_order)} solutions integrated")
        return assembled_content, integration_order

    def _assemble_linear(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Tuple[str, List[str]]:
        """
        Assemble in linear sequence.

        Simple sequential assembly in the order sub-problems were created.
        """
        logger.info("Assembling using linear strategy")

        integration_order = list(sub_solutions.keys())
        assembled_parts = []

        for sol_id in integration_order:
            sol = sub_solutions[sol_id]
            assembled_parts.append(f"## {sol_id}\n\n{sol.solution_content}")

        assembled_content = '\n\n'.join(assembled_parts)

        logger.info(f"Linear assembly complete: {len(integration_order)} solutions integrated")
        return assembled_content, integration_order

    def _assemble_parallel(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Tuple[str, List[str]]:
        """
        Assemble independent sub-solutions in parallel.

        Groups sub-solutions by dependency level and assembles each group.
        """
        logger.info("Assembling using parallel strategy")

        # Identify parallelizable groups
        groups = self._identify_parallel_groups(plan.sub_problems)
        integration_order = []

        assembled_parts = []
        for group in groups:
            # Add all solutions in this group
            for sol_id in group:
                if sol_id in sub_solutions:
                    sol = sub_solutions[sol_id]
                    assembled_parts.append(f"## {sol_id}\n\n{sol.solution_content}")
                    integration_order.append(sol_id)

        assembled_content = '\n\n'.join(assembled_parts)

        logger.info(f"Parallel assembly complete: {len(groups)} groups, {len(integration_order)} solutions")
        return assembled_content, integration_order

    def _assemble_adaptive(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Tuple[str, List[str]]:
        """
        Choose assembly strategy based on structure.

        Analyzes dependency structure and selects the best strategy.
        """
        logger.info("Assembling using adaptive strategy")

        # Analyze dependency structure
        structure = self._analyze_structure(plan)

        # Select strategy based on structure
        if structure['complexity'] == 'high':
            logger.info("High complexity detected, using hierarchical assembly")
            return self._assemble_hierarchical(plan, sub_solutions)
        elif structure['parallelism'] == 'high':
            logger.info("High parallelism detected, using parallel assembly")
            return self._assemble_parallel(plan, sub_solutions)
        else:
            logger.info("Simple structure detected, using linear assembly")
            return self._assemble_linear(plan, sub_solutions)

    def _assemble_with_roma(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict],
        resolved_conflicts: List[Conflict],
        track_in_hephaestus: bool = False,
        **roma_kwargs
    ) -> Tuple[str, List[str]]:
        """
        Assemble using ROMA's recursive decomposition/aggregation logic.

        This method leverages ROMA's intelligent recomposition capabilities to:
        1. Understand semantic relationships between sub-solutions
        2. Identify and resolve conflicts using domain-aware reasoning
        3. Create coherent transitions between solution components
        4. Optimize the final integrated solution structure

        DETERMINISTIC MODE:
        When roma_deterministic=True (default), ROMA only decides structure and organization.
        Sub-solutions are inserted verbatim without modification, ensuring:
        - No mutation of original content
        - Reproducible assembly
        - Preserved technical accuracy
        - Deterministic output

        Args:
            plan: Original decomposition plan with dependencies
            sub_solutions: Dict of solved sub-problems
            conflicts: List of detected conflicts
            resolved_conflicts: List of resolved conflicts
            track_in_hephaestus: Whether to track in Hephaestus
            **roma_kwargs: Additional ROMA parameters:
                - roma_deterministic: Use deterministic assembly (default: True)
                - roma_context: Custom context string
                - roma_extra_context: Extra context appended to auto-generated
                - roma_strategy: ROMA strategy (default: "chain_of_thought")
                - roma_temperature: LLM temperature (default: 0.7)
                - roma_max_tokens: Max tokens for generation (default: 4000)

        Returns:
            Tuple of (assembled_content, integration_order)
        """
        if not (self.enable_roma and ROMA_RECOMPOSITION_AVAILABLE and solve_with_roma):
            logger.warning("ROMA recomposition not available; falling back to hierarchical assembly")
            return self._assemble_hierarchical(plan, sub_solutions)

        # Check if deterministic mode is requested
        use_deterministic = roma_kwargs.get("roma_deterministic", True)

        if use_deterministic:
            logger.info("Using DETERMINISTIC ROMA assembly (sub-solutions remain immutable)")
            return self._assemble_with_roma_deterministic(
                plan=plan,
                sub_solutions=sub_solutions,
                conflicts=conflicts,
                resolved_conflicts=resolved_conflicts,
                track_in_hephaestus=track_in_hephaestus,
                **roma_kwargs
            )

        logger.info("Starting CREATIVE ROMA recomposition (sub-solutions may be rewritten)")

        # Track in Hephaestus if requested
        if track_in_hephaestus:
            self._create_hephaestus_recomposition_task(plan, sub_solutions, conflicts)

        # Build enhanced recomposition context
        integration_order = [sp.id for sp in plan.sub_problems if sp.id in sub_solutions]

        # Build domain-aware recomposition context
        roma_context = roma_kwargs.get("roma_context") or self._build_roma_recomposition_context(
            plan=plan,
            sub_solutions=sub_solutions,
            conflicts=conflicts,
            resolved_conflicts=resolved_conflicts,
            extra_context=roma_kwargs.get("roma_extra_context"),
        )

        # Build recomposition task
        task = self._build_roma_recomposition_task(
            plan=plan,
            sub_solutions=sub_solutions,
            conflicts=conflicts,
            resolved_conflicts=resolved_conflicts,
            context=roma_context,
        )

        # ROMA-specific parameters
        roma_strategy = roma_kwargs.get("roma_strategy", "chain_of_thought")
        roma_temperature = roma_kwargs.get("roma_temperature", 0.7)
        roma_max_tokens = roma_kwargs.get("roma_max_tokens", 4000)

        try:
            # Call ROMA with enhanced configuration
            result = solve_with_roma(
                task=task,
                max_depth=self.roma_max_depth,
                execution_mode=self.roma_execution_mode,
                provider=self.roma_provider,
                model=self.roma_model,
            )

            if not isinstance(result, dict) or "error" in result:
                logger.warning(f"ROMA recomposition failed: {result.get('error', 'Unknown error')}; falling back to hierarchical assembly")
                return self._assemble_hierarchical(plan, sub_solutions)

            assembled_content = str(result.get("result", "")).strip()
            if not assembled_content:
                logger.warning("ROMA recomposition returned empty content; falling back to hierarchical assembly")
                return self._assemble_hierarchical(plan, sub_solutions)

            logger.info(f"ROMA recomposition complete: {len(assembled_content)} chars assembled")
            return assembled_content, integration_order

        except Exception as exc:  # TODO: Catch specific exception instead of Exception
            logger.error(f"ROMA recomposition exception: {exc}; falling back to hierarchical assembly")
            return self._assemble_hierarchical(plan, sub_solutions)

    def _assemble_with_roma_deterministic(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict],
        resolved_conflicts: List[Conflict],
        track_in_hephaestus: bool = False,
        **roma_kwargs
    ) -> Tuple[str, List[str]]:
        """
        Deterministic ROMA assembly - sub-solutions remain immutable.

        This method uses ROMA ONLY for structural decisions:
        1. Extracts metadata from sub-solutions (immutable extraction)
        2. Uses ROMA to determine optimal organization and transitions
        3. Inserts sub-solutions verbatim without modification
        4. Generates deterministic, reproducible output

        Key difference from creative mode:
        - CREATIVE: LLM sees full content → rewrites everything → mutations occur
        - DETERMINISTIC: LLM sees metadata only → decides structure → insert verbatim → no mutations

        Args:
            plan: Original decomposition plan with dependencies
            sub_solutions: Dict of solved sub-problems
            conflicts: List of detected conflicts
            resolved_conflicts: List of resolved conflicts
            track_in_hephaestus: Whether to track in Hephaestus
            **roma_kwargs: Additional ROMA parameters

        Returns:
            Tuple of (assembled_content, integration_order)
        """
        if not (self.enable_roma and ROMA_RECOMPOSITION_AVAILABLE and solve_with_roma):
            logger.warning("ROMA recomposition not available; falling back to hierarchical assembly")
            return self._assemble_hierarchical(plan, sub_solutions)

        logger.info("Starting DETERMINISTIC ROMA assembly (metadata-based structural decisions)")

        # Track in Hephaestus if requested
        if track_in_hephaestus:
            self._create_hephaestus_recomposition_task(plan, sub_solutions, conflicts)

        # STEP 1: Extract immutable metadata from sub-solutions
        logger.info("Extracting metadata from sub-solutions (immutable extraction)")
        solution_metadata = self._extract_solution_metadata(sub_solutions)

        # STEP 2: Build structural planning task (NO full content)
        structure_task = self._build_structure_planning_task(
            plan=plan,
            solution_metadata=solution_metadata,
            conflicts=conflicts,
            resolved_conflicts=resolved_conflicts,
            context=roma_kwargs.get("roma_context") or self._build_roma_recomposition_context(
                plan=plan,
                sub_solutions=sub_solutions,
                conflicts=conflicts,
                resolved_conflicts=resolved_conflicts,
                extra_context=roma_kwargs.get("roma_extra_context"),
            ),
        )

        # STEP 3: Use ROMA to determine optimal structure
        logger.info("Using ROMA to determine optimal assembly structure")
        try:
            structure_result = solve_with_roma(
                task=structure_task,
                max_depth=1,  # Shallow depth for structural decisions only
                execution_mode="recursive",
                provider=self.roma_provider,
                model=self.roma_model,
            )

            if not isinstance(structure_result, dict) or "error" in structure_result:
                logger.warning(f"ROMA structure planning failed: {structure_result.get('error', 'Unknown error')}; using default hierarchical assembly")
                return self._assemble_hierarchical(plan, sub_solutions)

            structure_plan = self._parse_structure_plan(structure_result.get("result", ""))

        except Exception as exc:  # TODO: Catch specific exception instead of Exception
            logger.error(f"ROMA structure planning exception: {exc}; falling back to hierarchical assembly")
            return self._assemble_hierarchical(plan, sub_solutions)

        # STEP 4: Deterministic assembly using structure plan
        logger.info(f"Assembling solution using ROMA structure plan (order: {structure_plan['order']})")
        assembled_content = self._assemble_from_structure_plan(
            structure_plan=structure_plan,
            sub_solutions=sub_solutions,
            solution_metadata=solution_metadata,
        )

        integration_order = structure_plan.get('order', [sp.id for sp in plan.sub_problems if sp.id in sub_solutions])

        logger.info(f"Deterministic ROMA assembly complete: {len(assembled_content)} chars (sub-solutions verbatim)")
        return assembled_content, integration_order

    def _extract_solution_metadata(
        self,
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract immutable metadata from sub-solutions.

        This extracts ONLY metadata, never the full content:
        - ID, title, description
        - Confidence score
        - Content length and type
        - First line/heading
        - Dependencies (from solution content structure)

        CRITICAL: Never returns full solution_content to prevent LLM rewriting.

        Args:
            sub_solutions: Dict of sub-solutions

        Returns:
            Dict mapping sub_solution_id → metadata dict
        """
        metadata = {}

        for sol_id, solution in sub_solutions.items():
            content = solution.solution_content
            lines = content.split('\n')

            # Extract title/heading (first markdown heading or first line)
            title = None
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line.startswith('#'):
                    title = line.lstrip('#').strip()
                    break
            if not title:
                title = lines[0][:50] if lines else f"Solution {sol_id}"

            # Detect content type
            content_type = "unknown"
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ['def ', 'class ', 'import ', 'function', '```']):
                content_type = "code"
            elif any(keyword in content_lower for keyword in ['##', '###', '1.', '2.', '* ']):
                content_type = "markdown"

            # Extract dependencies from content structure
            dependencies = []
            if 'depends on' in content_lower:
                # Simple extraction - can be enhanced
                dep_section = content[content_lower.find('depends on'):content_lower.find('depends on') + 100]
                import re
                deps = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', dep_section)
                dependencies = [d for d in deps if len(d) > 2 and d not in ['depends', 'on', 'the']]

            metadata[sol_id] = {
                'id': sol_id,
                'solution_id': solution.solution_id,
                'title': title,
                'description': title,  # Same as title for now
                'confidence_score': solution.confidence_score,
                'content_length': len(content),
                'content_type': content_type,
                'line_count': len(lines),
                'first_line': lines[0] if lines else '',
                'dependencies': dependencies,
                'has_code_blocks': '```' in content,
                'has_headings': any(line.startswith('#') for line in lines),
            }

        return metadata

    def _build_structure_planning_task(
        self,
        plan: DecompositionPlan,
        solution_metadata: Dict[str, Dict[str, Any]],
        conflicts: List[Conflict],
        resolved_conflicts: List[Conflict],
        context: str,
    ) -> str:
        """
        Build task for ROMA to determine optimal assembly structure.

        CRITICAL: This task includes ONLY metadata, NOT full content.
        ROMA will decide organization and transitions, but never see full solutions.

        Args:
            plan: Original decomposition plan
            solution_metadata: Extracted metadata from sub-solutions
            conflicts: List of conflicts
            resolved_conflicts: List of resolved conflicts
            context: Enhanced domain context

        Returns:
            Formatted task for ROMA
        """
        # Build metadata summary
        metadata_summary = "SUB-SOLUTION METADATA:\n"
        for sol_id, meta in solution_metadata.items():
            metadata_summary += f"""
[{sol_id}] {meta['title']}
  - Type: {meta['content_type']}
  - Confidence: {meta['confidence_score']:.2f}
  - Length: {meta['content_length']} chars, {meta['line_count']} lines
  - Dependencies: {', '.join(meta['dependencies']) if meta['dependencies'] else 'None'}
  - First line: {meta['first_line'][:60]}...
"""

        # Add conflict summary
        conflict_summary = ""
        if conflicts:
            conflict_summary = "\nCONFLICTS TO ADDRESS:\n"
            for conflict in conflicts[:5]:  # Limit to prevent overwhelming
                conflict_summary += f"- {conflict.conflict_type}: {conflict.description}\n"

        # Build structure planning task
        task = f"""You are an expert solution architect specializing in organizing technical documentation.

OBJECTIVE:
Determine the OPTIMAL STRUCTURE for assembling {len(solution_metadata)} sub-solutions into a coherent integrated solution.

{context}

{metadata_summary}

{conflict_summary}

YOUR TASK - DECIDE STRUCTURE ONLY:
Analyze the metadata above and provide:

1. ASSEMBLY ORDER: List sub-solution IDs in optimal integration order
   Format: ORDER: [id1, id2, id3, ...]
   Consider: dependencies, logical flow, confidence scores

2. SECTION HEADERS: Suggest a heading for each sub-solution
   Format: HEADERS:
   [id1]: ## Suggested Heading
   [id2]: ## Suggested Heading
   ...

3. TRANSITIONS: Brief transitions between sections (2-3 sentences each)
   Format: TRANSITIONS:
   Between [id1] and [id2]: Transition text explaining connection...
   Between [id2] and [id3]: Transition text explaining connection...
   ...

4. INTRO: Optional introduction (2-3 sentences setting context)
   Format: INTRO: [Introduction text or "NONE"]

5. CONCLUSION: Optional conclusion (2-3 sentences summarizing integration)
   Format: CONCLUSION: [Conclusion text or "NONE"]

CRITICAL CONSTRAINTS:
- You are deciding STRUCTURE and ORGANIZATION ONLY
- Do NOT rewrite or summarize sub-solution content
- Sub-solutions will be inserted VERBATIM in your specified order
- Focus on logical flow, dependencies, and coherence
- Transitions should be brief and connective only

Provide your structural plan now:"""

        return task

    def _parse_structure_plan(self, structure_text: str) -> Dict[str, Any]:
        """
        Parse ROMA's structural plan into structured data.

        Args:
            structure_text: ROMA's response with structure plan

        Returns:
            Dict with parsed structure plan:
            {
                'order': [id1, id2, ...],
                'headers': {id1: 'Heading 1', id2: 'Heading 2', ...},
                'transitions': [(id1, id2, 'transition text'), ...],
                'intro': 'Intro text or None',
                'conclusion': 'Conclusion text or None'
            }
        """
        import re

        structure = {
            'order': [],
            'headers': {},
            'transitions': [],
            'intro': None,
            'conclusion': None,
        }

        lines = structure_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            # Parse ORDER
            if line.startswith('ORDER:') or line.startswith('Order:'):
                match = re.search(r'\[([^\]]+)\]', line)
                if match:
                    ids_str = match.group(1)
                    structure['order'] = [id.strip() for id in ids_str.split(',')]
                else:
                    # Try to find list of IDs after colon
                    ids = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line.split(':', 1)[1])
                    structure['order'] = ids

            # Parse HEADERS
            elif line.startswith('HEADERS:') or line.startswith('Headers:') or current_section == 'headers':
                if 'HEADERS' in line.upper():
                    current_section = 'headers'
                    continue
                match = re.match(r'\[([^\]]+)\]:\s*(.+)', line)
                if match:
                    sol_id, header = match.groups()
                    structure['headers'][sol_id.strip()] = header.strip()

            # Parse TRANSITIONS
            elif line.startswith('TRANSITIONS:') or line.startswith('Transitions:') or current_section == 'transitions':
                if 'TRANSITION' in line.upper():
                    current_section = 'transitions'
                    continue
                match = re.match(r'Between\s+([^\s]+)\s+and\s+([^\s]+):\s*(.+)', line, re.IGNORECASE)
                if match:
                    id1, id2, transition = match.groups()
                    structure['transitions'].append((id1.strip(), id2.strip(), transition.strip()))

            # Parse INTRO
            elif line.upper().startswith('INTRO:'):
                intro_text = line.split(':', 1)[1].strip()
                if intro_text and intro_text.upper() != 'NONE':
                    structure['intro'] = intro_text

            # Parse CONCLUSION
            elif line.upper().startswith('CONCLUSION:'):
                conclusion_text = line.split(':', 1)[1].strip()
                if conclusion_text and conclusion_text.upper() != 'NONE':
                    structure['conclusion'] = conclusion_text

        # Fallback: if no order parsed, try to extract any sequence
        if not structure['order']:
            # Look for numbered list or bullet points
            ids = re.findall(r'(?:^|\D)\s*(?:\d+\.|[-*])\s*([a-zA-Z_][a-zA-Z0-9_]*)', structure_text)
            if ids:
                structure['order'] = ids

        return structure

    def _assemble_from_structure_plan(
        self,
        structure_plan: Dict[str, Any],
        sub_solutions: Dict[str, SolutionAttempt],
        solution_metadata: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Assemble final solution using ROMA's structure plan.

        Sub-solutions are INSERTED VERBATIM - no modifications, no rewriting.
        Only structure (headers, transitions, order) is from ROMA.

        Args:
            structure_plan: ROMA's structural decisions
            sub_solutions: Original sub-solutions (will be inserted verbatim)
            solution_metadata: Extracted metadata

        Returns:
            Assembled solution with verbatim sub-solutions
        """
        parts = []

        # Add intro if provided
        if structure_plan.get('intro'):
            parts.append(structure_plan['intro'])
            parts.append("")  # Blank line

        # Add sub-solutions in specified order
        order = structure_plan.get('order', list(sub_solutions.keys()))
        headers = structure_plan.get('headers', {})
        transitions = structure_plan.get('transitions', [])

        # Build transition map for quick lookup
        transition_map = {}
        for id1, id2, text in transitions:
            transition_map[(id1, id2)] = text

        for i, sol_id in enumerate(order):
            if sol_id not in sub_solutions:
                logger.warning(f"Sub-solution {sol_id} in order but not in sub_solutions")
                continue

            # Add transition if not first section
            if i > 0:
                prev_id = order[i - 1]
                transition_key = (prev_id, sol_id)
                if transition_key in transition_map:
                    parts.append(transition_map[transition_key])
                    parts.append("")  # Blank line

            # Add header (from ROMA or fallback to metadata)
            header = headers.get(sol_id) or solution_metadata[sol_id].get('title', sol_id)
            parts.append(f"## {header}")
            parts.append("")  # Blank line

            # Insert sub-solution VERBATIM (no modification)
            parts.append(sub_solutions[sol_id].solution_content)
            parts.append("")  # Blank line

        # Add conclusion if provided
        if structure_plan.get('conclusion'):
            parts.append(structure_plan['conclusion'])
            parts.append("")  # Blank line

        assembled = '\n'.join(parts)

        logger.info(f"Assembly complete: {len(assembled)} chars ({len(order)} sub-solutions inserted verbatim)")
        return assembled

    def _build_recomposition_summary(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict],
        resolved_conflicts: List[Conflict],
        max_chars: int = 2000,
    ) -> str:
        """
        Build a summary of the recomposition task.

        This provides context to ROMA about what needs to be recomposed.

        Args:
            plan: Original decomposition plan
            sub_solutions: Dict of sub-solutions
            conflicts: List of detected conflicts
            resolved_conflicts: List of resolved conflicts
            max_chars: Max characters per sub-solution content

        Returns:
            Formatted summary string
        """
        parts = [f"Problem: {plan.problem_statement}\n"]
        parts.append("Sub-solutions:\n")
        for sp in plan.sub_problems:
            if sp.id not in sub_solutions:
                continue
            sol = sub_solutions[sp.id]
            content = sol.solution_content or ""
            if len(content) > max_chars:
                content = f"{content[:max_chars]}...\n[truncated]"
            parts.append(f"- {sp.id} ({sp.description}):\n{content}\n")

        if conflicts:
            parts.append("Detected conflicts:\n")
            for conflict in conflicts:
                parts.append(
                    f"- {conflict.conflict_id}: {conflict.conflict_type} | severity={conflict.severity}\n"
                    f"  description: {conflict.description}\n"
                )

        if resolved_conflicts:
            parts.append("Resolved conflicts:\n")
            for conflict in resolved_conflicts:
                if conflict.resolution:
                    parts.append(
                        f"- {conflict.conflict_id}: {conflict.resolution_strategy}\n"
                        f"  resolution: {conflict.resolution}\n"
                    )

        return "\n".join(parts)

    def _build_roma_recomposition_context(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict],
        resolved_conflicts: List[Conflict],
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Build enhanced domain-aware context for ROMA recomposition.

        This provides ROMA with:
        1. Domain information from the original problem
        2. Solution quality metrics
        3. Conflict resolution guidance
        4. Assembly strategy recommendations

        Args:
            plan: Original decomposition plan
            sub_solutions: Dict of sub-solutions
            conflicts: List of detected conflicts
            resolved_conflicts: List of resolved conflicts
            extra_context: Additional context to append

        Returns:
            Enhanced context string for ROMA
        """
        context_parts = []

        # Add problem domain context
        if hasattr(plan, 'domain_context') and plan.domain_context:
            domain = plan.domain_context
            context_parts.append("Domain Context:")
            if hasattr(domain, 'domain'):
                context_parts.append(f"- Domain: {domain.domain}")
            if hasattr(domain, 'subdomain'):
                context_parts.append(f"- Subdomain: {domain.subdomain}")
            if hasattr(domain, 'key_concepts'):
                concepts = getattr(domain, 'key_concepts', [])
                if concepts:
                    context_parts.append(f"- Key Concepts: {', '.join(concepts[:5])}")
            context_parts.append("")

        # Add solution quality overview
        context_parts.append("Solution Quality Overview:")
        num_solutions = len(sub_solutions)
        num_conflicts = len(conflicts)
        num_resolved = len([c for c in resolved_conflicts if c.status == 'resolved'])
        context_parts.append(f"- Sub-solutions: {num_solutions}")
        context_parts.append(f"- Conflicts detected: {num_conflicts}")
        context_parts.append(f"- Conflicts resolved: {num_resolved}")
        context_parts.append("")

        # Add conflict resolution guidance
        if conflicts:
            context_parts.append("Conflict Resolution Guidance:")
            critical_conflicts = [c for c in conflicts if c.severity == 'critical']
            high_conflicts = [c for c in conflicts if c.severity == 'high']

            if critical_conflicts:
                context_parts.append(f"- CRITICAL: {len(critical_conflicts)} critical conflicts must be resolved")
            if high_conflicts:
                context_parts.append(f"- HIGH: {len(high_conflicts)} high-severity conflicts need attention")

            # Add resolution strategies used
            strategies_used = set(c.resolution_strategy for c in resolved_conflicts if c.resolution_strategy)
            if strategies_used:
                context_parts.append(f"- Resolution strategies: {', '.join(strategies_used)}")
            context_parts.append("")

        # Add assembly recommendations
        context_parts.append("Assembly Recommendations:")
        structure = self._analyze_structure(plan)
        context_parts.append(f"- Structure complexity: {structure['complexity']}")
        context_parts.append(f"- Parallelism potential: {structure['parallelism']}")
        context_parts.append(f"- Recommended approach: {self._get_recommended_approach(structure)}")
        context_parts.append("")

        # Add extra context if provided
        if extra_context:
            context_parts.append(str(extra_context).strip())

        return "\n".join(context_parts).strip()

    def _build_roma_recomposition_task(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict],
        resolved_conflicts: List[Conflict],
        context: str,
    ) -> str:
        """
        Build the ROMA recomposition task prompt.

        This creates a comprehensive prompt that guides ROMA to:
        1. Understand the recomposition objectives
        2. Apply domain knowledge appropriately
        3. Follow conflict resolution guidance
        4. Create coherent integrated solutions

        Args:
            plan: Original decomposition plan
            sub_solutions: Dict of sub-solutions
            conflicts: List of detected conflicts
            resolved_conflicts: List of resolved conflicts
            context: Enhanced domain context

        Returns:
            Formatted task prompt for ROMA
        """
        # Build base summary
        summary = self._build_recomposition_summary(
            plan, sub_solutions, conflicts, resolved_conflicts
        )

        # Add context if available
        if context:
            summary = f"{context}\n\n{summary}"

        # Build comprehensive task
        task = f"""You are an expert solution integrator specializing in recomposing sub-solutions into coherent integrated solutions.

OBJECTIVE:
Recompose the following sub-solutions into a single, coherent, and well-structured integrated solution.

{summary}

RECOMPOSITION GUIDELINES:
1. Semantic Integration: Create smooth transitions between sub-solutions
2. Logical Flow: Organize content to follow natural problem-solving progression
3. Coherence: Ensure consistent terminology, style, and voice throughout
4. Completeness: Include all essential elements from each sub-solution
5. Conflict Resolution: Apply the provided conflict resolutions consistently
6. Structure: Use clear headings and organization appropriate for the content type

OUTPUT REQUIREMENTS:
- Return ONLY the final integrated solution content
- Use Markdown formatting for structure (headings, lists, code blocks, etc.)
- Ensure the final solution is self-contained and immediately actionable
- Do NOT include meta-commentary or explanations of the recomposition process

Begin recomposition now:"""

        return task

    def _get_recommended_approach(self, structure: Dict[str, str]) -> str:
        """
        Get recommended assembly approach based on structure analysis.

        Args:
            structure: Structure analysis from _analyze_structure

        Returns:
            Recommended approach description
        """
        complexity = structure.get('complexity', 'medium')
        parallelism = structure.get('parallelism', 'medium')

        if complexity == 'high' and parallelism == 'low':
            return "hierarchical assembly with careful dependency ordering"
        elif complexity == 'low' and parallelism == 'high':
            return "parallel assembly of independent components"
        elif complexity == 'high':
            return "adaptive assembly with conflict prioritization"
        else:
            return "linear assembly with quality checks"

    def _create_hephaestus_recomposition_task(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict],
    ) -> None:
        if not HEPHAESTUS_RECOMPOSITION_AVAILABLE or not requests:
            return
        if not (self.hephaestus_api_base and self.hephaestus_api_key and self.hephaestus_workflow_id):
            return

        task_description = (
            "Recomposition task: Integrate sub-solutions into a single coherent solution. "
            f"Problem: {plan.problem_statement}"
        )
        done_definition = "Integrated solution assembled with conflicts addressed."
        payload = {
            "task_description": task_description,
            "done_definition": done_definition,
            "ai_agent_id": self.hephaestus_agent_id,
            "workflow_id": self.hephaestus_workflow_id,
            "priority": "medium",
        }

        try:
            response = requests.post(
                f"{self.hephaestus_api_base.rstrip('/')}/create_task",
                json=payload,
                headers={"X-API-Key": self.hephaestus_api_key},
                timeout=10,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Hephaestus recomposition task creation failed: %s",
                    response.text,
                )
            else:
                logger.info("Hephaestus recomposition task created")
        except Exception as exc:  # TODO: Catch specific exception instead of Exception
            logger.warning("Hephaestus recomposition task creation error: %s", exc)

    def _build_dependency_graph(self, sub_problems: List[SubProblem]) -> Dict[str, List[str]]:
        """Build dependency graph from sub-problems."""
        graph = {}
        for sp in sub_problems:
            graph[sp.id] = sp.dependencies
        return graph

    def _topological_sort(
        self,
        graph: Dict[str, List[str]],
        sub_solutions: Dict[str, SolutionAttempt]
    ) -> List[str]:
        """
        Perform topological sort on dependency graph.

        Returns order in which solutions should be integrated.
        """
        # Build adjacency list and in-degree count
        in_degree = {node: 0 for node in sub_solutions.keys()}
        adjacency = {node: [] for node in sub_solutions.keys()}

        for node in sub_solutions.keys():
            if node in graph:
                for dep in graph[node]:
                    if dep in sub_solutions:
                        adjacency[dep].append(node)
                        in_degree[node] += 1

        # Kahn's algorithm
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _identify_parallel_groups(self, sub_problems: List[SubProblem]) -> List[List[str]]:
        """
        Identify groups of sub-problems that can be solved in parallel.

        Groups are based on dependency levels.
        """
        # Build dependency graph
        graph = self._build_dependency_graph(sub_problems)
        all_ids = [sp.id for sp in sub_problems]

        # Calculate dependency depth for each node
        depth = {}
        for node_id in all_ids:
            depth[node_id] = self._calculate_depth(node_id, graph, {})

        # Group by depth
        groups = defaultdict(list)
        for node_id, d in depth.items():
            groups[d].append(node_id)

        # Return groups in order
        sorted_groups = [groups[d] for d in sorted(groups.keys())]
        return sorted_groups

    def _calculate_depth(
        self,
        node_id: str,
        graph: Dict[str, List[str]],
        memo: Dict[str, int]
    ) -> int:
        """Calculate dependency depth for a node."""
        if node_id in memo:
            return memo[node_id]

        if node_id not in graph or not graph[node_id]:
            memo[node_id] = 0
            return 0

        max_dep_depth = 0
        for dep in graph[node_id]:
            dep_depth = self._calculate_depth(dep, graph, memo)
            max_dep_depth = max(max_dep_depth, dep_depth)

        memo[node_id] = max_dep_depth + 1
        return memo[node_id]

    def _analyze_structure(self, plan: DecompositionPlan) -> Dict[str, str]:
        """
        Analyze dependency structure to inform adaptive strategy selection.

        Returns dict with 'complexity' and 'parallelism' assessments.
        """
        num_sub_problems = len(plan.sub_problems)

        # Count dependencies
        total_deps = sum(len(sp.dependencies) for sp in plan.sub_problems)
        avg_deps = total_deps / num_sub_problems if num_sub_problems > 0 else 0

        # Assess complexity
        if num_sub_problems > 7 or avg_deps > 2:
            complexity = "high"
        elif num_sub_problems > 4 or avg_deps > 1:
            complexity = "medium"
        else:
            complexity = "low"

        # Assess parallelism (low avg deps = high parallelism potential)
        if avg_deps < 0.5:
            parallelism = "high"
        elif avg_deps < 1.5:
            parallelism = "medium"
        else:
            parallelism = "low"

        return {
            'complexity': complexity,
            'parallelism': parallelism,
            'num_sub_problems': num_sub_problems,
            'avg_dependencies': avg_deps
        }

    def _calculate_quality_metrics(
        self,
        assembled_content: str,
        sub_solutions: Dict[str, SolutionAttempt],
        conflicts: List[Conflict]
    ) -> SolutionQualityMetrics:
        """
        Calculate enhanced quality metrics for the integrated solution.

        Enhanced metrics include:
        - Core metrics (backward compatible)
        - Readability analysis
        - Information density
        - Integration quality
        - Conflict impact scoring
        - Quality predictions
        - Multi-dimensional quality profile
        """
        logger.info("Calculating enhanced quality metrics")

        # =========================================================================
        # CORE METRICS (backward compatible)
        # =========================================================================

        # Completeness: based on having solutions for all sub-problems
        completeness = 1.0  # Will be adjusted based on actual coverage

        # Consistency: inverse of conflict severity
        critical_conflicts = len([c for c in conflicts if c.severity == 'critical'])
        high_conflicts = len([c for c in conflicts if c.severity == 'high'])
        medium_conflicts = len([c for c in conflicts if c.severity == 'medium'])
        low_conflicts = len([c for c in conflicts if c.severity == 'low'])
        total_conflicts = len(conflicts)

        conflict_penalty = (critical_conflicts * 0.3 + high_conflicts * 0.1 + total_conflicts * 0.05)
        consistency = max(0.0, 1.0 - conflict_penalty)

        # Enhanced coherence with multiple dimensions
        coherence = self._assess_enhanced_coherence(assembled_content)

        # Integration quality: based on how well solutions fit together
        integration_quality = (consistency + coherence) / 2

        # Conflict score: lower is better
        conflict_score = min(1.0, total_conflicts * 0.1)

        # =========================================================================
        # ENHANCED READABILITY METRICS
        # =========================================================================

        readability_metrics = self._calculate_readability_metrics(assembled_content)
        readability_score = readability_metrics['flesch_score']
        sentence_complexity = readability_metrics['sentence_complexity']
        paragraph_quality = readability_metrics['paragraph_quality']
        jargon_density = readability_metrics['jargon_density']

        # =========================================================================
        # INFORMATION DENSITY METRICS
        # =========================================================================

        density_metrics = self._calculate_information_density(assembled_content, sub_solutions)
        information_density = density_metrics['density']
        redundancy_score = density_metrics['redundancy']
        information_balance = density_metrics['balance']

        # =========================================================================
        # ADVANCED INTEGRATION QUALITY METRICS
        # =========================================================================

        integration_metrics = self._calculate_integration_quality(
            assembled_content, sub_solutions
        )
        seamlessness_score = integration_metrics['seamlessness']
        terminology_consistency = integration_metrics['terminology']
        style_consistency = integration_metrics['style']
        voice_consistency = integration_metrics['voice']

        # =========================================================================
        # ENHANCED CONFLICT IMPACT SCORING
        # =========================================================================

        conflict_impact = self._calculate_conflict_impact(conflicts)
        conflict_impact_score = conflict_impact['impact_score']
        critical_conflict_resolved = conflict_impact['critical_resolved_fraction']
        deferred_conflict_penalty = conflict_impact['deferred_penalty']

        # =========================================================================
        # QUALITY PREDICTION AND COMPARATIVE METRICS
        # =========================================================================

        prediction_metrics = self._calculate_quality_prediction(
            assembled_content, sub_solutions, conflicts
        )
        predicted_quality = prediction_metrics['predicted_quality']
        quality_percentile = prediction_metrics['percentile']
        improvement_trend = prediction_metrics['trend']

        # =========================================================================
        # MULTI-DIMENSIONAL QUALITY DASHBOARD
        # =========================================================================

        quality_dashboard = self._generate_quality_dashboard({
            'completeness': completeness,
            'consistency': consistency,
            'coherence': coherence,
            'readability': readability_score,
            'information_density': information_density,
            'seamlessness': seamlessness_score,
            'terminology': terminology_consistency,
            'style': style_consistency
        })

        quality_profile = quality_dashboard['profile']
        weak_areas = quality_dashboard['weak_areas']
        improvement_suggestions = quality_dashboard['suggestions']

        # =========================================================================
        # CALCULATE ENHANCED OVERALL SCORE
        # =========================================================================

        # Weighted average with enhanced metrics
        overall_score = (
            completeness * 0.20 +
            consistency * 0.15 +
            coherence * 0.15 +
            readability_score * 0.10 +
            information_density * 0.10 +
            seamlessness_score * 0.10 +
            terminology_consistency * 0.05 +
            style_consistency * 0.05 +
            (1.0 - conflict_impact_score) * 0.10
        )

        # Build details dictionary
        details = {
            # Basic info
            'num_sub_solutions': len(sub_solutions),
            'num_conflicts': total_conflicts,
            'critical_conflicts': critical_conflicts,
            'high_conflicts': high_conflicts,
            'medium_conflicts': medium_conflicts,
            'low_conflicts': low_conflicts,

            # Readability breakdown
            'flesch_raw': readability_metrics['flesch_raw'],
            'avg_sentence_length': readability_metrics['avg_sentence_length'],
            'avg_syllables_per_word': readability_metrics['avg_syllables'],
            'total_sentences': readability_metrics['total_sentences'],
            'total_words': readability_metrics['total_words'],

            # Information density breakdown
            'sections_analyzed': density_metrics['num_sections'],
            'avg_info_per_section': density_metrics['avg_info_per_section'],
            'redundant_phrases_found': density_metrics['redundant_phrases'],

            # Integration breakdown
            'seams_detected': integration_metrics['seams_detected'],
            'terminology_variations': integration_metrics['terminology_variations'],
            'style_transitions': integration_metrics['style_transitions'],

            # Conflict breakdown
            'resolved_conflicts': len([c for c in conflicts if c.status == 'resolved']),
            'deferred_conflicts': len([c for c in conflicts if c.status == 'deferred']),
            'unresolved_conflicts': len([c for c in conflicts if c.status == 'unresolved']),

            # Metrics version
            'metrics_version': '2.0-enhanced',
            'metrics_timestamp': datetime.now().isoformat()
        }

        logger.info(f"Quality metrics calculated - Overall: {overall_score:.2f}")

        return SolutionQualityMetrics(
            # Core metrics (backward compatible)
            completeness_score=completeness,
            consistency_score=consistency,
            coherence_score=coherence,
            integration_quality=integration_quality,
            conflict_score=conflict_score,
            overall_score=overall_score,

            # Enhanced readability
            readability_score=readability_score,
            sentence_complexity=sentence_complexity,
            paragraph_quality=paragraph_quality,
            jargon_density=jargon_density,

            # Information density
            information_density=information_density,
            redundancy_score=redundancy_score,
            information_balance=information_balance,

            # Integration quality
            seamlessness_score=seamlessness_score,
            terminology_consistency=terminology_consistency,
            style_consistency=style_consistency,
            voice_consistency=voice_consistency,

            # Conflict metrics
            conflict_impact_score=conflict_impact_score,
            critical_conflict_resolved=critical_conflict_resolved,
            deferred_conflict_penalty=deferred_conflict_penalty,

            # Prediction and comparative
            predicted_quality=predicted_quality,
            quality_percentile=quality_percentile,
            improvement_trend=improvement_trend,

            # Quality profile
            quality_profile=quality_profile,
            weak_areas=weak_areas,
            improvement_suggestions=improvement_suggestions,

            details=details
        )

    def _assess_coherence(self, content: str) -> float:
        """
        Assess coherence of assembled content.

        Simple heuristic based on paragraph connectivity.
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        if len(paragraphs) < 2:
            return 1.0

        # Check for transitional phrases between paragraphs
        transitional_phrases = [
            'therefore', 'however', 'furthermore', 'consequently',
            'additionally', 'moreover', 'thus', 'hence'
        ]

        transition_count = 0
        for i in range(len(paragraphs) - 1):
            # Check if paragraph starts with transitional phrase
            next_para_start = paragraphs[i + 1].lower().split()[:3]
            if any(phrase in ' '.join(next_para_start) for phrase in transitional_phrases):
                transition_count += 1

        # Coherence score based on transitions
        coherence = 0.5 + (transition_count / (len(paragraphs) - 1)) * 0.5
        return min(1.0, coherence)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_solution_assembler(
    openevolve_client: Optional['OpenEvolveClient'] = None
) -> SolutionAssembler:
    """
    Factory function to create a SolutionAssembler.

    Args:
        openevolve_client: Optional OpenEvolve client for enhanced conflict resolution

    Returns:
        Configured SolutionAssembler instance
    """
    detector = ConflictDetector(openevolve_client)
    resolver = ConflictResolver(openevolve_client)
    return SolutionAssembler(detector, resolver, openevolve_client)
