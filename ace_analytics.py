"""
ACE Analytics Module

This module provides ML-based pattern mining, team performance tracking,
and gauntlet effectiveness analysis for Stage 6 Knowledge Extraction.

Components:
- SolutionPatternMiner: ML clustering for pattern discovery
- TeamPerformanceTracker: Team effectiveness analytics
- GauntletEffectivenessAnalyzer: Gauntlet performance metrics

SECURITY FIXES APPLIED:
- Phase 1: Input validation for all numeric parameters
- Phase 2: Thread safety with locks and synchronization
- Phase 3: Resource management with history limits and cleanup
- Phase 4: File path validation for all file operations
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import json
import logging
import os
import sys
import copy

# SECURITY FIX: Import all security utilities
try:
    from ace_security_utils import (
        validate_numeric_range,
        validate_list_size,
        validate_file_path_safe,
        safe_load_json_file,
        atomic_save_json_file,
        get_global_lock,
        synchronized,
    )
    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    # Fallback implementations if security utils not available
    SECURITY_UTILS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ace_security_utils not available, using fallback implementations")

    def validate_numeric_range(value, name, min_val=None, max_val=None, value_type=(int, float), allow_nan=False, allow_infinity=False):
        """Fallback validation for numeric range."""
        if not isinstance(value, value_type):
            raise ValueError(f"{name} must be {value_type.__name__}, got {type(value).__name__}")
        if isinstance(value, float):
            if not allow_nan and hasattr(value, 'isnan') and value.isnan():
                raise ValueError(f"{name} cannot be NaN")
            if not allow_infinity and hasattr(value, 'isinf') and value.isinf():
                raise ValueError(f"{name} cannot be Infinity")
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        return value

    def validate_list_size(items, name, max_size=10000, min_size=0, allow_empty=True):
        """Fallback validation for list size."""
        if not isinstance(items, list):
            raise ValueError(f"{name} must be a list, got {type(items).__name__}")
        list_len = len(items)
        if not allow_empty and list_len == 0:
            raise ValueError(f"{name} cannot be empty")
        if list_len < min_size:
            raise ValueError(f"{name} too small: {list_len} items (min: {min_size})")
        if list_len > max_size:
            raise ValueError(f"{name} too large: {list_len} items (max: {max_size})")
        return items

    def validate_file_path_safe(filepath, base_dir="."):
        """Fallback validation for file path."""
        if not filepath or not isinstance(filepath, str):
            raise ValueError("File path must be a non-empty string")
        if len(filepath) > 1000:
            raise ValueError("File path too long")
        suspicious = ['..', '~', '$', '|', ';', '&', '`', '\n', '\r', '\x00']
        if any(pattern in filepath for pattern in suspicious):
            raise ValueError(f"File path contains suspicious characters: {filepath}")
        return filepath

    def safe_load_json_file(filepath, max_size=10485760):
        """Fallback for loading JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def atomic_save_json_file(filepath, data):
        """Fallback for saving JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_global_lock(name):
        """Fallback for global lock."""
        return threading.RLock()

    def synchronized(lock=None):
        """Fallback for synchronized decorator."""
        def decorator(func):
            return func
        return decorator

# Optional ML dependencies
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# Import knowledge artifacts
from ace_knowledge_artifacts import (
    KnowledgeArtifact,
    SolutionPattern,
    TeamPerformanceData,
    GauntletEffectivenessData,
    WorkflowExtractionResult,
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SOLUTION PATTERN MINER (ML-BASED)
# ============================================================================

class SolutionPatternMiner:
    """
    Mine solution patterns from artifacts using ML clustering.

    Uses TF-IDF and clustering algorithms to discover groups of
    similar solution patterns that can be consolidated into
    higher-level patterns.

    Security Fixes:
    - Phase 1: Input validation for all parameters (EC-2, EC-8)
    - Phase 3: Resource limits for artifact lists (EC-1)
    - Phase 3: ML object cleanup in finally blocks (RM-1)
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.7,
        clustering_algorithm: str = "kmeans",  # "kmeans" or "dbscan"
        max_artifacts: int = 10000,  # MEMORY FIX: Add max artifacts limit
    ):
        """
        Initialize the pattern miner.

        Args:
            min_cluster_size: Minimum artifacts to form a pattern cluster (must be >= 2)
            similarity_threshold: Minimum similarity for clustering (0-1)
            clustering_algorithm: Algorithm to use ("kmeans" or "dbscan")
            max_artifacts: Maximum artifacts to process (memory limit)

        Raises:
            ValueError: If parameters are invalid
        """
        # SECURITY FIX: EC-2 - Validate min_cluster_size (removed duplicate validation)
        min_cluster_size = validate_numeric_range(
            min_cluster_size, "min_cluster_size",
            min_val=2, max_val=1000,
            value_type=int, allow_nan=False, allow_infinity=False
        )

        # SECURITY FIX: EC-2 - Validate similarity_threshold (no NaN/Inf)
        similarity_threshold = validate_numeric_range(
            similarity_threshold, "similarity_threshold",
            min_val=0.0, max_val=1.0,
            value_type=float, allow_nan=False, allow_infinity=False
        )

        # SECURITY FIX: EC-8 - Validate clustering_algorithm enum
        if clustering_algorithm not in ("kmeans", "dbscan"):
            raise ValueError(f"clustering_algorithm must be 'kmeans' or 'dbscan', got '{clustering_algorithm}'")

        # MEMORY FIX: Validate max_artifacts
        max_artifacts = validate_numeric_range(
            max_artifacts, "max_artifacts",
            min_val=1, max_val=1000000,
            value_type=int, allow_nan=False, allow_infinity=False
        )

        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.clustering_algorithm = clustering_algorithm
        self.max_artifacts = max_artifacts  # MEMORY FIX
        self.ml_available = ML_AVAILABLE

        # CONCURRENCY FIX: Add lock for ML operations
        self._ml_lock = threading.Lock()

        if not self.ml_available:
            logger.warning("ML not available - using fallback pattern mining")

    def mine_patterns_from_artifacts(
        self,
        artifacts: List[KnowledgeArtifact],
        max_patterns: int = 10,
    ) -> List[SolutionPattern]:
        """
        Mine solution patterns from a list of artifacts.

        Args:
            artifacts: List of knowledge artifacts to analyze
            max_patterns: Maximum number of patterns to extract

        Returns:
            List of mined solution patterns
        """
        if not artifacts:
            return []

        logger.info(f"Mining patterns from {len(artifacts)} artifacts")

        if self.ml_available:
            return self._mine_patterns_with_ml(artifacts, max_patterns)
        else:
            return self._mine_patterns_fallback(artifacts, max_patterns)

    def _mine_patterns_with_ml(
        self,
        artifacts: List[KnowledgeArtifact],
        max_patterns: int,
    ) -> List[SolutionPattern]:
        """
        Mine patterns using ML clustering.

        Security Fixes:
        - EC-1: Validate list size to prevent DoS
        - RM-1: Cleanup ML objects in finally block
        """
        # SECURITY FIX: EC-1 - Validate artifact list size
        artifacts = validate_list_size(
            artifacts, "artifacts",
            max_size=10000,
            min_size=0,
            allow_empty=True
        )

        # DEEP COPY FIX: Deep copy artifacts to prevent external modification
        artifacts = copy.deepcopy(artifacts)

        # MEMORY FIX: Limit artifacts before processing
        if len(artifacts) > self.max_artifacts:
            logger.warning(f"Too many artifacts ({len(artifacts)}), using first {self.max_artifacts}")
            artifacts = artifacts[:self.max_artifacts]

        patterns = []
        vectorizer = None
        tfidf_matrix = None
        cluster_model = None

        # CONCURRENCY FIX: Synchronize ML operations
        with self._ml_lock:
            try:
                # Extract artifact contents
                contents = [artifact.content for artifact in artifacts]

                # Create TF-IDF vectors
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words="english",
                    ngram_range=(1, 2),
                )
                tfidf_matrix = vectorizer.fit_transform(contents)

                # Perform clustering
                if self.clustering_algorithm == "kmeans":
                    # BUG FIX #4: Fix infinite loop potential - ensure n_clusters >= 2
                    n_clusters = min(max_patterns, len(artifacts) // self.min_cluster_size)
                    if n_clusters < 2:
                        logger.warning(f"n_clusters={n_clusters} too small, using fallback")
                        return self._mine_patterns_fallback(artifacts, max_patterns)

                    cluster_model = KMeans(
                        n_clusters=max(2, n_clusters),  # Ensure at least 2
                        random_state=42,
                        n_init=10,
                    )
                    clusters = cluster_model.fit_predict(tfidf_matrix)
                else:  # dbscan
                    # Calculate eps from similarity threshold
                    # For cosine distance: distance = 1 - similarity
                    # eps should be positive and typically between 0.1 and 1.0
                    eps_value = 1.0 - self.similarity_threshold
                    # BUG FIX #5: Fix floating point equality - use epsilon comparison
                    if eps_value < 0.001:  # Use epsilon comparison instead of <= 0
                        logger.warning(f"Invalid eps value {eps_value}, using fallback 0.3")
                        eps_value = 0.3
                    elif eps_value > 1.0:
                        logger.warning(f"Eps value {eps_value} too large, clamping to 1.0")
                        eps_value = 1.0

                    cluster_model = DBSCAN(
                        eps=eps_value,
                        min_samples=self.min_cluster_size,
                        metric="cosine",
                    )
                    clusters = cluster_model.fit_predict(tfidf_matrix)

                # Create patterns from clusters
                cluster_dict = defaultdict(list)
                for idx, cluster_id in enumerate(clusters):
                    if cluster_id >= 0:  # Ignore noise points
                        cluster_dict[cluster_id].append(artifacts[idx])

                # Generate pattern from each cluster
                for cluster_id, cluster_artifacts in cluster_dict.items():
                    if len(cluster_artifacts) >= self.min_cluster_size:
                        pattern = self._create_pattern_from_cluster(cluster_artifacts, cluster_id)
                        if pattern:
                            patterns.append(pattern)

                logger.info(f"Mined {len(patterns)} patterns using {self.clustering_algorithm}")

            except Exception as e:
                logger.error(f"ML pattern mining failed: {e}")
                return self._mine_patterns_fallback(artifacts, max_patterns)

            finally:
                # SECURITY FIX: RM-1 - Cleanup ML objects to free memory
                # This prevents memory leaks from scikit-learn objects
                try:
                    if vectorizer is not None:
                        del vectorizer
                    if tfidf_matrix is not None:
                        del tfidf_matrix
                    if cluster_model is not None:
                        del cluster_model
                except (AttributeError, RuntimeError, TypeError):
                    # Handle cleanup errors during destruction (interpreter shutdown)
                    pass

        return patterns

    def _mine_patterns_fallback(
        self,
        artifacts: List[KnowledgeArtifact],
        max_patterns: int,
    ) -> List[SolutionPattern]:
        """Fallback pattern mining without ML (keyword-based)."""
        patterns = []

        try:
            # Group by tags
            tag_groups = defaultdict(list)
            for artifact in artifacts:
                for tag in artifact.metadata.tags:
                    tag_groups[tag].append(artifact)

            # Group by domain
            domain_groups = defaultdict(list)
            for artifact in artifacts:
                if artifact.metadata.domain:
                    domain_groups[artifact.metadata.domain].append(artifact)

            # Create patterns from groups
            for tag, tag_artifacts in list(tag_groups.items())[:max_patterns]:
                if len(tag_artifacts) >= self.min_cluster_size:
                    pattern = self._create_pattern_from_group(tag_artifacts, tag)
                    if pattern:
                        patterns.append(pattern)

            for domain, domain_artifacts in list(domain_groups.items())[:max_patterns]:
                if len(domain_artifacts) >= self.min_cluster_size:
                    pattern = self._create_pattern_from_group(domain_artifacts, f"Domain: {domain}")
                    if pattern:
                        patterns.append(pattern)

            logger.info(f"Mined {len(patterns)} patterns using fallback method")

        except Exception as e:
            logger.error(f"Fallback pattern mining failed: {e}")

        return patterns

    def _create_pattern_from_cluster(
        self,
        cluster_artifacts: List[KnowledgeArtifact],
        cluster_id: int,
    ) -> Optional[SolutionPattern]:
        """Create a consolidated pattern from a cluster of artifacts."""
        try:
            # DEEP COPY FIX: Deep copy cluster_artifacts to prevent external modification
            cluster_artifacts = copy.deepcopy(cluster_artifacts)

            # Extract common themes
            all_tags = []
            all_descriptions = []
            for artifact in cluster_artifacts:
                all_tags.extend(artifact.metadata.tags)
                all_descriptions.append(artifact.description)

            # Find most common tags
            tag_counts = defaultdict(int)
            for tag in all_tags:
                tag_counts[tag] += 1
            # PERFORMANCE FIX: Use heapq.nlargest for O(n log k) instead of O(n log n)
            import heapq
            top_tags = heapq.nlargest(5, tag_counts.items(), key=lambda x: x[1])

            # Consolidate content
            combined_content = "\n\n".join([
                f"Pattern {i+1}: {artifact.content[:200]}..."
                for i, artifact in enumerate(cluster_artifacts[:3])
            ])

            # Create pattern
            from ace_knowledge_artifacts import create_solution_pattern

            pattern = create_solution_pattern(
                title=f"Mined Pattern {cluster_id}",
                description=f"Pattern mined from {len(cluster_artifacts)} similar artifacts",
                content=combined_content,
                problem_category=top_tags[0][0] if top_tags else "general",
                domain=cluster_artifacts[0].metadata.domain,
                tags=[tag for tag, _ in top_tags],
            )

            # Add related artifacts
            pattern.related_artifacts = [a.metadata.artifact_id for a in cluster_artifacts]

            return pattern

        except Exception as e:
            logger.warning(f"Failed to create pattern from cluster: {e}")
            return None

    def _create_pattern_from_group(
        self,
        group_artifacts: List[KnowledgeArtifact],
        group_name: str,
    ) -> Optional[SolutionPattern]:
        """Create a pattern from a group of artifacts."""
        try:
            # DEEP COPY FIX: Deep copy group_artifacts to prevent external modification
            group_artifacts = copy.deepcopy(group_artifacts)

            # Use first artifact as base
            base = group_artifacts[0]

            # Consolidate
            from ace_knowledge_artifacts import create_solution_pattern

            pattern = create_solution_pattern(
                title=f"Pattern: {group_name}",
                description=f"Pattern from {len(group_artifacts)} artifacts",
                content=base.content,
                problem_category=group_name,
                domain=base.metadata.domain,
                tags=[group_name],
            )

            pattern.related_artifacts = [a.metadata.artifact_id for a in group_artifacts]

            return pattern

        except Exception as e:
            logger.warning(f"Failed to create pattern from group: {e}")
            return None

    def cleanup(self):
        """
        Release ML resources.

        Memory Management:
        - ML objects (TF-IDF vectorizer, clustering models) are typically cleaned up in finally blocks
        - This method is provided for explicit cleanup when needed
        - Large matrices are cleared immediately after use in _mine_patterns_with_ml()
        """
        # No persistent ML state to clean up - all ML objects are local to methods
        # and cleaned up in finally blocks
        pass


# ============================================================================
# TEAM PERFORMANCE TRACKER
# ============================================================================

class TeamPerformanceTracker:
    """
    Track and analyze team performance metrics.

    Maintains historical data on team effectiveness, preferred problem types,
    skill affinities, and collaboration patterns.

    Security Fixes:
    - TS-6: Thread-safe operations with locks
    - RM-2: History limit to prevent unbounded memory growth
    - CVE-1: File path validation for all file operations

    Memory Management:
    - max_history_per_team: Default 1000, adjust based on available memory
    - Each entry ~1-5 KB
    - 1000 entries = ~1-5 MB per team
    - Set to None for unlimited (not recommended in production)
    """

    def __init__(self, storage_path: Optional[str] = None, max_history_per_team: int = 1000):
        """
        Initialize the team performance tracker.

        Args:
            storage_path: Path to persist performance data
            max_history_per_team: Maximum history entries per team (resource limit)

        Raises:
            ValueError: If storage_path is invalid
        """
        # SECURITY FIX: CVE-1 - Validate storage_path if provided
        if storage_path is not None:
            try:
                self.storage_path = validate_file_path_safe(storage_path)
            except ValueError as e:
                raise ValueError(f"Invalid storage_path: {e}")
        else:
            self.storage_path = None

        # SECURITY FIX: RM-2 - Add max_history_per_team parameter
        self.max_history_per_team = max_history_per_team
        self.team_history: Dict[str, List[TeamPerformanceData]] = defaultdict(list)
        self.team_aggregates: Dict[str, TeamPerformanceData] = {}

        # SECURITY FIX: TS-6 - Add thread lock for all operations
        self._lock = threading.Lock()

        # Load existing data if path provided (TOCTOU fix: validate before use)
        if self.storage_path:
            try:
                self.load_from_file(self.storage_path)
            except (FileNotFoundError, json.JSONDecodeError, IOError):
                pass  # File doesn't exist yet, that's OK

    def record_workflow_performance(
        self,
        workflow_id: str,
        team_performances: List[TeamPerformanceData],
    ):
        """
        Record team performance from a workflow execution.

        SECURITY FIX: TS-6 - Thread-safe with lock
        SECURITY FIX: RM-2 - Enforce history limit
        BUG FIX #8: Fix history append atomicity

        Args:
            workflow_id: Workflow identifier
            team_performances: List of team performance data
        """
        # DEEP COPY FIX: Deep copy team_performances to prevent external modification
        team_performances = copy.deepcopy(team_performances)

        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            for perf_data in team_performances:
                team_id = perf_data.team_id

                # BUG FIX #8: Fix history append atomicity - calculate truncation first
                # Store in history
                self.team_history[team_id].append(perf_data)

                # SECURITY FIX: RM-2 - Limit history size (atomic operation)
                if self.max_history_per_team is not None and len(self.team_history[team_id]) > self.max_history_per_team:
                    # Calculate and apply truncation atomically
                    removed = len(self.team_history[team_id]) - self.max_history_per_team
                    self.team_history[team_id] = self.team_history[team_id][-self.max_history_per_team:]
                    logger.warning(f"Team {team_id}: Removed {removed} old entries (limit: {self.max_history_per_team})")

                # Update aggregate
                if team_id not in self.team_aggregates:
                    self.team_aggregates[team_id] = copy.deepcopy(perf_data)
                else:
                    self._update_aggregate(team_id, perf_data)

            logger.info(f"Recorded performance for {len(team_performances)} teams")

    def _update_aggregate(self, team_id: str, new_perf: TeamPerformanceData):
        """
        Update aggregate performance data for a team.

        NOTE: Caller must hold self._lock
        BUG FIX #7: Fix aggregate update atomicity using try-except with rollback
        """
        # DEEP COPY FIX: Deep copy new_perf to prevent external modification during update
        new_perf = copy.deepcopy(new_perf)

        current = self.team_aggregates[team_id]

        # BUG FIX #7: Atomic update with rollback on error
        try:
            # Save current state for potential rollback
            saved_total_tasks = current.total_tasks
            saved_successful_tasks = current.successful_tasks
            saved_failed_tasks = current.failed_tasks
            saved_avg_exec_time = current.avg_execution_time
            saved_avg_quality = current.avg_quality_score
            saved_preferred_types = list(current.preferred_problem_types)
            saved_skill_affinities = dict(current.skill_affinities)

            # Update totals
            current.total_tasks += new_perf.total_tasks
            current.successful_tasks += new_perf.successful_tasks
            current.failed_tasks += new_perf.failed_tasks

            # Update averages correctly (track total sum and count)
            n = len(self.team_history[team_id])

            # CRITICAL BUG FIX #6: Prevent division by zero on first entry
            # When this is the first aggregate update, current.total_tasks was just set from new_perf
            # We need to check if total_tasks was 0 BEFORE the update above
            if n == 1 or current.total_tasks == new_perf.total_tasks:
                # First entry - use new_perf values directly
                current.avg_execution_time = new_perf.avg_execution_time
                current.avg_quality_score = new_perf.avg_quality_score
            else:
                # Get the previous total sum from the previous average
                previous_total = current.avg_execution_time * (n - 1)
                new_total = previous_total + (new_perf.avg_execution_time * new_perf.total_tasks)
                current.avg_execution_time = new_total / current.total_tasks

                # Same for quality score
                previous_quality_total = current.avg_quality_score * (n - 1)
                new_quality_total = previous_quality_total + (new_perf.avg_quality_score * new_perf.total_tasks)
                current.avg_quality_score = new_quality_total / current.total_tasks

            # Update preferred types
            for ptype in new_perf.preferred_problem_types:
                if ptype not in current.preferred_problem_types:
                    current.preferred_problem_types.append(ptype)

            # BUG FIX #6: Fix NaN check in skill affinity
            for skill, affinity in new_perf.skill_affinities.items():
                if skill in current.skill_affinities:
                    existing = current.skill_affinities[skill]
                    # Check for None or NaN before averaging
                    if existing is not None and not (isinstance(existing, float) and (existing != existing)):
                        current.skill_affinities[skill] = (existing + affinity) / 2
                    else:
                        current.skill_affinities[skill] = affinity
                else:
                    current.skill_affinities[skill] = affinity

            # Update timestamp
            current.last_updated = datetime.utcnow()

        except Exception as e:
            # BUG FIX #7: Rollback on error
            logger.error(f"Error updating aggregate for team {team_id}, rolling back: {e}")
            current.total_tasks = saved_total_tasks
            current.successful_tasks = saved_successful_tasks
            current.failed_tasks = saved_failed_tasks
            current.avg_execution_time = saved_avg_exec_time
            current.avg_quality_score = saved_avg_quality
            current.preferred_problem_types = saved_preferred_types
            current.skill_affinities = saved_skill_affinities
            raise

    def get_team_summary(self, team_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for a team.

        SECURITY FIX: TS-6 - Thread-safe with lock
        BUG FIX #1: Fix lock released too early - copy data inside lock
        """
        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            if team_id not in self.team_aggregates:
                return None

            perf = self.team_aggregates[team_id]
            history = self.team_history.get(team_id, [])

            # BUG FIX #1: Copy data inside lock to prevent race conditions
            # Build the entire dict while holding the lock
            summary = {
                "team_id": team_id,
                "team_name": perf.team_name,
                "team_type": perf.team_type,
                "total_workflows": len(history),
                "total_tasks": perf.total_tasks,
                "successful_tasks": perf.successful_tasks,
                "failed_tasks": perf.failed_tasks,
                "success_rate": perf.calculate_success_rate(),
                "avg_execution_time": perf.avg_execution_time,
                "avg_quality_score": perf.avg_quality_score,
                "preferred_problem_types": list(perf.preferred_problem_types),
                "skill_affinities": dict(sorted(
                    perf.skill_affinities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),  # Top 10 skills
                "collaboration_effectiveness": perf.collaboration_effectiveness,
                "last_updated": perf.last_updated.isoformat(),
            }

        return summary

    def get_top_teams(
        self,
        team_type: Optional[str] = None,
        metric: str = "success_rate",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top performing teams.

        SECURITY FIX: TS-6 - Thread-safe with lock

        Args:
            team_type: Filter by team type (optional)
            metric: Metric to rank by ("success_rate", "quality_score", "execution_time")
            limit: Maximum teams to return

        Returns:
            List of team summaries ranked by metric
        """
        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            teams = []

            for team_id, perf in self.team_aggregates.items():
                if team_type and perf.team_type != team_type:
                    continue

                # Get summary without re-acquiring lock (we already hold it)
                if team_id not in self.team_aggregates:
                    continue

                perf_data = self.team_aggregates[team_id]
                history = self.team_history.get(team_id, [])

                summary = {
                    "team_id": team_id,
                    "team_name": perf_data.team_name,
                    "team_type": perf_data.team_type,
                    "total_workflows": len(history),
                    "total_tasks": perf_data.total_tasks,
                    "successful_tasks": perf_data.successful_tasks,
                    "failed_tasks": perf_data.failed_tasks,
                    "success_rate": perf_data.calculate_success_rate(),
                    "avg_execution_time": perf_data.avg_execution_time,
                    "avg_quality_score": perf_data.avg_quality_score,
                    "preferred_problem_types": perf_data.preferred_problem_types,
                    "skill_affinities": dict(sorted(
                        perf_data.skill_affinities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]),
                    "collaboration_effectiveness": perf_data.collaboration_effectiveness,
                    "last_updated": perf_data.last_updated.isoformat(),
                }
                teams.append(summary)

        # Sort by metric
        reverse_sort = metric != "execution_time"  # Lower is better for execution time
        teams.sort(key=lambda x: x.get(metric, 0), reverse=reverse_sort)

        return teams[:limit]

    def recommend_team_for_task(
        self,
        problem_type: str,
        required_skills: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Recommend the best team for a given task.

        SECURITY FIX: TS-6 - Thread-safe with lock

        Args:
            problem_type: Type of problem to solve
            required_skills: Required skills (optional)

        Returns:
            Team recommendation with rationale
        """
        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            if not self.team_aggregates:
                return None

            candidates = []

            for team_id, perf in self.team_aggregates.items():
                score = 0.0
                rationale = []

                # Check problem type affinity
                if problem_type in perf.preferred_problem_types:
                    score += 10
                    rationale.append(f"Team prefers {problem_type} problems")

                # Check skill affinity
                if required_skills:
                    skill_match_score = 0.0
                    for skill in required_skills:
                        if skill in perf.skill_affinities:
                            skill_match_score += perf.skill_affinities[skill] * 5
                    score += skill_match_score
                    if skill_match_score > 0:
                        rationale.append(f"Team has {len(required_skills)} required skills")

                # Check success rate
                success_rate = perf.calculate_success_rate()
                score += success_rate * 20
                rationale.append(f"Success rate: {success_rate:.1%}")

                # Check quality score
                score += perf.avg_quality_score * 10
                rationale.append(f"Quality score: {perf.avg_quality_score:.1f}")

                # Get summary without re-acquiring lock
                history = self.team_history.get(team_id, [])
                summary = {
                    "team_id": team_id,
                    "team_name": perf.team_name,
                    "team_type": perf.team_type,
                    "total_workflows": len(history),
                    "total_tasks": perf.total_tasks,
                    "successful_tasks": perf.successful_tasks,
                    "failed_tasks": perf.failed_tasks,
                    "success_rate": success_rate,
                    "avg_execution_time": perf.avg_execution_time,
                    "avg_quality_score": perf.avg_quality_score,
                    "preferred_problem_types": perf.preferred_problem_types,
                    "skill_affinities": dict(sorted(
                        perf.skill_affinities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]),
                    "collaboration_effectiveness": perf.collaboration_effectiveness,
                    "last_updated": perf.last_updated.isoformat(),
                }

                candidates.append({
                    "team_id": team_id,
                    "score": score,
                    "rationale": rationale,
                    "summary": summary,
                })

        if not candidates:
            return None

        # Return top candidate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[0]

        return {
            "team_id": top["team_id"],
            "team_name": top["summary"]["team_name"],
            "team_type": top["summary"]["team_type"],
            "recommendation_score": top["score"],
            "rationale": top["rationale"],
            "team_summary": top["summary"],
        }

    def save_to_file(self, filepath: str):
        """
        Save performance data to file.

        SECURITY FIX: TS-6 - Thread-safe with lock
        SECURITY FIX: CVE-1 - Validate file path
        """
        # SECURITY FIX: CVE-1 - Validate filepath
        try:
            filepath = validate_file_path_safe(filepath)
        except ValueError as e:
            logger.error(f"Invalid filepath for save: {e}")
            raise

        try:
            # SECURITY FIX: TS-6 - Synchronize access
            with self._lock:
                data = {
                    "team_aggregates": {
                        team_id: perf.to_dict()
                        for team_id, perf in self.team_aggregates.items()
                    },
                    "team_history": {
                        team_id: [perf.to_dict() for perf in history]
                        for team_id, history in self.team_history.items()
                    },
                }

            # SECURITY FIX: Use atomic save if available
            if SECURITY_UTILS_AVAILABLE:
                atomic_save_json_file(filepath, data)
            else:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)

            logger.info(f"Saved team performance data to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save team performance data: {e}")
            raise

    def load_from_file(self, filepath: str):
        """
        Load performance data from file.

        SECURITY FIX: TS-6 - Thread-safe with lock
        SECURITY FIX: CVE-1 - Validate file path
        """
        # SECURITY FIX: CVE-1 - Validate filepath
        try:
            filepath = validate_file_path_safe(filepath)
        except ValueError as e:
            logger.error(f"Invalid filepath for load: {e}")
            raise

        try:
            # SECURITY FIX: Use safe load if available
            if SECURITY_UTILS_AVAILABLE:
                data = safe_load_json_file(filepath)
            else:
                with open(filepath, "r") as f:
                    data = json.load(f)

            # SECURITY FIX: TS-6 - Synchronize access after loading
            with self._lock:
                # Load aggregates
                for team_id, perf_dict in data.get("team_aggregates", {}).items():
                    perf = TeamPerformanceData(
                        team_id=perf_dict["team_id"],
                        team_name=perf_dict["team_name"],
                        team_type=perf_dict["team_type"],
                        total_tasks=perf_dict["total_tasks"],
                        successful_tasks=perf_dict["successful_tasks"],
                        failed_tasks=perf_dict["failed_tasks"],
                        avg_execution_time=perf_dict["avg_execution_time"],
                        avg_quality_score=perf_dict["avg_quality_score"],
                        preferred_problem_types=perf_dict["preferred_problem_types"],
                        skill_affinities=perf_dict["skill_affinities"],
                        collaboration_effectiveness=perf_dict["collaboration_effectiveness"],
                        last_updated=datetime.fromisoformat(perf_dict["last_updated"]),
                    )
                    self.team_aggregates[team_id] = perf

                # Load history
                for team_id, history_list in data.get("team_history", {}).items():
                    for perf_dict in history_list:
                        perf = TeamPerformanceData(
                            team_id=perf_dict["team_id"],
                            team_name=perf_dict["team_name"],
                            team_type=perf_dict["team_type"],
                            total_tasks=perf_dict["total_tasks"],
                            successful_tasks=perf_dict["successful_tasks"],
                            failed_tasks=perf_dict["failed_tasks"],
                            avg_execution_time=perf_dict["avg_execution_time"],
                            avg_quality_score=perf_dict["avg_quality_score"],
                            preferred_problem_types=perf_dict["preferred_problem_types"],
                            skill_affinities=perf_dict["skill_affinities"],
                            collaboration_effectiveness=perf_dict["collaboration_effectiveness"],
                            last_updated=datetime.fromisoformat(perf_dict["last_updated"]),
                        )
                        self.team_history[team_id].append(perf)

            logger.info(f"Loaded team performance data from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load team performance data: {e}")
            raise

    def cleanup(self):
        """
        Release resources held by this object.

        SECURITY FIX: RM-3 - Proper resource cleanup
        """
        try:
            with self._lock:
                self.team_history.clear()
                self.team_aggregates.clear()
            logger.info("TeamPerformanceTracker resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """
        Destructor to ensure cleanup.

        SECURITY FIX: RM-3 - Ensure cleanup on destruction
        """
        try:
            self.cleanup()
        except (AttributeError, RuntimeError, OSError):
            # Handle cleanup errors during destruction (interpreter may be shutting down)
            pass

    def __enter__(self):
        """
        Context manager entry.

        SECURITY FIX: RM-3 - Support context manager protocol
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup.

        SECURITY FIX: RM-3 - Ensure cleanup on context exit
        """
        self.cleanup()
        return False


# ============================================================================
# GAUNTLET EFFECTIVENESS ANALYZER
# ============================================================================

class GauntletEffectivenessAnalyzer:
    """
    Analyze gauntlet effectiveness metrics.

    Tracks which gauntlets are most effective at finding issues,
    their precision, false positive rates, and optimal use cases.

    Security Fixes:
    - TS-6: Thread-safe operations with locks
    - RM-2: History limit to prevent unbounded memory growth
    - CVE-1: File path validation for all file operations

    Memory Management:
    - max_history_per_gauntlet: Default 1000, adjust based on available memory
    - Each entry ~1-3 KB
    - 1000 entries = ~1-3 MB per gauntlet
    - Set to None for unlimited (not recommended in production)
    """

    def __init__(self, storage_path: Optional[str] = None, max_history_per_gauntlet: int = 1000):
        """
        Initialize the gauntlet effectiveness analyzer.

        Args:
            storage_path: Path to persist effectiveness data
            max_history_per_gauntlet: Maximum history entries per gauntlet (resource limit)

        Raises:
            ValueError: If storage_path is invalid
        """
        # SECURITY FIX: CVE-1 - Validate storage_path if provided
        if storage_path is not None:
            try:
                self.storage_path = validate_file_path_safe(storage_path)
            except ValueError as e:
                raise ValueError(f"Invalid storage_path: {e}")
        else:
            self.storage_path = None

        # SECURITY FIX: RM-2 - Add max_history_per_gauntlet parameter
        self.max_history_per_gauntlet = max_history_per_gauntlet
        self.gauntlet_history: Dict[str, List[GauntletEffectivenessData]] = defaultdict(list)
        self.gauntlet_aggregates: Dict[str, GauntletEffectivenessData] = {}

        # SECURITY FIX: TS-6 - Add thread lock for all operations
        self._lock = threading.Lock()

        # Load existing data if path provided (TOCTOU fix: validate before use)
        if self.storage_path:
            try:
                self.load_from_file(self.storage_path)
            except (FileNotFoundError, json.JSONDecodeError, IOError):
                pass  # File doesn't exist yet, that's OK

    def record_gauntlet_run(
        self,
        workflow_id: str,
        gauntlet_effectiveness: List[GauntletEffectivenessData],
    ):
        """
        Record gauntlet effectiveness from a workflow execution.

        SECURITY FIX: TS-6 - Thread-safe with lock
        SECURITY FIX: RM-2 - Enforce history limit
        BUG FIX #8: Fix history append atomicity

        Args:
            workflow_id: Workflow identifier
            gauntlet_effectiveness: List of gauntlet effectiveness data
        """
        # DEEP COPY FIX: Deep copy gauntlet_effectiveness to prevent external modification
        gauntlet_effectiveness = copy.deepcopy(gauntlet_effectiveness)

        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            for ge_data in gauntlet_effectiveness:
                gauntlet_id = ge_data.gauntlet_id

                # BUG FIX #8: Fix history append atomicity - calculate truncation first
                # Store in history
                self.gauntlet_history[gauntlet_id].append(ge_data)

                # SECURITY FIX: RM-2 - Limit history size (atomic operation)
                if self.max_history_per_gauntlet is not None and len(self.gauntlet_history[gauntlet_id]) > self.max_history_per_gauntlet:
                    # Calculate and apply truncation atomically
                    removed = len(self.gauntlet_history[gauntlet_id]) - self.max_history_per_gauntlet
                    self.gauntlet_history[gauntlet_id] = self.gauntlet_history[gauntlet_id][-self.max_history_per_gauntlet:]
                    logger.warning(f"Gauntlet {gauntlet_id}: Removed {removed} old entries (limit: {self.max_history_per_gauntlet})")

                # Update aggregate
                if gauntlet_id not in self.gauntlet_aggregates:
                    self.gauntlet_aggregates[gauntlet_id] = copy.deepcopy(ge_data)
                else:
                    self._update_aggregate(gauntlet_id, ge_data)

            logger.info(f"Recorded effectiveness for {len(gauntlet_effectiveness)} gauntlets")

    def _update_aggregate(self, gauntlet_id: str, new_ge: GauntletEffectivenessData):
        """
        Update aggregate effectiveness data for a gauntlet.

        NOTE: Caller must hold self._lock
        BUG FIX #3: Fix wrong weighted average formula
        BUG FIX #7: Fix aggregate update atomicity with rollback
        """
        # DEEP COPY FIX: Deep copy new_ge to prevent external modification during update
        new_ge = copy.deepcopy(new_ge)

        current = self.gauntlet_aggregates[gauntlet_id]

        # BUG FIX #7: Atomic update with rollback on error
        try:
            # Save current state for potential rollback
            saved_total_runs = current.total_runs
            saved_issues_found = current.issues_found
            saved_false_positives = current.false_positives
            saved_true_positives = current.true_positives
            saved_avg_exec_time = current.avg_execution_time
            saved_effective_types = list(current.effective_problem_types)
            saved_violations = dict(current.common_violations)

            # Update totals
            current.total_runs += new_ge.total_runs
            current.issues_found += new_ge.issues_found
            current.false_positives += new_ge.false_positives
            current.true_positives += new_ge.true_positives

            # BUG FIX #3: Fix wrong weighted average formula for execution time
            if current.total_runs == 0:
                current.avg_execution_time = new_ge.avg_execution_time
            else:
                old_runs = current.total_runs - new_ge.total_runs
                previous_total = current.avg_execution_time * old_runs
                new_total = previous_total + (new_ge.avg_execution_time * new_ge.total_runs)
                current.avg_execution_time = new_total / current.total_runs

            # Update effective problem types
            for ptype in new_ge.effective_problem_types:
                if ptype not in current.effective_problem_types:
                    current.effective_problem_types.append(ptype)

            # Update common violations
            for violation, count in new_ge.common_violations.items():
                current.common_violations[violation] = (
                    current.common_violations.get(violation, 0) + count
                )

            # Recalculate rates
            current.detection_rate = current.calculate_detection_rate()

            # Update timestamp
            current.last_updated = datetime.utcnow()

        except Exception as e:
            # BUG FIX #7: Rollback on error
            logger.error(f"Error updating aggregate for gauntlet {gauntlet_id}, rolling back: {e}")
            current.total_runs = saved_total_runs
            current.issues_found = saved_issues_found
            current.false_positives = saved_false_positives
            current.true_positives = saved_true_positives
            current.avg_execution_time = saved_avg_exec_time
            current.effective_problem_types = saved_effective_types
            current.common_violations = saved_violations
            raise

    def get_gauntlet_summary(self, gauntlet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get effectiveness summary for a gauntlet.

        SECURITY FIX: TS-6 - Thread-safe with lock
        BUG FIX #1: Fix lock released too early - copy data inside lock
        """
        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            if gauntlet_id not in self.gauntlet_aggregates:
                return None

            ge = self.gauntlet_aggregates[gauntlet_id]
            history = self.gauntlet_history[gauntlet_id]

            # BUG FIX #1: Copy data inside lock to prevent race conditions
            # Build the entire dict while holding the lock
            summary = {
                "gauntlet_id": gauntlet_id,
                "gauntlet_name": ge.gauntlet_name,
                "gauntlet_type": ge.gauntlet_type,
                "total_runs": ge.total_runs,
                "total_workflows": len(history),
                "issues_found": ge.issues_found,
                "detection_rate": ge.detection_rate,
                "precision": ge.calculate_precision(),
                "avg_execution_time": ge.avg_execution_time,
                "effective_problem_types": list(ge.effective_problem_types),
                "common_violations": dict(sorted(
                    ge.common_violations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),  # Top 10 violations
                "last_updated": ge.last_updated.isoformat(),
            }

        return summary

    def get_most_effective_gauntlets(
        self,
        gauntlet_type: Optional[str] = None,
        metric: str = "detection_rate",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get most effective gauntlets.

        SECURITY FIX: TS-6 - Thread-safe with lock

        Args:
            gauntlet_type: Filter by gauntlet type (optional)
            metric: Metric to rank by ("detection_rate", "precision", "issues_found")
            limit: Maximum gauntlets to return

        Returns:
            List of gauntlet summaries ranked by metric
        """
        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            gauntlets = []

            for gauntlet_id, ge in self.gauntlet_aggregates.items():
                if gauntlet_type and ge.gauntlet_type != gauntlet_type:
                    continue

                summary = self.get_gauntlet_summary(gauntlet_id)
                if summary:
                    gauntlets.append(summary)

        # Sort by metric
        gauntlets.sort(key=lambda x: x.get(metric, 0), reverse=True)

        return gauntlets[:limit]

    def recommend_gauntlets_for_task(
        self,
        problem_type: str,
        gauntlet_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommend gauntlets for a given task.

        SECURITY FIX: TS-6 - Thread-safe with lock

        Args:
            problem_type: Type of problem to validate
            gauntlet_type: Filter by gauntlet type (optional)

        Returns:
            List of recommended gauntlets with rationale
        """
        # SECURITY FIX: TS-6 - Synchronize access
        with self._lock:
            recommendations = []

            for gauntlet_id, ge in self.gauntlet_aggregates.items():
                if gauntlet_type and ge.gauntlet_type != gauntlet_type:
                    continue

                # Check if effective for this problem type
                if problem_type in ge.effective_problem_types:
                    summary = self.get_gauntlet_summary(gauntlet_id)
                    if summary:
                        recommendations.append({
                            "gauntlet_id": gauntlet_id,
                            "gauntlet_name": ge.gauntlet_name,
                            "gauntlet_type": ge.gauntlet_type,
                            "recommendation_score": ge.detection_rate * 10 + ge.calculate_precision() * 5,
                            "rationale": [
                                f"Effective for {problem_type} problems",
                                f"Detection rate: {ge.detection_rate:.1%}",
                                f"Precision: {ge.calculate_precision():.1%}",
                            ],
                            "summary": summary,
                        })

        # Sort by recommendation score
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)

        return recommendations

    def save_to_file(self, filepath: str):
        """
        Save effectiveness data to file.

        SECURITY FIX: TS-6 - Thread-safe with lock
        SECURITY FIX: CVE-1 - Validate file path
        """
        # SECURITY FIX: CVE-1 - Validate filepath
        try:
            filepath = validate_file_path_safe(filepath)
        except ValueError as e:
            logger.error(f"Invalid filepath for save: {e}")
            raise

        try:
            # SECURITY FIX: TS-6 - Synchronize access
            with self._lock:
                data = {
                    "gauntlet_aggregates": {
                        gauntlet_id: ge.to_dict()
                        for gauntlet_id, ge in self.gauntlet_aggregates.items()
                    },
                    "gauntlet_history": {
                        gauntlet_id: [ge.to_dict() for ge in history]
                        for gauntlet_id, history in self.gauntlet_history.items()
                    },
                }

            # SECURITY FIX: Use atomic save if available
            if SECURITY_UTILS_AVAILABLE:
                atomic_save_json_file(filepath, data)
            else:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)

            logger.info(f"Saved gauntlet effectiveness data to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save gauntlet effectiveness data: {e}")
            raise

    def load_from_file(self, filepath: str):
        """
        Load effectiveness data from file.

        SECURITY FIX: TS-6 - Thread-safe with lock
        SECURITY FIX: CVE-1 - Validate file path
        """
        # SECURITY FIX: CVE-1 - Validate filepath
        try:
            filepath = validate_file_path_safe(filepath)
        except ValueError as e:
            logger.error(f"Invalid filepath for load: {e}")
            raise

        try:
            # SECURITY FIX: Use safe load if available
            if SECURITY_UTILS_AVAILABLE:
                data = safe_load_json_file(filepath)
            else:
                with open(filepath, "r") as f:
                    data = json.load(f)

            # SECURITY FIX: TS-6 - Synchronize access after loading
            with self._lock:
                # Load aggregates
                for gauntlet_id, ge_dict in data.get("gauntlet_aggregates", {}).items():
                    ge = GauntletEffectivenessData(
                        gauntlet_id=ge_dict["gauntlet_id"],
                        gauntlet_name=ge_dict["gauntlet_name"],
                        gauntlet_type=ge_dict["gauntlet_type"],
                        total_runs=ge_dict["total_runs"],
                        issues_found=ge_dict["issues_found"],
                        false_positives=ge_dict["false_positives"],
                        true_positives=ge_dict["true_positives"],
                        detection_rate=ge_dict["detection_rate"],
                        avg_execution_time=ge_dict["avg_execution_time"],
                        effective_problem_types=ge_dict["effective_problem_types"],
                        common_violations=ge_dict["common_violations"],
                        last_updated=datetime.fromisoformat(ge_dict["last_updated"]),
                    )
                    self.gauntlet_aggregates[gauntlet_id] = ge

                # Load history
                for gauntlet_id, history_list in data.get("gauntlet_history", {}).items():
                    for ge_dict in history_list:
                        ge = GauntletEffectivenessData(
                            gauntlet_id=ge_dict["gauntlet_id"],
                            gauntlet_name=ge_dict["gauntlet_name"],
                            gauntlet_type=ge_dict["gauntlet_type"],
                            total_runs=ge_dict["total_runs"],
                            issues_found=ge_dict["issues_found"],
                            false_positives=ge_dict["false_positives"],
                            true_positives=ge_dict["true_positives"],
                            detection_rate=ge_dict["detection_rate"],
                            avg_execution_time=ge_dict["avg_execution_time"],
                            effective_problem_types=ge_dict["effective_problem_types"],
                            common_violations=ge_dict["common_violations"],
                            last_updated=datetime.fromisoformat(ge_dict["last_updated"]),
                        )
                        self.gauntlet_history[gauntlet_id].append(ge)

            logger.info(f"Loaded gauntlet effectiveness data from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load gauntlet effectiveness data: {e}")
            raise

    def cleanup(self):
        """
        Release resources held by this object.

        SECURITY FIX: RM-3 - Proper resource cleanup
        """
        try:
            with self._lock:
                self.gauntlet_history.clear()
                self.gauntlet_aggregates.clear()
            logger.info("GauntletEffectivenessAnalyzer resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """
        Destructor to ensure cleanup.

        SECURITY FIX: RM-3 - Ensure cleanup on destruction
        """
        try:
            self.cleanup()
        except (AttributeError, RuntimeError, OSError):
            # Handle cleanup errors during destruction (interpreter may be shutting down)
            pass

    def __enter__(self):
        """
        Context manager entry.

        SECURITY FIX: RM-3 - Support context manager protocol
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with cleanup.

        SECURITY FIX: RM-3 - Ensure cleanup on context exit
        """
        self.cleanup()
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SolutionPatternMiner",
    "TeamPerformanceTracker",
    "GauntletEffectivenessAnalyzer",
]
