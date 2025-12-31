"""
ACE Stage 6 Knowledge Extraction - MCP Tools

This module provides Model Context Protocol (MCP) tools for Stage 6:
Knowledge Extraction & Learning. These tools integrate the new components
(WORKflowKnowledgeExtractor, SolutionPatternMiner, TeamPerformanceTracker,
GauntletEffectivenessAnalyzer) with the existing ACE infrastructure.

MCP Tools:
- extract_knowledge_from_workflow: Extract artifacts from workflow
- mine_solution_patterns: Mine patterns using ML
- track_team_performance: Record and analyze team performance
- analyze_gauntlet_effectiveness: Track gauntlet metrics
- recommend_team_for_task: Recommend best team for a task
- recommend_gauntlets_for_task: Recommend gauntlets for validation
- get_knowledge_statistics: Get statistics about extracted knowledge

SECURITY HARDENING:
- All inputs validated using ace_security_utils
- Thread-safe MCP tools registry
- Safe error handling without information disclosure
- Path traversal protection
- Resource exhaustion protection
"""

from typing import Any, Dict, List, Optional
import sys
import os
import json
import logging
from datetime import datetime
import threading
import copy
import numpy as np

# NUMPY SERIALIZATION FIX: Add custom JSON encoder for numpy arrays
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

# ============================================================================
# SECURITY FIX #1: Import Security Utilities
# ============================================================================
try:
    from ace_security_utils import (
        validate_string_length,
        validate_list_size,
        validate_numeric_range,
        validate_dict_structure,
        validate_file_path_safe,
        create_safe_error,
        get_global_lock,
        synchronized,
        DEFAULT_SKILLBOOK_DIR,
        DEFAULT_ANALYTICS_DIR,
    )
    SECURITY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Security utilities not available: {e}")
    SECURITY_AVAILABLE = False

    # Fallback implementations (if security utils unavailable)
    def validate_string_length(value, name, max_length=100000, min_length=0, allow_empty=True):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        if not allow_empty and len(value) == 0:
            raise ValueError(f"{name} cannot be empty")
        return value

    def validate_list_size(items, name, max_size=10000, min_size=0, allow_empty=True):
        if not isinstance(items, list):
            raise ValueError(f"{name} must be a list")
        return items

    def validate_numeric_range(value, name, min_val=None, max_val=None, **kwargs):
        return value

    def validate_dict_structure(data, expected_fields, **kwargs):
        return data if isinstance(data, dict) else {}

    def validate_file_path_safe(filepath, base_dir="."):
        return filepath

    def create_safe_error(user_message, internal_error, include_details=False):
        return {"success": False, "error": user_message}

    def get_global_lock(name):
        return threading.RLock()

    def synchronized(lock):
        def decorator(func):
            return func
        return decorator

    DEFAULT_SKILLBOOK_DIR = "./ace_skillbooks"
    DEFAULT_ANALYTICS_DIR = "./ace_analytics"

# Add agentic-context-engine to path
ACE_PATH = os.path.join(os.path.dirname(__file__), "agentic-context-engine")
if os.path.exists(ACE_PATH) and ACE_PATH not in sys.path:
    sys.path.insert(0, ACE_PATH)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY FIX #2: Thread-Safe MCP Tools Registry
# ============================================================================

_MCP_TOOLS: Dict[str, Any] = {}
_MCP_TOOLS_LOCK = get_global_lock("ace_stage6_mcp_tools_registry")

def mcp_tool(name: str):
    """
    Decorator to register MCP tools (thread-safe).

    SECURITY FIX: Thread-safe registration prevents race conditions
    """
    def decorator(func):
        with _MCP_TOOLS_LOCK:
            _MCP_TOOLS[name] = func
        return func
    return decorator


def clear_stage6_mcp_tools() -> int:
    """
    SECURITY FIX: Clear all registered Stage 6 MCP tools (thread-safe).

    This should be called when you want to free up memory
    by clearing the global MCP tool registry.

    Returns:
        int: Number of tools that were cleared
    """
    global _MCP_TOOLS
    with _MCP_TOOLS_LOCK:
        count = len(_MCP_TOOLS)
        _MCP_TOOLS.clear()
        logger.info(f"Cleared {count} Stage 6 MCP tools from global registry")
        return count

# Import components
try:
    from ace_workflow_knowledge_extractor import (
        WorkflowKnowledgeExtractor,
        extract_knowledge_from_workflow,
    )
    from ace_analytics import (
        SolutionPatternMiner,
        TeamPerformanceTracker,
        GauntletEffectivenessAnalyzer,
    )
    from ace_knowledge_artifacts import (
        KnowledgeArtifact,
        SolutionPattern,
        TeamPerformanceData,
        GauntletEffectivenessData,
        WorkflowExtractionResult,
        ArtifactType,
    )
    ACE_STAGE6_AVAILABLE = True
except ImportError as e:
    ACE_STAGE6_AVAILABLE = False
    logging.warning(f"ACE Stage 6 components not available: {e}")


# ============================================================================
# MCP TOOL 1: Extract Knowledge from Workflow
# ============================================================================

@mcp_tool("extract_knowledge_from_workflow")
def extract_knowledge_from_workflow_tool(
    workflow_id: str,
    problem_statement: str,
    workflow_results: Dict[str, Any],
    model: str = "gpt-4o-mini",
    skillbook_path: Optional[str] = None,
    output_file: Optional[str] = None,
    extract_team_metrics: bool = True,
    extract_gauntlet_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Extract knowledge artifacts from a complete workflow execution.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        workflow_id: Unique identifier for the workflow
        problem_statement: The original problem statement
        workflow_results: Complete results from all workflow stages
        model: LiteLLM model name for ACE
        skillbook_path: Path to load existing skillbook
        output_file: Optional file to save extraction results
        extract_team_metrics: Extract team performance data
        extract_gauntlet_metrics: Extract gauntlet effectiveness data

    Returns:
        Dict with extraction results and summary
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        # DEEP COPY FIX: Deep copy all inputs to prevent external modification
        workflow_id = validate_string_length(
            copy.deepcopy(workflow_id), "workflow_id",
            max_length=200, min_length=1, allow_empty=False
        )

        problem_statement = validate_string_length(
            copy.deepcopy(problem_statement), "problem_statement",
            max_length=50000, min_length=10, allow_empty=False
        )

        # Validate workflow_results is a dict with expected structure
        # DEEP COPY FIX: Deep copy workflow_results to prevent external modification
        workflow_results = copy.deepcopy(workflow_results)
        expected_fields = {
            "stages": dict,
            "timestamp": str,
        }
        workflow_results = validate_dict_structure(
            workflow_results, expected_fields,
            allow_extra=True, require_all=False
        )

        # Validate file paths if provided
        if skillbook_path:
            skillbook_path = validate_file_path_safe(
                skillbook_path,
                base_dir=DEFAULT_SKILLBOOK_DIR
            )

        if output_file:
            output_file = validate_file_path_safe(
                output_file,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Extract knowledge
        result = extract_knowledge_from_workflow(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
            workflow_results=workflow_results,
            model=model,
            skillbook_path=skillbook_path,
            output_file=output_file,
        )

        return {
            "success": True,
            "available": True,
            "workflow_id": workflow_id,
            "total_artifacts": result.total_artifacts,
            "artifacts": [artifact.to_dict() for artifact in result.extracted_artifacts],
            "team_performances": [tp.to_dict() for tp in result.team_performances],
            "gauntlet_effectiveness": [ge.to_dict() for ge in result.gauntlet_effectiveness],
            "summary": result.to_summary(),
            "message": f"Extracted {result.total_artifacts} artifacts from workflow",
        }

    except Exception as e:
        logger.error(f"Failed to extract knowledge: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to extract knowledge from workflow",
            e
        )


# ============================================================================
# MCP TOOL 2: Mine Solution Patterns
# ============================================================================

@mcp_tool("mine_solution_patterns")
def mine_solution_patterns_tool(
    artifacts: List[Dict[str, Any]],
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.7,
    clustering_algorithm: str = "kmeans",
    max_patterns: int = 10,
) -> Dict[str, Any]:
    """
    Mine solution patterns from artifacts using ML clustering.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        artifacts: List of knowledge artifacts (as dicts)
        min_cluster_size: Minimum artifacts to form a cluster
        similarity_threshold: Minimum similarity for clustering (0-1)
        clustering_algorithm: Algorithm ("kmeans" or "dbscan")
        max_patterns: Maximum number of patterns to extract

    Returns:
        Dict with mined solution patterns
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        # DEEP COPY FIX: Deep copy artifacts list to prevent external modification
        artifacts = copy.deepcopy(artifacts)
        artifacts = validate_list_size(
            artifacts, "artifacts",
            max_size=1000, min_size=1, allow_empty=False
        )

        min_cluster_size = validate_numeric_range(
            min_cluster_size, "min_cluster_size",
            min_val=2, max_val=1000, value_type=int
        )

        similarity_threshold = validate_numeric_range(
            similarity_threshold, "similarity_threshold",
            min_val=0.0, max_val=1.0, value_type=float
        )

        # Validate clustering algorithm
        valid_algorithms = ["kmeans", "dbscan", "hierarchical"]
        if clustering_algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid clustering_algorithm: {clustering_algorithm}. "
                f"Must be one of: {valid_algorithms}"
            )

        max_patterns = validate_numeric_range(
            max_patterns, "max_patterns",
            min_val=1, max_val=1000, value_type=int
        )

        # Convert dicts to artifacts
        artifact_objects = []
        for artifact_dict in artifacts:
            if artifact_dict is None:
                logger.warning("Skipping None artifact_dict")
                continue
            if not isinstance(artifact_dict, dict):
                logger.warning(f"Skipping non-dict artifact: {type(artifact_dict)}")
                continue

            try:
                artifact = KnowledgeArtifact.from_dict(artifact_dict)
                # ARTIFACTS OBJECT DEEP COPY FIX: Deep copy artifact after creation
                # to prevent external modification through shared references
                artifact = copy.deepcopy(artifact)
                artifact_objects.append(artifact)
            except Exception as e:
                logger.warning(f"Failed to parse artifact: {e}")

        # Create pattern miner
        miner = SolutionPatternMiner(
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
            clustering_algorithm=clustering_algorithm,
        )

        # Mine patterns
        patterns = miner.mine_patterns_from_artifacts(
            artifact_objects,
            max_patterns=max_patterns,
        )

        return {
            "success": True,
            "available": True,
            "patterns_found": len(patterns),
            "patterns": [pattern.to_dict() for pattern in patterns],
            "clustering_algorithm": clustering_algorithm,
            "message": f"Mined {len(patterns)} solution patterns",
        }

    except Exception as e:
        logger.error(f"Failed to mine patterns: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to mine solution patterns",
            e
        )


# ============================================================================
# MCP TOOL 3: Track Team Performance
# ============================================================================

@mcp_tool("track_team_performance")
def track_team_performance_tool(
    workflow_id: str,
    team_performances: List[Dict[str, Any]],
    storage_path: Optional[str] = "./team_performance.json",
) -> Dict[str, Any]:
    """
    Record team performance from a workflow execution.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        workflow_id: Workflow identifier
        team_performances: List of team performance data (as dicts)
        storage_path: Path to persist performance data

    Returns:
        Dict with recorded performance data
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        # DEEP COPY FIX: Deep copy all inputs to prevent external modification
        workflow_id = validate_string_length(
            copy.deepcopy(workflow_id), "workflow_id",
            max_length=200, min_length=1, allow_empty=False
        )

        # DEEP COPY FIX: Deep copy team_performances list to prevent external modification
        team_performances = copy.deepcopy(team_performances)
        team_performances = validate_list_size(
            team_performances, "team_performances",
            max_size=100, min_size=1, allow_empty=False
        )

        # Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create tracker
        tracker = TeamPerformanceTracker(storage_path=storage_path)

        # Convert dicts to performance data
        perf_objects = []
        for perf_dict in team_performances:
            # TEAM PERF DICT VALIDATION FIX: Validate dict structure before access
            if not isinstance(perf_dict, dict):
                logger.warning(f"Skipping non-dict performance data: {type(perf_dict)}")
                continue

            if "team_id" not in perf_dict:
                logger.warning("Skipping performance data without team_id")
                continue

            try:
                perf = TeamPerformanceData(
                    team_id=perf_dict["team_id"],
                    team_name=perf_dict.get("team_name", ""),
                    team_type=perf_dict.get("team_type", "blue_team"),
                    total_tasks=perf_dict.get("total_tasks", 0),
                    successful_tasks=perf_dict.get("successful_tasks", 0),
                    failed_tasks=perf_dict.get("failed_tasks", 0),
                    avg_execution_time=perf_dict.get("avg_execution_time", 0.0),
                    avg_quality_score=perf_dict.get("avg_quality_score", 0.0),
                    preferred_problem_types=perf_dict.get("preferred_problem_types", []),
                    skill_affinities=perf_dict.get("skill_affinities", {}),
                    collaboration_effectiveness=perf_dict.get("collaboration_effectiveness", 0.0),
                )
                perf_objects.append(perf)
            except Exception as e:
                logger.warning(f"Failed to create TeamPerformanceData: {e}")

        # Record performance
        tracker.record_workflow_performance(workflow_id, perf_objects)

        # Save to file
        if storage_path:
            tracker.save_to_file(storage_path)

        return {
            "success": True,
            "available": True,
            "workflow_id": workflow_id,
            "teams_recorded": len(perf_objects),
            "team_summaries": {
                perf.team_id: tracker.get_team_summary(perf.team_id)
                for perf in perf_objects
            },
            "message": f"Recorded performance for {len(perf_objects)} teams",
        }

    except Exception as e:
        logger.error(f"Failed to track team performance: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to track team performance",
            e
        )


# ============================================================================
# MCP TOOL 4: Analyze Gauntlet Effectiveness
# ============================================================================

@mcp_tool("analyze_gauntlet_effectiveness")
def analyze_gauntlet_effectiveness_tool(
    workflow_id: str,
    gauntlet_effectiveness: List[Dict[str, Any]],
    storage_path: Optional[str] = "./gauntlet_effectiveness.json",
) -> Dict[str, Any]:
    """
    Record gauntlet effectiveness from a workflow execution.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        workflow_id: Workflow identifier
        gauntlet_effectiveness: List of gauntlet effectiveness data (as dicts)
        storage_path: Path to persist effectiveness data

    Returns:
        Dict with recorded effectiveness data
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        # DEEP COPY FIX: Deep copy all inputs to prevent external modification
        workflow_id = validate_string_length(
            copy.deepcopy(workflow_id), "workflow_id",
            max_length=200, min_length=1, allow_empty=False
        )

        # DEEP COPY FIX: Deep copy gauntlet_effectiveness list to prevent external modification
        gauntlet_effectiveness = copy.deepcopy(gauntlet_effectiveness)
        gauntlet_effectiveness = validate_list_size(
            gauntlet_effectiveness, "gauntlet_effectiveness",
            max_size=100, min_size=1, allow_empty=False
        )

        # Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create analyzer
        analyzer = GauntletEffectivenessAnalyzer(storage_path=storage_path)

        # Convert dicts to effectiveness data
        ge_objects = []
        for ge_dict in gauntlet_effectiveness:
            # GAUNTLET DATA VALIDATION FIX: Validate dict structure before access
            if not isinstance(ge_dict, dict):
                logger.warning(f"Skipping non-dict gauntlet effectiveness data: {type(ge_dict)}")
                continue

            if "gauntlet_id" not in ge_dict:
                logger.warning("Skipping gauntlet effectiveness data without gauntlet_id")
                continue

            try:
                ge = GauntletEffectivenessData(
                    gauntlet_id=ge_dict["gauntlet_id"],
                    gauntlet_name=ge_dict.get("gauntlet_name", ""),
                    gauntlet_type=ge_dict.get("gauntlet_type", "red_team"),
                    total_runs=ge_dict.get("total_runs", 0),
                    issues_found=ge_dict.get("issues_found", 0),
                    false_positives=ge_dict.get("false_positives", 0),
                    true_positives=ge_dict.get("true_positives", 0),
                    detection_rate=ge_dict.get("detection_rate", 0.0),
                    avg_execution_time=ge_dict.get("avg_execution_time", 0.0),
                    effective_problem_types=ge_dict.get("effective_problem_types", []),
                    common_violations=ge_dict.get("common_violations", {}),
                )
                ge_objects.append(ge)
            except Exception as e:
                logger.warning(f"Failed to create GauntletEffectivenessData: {e}")

        # Record effectiveness
        analyzer.record_gauntlet_run(workflow_id, ge_objects)

        # Save to file
        if storage_path:
            analyzer.save_to_file(storage_path)

        return {
            "success": True,
            "available": True,
            "workflow_id": workflow_id,
            "gauntlets_recorded": len(ge_objects),
            "gauntlet_summaries": {
                ge.gauntlet_id: analyzer.get_gauntlet_summary(ge.gauntlet_id)
                for ge in ge_objects
            },
            "message": f"Recorded effectiveness for {len(ge_objects)} gauntlets",
        }

    except Exception as e:
        logger.error(f"Failed to analyze gauntlet effectiveness: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to analyze gauntlet effectiveness",
            e
        )


# ============================================================================
# MCP TOOL 5: Recommend Team for Task
# ============================================================================

@mcp_tool("recommend_team_for_task")
def recommend_team_for_task_tool(
    problem_type: str,
    required_skills: Optional[List[str]] = None,
    team_type: Optional[str] = None,
    storage_path: Optional[str] = "./team_performance.json",
) -> Dict[str, Any]:
    """
    Recommend the best team for a given task.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        problem_type: Type of problem to solve
        required_skills: Required skills (optional)
        team_type: Filter by team type (optional)
        storage_path: Path to performance data file

    Returns:
        Dict with team recommendation
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        # DEEP COPY FIX: Deep copy all inputs to prevent external modification
        problem_type = validate_string_length(
            copy.deepcopy(problem_type), "problem_type",
            max_length=200, min_length=1, allow_empty=False
        )

        if required_skills is not None:
            # DEEP COPY FIX: Deep copy required_skills list to prevent external modification
            required_skills = copy.deepcopy(required_skills)
            required_skills = validate_list_size(
                required_skills, "required_skills",
                max_size=50, min_size=0, allow_empty=True
            )

        if team_type is not None:
            team_type = validate_string_length(
                copy.deepcopy(team_type), "team_type",
                max_length=50, min_length=1, allow_empty=False
            )

        # Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create tracker and load data
        tracker = TeamPerformanceTracker(storage_path=storage_path)

        # Get recommendation
        recommendation = tracker.recommend_team_for_task(
            problem_type=problem_type,
            required_skills=required_skills or [],
        )

        if not recommendation:
            return {
                "success": True,
                "available": True,
                "recommendation": None,
                "message": "No suitable team found",
            }

        # Filter by team type if specified
        if team_type and recommendation["team_type"] != team_type:
            # Get next best team of specified type
            top_teams = tracker.get_top_teams(
                team_type=team_type,
                limit=5,
            )
            # INDEXERROR FIX: Check for empty list before accessing [0]
            if not top_teams:
                return {
                    "success": False,
                    "available": True,
                    "recommendation": None,
                    "message": f"No suitable team found for task: {problem_type}",
                }
            recommendation_score = top_teams[0].get("success_rate", 0) * 20
            recommendation = {
                "team_id": top_teams[0]["team_id"],
                "team_name": top_teams[0]["team_name"],
                "team_type": top_teams[0]["team_type"],
                "recommendation_score": recommendation_score,
                "rationale": [f"Best {team_type} team for this task"],
                "team_summary": top_teams[0],
            }

        return {
            "success": True,
            "available": True,
            "recommendation": recommendation,
            "message": f"Recommended team: {recommendation['team_name']}",
        }

    except Exception as e:
        logger.error(f"Failed to recommend team: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to recommend team for task",
            e
        )


# ============================================================================
# MCP TOOL 6: Recommend Gauntlets for Task
# ============================================================================

@mcp_tool("recommend_gauntlets_for_task")
def recommend_gauntlets_for_task_tool(
    problem_type: str,
    gauntlet_type: Optional[str] = None,
    limit: int = 5,
    storage_path: Optional[str] = "./gauntlet_effectiveness.json",
) -> Dict[str, Any]:
    """
    Recommend gauntlets for validating a given task.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        problem_type: Type of problem to validate
        gauntlet_type: Filter by gauntlet type (optional)
        limit: Maximum gauntlets to recommend
        storage_path: Path to effectiveness data file

    Returns:
        Dict with gauntlet recommendations
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        problem_type = validate_string_length(
            problem_type, "problem_type",
            max_length=200, min_length=1, allow_empty=False
        )

        if gauntlet_type is not None:
            gauntlet_type = validate_string_length(
                gauntlet_type, "gauntlet_type",
                max_length=50, min_length=1, allow_empty=False
            )

        limit = validate_numeric_range(
            limit, "limit",
            min_val=1, max_val=100, value_type=int
        )

        # LIMIT VALIDATION FIX: Ensure limit is valid (double-check)
        if limit < 1:
            logger.warning(f"Invalid limit {limit}, using 1")
            limit = 1

        # Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create analyzer and load data
        analyzer = GauntletEffectivenessAnalyzer(storage_path=storage_path)

        # Get recommendations
        recommendations = analyzer.recommend_gauntlets_for_task(
            problem_type=problem_type,
            gauntlet_type=gauntlet_type,
        )

        # Limit results
        recommendations = recommendations[:limit]

        return {
            "success": True,
            "available": True,
            "problem_type": problem_type,
            "gauntlet_type": gauntlet_type,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations,
            "message": f"Found {len(recommendations)} recommended gauntlets",
        }

    except Exception as e:
        logger.error(f"Failed to recommend gauntlets: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to recommend gauntlets for task",
            e
        )


# ============================================================================
# MCP TOOL 7: Get Knowledge Statistics
# ============================================================================

@mcp_tool("get_knowledge_statistics")
def get_knowledge_statistics_tool(
    storage_path: Optional[str] = "./team_performance.json",
) -> Dict[str, Any]:
    """
    Get statistics about extracted knowledge and performance data.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        storage_path: Path to performance data file

    Returns:
        Dict with knowledge statistics
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create tracker and analyzer
        tracker = TeamPerformanceTracker(storage_path=storage_path)
        analyzer = GauntletEffectivenessAnalyzer(storage_path=storage_path)

        # Get statistics
        stats = {
            "team_performance": {
                "total_teams": len(tracker.team_aggregates),
                "total_workflows_tracked": sum(
                    len(history) for history in tracker.team_history.values()
                ),
                "top_teams": tracker.get_top_teams(limit=5),
            },
            "gauntlet_effectiveness": {
                "total_gauntlets": len(analyzer.gauntlet_aggregates),
                "total_runs_tracked": sum(
                    ge.total_runs for ge in analyzer.gauntlet_aggregates.values()
                ),
                "top_gauntlets": analyzer.get_most_effective_gauntlets(limit=5),
            },
        }

        return {
            "success": True,
            "available": True,
            "statistics": stats,
            "message": "Knowledge statistics retrieved successfully",
        }

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to get knowledge statistics",
            e
        )


# ============================================================================
# MCP TOOL 8: Get Top Teams
# ============================================================================

@mcp_tool("get_top_teams")
def get_top_teams_tool(
    team_type: Optional[str] = None,
    metric: str = "success_rate",
    limit: int = 10,
    storage_path: Optional[str] = "./team_performance.json",
) -> Dict[str, Any]:
    """
    Get top performing teams.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        team_type: Filter by team type (optional)
        metric: Metric to rank by ("success_rate", "quality_score", "execution_time")
        limit: Maximum teams to return
        storage_path: Path to performance data file

    Returns:
        Dict with top teams
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        if team_type is not None:
            team_type = validate_string_length(
                team_type, "team_type",
                max_length=50, min_length=1, allow_empty=False
            )

        # Validate metric
        valid_metrics = ["success_rate", "quality_score", "execution_time"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: {valid_metrics}"
            )

        limit = validate_numeric_range(
            limit, "limit",
            min_val=1, max_val=100, value_type=int
        )

        # Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create tracker and load data
        tracker = TeamPerformanceTracker(storage_path=storage_path)

        # Get top teams
        top_teams = tracker.get_top_teams(
            team_type=team_type,
            metric=metric,
            limit=limit,
        )

        return {
            "success": True,
            "available": True,
            "team_type": team_type,
            "metric": metric,
            "teams_count": len(top_teams),
            "top_teams": top_teams,
            "message": f"Retrieved {len(top_teams)} top teams",
        }

    except Exception as e:
        logger.error(f"Failed to get top teams: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to get top teams",
            e
        )


# ============================================================================
# MCP TOOL 9: Get Most Effective Gauntlets
# ============================================================================

@mcp_tool("get_most_effective_gauntlets")
def get_most_effective_gauntlets_tool(
    gauntlet_type: Optional[str] = None,
    metric: str = "detection_rate",
    limit: int = 10,
    storage_path: Optional[str] = "./gauntlet_effectiveness.json",
) -> Dict[str, Any]:
    """
    Get most effective gauntlets.

    SECURITY FIX: All inputs validated, safe error handling

    Args:
        gauntlet_type: Filter by gauntlet type (optional)
        metric: Metric to rank by ("detection_rate", "precision", "issues_found")
        limit: Maximum gauntlets to return
        storage_path: Path to effectiveness data file

    Returns:
        Dict with most effective gauntlets
    """
    if not ACE_STAGE6_AVAILABLE:
        return {
            "success": False,
            "available": False,
            "error": "ACE Stage 6 components not available",
        }

    try:
        # SECURITY FIX: Validate all inputs
        if gauntlet_type is not None:
            gauntlet_type = validate_string_length(
                gauntlet_type, "gauntlet_type",
                max_length=50, min_length=1, allow_empty=False
            )

        # Validate metric
        valid_metrics = ["detection_rate", "precision", "issues_found"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: {valid_metrics}"
            )

        limit = validate_numeric_range(
            limit, "limit",
            min_val=1, max_val=100, value_type=int
        )

        # Validate storage_path if provided
        if storage_path:
            storage_path = validate_file_path_safe(
                storage_path,
                base_dir=DEFAULT_ANALYTICS_DIR
            )

        # Create analyzer and load data
        analyzer = GauntletEffectivenessAnalyzer(storage_path=storage_path)

        # Get top gauntlets
        top_gauntlets = analyzer.get_most_effective_gauntlets(
            gauntlet_type=gauntlet_type,
            metric=metric,
            limit=limit,
        )

        return {
            "success": True,
            "available": True,
            "gauntlet_type": gauntlet_type,
            "metric": metric,
            "gauntlets_count": len(top_gauntlets),
            "top_gauntlets": top_gauntlets,
            "message": f"Retrieved {len(top_gauntlets)} most effective gauntlets",
        }

    except Exception as e:
        logger.error(f"Failed to get most effective gauntlets: {e}")
        # SECURITY FIX: Use safe error handling
        return create_safe_error(
            "Failed to get most effective gauntlets",
            e
        )


# ============================================================================
# MCP Tool Registry Access
# ============================================================================

def get_registered_tools() -> Dict[str, Any]:
    """Get all registered Stage 6 MCP tools (thread-safe)."""
    # THREAD SAFETY FIX: TS-1 - Synchronize registry access
    with _MCP_TOOLS_LOCK:
        return _MCP_TOOLS.copy()

def list_mcp_tools() -> List[str]:
    """List names of all registered Stage 6 MCP tools (thread-safe)."""
    # THREAD SAFETY FIX: TS-1 - Synchronize registry access
    with _MCP_TOOLS_LOCK:
        return list(_MCP_TOOLS.keys())

# Export all MCP tools
__all__ = [
    # MCP Tools
    "extract_knowledge_from_workflow_tool",
    "mine_solution_patterns_tool",
    "track_team_performance_tool",
    "analyze_gauntlet_effectiveness_tool",
    "recommend_team_for_task_tool",
    "recommend_gauntlets_for_task_tool",
    "get_knowledge_statistics_tool",
    "get_top_teams_tool",
    "get_most_effective_gauntlets_tool",
    # Utilities
    "get_registered_tools",
    "list_mcp_tools",
    "clear_stage6_mcp_tools",
    "ACE_STAGE6_AVAILABLE",
]

# Module initialization
if __name__ == "__main__":
    print("ACE Stage 6 Knowledge Extraction MCP Tools")
    print(f"ACE Stage 6 Available: {ACE_STAGE6_AVAILABLE}")
    print(f"Registered Tools: {len(_MCP_TOOLS)}")
    print("\nTools:")
    for tool_name in sorted(_MCP_TOOLS.keys()):
        print(f"  - {tool_name}")
