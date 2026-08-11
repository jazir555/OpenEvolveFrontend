"""
Integration of Knowledge Extraction and Performance Tracking with Decomposition Engine

This module provides integration hooks to incorporate continuous learning capabilities
into the decomposition engine.
"""

import time
import logging
from typing import Dict, List, Any, Optional

from knowledge_artifact_extractor import KnowledgeArtifactExtractor
from performance_metrics_tracker import PerformanceMetricsTracker

logger = logging.getLogger(__name__)


def integrate_with_decomposition(
    problem,
    strategy: str = "semantic",
    extract_knowledge: bool = True,
    track_performance: bool = True,
    artifact_store_path: str = "knowledge_artifacts.json",
    metrics_store_path: str = "performance_metrics.json"
):
    """
    Integration wrapper for decomposition with knowledge extraction and performance tracking.

    This function demonstrates how to integrate the knowledge and performance systems
    with the decomposition process. It should be called within the decompose() method
    of the DecompositionEngine.

    Args:
        problem: The ProblemDefinition to decompose
        strategy: The decomposition strategy to use
        extract_knowledge: Whether to extract and use knowledge artifacts
        track_performance: Whether to track performance metrics
        artifact_store_path: Path to artifact storage
        metrics_store_path: Path to metrics storage

    Returns:
        Dictionary containing:
        - relevant_artifacts: List of relevant knowledge artifacts (if extract_knowledge=True)
        - performance_tracker: PerformanceMetricsTracker instance (if track_performance=True)
        - artifact_extractor: KnowledgeArtifactExtractor instance (if extract_knowledge=True)
    """
    result = {
        'relevant_artifacts': [],
        'performance_tracker': None,
        'artifact_extractor': None
    }

    try:
        # Step 1: Retrieve relevant knowledge artifacts if enabled
        if extract_knowledge:
            logger.info("Initializing knowledge artifact extraction...")
            artifact_extractor = KnowledgeArtifactExtractor(artifact_store_path)
            result['artifact_extractor'] = artifact_extractor

            # Get relevant artifacts for this problem
            domain = problem.domain_context.domain if hasattr(problem, 'domain_context') else "general"
            relevant_artifacts = artifact_extractor.retrieve_relevant_artifacts(problem, domain)
            result['relevant_artifacts'] = relevant_artifacts

            logger.info(f"Retrieved {len(relevant_artifacts)} relevant knowledge artifacts")

            # Store artifacts in problem metadata for use during decomposition
            if not hasattr(problem, 'metadata'):
                problem.metadata = {}
            problem.metadata['relevant_artifacts'] = [a.to_dict() for a in relevant_artifacts]
            problem.metadata['extract_knowledge'] = True

        # Step 2: Initialize performance tracker if enabled
        if track_performance:
            logger.info("Initializing performance metrics tracking...")
            performance_tracker = PerformanceMetricsTracker(metrics_store_path)
            result['performance_tracker'] = performance_tracker

            # Store reference in problem metadata
            if not hasattr(problem, 'metadata'):
                problem.metadata = {}
            problem.metadata['track_performance'] = True

    except (ValueError, TypeError, AttributeError, OSError, IOError) as e:
        logger.error(f"Error during integration setup: {e}", exc_info=True)

    return result


def record_decomposition_completion(
    decomposition_plan,
    problem,
    decomposition_time: float,
    performance_tracker: Optional[PerformanceMetricsTracker] = None
):
    """
    Record decomposition metrics after plan creation.

    Args:
        decomposition_plan: The created decomposition plan
        problem: The original problem
        decomposition_time: Time taken for decomposition
        performance_tracker: Performance tracker instance
    """
    try:
        if performance_tracker:
            performance_tracker.record_decomposition_metrics(
                decomposition_plan,
                problem,
                decomposition_time
            )
            logger.info(f"Recorded decomposition metrics for problem {problem.id}")

    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to record decomposition metrics: {e}", exc_info=True)


def record_solution_completion(
    sub_problem_id: str,
    solution,
    validation,
    generation_time: float,
    performance_tracker: Optional[PerformanceMetricsTracker] = None
):
    """
    Record solution metrics after solution generation.

    Args:
        sub_problem_id: ID of the sub-problem
        solution: The generated solution
        validation: Validation results
        generation_time: Time taken for generation
        performance_tracker: Performance tracker instance
    """
    try:
        if performance_tracker:
            performance_tracker.record_solution_metrics(
                sub_problem_id,
                solution,
                validation,
                generation_time
            )
            logger.debug(f"Recorded solution metrics for {sub_problem_id}")

    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to record solution metrics: {e}", exc_info=True)


def extract_and_store_knowledge(
    decomposition_plan,
    solutions: Dict[str, Any],
    validation_results: Dict[str, Any],
    artifact_extractor: Optional[KnowledgeArtifactExtractor] = None
) -> List[Any]:
    """
    Extract knowledge artifacts from completed problem solving.

    Args:
        decomposition_plan: The decomposition plan used
        solutions: Dictionary of solutions generated
        validation_results: Dictionary of validation results
        artifact_extractor: Artifact extractor instance

    Returns:
        List of extracted knowledge artifacts
    """
    try:
        if artifact_extractor:
            artifacts = artifact_extractor.extract_artifacts(
                decomposition_plan,
                solutions,
                validation_results
            )
            logger.info(f"Extracted and stored {len(artifacts)} knowledge artifacts")
            return artifacts

        return []

    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to extract knowledge artifacts: {e}", exc_info=True)
        return []


def generate_performance_summary(
    performance_tracker: PerformanceMetricsTracker,
    time_period: str = "all"
) -> Dict[str, Any]:
    """
    Generate comprehensive performance summary.

    Args:
        performance_tracker: Performance tracker instance
        time_period: Time period for report

    Returns:
        Dictionary with performance summary
    """
    try:
        report = performance_tracker.generate_performance_report(time_period)

        return {
            'report_id': report.report_id,
            'generated_at': report.generated_at,
            'time_period': report.time_period,
            'overall_metrics': report.overall_metrics.to_dict(),
            'total_strategies': len(report.strategy_metrics),
            'total_teams': len(report.team_metrics),
            'total_domains': len(report.domain_metrics),
            'improvement_areas': report.improvement_areas,
            'trend_analyses': {k: v.to_dict() for k, v in report.trend_analyses.items()}
        }

    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to generate performance summary: {e}", exc_info=True)
        return {}


def get_knowledge_statistics(
    artifact_extractor: KnowledgeArtifactExtractor
) -> Dict[str, Any]:
    """
    Get statistics about stored knowledge artifacts.

    Args:
        artifact_extractor: Artifact extractor instance

    Returns:
        Dictionary with artifact statistics
    """
    try:
        stats = artifact_extractor.get_artifact_statistics()

        # Add additional details
        by_type = {}
        for artifact_type, count in stats['by_type'].items():
            by_type[artifact_type] = count

        by_domain = {}
        for domain, count in stats['by_domain'].items():
            by_domain[domain] = count

        return {
            'total_artifacts': stats['total_artifacts'],
            'by_type': by_type,
            'by_domain': by_domain,
            'average_confidence': stats['avg_confidence'],
            'average_success_rate': stats['avg_success_rate'],
            'high_confidence_artifacts': stats['high_confidence_count']
        }

    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to get knowledge statistics: {e}", exc_info=True)
        return {}


# Example usage integration function
def example_decomposition_with_learning(
    decomposition_engine,
    problem,
    strategy: str = "semantic"
):
    """
    Example showing complete integration with decomposition engine.

    This demonstrates the full workflow:
    1. Setup with knowledge retrieval
    2. Decompose with timing
    3. Record metrics
    4. Extract knowledge from results
    """
    # Initialize systems
    integration_result = integrate_with_decomposition(
        problem=problem,
        strategy=strategy,
        extract_knowledge=True,
        track_performance=True
    )

    artifact_extractor = integration_result.get('artifact_extractor')
    performance_tracker = integration_result.get('performance_tracker')
    relevant_artifacts = integration_result.get('relevant_artifacts', [])

    logger.info(f"Starting decomposition with {len(relevant_artifacts)} relevant artifacts")

    # Decompose (with timing)
    start_time = time.time()

    # Here you would call the actual decomposition engine
    # plan = decomposition_engine.decompose(problem, strategy=strategy)

    decomposition_time = time.time() - start_time

    # For this example, we'll create a mock plan
    # In real usage, this would be the actual plan from the engine
    plan = None  # decomposition_engine.decompose(problem, strategy=strategy)

    # Record decomposition metrics
    if plan and performance_tracker:
        record_decomposition_completion(
            plan,
            problem,
            decomposition_time,
            performance_tracker
        )

    # In a complete workflow, you would also:
    # 1. Generate solutions for each sub-problem
    # 2. Validate solutions
    # 3. Record solution metrics
    # 4. Extract knowledge artifacts

    # Example of what would happen after solutions are generated:
    # solutions = {...}  # Generated solutions
    # validation_results = {...}  # Validation results

    # Extract knowledge from completed work
    # if solutions and validation_results and artifact_extractor:
    #     artifacts = extract_and_store_knowledge(
    #         plan,
    #         solutions,
    #         validation_results,
    #         artifact_extractor
    #     )

    # Generate performance summary
    if performance_tracker:
        summary = generate_performance_summary(performance_tracker)
        logger.info(f"Performance summary: {summary['overall_metrics']}")

    return {
        'plan': plan,
        'relevant_artifacts_used': len(relevant_artifacts),
        'decomposition_time': decomposition_time
    }
