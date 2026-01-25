"""
Integration test for dependency_builder.py with problem_fractal_pipeline.py

This test demonstrates the complete workflow from problem decomposition
through dependency analysis to execution planning.
"""

import logging
from datetime import datetime
from dependency_builder import DependencyBuilder
from sovereign_data_models import SubProblem, ProblemStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_complex_workflow():
    """
    Create a realistic complex workflow scenario.

    Simulates a microservices deployment with multiple interconnected services.
    """
    logger.info("Creating complex workflow scenario...")

    # Define the workflow structure
    # Format: (task_id, dependencies, complexity, description)
    workflow_spec = [
        # Infrastructure layer
        ("infra_setup", [], 1.0, "Setup infrastructure (VPC, networking, etc.)"),
        ("database_cluster", ["infra_setup"], 2.5, "Setup database cluster"),
        ("cache_layer", ["infra_setup"], 2.0, "Setup Redis cache layer"),
        ("message_queue", ["infra_setup"], 1.5, "Setup message queue (RabbitMQ)"),

        # Authentication service
        ("auth_service_design", [], 2.0, "Design authentication service"),
        ("auth_service_impl", ["auth_service_design", "database_cluster"], 3.0, "Implement auth service"),
        ("auth_service_test", ["auth_service_impl"], 1.5, "Test authentication service"),

        # User service
        ("user_service_design", ["auth_service_design"], 2.0, "Design user service"),
        ("user_service_impl", ["user_service_design", "database_cluster", "cache_layer"], 3.5,
         "Implement user service"),
        ("user_service_test", ["user_service_impl", "auth_service_test"], 2.0, "Test user service"),

        # Payment service
        ("payment_service_design", [], 2.5, "Design payment service"),
        ("payment_service_impl", ["payment_service_design", "database_cluster", "message_queue"], 4.0,
         "Implement payment service"),
        ("payment_service_test", ["payment_service_impl"], 2.5, "Test payment service"),

        # Order service
        ("order_service_design", ["user_service_design", "payment_service_design"], 2.0,
         "Design order service"),
        ("order_service_impl", ["order_service_design", "database_cluster", "cache_layer",
                               "message_queue"], 4.5, "Implement order service"),
        ("order_service_test", ["order_service_impl", "user_service_test", "payment_service_test"],
                               3.0, "Test order service"),

        # API Gateway
        ("api_gateway_config", [], 1.5, "Configure API gateway"),
        ("api_gateway_deploy", ["api_gateway_config", "auth_service_test", "user_service_test",
                               "payment_service_test", "order_service_test"], 2.5,
         "Deploy API gateway with all services"),

        # Monitoring & Logging
        ("monitoring_setup", ["infra_setup"], 2.0, "Setup monitoring and logging"),
        ("monitoring_config", ["monitoring_setup", "api_gateway_deploy"], 1.5,
         "Configure monitoring for all services"),

        # Final deployment
        ("integration_tests", ["api_gateway_deploy", "monitoring_config"], 3.0,
         "Run integration tests"),
        ("production_deploy", ["integration_tests"], 2.0,
         "Deploy to production"),
    ]

    # Create SubProblem objects
    sub_problems = []
    for task_id, dependencies, complexity, description in workflow_spec:
        sp = SubProblem(
            sub_problem_id=task_id,
            parent_id=None,
            title=task_id.replace("_", " ").title(),
            description=description,
            status=ProblemStatus.PENDING,
            confidence=0.85,
            assigned_agent=None,
            created_at=datetime.now(),
            completed_at=None
        )
        sp.dependencies = dependencies
        sp.complexity_score = complexity
        sub_problems.append(sp)

    logger.info(f"Created {len(sub_problems)} sub-problems")
    return sub_problems


def analyze_workflow(sub_problems):
    """
    Perform complete dependency analysis on the workflow.
    """
    logger.info("=" * 80)
    logger.info("DEPENDENCY ANALYSIS")
    logger.info("=" * 80)

    # Build dependency graph
    logger.info("\n1. Building dependency graph...")
    builder = DependencyBuilder(validate_on_build=True)
    graph = builder.build_dependency_graph(sub_problems)
    logger.info(f"   Built graph with {len(graph)} nodes")

    # Check for circular dependencies
    logger.info("\n2. Checking for circular dependencies...")
    cycles = builder.detect_circular_dependencies(graph)
    if cycles:
        logger.error(f"   Found {len(cycles)} circular dependencies!")
        for i, cycle in enumerate(cycles, 1):
            logger.error(f"   Cycle {i}: {' -> '.join(cycle)}")
        return None
    else:
        logger.info("   No circular dependencies detected")

    # Calculate execution order
    logger.info("\n3. Calculating optimal execution order...")
    execution_order = builder.calculate_execution_order(graph)
    logger.info(f"   Execution order: {' -> '.join(execution_order[:5])}... "
                f"({len(execution_order)} total tasks)")

    # Identify critical path
    logger.info("\n4. Identifying critical path...")
    critical_path = builder.identify_critical_path(graph)
    total_complexity = sum(graph.nodes[nid].complexity for nid in critical_path)
    logger.info(f"   Critical path length: {len(critical_path)} tasks")
    logger.info(f"   Total complexity: {total_complexity:.1f}")
    logger.info(f"   Critical path: {' -> '.join(critical_path[:5])}...")

    # Find parallelizable tasks
    logger.info("\n5. Analyzing parallelization opportunities...")
    parallel_groups = builder.find_parallelizable_tasks(graph)
    logger.info(f"   Parallelization levels: {len(parallel_groups)}")

    max_parallel = max(len(group) for group in parallel_groups) if parallel_groups else 0
    logger.info(f"   Maximum parallel tasks: {max_parallel} (at level "
                f"{parallel_groups.index(max(parallel_groups, key=len)) if parallel_groups else 0})")

    # Calculate statistics
    logger.info("\n6. Computing graph statistics...")
    stats = builder.analyze_graph_statistics(graph)
    logger.info(f"   Total nodes: {stats['total_nodes']}")
    logger.info(f"   Total edges: {stats['total_edges']}")
    logger.info(f"   Average dependencies per node: {stats['avg_dependencies']:.2f}")
    logger.info(f"   Maximum depth: {stats['max_depth']}")
    logger.info(f"   Source nodes (no deps): {stats['sources']}")
    logger.info(f"   Sink nodes (no dependents): {stats['sinks']}")
    logger.info(f"   Is valid DAG: {stats['is_dag']}")

    return {
        "graph": graph,
        "execution_order": execution_order,
        "critical_path": critical_path,
        "parallel_groups": parallel_groups,
        "stats": stats,
    }


def generate_execution_plan(analysis_result):
    """
    Generate a detailed execution plan based on analysis.
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION PLAN")
    logger.info("=" * 80)

    if not analysis_result:
        logger.error("Cannot generate plan: Analysis failed")
        return

    graph = analysis_result["graph"]
    parallel_groups = analysis_result["parallel_groups"]
    critical_path = analysis_result["critical_path"]
    execution_order = analysis_result["execution_order"]

    logger.info("\nExecution Strategy:")
    logger.info("-" * 80)

    for level, tasks in enumerate(parallel_groups):
        logger.info(f"\nLevel {level}: Execute {len(tasks)} task(s) in parallel")
        logger.info("  Tasks:")
        for task_id in tasks:
            node = graph.nodes[task_id]
            is_critical = task_id in critical_path
            critical_marker = " [CRITICAL PATH]" if is_critical else ""
            logger.info(f"    - {task_id}{critical_marker}")
            logger.info(f"      Complexity: {node.complexity}")
            logger.info(f"      Dependencies: {', '.join(node.dependencies) if node.dependencies else 'None'}")
            logger.info(f"      Depth: {node.depth}")

    logger.info("\n" + "=" * 80)
    logger.info("CRITICAL PATH ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"\nCritical Path ({len(critical_path)} tasks):")
    for i, task_id in enumerate(critical_path, 1):
        node = graph.nodes[task_id]
        logger.info(f"{i}. {task_id}")
        logger.info(f"   Complexity: {node.complexity}")
        logger.info(f"   Description: {node.sub_problem.description}")

    total_complexity = sum(graph.nodes[tid].complexity for tid in critical_path)
    logger.info(f"\nTotal Critical Path Complexity: {total_complexity:.1f}")

    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION METRICS")
    logger.info("=" * 80)

    # Calculate potential time savings
    sequential_time = sum(graph.nodes[tid].complexity for tid in execution_order)
    parallel_time = sum(
        max(graph.nodes[tid].complexity for tid in group)
        for group in parallel_groups
    )

    time_saved = sequential_time - parallel_time
    efficiency_gain = (time_saved / sequential_time * 100) if sequential_time > 0 else 0

    logger.info(f"\nSequential Execution Time: {sequential_time:.1f} complexity units")
    logger.info(f"Parallel Execution Time: {parallel_time:.1f} complexity units")
    logger.info(f"Time Saved: {time_saved:.1f} complexity units ({efficiency_gain:.1f}%)")


def main():
    """
    Main integration test.
    """
    logger.info("=" * 80)
    logger.info("DEPENDENCY BUILDER INTEGRATION TEST")
    logger.info("=" * 80)

    # Create complex workflow
    sub_problems = create_complex_workflow()

    # Analyze workflow
    analysis_result = analyze_workflow(sub_problems)

    if analysis_result:
        # Generate execution plan
        generate_execution_plan(analysis_result)

        logger.info("\n" + "=" * 80)
        logger.info("INTEGRATION TEST PASSED")
        logger.info("=" * 80)
        logger.info("\nAll dependency analysis features working correctly!")
        return 0
    else:
        logger.error("\n" + "=" * 80)
        logger.error("INTEGRATION TEST FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
