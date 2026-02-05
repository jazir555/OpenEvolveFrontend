"""
Complete Gauntlet System Integration Example

This example demonstrates how to use all components of the OpenEvolve
Gauntlet system together in a comprehensive problem-solving pipeline.

Components Demonstrated:
- Configuration management
- Metrics collection and monitoring
- Checkpointing and recovery
- Parallel execution
- Solution caching
- Circuit breakers
- Dynamic difficulty adjustment
- Team performance tracking
- Visualization
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

# Import all Gauntlet components
from bubblelabs_nodes import (
    # Configuration
    create_config,
    StrategyProfile,
    CheckpointFrequency,
    CacheType,
    CircuitBreakerStrategy,

    # Core Components
    ParallelProblemExecutor,
    AtomicSolutionCache,
    CheckpointManager,
    create_checkpoint_manager,
    create_solution_cache,
    create_parallel_executor,

    # Metrics
    get_metrics_collector,
    track_performance,

    # Testing
    TestDataGenerator,
    ValidationHelper,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GauntletSystem:
    """
    Complete Gauntlet system integrating all components.
    """

    def __init__(self, config=None):
        # Create configuration
        self.config = config or create_config(
            profile=StrategyProfile.BALANCED,
            from_env=True
        )

        logger.info(f"Initializing Gauntlet System with {self.config.strategy_profile.value} profile")

        # Initialize metrics collector
        self.metrics = get_metrics_collector()
        self.metrics.start_resource_monitoring(interval_seconds=5.0)

        # Initialize cache
        self.cache = create_solution_cache(
            cache_type=self.config.cache.cache_type,
            ttl_seconds=self.config.cache.ttl_seconds,
            max_size=self.config.cache.max_size,
            redis_url=self.config.cache.redis_url
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(
            storage_path=self.config.checkpointing.storage_path,
            compression=self.config.checkpointing.compression,
            frequency=self.config.checkpointing.frequency,
            retention_count=self.config.checkpointing.retention_count
        )

        # Initialize parallel executor
        self.parallel_executor = create_parallel_executor(
            max_parallelism=self.config.parallel_execution.max_parallelism,
            timeout_seconds=self.config.parallel_execution.timeout_seconds
        )

        logger.info("Gauntlet System initialized successfully")

    async def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a problem using the complete Gauntlet system.

        Args:
            problem: Problem to solve

        Returns:
            Solution result
        """
        problem_id = problem.get('id', 'unknown')
        logger.info(f"Solving problem: {problem_id}")

        start_time = datetime.utcnow()

        try:
            # Check cache first
            cached_solution = await self.cache.get(problem)
            if cached_solution:
                logger.info(f"Cache hit for problem {problem_id}")
                self.metrics.record_cache_operation(
                    operation="hit",
                    cache_type=self.config.cache.cache_type.value,
                    key=problem_id
                )

                self.metrics.increment("problems_solved_from_cache")
                return cached_solution

            self.metrics.record_cache_operation(
                operation="miss",
                cache_type=self.config.cache.cache_type.value,
                key=problem_id
            )

            # Create checkpoint before solving
            if self.config.checkpointing.enabled:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    problem=problem,
                    context={'stage': 'before_solve'},
                    solutions={},
                    level=0,
                    stage='before_solve'
                )
                logger.info(f"Created checkpoint: {checkpoint_id}")

            # Solve the problem
            solution = await self._solve_internal(problem)

            # Cache the solution
            await self.cache.set(problem, solution)
            logger.info(f"Cached solution for problem {problem_id}")

            # Record metrics
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.record_performance(
                operation="solve_problem",
                duration_ms=duration_ms,
                success=solution.get('success', False),
                metadata={'problem_id': problem_id}
            )

            # Record team performance
            if solution.get('team_id'):
                self.metrics.record_team_performance(
                    team_id=solution['team_id'],
                    problem_id=problem_id,
                    domain=problem.get('domain', 'general'),
                    difficulty=problem.get('difficulty', 3),
                    success=solution.get('success', False),
                    score=solution.get('score', 0.0),
                    execution_time=duration_ms / 1000.0
                )

            return solution

        except Exception as e:
            logger.error(f"Error solving problem {problem_id}: {e}")
            self.metrics.increment("problem_solve_errors")
            raise

    async def solve_parallel_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Solve multiple problems in parallel.

        Args:
            problems: List of problems to solve

        Returns:
            List of solutions
        """
        logger.info(f"Solving {len(problems)} problems in parallel")

        # Analyze dependencies
        analyzer = self.parallel_executor.dependency_analyzer
        independent_problems = analyzer.find_independent_problems(problems)

        logger.info(f"Found {len(independent_problems)} independent problems")

        # Execute in parallel
        async def solve_single(p):
            return await self.solve_problem(p)

        result = await self.parallel_executor.execute_in_parallel(
            problems=independent_problems,
            executor_func=solve_single,
            context={}
        )

        logger.info(
            f"Parallel execution complete: "
            f"{result.successful_count}/{result.total_count} successful"
        )

        # Record metrics
        self.metrics.set_gauge("parallel_execution_success_rate", result.success_rate)

        return result.results

    async def create_checkpoint(self, problem, context, solutions, level, stage):
        """Create a checkpoint"""
        if not self.config.checkpointing.enabled:
            return None

        return await self.checkpoint_manager.create_checkpoint(
            problem=problem,
            context=context,
            solutions=solutions,
            level=level,
            stage=stage
        )

    async def load_checkpoint(self, checkpoint_id: str):
        """Load a checkpoint"""
        return await self.checkpoint_manager.load_checkpoint(checkpoint_id)

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report"""
        return self.metrics.get_all_metrics()

    def visualize_problem(self, problem: Dict[str, Any]) -> str:
        """Visualize problem hierarchy"""
        from bubblelabs_nodes import visualize_ascii

        return visualize_ascii(problem)

    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down Gauntlet System")

        # Stop resource monitoring
        self.metrics.stop_resource_monitoring()

        logger.info("Gauntlet System shutdown complete")

    async def _solve_internal(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal problem solving logic.

        This is a simplified example - real implementation would
        use Blue/Red/Gold teams, decomposition, etc.
        """
        # Simulate solving time
        await asyncio.sleep(0.1)

        # Return mock solution
        return {
            'problem_id': problem.get('id'),
            'success': True,
            'score': 0.85,
            'solution': f"Solution for {problem.get('id')}",
            'confidence': 0.85,
            'team_id': 'blue_team_1',
            'timestamp': datetime.utcnow().isoformat(),
        }


async def demo_complete_system():
    """Demonstration of complete Gauntlet system"""

    print("\n" + "=" * 80)
    print("COMPLETE GAUNTLET SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Create system with Conservative profile
    print("\n1. Initializing Gauntlet System...")
    system = GauntletSystem(config=create_config(profile=StrategyProfile.CONSERVATIVE))
    print("   [OK] System initialized")

    # Generate test problems
    print("\n2. Generating Test Problems...")
    generator = TestDataGenerator(seed=42)

    problems = [
        generator.generate_problem("medium", "web"),
        generator.generate_problem("easy", "ml"),
        generator.generate_problem("hard", "data"),
    ]

    print(f"   Generated {len(problems)} test problems")

    # Visualize first problem
    print("\n3. Problem Visualization:")
    visualization = system.visualize_problem(problems[0])
    print(visualization)

    # Solve single problem
    print("\n4. Solving Single Problem...")
    solution = await system.solve_problem(problems[0])
    print(f"   Problem ID: {solution['problem_id']}")
    print(f"   Success: {solution['success']}")
    print(f"   Score: {solution['score']}")

    # Test cache
    print("\n5. Testing Cache...")
    cached_solution = await system.cache.get(problems[0])
    print(f"   Cache hit: {cached_solution is not None}")

    # Solve multiple problems in parallel
    print("\n6. Solving Multiple Problems in Parallel...")
    solutions = await system.solve_parallel_problems(problems)
    print(f"   Solved {len(solutions)} problems")

    # Create checkpoint
    print("\n7. Creating Checkpoint...")
    checkpoint_id = await system.create_checkpoint(
        problem=problems[0],
        context={'stage': 'demo'},
        solutions={'test': 'solution'},
        level=0,
        stage='demo'
    )
    print(f"   Checkpoint ID: {checkpoint_id}")

    # Get metrics report
    print("\n8. Metrics Report:")
    metrics = system.get_metrics_report()

    print(f"   Timestamp: {metrics['timestamp']}")
    print(f"   Counters: {len(metrics['counters'])}")
    print(f"   Gauges: {len(metrics['gauges'])}")

    # Performance summary
    if 'performance' in metrics and 'all' in metrics['performance']:
        perf = metrics['performance']['all']
        print(f"\n   Performance:")
        print(f"   Total requests: {perf.get('total_requests', 0)}")
        print(f"   Success rate: {perf.get('success_rate', 0):.1%}")

    # Team performance
    if 'team_performance' in metrics:
        team = metrics['team_performance']
        print(f"\n   Team Performance:")
        print(f"   Total problems: {team.get('total_problems', 0)}")
        print(f"   Success rate: {team.get('success_rate', 0):.1%}")

    # Cache summary
    if 'cache' in metrics:
        cache = metrics['cache']
        print(f"\n   Cache:")
        print(f"   Hit rate: {cache.get('hit_rate', 0):.1%}")
        print(f"   Total requests: {cache.get('total_requests', 0)}")

    # Validation
    print("\n9. Validation:")

    # Validate problem
    problem_report = ValidationHelper.validate_problem(problems[0])
    print(f"   Problem valid: {problem_report.is_valid}")

    # Validate solution
    solution_report = ValidationHelper.validate_solution(solution)
    print(f"   Solution valid: {solution_report.is_valid}")

    # Cleanup
    print("\n10. Cleanup...")
    system.shutdown()
    print("   [OK] System shutdown complete")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


async def demo_configuration_profiles():
    """Demonstrate different configuration profiles"""

    print("\n" + "=" * 80)
    print("CONFIGURATION PROFILES DEMONSTRATION")
    print("=" * 80)

    profiles = [
        StrategyProfile.CONSERVATIVE,
        StrategyProfile.BALANCED,
        StrategyProfile.AGGRESSIVE,
        StrategyProfile.FAST,
        StrategyProfile.THOROUGH,
    ]

    for profile in profiles:
        print(f"\n{profile.value.upper()} Profile:")
        config = create_config(profile=profile)

        print(f"  Max gauntlet rounds: {config.max_gauntlet_rounds}")
        print(f"  Pass threshold: {config.pass_threshold}")
        print(f"  Max decomposition depth: {config.max_decomposition_depth}")
        print(f"  Initial difficulty: {config.difficulty.initial_level.value}")
        print(f"  Parallel execution: {config.parallel_execution.enabled}")
        print(f"  Cache enabled: {config.cache.enabled}")
        print(f"  Fuzzing enabled: {config.fuzzing.enabled}")

    print("\n" + "=" * 80)


async def demo_metrics_collection():
    """Demonstrate comprehensive metrics collection"""

    print("\n" + "=" * 80)
    print("METRICS COLLECTION DEMONSTRATION")
    print("=" * 80)

    collector = get_metrics_collector()

    # Counter metrics
    print("\n1. Counter Metrics:")
    collector.increment("problems_solved", labels={"domain": "web"})
    collector.increment("problems_solved", labels={"domain": "ml"})
    collector.increment("problems_solved", labels={"domain": "web"})
    print(f"   Web problems solved: {collector.get_counter('problems_solved', {'domain': 'web'})}")

    # Gauge metrics
    print("\n2. Gauge Metrics:")
    collector.set_gauge("active_problems", 5)
    collector.set_gauge("queue_size", 10)
    print(f"   Active problems: {collector.get_gauge('active_problems')}")
    print(f"   Queue size: {collector.get_gauge('queue_size')}")

    # Histogram metrics
    print("\n3. Histogram Metrics:")
    for duration in [100, 150, 200, 120, 180]:
        collector.record_histogram("solve_duration_ms", duration)
    stats = collector.get_histogram_stats("solve_duration_ms")
    print(f"   Average: {stats['avg']:.1f}ms")
    print(f"   P95: {stats['p95']:.1f}ms")

    # Performance metrics
    print("\n4. Performance Metrics:")
    collector.record_performance("solve_problem", 150.5, True)
    collector.record_performance("solve_problem", 200.3, True)
    collector.record_performance("solve_problem", 100.2, False)
    summary = collector.get_performance_summary("solve_problem")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Average duration: {summary['avg_duration_ms']:.1f}ms")

    # Team performance
    print("\n5. Team Performance:")
    collector.record_team_performance(
        team_id="blue_team_1",
        problem_id="problem_123",
        domain="web",
        difficulty=3,
        success=True,
        score=0.85,
        execution_time=150.0
    )
    team_summary = collector.get_team_performance_summary("blue_team_1")
    print(f"   Success rate: {team_summary['success_rate']:.1%}")
    print(f"   Average score: {team_summary['avg_score']:.2f}")

    print("\n" + "=" * 80)


async def main():
    """Run all demonstrations"""

    # Complete system demo
    await demo_complete_system()

    # Configuration profiles demo
    await demo_configuration_profiles()

    # Metrics collection demo
    await demo_metrics_collection()

    print("\n[OK] ALL DEMONSTRATIONS COMPLETE")


if __name__ == '__main__':
    asyncio.run(main())
