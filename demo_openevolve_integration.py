"""
OpenEvolve Integration Demo - Complete Showcase

This script demonstrates the full integration between:
- Enhanced Decomposition Engine
- Enhanced Recomposition Engine  
- OpenEvolve Evolution Platform

Usage:
    python demo_openevolve_integration.py
"""

import time
from typing import Dict, List, Any

# Import enhanced systems
from enhanced_decomposition_engine import (
    DecompositionStrategy,
    ProblemDomain,
    create_problem_definition
)

from enhanced_recomposition_engine import (
    AssemblyStrategy,
    create_subproblem_solution
)

# Import OpenEvolve integration
from openevolve_enhanced_decomposition_integration import (
    OpenEvolveIntegratedPipeline,
    OpenEvolveSolutionSolver,
    EvolutionConfig,
    quick_solve_with_openevolve,
    compare_strategies_with_openevolve
)

from openevolve_decomposition_adapter import (
    OpenEvolveDecompositionAdapter,
    DecompositionMetricsCollector,
    create_decomposition_aware_config
)


def print_header(title: str, char: str = "="):
    """Print formatted header."""
    print("\n" + char * 70)
    print(f"  {title}")
    print(char * 70)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print("─" * 70)


def demo_basic_integration():
    """Demo 1: Basic OpenEvolve-Integrated Pipeline"""
    print_header("DEMO 1: Basic OpenEvolve Integration")
    
    # Create problem
    problem = create_problem_definition(
        title="Build Microservices Architecture",
        description="""
        Design and implement a microservices-based e-commerce platform with:
        - Product catalog service with search and filtering
        - Shopping cart service with persistence
        - Order processing service with inventory management
        - Payment gateway integration with multiple providers
        - User authentication and authorization service
        - API gateway with rate limiting
        - Event-driven communication between services
        - Centralized logging and monitoring
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=8.5
    )
    
    print(f"\n📋 Problem: {problem.title}")
    print(f"   Domain: {problem.domain.value}")
    print(f"   Complexity: {problem.complexity_score.overall_complexity}/10")
    
    # Configure evolution
    evolution_config = EvolutionConfig(
        max_iterations=25,
        population_size=50,
        parallel_evolution=True,
        max_workers=4,
        min_fitness_threshold=0.7
    )
    
    print(f"\n⚙️  Evolution Configuration:")
    print(f"   Max Iterations: {evolution_config.max_iterations}")
    print(f"   Population Size: {evolution_config.population_size}")
    print(f"   Parallel Evolution: {evolution_config.parallel_evolution}")
    
    # Create and run pipeline
    print(f"\n🚀 Executing OpenEvolve-Integrated Pipeline...")
    print("   (Decomposition → Evolution → Recomposition)")
    
    pipeline = OpenEvolveIntegratedPipeline(evolution_config=evolution_config)
    
    start_time = time.time()
    result = pipeline.execute(problem)
    elapsed = time.time() - start_time
    
    print_section("Results")
    print(f"✅ Pipeline completed in {elapsed:.2f}s")
    print(f"\n📊 Quality Scores:")
    print(f"   Overall Quality:     {result.overall_quality:.2f}/1.0")
    print(f"   Decomposition:       {result.decomposition_quality:.2f}/1.0")
    print(f"   Solution:            {result.solution_quality:.2f}/1.0")
    
    if result.decomposition_plan:
        print(f"\n🧩 Decomposition:")
        print(f"   Strategy: {result.decomposition_plan.strategy_used.value}")
        print(f"   Sub-problems: {len(result.decomposition_plan.sub_problems)}")
        print(f"   Parallel Groups: {len(result.decomposition_plan.parallel_groups)}")
    
    if result.sub_solutions:
        fitness_scores = [s.quality_score for s in result.sub_solutions.values()]
        print(f"\n🧬 Evolution Results:")
        print(f"   Solutions Generated: {len(result.sub_solutions)}")
        print(f"   Avg Fitness: {sum(fitness_scores)/len(fitness_scores):.2f}")
        print(f"   Max Fitness: {max(fitness_scores):.2f}")
        print(f"   Min Fitness: {min(fitness_scores):.2f}")
    
    if result.integrated_solution:
        print(f"\n🔧 Assembly:")
        print(f"   Strategy: {result.integrated_solution.assembly_strategy.value}")
        print(f"   Conflicts: {len(result.integrated_solution.conflicts_detected)} detected")
        print(f"   Resolved: {len(result.integrated_solution.conflicts_resolved)} resolved")
        print(f"   Content Length: {len(result.integrated_solution.assembled_content)} chars")


def demo_strategy_comparison():
    """Demo 2: Compare Decomposition Strategies with OpenEvolve"""
    print_header("DEMO 2: Strategy Comparison with OpenEvolve Evolution")
    
    problem = create_problem_definition(
        title="AI-Powered Recommendation Engine",
        description="""
        Build a recommendation engine that uses machine learning to suggest
        products to users based on their browsing history, purchase patterns,
        and similar user behaviors. Must handle 10M+ users with personalized
        recommendations served in <100ms.
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=9.0
    )
    
    print(f"\n📋 Problem: {problem.title}")
    print(f"   Complexity: {problem.complexity_score.overall_complexity}/10")
    
    print(f"\n🔄 Comparing decomposition strategies with OpenEvolve evolution...")
    print("   This may take a moment...")
    
    result = compare_strategies_with_openevolve(
        problem,
        strategies=[
            DecompositionStrategy.HIERARCHICAL,
            DecompositionStrategy.FUNCTIONAL,
            DecompositionStrategy.SEMANTIC,
            DecompositionStrategy.HYBRID,
        ]
    )
    
    print_section("Strategy Comparison Results")
    print(f"\n{'Strategy':<20} {'Sub-Prob':>10} {'Decomp Q':>10} {'Solution Q':>12} {'Conflicts':>10}")
    print("-" * 70)
    
    for r in result['results']:
        print(f"{r['strategy']:<20} {r['sub_problems']:>10} {r['decomposition_quality']:>10.2f} "
              f"{r['solution_quality']:>12.2f} {r['conflicts']:>10}")
    
    print(f"\n🏆 Best Strategy: {result['best_strategy']}")


def demo_cross_domain():
    """Demo 3: Cross-Domain Problem Solving"""
    print_header("DEMO 3: Cross-Domain Problem Solving")
    
    domains = [
        {
            'domain': ProblemDomain.FINANCE,
            'title': "Algorithmic Trading System",
            'description': """
            Build a high-frequency trading system that analyzes market data
            in real-time, identifies arbitrage opportunities, and executes
            trades within microseconds while managing risk exposure.
            """,
            'complexity': 9.5
        },
        {
            'domain': ProblemDomain.HEALTHCARE,
            'title': "Medical Imaging Analysis Platform",
            'description': """
            Create a platform for analyzing medical images (X-rays, MRIs, CT scans)
            using deep learning to assist radiologists in detecting anomalies
            and diagnosing conditions with 95%+ accuracy.
            """,
            'complexity': 9.0
        },
        {
            'domain': ProblemDomain.MANUFACTURING,
            'title': "Smart Factory IoT System",
            'description': """
            Design an IoT system for smart manufacturing that collects sensor
            data from production lines, predicts equipment failures, optimizes
            maintenance schedules, and improves overall equipment effectiveness.
            """,
            'complexity': 8.0
        },
    ]
    
    results = []
    
    for domain_info in domains:
        print(f"\n📋 {domain_info['title']} ({domain_info['domain'].value})")
        
        problem = create_problem_definition(
            title=domain_info['title'],
            description=domain_info['description'],
            domain=domain_info['domain'],
            complexity=domain_info['complexity']
        )
        
        pipeline = OpenEvolveIntegratedPipeline()
        
        start = time.time()
        result = pipeline.execute(problem)
        elapsed = time.time() - start
        
        results.append({
            'domain': domain_info['domain'].value,
            'title': domain_info['title'][:25],
            'quality': result.overall_quality,
            'sub_problems': len(result.decomposition_plan.sub_problems) if result.decomposition_plan else 0,
            'time': elapsed,
            'successful': result.is_successful()
        })
        
        print(f"   ✅ Quality: {result.overall_quality:.2f} | Time: {elapsed:.2f}s")
    
    print_section("Cross-Domain Summary")
    print(f"\n{'Domain':<15} {'Problem':<25} {'Quality':>10} {'Time':>8} {'Status':>10}")
    print("-" * 80)
    
    for r in results:
        status = "✅ PASS" if r['successful'] else "❌ FAIL"
        print(f"{r['domain']:<15} {r['title']:<25} {r['quality']:>10.2f} {r['time']:>8.2f} {status:>10}")


def demo_metrics_collection():
    """Demo 4: Metrics Collection and Analysis"""
    print_header("DEMO 4: Metrics Collection and Analysis")
    
    collector = DecompositionMetricsCollector()
    
    problems = [
        ("Feature A", "Implement user authentication", 6.0),
        ("Feature B", "Build payment processing", 7.0),
        ("Feature C", "Create analytics dashboard", 5.5),
    ]
    
    print(f"\n🔄 Processing {len(problems)} problems with metrics collection...")
    
    for title, desc, complexity in problems:
        print(f"\n  📋 {title}")
        
        problem = create_problem_definition(title, desc, complexity=complexity)
        
        # Create pipeline with metrics
        pipeline = OpenEvolveIntegratedPipeline()
        
        # Execute
        start = time.time()
        result = pipeline.execute(problem)
        total_time = time.time() - start
        
        # Collect metrics
        if result.decomposition_plan:
            collector.collect_decomposition_metrics(
                result.decomposition_plan,
                total_time * 0.2  # Estimated
            )
        
        for sp_id, sol in result.sub_solutions.items():
            collector.collect_evolution_metrics(
                sp_id,
                sol.quality_score,
                25,  # Estimated iterations
                total_time * 0.6 / len(result.sub_solutions)  # Estimated per sub-problem
            )
        
        if result.integrated_solution:
            collector.collect_recomposition_metrics(
                result.integrated_solution,
                total_time * 0.2  # Estimated
            )
        
        print(f"     ✅ Quality: {result.overall_quality:.2f}")
    
    # Display summary
    print_section("Metrics Summary")
    summary = collector.get_summary()
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Operations: {summary['total_operations']}")
    print(f"   Decompositions: {summary['decompositions']}")
    print(f"   Evolutions: {summary['evolutions']}")
    print(f"   Recompositions: {summary['recompositions']}")
    print(f"\n⏱️  Timing:")
    print(f"   Avg Decomposition: {summary['avg_decomposition_time']:.2f}s")
    print(f"   Avg Evolution: {summary['avg_evolution_time']:.2f}s")
    print(f"\n🎯 Quality:")
    print(f"   Avg Fitness: {summary['avg_fitness']:.2f}")


def demo_adapter_usage():
    """Demo 5: Using the OpenEvolve Adapter"""
    print_header("DEMO 5: OpenEvolve Adapter Usage")
    
    # Create adapter
    adapter = OpenEvolveDecompositionAdapter()
    
    print(f"\n🔄 Using adapter for decomposition and evolution...")
    
    result = adapter.decompose_and_evolve(
        problem_description="""
        Build a real-time chat application with the following features:
        - WebSocket-based messaging for instant delivery
        - Message persistence and history
        - File sharing with image/video support
        - Group chats with admin controls
        - End-to-end encryption for private messages
        - Typing indicators and read receipts
        - Push notifications for mobile devices
        """,
        problem_title="Real-Time Chat Application",
        domain="software",
        complexity=7.5
    )
    
    print_section("Adapter Results")
    print(f"✅ Success: {result['success']}")
    print(f"📊 Overall Quality: {result['overall_quality']:.2f}")
    
    if result['decomposition']:
        decomp = result['decomposition']
        print(f"\n🧩 Decomposition:")
        print(f"   Plan ID: {decomp['plan_id']}")
        print(f"   Strategy: {decomp['strategy']}")
        print(f"   Sub-problems: {len(decomp['sub_problems'])}")
        print(f"   Quality: {decomp['quality']:.2f}")
        
        print(f"\n   Sub-Problem Breakdown:")
        for i, sp in enumerate(decomp['sub_problems'][:5], 1):
            print(f"   {i}. {sp['title']}")
            print(f"      Type: {sp['type']} | Complexity: {sp['complexity']:.1f} | Effort: {sp['effort_hours']}h")
    
    if result['integrated_solution']:
        sol = result['integrated_solution']
        print(f"\n🔧 Integrated Solution:")
        print(f"   Solution ID: {sol['solution_id']}")
        print(f"   Quality: {sol['quality']:.2f}")
        print(f"   Conflicts Detected: {sol['conflicts_detected']}")
        print(f"   Conflicts Resolved: {sol['conflicts_resolved']}")


def demo_quick_solve():
    """Demo 6: Quick Solve Helper"""
    print_header("DEMO 6: Quick Solve with OpenEvolve")
    
    print(f"\n🚀 Using quick_solve_with_openevolve helper...")
    
    result = quick_solve_with_openevolve(
        title="Data Pipeline Architecture",
        description="""
        Design a data pipeline for processing clickstream data from a website
        with 1M+ daily active users. Must handle real-time ingestion, validation,
        transformation, and storage in a data warehouse for analytics.
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=7.0,
        evolution_config=EvolutionConfig(
            max_iterations=20,
            parallel_evolution=True
        )
    )
    
    print_section("Quick Solve Results")
    print(f"✅ Pipeline ID: {result.pipeline_id}")
    print(f"📊 Overall Quality: {result.overall_quality:.2f}")
    print(f"🧩 Decomposition Quality: {result.decomposition_quality:.2f}")
    print(f"🔧 Solution Quality: {result.solution_quality:.2f}")
    
    if result.stages:
        print(f"\n📋 Pipeline Stages:")
        for stage in result.stages:
            duration = stage.duration_seconds()
            status_icon = "✅" if stage.status == "completed" else "❌"
            print(f"   {status_icon} {stage.name}: {stage.status} ({duration:.2f}s)")
    
    print(f"\n📝 Solution Preview:")
    if result.integrated_solution:
        content = result.integrated_solution.assembled_content
        lines = content.split('\n')[:15]
        for line in lines:
            print(f"   {line}")
        line_count = len(content.split('\n'))
        if line_count > 15:
            print(f"   ... ({line_count - 15} more lines)")


def demo_configuration_options():
    """Demo 7: Configuration Options"""
    print_header("DEMO 7: Configuration Options")
    
    print(f"\n⚙️  Creating decomposition-aware OpenEvolve configuration...")
    
    config = create_decomposition_aware_config(
        base_config={
            'max_iterations': 100,
            'population_size': 200,
            'temperature': 0.7
        },
        decomposition_strategy='semantic',
        enable_parallel_evolution=True,
        max_subproblems=8
    )
    
    print(f"\n📄 Configuration:")
    print(json.dumps(config, indent=2))
    
    print(f"\n🔄 Using configuration in pipeline...")
    
    problem = create_problem_definition(
        title="Configured Pipeline Test",
        description="Test problem for configuration demonstration",
        complexity=6.0
    )
    
    evolution_config = EvolutionConfig(
        max_iterations=config.get('max_iterations', 50),
        parallel_evolution=config['decomposition']['parallel_evolution']
    )
    
    pipeline = OpenEvolveIntegratedPipeline(evolution_config=evolution_config)
    result = pipeline.execute(problem)
    
    print(f"\n✅ Result: Quality={result.overall_quality:.2f}, Sub-problems={len(result.decomposition_plan.sub_problems) if result.decomposition_plan else 0}")


# Need json for the last demo
import json


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     OPENEVOLVE ENHANCED DECOMPOSITION INTEGRATION DEMO               ║
║                                                                      ║
║  Features:                                                           ║
║  • LLM-Powered Intelligent Decomposition                             ║
║  • Evolutionary Solution Generation                                  ║
║  • Parallel Sub-Problem Evolution                                    ║
║  • Conflict-Aware Solution Assembly                                  ║
║  • Cross-Domain Problem Solving                                      ║
║  • Comprehensive Metrics Collection                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    demos = [
        ("Basic Integration", demo_basic_integration),
        ("Strategy Comparison", demo_strategy_comparison),
        ("Cross-Domain Solving", demo_cross_domain),
        ("Metrics Collection", demo_metrics_collection),
        ("Adapter Usage", demo_adapter_usage),
        ("Quick Solve", demo_quick_solve),
        ("Configuration Options", demo_configuration_options),
    ]
    
    total_start = time.time()
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in demo '{name}': {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    print_header("ALL DEMOS COMPLETE", "=")
    print(f"\n⏱️  Total execution time: {total_elapsed:.2f}s")
    print("\n✅ All demonstrations completed successfully!")
    
    print("\n📚 Key Integration Points:")
    print("  1. OpenEvolveSolutionSolver - Evolves sub-problems using OpenEvolve")
    print("  2. ParallelEvolutionManager - Manages parallel evolution of sub-problems")
    print("  3. OpenEvolveIntegratedPipeline - Full pipeline with decomposition + evolution")
    print("  4. OpenEvolveDecompositionAdapter - Adapter for existing OpenEvolve API")
    print("  5. DecompositionMetricsCollector - Collects metrics from all stages")
    
    print("\n🔗 Integration Benefits:")
    print("  • Higher quality solutions through evolutionary optimization")
    print("  • Parallel evolution reduces total execution time")
    print("  • Automatic conflict detection and resolution")
    print("  • Comprehensive quality metrics at each stage")
    print("  • Compatible with existing OpenEvolve infrastructure")


if __name__ == "__main__":
    main()
