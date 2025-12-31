"""
Example usage of Sovereign Integration System
Demonstrates real-world problem decomposition workflows.
"""

from sovereign_integration import (
    SovereignIntegrationOrchestrator,
    quick_decompose,
    decompose_with_strategy,
    full_solution_workflow
)


def example_1_quick_decomposition():
    """Example 1: Quick decomposition with defaults."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Decomposition")
    print("="*70)
    
    problem = """
    Build a recommendation system that:
    - Ingests user behavior data in real-time
    - Trains ML models with collaborative filtering
    - Serves recommendations with <100ms latency
    - Handles 10M+ users
    - Provides A/B testing capabilities
    """
    
    result = quick_decompose(problem, "Recommendation System")
    
    print(f"\nSuccess: {result.success}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Sub-problems: {len(result.final_plan.sub_problems)}")
    print(f"Refinement Cycles: {result.refinement_cycles}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    print("\nSub-problems created:")
    for i, sp in enumerate(result.final_plan.sub_problems, 1):
        print(f"  {i}. {sp.title}")
        print(f"     Type: {sp.type.value}, Priority: {sp.priority}, Effort: {sp.estimated_effort}h")


def example_2_strategy_comparison():
    """Example 2: Compare different strategies."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Strategy Comparison")
    print("="*70)
    
    problem = "Design a distributed caching system with Redis and Memcached"
    
    strategies = ['semantic', 'dependency', 'complexity', 'hybrid']
    results = {}
    
    for strategy in strategies:
        print(f"\nTrying {strategy} strategy...")
        result = decompose_with_strategy(problem, strategy, "Cache System")
        results[strategy] = result
        print(f"  Quality: {result.quality_score:.2f}, Sub-problems: {len(result.final_plan.sub_problems)}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].quality_score)
    print(f"\nBest strategy: {best_strategy[0]} (quality: {best_strategy[1].quality_score:.2f})")


def example_3_full_workflow():
    """Example 3: Complete solution workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Full Solution Workflow")
    print("="*70)
    
    problem = """
    Create a data pipeline that:
    - Extracts data from multiple sources (APIs, databases, files)
    - Transforms and validates data
    - Loads into data warehouse
    - Provides monitoring and alerting
    """
    
    result = full_solution_workflow(problem, "Data Pipeline", strategy='hybrid')
    
    print(f"\nWorkflow Success: {result['success']}")
    print(f"\nDecomposition:")
    print(f"  Strategy: {result['decomposition']['strategy']}")
    print(f"  Sub-problems: {result['decomposition']['sub_problem_count']}")
    print(f"  Quality: {result['decomposition']['quality_score']:.2f}")
    print(f"  Refinement Cycles: {result['decomposition']['refinement_cycles']}")
    
    print(f"\nSolutions:")
    print(f"  Total: {result['solutions']['total']}")
    print(f"  Successful: {result['solutions']['successful']}")
    print(f"  Pending: {result['solutions']['pending']}")
    
    print(f"\nFinal Quality:")
    print(f"  Overall: {result['final_quality']['overall_score']:.2f}")
    print(f"  Coherence: {result['final_quality']['coherence']:.2f}")
    print(f"  Completeness: {result['final_quality']['completeness']:.2f}")
    print(f"  Feasibility: {result['final_quality']['feasibility']:.2f}")
    print(f"  Meets Thresholds: {result['final_quality']['meets_thresholds']}")


def example_4_with_refinement():
    """Example 4: Workflow with refinement."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Workflow with Refinement")
    print("="*70)
    
    orchestrator = SovereignIntegrationOrchestrator()
    
    problem = "Optimize database queries for a high-traffic e-commerce site"
    
    result = orchestrator.run_complete_workflow(
        problem,
        title="DB Optimization",
        strategy='hybrid',
        max_refinement_cycles=5,  # Allow up to 5 refinement cycles
        enable_knowledge_extraction=True
    )
    
    print(f"\nSuccess: {result.success}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Refinement Cycles Used: {result.refinement_cycles}")
    print(f"Knowledge Extracted: {result.knowledge_extracted}")
    
    if result.refinement_cycles > 0:
        print(f"\nRefinement improved the decomposition through {result.refinement_cycles} iterations")


def example_5_pattern_application():
    """Example 5: Apply learned patterns."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Pattern Application")
    print("="*70)
    
    orchestrator = SovereignIntegrationOrchestrator()
    
    # First, decompose a problem to create patterns
    problem1 = "Build a REST API with authentication and rate limiting"
    result1 = orchestrator.run_complete_workflow(
        problem1,
        title="REST API v1",
        strategy='semantic',
        enable_knowledge_extraction=True
    )
    print(f"First decomposition: Quality={result1.quality_score:.2f}, Patterns extracted={result1.knowledge_extracted}")
    
    # Now try a similar problem - should use learned patterns
    problem2 = "Build a GraphQL API with authentication and caching"
    
    # Get strategy recommendation
    from problem_analyzer import ProblemAnalyzer
    analyzer = ProblemAnalyzer()
    problem_def = analyzer.analyze_problem(problem2, "GraphQL API")
    
    recommendation = orchestrator.get_strategy_recommendation(problem_def)
    print(f"\nStrategy recommendation for similar problem:")
    print(f"  Strategy: {recommendation['recommended_strategy']}")
    print(f"  Confidence: {recommendation['confidence']}")
    print(f"  Reasoning: {recommendation['reasoning']}")
    
    # Apply learned patterns
    pattern_guidance = orchestrator.apply_learned_patterns(problem_def)
    if pattern_guidance:
        print(f"\nApplied learned pattern:")
        print(f"  Success Rate: {pattern_guidance['success_rate']:.2f}")
        print(f"  Avg Quality: {pattern_guidance['avg_quality']:.2f}")
        print(f"  Recommendations: {len(pattern_guidance['recommendations'])}")


def example_6_dependency_analysis():
    """Example 6: Dependency graph analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Dependency Analysis")
    print("="*70)
    
    orchestrator = SovereignIntegrationOrchestrator()
    
    problem = """
    Implement a microservices architecture with:
    - User service
    - Product service
    - Order service
    - Payment service
    - Notification service
    """
    
    result = orchestrator.run_complete_workflow(
        problem,
        title="Microservices",
        strategy='dependency'  # Use dependency strategy
    )
    
    plan = result.final_plan
    
    if plan.dependency_graph:
        print(f"\nDependency Graph Analysis:")
        print(f"  Nodes: {len(plan.dependency_graph.nodes)}")
        print(f"  Critical Path Length: {len(plan.dependency_graph.critical_path)}")
        print(f"  Parallel Groups: {len(plan.dependency_graph.parallel_groups)}")
        
        print(f"\nCritical Path:")
        for node_id in plan.dependency_graph.critical_path:
            node = plan.dependency_graph.nodes[node_id]
            print(f"  → {node.title} ({node.estimated_effort}h)")
        
        print(f"\nParallel Opportunities:")
        for i, group in enumerate(plan.dependency_graph.parallel_groups, 1):
            print(f"  Group {i}: {len(group)} tasks can run in parallel")
            for node_id in group:
                node = plan.dependency_graph.nodes[node_id]
                print(f"    - {node.title}")
        
        print(f"\nExecution Order:")
        for i, node_id in enumerate(plan.dependency_graph.execution_order, 1):
            node = plan.dependency_graph.nodes[node_id]
            print(f"  {i}. {node.title}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("SOVEREIGN INTEGRATION SYSTEM - USAGE EXAMPLES")
    print("="*70)
    
    try:
        example_1_quick_decomposition()
        example_2_strategy_comparison()
        example_3_full_workflow()
        example_4_with_refinement()
        example_5_pattern_application()
        example_6_dependency_analysis()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
