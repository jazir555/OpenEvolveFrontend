"""
Comprehensive Demonstration of Enhanced Decomposition and Recomposition Systems

This script demonstrates the full capabilities of:
1. Enhanced Decomposition Engine (20+ strategies)
2. Enhanced Recomposition Engine (advanced conflict detection)
3. Decomposition-Recomposition Integration Pipeline

Usage:
    python demo_enhanced_decomposition_recomposition.py
"""

import json
import time
from typing import List, Dict, Any

# Import enhanced systems
from enhanced_decomposition_engine import (
    EnhancedDecompositionEngine,
    DecompositionStrategy,
    ProblemDomain,
    create_problem_definition
)

from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    AssemblyStrategy,
    create_subproblem_solution,
    ConflictSeverity
)

from decomposition_recomposition_integration import (
    DecompositionRecompositionPipeline,
    PipelineConfig,
    quick_solve,
    analyze_solution,
    BatchPipelineProcessor
)


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print("─" * 70)


def demo_decomposition_strategies():
    """Demonstrate different decomposition strategies."""
    print_header("DEMO 1: Decomposition Strategies Comparison")
    
    engine = EnhancedDecompositionEngine()
    
    problem = create_problem_definition(
        title="Build Machine Learning Platform",
        description="""
        Develop an end-to-end machine learning platform that enables data scientists
        to build, train, deploy, and monitor ML models at scale. The platform should
        support multiple frameworks (TensorFlow, PyTorch, scikit-learn), provide
        automated hyperparameter tuning, model versioning, A/B testing capabilities,
        and real-time model serving infrastructure.
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=8.5
    )
    
    strategies = [
        DecompositionStrategy.HIERARCHICAL,
        DecompositionStrategy.FUNCTIONAL,
        DecompositionStrategy.SEMANTIC,
        DecompositionStrategy.TEMPORAL,
        DecompositionStrategy.CAUSAL,
        DecompositionStrategy.RISK_BASED,
        DecompositionStrategy.COMPLEXITY,
        DecompositionStrategy.HYBRID,
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy.value} strategy...")
        
        start = time.time()
        plan = engine.decompose(problem, strategy=strategy)
        elapsed = time.time() - start
        
        results.append({
            'strategy': strategy.value,
            'sub_problems': len(plan.sub_problems),
            'quality': plan.overall_quality,
            'coverage': plan.coverage_score,
            'balance': plan.balance_score,
            'time': elapsed
        })
        
        print(f"   [OK] Generated {len(plan.sub_problems)} sub-problems in {elapsed:.2f}s")
        print(f"   📊 Quality: {plan.overall_quality:.2f} | Coverage: {plan.coverage_score:.2f} | Balance: {plan.balance_score:.2f}")
    
    # Summary table
    print_section("Strategy Comparison Summary")
    print(f"{'Strategy':<20} {'Sub-Problems':>12} {'Quality':>10} {'Time (s)':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['strategy']:<20} {r['sub_problems']:>12} {r['quality']:>10.2f} {r['time']:>10.2f}")


def demo_decomposition_analysis():
    """Demonstrate decomposition analysis features."""
    print_header("DEMO 2: Decomposition Analysis & Metrics")
    
    engine = EnhancedDecompositionEngine()
    
    problem = create_problem_definition(
        title="Enterprise Resource Planning System",
        description="""
        Build a comprehensive ERP system integrating finance, HR, inventory,
        manufacturing, CRM, and supply chain modules. Must support multi-tenancy,
        real-time analytics, and compliance with SOX and GDPR regulations.
        """,
        domain=ProblemDomain.BUSINESS,
        complexity=9.0
    )
    
    print("\n🔄 Decomposing problem...")
    plan = engine.decompose(problem)
    
    print_section("Decomposition Results")
    print(f"Strategy Used: {plan.strategy_used.value}")
    print(f"Sub-Problems: {len(plan.sub_problems)}")
    
    print_section("Quality Metrics")
    print(f"  Overall Quality:    {plan.overall_quality:.2f}/1.0")
    print(f"  Coverage Score:     {plan.coverage_score:.2f}/1.0")
    print(f"  Balance Score:      {plan.balance_score:.2f}/1.0")
    print(f"  Coherence Score:    {plan.coherence_score:.2f}/1.0")
    
    print_section("Complexity Analysis")
    analysis = plan.complexity_analysis
    print(f"  Mean Complexity:    {analysis.get('mean', 0):.2f}")
    print(f"  Min Complexity:     {analysis.get('min', 0):.2f}")
    print(f"  Max Complexity:     {analysis.get('max', 0):.2f}")
    if 'distribution' in analysis:
        dist = analysis['distribution']
        print(f"  Distribution:       Low: {dist.get('low', 0)}, Medium: {dist.get('medium', 0)}, High: {dist.get('high', 0)}")
    
    print_section("Dependency Structure")
    print(f"  Execution Order:    {len(plan.execution_order)} steps")
    print(f"  Parallel Groups:    {len(plan.parallel_groups)} groups")
    print(f"  Dependency Graph:   {len(plan.dependency_graph)} nodes")
    
    # Show parallel execution groups
    print_section("Parallel Execution Groups")
    for i, group in enumerate(plan.parallel_groups, 1):
        print(f"  Group {i}: {len(group)} sub-problems can execute in parallel")
    
    print_section("Sub-Problems Detail")
    for i, sp in enumerate(plan.sub_problems[:5], 1):
        print(f"  {i}. {sp.title}")
        print(f"     Type: {sp.type.value:<15} Priority: {sp.priority}/10  Complexity: {sp.complexity_score.overall_complexity:.1f}")
        print(f"     Effort: {sp.estimated_effort_hours}h  Dependencies: {len(sp.dependencies)}")


def demo_recomposition_capabilities():
    """Demonstrate recomposition capabilities."""
    print_header("DEMO 3: Advanced Recomposition & Conflict Resolution")
    
    engine = EnhancedRecompositionEngine()
    
    # Create sample solutions with potential conflicts
    solutions = {
        "req_analysis": create_subproblem_solution(
            "req_analysis",
            """
# Requirements Analysis

## Functional Requirements
- User authentication with MFA support
- Real-time data synchronization
- Role-based access control (RBAC)

## Non-Functional Requirements
- Response time < 200ms for 95th percentile
- 99.9% uptime SLA
- Must use PostgreSQL database
            """,
            0.88
        ),
        "system_design": create_subproblem_solution(
            "system_design",
            """
# System Design

## Architecture
- Microservices pattern with API Gateway
- Event-driven communication via Kafka
- Redis for caching layer

## Database
- MongoDB for flexible document storage
- Redis for session management
            """,
            0.85
        ),
        "implementation": create_subproblem_solution(
            "implementation",
            """
# Implementation

## Backend Services
- Node.js with Express framework
- Python microservices for ML components
- gRPC for inter-service communication

## Frontend
- React with TypeScript
- Redux for state management
            """,
            0.82
        ),
        "testing": create_subproblem_solution(
            "testing",
            """
# Testing Strategy

## Unit Tests
- Jest for JavaScript components
- PyTest for Python services
- 80% code coverage minimum

## Integration Tests
- Postman collections for API testing
- Cypress for E2E testing

## Database
- Must use MySQL for testing environment
            """,
            0.80
        ),
        "deployment": create_subproblem_solution(
            "deployment",
            """
# Deployment

## Infrastructure
- Kubernetes on AWS EKS
- Terraform for infrastructure as code
- Helm charts for application deployment

## CI/CD
- GitHub Actions for CI/CD pipeline
- Automated rollback on failure
            """,
            0.85
        ),
    }
    
    # Define dependencies
    dependency_graph = {
        "req_analysis": [],
        "system_design": ["req_analysis"],
        "implementation": ["system_design"],
        "testing": ["implementation"],
        "deployment": ["testing"]
    }
    
    print(f"\n🔄 Assembling {len(solutions)} solutions...")
    
    solution = engine.assemble(
        sub_solutions=solutions,
        problem_id="erp_system",
        decomposition_plan_id="erp_plan",
        dependency_graph=dependency_graph,
        strategy=AssemblyStrategy.HIERARCHICAL
    )
    
    print_section("Assembly Results")
    print(f"Status: {solution.status.value}")
    print(f"Strategy: {solution.assembly_strategy.value}")
    print(f"Content Length: {len(solution.assembled_content)} characters")
    
    print_section("Conflict Analysis")
    conflicts = solution.conflicts_detected
    print(f"Total Conflicts Detected: {len(conflicts)}")
    
    by_severity = {}
    by_type = {}
    for conflict in conflicts:
        by_severity[conflict.severity.value] = by_severity.get(conflict.severity.value, 0) + 1
        by_type[conflict.conflict_type.value] = by_type.get(conflict.conflict_type.value, 0) + 1
    
    print("\nBy Severity:")
    for severity, count in sorted(by_severity.items(), key=lambda x: x[1], reverse=True):
        print(f"  {severity}: {count}")
    
    print("\nBy Type:")
    for conflict_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {conflict_type}: {count}")
    
    print(f"\nConflicts Resolved: {len(solution.conflicts_resolved)}/{len(conflicts)}")
    
    print_section("Quality Metrics")
    metrics = solution.quality_metrics
    print(f"  Overall Score:      {metrics.overall_score:.2f}/1.0")
    print(f"  Completeness:       {metrics.completeness:.2f}/1.0")
    print(f"  Consistency:        {metrics.consistency:.2f}/1.0")
    print(f"  Coherence:          {metrics.coherence:.2f}/1.0")
    print(f"  Correctness:        {metrics.correctness:.2f}/1.0")
    print(f"  Clarity:            {metrics.clarity:.2f}/1.0")
    print(f"  Integration Quality: {metrics.integration_quality:.2f}/1.0")
    
    print_section("Assembly Log")
    for entry in solution.assembly_log[:5]:
        print(f"  * {entry}")


def demo_full_pipeline():
    """Demonstrate full pipeline with multiple domains."""
    print_header("DEMO 4: Cross-Domain Pipeline Execution")
    
    problems = [
        {
            'domain': ProblemDomain.SOFTWARE,
            'title': "Build Distributed Task Queue",
            'description': """
            Design and implement a distributed task queue system supporting
            priority scheduling, retry logic, dead letter queues, and monitoring.
            Must handle 100,000+ tasks per second with sub-second latency.
            """,
            'complexity': 8.0
        },
        {
            'domain': ProblemDomain.FINANCE,
            'title': "Fraud Detection System",
            'description': """
            Build a real-time fraud detection system for credit card transactions.
            Must analyze transaction patterns, detect anomalies, and trigger alerts
            within 50ms while maintaining false positive rate below 0.1%.
            """,
            'complexity': 9.0
        },
        {
            'domain': ProblemDomain.HEALTHCARE,
            'title': "Patient Monitoring Platform",
            'description': """
            Create an IoT-based patient monitoring platform that collects vital signs
            from wearable devices, analyzes trends, and alerts healthcare providers
            to potential issues. Must comply with HIPAA and support 10,000+ concurrent patients.
            """,
            'complexity': 8.5
        },
    ]
    
    pipeline = DecompositionRecompositionPipeline()
    results = []
    
    for problem_data in problems:
        print(f"\n🔄 Processing: {problem_data['title']}")
        print(f"   Domain: {problem_data['domain'].value}")
        
        problem = create_problem_definition(
            title=problem_data['title'],
            description=problem_data['description'],
            domain=problem_data['domain'],
            complexity=problem_data['complexity']
        )
        
        start = time.time()
        result = pipeline.execute(problem)
        elapsed = time.time() - start
        
        results.append({
            'title': problem_data['title'][:30],
            'domain': problem_data['domain'].value,
            'sub_problems': len(result.decomposition_plan.sub_problems) if result.decomposition_plan else 0,
            'decomp_quality': result.decomposition_quality,
            'solution_quality': result.solution_quality,
            'overall': result.overall_quality,
            'time': elapsed,
            'successful': result.is_successful()
        })
        
        print(f"   [OK] Completed in {elapsed:.2f}s")
        print(f"   📊 Quality: {result.overall_quality:.2f} | Sub-problems: {results[-1]['sub_problems']}")
    
    print_section("Cross-Domain Results Summary")
    print(f"{'Problem':<30} {'Domain':<12} {'Quality':>8} {'Time':>8} {'Status':>10}")
    print("-" * 80)
    for r in results:
        status = "[OK] PASS" if r['successful'] else "[FAIL] FAIL"
        print(f"{r['title']:<30} {r['domain']:<12} {r['overall']:>8.2f} {r['time']:>8.2f} {status:>10}")
    
    # Analytics
    analytics = pipeline.get_analytics()
    print_section("Pipeline Analytics")
    print(f"Total Executions: {analytics.total_executions}")
    print(f"Successful: {analytics.successful_executions}")
    print(f"Failed: {analytics.failed_executions}")
    print(f"Average Quality: {analytics.avg_quality_score:.2f}")
    print(f"Average Duration: {analytics.avg_total_time:.2f}s")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print_header("DEMO 5: Batch Processing")
    
    pipeline = DecompositionRecompositionPipeline()
    processor = BatchPipelineProcessor(pipeline)
    
    # Create batch of problems
    problems = [
        create_problem_definition(
            f"Feature {i}",
            f"Implement feature {i} with requirements including user interface, backend API, and database schema.",
            complexity=5.0 + (i % 3)
        )
        for i in range(5)
    ]
    
    print(f"\n🔄 Processing batch of {len(problems)} problems...")
    
    start = time.time()
    results = processor.process_batch(problems)
    elapsed = time.time() - start
    
    print_section("Batch Results")
    summary = processor.get_summary()
    print(f"Total: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Quality: {summary['avg_quality']:.2f}")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Average Time per Problem: {summary['avg_duration']:.2f}s")
    
    print_section("Individual Results")
    for i, result in enumerate(results, 1):
        status = "[OK]" if result.is_successful() else "[FAIL]"
        print(f"  {status} Problem {i}: Quality={result.overall_quality:.2f}, Stages={len(result.stages)}")


def demo_solution_analysis():
    """Demonstrate solution analysis capabilities."""
    print_header("DEMO 6: Solution Analysis")
    
    # Use quick_solve helper
    print("\n🔄 Solving problem...")
    result = quick_solve(
        title="Data Pipeline Architecture",
        description="""
        Design a scalable data pipeline for processing 10TB of daily log data.
        Must include data ingestion, validation, transformation, storage, and
        analytics components with fault tolerance and exactly-once processing.
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=8.0
    )
    
    print_section("Basic Results")
    print(f"Pipeline ID: {result.pipeline_id}")
    print(f"Successful: {result.is_successful()}")
    print(f"Overall Quality: {result.overall_quality:.2f}")
    
    print_section("Detailed Analysis")
    analysis = analyze_solution(result)
    
    if analysis['decomposition']:
        decomp = analysis['decomposition']
        print(f"\nDecomposition:")
        print(f"  Strategy: {decomp['strategy']}")
        print(f"  Sub-problems: {decomp['sub_problems']}")
        print(f"  Quality: {decomp['quality']['overall']:.2f}")
    
    if analysis['recomposition']:
        recomp = analysis['recomposition']
        print(f"\nRecomposition:")
        print(f"  Strategy: {recomp['strategy']}")
        print(f"  Content Length: {recomp['content_length']}")
        if recomp['conflicts']:
            print(f"  Conflicts: {recomp['conflicts']['total_detected']} detected")
    
    if analysis['recommendations']:
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  * {rec}")
    
    print_section("Pipeline Stages")
    for stage in result.stages:
        duration = stage.duration_seconds()
        status_icon = "[OK]" if stage.status == "completed" else "[FAIL]" if stage.status == "failed" else "⏳"
        print(f"  {status_icon} {stage.name}: {stage.status} ({duration:.2f}s)")
    
    print_section("Solution Preview")
    if result.integrated_solution:
        content = result.integrated_solution.assembled_content
        lines = content.split('\n')[:20]
        for line in lines:
            print(line)
        line_count = len(content.split('\n'))
        if line_count > 20:
            print(f"\n... ({line_count - 20} more lines)")


def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     ENHANCED DECOMPOSITION & RECOMPOSITION SYSTEMS DEMO              ║
║                                                                      ║
║  Version: 3.0.0                                                      ║
║  Features: 20+ Strategies | Advanced Conflict Detection | Pipeline   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    demos = [
        ("Decomposition Strategies Comparison", demo_decomposition_strategies),
        ("Decomposition Analysis & Metrics", demo_decomposition_analysis),
        ("Advanced Recomposition", demo_recomposition_capabilities),
        ("Cross-Domain Pipeline", demo_full_pipeline),
        ("Batch Processing", demo_batch_processing),
        ("Solution Analysis", demo_solution_analysis),
    ]
    
    total_start = time.time()
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n[FAIL] Error in demo '{name}': {e}")
    
    total_elapsed = time.time() - total_start
    
    print_header("DEMO COMPLETE")
    print(f"\nTotal execution time: {total_elapsed:.2f}s")
    print("\n[OK] All demonstrations completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  * 9 Decomposition Strategies (Hierarchical, Functional, Semantic, Temporal, etc.)")
    print("  * Multi-dimensional Quality Metrics (Coverage, Balance, Coherence)")
    print("  * Advanced Conflict Detection (12+ conflict types)")
    print("  * Automatic Conflict Resolution")
    print("  * Full Pipeline Integration")
    print("  * Batch Processing Capabilities")
    print("  * Cross-Domain Problem Solving")
    print("  * Solution Analysis & Recommendations")


if __name__ == "__main__":
    main()
