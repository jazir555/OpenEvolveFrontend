# Sovereign-Grade Problem Decomposition System - User Guide

## Introduction

The Sovereign-Grade Problem Decomposition System helps you solve complex problems by intelligently breaking them down into manageable sub-problems with clear success criteria, dependencies, and validation.

## Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from sovereign_persistence import SovereignDatabase; SovereignDatabase().init_database()"
```

### Quick Start

```python
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine

# Analyze your problem
analyzer = ProblemAnalyzer()
problem = analyzer.analyze_problem(
    "Build a recommendation system",
    title="Recommendation System"
)

# Decompose it
engine = DecompositionEngine(analyzer)
plan = engine.decompose(problem)

# View sub-problems
for sp in plan.sub_problems:
    print(f"- {sp.title}: {sp.description}")
```

## Core Concepts

### 1. Problem Analysis

The system analyzes your problem to understand:
- **Domain**: What field does this belong to?
- **Complexity**: How difficult is this problem?
- **Constraints**: What limitations exist?
- **Success Criteria**: How do we know when it's solved?

### 2. Decomposition Strategies

Choose from multiple strategies:

- **Semantic**: Groups related concepts together
- **Dependency**: Orders by prerequisites
- **Complexity**: Balances difficulty across sub-problems
- **Hybrid**: Combines multiple strategies (recommended)

### 3. Validation

Every decomposition is validated through:
- **Gauntlets**: Automated quality checks
- **Quality Metrics**: 6-dimensional scoring
- **Team Review**: Optional AI team validation

### 4. Refinement

If quality isn't sufficient, the system:
- Collects feedback from gauntlets and teams
- Generates improvement suggestions
- Iteratively refines until convergence

## Common Use Cases

### Use Case 1: Research Problem

```python
problem = analyzer.analyze_problem(
    "Research the effectiveness of different neural network architectures for NLP tasks",
    title="NN Architecture Research"
)

plan = engine.decompose(problem, strategy='semantic')

# Research problems get hypothesis-driven sub-problems
# - Literature Review
# - Hypothesis Formation
# - Experimental Design
# - Data Collection and Analysis
```

### Use Case 2: Implementation Problem

```python
problem = analyzer.analyze_problem(
    "Build a REST API with authentication, rate limiting, and caching",
    title="REST API"
)

plan = engine.decompose(problem, strategy='dependency')

# Implementation problems get ordered sub-problems
# - Requirements Analysis
# - Architecture Design
# - Core Implementation
# - Integration and Testing
```

### Use Case 3: Complex System Design

```python
problem = analyzer.analyze_problem(
    "Design a distributed caching system with consistency guarantees",
    title="Distributed Cache"
)

plan = engine.decompose(problem, strategy='hybrid')

# Hybrid strategy combines multiple approaches for optimal decomposition
```

## Advanced Features

### Knowledge Learning

The system learns from successful decompositions:

```python
from sovereign_knowledge_manager import KnowledgeManager

knowledge_mgr = KnowledgeManager()

# After successful decomposition
patterns = knowledge_mgr.extract_patterns(
    plan,
    success=True,
    quality_score=0.85
)

# For future problems
best_strategy = knowledge_mgr.get_best_strategy(problem.problem_type)
```

### Iterative Refinement

Automatically refine until quality thresholds are met:

```python
from sovereign_refinement import RefinementCoordinator

coordinator = RefinementCoordinator()
result = coordinator.track_refinement_cycles(
    plan,
    max_cycles=5,
    convergence_threshold=0.01
)

if result['converged']:
    print(f"Converged after {result['total_cycles']} cycles")
    print(f"Final quality: {result['final_quality']:.2f}")
```

### Performance Optimization

Cache expensive operations:

```python
from sovereign_performance_optimization import cached

@cached("problem_analysis")
def analyze_expensive_problem(text):
    return analyzer.analyze_problem(text)
```

Monitor performance:

```python
from sovereign_performance_optimization import timed, get_performance_stats

@timed("decomposition")
def decompose_problem(problem):
    return engine.decompose(problem)

# Later, check stats
stats = get_performance_stats()
print(f"Average decomposition time: {stats['decomposition']['avg_duration']:.2f}s")
```

## UI Components

### Streamlit Integration

```python
import streamlit as st
from sovereign_ui_components import (
    render_problem_input_form,
    render_decomposition_plan,
    render_quality_dashboard
)

# Problem input
problem_data = render_problem_input_form()

if problem_data:
    problem = analyzer.analyze_problem(
        problem_data['description'],
        title=problem_data['title']
    )
    
    plan = engine.decompose(problem)
    
    # Display results
    render_decomposition_plan(plan)
    
    report = assessor.generate_quality_report(plan)
    render_quality_dashboard(report)
```

### Sidebar Controls

```python
from sovereign_sidebar_integration import render_sovereign_sidebar

# Add to sidebar
sidebar_input = render_sovereign_sidebar()

if sidebar_input and sidebar_input['mode'] == 'quick':
    problem = analyzer.analyze_problem(sidebar_input['problem_text'])
    plan = engine.decompose(problem, strategy=sidebar_input['strategy'])
```

## Best Practices

### 1. Problem Description

Write clear, detailed problem descriptions:

✅ Good:
```
"Build a recommendation system that uses collaborative filtering and content-based 
algorithms to suggest products. The system must handle 10,000 requests per second 
and provide sub-100ms response times."
```

❌ Bad:
```
"Make a recommender"
```

### 2. Strategy Selection

- Use `hybrid` for complex, multi-faceted problems
- Use `semantic` for concept-heavy problems
- Use `dependency` for sequential workflows
- Use `complexity` when balancing workload is critical

### 3. Quality Thresholds

Adjust thresholds based on your needs:

```python
assessor = QualityAssessor()
assessor.thresholds = {
    'coherence': 0.80,      # Higher for critical systems
    'completeness': 0.85,   # Higher for comprehensive coverage
    'feasibility': 0.75,    # Lower if exploring new territory
    'overall': 0.80
}
```

### 4. Refinement Cycles

Balance quality vs. time:

```python
# Quick iteration
result = coordinator.track_refinement_cycles(plan, max_cycles=2)

# Thorough refinement
result = coordinator.track_refinement_cycles(plan, max_cycles=10)
```

## Troubleshooting

### Problem: Low Quality Scores

**Solution**: 
1. Provide more detailed problem description
2. Increase refinement cycles
3. Try different decomposition strategy

### Problem: Too Many Sub-Problems

**Solution**:
1. Use `complexity` strategy to balance
2. Increase complexity threshold
3. Merge related sub-problems manually

### Problem: Circular Dependencies

**Solution**:
1. System automatically detects these
2. Review dependency gauntlet feedback
3. Refine decomposition to remove cycles

### Problem: Slow Performance

**Solution**:
1. Enable caching: `@cached` decorator
2. Use lazy loading for large datasets
3. Check performance stats: `get_performance_stats()`

## Examples

### Example 1: Machine Learning Pipeline

```python
problem = analyzer.analyze_problem("""
Build an end-to-end machine learning pipeline that:
- Ingests data from multiple sources
- Performs feature engineering
- Trains multiple models
- Evaluates and selects best model
- Deploys to production with monitoring
""", title="ML Pipeline")

plan = engine.decompose(problem, strategy='hybrid')

# Result: 5-7 sub-problems covering data, training, deployment
```

### Example 2: Microservices Architecture

```python
problem = analyzer.analyze_problem("""
Design a microservices architecture for an e-commerce platform with:
- User service (authentication, profiles)
- Product catalog service
- Order management service
- Payment processing service
- Notification service
All services must communicate via message queue and have independent databases.
""", title="E-commerce Microservices")

plan = engine.decompose(problem, strategy='dependency')

# Result: Ordered sub-problems from infrastructure to services
```

## Support

For issues or questions:
1. Check API documentation: `SOVEREIGN_API_DOCUMENTATION.md`
2. Review test examples: `test_sovereign_integration.py`
3. Check system health: `HealthMonitor().run_health_checks()`

## Next Steps

1. Try the quick start example
2. Experiment with different strategies
3. Explore advanced features (refinement, knowledge learning)
4. Integrate with your UI using provided components
5. Monitor performance and optimize as needed

Happy problem solving!
