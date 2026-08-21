# Sovereign-Grade Problem Decomposition System - API Documentation

> **STATUS: partially implemented.** Documents the Decomposition System API. `problem_analyzer.py` (`ProblemAnalyzer`) and `DecompositionEngine` (e.g. `engines/decomposition/decomposition_engine_adaptive_enhancement.py`) exist and are served by `engines/other/api_server.py` (port 8001, the Decomposition-Workflow server — NOT the BubbleLab integration backend at `core-projects/BubbleLab/services/openevolve-api`, port 8000).
> **Last reconciled: 2026-08-20**

## Overview

The Sovereign-Grade Problem Decomposition System provides a comprehensive API for analyzing, decomposing, validating, and solving complex problems through intelligent decomposition strategies.

## Core Components

### 1. Problem Analyzer

**Module**: `problem_analyzer.py`

#### ProblemAnalyzer

Analyzes problems to extract semantic information and structure.

```python
from problem_analyzer import ProblemAnalyzer

analyzer = ProblemAnalyzer()
problem = analyzer.analyze_problem(
    problem_text="Build a recommendation system with ML models",
    title="Recommendation System"
)
```

**Methods**:
- `analyze_problem(problem_text: str, title: str) -> ProblemDefinition`
  - Performs comprehensive problem analysis
  - Returns: ProblemDefinition with domain, complexity, constraints

### 2. Decomposition Engine

**Module**: `decomposition_engine.py`

#### DecompositionEngine

Orchestrates problem decomposition using multiple strategies.

```python
from decomposition_engine import DecompositionEngine

engine = DecompositionEngine(analyzer)
plan = engine.decompose(problem, strategy='hybrid')
```

**Strategies Available**:
- `semantic`: Concept-based decomposition
- `dependency`: Prerequisite-based decomposition
- `complexity`: Complexity-balanced decomposition
- `hybrid`: Multi-strategy combination

**Methods**:
- `decompose(problem: ProblemDefinition, strategy: str = 'auto') -> DecompositionPlan`
  - Decomposes problem using specified strategy
  - Returns: Complete decomposition plan with sub-problems

### 3. Gauntlet System

**Module**: `sovereign_gauntlets.py`

#### GauntletSystem

Validates decomposition quality through specialized gauntlets.

```python
from sovereign_gauntlets import GauntletSystem

gauntlet_system = GauntletSystem()
results = gauntlet_system.run_decomposition_gauntlets(plan)
```

**Gauntlets**:
- `CoherenceGauntlet`: Logical consistency
- `CompletenessGauntlet`: Problem coverage
- `FeasibilityGauntlet`: Solvability assessment
- `DependencyGauntlet`: Graph validation

**Methods**:
- `run_decomposition_gauntlets(plan: DecompositionPlan) -> Dict[str, ValidationResult]`
- `process_gauntlet_feedback(results: Dict) -> List[Feedback]`
- `get_overall_quality(results: Dict) -> float`

### 4. Quality Assessment

**Module**: `sovereign_quality_assessment.py`

#### QualityAssessor

Assesses decomposition and solution quality.

```python
from sovereign_quality_assessment import QualityAssessor

assessor = QualityAssessor()
report = assessor.generate_quality_report(plan)
```

**Metrics**:
- Coherence (0-1)
- Completeness (0-1)
- Feasibility (0-1)
- Integration (0-1)
- Balance (0-1)
- Clarity (0-1)

**Methods**:
- `generate_quality_report(plan: DecompositionPlan) -> QualityReport`
- `check_quality_thresholds(scores: QualityScores) -> bool`

### 5. Refinement Coordinator

**Module**: `sovereign_refinement.py`

#### RefinementCoordinator

Coordinates iterative refinement with feedback loops.

```python
from sovereign_refinement import RefinementCoordinator

coordinator = RefinementCoordinator()
result = coordinator.track_refinement_cycles(
    plan,
    max_cycles=5,
    convergence_threshold=0.01
)
```

**Methods**:
- `process_feedback(plan, feedback_list) -> Dict`
- `generate_refinement_plan(plan, feedback) -> RefinementPlan`
- `execute_refinement(plan, refinement_plan) -> Tuple[DecompositionPlan, RefinementMetrics]`
- `track_refinement_cycles(plan, max_cycles, convergence_threshold) -> Dict`

### 6. Knowledge Manager

**Module**: `sovereign_knowledge_manager.py`

#### KnowledgeManager

Manages knowledge extraction and pattern learning.

```python
from sovereign_knowledge_manager import KnowledgeManager

knowledge_mgr = KnowledgeManager()
patterns = knowledge_mgr.extract_patterns(plan, success=True, quality_score=0.85)
best_strategy = knowledge_mgr.get_best_strategy(problem_type)
```

**Methods**:
- `extract_patterns(plan, success, quality_score) -> List[Pattern]`
- `retrieve_patterns(problem_type, domain, min_success_rate) -> List[Pattern]`
- `get_best_strategy(problem_type, domain) -> DecompositionStrategy`
- `track_strategy_performance(strategy, quality_score) -> None`

### 7. Performance Optimization

**Module**: `sovereign_performance_optimization.py`

#### PerformanceCache

High-performance caching layer.

```python
from sovereign_performance_optimization import PerformanceCache, cached

cache = PerformanceCache(max_size=1000, ttl_seconds=3600)
cache.set("key", "value")
value = cache.get("key")

# Or use decorator
@cached("my_function")
def expensive_function(x):
    return x * 2
```

#### PerformanceMonitor

Tracks system performance.

```python
from sovereign_performance_optimization import PerformanceMonitor, timed

monitor = PerformanceMonitor()

@timed("operation_name")
def my_operation():
    pass

stats = monitor.get_stats("operation_name")
```

### 8. Reliability

**Module**: `sovereign_reliability.py`

#### Error Handling

```python
from sovereign_reliability import with_retry, ErrorHandler

@with_retry(max_attempts=3, retry_on=(ValueError,))
def unreliable_operation():
    pass

handler = ErrorHandler()
handler.handle_error(exception, context={'operation': 'decompose'})
```

## API Reference

... (Content of API Reference) ...

---

## Real-World Examples

This section will contain practical examples demonstrating how to use the Sovereign API in various real-world scenarios.

---

## Troubleshooting

... (Content of Troubleshooting) ...
## Complete Workflow Example

```python
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_gauntlets import GauntletSystem
from sovereign_quality_assessment import QualityAssessor
from sovereign_refinement import RefinementCoordinator
from sovereign_knowledge_manager import KnowledgeManager

# Step 1: Analyze Problem
analyzer = ProblemAnalyzer()
problem = analyzer.analyze_problem(
    "Build a distributed ML system with real-time inference",
    title="ML System"
)

# Step 2: Get Best Strategy from Knowledge Base
knowledge_mgr = KnowledgeManager()
best_strategy = knowledge_mgr.get_best_strategy(problem.problem_type)

# Step 3: Decompose Problem
engine = DecompositionEngine(analyzer)
plan = engine.decompose(problem, strategy=best_strategy or 'hybrid')

# Step 4: Validate with Gauntlets
gauntlet_system = GauntletSystem()
results = gauntlet_system.run_decomposition_gauntlets(plan)

# Step 5: Assess Quality
assessor = QualityAssessor()
report = assessor.generate_quality_report(plan)

# Step 6: Refine if Needed
if not report.meets_thresholds:
    coordinator = RefinementCoordinator()
    feedback = gauntlet_system.process_gauntlet_feedback(results)
    refinement_result = coordinator.track_refinement_cycles(
        plan,
        max_cycles=5,
        convergence_threshold=0.01
    )
    plan = refinement_result['final_plan']

# Step 7: Extract and Store Knowledge
if report.meets_thresholds:
    patterns = knowledge_mgr.extract_patterns(
        plan,
        success=True,
        quality_score=report.metrics.overall_score
    )

print(f"Decomposition complete: {len(plan.sub_problems)} sub-problems")
print(f"Quality score: {report.metrics.overall_score:.2f}")
```

## Data Models

### ProblemDefinition

```python
@dataclass
class ProblemDefinition:
    id: str
    title: str
    description: str
    problem_type: ProblemType
    domain_context: DomainContext
    complexity_score: ComplexityScore
    constraints: List[Constraint]
    success_criteria: List[SuccessCriterion]
    # ... additional fields
```

### DecompositionPlan

```python
@dataclass
class DecompositionPlan:
    id: str
    problem_id: str
    strategy: DecompositionStrategy
    sub_problems: List[SubProblem]
    dependency_graph: DependencyGraph
    quality_scores: QualityScores
    confidence_level: float
    # ... additional fields
```

### SubProblem

```python
@dataclass
class SubProblem:
    id: str
    parent_id: str
    title: str
    description: str
    type: SubProblemType
    complexity_score: ComplexityScore
    dependencies: List[str]
    success_criteria: List[SuccessCriterion]
    priority: int
    estimated_effort: int
    # ... additional fields
```

## Error Handling

All API methods may raise:
- `AnalysisError`: Problem analysis failed
- `DecompositionError`: Decomposition failed
- `ValidationError`: Validation failed
- `PersistenceError`: Database operation failed

Use the retry decorator for automatic retry:

```python
from sovereign_reliability import with_retry

@with_retry(max_attempts=3)
def my_operation():
    # Your code here
    pass
```

## Performance Considerations

1. **Caching**: Use `@cached` decorator for expensive operations
2. **Monitoring**: Use `@timed` decorator to track performance
3. **Lazy Loading**: Use `LazyLoader` for deferred resource loading
4. **Batch Operations**: Use `BatchProcessor` for database operations

## Testing

Run all tests:
```bash
python -m pytest test_sovereign*.py -v
```

Current test coverage: 198 passing tests

## Version

Current Version: 1.0.0
API Stability: Stable
