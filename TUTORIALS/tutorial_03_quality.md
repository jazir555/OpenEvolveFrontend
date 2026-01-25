# Tutorial 3: Quality Assessment

**Level:** Intermediate
**Time:** 40 minutes
**Prerequisites:** Tutorial 1 (Getting Started)

---

## Learning Objectives

After this tutorial, you will be able to:
- Understand decomposition quality metrics
- Assess quality of sub-problems
- Identify and fix quality issues
- Validate decompositions
- Measure effectiveness

---

## Understanding Quality

### What Makes a "Good" Decomposition?

A good decomposition has these characteristics:

```
✓ COMPLETENESS
  - All aspects of the problem are covered
  - No missing requirements
  - All constraints addressed

✓ MANAGEABILITY
  - Each sub-problem is solvable
  - Appropriate granularity
  - Clear scope boundaries

✓ INDEPENDENCE
  - Minimal overlap between sub-problems
  - Clear dependencies
  - Can be worked on in parallel

✓ VERIFIABILITY
  - Clear success criteria
  - Testable acceptance criteria
  - Measurable outcomes

✓ BALANCE
  - Balanced complexity across sub-problems
  - Reasonable effort distribution
  - Logical priority ordering
```

---

## Quality Metrics

The decomposition engine provides several quality metrics:

### 1. Complexity Score

```python
from sovereign_data_models import ComplexityScore

# Good complexity distribution
good_scores = {
    'sub-001': ComplexityScore(overall=6, cognitive=5, computational=7, domain=6, integration=6),
    'sub-002': ComplexityScore(overall=5, cognitive=6, computational=5, domain=5, integration=5),
    'sub-003': ComplexityScore(overall=7, cognitive=6, computational=8, domain=7, integration=7),
}

# All in 4-7 range - balanced!
```

**Target**: Overall complexity 4-7 for most sub-problems

### 2. Effort Distribution

```python
# Calculate effort distribution
efforts = [sp.estimated_effort for sp in sub_problems]

import statistics
mean_effort = statistics.mean(efforts)
median_effort = statistics.median(efforts)

# Good: Mean ≈ Median, small standard deviation
std_dev = statistics.stdev(efforts)
print(f"Mean effort: {mean_effort:.1f}h")
print(f"Median effort: {median_effort:.1f}h")
print(f"Std deviation: {std_dev:.1f}h")

# Target: std_dev < mean * 0.5 (not too spread out)
```

### 3. Dependency Depth

```python
def calculate_max_depth(sub_problems):
    """Calculate maximum dependency depth"""
    id_to_sp = {sp.id: sp for sp in sub_problems}

    def depth(sp_id, visited=None):
        if visited is None:
            visited = set()

        if sp_id in visited:
            return 0  # Circular dependency!

        visited.add(sp_id)
        sp = id_to_sp[sp_id]

        if not sp.dependencies:
            return 1

        max_dep_depth = 0
        for dep_id in sp.dependencies:
            max_dep_depth = max(max_dep_depth, depth(dep_id, visited))

        return max_dep_depth + 1

    return max(depth(sp.id) for sp in sub_problems)

max_depth = calculate_max_depth(sub_problems)
print(f"Max dependency depth: {max_depth}")

# Target: depth < 5 (not too deep)
```

### 4. Coverage

```python
def assess_coverage(problem: ProblemDefinition, sub_problems: List[SubProblem]) -> Dict[str, float]:
    """Assess how well sub-problems cover the original problem"""

    # Extract keywords from problem
    problem_keywords = set(problem.description.lower().split())

    # Extract keywords from all sub-problems
    all_sp_keywords = set()
    for sp in sub_problems:
        all_sp_keywords.update(sp.description.lower().split())

    # Calculate coverage
    covered_keywords = problem_keywords & all_sp_keywords
    coverage_ratio = len(covered_keywords) / len(problem_keywords)

    return {
        'keyword_coverage': coverage_ratio,
        'covered_keywords': len(covered_keywords),
        'total_keywords': len(problem_keywords)
    }

coverage = assess_coverage(problem, sub_problems)
print(f"Coverage: {coverage['keyword_coverage']:.1%}")

# Target: > 80% keyword coverage
```

---

## Quality Assessment Example

```python
# example_quality_assessment.py
from decomposition_engine import DecompositionEngine
from problem_analyzer import ProblemAnalyzer
from sovereign_data_models import ProblemDefinition, ProblemType, DomainContext, ComplexityScore
import statistics

def assess_decomposition_quality(problem: ProblemDefinition, result) -> Dict[str, Any]:
    """Comprehensive quality assessment"""

    sub_problems = result.sub_problems

    # 1. Count metrics
    num_sub_problems = len(sub_problems)

    # 2. Complexity analysis
    complexities = [sp.complexity_score.overall_complexity for sp in sub_problems]
    mean_complexity = statistics.mean(complexities)
    std_complexity = statistics.stdev(complexities) if len(complexities) > 1 else 0

    # 3. Effort analysis
    efforts = [sp.estimated_effort for sp in sub_problems]
    total_effort = sum(efforts)
    mean_effort = statistics.mean(efforts)
    std_effort = statistics.stdev(efforts) if len(efforts) > 1 else 0

    # 4. Dependency analysis
    with_no_deps = sum(1 for sp in sub_problems if not sp.dependencies)
    max_depth = calculate_max_depth(sub_problems)

    # 5. Coverage analysis
    coverage = assess_coverage(problem, sub_problems)

    # 6. Quality score
    quality_issues = []

    # Check number of sub-problems
    if num_sub_problems < 3:
        quality_issues.append("Too few sub-problems (< 3)")
    elif num_sub_problems > 15:
        quality_issues.append("Too many sub-problems (> 15)")

    # Check complexity balance
    if std_complexity > 2.0:
        quality_issues.append(f"High complexity variance (std={std_complexity:.1f})")

    # Check effort balance
    if std_effort > mean_effort * 0.5:
        quality_issues.append(f"High effort variance (std={std_effort:.1f})")

    # Check dependency depth
    if max_depth > 5:
        quality_issues.append(f"Dependency chain too deep (depth={max_depth})")

    # Check coverage
    if coverage['keyword_coverage'] < 0.8:
        quality_issues.append(f"Low coverage ({coverage['keyword_coverage']:.1%})")

    # Calculate overall quality score
    quality_score = 1.0
    quality_score -= len(quality_issues) * 0.1
    quality_score = max(0.0, quality_score)

    return {
        'quality_score': quality_score,
        'num_sub_problems': num_sub_problems,
        'complexity': {
            'mean': mean_complexity,
            'std': std_complexity,
            'min': min(complexities),
            'max': max(complexities)
        },
        'effort': {
            'total': total_effort,
            'mean': mean_effort,
            'std': std_effort,
            'min': min(efforts),
            'max': max(efforts)
        },
        'dependencies': {
            'with_no_deps': with_no_deps,
            'max_depth': max_depth,
            'parallelizable': with_no_deps
        },
        'coverage': coverage,
        'issues': quality_issues
    }

# Use it
engine = DecompositionEngine()
result = engine.decompose(problem, strategy="semantic")

quality = assess_decomposition_quality(problem, result)

print("\n=== Quality Assessment ===")
print(f"Quality Score: {quality['quality_score']:.1%}")
print(f"\nSub-Problems: {quality['num_sub_problems']}")
print(f"Complexity: {quality['complexity']['mean']:.1f} ± {quality['complexity']['std']:.1f}")
print(f"Effort: {quality['effort']['mean']:.1f}h ± {quality['effort']['std']:.1f}h")
print(f"Parallelizable: {quality['dependencies']['parallelizable']}/{quality['num_sub_problems']}")
print(f"Coverage: {quality['coverage']['keyword_coverage']:.1%}")

if quality['issues']:
    print(f"\n⚠️  Issues Found:")
    for issue in quality['issues']:
        print(f"   - {issue}")
else:
    print(f"\n✓ No quality issues found!")
```

---

## Improving Quality

### Issue 1: Too Many Sub-Problems

**Problem**: Generated 20+ sub-problems

**Solution 1: Increase Complexity Target**
```python
# Original
result = engine.decompose(problem, complexity_target=5)

# Increase to merge sub-problems
result = engine.decompose(problem, complexity_target=7)
```

**Solution 2: Merge Similar Sub-Problems**
```python
def merge_similar_subproblems(sub_problems, threshold=0.7):
    """Merge sub-problems with similar descriptions"""
    # Implementation uses text similarity
    merged = []
    used = set()

    for i, sp1 in enumerate(sub_problems):
        if sp1.id in used:
            continue

        similar = [sp1]
        for j, sp2 in enumerate(sub_problems[i+1:], i+1):
            if sp2.id in used:
                continue

            # Calculate similarity (simplified)
            similarity = calculate_similarity(sp1.description, sp2.description)

            if similarity > threshold:
                similar.append(sp2)
                used.add(sp2.id)

        if len(similar) > 1:
            # Merge into one
            merged_sp = merge_subproblems(similar)
            merged.append(merged_sp)
            used.add(sp1.id)
        else:
            merged.append(sp1)
            used.add(sp1.id)

    return merged
```

### Issue 2: High Complexity Variance

**Problem**: Some sub-problems are 3/10, others are 9/10

**Solution: Rebalance Complexity**
```python
def rebalance_complexity(sub_problems, target_mean=6):
    """Rebalance complexity by splitting/merging"""

    balanced = []
    for sp in sub_problems:
        if sp.complexity_score.overall_complexity > 8:
            # Split into smaller sub-problems
            splits = split_subproblem(sp, num_parts=2)
            balanced.extend(splits)
        elif sp.complexity_score.overall_complexity < 4:
            # Merge with next sub-problem
            # (Implementation depends on context)
            balanced.append(sp)
        else:
            balanced.append(sp)

    return balanced
```

### Issue 3: Deep Dependency Chains

**Problem**: Dependency depth is 8+

**Solution: Flatten Hierarchy**
```python
def flatten_dependencies(sub_problems, max_depth=4):
    """Flatten deep dependency chains"""

    # Build dependency graph
    id_to_sp = {sp.id: sp for sp in sub_problems}

    def calculate_depth(sp_id, depth=0, visited=None):
        if visited is None:
            visited = set()

        if sp_id in visited or depth > max_depth:
            return depth

        visited.add(sp_id)
        sp = id_to_sp[sp_id]

        if not sp.dependencies:
            return depth

        return max(calculate_depth(dep_id, depth+1, visited)
                   for dep_id in sp.dependencies)

    # Find sub-problems causing deep chains
    problem_ids = []
    for sp in sub_problems:
        depth = calculate_depth(sp.id)
        if depth > max_depth:
            problem_ids.append(sp.id)

    # Remove intermediate dependencies (simplified)
    for sp in sub_problems:
        if sp.id in problem_ids:
            # Keep only direct dependencies, not transitive
            sp.dependencies = sp.dependencies[:1]  # Keep first only

    return sub_problems
```

### Issue 4: Low Coverage

**Problem**: Only 60% of problem keywords covered

**Solution: Add Missing Sub-Problems**
```python
def add_missing_coverage(problem: ProblemDefinition, sub_problems):
    """Add sub-problems for missing aspects"""

    # Extract missing keywords
    problem_keywords = set(problem.description.lower().split())
    covered_keywords = set()

    for sp in sub_problems:
        covered_keywords.update(sp.description.lower().split())

    missing = problem_keywords - covered_keywords

    if missing:
        # Create sub-problem for missing aspects
        missing_sp = SubProblem(
            id=f"{problem.id}-missing-coverage",
            parent_id=problem.id,
            title="Additional Requirements",
            description=f"Address: {', '.join(missing)}",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                overall_complexity=5,
                cognitive_complexity=5,
                computational_complexity=5,
                domain_complexity=5,
                integration_complexity=5
            ),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="",
            priority=5,
            estimated_effort=8
        )
        sub_problems.append(missing_sp)

    return sub_problems
```

---

## Validation Checklist

Use this checklist to validate decompositions:

```python
# validation_checklist.py
def validate_decomposition(problem: ProblemDefinition, result) -> Dict[str, bool]:
    """Comprehensive validation checklist"""

    sub_problems = result.sub_problems

    checks = {
        'has_min_sub_problems': len(sub_problems) >= 3,
        'has_max_sub_problems': len(sub_problems) <= 15,
        'all_have_titles': all(bool(sp.title.strip()) for sp in sub_problems),
        'all_have_descriptions': all(bool(sp.description.strip()) for sp in sub_problems),
        'all_have_types': all(sp.type for sp in sub_problems),
        'all_have_priorities': all(1 <= sp.priority <= 10 for sp in sub_problems),
        'all_have_efforts': all(sp.estimated_effort > 0 for sp in sub_problems),
        'all_have_success_criteria': all(sp.success_criteria for sp in sub_problems),
        'complexities_reasonable': all(3 <= sp.complexity_score.overall_complexity <= 9
                                      for sp in sub_problems),
        'efforts_reasonable': all(4 <= sp.estimated_effort <= 40 for sp in sub_problems),
        'no_duplicate_titles': len(set(sp.title for sp in sub_problems)) == len(sub_problems),
        'dependencies_valid': all(
            all(dep in [sp2.id for sp2 in sub_problems] for dep in sp.dependencies)
            for sp in sub_problems
        )
    }

    return checks

# Use it
checks = validate_decomposition(problem, result)

print("\n=== Validation Checklist ===")
for check_name, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"{status} {check_name}")

all_passed = all(checks.values())
if all_passed:
    print(f"\n✓ All checks passed!")
else:
    failed = sum(1 for passed in checks.values() if not passed)
    print(f"\n✗ {failed} check(s) failed")
```

---

## Exercise: Quality Improvement

Given this decomposition result with quality issues, fix them:

```python
# exercise_quality.py
# TODO: Fix quality issues in this decomposition

quality_issues = [
    "Too many sub-problems (20)",
    "High complexity variance (std=3.2)",
    "Dependency chain too deep (depth=7)",
    "Low coverage (65%)"
]

# Your task: Implement fixes
# 1. Reduce sub-problems to 8-12
# 2. Rebalance complexity (std < 2.0)
# 3. Flatten dependencies (depth < 5)
# 4. Improve coverage (> 80%)

# Hints:
# - Use merge_similar_subproblems()
# - Use rebalance_complexity()
# - Use flatten_dependencies()
# - Use add_missing_coverage()
```

---

## Summary

In this tutorial, you learned:

✓ Quality metrics for decomposition
✓ How to assess quality comprehensively
✓ Common quality issues and fixes
✓ Validation checklist
✓ Quality improvement techniques

---

## Next Steps

**Next Tutorial:** [Tutorial 4: Solution Integration](tutorial_04_integration.md)

---

**Tutorial Version:** 1.0.0
**Last Updated:** 2025-01-03
