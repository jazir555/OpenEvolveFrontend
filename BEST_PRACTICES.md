# Best Practices Guide

**Version:** 1.0.0
**Last Updated:** 2025-01-03

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Decomposition Strategy](#decomposition-strategy)
3. [Solution Development](#solution-development)
4. [Quality Assurance](#quality-assurance)
5. [Performance Optimization](#performance-optimization)
6. [Security](#security)
7. [Integration Patterns](#integration-patterns)
8. [Common Pitfalls](#common-pitfalls)

---

## Problem Definition

### DO: Be Specific and Clear

✓ **Good:**
```python
problem = ProblemDefinition(
    title="User Authentication System with 2FA",
    description="""Build secure user authentication with:
    - Password hashing using Argon2id (256 MB memory)
    - Session management with JWT tokens
    - Email-based password reset
    - TOTP-based 2FA (Google Authenticator compatible)
    - Brute-force protection (rate limiting)
    - PCI-DSS compliant
    """,
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(domain="Security", subdomain="Authentication"),
    complexity_score=ComplexityScore(overall=7, cognitive=6, computational=7, domain=8, integration=7)
)
```

✗ **Bad:**
```python
problem = ProblemDefinition(
    title="Auth",
    description="Make login work",  # Too vague
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(domain="Software"),
    complexity_score=ComplexityScore(overall=5, cognitive=5, computational=5, domain=5, integration=5)
)
```

### DO: Estimate Complexity Honestly

```python
# Consider all aspects of complexity
complexity = ComplexityScore(
    overall_complexity=7,
    cognitive_complexity=6,      # How hard to understand?
    computational_complexity=7,  # How much computation?
    domain_complexity=8,         # How specialized?
    integration_complexity=7     # How many integrations?
)
```

### DO: Define Success Criteria

```python
problem = ProblemDefinition(
    # ... other fields
    success_criteria=[
        SuccessCriterion(
            id="sc-001",
            description="Handle 1000 login requests/second",
            metric="throughput",
            threshold=1000.0,
            validation_method="automated"
        ),
        SuccessCriterion(
            id="sc-002",
            description="99.9% authentication accuracy",
            metric="accuracy",
            threshold=0.999,
            validation_method="automated"
        )
    ]
)
```

### DON'T: Ignore Constraints

```python
# ✓ Document constraints clearly
problem = ProblemDefinition(
    # ... other fields
    constraints=[
        Constraint(
            id="const-001",
            description="Must comply with GDPR",
            type="regulatory",
            severity="critical"
        ),
        Constraint(
            id="const-002",
            description="Maximum latency 100ms",
            type="performance",
            severity="high"
        )
    ]
)
```

---

## Decomposition Strategy

### DO: Choose Strategy Based on Problem Type

| Problem Type | Best Strategy | Why |
|--------------|---------------|-----|
| Research / Design | Semantic | Needs conceptual understanding |
| System Architecture | Hierarchical | Clear structure needed |
| Workflow / Pipeline | Flow-Based | Sequential stages |
| Unknown | Semantic | Most flexible |

```python
# Example: Correct strategy selection
if problem.problem_type == ProblemType.RESEARCH:
    strategy = "semantic"
elif problem.problem_type == ProblemType.DESIGN and "architecture" in problem.description.lower():
    strategy = "hierarchical"
elif "pipeline" in problem.description.lower() or "workflow" in problem.description.lower():
    strategy = "flow"
else:
    strategy = "semantic"  # Default
```

### DO: Set Appropriate Parameters

```python
# For simple problems
result = decompose_problem_into_sub_problems(
    problem_statement=simple_problem,
    max_sub_problems=5,  # Fewer sub-problems
    complexity_target=5  # Moderate complexity
)

# For complex problems
result = decompose_problem_into_sub_problems(
    problem_statement=complex_problem,
    max_sub_problems=15,  # More sub-problems
    complexity_target=6  # Slightly higher complexity
)
```

### DO: Review and Adjust Sub-Problems

```python
result = engine.decompose(problem, strategy="semantic")

# Review each sub-problem
for sp in result.sub_problems:
    # Check if complexity is reasonable
    if sp.complexity_score.overall_complexity > 8:
        print(f"Warning: {sp.title} has high complexity ({sp.complexity_score.overall_complexity}/10)")
        print("  Consider splitting into smaller sub-problems")

    # Check if effort is reasonable
    if sp.estimated_effort > 40:
        print(f"Warning: {sp.title} has high effort estimate ({sp.estimated_effort}h)")
        print("  Consider breaking down further")

    # Check dependencies
    if len(sp.dependencies) > 3:
        print(f"Warning: {sp.title} has many dependencies ({len(sp.dependencies)})")
        print("  Consider reordering to reduce dependencies")
```

### DON'T: Accept First Decomposition Blindly

```python
# ✗ Bad: Blind acceptance
result = engine.decompose(problem, strategy="semantic")
for sp in result.sub_problems:
    execute(sp)

# ✓ Good: Review first
result = engine.decompose(problem, strategy="semantic")

# Validate quality
quality = assess_decomposition_quality(problem, result)
if quality['quality_score'] < 0.7:
    print("Low quality score, re-decomposing with different parameters...")
    result = decompose_problem_into_sub_problems(
        problem_statement=problem.description,
        max_sub_problems=10,  # Adjusted
        complexity_target=6   # Adjusted
    )
```

---

## Solution Development

### DO: Use Appropriate Execution Methods

```python
# Match execution method to sub-problem characteristics
def select_execution_method(sub_problem):
    """Select best execution method"""

    desc = sub_problem.description.lower()

    # Critical zero-error tasks
    if any(kw in desc for kw in ['critical', 'safety', 'zero-error']):
        return 'roma_mdap_maker'

    # Code generation
    if any(kw in desc for kw in ['implement', 'code', 'function', 'class']):
        return 'claudiomiro'

    # Research/analysis
    if sub_problem.type.value in ['research', 'analysis']:
        return 'datapizza'

    # Hierarchical decomposition
    if 'decompose' in desc or 'break down' in desc:
        return 'roma'

    # Default
    return 'traditional'
```

### DO: Solve Independent Sub-Problems in Parallel

```python
from concurrent.futures import ThreadPoolExecutor

# Find independent sub-problems (no dependencies)
independent = [sp for sp in sub_problems if not sp['dependencies']]

print(f"Found {len(independent)} independent sub-problems")
print("Executing in parallel...")

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(solve_subproblem, independent)
```

### DO: Use Evolution for Complex Problems

```python
# Enable evolution for better quality solutions
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="Optimize database query performance",
    team_name="Performance-Blue",
    execution_method="traditional",
    use_evolution=True,
    evolution_iterations=100  # More iterations = better quality
)
```

### DON'T: Ignore Dependencies

```python
# ✗ Bad: Execute in arbitrary order
for sp in sub_problems:
    solve(sp)

# ✓ Good: Respect dependency order
def execution_order(sub_problems):
    """Calculate valid execution order"""
    executed = set()
    order = []

    while len(executed) < len(sub_problems):
        ready = [
            sp for sp in sub_problems
            if sp.id not in executed and
            all(dep in executed for dep in sp.dependencies)
        ]

        if not ready:
            raise Exception("Circular dependency!")

        order.extend(ready)
        for sp in ready:
            executed.add(sp.id)

    return order

for sp in execution_order(sub_problems):
    solve(sp)
```

---

## Quality Assurance

### DO: Always Validate Solutions

```python
# Red Team critique
critique = critique_solution_with_gauntlet(
    solution=solution_text,
    sub_problem_id=sp.id,
    gauntlet_name="Security-RedTeam-Gauntlet"
)

# Gold Team verification
verification = verify_solution_with_gauntlet(
    solution=revised_solution,
    critique=critique,
    sub_problem_id=sp.id,
    gauntlet_name="Quality-GoldTeam-Gauntlet",
    requirements=requirements
)

# Only accept approved solutions
if verification['approved']:
    print(f"✓ {sp.title} approved")
else:
    print(f"✗ {sp.title} needs revision")
    print(f"  Issues: {verification['feedback']}")
```

### DO: Use Custom Gauntlets for Domain-Specific Validation

```python
class SecurityGauntlet(Gauntlet):
    """Custom security validation"""

    def __init__(self):
        super().__init__(
            name="Security-Gauntlet",
            description="Security-focused validation"
        )

        # Add security-specific rounds
        self.add_round(GauntletRound(
            name="SQL Injection Check",
            test_function=self._check_sql_injection
        ))

        self.add_round(GauntletRound(
            name="XSS Check",
            test_function=self._check_xss
        ))

    def _check_sql_injection(self, solution: str) -> dict:
        # Implementation
        pass

    def _check_xss(self, solution: str) -> dict:
        # Implementation
        pass
```

### DO: Track Quality Metrics

```python
def track_quality_metrics(sub_problems, solutions, verifications):
    """Track quality metrics across all solutions"""

    metrics = {
        'total_sub_problems': len(sub_problems),
        'solutions_generated': len(solutions),
        'solutions_approved': sum(1 for v in verifications.values() if v.get('approved', False)),
        'average_correctness': sum(v.get('correctness_score', 0) for v in verifications.values()) / len(verifications),
        'average_completeness': sum(v.get('completeness_score', 0) for v in verifications.values()) / len(verifications),
        'average_quality': sum(v.get('quality_score', 0) for v in verifications.values()) / len(verifications),
    }

    metrics['approval_rate'] = metrics['solutions_approved'] / metrics['solutions_generated']

    return metrics

metrics = track_quality_metrics(sub_problems, solutions, verifications)

print(f"Approval Rate: {metrics['approval_rate']:.1%}")
print(f"Average Quality: {metrics['average_quality']:.2f}/1.0")
```

---

## Performance Optimization

### DO: Enable Caching

```python
from openevolve_client import OpenEvolveClient

# Create client with caching
client = OpenEvolveClient(
    enable_cache=True,
    cache_ttl=3600,  # Cache for 1 hour
    cache_dir="./cache/openevolve"
)

# Cache is automatically used for similar requests
```

### DO: Use Appropriate Evolution Parameters

```python
# Fast prototyping (lower quality, faster)
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="...",
    execution_method="traditional",
    use_evolution=True,
    evolution_iterations=20  # Fewer iterations
)

# Production quality (slower, better)
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="...",
    execution_method="traditional",
    use_evolution=True,
    evolution_iterations=100  # More iterations
)
```

### DO: Profile Performance

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")

# Time decomposition workflow
with timer("Analysis"):
    analysis = analyze_problem_for_decomposition(problem_statement)

with timer("Decomposition"):
    result = decompose_problem_into_sub_problems(
        problem_statement=problem_statement,
        analysis=analysis
   )

with timer("Solving"):
    for sp in result['sub_problems']:
        with timer(f"  {sp['title']}"):
            solve(sp)
```

---

## Security

### DO: Never Hardcode API Keys

```python
# ✗ Bad: Hardcoded
api_key = "sk-1234567890abcdef"

# ✓ Good: Environment variable
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

### DO: Validate Input

```python
def validate_problem_statement(statement: str) -> bool:
    """Validate problem statement"""

    if not statement or len(statement.strip()) < 10:
        raise ValueError("Problem statement too short")

    if len(statement) > 10000:
        raise ValueError("Problem statement too long (max 10000 chars)")

    # Check for malicious patterns
    dangerous_patterns = ["__import__", "eval(", "exec("]
    for pattern in dangerous_patterns:
        if pattern in statement:
            raise ValueError(f"Dangerous pattern detected: {pattern}")

    return True
```

### DO: Use Safe Code Execution

```python
# Safe globals for evolved code execution
SAFE_GLOBALS = {
    "__builtins__": {
        "dict": dict,
        "list": list,
        "str": str,
        "int": int,
        "float": float,
        "len": len,
        "range": range,
        # Only safe builtins
    }
}

# Execute with restricted environment
exec(code, SAFE_GLOBALS, local_vars)
```

---

## Integration Patterns

### DO: Use MCP Tools for CrewAI Integration

```python
from decomposition_mcp_tools import (
    analyze_problem_for_decomposition,
    decompose_problem_into_sub_problems,
    solve_sub_problem_with_team,
    critique_solution_with_gauntlet,
    verify_solution_with_gauntlet
)

# Complete workflow
def crewai_workflow(problem_statement: str):
    """Complete decomposition workflow for CrewAI"""

    # Analyze
    analysis = analyze_problem_for_decomposition(problem_statement)

    # Decompose
    decomp = decompose_problem_into_sub_problems(
        problem_statement=problem_statement,
        analysis=analysis
    )

    # Solve, critique, verify each sub-problem
    final_solutions = {}
    for sp in decomp['sub_problems']:
        # Solve
        solution = solve_sub_problem_with_team(
            sub_problem_id=sp['id'],
            sub_problem_description=sp['description'],
            team_name="Default-Blue"
        )

        # Critique
        critique = critique_solution_with_gauntlet(
            solution=solution['solution'],
            sub_problem_id=sp['id'],
            gauntlet_name="Default-RedTeam-Gauntlet"
        )

        # Verify
        verification = verify_solution_with_gauntlet(
            solution=solution['solution'],
            critique=critique,
            sub_problem_id=sp['id'],
            gauntlet_name="Default-GoldTeam-Gauntlet"
        )

        if verification['approved']:
            final_solutions[sp['id']] = solution['solution']

    return final_solutions
```

---

## Common Pitfalls

### Pitfall 1: Over-Decomposition

**Problem:** Too many small sub-problems

```python
# ✗ Bad: 30+ sub-problems
result = decompose_problem_into_sub_problems(
    problem_statement=problem,
    max_sub_problems=30  # Too many!
)

# ✓ Good: 5-10 sub-problems
result = decompose_problem_into_sub_problems(
    problem_statement=problem,
    max_sub_problems=10  # Manageable
)
```

### Pitfall 2: Under-Decomposition

**Problem:** Too few large sub-problems

```python
# ✗ Bad: Only 2 sub-problems for complex problem
result = decompose_problem_into_sub_problems(
    problem_statement=complex_problem,
    max_sub_problems=3  # Too few!
)

# ✓ Good: Appropriate number
result = decompose_problem_into_sub_problems(
    problem_statement=complex_problem,
    max_sub_problems=12  # Better
)
```

### Pitfall 3: Ignoring Quality Metrics

```python
# ✗ Bad: Don't check quality
result = engine.decompose(problem)
execute_all(result.sub_problems)

# ✓ Good: Check quality first
result = engine.decompose(problem)
quality = assess_decomposition_quality(problem, result)

if quality['quality_score'] < 0.7:
    # Re-decompose with different parameters
    result = decompose_problem_into_sub_problems(
        problem_statement=problem.description,
        max_sub_problems=quality['num_sub_problems'] - 2
    )
```

### Pitfall 4: Not Using Evolution

```python
# ✗ Bad: Don't use evolution (lower quality)
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="Optimize algorithm",
    use_evolution=False  # Missing out on better solutions
)

# ✓ Good: Use evolution for complex problems
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="Optimize algorithm",
    use_evolution=True,
    evolution_iterations=100
)
```

---

## Summary

### Key Principles

1. **Be Specific**: Clear, detailed problem definitions
2. **Choose Wisely**: Select appropriate strategy and execution method
3. **Review Always**: Validate decompositions and solutions
4. **Optimize**: Use caching, parallelization, evolution
5. **Secure**: Never hardcode secrets, validate inputs
6. **Track Metrics**: Monitor quality and performance

### Checklist

Before decomposing:
- [ ] Problem statement is clear and specific
- [ ] Complexity is estimated honestly
- [ ] Success criteria are defined
- [ ] Constraints are documented
- [ ] Appropriate strategy is selected

During decomposition:
- [ ] Quality metrics are acceptable
- [ ] Sub-problems are manageable
- [ ] Dependencies are logical
- [ ] Effort is distributed evenly

After decomposition:
- [ ] Solutions are validated
- [ ] Quality metrics are tracked
- [ ] Performance is acceptable
- [ ] Security best practices are followed

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-03
