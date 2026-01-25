# Tutorial 5: Advanced Features

**Level:** Advanced
**Time:** 60 minutes
**Prerequisites:** Tutorials 1-4

---

## Learning Objectives

After this tutorial, you will be able to:
- Use evolutionary optimization
- Implement custom quality metrics
- Create custom gauntlets
- Handle large-scale decompositions
- Optimize performance

---

## Advanced Feature 1: Evolutionary Optimization

### Basic Evolution

```python
# example_evolution_basic.py
from decomposition_mcp_tools import solve_sub_problem_with_team

# Enable evolution for better solutions
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="Optimize database query performance",
    team_name="Performance-Blue",
    execution_method="traditional",
    use_evolution=True,  # Enable evolution
    evolution_iterations=100  # Number of generations
)

print(f"Evolution metrics: {solution['evolution_metrics']}")
print(f"Best fitness: {solution['evolution_metrics']['best_fitness']}")
print(f"Iterations: {solution['evolution_metrics']['iterations']}")
```

### Custom Fitness Function

```python
# example_custom_fitness.py
from openevolve_client import OpenEvolveClient

def custom_fitness_evaluator(solution_code: str) -> float:
    """Custom fitness function for database optimization"""

    fitness = 0.0

    # Criterion 1: Uses indexing (20%)
    if "CREATE INDEX" in solution_code or "add_index" in solution_code:
        fitness += 0.2

    # Criterion 2: Avoids N+1 queries (25%)
    if "JOIN" in solution_code or "select_related" in solution_code or "prefetch" in solution_code:
        fitness += 0.25

    # Criterion 3: Uses caching (25%)
    if "cache" in solution_code.lower():
        fitness += 0.25

    # Criterion 4: Includes pagination (15%)
    if "limit" in solution_code.lower() or "paginate" in solution_code.lower():
        fitness += 0.15

    # Criterion 5: Has error handling (15%)
    if "try:" in solution_code and "except" in solution_code:
        fitness += 0.15

    return fitness

# Use custom fitness with OpenEvolve
client = OpenEvolveClient()
result = client.evolve(
    content="# Optimize this database query\nSELECT * FROM users;",
    evolution_mode="standard",
    fitness_function=custom_fitness_evaluator,
    iterations=100
)
```

### Adversarial Evolution

```python
# example_adversarial_evolution.py
from openevolve_client import OpenEvolveClient

# Adversarial mode for robust solutions
client = OpenEvolveClient()

result = client.evolve(
    content="def authenticate_user(username, password):\n    pass",
    evolution_mode="adversarial",  # Adversarial evolution
    iterations=150,
    adversarial_examples=[
        ("admin', --", "SQL injection"),
        ("' OR '1'='1", "SQL injection"),
        ("../../../etc/passwd", "Path traversal"),
        ("<script>alert('xss')</script>", "XSS")
    ]
)

print("Adversarially evolved solution:")
print(result.best_code)
```

---

## Advanced Feature 2: Parallel Execution

### Parallelizing Independent Sub-Problems

```python
# example_parallel_execution.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from decomposition_mcp_tools import solve_sub_problem_with_team

def solve_single_subproblem(sp):
    """Solve a single sub-problem"""
    return solve_sub_problem_with_team(
        sub_problem_id=sp['id'],
        sub_problem_description=sp['description'],
        team_name="Default-Blue",
        execution_method="traditional",
        use_evolution=False
    )

# Find independent sub-problems (no dependencies)
independent_sps = [
    sp for sp in decomposition_result['sub_problems']
    if not sp['dependencies']
]

print(f"Found {len(independent_sps)} independent sub-problems")
print("Executing in parallel...\n")

# Execute in parallel (4 workers)
solutions = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all tasks
    future_to_sp = {
        executor.submit(solve_single_subproblem, sp): sp
        for sp in independent_sps
    }

    # Collect results as they complete
    for future in as_completed(future_to_sp):
        sp = future_to_sp[future]
        try:
            solution = future.result()
            solutions[sp['id']] = solution
            print(f"✓ {sp['title']}: {solution['status']}")
        except Exception as e:
            print(f"✗ {sp['title']}: ERROR - {e}")

print(f"\nCompleted {len(solutions)}/{len(independent_sps)}")
```

### Parallelizing with Progress Tracking

```python
# example_parallel_with_progress.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # pip install tqdm

def solve_with_progress(sub_problems, max_workers=4):
    """Solve sub-problems with progress bar"""

    solutions = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(solve_single_subproblem, sp): sp
            for sp in sub_problems
        }

        # Process with progress bar
        with tqdm(total=len(futures), desc="Solving") as pbar:
            for future in as_completed(futures):
                sp = futures[future]
                try:
                    solution = future.result()
                    solutions[sp['id']] = solution
                    pbar.set_postfix_str(f"Last: {sp['title'][:30]}")
                except Exception as e:
                    print(f"\nError: {sp['title']}: {e}")

                pbar.update(1)

    return solutions

solutions = solve_with_progress(independent_sps, max_workers=4)
```

---

## Advanced Feature 3: Custom Gauntlets

### Creating Custom Red Team Gauntlet

```python
# example_custom_gauntlet.py
from gauntlet_manager import GauntletManager, Gauntlet, GauntletRound

class SecurityRedTeamGauntlet(Gauntlet):
    """Custom security-focused Red Team gauntlet"""

    def __init__(self):
        super().__init__(
            name="Security-RedTeam-Gauntlet",
            team_name="Security-RedTeam",
            description="Security-focused adversarial testing"
        )

        # Add gauntlet rounds
        self.add_round(GauntletRound(
            name="SQL Injection Check",
            description="Check for SQL injection vulnerabilities",
            test_function=self._check_sql_injection
        ))

        self.add_round(GauntletRound(
            name="XSS Check",
            description="Check for XSS vulnerabilities",
            test_function=self._check_xss
        ))

        self.add_round(GauntletRound(
            name="Authentication Check",
            description="Verify secure authentication",
            test_function=self._check_authentication
        ))

    def _check_sql_injection(self, solution: str) -> dict:
        """Check for SQL injection vulnerabilities"""

        issues = []

        # Check for string concatenation in SQL
        dangerous_patterns = [
            'f"SELECT * FROM',
            '"SELECT * FROM' +',
            "'SELECT * FROM" +',
            "format('SELECT",
        ]

        for pattern in dangerous_patterns:
            if pattern in solution:
                issues.append({
                    'severity': 'HIGH',
                    'description': f'SQL injection vulnerability: {pattern}',
                    'location': 'Database query construction',
                    'suggestion': 'Use parameterized queries or ORM'
                })

        return {
            'passed': len(issues) == 0,
            'issues': issues
        }

    def _check_xss(self, solution: str) -> dict:
        """Check for XSS vulnerabilities"""

        issues = []

        # Check for unsafe HTML rendering
        if 'innerHTML' in solution and 'sanitize' not in solution:
            issues.append({
                'severity': 'HIGH',
                'description': 'Unsafe innerHTML usage without sanitization',
                'location': 'HTML rendering',
                'suggestion': 'Use DOMPurify or template with auto-escaping'
            })

        if 'dangerouslySetInnerHTML' in solution:
            issues.append({
                'severity': 'HIGH',
                'description': 'React dangerouslySetInnerHTML without sanitization',
                'location': 'React component',
                'suggestion': 'Use DOMPurify before dangerouslySetInnerHTML'
            })

        return {
            'passed': len(issues) == 0,
            'issues': issues
        }

    def _check_authentication(self, solution: str) -> dict:
        """Check authentication implementation"""

        issues = []

        # Check for plaintext password storage
        if 'password' in solution.lower() and 'hash' not in solution:
            issues.append({
                'severity': 'CRITICAL',
                'description': 'Possible plaintext password storage',
                'location': 'Password handling',
                'suggestion': 'Use bcrypt/argon2 password hashing'
            })

        # Check for weak password hashing
        if 'md5' in solution.lower() or 'sha1' in solution.lower():
            issues.append({
                'severity': 'HIGH',
                'description': 'Weak password hashing algorithm (MD5/SHA1)',
                'location': 'Password hashing',
                'suggestion': 'Use bcrypt or argon2'
            })

        return {
            'passed': len(issues) == 0,
            'issues': issues
        }

# Register custom gauntlet
manager = GauntletManager()
manager.register_gauntlet(SecurityRedTeamGauntlet())

# Use it
gauntlet_result = manager.run_gauntlet(
    gauntlet_name="Security-RedTeam-Gauntlet",
    content=solution_code,
    context={'sub_problem_id': 'sub-001'}
)

print(f"Security check passed: {gauntlet_result['approved']}")
print(f"Issues found: {len(gauntlet_result['issues'])}")
```

---

## Advanced Feature 4: Caching and Performance

### Enabling OpenEvolve Caching

```python
# example_caching.py
from openevolve_client import OpenEvolveClient
import hashlib

# Create client with caching
client = OpenEvolveClient(
    enable_cache=True,
    cache_ttl=3600,  # Cache for 1 hour
    cache_dir="./cache/openevolve"
)

# Cache key generation
def generate_cache_key(problem_statement: str, strategy: str) -> str:
    """Generate cache key for decomposition"""

    content = f"{strategy}:{problem_statement}"
    return hashlib.sha256(content.encode()).hexdigest()

# Check cache before decomposing
cache_key = generate_cache_key(problem_statement, "semantic")

if client.is_cached(cache_key):
    print("✓ Loading from cache...")
    result = client.get_from_cache(cache_key)
else:
    print("Decomposing (not in cache)...")
    result = engine.decompose(problem, strategy="semantic")
    client.save_to_cache(cache_key, result)
```

### Batch Processing

```python
# example_batch_processing.py
from decomposition_mcp_tools import decompose_problem_into_sub_problems

# Multiple problems to decompose
problems = [
    "Build a REST API for user management",
    "Create a data pipeline for analytics",
    "Design a microservices architecture",
    "Implement real-time notifications"
]

# Batch decompose (without evolution for speed)
results = []
for i, problem in enumerate(problems, 1):
    print(f"[{i}/{len(problems)}] Decomposing: {problem[:50]}...")

    result = decompose_problem_into_sub_problems(
        problem_statement=problem,
        use_evolution=False  # Faster without evolution
    )

    results.append(result)

print(f"\n✓ Batch complete: {len(results)} problems decomposed")
```

---

## Advanced Feature 5: Large-Scale Decomposition

### Hierarchical Decomposition for Large Problems

```python
# example_large_scale.py
from decomposition_engine import DecompositionEngine

def hierarchical_decompose_large_problem(problem, max_sub_problems=15):
    """Decompose large problems hierarchically"""

    print(f"Decomposing large problem (target: {max_sub_problems} sub-problems)")

    # Level 1: Top-level decomposition
    engine = DecompositionEngine()
    result = engine.decompose(problem, strategy="hierarchical")

    top_level = result.sub_problems
    print(f"Level 1: {len(top_level)} top-level components")

    # Level 2: Decompose large components further
    all_sub_problems = []

    for component in top_level:
        if component.complexity_score.overall_complexity > 7:
            # Need to decompose further
            print(f"  Decomposing: {component.title}...")

            sub_result = engine.decompose(
                problem,
                strategy="semantic"
            )

            # Add sub-sub-problems
            for sub_sp in sub_result.sub_problems:
                sub_sp.id = f"{component.id}-{sub_sp.id}"
                sub_sp.parent_id = component.id
                sub_sp.dependencies = [
                    f"{component.id}-{dep}" for dep in sub_sp.dependencies
                ]
                all_sub_problems.append(sub_sp)
        else:
            # Keep as-is
            all_sub_problems.append(component)

    print(f"Total sub-problems: {len(all_sub_problems)}")
    return all_sub_problems

# Use on large problem
large_problem = ProblemDefinition(
    id="large-001",
    title="Enterprise Resource Planning System",
    description="Complete ERP system with modules for: finance, HR, inventory, sales, CRM, accounting...",
    # ... other fields
)

sub_problems = hierarchical_decompose_large_problem(large_problem)
```

---

## Advanced Feature 6: Monitoring and Logging

### Comprehensive Logging

```python
# example_logging.py
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('decomposition.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('decomposition')

def log_decomposition_workflow(problem, result):
    """Log decomposition workflow details"""

    logger.info("=" * 80)
    logger.info(f"DECOMPOSITION WORKFLOW: {problem.title}")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info(f"Domain: {problem.domain_context.domain}")
    logger.info(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    logger.info("=" * 80)

    logger.info(f"Generated {len(result.sub_problems)} sub-problems")

    for sp in result.sub_problems:
        logger.info(f"  - {sp.title}")
        logger.info(f"    Type: {sp.type.value}")
        logger.info(f"    Priority: {sp.priority}/10")
        logger.info(f"    Complexity: {sp.complexity_score.overall_complexity}/10")
        logger.info(f"    Effort: {sp.estimated_effort}h")
        if sp.dependencies:
            logger.info(f"    Dependencies: {', '.join(sp.dependencies)}")

    logger.info("=" * 80)

# Use it
result = engine.decompose(problem, strategy="semantic")
log_decomposition_workflow(problem, result)
```

### Performance Metrics

```python
# example_metrics.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Context manager for timing"""
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

with timer("Solving (all sub-problems)"):
    for sp in result['sub_problems']:
        with timer(f"  {sp['title']}"):
            solution = solve_sub_problem_with_team(
                sub_problem_id=sp['id'],
                sub_problem_description=sp['description'],
                team_name="Default-Blue"
            )

print("\n=== Performance Summary ===")
print(f"Total sub-problems: {len(result['sub_problems'])}")
print(f"Average solve time: {total_time / len(result['sub_problems']):.2f}s")
```

---

## Summary

In this tutorial, you learned:

✓ Evolutionary optimization techniques
✓ Parallel execution strategies
✓ Custom gauntlet creation
✓ Caching and performance optimization
✓ Large-scale decomposition
✓ Monitoring and logging

---

## Complete Advanced Example

```python
# advanced_complete_example.py
"""
Advanced example combining all features:
- Evolutionary optimization
- Parallel execution
- Custom gauntlets
- Caching
- Performance monitoring
"""

from concurrent.futures import ThreadPoolExecutor
from decomposition_mcp_tools import (
    decompose_problem_into_sub_problems,
    solve_sub_problem_with_team
)
from openevolve_client import OpenEvolveClient
import time

class AdvancedDecompositionWorkflow:
    """Advanced decomposition workflow with all features"""

    def __init__(self):
        self.client = OpenEvolveClient(enable_cache=True)
        self.engine = DecompositionEngine()
        self.metrics = {}

    def run(self, problem_statement: str, max_workers=4):
        """Run complete advanced workflow"""

        start_time = time.time()

        # Decompose
        print("[1/4] Decomposing...")
        decomp = decompose_problem_into_sub_problems(
            problem_statement=problem_statement,
            use_evolution=True
        )

        # Find independent sub-problems
        independent = [sp for sp in decomp['sub_problems'] if not sp['dependencies']]
        print(f"[2/4] Solving {len(independent)} independent sub-problems in parallel...")

        # Solve in parallel with evolution
        solutions = self._solve_parallel(independent, max_workers)

        # Custom gauntlet validation
        print(f"[3/4] Running custom gauntlets...")
        validated = self._custom_validate(solutions)

        # Assemble
        print(f"[4/4] Assembling final solution...")
        final = self._assemble(validated)

        elapsed = time.time() - start_time
        self.metrics['total_time'] = elapsed
        self.metrics['sub_problems'] = len(decomp['sub_problems'])
        self.metrics['approved'] = len(final)

        return final

    def _solve_parallel(self, sub_problems, max_workers):
        """Solve sub-problems in parallel with evolution"""

        def solve(sp):
            return solve_sub_problem_with_team(
                sub_problem_id=sp['id'],
                sub_problem_description=sp['description'],
                team_name="Default-Blue",
                execution_method="traditional",
                use_evolution=True,
                evolution_iterations=100
            )

        solutions = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(solve, sp): sp for sp in sub_problems}

            for future in futures:
                sp = futures[future]
                try:
                    solutions[sp['id']] = future.result()
                    print(f"  ✓ {sp['title']}")
                except Exception as e:
                    print(f"  ✗ {sp['title']}: {e}")

        return solutions

    def _custom_validate(self, solutions):
        """Custom validation logic"""
        # Implementation here
        return solutions

    def _assemble(self, validated):
        """Assemble final solution"""
        # Implementation here
        return validated

# Use it
workflow = AdvancedDecompositionWorkflow()
final = workflow.run("Build enterprise-grade authentication system")

print(f"\n✓ Complete in {workflow.metrics['total_time']:.1f}s")
print(f"  Sub-problems: {workflow.metrics['sub_problems']}")
print(f"  Approved: {workflow.metrics['approved']}")
```

---

## Next Steps

Congratulations! You've mastered all decomposition features. Explore:
- [API Reference](../OPNEEVOLVE_DECOMPOSITION_API_REFERENCE.md)
- [Example Gallery](../EXAMPLES/)
- [Best Practices](../BEST_PRACTICES.md)

---

**Tutorial Version:** 1.0.0
**Last Updated:** 2025-01-03
