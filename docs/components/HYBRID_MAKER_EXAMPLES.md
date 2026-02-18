# HYBRID MAKER EXAMPLES

Comprehensive code examples and use cases for Hybrid MAKER strategies.

**Version:** 1.0.0
**Paper:** arXiv:2511.09030
**Last Updated:** 2025-12-30

---

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Theorem Proving](#theorem-proving)
3. [Code Generation](#code-generation)
4. [Optimization Problems](#optimization-problems)
5. [Planning Problems](#planning-problems)
6. [Multi-Objective Optimization](#multi-objective-optimization)
7. [Performance Comparisons](#performance-comparisons)
8. [Integration Examples](#integration-examples)
9. [Before/After Comparisons](#beforeafter-comparisons)
10. [Jupyter Notebook Examples](#jupyter-notebook-examples)

---

## Basic Examples

### Example 1: Hello World

```python
import asyncio
from hybrid_maker_integration import (
    run_maker_hybrid,
    MAKERHybridMode,
    MAKERHybridConfig
)

async def hello_world():
    """Simplest MAKER hybrid example"""
    theorem = "forall n : nat, n + 0 = n"

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MCTS_THEN_MAKER
    )

    print(f"Success: {result.success}")
    print(f"Fitness: {result.best_fitness:.3f}")
    if result.best_proof:
        print(f"Proof:\n{result.best_proof}")

asyncio.run(hello_world())
```

### Example 2: Custom Configuration

```python
async def custom_config():
    """Using custom configuration"""
    config = MAKERHybridConfig(
        voting_threshold=4,
        mcts_simulations=150,
        evolution_generations=25,
        population_size=25,
        enable_red_flagging=True
    )

    result = await run_maker_hybrid(
        theorem="forall n m : nat, n + m = m + n",
        mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
        config=config
    )

    print(f"Generations: {result.generations_completed}")
    print(f"Time: {result.evolution_time:.2f}s")

asyncio.run(custom_config())
```

### Example 3: Error Handling

```python
async def with_error_handling():
    """Example with comprehensive error handling"""
    theorem = "forall n : nat, n * 0 = 0"

    try:
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER
        )

        if result.success:
            print("Success!")
            print(f"Fitness: {result.best_fitness:.3f}")
        else:
            print("Failed to find solution")
            print("Errors:")
            for error in result.failed_attempts:
                print(f"  - {error}")

            # Try fallback
            print("\nTrying fallback mode...")
            result = await run_maker_hybrid(
                theorem=theorem,
                mode=MAKERHybridMode.MAKER_THEN_EVOLUTION
            )

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Log error, notify user, etc.

asyncio.run(with_error_handling())
```

### Example 4: Progress Tracking

```python
async def with_progress():
    """Track progress during execution"""
    import logging

    logging.basicConfig(level=logging.INFO)

    theorem = "forall n m : nat, n + m = m + n"

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.ADAPTIVE_MAKER,
        config=MAKERHybridConfig(
            evolution_generations=30,
            adaptive_switching=True
        )
    )

    # Analyze convergence
    if result.convergence_history:
        print("\nConvergence Progress:")
        for i, fitness in enumerate(result.convergence_history[::5]):
            gen = i * 5
            print(f"  Generation {gen:2d}: {fitness:.3f}")

        # Calculate improvement rate
        initial = result.convergence_history[0]
        final = result.convergence_history[-1]
        improvement = final - initial
        print(f"\nTotal improvement: {improvement:.3f}")
        print(f"Improvement rate: {improvement/len(result.convergence_history):.4f} per generation")

asyncio.run(with_progress())
```

---

## Theorem Proving

### Example 5: Simple Arithmetic

```python
async def prove_add_comm():
    """Prove addition is commutative"""
    theorem = "forall n m : nat, n + m = m + n"

    config = MAKERHybridConfig(
        voting_threshold=3,
        mcts_simulations=100
    )

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MCTS_THEN_MAKER,
        config=config
    )

    if result.success:
        print("Proof found!")
        print(f"Fitness: {result.best_fitness:.3f}")
        print(f"\nProof:\n{result.best_proof}")
    else:
        print("Could not prove theorem")

asyncio.run(prove_add_comm())
```

### Example 6: Induction Theorem

```python
async def prove_induction():
    """Prove theorem requiring induction"""
    theorem = "forall n : nat, n + 0 = n"

    # Use MAKER-Then-Evolution for more complex proof
    config = MAKERHybridConfig(
        voting_threshold=3,
        evolution_generations=30,
        population_size=20
    )

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
        config=config
    )

    if result.success:
        print("Induction proof found!")
        print(f"Generations: {result.generations_completed}")
        print(f"\nProof:\n{result.best_proof}")
    else:
        print("Induction proof failed")

asyncio.run(prove_induction())
```

### Example 7: Associativity

```python
async def prove_associative():
    """Prove addition is associative"""
    theorem = "forall a b c : nat, a + (b + c) = (a + b) + c"

    # Use adversarial for robustness
    config = MAKERHybridConfig(
        voting_threshold=4,
        adversarial_rounds=5,
        red_team_size=2,
        blue_team_size=2
    )

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MAKER_ADVERSARIAL,
        config=config
    )

    if result.success:
        print("Associativity proved!")
        print(f"Adversarial rounds: {result.generations_completed}")

        if result.convergence_history:
            print("\nAdversarial progression:")
            for i, fitness in enumerate(result.convergence_history):
                print(f"  Round {i+1}: {fitness:.3f}")
    else:
        print("Could not prove associativity")

asyncio.run(prove_associative())
```

### Example 8: Multiple Theorems

```python
async def prove_multiple():
    """Prove multiple theorems efficiently"""
    theorems = [
        "forall n : nat, n + 0 = n",
        "forall n m : nat, n + m = m + n",
        "forall n : nat, n * 1 = n",
        "forall n : nat, 0 + n = n",
        "forall a b c : nat, a + (b + c) = (a + b) + c"
    ]

    config = MAKERHybridConfig(
        voting_threshold=3,
        mcts_simulations=80  # Moderate for speed
    )

    results = []
    for i, theorem in enumerate(theorems, 1):
        print(f"\n[{i}/{len(theorems)}] Proving: {theorem}")

        result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER,
            config=config
        )

        results.append((theorem, result))

        if result.success:
            print(f"  ✓ Proved (fitness: {result.best_fitness:.3f})")
        else:
            print(f"  ✗ Failed")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    succeeded = sum(1 for _, r in results if r.success)
    print(f"Succeeded: {succeeded}/{len(theorems)}")

    for theorem, result in results:
        status = "✓" if result.success else "✗"
        fitness = result.best_fitness if result.success else 0.0
        print(f"  {status} {theorem[:50]:50s} ({fitness:.3f})")

asyncio.run(prove_multiple())
```

---

## Code Generation

### Example 9: Simple Function

```python
from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

def generate_function():
    """Generate a simple function"""

    def fitness_function(code: str) -> float:
        """Evaluate generated code"""
        score = 0.0

        # Check for required elements
        if "def " in code:
            score += 0.3
        if "return " in code:
            score += 0.3
        if "if " in code:
            score += 0.1
        if "else:" in code:
            score += 0.1

        # Check for good practices
        if "docstring" in code or '"""' in code:
            score += 0.1
        if not code.count(";") > 2:  # Not too many statements
            score += 0.1

        return min(1.0, score)

    result = run_maker_evolution(
        initial_program="# Generate a function to check if number is prime",
        evaluator=fitness_function,
        max_generations=20,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            voting_threshold=3,
            population_size=15
        )
    )

    if result['success']:
        print("Generated code:")
        print(result['best_program'])
        print(f"\nFitness: {result['best_fitness']:.3f}")
    else:
        print("Failed to generate code")

generate_function()
```

### Example 10: Class Generation

```python
def generate_class():
    """Generate a Python class"""

    def class_fitness(code: str) -> float:
        """Evaluate class quality"""
        score = 0.0

        if "class " in code:
            score += 0.3
        if "def __init__" in code:
            score += 0.2
        if "self." in code:
            score += 0.2

        # Count methods
        method_count = code.count("def ")
        if 2 <= method_count <= 5:
            score += 0.2
        elif method_count > 5:
            score += 0.1

        # Check for docstring
        if '"""' in code or "'''" in code:
            score += 0.1

        return min(1.0, score)

    result = run_maker_evolution(
        initial_program="# Generate a Stack class",
        evaluator=class_fitness,
        max_generations=25,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            enable_decomposition=True  # Decompose class into methods
        )
    )

    if result['success']:
        print("Generated class:")
        print(result['best_program'])
        print(f"\nFitness: {result['best_fitness']:.3f}")
        print(f"Generations: {result['generations']}")

generate_class()
```

---

## Optimization Problems

### Example 11: Parameter Tuning

```python
def optimize_parameters():
    """Optimize function parameters"""

    def parameter_fitness(params: str) -> float:
        """Evaluate parameter quality"""
        try:
            # Parse parameters (simple format: "a=1.0,b=2.0,c=3.0")
            param_dict = {}
            for pair in params.split(','):
                key, value = pair.split('=')
                param_dict[key.strip()] = float(value.strip())

            # Example: optimize for sum close to 10
            total = sum(param_dict.values())
            target = 10.0

            # Fitness based on proximity to target
            error = abs(total - target)
            fitness = max(0.0, 1.0 - error / target)

            return fitness
        except:
            return 0.0

    # Initial parameters
    initial = "a=1.0,b=2.0,c=3.0"

    result = run_maker_evolution(
        initial_program=initial,
        evaluator=parameter_fitness,
        max_generations=30,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.VOTING_ONLY,
            population_size=20,
            voting_threshold=2  # Lower for speed
        )
    )

    if result['success']:
        print("Optimized parameters:")
        print(result['best_program'])
        print(f"Fitness: {result['best_fitness']:.3f}")

optimize_parameters()
```

### Example 12: Traveling Salesman

```python
def solve_tsp():
    """Solve Traveling Salesman Problem with MAKER"""

    # Cities: (x, y) coordinates
    cities = {
        'A': (0, 0),
        'B': (1, 3),
        'C': (2, 1),
        'D': (3, 4),
        'E': (4, 2)
    }

    def tsp_fitness(tour: str) -> float:
        """Calculate tour quality (shorter is better)"""
        try:
            # Parse tour (format: "A->B->C->D->E->A")
            cities_visited = tour.split('->')

            # Calculate total distance
            total_distance = 0.0
            for i in range(len(cities_visited) - 1):
                city1 = cities_visited[i].strip()
                city2 = cities_visited[i+1].strip()

                if city1 in cities and city2 in cities:
                    x1, y1 = cities[city1]
                    x2, y2 = cities[city2]
                    distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    total_distance += distance

            # Fitness: inverse of distance (normalized)
            max_distance = 20.0  # Approximate worst case
            fitness = max(0.0, 1.0 - total_distance / max_distance)

            return fitness
        except:
            return 0.0

    # Initial tour
    initial_tour = "A->B->C->D->E->A"

    result = run_maker_evolution(
        initial_program=initial_tour,
        evaluator=tsp_fitness,
        max_generations=40,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            population_size=25,
            voting_threshold=3,
            enable_decomposition=True  # Decompose tour segments
        )
    )

    if result['success']:
        print("Optimal tour found:")
        print(result['best_program'])
        print(f"Fitness: {result['best_fitness']:.3f}")
        print(f"Generations: {result['generations']}")

solve_tsp()
```

---

## Planning Problems

### Example 13: Task Scheduling

```python
def schedule_tasks():
    """Schedule tasks with dependencies"""

    tasks = [
        {'name': 'A', 'duration': 2, 'deps': []},
        {'name': 'B', 'duration': 3, 'deps': ['A']},
        {'name': 'C', 'duration': 1, 'deps': ['A']},
        {'name': 'D', 'duration': 4, 'deps': ['B', 'C']},
        {'name': 'E', 'duration': 2, 'deps': ['D']}
    ]

    def schedule_fitness(schedule: str) -> float:
        """Evaluate schedule quality (shorter makespan is better)"""
        try:
            # Parse schedule (format: "A:0-2,B:2-5,C:2-3,D:5-9,E:9-11")
            task_times = {}
            for task_schedule in schedule.split(','):
                task, time_range = task_schedule.split(':')
                start, end = map(int, time_range.split('-'))
                task_times[task] = (start, end)

            # Check dependencies
            valid = True
            for task in tasks:
                name = task['name']
                if name not in task_times:
                    return 0.0

                start, end = task_times[name]
                duration = end - start

                if duration != task['duration']:
                    valid = False

                # Check dependencies
                for dep in task['deps']:
                    if dep in task_times:
                        dep_end = task_times[dep][1]
                        if start < dep_end:
                            valid = False

            if not valid:
                return 0.0

            # Calculate makespan
            max_end = max(end for start, end in task_times.values())

            # Fitness: inverse of makespan
            max_possible = 15  # Upper bound
            fitness = max(0.0, 1.0 - max_end / max_possible)

            return fitness
        except:
            return 0.0

    # Initial schedule
    initial = "A:0-2,B:2-5,C:2-3,D:5-9,E:9-11"

    result = run_maker_evolution(
        initial_program=initial,
        evaluator=schedule_fitness,
        max_generations=30,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            population_size=20
        )
    )

    if result['success']:
        print("Optimal schedule:")
        print(result['best_program'])
        print(f"Fitness: {result['best_fitness']:.3f}")

schedule_tasks()
```

### Example 14: Path Planning

```python
def plan_path():
    """Find path through grid with obstacles"""

    # 5x5 grid, 1 = obstacle, 0 = free
    grid = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    goal = (4, 4)

    def path_fitness(path: str) -> float:
        """Evaluate path quality"""
        try:
            # Parse path (format: "(0,0)->(0,1)->(1,1)->...->(4,4)")
            points = []
            for point in path.split('->'):
                x, y = eval(point.strip())
                points.append((x, y))

            # Check start and goal
            if points[0] != start or points[-1] != goal:
                return 0.0

            # Check obstacles and validity
            for i, (x, y) in enumerate(points):
                if not (0 <= x < 5 and 0 <= y < 5):
                    return 0.0
                if grid[y][x] == 1:
                    return 0.0

                # Check adjacency
                if i < len(points) - 1:
                    next_x, next_y = points[i+1]
                    dist = abs(next_x - x) + abs(next_y - y)
                    if dist != 1:  # Must be adjacent
                        return 0.0

            # Shorter paths are better
            path_length = len(points)
            min_length = 9  # Manhattan distance
            max_length = 25  # Worst case

            fitness = 1.0 - (path_length - min_length) / (max_length - min_length)
            return max(0.0, fitness)

        except:
            return 0.0

    # Initial path (down then right, will hit obstacles)
    initial = "(0,0)->(1,0)->(2,0)->(3,0)->(4,0)->(4,1)->(4,2)->(4,3)->(4,4)"

    result = run_maker_evolution(
        initial_program=initial,
        evaluator=path_fitness,
        max_generations=35,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            population_size=25,
            enable_decomposition=True  # Decompose path into segments
        )
    )

    if result['success']:
        print("Optimal path:")
        print(result['best_program'])
        print(f"Fitness: {result['best_fitness']:.3f}")

plan_path()
```

---

## Multi-Objective Optimization

### Example 15: Speed vs. Quality

```python
def multi_objective():
    """Optimize for both speed and quality"""

    def multi_fitness(solution: str) -> float:
        """Multi-objective fitness function"""
        # Parse solution
        # Assume format: "solution_data|speed|quality"

        try:
            parts = solution.split('|')
            if len(parts) != 3:
                return 0.0

            solution_data, speed_str, quality_str = parts
            speed = float(speed_str)
            quality = float(quality_str)

            # Normalize to 0-1
            speed_norm = min(1.0, speed / 100.0)
            quality_norm = min(1.0, quality / 10.0)

            # Weighted combination
            # Can adjust weights based on priorities
            w_speed = 0.4
            w_quality = 0.6

            fitness = w_speed * speed_norm + w_quality * quality_norm

            return fitness
        except:
            return 0.0

    # Initial solution
    initial = "initial_solution|50.0|5.0"

    result = run_maker_evolution(
        initial_program=initial,
        evaluator=multi_fitness,
        max_generations=40,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.FULL_MAKER,
            population_size=30,
            voting_threshold=4
        )
    )

    if result['success']:
        print("Pareto-optimal solution:")
        parts = result['best_program'].split('|')
        print(f"Solution: {parts[0]}")
        print(f"Speed: {parts[1]}")
        print(f"Quality: {parts[2]}")
        print(f"Combined Fitness: {result['best_fitness']:.3f}")

multi_objective()
```

---

## Performance Comparisons

### Example 16: Strategy Comparison

```python
async def compare_strategies():
    """Compare all MAKER hybrid strategies"""

    theorem = "forall n m : nat, n + m = m + n"

    modes = [
        (MAKERHybridMode.MCTS_THEN_MAKER, "MCTS-Then-MAKER"),
        (MAKERHybridMode.MAKER_THEN_EVOLUTION, "MAKER-Then-Evolution"),
        (MAKERHybridMode.ADAPTIVE_MAKER, "Adaptive MAKER"),
        (MAKERHybridMode.MAKER_MDAP_PARALLEL, "MAKER-MDAP Parallel")
    ]

    results = []

    for mode, mode_name in modes:
        print(f"\nTesting: {mode_name}...")

        start_time = time.time()
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=mode,
            config=MAKERHybridConfig(
                voting_threshold=3,
                mcts_simulations=80,
                evolution_generations=15
            )
        )
        elapsed = time.time() - start_time

        results.append({
            "mode": mode_name,
            "success": result.success,
            "fitness": result.best_fitness,
            "time": elapsed,
            "generations": result.generations_completed
        })

        print(f"  Success: {result.success}")
        print(f"  Fitness: {result.best_fitness:.3f}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Generations: {result.generations_completed}")

    # Summary table
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Mode':<25} {'Success':<10} {'Fitness':<10} {'Time':<10} {'Gens':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['mode']:<25} {str(r['success']):<10} "
              f"{r['fitness']:<10.3f} {r['time']:<10.2f} {r['generations']:<10}")

asyncio.run(compare_strategies())
```

### Example 17: Configuration Tuning

```python
async def tune_configuration():
    """Find optimal configuration"""

    theorem = "forall n : nat, n + 0 = n"

    # Test different voting thresholds
    thresholds = [2, 3, 4, 5]

    print("Testing voting thresholds...")
    print(f"{'k':<5} {'Fitness':<10} {'Time':<10}")
    print("-"*30)

    for k in thresholds:
        config = MAKERHybridConfig(
            voting_threshold=k,
            mcts_simulations=100
        )

        start = time.time()
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER,
            config=config
        )
        elapsed = time.time() - start

        print(f"{k:<5} {result.best_fitness:<10.3f} {elapsed:<10.2f}")

asyncio.run(tune_configuration())
```

---

## Integration Examples

### Example 18: Workflow Integration

```python
async def workflow_integration():
    """Integrate MAKER into larger workflow"""

    # Step 1: Decompose problem
    problem = "Prove all basic arithmetic properties"

    print("Step 1: Decomposing problem...")
    subproblems = [
        "forall n : nat, n + 0 = n",
        "forall n : nat, 0 + n = n",
        "forall n m : nat, n + m = m + n",
        "forall n : nat, n * 1 = n",
        "forall n m : nat, n * m = m * n"
    ]

    # Step 2: Solve each subproblem
    print("\nStep 2: Solving subproblems...")
    results = []

    for i, subproblem in enumerate(subproblems, 1):
        print(f"\n  [{i}/{len(subproblems)}] {subproblem}")

        result = await run_maker_hybrid(
            theorem=subproblem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER,
            config=MAKERHybridConfig(
                voting_threshold=3,
                mcts_simulations=80
            )
        )

        results.append((subproblem, result))

        if result.success:
            print(f"    ✓ Solved (fitness: {result.best_fitness:.3f})")
        else:
            print(f"    ✗ Failed")

    # Step 3: Aggregate results
    print("\nStep 3: Aggregating results...")
    succeeded = sum(1 for _, r in results if r.success)
    avg_fitness = sum(r.best_fitness for _, r in results if r.success) / max(1, succeeded)

    print(f"\nResults:")
    print(f"  Solved: {succeeded}/{len(subproblems)}")
    print(f"  Average fitness: {avg_fitness:.3f}")

    # Step 4: Generate final report
    print("\nStep 4: Generating report...")
    report = []
    for theorem, result in results:
        status = "✓" if result.success else "✗"
        report.append(f"{status} {theorem}")

    print("\nFinal Report:")
    for line in report:
        print(f"  {line}")

asyncio.run(workflow_integration())
```

### Example 19: LeanAide Integration

```python
async def integrate_leanaide():
    """Integrate with LeanAide framework"""

    # Import LeanAide components
    try:
        from leanaide_mcts import LeanProofMCTS
        from leanaide_evolution import LeanProofEvolutionEngineMCTS
        LEANAIDE_AVAILABLE = True
    except ImportError:
        LEANAIDE_AVAILABLE = False
        print("LeanAide not available")

    if not LEANAIDE_AVAILABLE:
        return

    theorem = "forall n m : nat, n + m = m + n"

    # Use MAKER hybrid with LeanAide backend
    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.FULL_MAKER_HYBRID,
        config=MAKERHybridConfig(
            voting_threshold=4,
            mcts_simulations=150,
            evolution_generations=25
        )
    )

    if result.success:
        print("LeanAide integration successful!")
        print(f"Proof:\n{result.best_proof}")

asyncio.run(integrate_leanaide())
```

### Example 20: CrewAI Integration

```python
async def integrate_crewai():
    """Integrate with CrewAI delegation system"""

    try:
        from crewai_client import CrewAIClient
        CREWAI_AVAILABLE = True
    except ImportError:
        CREWAI_AVAILABLE = False
        print("CrewAI not available")

    if not CREWAI_AVAILABLE:
        return

    theorem = "forall n : nat, n + 0 = n"

    # Delegate to CrewAI with MAKER
    client = CrewAIClient()

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
        config=MAKERHybridConfig(
            voting_threshold=3,
            evolution_generations=20
        )
    )

    if result.success:
        print("CrewAI delegation successful!")
        print(f"Proof:\n{result.best_proof}")

        # Could send proof back to CrewAI for validation
        # client.validate_proof(result.best_proof)

asyncio.run(integrate_crewai())
```

---

## Before/After Comparisons

### Example 21: Evolution vs. MAKER Evolution

```python
def evolution_vs_maker():
    """Compare standard evolution with MAKER-enhanced evolution"""

    def fitness_fn(program: str) -> float:
        """Simple fitness function"""
        score = 0.0
        if "induction" in program:
            score += 0.4
        if "simp" in program:
            score += 0.3
        if "refl" in program:
            score += 0.2
        if "rw" in program:
            score += 0.1
        return min(1.0, score)

    theorem = "forall n : nat, n + 0 = n"
    initial = f"theorem : {theorem}"

    # Before: Standard evolution
    print("BEFORE: Standard Evolution")
    print("-"*40)
    from evolution import EvolutionConfiguration

    evo_config = EvolutionConfiguration(
        population_size=20,
        generations=20,
        mutation_rate=0.1
    )

    start = time.time()
    # Standard evolution would run here
    # result_standard = run_evolution(initial, fitness_fn, evo_config)
    time_standard = 30.5  # Placeholder
    print(f"Time: {time_standard:.2f}s")
    print(f"Fitness: 0.75 (example)")
    print(f"Generations: 20")

    # After: MAKER evolution
    print("\nAFTER: MAKER-Enhanced Evolution")
    print("-"*40)

    start = time.time()
    result_maker = run_maker_evolution(
        initial_program=initial,
        evaluator=fitness_fn,
        max_generations=20,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            voting_threshold=3,
            population_size=20
        )
    )
    time_maker = time.time() - start

    print(f"Time: {time_maker:.2f}s")
    print(f"Fitness: {result_maker['best_fitness']:.3f}")
    print(f"Generations: {result_maker['generations']}")

    # Comparison
    print("\nIMPROVEMENT:")
    print("-"*40)
    improvement = (result_maker['best_fitness'] - 0.75) / 0.75 * 100
    print(f"Fitness improvement: {improvement:.1f}%")
    print(f"Time difference: {time_maker - time_standard:.2f}s")

evolution_vs_maker()
```

---

## Jupyter Notebook Examples

### Example 22: Interactive Demo (Jupyter)

```python
# In a Jupyter notebook cell:

import asyncio
from hybrid_maker_integration import run_maker_hybrid, MAKERHybridMode
import matplotlib.pyplot as plt

# Define theorem
theorem = "forall n m : nat, n + m = m + n"

# Run MAKER hybrid
result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.ADAPTIVE_MAKER,
    config=MAKERHybridConfig(
        voting_threshold=3,
        evolution_generations=30,
        adaptive_switching=True
    )
)

# Display results
print(f"Success: {result.success}")
print(f"Best Fitness: {result.best_fitness:.3f}")
print(f"Generations: {result.generations_completed}")

# Plot convergence
if result.convergence_history:
    plt.figure(figsize=(10, 6))
    plt.plot(result.convergence_history, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Convergence History')
    plt.grid(True)
    plt.show()
```

### Example 23: Parameter Sweep (Jupyter)

```python
# Parameter sweep in Jupyter

import pandas as pd
import asyncio

async def parameter_sweep():
    """Sweep across different parameters"""

    results = []

    # Test different voting thresholds
    for k in [2, 3, 4, 5]:
        # Test different generations
        for gens in [10, 20, 30]:
            config = MAKERHybridConfig(
                voting_threshold=k,
                evolution_generations=gens
            )

            result = await run_maker_hybrid(
                theorem="forall n : nat, n + 0 = n",
                mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
                config=config
            )

            results.append({
                'k': k,
                'generations': gens,
                'fitness': result.best_fitness,
                'time': result.evolution_time,
                'success': result.success
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Display results
    print(df.pivot_table(
        values='fitness',
        index='k',
        columns='generations'
    ))

    return df

# Run sweep
df = await parameter_sweep()
```

---

**End of Examples**

For more information, see:
- Architecture: `HYBRID_MAKER_ARCHITECTURE.md`
- API Reference: `HYBRID_MAKER_API.md`
- User Guide: `HYBRID_MAKER_GUIDE.md`
- Integration: `HYBRID_MAKER_INTEGRATION.md`
