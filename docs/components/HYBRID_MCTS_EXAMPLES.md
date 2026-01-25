# Hybrid MCTS-Evolution Examples

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Intermediate Examples](#intermediate-examples)
3. [Advanced Examples](#advanced-examples)
4. [Integration Examples](#integration-examples)
5. [Real-World Scenarios](#real-world-scenarios)
6. [Complete Workflows](#complete-workflows)

---

## Basic Examples

### Example 1: Simple Evolved Policies

Evolve a policy for simple arithmetic theorems.

```python
import asyncio
from hybrid_mcts import (
    PolicyEvolutionEngine,
    HybridMCTSPresets,
    EvolvedPolicyMCTS
)

async def main():
    # 1. Prepare training data
    training_theorems = [
        "For all n, n + 0 = n",
        "For all a b, a + b = b + a",
        "For all a b c, (a + b) + c = a + (b + c)",
        "For all n, n * 1 = n",
        "For all a b, a * b = b * a"
    ]

    # 2. Configure for fast training
    config = HybridMCTSPresets.fast()
    config.policy_training_generations = 30
    config.policy_population_size = 20

    # 3. Train policy
    engine = PolicyEvolutionEngine(config)
    best_policy = await engine.evolve_policies(
        test_theorems=training_theorems,
        generations=30
    )

    print(f"Policy fitness: {best_policy.fitness:.3f}")

    # 4. Use policy for new theorem
    mcts_config = HybridMCTSPresets.fast()
    mcts = EvolvedPolicyMCTS(best_policy, mcts_config.mcts_config)

    result = await mcts.search("For all n, 0 + n = n")

    if result.success:
        print("Proof found!")
        print(result.best_proof.lean_code)
    else:
        print("Proof not found")

    # 5. Save policy for reuse
    engine.save_policy(best_policy, "arithmetic_policy.json")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
Generation 1: Best fitness=0.400
Generation 10: Best fitness=0.720
Generation 20: Best fitness=0.880
Generation 30: Best fitness=0.920
Policy fitness: 0.920
Proof found!
theorem add_zero_right (n : Nat) : 0 + n = n := by
  rw [Nat.add_comm]
  rw [Nat.add_zero]
```

---

### Example 2: Evolutionary Nodes for Complex Proofs

Use evolutionary nodes for a complex inductive proof.

```python
import asyncio
from hybrid_mcts import (
    EvolutionaryMCTS,
    HybridMCTSPresets,
    HybridMCTSApproach
)

async def main():
    # Complex theorem requiring induction
    complex_theorem = """
    For all n m,
      (n + m) * (n + m) = n*n + 2*n*m + m*m
    """

    # Configure for evolutionary nodes
    config = HybridMCTSPresets.evolutionary_nodes()
    config.node_population_size = 15
    config.node_evolution_frequency = 5
    config.sequence_length_range = (5, 15)
    config.mcts_simulations = 2000

    # Initialize engine
    engine = EvolutionaryMCTS(config)

    # Search with progress tracking
    def on_iteration(iteration, metrics):
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: "
                  f"Best={metrics.best_fitness:.3f}, "
                  f"Nodes={metrics.nodes_visited}")

    result = await engine.search(
        complex_theorem,
        time_budget=120.0,
        progress_callback=on_iteration
    )

    print(f"\nSuccess: {result.success}")
    print(f"Approach: {result.approach_used}")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Iterations: {result.iterations_completed}")

    if result.success:
        print("\nProof:")
        print(result.best_proof.lean_code)

if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
Iteration 0: Best=0.500, Nodes=50
Iteration 100: Best=0.650, Nodes=1250
Iteration 200: Best=0.780, Nodes=2450
Iteration 300: Best=0.850, Nodes=3680
Iteration 400: Best=0.920, Nodes=4890

Success: True
Approach: HybridMCTSApproach.EVOLUTIONARY_NODES
Time: 118.43s
Iterations: 489

Proof:
theorem mul_add (n m : Nat) :
  (n + m) * (n + m) = n * n + 2 * n * m + m * m := by
  induction n
  case zero =>
    simp
    ring
  case succ n ih =>
    simp [ih]
    ring
```

---

### Example 3: Coevolution for Domain Adaptation

Coevolve proof trees and evaluators for specific domain.

```python
import asyncio
from hybrid_mcts import (
    TreeCoevolution,
    HybridMCTSPresets,
    ProofDecisionTree,
    MCTreeEvaluator
)

async def main():
    # Domain-specific theorems (e.g., algebra)
    domain_theorems = [
        "For all x, x * 0 = 0",
        "For all x y, x * (y + z) = x*y + x*z",
        "For all x y z, (x + y) * z = x*z + y*z",
        "For all x y z, x * (y * z) = (x * y) * z",
        "For all x, x * x = x²"
    ]

    # Configure coevolution
    config = HybridMCTSPresets.coevolution()
    config.tree_population_size = 25
    config.evaluator_population_size = 20
    config.coevolution_generations = 40

    # Initialize coevolution
    coevolution = TreeCoevolution(config, domain_theorems)

    # Track arms race
    def track_arms_race(gen, metrics):
        print(f"Gen {gen}: "
              f"Best tree={metrics.best_tree_fitness:.3f}, "
              f"Best eval={metrics.best_evaluator_accuracy:.3f}")

    # Run coevolution
    best_tree, best_evaluator = await coevolution.coevolve(
        generations=40,
        progress_callback=track_arms_race
    )

    print("\n=== Best Tree ===")
    print(f"Depth: {best_tree.get_depth()}")
    print(f"Size: {best_tree.get_size()}")

    print("\n=== Best Evaluator ===")
    print(f"Feature weights: {best_evaluator.get_weights()}")

    # Test on new theorem
    test_theorem = "For all a b c, (a * b) * c = a * (b * c)"
    proof = best_tree.generate_proof(test_theorem)

    if proof:
        score = best_evaluator.evaluate(best_tree, test_theorem)
        print(f"\nTest score: {score:.3f}")
        print("Proof:", proof.lean_code[:200])

    # Save coevolved components
    coevolution.save_tree(best_tree, "algebra_tree.json")
    coevolution.save_evaluator(best_evaluator, "algebra_evaluator.json")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
Gen 0: Best tree=0.450, Best eval=0.520
Gen 10: Best tree=0.680, Best eval=0.710
Gen 20: Best tree=0.820, Best eval=0.840
Gen 30: Best tree=0.910, Best eval=0.920
Gen 40: Best tree=0.950, Best eval=0.960

=== Best Tree ===
Depth: 8
Size: 24

=== Best Evaluator ===
Feature weights: {'success': 0.7, 'depth': 0.1, 'diversity': 0.2}

Test score: 0.930
Proof: theorem mul_assoc (a b c : Nat) :
  (a * b) * c = a * (b * c) := by
  induction a
  case zero =>
    simp
  case succ a ih =>
    simp [ih]
    ring
```

---

## Intermediate Examples

### Example 4: Adaptive Approach Selection

Let framework automatically select best approach.

```python
import asyncio
from hybrid_mcts import (
    HybridMCTSEngine,
    AdaptiveHybridSelector,
    HybridMCTSPresets,
    HybridMCTSApproach
)

async def main():
    # Configure with adaptive selection
    config = HybridMCTSPresets.balanced()
    config.enable_adaptive_selection = True
    config.adaptive_window_size = 10
    config.switch_threshold = 0.3

    # Initialize selector
    selector = AdaptiveHybridSelector()

    # Test theorems of varying complexity
    test_theorems = [
        ("Simple", "For all n, n + 0 = n"),
        ("Medium", "For all n m, n + m = m + n"),
        ("Complex", "For all n, sum i from 0 to n of i = n*(n+1)/2"),
    ]

    engine = HybridMCTSEngine(config)

    for name, theorem in test_theorems:
        # Extract features
        features = selector.extract_features(theorem)
        print(f"\n=== {name} ===")
        print(f"Complexity: {features['complexity']:.2f}")
        print(f"Domain: {features['domain']}")

        # Get recommendation
        recommended = selector.select_approach(theorem, features)
        print(f"Recommended: {recommended.value}")

        # Search
        result = await engine.search(theorem, time_budget=30.0)

        print(f"Used: {result.approach_used.value}")
        print(f"Success: {result.success}")
        print(f"Time: {result.time_elapsed:.2f}s")
        print(f"Fitness: {result.best_fitness:.3f}")

        # Update selector performance
        selector.update_performance(
            result.approach_used,
            result.best_fitness
        )

    # Get performance summary
    history = selector.get_performance_history()
    print("\n=== Performance Summary ===")
    for approach, scores in history.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{approach.value}: {avg:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
=== Simple ===
Complexity: 0.25
Domain: arithmetic
Recommended: evolved_policies
Used: evolved_policies
Success: True
Time: 5.23s
Fitness: 0.950

=== Medium ===
Complexity: 0.52
Domain: arithmetic
Recommended: evolved_policies
Used: evolved_policies
Success: True
Time: 12.87s
Fitness: 0.880

=== Complex ===
Complexity: 0.85
Domain: arithmetic
Recommended: evolutionary_nodes
Used: evolutionary_nodes
Success: True
Time: 28.43s
Fitness: 0.820

=== Performance Summary ===
evolved_policies: 0.915
evolutionary_nodes: 0.820
```

---

### Example 5: Combined Approaches with Voting

Combine multiple approaches and vote on best result.

```python
import asyncio
from hybrid_mcts import (
    CombinedHybridMCTS,
    HybridMCTSPresets,
    HybridMCTSApproach
)

async def main():
    theorem = """
    For all n,
      sum of first n natural numbers = n*(n+1)/2
    """

    # Configure combined approach
    combined = CombinedHybridMCTS(
        approaches=[
            HybridMCTSApproach.EVOLVED_POLICIES,
            HybridMCTSApproach.EVOLUTIONARY_NODES,
            HybridMCTSApproach.COEVOLUTION
        ],
        combination_method="voting",
        config=HybridMCTSPresets.balanced()
    )

    # Search with all approaches
    result = await combined.search_combined(
        theorem,
        time_budget=60.0
    )

    print(f"=== Combined Result ===")
    print(f"Success: {result.success}")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Fitness: {result.best_fitness:.3f}")

    if result.success:
        print("\nProof:")
        print(result.best_proof.lean_code)

    # Get individual approach results
    individual_results = await combined.search_parallel(
        theorem,
        time_per_approach=30.0
    )

    print("\n=== Individual Results ===")
    for r in individual_results:
        print(f"{r.approach_used.value}: "
              f"Success={r.success}, "
              f"Fitness={r.best_fitness:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
=== Combined Result ===
Success: True
Time: 58.72s
Fitness: 0.910

Proof:
theorem sum_nat (n : Nat) :
  (∑ i in range (n + 1), i) = n * (n + 1) / 2 := by
  induction n
  case zero =>
    simp
  case succ n ih =>
    simp [ih]
    ring

=== Individual Results ===
evolved_policies: Success=True, Fitness=0.880
evolutionary_nodes: Success=True, Fitness=0.920
coevolution: Success=False, Fitness=0.650
```

---

### Example 6: LeanAide Integration

Enable formal verification with LeanAide.

```python
import asyncio
from hybrid_mcts import (
    HybridMCTSEngine,
    HybridMCTSPresets
)

async def main():
    theorem = "For all a b c, a + (b + c) = (a + b) + c"

    # Configure with LeanAide
    config = HybridMCTSPresets.thorough()
    config.leanaide_enabled = True
    config.leanaide_host = "localhost"
    config.leanaide_port = 7654
    config.leanaide_timeout = 30.0

    engine = HybridMCTSEngine(config)

    result = await engine.search(
        theorem,
        time_budget=60.0
    )

    print(f"Search completed: {result.success}")
    print(f"Time: {result.time_elapsed:.2f}s")

    # Verification results
    if result.leanaide_metrics:
        la_metrics = result.leanaide_metrics
        print(f"\n=== LeanAide Verification ===")
        print(f"Translation: {la_metrics.translation_success}")
        print(f"Verification: {la_metrics.verification_success}")
        print(f"Elaboration: {la_metrics.elaboration_success}")

        if la_metrics.verification_success:
            print("\n✓ Formally verified proof")
        else:
            print("\n✗ Verification failed")
            if la_metrics.errors:
                print("Errors:")
                for error in la_metrics.errors:
                    print(f"  - {error}")

    if result.best_proof:
        print("\n=== Generated Proof ===")
        print(result.best_proof.lean_code)

if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
Search completed: True
Time: 45.32s

=== LeanAide Verification ===
Translation: True
Verification: True
Elaboration: True

✓ Formally verified proof

=== Generated Proof ===
theorem add_assoc (a b c : Nat) :
  a + (b + c) = (a + b) + c := by
  induction b
  case zero =>
    simp
  case succ b ih =>
    simp [ih]
    ring
```

---

## Advanced Examples

### Example 7: Batch Processing with Policy Evolution

Evolve policy for multiple theorems efficiently.

```python
import asyncio
from hybrid_mcts import (
    PolicyEvolutionEngine,
    HybridMCTSPresets,
    EvolvedPolicyMCTS
)

async def main():
    # Load theorem corpus
    theorem_corpus = [
        # Arithmetic
        "∀ n, n + 0 = n",
        "∀ a b, a + b = b + a",
        "∀ a b c, (a + b) + c = a + (b + c)",
        "∀ n, n * 1 = n",
        "∀ a b, a * b = b * a",

        # Algebra
        "∀ x, x * 0 = 0",
        "∀ x y z, x * (y + z) = x*y + x*z",
        "∀ x y z, (x + y) * z = x*z + y*z",

        # Properties
        "∀ n, n ≤ n",
        "∀ n m, n ≤ m → n + k ≤ m + k",
    ]

    # Split into train/test
    train_set = theorem_corpus[:7]
    test_set = theorem_corpus[7:]

    # Configure evolution
    config = HybridMCTSPresets.balanced()
    config.policy_training_generations = 50
    config.policy_population_size = 40

    engine = PolicyEvolutionEngine(config)

    # Evolve on training set
    print("Evolving policy...")
    best_policy = await engine.evolve_policies(
        test_theorems=train_set,
        generations=50,
        mcts_config=config.mcts_config
    )

    print(f"Training fitness: {best_policy.fitness:.3f}")

    # Test on test set
    print("\nTesting on test set...")
    mcts = EvolvedPolicyMCTS(best_policy, config.mcts_config)

    test_results = []
    for theorem in test_set:
        result = await mcts.search(theorem, time_budget=20.0)
        test_results.append(result.success)
        print(f"  {result.success} - fitness={result.best_fitness:.3f}")

    success_rate = sum(test_results) / len(test_results)
    print(f"\nTest success rate: {success_rate:.2%}")

    # Plot convergence
    history = engine.get_training_history()
    generations = [m.generations_completed for m in history]
    fitnesses = [m.best_fitness for m in history]

    import matplotlib.pyplot as plt

    plt.plot(generations, fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Policy Evolution Convergence')
    plt.grid(True)
    plt.savefig('policy_convergence.png')
    print("\nSaved convergence plot to policy_convergence.png")

    # Save best policy
    engine.save_policy(best_policy, "best_policy.json")
    print("Saved best policy to best_policy.json")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Example 8: Transfer Learning Between Domains

Transfer learned policy from source to target domain.

```python
import asyncio
from hybrid_mcts import (
    PolicyEvolutionEngine,
    HybridMCTSPresets,
    RolloutPolicyGenome
)

async def main():
    # Source domain: Arithmetic
    source_theorems = [
        "∀ n, n + 0 = n",
        "∀ a b, a + b = b + a",
        "∀ a b c, (a + b) + c = a + (b + c)",
    ]

    # Target domain: Algebra
    target_theorems = [
        "∀ x, x * 0 = 0",
        "∀ x y, x * (y + z) = x*y + x*z",
        "∀ x y z, (x + y) * z = x*z + y*z",
    ]

    # Train on source domain
    print("=== Source Domain Training ===")
    config = HybridMCTSPresets.balanced()
    engine = PolicyEvolutionEngine(config)

    source_policy = await engine.evolve_policies(
        test_theorems=source_theorems,
        generations=30
    )

    print(f"Source fitness: {source_policy.fitness:.3f}")

    # Transfer to target domain
    print("\n=== Transfer Learning ===")

    # Initialize with source policy
    target_engine = PolicyEvolutionEngine(config)

    # Fine-tune on target domain
    target_policy = await target_engine.evolve_policies(
        test_theorems=target_theorems,
        generations=15,
        initial_population=[source_policy]  # Seed with source
    )

    print(f"Target fitness: {target_policy.fitness:.3f}")

    # Compare: from scratch vs transfer
    print("\n=== Comparison ===")

    from_scratch = await engine.evolve_policies(
        test_theorems=target_theorems,
        generations=15
    )

    print(f"From scratch: {from_scratch.fitness:.3f}")
    print(f"Transfer learning: {target_policy.fitness:.3f}")

    improvement = (target_policy.fitness - from_scratch.fitness)
    print(f"Improvement: {improvement:+.3f}")

    # Compare weights
    print("\n=== Weight Comparison ===")
    print("Source:", source_policy.tactic_weights)
    print("Target:", target_policy.tactic_weights)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Example 9: Multi-Objective Coevolution

Optimize multiple objectives simultaneously.

```python
import asyncio
from hybrid_mcts import MultiObjectiveCoevolution

async def main():
    theorems = [
        "∀ n, n + 0 = n",
        "∀ a b, a + b = b + a",
        "∀ n m, n * m = m * n",
    ]

    # Configure multi-objective
    multi = MultiObjectiveCoevolution(
        objectives=["success", "speed", "elegance"],
        objective_weights=[0.5, 0.3, 0.2],  # Prioritize success
        config=HybridMCTSPresets.coevolution()
    )

    # Evolve Pareto front
    pareto_front = await multi.coevolve_multi_objective(
        test_theorems=theorems,
        generations=50
    )

    print(f"=== Pareto Front ({len(pareto_front)} solutions) ===")

    # Show Pareto optimal solutions
    for i, solution in enumerate(pareto_front, 1):
        print(f"\nSolution {i}:")
        print(f"  Success: {solution.success_score:.3f}")
        print(f"  Speed: {solution.speed_score:.3f}")
        print(f"  Elegance: {solution.elegance_score:.3f}")
        print(f"  Tree depth: {solution.tree.get_depth()}")
        print(f"  Tree size: {solution.tree.get_size()}")

    # Select based on preference
    print("\n=== Select Best by Objective ===")

    # Best success rate
    best_success = max(pareto_front, key=lambda s: s.success_score)
    print(f"Best success: {best_success.success_score:.3f}")

    # Fastest
    best_speed = max(pareto_front, key=lambda s: s.speed_score)
    print(f"Best speed: {best_speed.speed_score:.3f}")

    # Most elegant
    best_elegance = max(pareto_front, key=lambda s: s.elegance_score)
    print(f"Best elegance: {best_elegance.elegance_score:.3f}")

    # Plot Pareto front
    multi.plot_pareto_front(pareto_front)
    print("\nSaved Pareto front plot")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Example 10: Monitoring and Analysis

Detailed monitoring and analysis of search process.

```python
import asyncio
from hybrid_mcts import (
    HybridMCTSEngine,
    HybridMCTSPresets,
    HybridMCTSMonitor
)

async def main():
    theorem = "∀ n, sum from 0 to n = n*(n+1)/2"

    # Configure with monitoring
    config = HybridMCTSPresets.thorough()
    config.enable_progress_tracking = True
    config.log_metrics = True

    # Create monitor
    monitor = HybridMCTSMonitor(
        log_iterations=True,
        log_nodes=True,
        log_time=True,
        save_tree=True
    )

    # Initialize engine with monitor
    engine = HybridMCTSEngine(config, monitor=monitor)

    # Search
    result = await engine.search(theorem, time_budget=120.0)

    # Get detailed summary
    summary = monitor.get_summary()

    print("=== Search Summary ===")
    print(f"Success: {summary['success']}")
    print(f"Time: {summary['time_elapsed']:.2f}s")
    print(f"Iterations: {summary['iterations_completed']}")
    print(f"Nodes visited: {summary['nodes_visited']}")
    print(f"Max depth: {summary['max_depth']}")
    print(f"Avg branching: {summary['avg_branching_factor']:.2f}")

    # Time breakdown
    print("\n=== Time Breakdown ===")
    for phase, time_val in summary['time_breakdown'].items():
        print(f"{phase}: {time_val:.2f}s")

    # Evolution metrics
    if 'evolution_metrics' in summary:
        evo = summary['evolution_metrics']
        print("\n=== Evolution Metrics ===")
        print(f"Generations: {evo['generations_completed']}")
        print(f"Best fitness: {evo['best_fitness']:.3f}")
        print(f"Avg fitness: {evo['average_fitness']:.3f}")
        print(f"Diversity: {evo['diversity']:.3f}")

    # MCTS metrics
    if 'mcts_metrics' in summary:
        mcts = summary['mcts_metrics']
        print("\n=== MCTS Metrics ===")
        print(f"Win rate: {mcts['win_rate']:.3f}")
        print(f"Confidence: {mcts['confidence']:.3f}")
        print(f"Root visits: {mcts['root_visits']}")

    # Generate plots
    monitor.plot_convergence()
    monitor.plot_tree_depth()
    monitor.plot_fitness_distribution()

    print("\nSaved analysis plots")

    # Export data
    monitor.export_metrics("search_metrics.json")
    monitor.export_tree("search_tree.json")

    print("\nExported metrics and tree data")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Real-World Scenarios

### Scenario 1: Production Theorem Proving

Deploy hybrid MCTS in production environment.

```python
import asyncio
from hybrid_mcts import (
    HybridMCTSEngine,
    HybridMCTSPresets,
    PolicyEvolutionEngine,
    AdaptiveHybridSelector
)

async def production_setup():
    """Setup for production use."""

    # 1. Train policies on domain corpus
    domain_corpus = load_production_corpus()  # Your data

    config = HybridMCTSPresets.balanced()
    engine = PolicyEvolutionEngine(config)

    print("Training production policy...")
    policy = await engine.evolve_policies(
        test_theorems=domain_corpus,
        generations=100
    )

    # Save policy
    engine.save_policy(policy, "production_policy.json")

    # 2. Create production engine
    prod_config = HybridMCTSPresets.fast()
    prod_config.enable_adaptive_selection = True
    prod_config.enable_caching = True
    prod_config.max_workers = 8

    production_engine = HybridMCTSEngine(prod_config)

    return production_engine

async def handle_request(engine: HybridMCTSEngine, theorem: str):
    """Handle production request."""
    try:
        result = await engine.search(
            theorem,
            time_budget=30.0
        )

        return {
            "success": result.success,
            "proof": result.best_proof.lean_code if result.best_proof else None,
            "confidence": result.mcts_confidence,
            "approach": result.approach_used.value,
            "time": result.time_elapsed
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def main():
    # Setup
    engine = await production_setup()

    # Handle requests
    theorems = [
        "User theorem 1",
        "User theorem 2",
        "User theorem 3"
    ]

    for theorem in theorems:
        response = await handle_request(engine, theorem)
        print(f"\n{theorem}:")
        print(f"  Success: {response['success']}")
        if response['success']:
            print(f"  Confidence: {response['confidence']:.2f}")
            print(f"  Approach: {response['approach']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Scenario 2: Research and Exploration

Use for exploring new mathematical domains.

```python
import asyncio
from hybrid_mcts import (
    TreeCoevolution,
    HybridMCTSPresets
)

async def research_setup():
    """Setup for research/exploration."""

    # Unknown domain theorems
    exploration_set = [
        "Novel theorem 1",
        "Novel theorem 2",
        "Novel theorem 3",
    ]

    # Use coevolution for exploration
    config = HybridMCTSPresets.coevolution()
    config.coevolution_generations = 100
    config.tree_population_size = 30
    config.evaluator_population_size = 25

    coevolution = TreeCoevolution(config, exploration_set)

    # Track arms race
    def track(gen, metrics):
        if gen % 10 == 0:
            print(f"Gen {gen}: Tree={metrics.best_tree_fitness:.3f}, "
                  f"Eval={metrics.best_evaluator_accuracy:.3f}")

    # Run coevolution
    best_tree, best_evaluator = await coevolution.coevolve(
        generations=100,
        progress_callback=track
    )

    # Analyze results
    print("\n=== Analysis ===")
    print(f"Best tree depth: {best_tree.get_depth()}")
    print(f"Best tree size: {best_tree.get_size()}")

    # Extract strategies
    strategies = coevolution.extract_strategies(best_tree)
    print("\nDiscovered strategies:")
    for strategy in strategies:
        print(f"  - {strategy}")

    # Save for publication
    coevolution.save_tree(best_tree, "research_tree.json")
    coevolution.save_evaluator(best_evaluator, "research_evaluator.json")
    coevolution.plot_arms_race()

    return best_tree, best_evaluator

if __name__ == "__main__":
    asyncio.run(research_setup())
```

---

## Complete Workflows

### Workflow 1: End-to-End Pipeline

Complete pipeline from data collection to deployment.

```python
import asyncio
from hybrid_mcts import *

async def full_pipeline():
    """Complete hybrid MCTS pipeline."""

    # ===== PHASE 1: Data Collection =====
    print("=== Phase 1: Data Collection ===")

    raw_theorems = collect_theorems()  # Your data source

    # Split into train/val/test
    train = raw_theorems[:70]
    val = raw_theorems[70:85]
    test = raw_theorems[85:]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # ===== PHASE 2: Configuration =====
    print("\n=== Phase 2: Configuration ===")

    config = HybridMCTSPresets.balanced()
    config.policy_training_generations = 50
    config.enable_caching = True

    # ===== PHASE 3: Training =====
    print("\n=== Phase 3: Training ===")

    engine = PolicyEvolutionEngine(config)

    def on_progress(gen, metrics):
        if gen % 10 == 0:
            print(f"  Gen {gen}: fitness={metrics.best_fitness:.3f}")

    best_policy = await engine.evolve_policies(
        test_theorems=train,
        generations=50,
        progress_callback=on_progress
    )

    print(f"Training complete: {best_policy.fitness:.3f}")

    # ===== PHASE 4: Validation =====
    print("\n=== Phase 4: Validation ===")

    mcts = EvolvedPolicyMCTS(best_policy, config.mcts_config)

    val_results = []
    for theorem in val:
        result = await mcts.search(theorem, time_budget=20.0)
        val_results.append(result.success)

    val_accuracy = sum(val_results) / len(val_results)
    print(f"Validation accuracy: {val_accuracy:.2%}")

    # ===== PHASE 5: Testing =====
    print("\n=== Phase 5: Testing ===")

    test_results = []
    for theorem in test:
        result = await mcts.search(theorem, time_budget=20.0)
        test_results.append(result.success)

    test_accuracy = sum(test_results) / len(test_results)
    print(f"Test accuracy: {test_accuracy:.2%}")

    # ===== PHASE 6: Deployment =====
    print("\n=== Phase 6: Deployment ===")

    # Save artifacts
    engine.save_policy(best_policy, "production_policy.json")

    # Create production engine
    prod_config = HybridMCTSPresets.fast()
    prod_engine = HybridMCTSEngine(prod_config)

    print("Ready for production")

    return {
        "policy": best_policy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "engine": prod_engine
    }

if __name__ == "__main__":
    results = asyncio.run(full_pipeline())
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-30
**Author**: OpenEvolve Frontend Team
**Related Docs**:
- [HYBRID_MCTS_ARCHITECTURE.md](./HYBRID_MCTS_ARCHITECTURE.md)
- [HYBRID_MCTS_API.md](./HYBRID_MCTS_API.md)
- [HYBRID_MCTS_GUIDE.md](./HYBRID_MCTS_GUIDE.md)
