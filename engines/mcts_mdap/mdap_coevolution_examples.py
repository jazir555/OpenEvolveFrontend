"""
MDAP/MAKER Coevolution Examples

This module provides practical examples of using the MDAP/MAKER integration
with coevolving decision trees.
"""
from __future__ import annotations


import asyncio
from mcts_coevolution_mdap import (
    MDAPProofDecisionTree,
    MDAPTreeCoevolution,
    DecompositionTreeCoevolution,
    MDAPCompetitiveCoevolution,
    MDAPMultiObjectiveCoevolution,
    MDAPTreeEnsemble,
    MDAPCoevolutionMonitor,
    create_mdap_config,
    run_mdap_coevolution_pipeline
)


async def example_basic_mdap_coevolution():
    """Basic MDAP coevolution example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic MDAP Coevolution")
    print("=" * 80)

    # Define test theorems
    test_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a",
        "∀ n: Nat, 2 * n = n + n"
    ]

    # Create MDAP coevolution configuration
    config = create_mdap_config(
        num_agents=5,
        k_ahead=3,
        voting_strategy="first_k_ahead",
        enable_decomposition=True
    )

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run coevolution
    print("\nStarting coevolution...")
    best_tree = await run_mdap_coevolution_pipeline(test_theorems, config)

    print("\nResults:")
    print(f"  Best tree ID: {best_tree.tree_id}")
    print(f"  Consensus score: {best_tree.consensus_score:.4f}")
    print(f"  Agreement level: {best_tree.agreement_level:.4f}")
    print(f"  Fitness: {best_tree.fitness:.4f}")
    print(f"  Depth: {best_tree.depth}")
    print(f"  Nodes: {best_tree.node_count}")


async def example_multi_objective_optimization():
    """Multi-objective optimization example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multi-Objective Optimization")
    print("=" * 80)

    test_theorems = [
        "∀ a b: Nat, a + b = b + a",
        "∀ a b c: Nat, (a + b) + c = a + (b + c)"
    ]

    # Create multi-objective coevolution
    multi_obj = MDAPMultiObjectiveCoevolution(
        objectives=["success", "elegance", "simplicity"],
        population_size=30,
        generations=20,
        num_agents=5
    )

    print("\nObjectives to optimize:")
    for obj in multi_obj.objectives:
        print(f"  - {obj}")

    print("\nRunning multi-objective coevolution...")
    pareto_front = await multi_obj.coevolve_multi_objective_mdap(test_theorems)

    print(f"\nPareto front size: {len(pareto_front)}")

    print("\nTop 5 Pareto-optimal solutions:")
    for i, tree in enumerate(pareto_front[:5]):
        obj_fit = getattr(tree, 'objective_fitness', {})
        print(f"\n  Solution {i+1}:")
        print(f"    Tree ID: {tree.tree_id[:8]}...")
        for obj, value in obj_fit.items():
            print(f"    {obj}: {value:.4f}")


async def example_competitive_coevolution():
    """Competitive coevolution example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Competitive Coevolution")
    print("=" * 80)

    initial_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a"
    ]

    # Create competitive coevolution
    competitive = MDAPCompetitiveCoevolution(
        solver_pop_size=30,
        problem_pop_size=10,
        generations=25,
        num_agents=5,
        k_ahead=3
    )

    print("\nStarting competitive coevolution...")
    print("Solvers and problems will coevolve:")
    print("  - Solvers evolve to prove theorems better")
    print("  - Problems evolve to be more challenging")

    best_solver = await competitive.competitive_coevolve_mdap(initial_theorems)

    print(f"\nBest solver found:")
    print(f"  Tree ID: {best_solver.tree_id}")
    print(f"  Fitness: {best_solver.fitness:.4f}")
    print(f"  Consensus: {best_solver.consensus_score:.4f}")


async def example_ensemble_methods():
    """Ensemble methods example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Ensemble Methods")
    print("=" * 80)

    # Create a simple ensemble with mock trees
    from mcts_coevolution import TreeGenerator, DecisionNode, NodeType

    generator = TreeGenerator()
    base_trees = generator.generate_ramped_half_and_half(10, 10)

    # Convert to MDAP trees
    mdap_trees = [
        MDAPProofDecisionTree(
            root=t.root,
            tree_id=t.tree_id,
            num_agents=3
        )
        for t in base_trees[:5]
    ]

    # Set mock performance
    for tree in mdap_trees:
        tree.consensus_score = 0.7 + hash(tree.tree_id) % 30 / 100
        tree.agreement_level = 0.6 + hash(tree.tree_id) % 40 / 100

    # Create ensemble
    ensemble = MDAPTreeEnsemble(
        trees=mdap_trees,
        voting_strategy="first_k_ahead",
        k_ahead=2
    )

    print("\nEnsemble configuration:")
    print(f"  Number of trees: {len(ensemble.trees)}")
    print(f"  Voting strategy: {ensemble.voting_strategy}")
    print(f"  K-ahead: {ensemble.k_ahead}")

    print("\nTree consensus scores:")
    for i, tree in enumerate(ensemble.trees):
        print(f"  Tree {i+1}: {tree.consensus_score:.4f}")

    # Note: Actual evaluation requires ProofContext
    print("\nEnsemble ready for evaluation!")


async def example_monitoring():
    """Performance monitoring example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Performance Monitoring")
    print("=" * 80)

    # Create monitor
    monitor = MDAPCoevolutionMonitor()

    # Simulate tracking across generations
    print("\nSimulating coevolution tracking...")

    for gen in range(10):
        # Create mock population data
        from mcts_coevolution import TreeGenerator

        generator = TreeGenerator()
        base_tree = generator.generate_grow_tree(10, min_depth=2)
        mdap_tree = MDAPProofDecisionTree(
            root=base_tree.root,
            num_agents=5
        )

        # Simulate improving performance
        mdap_tree.consensus_score = 0.5 + gen * 0.04
        mdap_tree.agreement_level = 0.6 + gen * 0.03

        from mcts_coevolution_mdap import MDAPTreeEvaluation, AgentEvaluation

        evaluation = MDAPTreeEvaluation(
            tree_id=mdap_tree.tree_id,
            agent_results=[
                AgentEvaluation(
                    agent_id=f"agent_{i}",
                    success_rate=mdap_tree.consensus_score + (i - 2) * 0.02,
                    avg_depth=10.0,
                    avg_time=1.0,
                    elegance_score=0.7,
                    simplicity_score=0.8,
                    robustness=0.75
                )
                for i in range(5)
            ],
            consensus_score=mdap_tree.consensus_score,
            agreement_level=mdap_tree.agreement_level,
            voting_details={"success_votes": 3, "total_agents": 5, "avg_success": 0.75},
            std_dev_success=0.1,
            std_dev_depth=0.2
        )

        monitor.track_generation(gen, [mdap_tree], [evaluation])

    print("\nMonitoring complete!")

    # Generate agent reliability report
    print("\nAgent Reliability Report:")
    print("-" * 40)
    report = monitor.get_agent_reliability_report()

    for agent_id, metrics in sorted(report.items()):
        print(f"\n{agent_id}:")
        print(f"  Average score: {metrics['avg_score']:.4f}")
        print(f"  Std deviation: {metrics['std_dev']:.4f}")
        print(f"  Min score: {metrics['min_score']:.4f}")
        print(f"  Max score: {metrics['max_score']:.4f}")
        print(f"  Evaluations: {metrics['num_evaluations']}")

    # Plot progress (optional)
    print("\nGenerating progress plots...")
    try:
        monitor.plot_progress()
    except Exception as e:
        print(f"Plotting skipped: {e}")


async def example_custom_configuration():
    """Custom configuration example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custom Configuration")
    print("=" * 80)

    # Custom MDAP configuration
    config = create_mdap_config(
        num_agents=7,           # More agents for robust evaluation
        k_ahead=5,              # Higher threshold for stronger consensus
        voting_strategy="first_k_ahead",
        enable_decomposition=True
    )

    print("\nCustom MDAP Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create coevolution with custom settings
    coevolution = MDAPTreeCoevolution(
        population_size=40,
        generations=20,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism=3,
        max_depth=15,
        simulations=50,
        num_agents=config["num_agents"],
        k_ahead=config["k_ahead"],
        voting_strategy=config["voting_strategy"]
    )

    print("\nCoevolution Parameters:")
    print(f"  Population size: {coevolution.population_size}")
    print(f"  Generations: {coevolution.generations}")
    print(f"  Crossover rate: {coevolution.crossover_rate}")
    print(f"  Mutation rate: {coevolution.mutation_rate}")
    print(f"  Simulations per evaluation: {coevolution.mdap_evaluator.simulations}")

    test_theorems = ["∀ n: Nat, n + 0 = n"]

    print("\nConfiguration ready for coevolution!")
    print("Run: best_tree = await coevolution.coevolve_mdap(test_theorems)")


async def example_decomposition_enhanced():
    """Decomposition-enhanced coevolution example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Decomposition-Enhanced Coevolution")
    print("=" * 80)

    # Create base MDAP coevolution
    base_coevolution = MDAPTreeCoevolution(
        population_size=30,
        generations=15,
        num_agents=5
    )

    # Create decomposition-enhanced coevolution
    decomp_coevolution = DecompositionTreeCoevolution(
        mdap_coevolution=base_coevolution,
        max_decomposition_depth=3,
        decomposition_threshold=0.7
    )

    print("\nDecomposition Configuration:")
    print(f"  Max decomposition depth: {decomp_coevolution.max_decomposition_depth}")
    print(f"  Decomposition threshold: {decomp_coevolution.decomposition_threshold}")

    test_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a"
    ]

    print("\nDecomposition will be used when:")
    print("  - Problem complexity is high")
    print("  - Success rate is below threshold")
    print("  - Agent agreement is low")

    print("\nStarting decomposition-enhanced coevolution...")
    best_tree = await decomp_coevolution.coevolve_with_decomposition(test_theorems)

    print(f"\nBest tree fitness: {best_tree.fitness:.4f}")


async def main():
    """Run all examples"""
    print("\n")
    print("=" * 80)
    print("MDAP/MAKER COEVOLVING DECISION TREES - EXAMPLES")
    print("=" * 80)

    examples = [
        ("Basic MDAP Coevolution", example_basic_mdap_coevolution),
        ("Multi-Objective Optimization", example_multi_objective_optimization),
        ("Competitive Coevolution", example_competitive_coevolution),
        ("Ensemble Methods", example_ensemble_methods),
        ("Performance Monitoring", example_monitoring),
        ("Custom Configuration", example_custom_configuration),
        ("Decomposition-Enhanced", example_decomposition_enhanced),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\n" + "-" * 80)
    selection = input("\nEnter example number to run (or 'all' for all examples): ").strip()

    if selection.lower() == 'all':
        for name, example_func in examples:
            try:
                await example_func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
            print("\n" + "=" * 80 + "\n")
    elif selection.isdigit() and 1 <= int(selection) <= len(examples):
        idx = int(selection) - 1
        name, example_func = examples[idx]
        try:
            await example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
    else:
        print("Invalid selection!")


if __name__ == "__main__":
    asyncio.run(main())
