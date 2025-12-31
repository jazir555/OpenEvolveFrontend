"""
Example Usage of Lean 4 Self-Play System

This script demonstrates various usage patterns for the LeanAide self-play system,
from basic usage to advanced training loops.
"""

import asyncio
import json
from pathlib import Path

from leanaide_selfplay import (
    LeanSelfPlayEngine,
    LeanTheorem,
    LeanProofStrategy,
    ProofDifficulty,
    LeanTactic,
    LeanProof
)


# ============================================================================
# Example 1: Basic Self-Play
# ============================================================================

async def example_1_basic_self_play():
    """Basic self-play for a single theorem"""
    print("=" * 60)
    print("Example 1: Basic Self-Play")
    print("=" * 60)

    # Create engine
    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=100
    )

    try:
        # Define a simple theorem
        theorem = "∀ n : Nat, n + 0 = n"

        print(f"\nRunning self-play for theorem: {theorem}")

        # Run self-play
        best_proof = await engine.run_self_play(
            theorem=theorem,
            games=5
        )

        # Display results
        print(f"\nResults:")
        print(f"  Best proof valid: {best_proof.is_valid if best_proof else False}")
        print(f"  Proof tactics: {best_proof.tactic_count if best_proof else 0}")
        print(f"  Confidence: {best_proof.confidence if best_proof else 0:.2f}")

        # Get progress
        progress = engine.get_training_progress()
        print(f"\nTraining progress:")
        print(f"  Total games: {progress['total_games']}")
        print(f"  Success rate: {progress['success_rate']:.1%}")

    finally:
        await engine.close()

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Example 2: Batch Self-Play
# ============================================================================

async def example_2_batch_self_play():
    """Self-play on multiple theorems"""
    print("=" * 60)
    print("Example 2: Batch Self-Play")
    print("=" * 60)

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=500
    )

    try:
        # Define multiple theorems
        theorems = [
            "∀ n : Nat, n + 0 = n",
            "∀ a b : Nat, a + b = b + a",
            "∀ n : Nat, 2 * n = n + n"
        ]

        print(f"\nRunning batch self-play for {len(theorems)} theorems")

        # Run batch self-play
        results = await engine.run_batch_self_play(
            theorems=theorems,
            games_per_theorem=3
        )

        # Display results
        print(f"\nResults:")
        for theorem, proof in results.items():
            if proof:
                print(f"  {theorem}:")
                print(f"    Valid: {proof.is_valid}")
                print(f"    Tactics: {proof.tactic_count}")
                print(f"    Status: {proof.status.value}")

        # Train from experiences
        print("\nTraining from experience buffer...")
        metrics = await engine.train_from_buffer(
            batch_size=8,
            iterations=5
        )

        print(f"\nTraining metrics:")
        print(f"  Success rate: {metrics.success_rate:.1%}")
        print(f"  Avg reward: {metrics.avg_reward:.3f}")
        print(f"  Avg proof length: {metrics.avg_proof_length:.1f} tactics")

    finally:
        await engine.close()

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Example 3: Training Loop with Checkpoints
# ============================================================================

async def example_3_training_loop():
    """Complete training loop with checkpoints"""
    print("=" * 60)
    print("Example 3: Training Loop with Checkpoints")
    print("=" * 60)

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=1000
    )

    try:
        # Training configuration
        num_epochs = 3
        games_per_epoch = 6
        training_iterations = 10

        # Define curriculum (easy to hard)
        curriculum = [
            ["∀ n : Nat, n + 0 = n"],  # Easy
            ["∀ a b : Nat, a + b = b + a"],  # Medium
            ["∀ n : Nat, 2 * n = n + n"]  # Medium
        ]

        print(f"\nStarting training loop:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Games per epoch: {games_per_epoch}")
        print(f"  Training iterations: {training_iterations}")

        initial_success = 0.0

        for epoch in range(num_epochs):
            print(f"\n{'='*20} Epoch {epoch + 1}/{num_epochs} {'='*20}")

            # Select theorems for this epoch
            theorems = curriculum[epoch % len(curriculum)]

            # Self-play phase
            print(f"\nSelf-play phase ({len(theorems)} theorems)...")
            for theorem in theorems:
                await engine.run_self_play(theorem, games=games_per_epoch)

            # Training phase
            print(f"\nTraining phase...")
            metrics = await engine.train_from_buffer(
                batch_size=16,
                iterations=training_iterations
            )

            # Display progress
            print(f"\nEpoch {epoch + 1} results:")
            print(f"  Success rate: {metrics.success_rate:.1%}")
            print(f"  Avg reward: {metrics.avg_reward:.3f}")
            print(f"  Total games: {metrics.total_games}")

            if epoch == 0:
                initial_success = metrics.success_rate

            # Save checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.json"
            engine.save_checkpoint(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Final summary
        print(f"\n{'='*20} Training Complete {'='*20}")
        progress = engine.get_training_progress()
        improvement = progress['improvement']

        print(f"\nFinal results:")
        print(f"  Initial success rate: {initial_success:.1%}")
        print(f"  Final success rate: {progress['success_rate']:.1%}")
        print(f"  Absolute improvement: {improvement['absolute']:.1%}")
        print(f"  Relative improvement: {improvement['relative']:.1%}")
        print(f"  Total games played: {progress['total_games']}")

    finally:
        await engine.close()

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Example 4: Custom Proof Strategies
# ============================================================================

async def example_4_custom_strategies():
    """Using custom proof strategies"""
    print("=" * 60)
    print("Example 4: Custom Proof Strategies")
    print("=" * 60)

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654"
    )

    try:
        # Define custom strategies
        custom_strategies = [
            LeanProofStrategy(
                name="calculation_strategy",
                tactic_sequence=["calc", "rw", "simp", "norm_num"],
                description="Step-by-step calculation",
                适用领域=["algebra", "analysis"],
                success_rate=0.7
            ),
            LeanProofStrategy(
                name="structural_induction",
                tactic_sequence=["induction", "case", "simp", "rfl"],
                description="Proof by structural induction",
                适用领域=["combinatorics", "algebra"],
                success_rate=0.6
            ),
            LeanProofStrategy(
                name="contradiction_strategy",
                tactic_sequence=["intro", "by_contradiction", "push_neg", "contradiction"],
                description="Proof by contradiction",
                适用领域=["logic", "set_theory"],
                success_rate=0.5
            )
        ]

        # Add custom strategies to agent
        print(f"\nAdding {len(custom_strategies)} custom strategies...")
        for strategy in custom_strategies:
            engine.agent.known_strategies.append(strategy)
            print(f"  - {strategy.name}: {strategy.description}")

        # Run self-play with custom strategies
        theorem = "∀ a b : Nat, a + b = b + a"
        print(f"\nRunning self-play with custom strategies...")

        await engine.run_self_play(theorem, games=5)

        # Analyze which strategies were used
        strategy_usage = {}
        for exp in engine.buffer.buffer:
            strategy_name = exp.strategy_used
            strategy_usage[strategy_name] = strategy_usage.get(strategy_name, 0) + 1

        print(f"\nStrategy usage:")
        for strategy, count in strategy_usage.items():
            print(f"  {strategy}: {count} times")

        # Analyze success rate by strategy
        print(f"\nSuccess rate by strategy:")
        for strategy_obj in engine.agent.known_strategies:
            if strategy_obj.name in strategy_usage:
                print(f"  {strategy_obj.name}: {strategy_obj.success_rate:.1%}")

    finally:
        await engine.close()

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Example 5: Experience Buffer Analysis
# ============================================================================

async def example_5_buffer_analysis():
    """Analyze experience buffer statistics"""
    print("=" * 60)
    print("Example 5: Experience Buffer Analysis")
    print("=" * 60)

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=200
    )

    try:
        # Generate diverse experiences
        theorems = [
            ("∀ n : Nat, n + 0 = n", "algebra", ProofDifficulty.EASY),
            ("∀ a b : Nat, a + b = b + a", "algebra", ProofDifficulty.MEDIUM),
            ("∀ n : Nat, 2 * n = n + n", "algebra", ProofDifficulty.EASY),
        ]

        print(f"\nGenerating experiences for {len(theorems)} theorems...")

        for theorem_stmt, domain, difficulty in theorems:
            theorem = LeanTheorem(
                id=f"theorem_{len(engine.buffer.buffer)}",
                statement=theorem_stmt,
                lean_code=f"theorem test : {theorem_stmt} := by",
                difficulty=difficulty,
                domain=domain
            )

            # Run self-play
            await engine.run_self_play(theorem_stmt, games=3)

        # Analyze buffer
        print(f"\nBuffer statistics:")
        stats = engine.buffer.get_statistics()

        print(f"  Size: {stats['size']}/{stats['capacity']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Average reward: {stats['avg_reward']:.3f}")
        print(f"  Reward std: {stats['reward_std']:.3f}")

        # Analyze by domain
        print(f"\nExperiences by domain:")
        domain_counts = {}
        for exp in engine.buffer.buffer:
            domain = exp.theorem.domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} experiences")

        # Analyze by difficulty
        print(f"\nExperiences by difficulty:")
        difficulty_counts = {}
        for exp in engine.buffer.buffer:
            difficulty = exp.theorem.difficulty.value
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count} experiences")

        # Analyze proof lengths
        proof_lengths = [exp.proof.tactic_count for exp in engine.buffer.buffer]
        print(f"\nProof length statistics:")
        print(f"  Min: {min(proof_lengths)} tactics")
        print(f"  Max: {max(proof_lengths)} tactics")
        print(f"  Average: {sum(proof_lengths)/len(proof_lengths):.1f} tactics")

        # Save buffer for later analysis
        buffer_path = "experience_buffer.json"
        engine.buffer.save(buffer_path)
        print(f"\nBuffer saved to: {buffer_path}")

    finally:
        await engine.close()

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Example 6: Resume from Checkpoint
# ============================================================================

async def example_6_resume_training():
    """Resume training from a checkpoint"""
    print("=" * 60)
    print("Example 6: Resume Training from Checkpoint")
    print("=" * 60)

    # First, create a checkpoint
    print("\nPhase 1: Create initial checkpoint")

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654"
    )

    try:
        # Run some initial games
        await engine.run_self_play("∀ n : Nat, n + 0 = n", games=3)

        # Save checkpoint
        checkpoint_path = "temp_checkpoint.json"
        engine.save_checkpoint(checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        print(f"  Iterations: {engine.iteration_count}")

        # Get initial progress
        initial_progress = engine.get_training_progress()
        print(f"  Success rate: {initial_progress['success_rate']:.1%}")

    finally:
        await engine.close()

    # Now resume from checkpoint
    print("\nPhase 2: Resume from checkpoint")

    new_engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654"
    )

    try:
        # Load checkpoint
        new_engine.load_checkpoint(checkpoint_path)
        print(f"  Checkpoint loaded: {checkpoint_path}")
        print(f"  Resumed from iteration: {new_engine.iteration_count}")

        # Continue training
        print(f"\nContinuing training...")
        await new_engine.run_self_play("∀ a b : Nat, a + b = b + a", games=3)

        # Get updated progress
        final_progress = new_engine.get_training_progress()
        print(f"\nFinal results:")
        print(f"  Total iterations: {final_progress['total_games']}")
        print(f"  Success rate: {final_progress['success_rate']:.1%}")

    finally:
        await new_engine.close()

    # Cleanup
    Path(checkpoint_path).unlink(missing_ok=True)

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Example 7: Performance Monitoring
# ============================================================================

async def example_7_performance_monitoring():
    """Real-time performance monitoring during training"""
    print("=" * 60)
    print("Example 7: Performance Monitoring")
    print("=" * 60)

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654"
    )

    try:
        theorems = [
            "∀ n : Nat, n + 0 = n",
            "∀ a b : Nat, a + b = b + a"
        ]

        print(f"\nTraining with real-time monitoring...")

        # Training with monitoring
        for round_num in range(3):
            print(f"\n--- Round {round_num + 1} ---")

            # Self-play
            for theorem in theorems:
                await engine.run_self_play(theorem, games=2)

            # Train
            metrics = await engine.train_from_buffer(batch_size=4, iterations=3)

            # Display metrics
            print(f"  Total games: {metrics.total_games}")
            print(f"  Success rate: {metrics.success_rate:.1%}")
            print(f"  Avg reward: {metrics.avg_reward:.3f}")
            print(f"  Avg proof length: {metrics.avg_proof_length:.1f}")

            # Show agent performance
            if engine.agent.performance_history:
                recent_perf = engine.agent.performance_history[-5:]
                print(f"\n  Recent agent performance:")
                for perf in recent_perf:
                    status = "✓" if perf['success'] else "✗"
                    print(f"    {status} {perf['theorem_id'][:20]}... "
                          f"(reward: {perf['reward']:.2f})")

        # Show improvement trajectory
        print(f"\n{'='*20} Improvement Trajectory {'='*20}")

        if len(engine.metrics_history) > 1:
            print(f"\n{'Round':<8} {'Success':<10} {'Reward':<10} {'Length':<10}")
            print("-" * 40)

            for i, metrics in enumerate(engine.metrics_history, 1):
                print(f"{i:<8} {metrics.success_rate:<10.1%} "
                      f"{metrics.avg_reward:<10.3f} "
                      f"{metrics.avg_proof_length:<10.1f}")

    finally:
        await engine.close()

    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Main Runner
# ============================================================================

async def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Self-Play", example_1_basic_self_play),
        ("Batch Self-Play", example_2_batch_self_play),
        ("Training Loop", example_3_training_loop),
        ("Custom Strategies", example_4_custom_strategies),
        ("Buffer Analysis", example_5_buffer_analysis),
        ("Resume Training", example_6_resume_training),
        ("Performance Monitoring", example_7_performance_monitoring)
    ]

    print("\n" + "=" * 60)
    print("Lean 4 Self-Play System - Examples")
    print("=" * 60 + "\n")

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run a specific example
    # asyncio.run(example_1_basic_self_play())

    # Or run all examples
    asyncio.run(run_all_examples())
