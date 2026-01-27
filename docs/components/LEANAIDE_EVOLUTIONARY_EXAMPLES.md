# LeanAide Evolutionary Examples - Real-World Usage

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide Evolutionary Integration

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Basic Examples](#2-basic-examples)
3. [Advanced Workflows](#3-advanced-workflows)
4. [Domain-Specific Examples](#4-domain-specific-examples)
5. [End-to-End Workflows](#5-end-to-end-workflows)
6. [Performance Tuning Examples](#6-performance-tuning-examples)
7. [Common Patterns](#7-common-patterns)
8. [Migration Examples](#8-migration-examples)

---

## 1. Introduction

This document provides real-world examples of using evolutionary LeanAide for automated proof generation. Each example includes complete, runnable code with explanations.

**Example Categories:**
- Basic: Getting started with simple theorems
- Advanced: Complex workflows and configurations
- Domain-Specific: Algebra, analysis, combinatorics, logic
- End-to-End: Complete workflows from theorem to verified proof
- Performance: Optimization and tuning examples
- Patterns: Reusable design patterns

---

## 2. Basic Examples

### 2.1 Simple Genetic Evolution

**Scenario:** Prove a basic arithmetic theorem using genetic evolution.

```python
import asyncio
from leanaide_evolution import evolve_proof

async def main():
    """Basic genetic evolution example"""

    theorem = "∀ n : Nat, n + 0 = n"

    result = await evolve_proof(
        theorem=theorem,
        theorem_name="add_zero",
        max_generations=30,
        population_size=20,
        server_url="http://localhost:7654"
    )

    # Report results
    print("=" * 60)
    print("GENETIC EVOLUTION RESULTS")
    print("=" * 60)
    print(f"Theorem: {theorem}")
    print(f"Success: {result.success}")
    print(f"Generations: {result.generations_completed}")
    print(f"Total Evaluations: {result.total_evaluations}")
    print(f"Time: {result.evolution_time:.2f}s")

    if result.success:
        print("\n✓ VERIFIED PROOF:")
        print("-" * 60)
        print(result.best_proof.lean_code)
    else:
        print("\n✗ NO VERIFIED PROOF FOUND")
        print(f"Best Fitness: {result.best_strategy.fitness:.3f}")
        print(f"Best Proof Attempt:")
        print("-" * 60)
        print(result.best_strategy.proof.lean_code)

asyncio.run(main())
```

**Expected Output:**
```
============================================================
GENETIC EVOLUTION RESULTS
============================================================
Theorem: ∀ n : Nat, n + 0 = n
Success: True
Generations: 12
Total Evaluations: 240
Time: 45.23s

✓ VERIFIED PROOF:
------------------------------------------------------------
theorem add_zero (n : Nat) : n + 0 = n := by
  rw [Nat.add_zero]
```

---

### 2.2 Adversarial Evolution for Robustness

**Scenario:** Use adversarial evolution to find edge cases in a proof.

```python
import asyncio
from leanaide_adversarial import LeanAdversarialEvolution

async def main():
    """Adversarial evolution example"""

    theorem = """
    theorem injective_comp {f : α → β} {g : β → γ}
        (hf : Function.Injective f)
        (hg : Function.Injective g) :
        Function.Injective (g ∘ f) := by
          -- Proof to be generated
    """

    evolution = LeanAdversarialEvolution(api_key="your-api-key")

    final_proof, round_results, stats = await evolution.run_adversarial_evolution(
        theorem=theorem,
        rounds=12
    )

    # Detailed reporting
    print("=" * 60)
    print("ADVERSARIAL EVOLUTION RESULTS")
    print("=" * 60)
    print(f"Total Rounds: {len(round_results)}")
    print(f"Blue Wins: {stats.blue_wins}")
    print(f"Red Wins: {stats.red_wins}")
    print(f"Blue Win Rate: {stats.blue_success_rate:.1%}")
    print(f"Red Win Rate: {stats.red_success_rate:.1%}")
    print(f"Counterexamples Found: {stats.unique_counterexamples_found}")

    if stats.most_effective_approach:
        print(f"\nMost Effective Approach: {stats.most_effective_approach.value}")

    # Round-by-round breakdown
    print("\n" + "=" * 60)
    print("ROUND-BY-ROUND BREAKDOWN")
    print("=" * 60)

    for round_result in round_results:
        status = "✓ SURVIVED" if round_result.blue_survived else "✗ FAILED"
        print(f"Round {round_result.round_number}: {status}")
        print(f"  Blue Score: {round_result.blue_score:.3f}")
        print(f"  Red Score: {round_result.red_score:.3f}")
        print(f"  Critiques: {len(round_result.red_critique)}")
        print(f"  Counterexamples: {len(round_result.counterexamples)}")

    # Final proof
    if final_proof.lean_code:
        print("\n" + "=" * 60)
        print("FINAL ROBUST PROOF")
        print("=" * 60)
        print(final_proof.lean_code)

asyncio.run(main())
```

---

### 2.3 Self-Play for Continuous Improvement

**Scenario:** Train an agent on multiple related theorems using self-play.

```python
import asyncio
from leanaide_selfplay import LeanSelfPlayEngine

async def main():
    """Self-play training example"""

    # Training set of related theorems
    training_theorems = [
        "∀ n : Nat, n + 0 = n",
        "∀ a b : Nat, a + b = b + a",
        "∀ a b c : Nat, (a + b) + c = a + (b + c)",
        "∀ n : Nat, 0 * n = 0",
        "∀ n m : Nat, n * (m + 1) = n * m + n"
    ]

    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=5000
    )

    try:
        # Phase 1: Generate experiences
        print("Phase 1: Generating experiences...")
        results = await engine.run_batch_self_play(
            theorems=training_theorems,
            games_per_theorem=15
        )

        # Report initial results
        print("\nInitial Results:")
        for theorem, proof in results.items():
            status = "✓" if proof.is_valid else "✗"
            print(f"  {status} {theorem[:40]}...")

        # Phase 2: Train from experiences
        print("\nPhase 2: Training from experiences...")
        metrics = await engine.train_from_buffer(
            batch_size=16,
            iterations=50
        )

        # Report training metrics
        print("\nTraining Metrics:")
        print(f"  Total Games: {metrics.total_games}")
        print(f"  Success Rate: {metrics.success_rate:.1%}")
        print(f"  Average Reward: {metrics.avg_reward:.3f}")
        print(f"  Value Loss: {metrics.value_loss:.4f}")

        # Phase 3: Test on new theorem
        print("\nPhase 3: Testing on new theorem...")
        test_theorem = "∀ a b c d : Nat, a + b + c + d = d + c + b + a"
        test_proof = await engine.run_self_play(
            theorem=test_theorem,
            games=10
        )

        print(f"\nTest Result: {'✓ Valid' if test_proof.is_valid else '✗ Invalid'}")
        if test_proof.is_valid:
            print("\nLearned Proof:")
            print(test_proof.lean_code)

        # Save training progress
        engine.save_checkpoint("lean_selfplay_checkpoint.json")
        print("\nCheckpoint saved.")

    finally:
        await engine.close()

asyncio.run(main())
```

---

## 3. Advanced Workflows

### 3.1 Hybrid Evolution (Genetic → Adversarial → Self-Play)

**Scenario:** Combine all three evolutionary approaches for maximum robustness.

```python
import asyncio
import logging
from leanaide_evolution import LeanProofEvolutionEngine
from leanaide_adversarial import LeanAdversarialEvolution
from leanaide_selfplay import LeanSelfPlayEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def hybrid_evolution(theorem: str):
    """
    Complete hybrid evolutionary pipeline:
    1. Genetic evolution for broad search
    2. Adversarial evolution for robustness
    3. Self-play for final refinement
    """

    logger.info(f"Starting hybrid evolution for: {theorem[:50]}...")

    # Phase 1: Genetic Evolution
    logger.info("=" * 60)
    logger.info("PHASE 1: GENETIC EVOLUTION")
    logger.info("=" * 60)

    genetic_engine = LeanProofEvolutionEngine(
        theorem=theorem,
        population_size=50,
        max_generations=40,
        parallel_evaluation=True,
        max_concurrent=8
    )

    genetic_result = await genetic_engine.evolve()
    await genetic_engine.close()

    if genetic_result.success:
        logger.info(f"✓ Genetic evolution succeeded in {genetic_result.generations_completed} generations")
        logger.info(f"Best proof: {genetic_result.best_proof.lean_code[:100]}...")
        return genetic_result.best_proof

    logger.info(f"Genetic evolution incomplete, best fitness: {genetic_result.best_strategy.fitness:.3f}")

    # Phase 2: Adversarial Evolution
    logger.info("=" * 60)
    logger.info("PHASE 2: ADVERSARIAL EVOLUTION")
    logger.info("=" * 60)

    adversarial_evolution = LeanAdversarialEvolution(api_key="your-api-key")

    # Use best genetic strategy as starting point
    adv_proof, adv_rounds, adv_stats = await adversarial_evolution.run_adversarial_evolution(
        theorem=theorem,
        rounds=15
    )

    logger.info(f"Adversarial evolution complete")
    logger.info(f"  Blue win rate: {adv_stats.blue_success_rate:.1%}")
    logger.info(f"  Most effective approach: {adv_stats.most_effective_approach.value if adv_stats.most_effective_approach else 'N/A'}")

    if adv_proof.lean_code and adv_proof.verified:
        logger.info(f"✓ Adversarial evolution succeeded")
        logger.info(f"Final proof: {adv_proof.lean_code[:100]}...")
        return adv_proof

    # Phase 3: Self-Play Refinement
    logger.info("=" * 60)
    logger.info("PHASE 3: SELF-PLAY REFINEMENT")
    logger.info("=" * 60)

    selfplay_engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=2000
    )

    final_proof = await selfplay_engine.run_self_play(
        theorem=theorem,
        games=20
    )

    await selfplay_engine.close()

    if final_proof.is_valid:
        logger.info(f"✓ Self-play succeeded")
        logger.info(f"Final proof: {final_proof.lean_code[:100]}...")
        return final_proof

    logger.warning("All evolutionary approaches failed")
    return None

async def main():
    theorem = """
    theorem mul_add_distrib (a b c : Nat) :
        a * (b + c) = a * b + a * c
    """

    final_proof = await hybrid_evolution(theorem)

    if final_proof and final_proof.lean_code:
        print("\n" + "=" * 60)
        print("FINAL HYBRID PROOF")
        print("=" * 60)
        print(final_proof.lean_code)
    else:
        print("\nNo verified proof found through hybrid evolution")

asyncio.run(main())
```

---

### 3.2 Parallel Batch Processing

**Scenario:** Process multiple theorems in parallel using genetic evolution.

```python
import asyncio
from typing import List, Dict
from leanaide_evolution import evolve_proof

async def batch_evolve_parallel(
    theorems: List[str],
    max_concurrent: int = 5
) -> Dict[str, dict]:
    """
    Evolve proofs for multiple theorems in parallel.

    Args:
        theorems: List of theorem statements
        max_concurrent: Maximum concurrent evolutions

    Returns:
        Dictionary mapping theorems to results
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def evolve_with_semaphore(theorem: str, index: int):
        """Evolve with concurrency limit"""
        async with semaphore:
            print(f"[{index+1}/{len(theorems)}] Starting: {theorem[:50]}...")

            try:
                result = await evolve_proof(
                    theorem=theorem,
                    max_generations=30,
                    population_size=20,
                    parallel_evaluation=True
                )

                return {
                    "theorem": theorem,
                    "success": result.success,
                    "generations": result.generations_completed,
                    "evaluations": result.total_evaluations,
                    "fitness": result.best_strategy.fitness if result.best_strategy else 0.0,
                    "time": result.evolution_time,
                    "proof": result.best_proof.lean_code if result.success else None
                }

            except Exception as e:
                print(f"  ✗ Error: {e}")
                return {
                    "theorem": theorem,
                    "success": False,
                    "error": str(e)
                }

    # Run all evolutions concurrently
    tasks = [
        evolve_with_semaphore(theorem, i)
        for i, theorem in enumerate(theorems)
    ]

    results = await asyncio.gather(*tasks)

    # Organize results
    return {r["theorem"]: r for r in results}

async def main():
    """Batch processing example"""

    theorems = [
        "∀ n : Nat, n + 0 = n",
        "∀ a b : Nat, a + b = b + a",
        "∀ a b c : Nat, (a + b) + c = a + (b + c)",
        "∀ n : Nat, 0 * n = 0",
        "∀ n m : Nat, n * 0 = 0",
        "∀ n : Nat, 1 * n = n",
        "∀ n m : Nat, n * m = m * n",
        "∀ a b c : Nat, a * (b + c) = a * b + a * c"
    ]

    print(f"Processing {len(theorems)} theorems in parallel...")
    print("=" * 60)

    results = await batch_evolve_parallel(theorems, max_concurrent=4)

    # Summary report
    print("\n" + "=" * 60)
    print("BATCH EVOLUTION SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["success"])
    print(f"Successful: {successful}/{len(theorems)} ({successful/len(theorems):.1%})")

    # Detailed results
    for theorem, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} {theorem[:50]}...")

        if result["success"]:
            print(f"  Generations: {result['generations']}")
            print(f"  Evaluations: {result['evaluations']}")
            print(f"  Time: {result['time']:.1f}s")
            print(f"  Fitness: {result['fitness']:.3f}")
        else:
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Best Fitness: {result.get('fitness', 0):.3f}")

asyncio.run(main())
```

---

### 3.3 Evolutionary Pipeline with Checkpointing

**Scenario:** Long-running evolution with periodic checkpoints and resume capability.

```python
import asyncio
import json
from pathlib import Path
from datetime import datetime
from leanaide_evolution import LeanProofEvolutionEngine

class EvolutionaryPipeline:
    """Managed evolutionary pipeline with checkpointing"""

    def __init__(self, checkpoint_dir: str = "./evolution_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    async def run_with_checkpointing(
        self,
        theorem: str,
        max_generations: int = 100,
        checkpoint_interval: int = 10,
        resume_from: str = None
    ):
        """
        Run evolution with automatic checkpointing.

        Args:
            theorem: Theorem to prove
            max_generations: Maximum generations
            checkpoint_interval: Save checkpoint every N generations
            resume_from: Checkpoint file to resume from
        """

        engine = LeanProofEvolutionEngine(
            theorem=theorem,
            max_generations=max_generations,
            population_size=50
        )

        try:
            # Resume from checkpoint if specified
            if resume_from:
                print(f"Resuming from checkpoint: {resume_from}")
                await self._load_checkpoint(engine, resume_from)

            # Run evolution with periodic checkpointing
            async for generation in self._run_with_monitoring(engine):
                print(f"Generation {generation.current_generation}: "
                      f"Best Fitness = {generation.best_fitness:.3f}, "
                      f"Diversity = {generation.diversity_score:.3f}")

                # Checkpoint periodically
                if generation.current_generation % checkpoint_interval == 0:
                    checkpoint_path = self.checkpoint_dir / f"gen_{generation.current_generation}.json"
                    await self._save_checkpoint(engine, generation.current_generation, checkpoint_path)
                    print(f"  → Checkpoint saved: {checkpoint_path}")

        finally:
            await engine.close()

    async def _run_with_monitoring(self, engine):
        """Generator that yields generation statistics"""
        while engine.current_generation < engine.max_generations:
            await engine.evaluate_population()

            # Yield current state
            stats = engine.population.calculate_statistics()
            yield stats

            # Check convergence
            if self._check_convergence(engine):
                print("Convergence detected, stopping early")
                break

            # Create next generation
            await engine.create_next_generation()
            engine.current_generation += 1

    def _check_convergence(self, engine, window: int = 10, threshold: float = 0.001):
        """Check if evolution has converged"""
        if len(engine.statistics_history) < window:
            return False

        recent = engine.statistics_history[-window:]
        improvements = [
            abs(s.best_fitness - t.best_fitness)
            for s, t in zip(recent, recent[1:])
        ]

        return all(imp < threshold for imp in improvements)

    async def _save_checkpoint(self, engine, generation: int, path: Path):
        """Save checkpoint to disk"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "population": [
                {
                    "strategy_id": s.strategy_id,
                    "fitness": s.fitness,
                    "verified": s.verified,
                    "proof": s.proof.lean_code
                }
                for s in engine.population.strategies
            ],
            "statistics": [
                {
                    "generation": s.generation,
                    "best_fitness": s.best_fitness,
                    "avg_fitness": s.average_fitness
                }
                for s in engine.statistics_history[-10:]
            ]
        }

        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    async def _load_checkpoint(self, engine, path: Path):
        """Load checkpoint from disk"""
        with open(path, 'r') as f:
            checkpoint = json.load(f)

        # Restore state
        engine.current_generation = checkpoint["generation"]
        # (Full implementation would restore population strategies)

async def main():
    pipeline = EvolutionaryPipeline()

    theorem = "∀ a b c : Nat, (a + b) + c = a + (b + c)"

    await pipeline.run_with_checkpointing(
        theorem=theorem,
        max_generations=100,
        checkpoint_interval=10
    )

asyncio.run(main())
```

---

## 4. Domain-Specific Examples

### 4.1 Algebra: Ring Properties

**Scenario:** Prove algebraic properties using specialized tactics.

```python
import asyncio
from leanaide_evolution import evolve_proof, LeanProofEvolutionEngine

async def algebra_example():
    """Prove ring properties using algebraic tactics"""

    theorem = """
    theorem ring_properties (R : Type) [Ring R] (a b c : R) :
        a * (b + c) = a * b + a * c
    """

    # Configure for algebraic proofs
    engine = LeanProofEvolutionEngine(
        theorem=theorem,
        population_size=30,
        max_generations=40
    )

    # Add algebraic tactics to mutator
    algebraic_tactics = [
        "rw [mul_add]",      # Distributive property
        "ring",              # Ring solver
        "abel",              # Abelian group tactics
        "linarith",          # Linear arithmetic
        "norm_num",          # Normalize numbers
        "field_simp"         # Field simplification
    ]

    engine.mutator.custom_tactics = algebraic_tactics

    # Adjust fitness for algebraic proofs
    engine.evaluator.verification_weight = 12.0
    engine.evaluator.elegance_weight = 0.5  # Reward elegant algebraic manipulation

    result = await engine.evolve()
    await engine.close()

    if result.success:
        print("✓ Algebraic proof verified:")
        print(result.best_proof.lean_code)
    else:
        print(f"Best attempt: {result.best_strategy.fitness:.3f}")

asyncio.run(algebra_example())
```

---

### 4.2 Combinatorics: Inductive Proofs

**Scenario:** Prove combinatorial identities using induction.

```python
import asyncio
from leanaide_evolution import LeanProofEvolutionEngine

async def combinatorics_example():
    """Prove combinatorial identity using induction"""

    theorem = """
    theorem sum_natural_numbers (n : Nat) :
        (∑ i in Finset.range (n + 1), i) = n * (n + 1) / 2
    """

    engine = LeanProofEvolutionEngine(
        theorem=theorem,
        population_size=40,
        max_generations=50
    )

    # Add inductive tactics
    inductive_tactics = [
        "induction n with",
        "case zero",
        "case succ",
        "simp",
        "linarith",
        "ring"
    ]

    engine.mutator.custom_tactics = inductive_tactics

    # Higher mutation rate for exploring inductive structures
    engine.mutator.mutation_rate = 0.15

    result = await engine.evolve()
    await engine.close()

    print(f"Proof found: {result.success}")
    if result.success:
        print(result.best_proof.lean_code)

asyncio.run(combinatorics_example())
```

---

### 4.3 Logic: Quantifier Manipulation

**Scenario:** Prove logical statements using quantifier tactics.

```python
import asyncio
from leanaide_adversarial import LeanAdversarialEvolution

async def logic_example():
    """Prove logical statement using quantifier manipulation"""

    theorem = """
    theorem logic_example (P Q : α → Prop) :
        (∀ x, P x → Q x) → (∃ x, P x) → (∃ x, Q x)
    """

    evolution = LeanAdversarialEvolution(api_key="your-api-key")

    # Configure blue team for logical proofs
    evolution.blue_team.approaches = [
        ProofApproach.CONSTRUCTIVE,  # For existential quantifiers
        ProofApproach.CLASSICAL      # For classical reasoning
    ]

    proof, rounds, stats = await evolution.run_adversarial_evolution(
        theorem=theorem,
        rounds=10
    )

    print(f"Logical proof robustness: {stats.blue_success_rate:.1%}")
    if proof.lean_code:
        print(proof.lean_code)

asyncio.run(logic_example())
```

---

## 5. End-to-End Workflows

### 5.1 Complete Research Workflow

**Scenario:** Full workflow from problem statement to publication-ready proof.

```python
import asyncio
from pathlib import Path
from datetime import datetime
from leanaide_evolution import evolve_proof
from leanaide_adversarial import LeanAdversarialEvolution

class ResearchWorkflow:
    """Complete research workflow for theorem proving"""

    def __init__(self, output_dir: str = "./research_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def full_workflow(
        self,
        problem_statement: str,
        theorem_name: str,
        researcher: str = "AI Assistant"
    ):
        """
        Complete workflow from problem to publication.

        Steps:
        1. Problem analysis
        2. Evolutionary proof search
        3. Adversarial robustness testing
        4. Documentation generation
        5. Export to Lean file
        """

        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print(f"RESEARCH WORKFLOW: {theorem_name}")
        print("=" * 70)
        print(f"Researcher: {researcher}")
        print(f"Timestamp: {timestamp}")
        print(f"Problem: {problem_statement[:100]}...")
        print()

        # Phase 1: Evolutionary Proof Search
        print("PHASE 1: EVOLUTIONARY PROOF SEARCH")
        print("-" * 70)

        genetic_result = await evolve_proof(
            theorem=problem_statement,
            theorem_name=theorem_name,
            max_generations=50,
            population_size=40
        )

        if not genetic_result.success:
            print("✗ Evolutionary search failed")
            return None

        print(f"✓ Proof found in {genetic_result.generations_completed} generations")
        print(f"  Fitness: {genetic_result.best_strategy.fitness:.3f}")
        print(f"  Evaluations: {genetic_result.total_evaluations}")

        # Phase 2: Adversarial Robustness Testing
        print("\nPHASE 2: ADVERSARIAL ROBUSTNESS TESTING")
        print("-" * 70)

        adversarial = LeanAdversarialEvolution()
        final_proof, rounds, stats = await adversarial.run_adversarial_evolution(
            theorem=problem_statement,
            rounds=12
        )

        robustness_score = stats.blue_success_rate
        print(f"✓ Robustness score: {robustness_score:.1%}")
        print(f"  Most effective approach: {stats.most_effective_approach.value}")

        # Phase 3: Documentation Generation
        print("\nPHASE 3: DOCUMENTATION GENERATION")
        print("-" * 70)

        documentation = self._generate_documentation(
            theorem_name=theorem_name,
            problem_statement=problem_statement,
            proof=final_proof.lean_code if final_proof.lean_code else genetic_result.best_proof.lean_code,
            statistics={
                "generations": genetic_result.generations_completed,
                "evaluations": genetic_result.total_evaluations,
                "robustness": robustness_score,
                "approaches_tested": len(stats.approach_success_rates)
            },
            researcher=researcher,
            timestamp=timestamp
        )

        # Phase 4: Export to Lean File
        print("\nPHASE 4: LEAN FILE EXPORT")
        print("-" * 70)

        lean_file = self._export_lean_file(
            theorem_name=theorem_name,
            proof=final_proof.lean_code if final_proof.lean_code else genetic_result.best_proof.lean_code
        )

        print(f"✓ Lean file exported: {lean_file}")

        # Phase 5: Generate Report
        report_path = self.output_dir / f"{theorem_name}_report.md"
        with open(report_path, 'w') as f:
            f.write(documentation)

        print(f"\n✓ Research report: {report_path}")
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETE")
        print("=" * 70)

        return {
            "proof": final_proof.lean_code if final_proof.lean_code else genetic_result.best_proof.lean_code,
            "lean_file": str(lean_file),
            "report": str(report_path),
            "statistics": {
                "generations": genetic_result.generations_completed,
                "robustness": robustness_score
            }
        }

    def _generate_documentation(
        self,
        theorem_name: str,
        problem_statement: str,
        proof: str,
        statistics: dict,
        researcher: str,
        timestamp: str
    ) -> str:
        """Generate markdown documentation"""

        return f"""# {theorem_name}

**Researcher:** {researcher}
**Date:** {timestamp}
**Method:** Evolutionary LeanAide (Genetic + Adversarial)

## Problem Statement

```lean
{problem_statement}
```

## Proof

```lean
{proof}
```

## Statistics

- **Generations:** {statistics['generations']}
- **Evaluations:** {statistics['evaluations']}
- **Robustness Score:** {statistics['robustness']:.1%}
- **Approaches Tested:** {statistics['approaches_tested']}

## Methodology

This proof was discovered using evolutionary algorithms:
1. **Genetic Evolution:** Population-based search across proof strategies
2. **Adversarial Testing:** Red team vs blue team for robustness verification
3. **Approaches:** Constructive, classical, computational, indirect, structural, algebraic

## Verification

The proof has been formally verified using Lean 4 theorem prover.

"""

    def _export_lean_file(self, theorem_name: str, proof: str) -> Path:
        """Export proof to standalone Lean file"""

        lean_content = f"""import Mathlib

-- {theorem_name}
-- Automatically generated by Evolutionary LeanAide

{proof}
"""

        lean_path = self.output_dir / f"{theorem_name}.lean"
        with open(lean_path, 'w') as f:
            f.write(lean_content)

        return lean_path

async def main():
    workflow = ResearchWorkflow()

    problem = """
    theorem nat_mul_comm (a b : Nat) : a * b = b * a
    """

    result = await workflow.full_workflow(
        problem_statement=problem,
        theorem_name="nat_mul_comm",
        researcher="Evolutionary LeanAide"
    )

    if result:
        print("\nProof:")
        print(result["proof"])

asyncio.run(main())
```

---

## 6. Performance Tuning Examples

### 6.1 Optimizing for Speed

**Scenario:** Get results quickly with acceptable quality.

```python
import asyncio
from leanaide_evolution import evolve_proof

async def fast_evolution():
    """Optimized for speed over quality"""

    result = await evolve_proof(
        theorem="∀ n : Nat, n + 0 = n",
        max_generations=15,       # Fewer generations
        population_size=15,       # Smaller population
        parallel_evaluation=True,
        max_concurrent=10,        # High parallelism
        cache_enabled=True        # Enable caching
    )

    print(f"Fast evolution: {result.evolution_time:.1f}s")
    print(f"Success: {result.success}")

asyncio.run(fast_evolution())
```

---

### 6.2 Optimizing for Quality

**Scenario:** Get best possible proof regardless of time.

```python
import asyncio
from leanaide_evolution import LeanProofEvolutionEngine

async def quality_evolution():
    """Optimized for quality over speed"""

    engine = LeanProofEvolutionEngine(
        theorem="∀ a b c : Nat, (a + b) + c = a + (b + c)",
        population_size=100,       # Large population
        max_generations=150,       # Many generations
        mutation_rate=0.15,        # Higher exploration
        elitism_ratio=0.15,        # Preserve more elites
        verification_weight=15.0,  # Emphasize verification
        elegance_weight=0.5,       # Reward elegance
        parallel_evaluation=True,
        max_concurrent=15
    )

    result = await engine.evolve()

    print(f"Quality evolution: {result.evolution_time:.1f}s")
    print(f"Fitness: {result.best_strategy.fitness:.3f}")

asyncio.run(quality_evolution())
```

---

## 7. Common Patterns

### 7.1 Retry Pattern

```python
import asyncio
from leanaide_evolution import evolve_proof

async def evolve_with_retry(theorem: str, max_retries: int = 3):
    """Retry evolution with different parameters"""

    for attempt in range(max_retries):
        try:
            # Increase parameters each retry
            population = 20 + attempt * 10
            generations = 30 + attempt * 10

            result = await evolve_proof(
                theorem=theorem,
                population_size=population,
                max_generations=generations
            )

            if result.success:
                print(f"✓ Success on attempt {attempt + 1}")
                return result

            print(f"Attempt {attempt + 1} incomplete, retrying...")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

    print("All retries exhausted")
    return None
```

---

### 7.2 Ensemble Pattern

```python
import asyncio
from leanaide_evolution import evolve_proof
from leanaide_adversarial import evolve_lean_proof

async def ensemble_evolution(theorem: str):
    """Run multiple evolutionary approaches and select best"""

    # Run all approaches
    genetic_task = evolve_proof(theorem, max_generations=30)
    adversarial_task = evolve_lean_proof(theorem, rounds=10)

    results = await asyncio.gather(genetic_task, adversarial_task)

    # Select best result
    genetic_success = results[0].success
    adversarial_confidence = results[1]['confidence']

    if genetic_success and adversarial_confidence > 0.8:
        print("Both approaches succeeded!")
        return results[0].best_proof

    if genetic_success:
        print("Genetic approach succeeded")
        return results[0].best_proof

    if adversarial_confidence > 0.8:
        print("Adversarial approach succeeded")
        # Convert adversarial result to proof
        return results[1]['proof']

    print("No approach succeeded")
    return None
```

---

## 8. Migration Examples

### 8.1 From Manual to Evolutionary

**Before (Manual):**
```python
from lean4_integration import verify_proof

proof_code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  induction b
  case zero =>
    rw [Nat.add_zero]
  case succ n ih =>
    rw [Nat.add_succ, ih, Nat.add_succ]
"""

result = verify_proof(proof_code)
print(f"Valid: {result.is_valid}")
```

**After (Evolutionary):**
```python
from leanaide_evolution import evolve_proof

async def migrate():
    result = await evolve_proof(
        theorem="∀ a b : Nat, a + b = b + a",
        max_generations=30
    )

    if result.success:
        print(f"Proof discovered automatically!")
        print(f"Generations: {result.generations_completed}")
        print(f"Proof: {result.best_proof.lean_code}")

asyncio.run(migrate())
```

---

**Document End**

For complete API reference, see `LEANAIDE_EVOLUTIONARY_API.md`
For usage guide, see `LEANAIDE_EVOLUTIONARY_GUIDE.md`
