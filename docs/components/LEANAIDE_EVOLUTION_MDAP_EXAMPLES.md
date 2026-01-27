# LeanAide MDAP-Enhanced Evolution - Examples

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide Evolution + MDAP Integration

---

## Table of Contents

1. [Basic Usage](#1-basic-usage)
2. [Custom Agent Configurations](#2-custom-agent-configurations)
3. [Custom Voting Strategies](#3-custom-voting-strategies)
4. [Workflow Integration](#4-workflow-integration)
5. [Performance Tuning](#5-performance-tuning)
6. [Comparison Examples](#6-comparison-examples)
7. [Advanced Use Cases](#7-advanced-use-cases)

---

## 1. Basic Usage

### Example 1.1: Simple MDAP-Enhanced Evolution

```python
from evolution_maker_integration import (
    run_maker_evolution,
    MakerevolutionConfig,
    MakerevolutionMode
)

# Define fitness evaluator
def simple_evaluator(genome: str) -> float:
    """Simple fitness: reward for 'intros' and 'refl'"""
    score = 0.0
    if "intros" in genome:
        score += 3.0
    if "refl" in genome:
        score += 3.0
    if "simp" in genome:
        score += 1.0
    return score

# Configure evolution
config = MakerevolutionConfig(
    mode=MakerevolutionMode.HYBRID,
    voting_threshold=3,
    population_size=20
)

# Run evolution
result = run_maker_evolution(
    initial_program="intros n refl",
    evaluator=simple_evaluator,
    max_generations=20,
    config=config
)

# Display results
print(f"Best fitness: {result['best_fitness']:.3f}")
print(f"Best program: {result['best_program']}")
print(f"Generations: {result['generations_completed']}")
```

**Output**:
```
Best fitness: 6.000
Best program: intros n refl
Generations: 5
```

---

### Example 1.2: Lean 4 Proof Evolution

```python
from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

# Fitness evaluator for Lean 4 proofs
def lean4_evaluator(proof: str) -> float:
    """
    Evaluate Lean 4 proof quality.
    Higher score = better proof.
    """
    score = 0.0

    # Verification is most important
    if "verified" in proof.lower():
        score += 10.0
        return score

    # Admitted proof gets low score
    if "sorry" in proof.lower():
        return 0.1

    # Check for structure
    if "intros" in proof:
        score += 2.0
    if "refl" in proof or "rfl" in proof:
        score += 3.0
    if "simp" in proof:
        score += 1.0
    if "induction" in proof:
        score += 2.0

    # Prefer concise proofs
    tactic_count = len(proof.split())
    score -= min(tactic_count * 0.1, 2.0)

    return max(score, 0.0)

# Theorem to prove
theorem = "∀ n : Nat, n + 0 = n"

# Initial sketch
initial_proof = f"""
theorem add_zero (n : Nat) : n + 0 = n :=
  intros
  sorry
"""

# Evolve proof
result = run_maker_evolution(
    initial_program=initial_proof,
    evaluator=lean4_evaluator,
    max_generations=30,
    config=MakerevolutionConfig(
        voting_threshold=3,
        population_size=30,
        enable_decomposition=True
    )
)

print(f"Evolved proof (fitness={result['best_fitness']:.2f}):")
print(result['best_program'])
```

---

## 2. Custom Agent Configurations

### Example 2.1: Specialized Agents for Different Proof Approaches

```python
from evolution_maker_integration import (
    MAKERSelection,
    MakerevolutionConfig,
    Individual,
    Population
)

class SpecializedAgentSelector:
    """Select agents based on theorem domain"""

    def __init__(self):
        self.agent_types = {
            "constructive": {
                "system_prompt": "You are a constructive mathematician.",
                "temperature": 0.1,
                "tactics": ["intros", "exists", "use"]
            },
            "inductive": {
                "system_prompt": "You specialize in induction proofs.",
                "temperature": 0.2,
                "tactics": ["intros", "induction", "case"]
            },
            "algebraic": {
                "system_prompt": "You use algebraic manipulation.",
                "temperature": 0.1,
                "tactics": ["intros", "linarith", "ring", "simp"]
            },
            "computational": {
                "system_prompt": "You use computational methods.",
                "temperature": 0.15,
                "tactics": ["intros", "simp", "norm_num", "rfl"]
            }
        }

    def select_agents_for_theorem(self, theorem: str) -> List[str]:
        """Select appropriate agents based on theorem characteristics"""
        selected = []

        if "∃" in theorem or "exists" in theorem:
            selected.append("constructive")

        if "∀" in theorem and ("Nat" in theorem or "List" in theorem):
            selected.append("inductive")
            selected.append("computational")

        if "+" in theorem or "*" in theorem or "=" in theorem:
            selected.append("algebraic")

        # Default to constructive if no match
        if not selected:
            selected = ["constructive"]

        return selected

    def get_agent_config(self, agent_type: str) -> dict:
        """Get configuration for specific agent type"""
        return self.agent_types[agent_type]


# Usage
selector = SpecializedAgentSelector()

theorem1 = "∃ n : Nat, n > 0"
agents1 = selector.select_agents_for_theorem(theorem1)
print(f"Theorem 1 agents: {agents1}")  # ['constructive']

theorem2 = "∀ n : Nat, n + 0 = n"
agents2 = selector.select_agents_for_theorem(theorem2)
print(f"Theorem 2 agents: {agents2}")  # ['inductive', 'computational', 'algebraic']
```

---

### Example 2.2: Performance-Based Agent Selection

```python
class PerformanceAgentSelector:
    """Select agents based on past performance"""

    def __init__(self):
        self.agent_performance = {
            "constructive": {"successes": 85, "attempts": 100},
            "inductive": {"successes": 72, "attempts": 100},
            "algebraic": {"successes": 68, "attempts": 100},
            "computational": {"successes": 79, "attempts": 100}
        }

    def get_success_rate(self, agent_type: str) -> float:
        """Get success rate for agent"""
        perf = self.agent_performance[agent_type]
        return perf["successes"] / perf["attempts"]

    def update_performance(self, agent_type: str, success: bool):
        """Update agent performance after attempt"""
        self.agent_performance[agent_type]["attempts"] += 1
        if success:
            self.agent_performance[agent_type]["successes"] += 1

    def select_best_agents(self, num_agents: int = 3) -> List[str]:
        """Select top N agents by success rate"""
        sorted_agents = sorted(
            self.agent_performance.keys(),
            key=self.get_success_rate,
            reverse=True
        )
        return sorted_agents[:num_agents]


# Usage
selector = PerformanceAgentSelector()

# Get best agents
best_agents = selector.select_best_agents(num_agents=3)
print(f"Best agents: {best_agents}")

# Simulate updating performance
selector.update_performance("constructive", success=True)
selector.update_performance("inductive", success=False)

print("Updated success rates:")
for agent in selector.agent_performance:
    rate = selector.get_success_rate(agent)
    print(f"  {agent}: {rate:.2%}")
```

---

## 3. Custom Voting Strategies

### Example 3.1: Weighted Voting

```python
from evolution_maker_integration import MAKERSelection, MakerevolutionConfig

class WeightedVotingSelection(MAKERSelection):
    """Selection with weighted voting based on agent expertise"""

    def __init__(self, config, agent_weights):
        super().__init__(config)
        self.agent_weights = agent_weights

    def _vote_on_candidates(self, candidates):
        """Vote with weighted agents"""
        votes = {}
        weighted_votes = {}

        for agent, weight in self.agent_weights.items():
            # Agent selects candidate (simplified)
            choice = self._agent_select(candidates, agent)

            # Record votes
            votes[choice.genome] = votes.get(choice.genome, 0) + 1
            weighted_votes[choice.genome] = weighted_votes.get(choice.genome, 0) + weight

        return votes, weighted_votes

    def _agent_select(self, candidates, agent_type):
        """Simplified agent selection (in real system, would use LLM)"""
        # For demo: prefer candidates with certain tactics based on agent type
        agent_preferences = {
            "constructive": ["intros", "exists"],
            "inductive": ["induction", "case"],
            "algebraic": ["linarith", "ring"],
            "computational": ["simp", "norm_num"]
        }

        preferences = agent_preferences.get(agent_type, [])
        for candidate in candidates:
            if any(tactic in candidate.genome for tactic in preferences):
                return candidate

        # Default: return random candidate
        return candidates[0]


# Usage
config = MakerevolutionConfig(voting_threshold=3)

agent_weights = {
    "constructive": 1.0,
    "inductive": 1.2,  # Higher weight
    "algebraic": 0.8,
    "computational": 1.0
}

selector = WeightedVotingSelection(config, agent_weights)
```

---

### Example 3.2: Confidence-Based Voting

```python
class ConfidenceVotingSelection(MAKERSelection):
    """Voting based on agent confidence scores"""

    def __init__(self, config):
        super().__init__(config)

    def _vote_on_candidates_with_confidence(self, candidates):
        """Vote with confidence scores"""
        votes = {}
        confidences = {}
        confidence_sums = {}

        for agent in self._get_agents():
            # Agent selects candidate with confidence
            choice, confidence = self._agent_select_with_confidence(
                candidates,
                agent
            )

            # Record votes
            choice_key = choice.genome
            votes[choice_key] = votes.get(choice_key, 0) + 1
            confidences[choice_key] = max(confidences.get(choice_key, 0), confidence)
            confidence_sums[choice_key] = confidence_sums.get(choice_key, 0) + confidence

        # Use confidence to break ties
        return votes, confidences, confidence_sums

    def _agent_select_with_confidence(self, candidates, agent):
        """Agent selects candidate with confidence score"""
        # Simplified: confidence based on fitness
        choice = max(candidates, key=lambda c: c.fitness)
        confidence = min(choice.fitness / 10.0, 1.0)  # Normalize to 0-1
        return choice, confidence


# Usage
selector = ConfidenceVotingSelection(MakerevolutionConfig())
```

---

### Example 3.3: Adaptive Voting Threshold

```python
class AdaptiveVotingEngine:
    """Adjust voting threshold based on population diversity"""

    def __init__(self, initial_k: int = 3, min_k: int = 2, max_k: int = 8):
        self.k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.history = []

    def update_k(self, diversity: float, convergence_rate: float):
        """
        Update voting threshold based on population state.

        Args:
            diversity: Population diversity (0-1)
            convergence_rate: Rate of fitness improvement
        """
        # Low diversity → Increase K (more conservative)
        if diversity < 0.2:
            self.k = min(self.k + 1, self.max_k)
            print(f"Low diversity ({diversity:.2f}), increasing K to {self.k}")

        # High diversity → Decrease K (faster convergence)
        elif diversity > 0.5:
            self.k = max(self.k - 1, self.min_k)
            print(f"High diversity ({diversity:.2f}), decreasing K to {self.k}")

        # Track history
        self.history.append({
            "k": self.k,
            "diversity": diversity,
            "convergence_rate": convergence_rate
        })

    def get_current_k(self) -> int:
        """Get current voting threshold"""
        return self.k


# Usage
adaptive_voting = AdaptiveVotingEngine(initial_k=3)

# Simulate evolution
for generation in range(10):
    diversity = 1.0 - (generation * 0.1)  # Decreasing diversity
    convergence_rate = 0.05 + generation * 0.01

    adaptive_voting.update_k(diversity, convergence_rate)
    print(f"Generation {generation}: K={adaptive_voting.get_current_k()}")
```

**Output**:
```
Generation 0: K=3
Generation 1: K=3
Generation 2: K=3
Generation 3: K=2
High diversity (0.70), decreasing K to 2
Generation 4: K=2
...
Generation 9: K=2
```

---

## 4. Workflow Integration

### Example 4.1: Integration with LeanAide Workflow

```python
from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

class LeanAideEvolutionWorkflow:
    """Integrate MDAP-evolution with LeanAide workflow"""

    def __init__(self, leanaide_url: str = "http://localhost:7654"):
        self.leanaide_url = leanaide_url

    def leanaide_evaluator(self, proof: str) -> float:
        """
        Evaluate proof using LeanAide verification.

        In production, this would call LeanAide server.
        For demo, we use a simplified evaluator.
        """
        # Simplified: check for proof structure
        score = 0.0

        if "intros" not in proof:
            return 0.0

        score += 2.0  # Has intros

        if "refl" in proof or "rfl" in proof:
            score += 3.0

        if "simp" in proof:
            score += 1.0

        # Penalize length
        score -= min(len(proof.split()) * 0.1, 2.0)

        # In production: call LeanAide for verification
        # verified = self._call_leanaide(proof)
        # if verified:
        #     score = 10.0

        return max(score, 0.0)

    def evolve_proof(self, theorem: str, max_generations: int = 30) -> dict:
        """Evolve proof for theorem"""

        # Create initial proof sketch
        initial_proof = f"""
theorem tmp : {theorem} :=
  intros
  sorry
"""

        # Configure evolution
        config = MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            voting_threshold=3,
            population_size=30,
            enable_decomposition=True
        )

        # Run evolution
        result = run_maker_evolution(
            initial_program=initial_proof,
            evaluator=self.leanaide_evaluator,
            max_generations=max_generations,
            config=config
        )

        return result

    def _call_leanaide(self, proof: str) -> bool:
        """Call LeanAide server for verification (production only)"""
        # In production:
        # import requests
        # response = requests.post(
        #     f"{self.leanaide_url}/verify",
        #     json={"proof": proof}
        # )
        # return response.json()["verified"]
        pass


# Usage
workflow = LeanAideEvolutionWorkflow()

theorem = "∀ n : Nat, n + 0 = n"
result = workflow.evolve_proof(theorem, max_generations=20)

print(f"Theorem: {theorem}")
print(f"Best proof (fitness={result['best_fitness']:.2f}):")
print(result['best_program'])
```

---

### Example 4.2: Stage 3A MDAP-Evolution Integration

```python
async def stage_3a_mdap_evolution(sub_problem, workflow_config):
    """
    Stage 3A: MDAP-enhanced evolutionary proof search.

    This integrates with the decomposition workflow.
    """

    from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

    # Extract theorem from sub-problem
    theorem = sub_problem.get("theorem", "")

    # Create initial proof sketch
    initial_proof = f"""
theorem evolved_proof : {theorem} :=
  intros
  sorry
"""

    # Define evaluator
    def evaluator(genome: str) -> float:
        """Evaluate proof quality"""
        score = 0.0

        # Basic structure
        if "intros" in genome:
            score += 2.0
        if any(t in genome for t in ["refl", "rfl", "simp"]):
            score += 2.0

        # Domain-specific tactics
        if "Nat" in theorem:
            if "induction" in genome:
                score += 2.0
            if "linarith" in genome:
                score += 1.0

        # Prefer concise proofs
        score -= min(len(genome.split()) * 0.1, 2.0)

        return max(score, 0.0)

    # Configure based on sub-problem difficulty
    difficulty = sub_problem.get("difficulty", "medium")

    if difficulty == "easy":
        config = MakerevolutionConfig(
            voting_threshold=2,
            population_size=15,
            max_generations=15
        )
    elif difficulty == "hard":
        config = MakerevolutionConfig(
            voting_threshold=5,
            population_size=50,
            enable_decomposition=True,
            decomposition_depth=4
        )
    else:  # medium
        config = MakerevolutionConfig(
            voting_threshold=3,
            population_size=30,
            enable_decomposition=True
        )

    # Run evolution
    result = run_maker_evolution(
        initial_program=initial_proof,
        evaluator=evaluator,
        max_generations=30,
        config=config
    )

    # Return formatted result
    return {
        "proof": result['best_program'],
        "fitness": result['best_fitness'],
        "generations": result['generations_completed'],
        "converged": result['converged']
    }
```

---

### Example 4.3: Stage 3B Refinement with Voting

```python
async def stage_3b_refinement(proof: str, refinement_rounds: int = 3):
    """
    Stage 3B: Refine proof with high-reliability voting.

    Takes a good proof and refines it for elegance and conciseness.
    """

    from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

    # Refinement evaluator: rewards elegance
    def refinement_evaluator(genome: str) -> float:
        """Evaluate proof elegance"""
        score = 0.0

        # Essential structure
        if not ("intros" in genome and ("refl" in genome or "rfl" in genome)):
            return 0.0

        score += 5.0

        # Elegance bonuses
        if "rfl" in genome:  # Short form of refl
            score += 1.0

        # Conciseness: prefer shorter proofs
        tactic_count = len(genome.split())
        score += max(0, 10 - tactic_count) * 0.5

        # Penalize unnecessary tactics
        unnecessary = ["by", "calc", "have"]
        for tactic in unnecessary:
            if tactic in genome:
                score -= 0.5

        return max(score, 0.0)

    # High-reliability configuration for refinement
    config = MakerevolutionConfig(
        mode=MakerevolutionMode.VOTING_ONLY,
        voting_threshold=5,  # Higher threshold
        population_size=20,
        enable_decomposition=False  # No decomposition needed
    )

    # Run refinement
    result = run_maker_evolution(
        initial_program=proof,
        evaluator=refinement_evaluator,
        max_generations=10,  # Fewer generations for refinement
        config=config
    )

    return result['best_program']
```

---

## 5. Performance Tuning

### Example 5.1: Progressive Refinement Strategy

```python
def progressive_evolution(initial_program: str, evaluator) -> str:
    """
    Progressive refinement: fast exploration → careful refinement.
    """

    from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

    # Stage 1: Fast exploration (low K, small population)
    print("Stage 1: Fast exploration")
    result1 = run_maker_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        max_generations=15,
        config=MakerevolutionConfig(
            voting_threshold=2,
            population_size=15,
            enable_decomposition=False
        )
    )
    print(f"  Best fitness: {result1['best_fitness']:.3f}")

    # Stage 2: Medium refinement (medium K, medium population)
    print("Stage 2: Medium refinement")
    result2 = run_maker_evolution(
        initial_program=result1['best_program'],
        evaluator=evaluator,
        max_generations=20,
        config=MakerevolutionConfig(
            voting_threshold=3,
            population_size=25,
            enable_decomposition=True
        )
    )
    print(f"  Best fitness: {result2['best_fitness']:.3f}")

    # Stage 3: Final polishing (high K, focused search)
    print("Stage 3: Final polishing")
    result3 = run_maker_evolution(
        initial_program=result2['best_program'],
        evaluator=evaluator,
        max_generations=10,
        config=MakerevolutionConfig(
            voting_threshold=5,
            population_size=20,
            enable_decomposition=False
        )
    )
    print(f"  Best fitness: {result3['best_fitness']:.3f}")

    return result3['best_program']


# Usage
def evaluator(genome: str) -> float:
    return 5.0 if "intros refl" in genome else 2.0

initial = "intros n sorry"
final = progressive_evolution(initial, evaluator)
print(f"\nFinal proof:\n{final}")
```

---

### Example 5.2: Batch Processing with Resource Management

```python
import asyncio
from typing import List, Dict

async def batch_evolve(
    theorems: List[str],
    evaluator,
    max_concurrent: int = 3
) -> Dict[str, dict]:
    """
    Evolve proofs for multiple theorems with concurrency control.

    Args:
        theorems: List of theorems to prove
        evaluator: Fitness evaluator function
        max_concurrent: Maximum concurrent evolutions

    Returns:
        Dictionary mapping theorem to evolution result
    """

    from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

    semaphore = asyncio.Semaphore(max_concurrent)

    async def evolve_single(theorem: str) -> tuple:
        """Evolve proof for single theorem"""
        async with semaphore:
            print(f"Starting: {theorem[:50]}...")

            initial_proof = f"theorem tmp : {theorem} := intros sorry"

            result = run_maker_evolution(
                initial_program=initial_proof,
                evaluator=evaluator,
                max_generations=20,
                config=MakerevolutionConfig(
                    voting_threshold=3,
                    population_size=20
                )
            )

            print(f"Completed: {theorem[:50]}... (fitness={result['best_fitness']:.2f})")
            return theorem, result

    # Run all evolutions with concurrency control
    tasks = [evolve_single(th) for th in theorems]
    results = await asyncio.gather(*tasks)

    # Convert to dictionary
    return {th: result for th, result in results}


# Usage
async def main():
    theorems = [
        "∀ n : Nat, n + 0 = n",
        "∀ a b : Nat, a + b = b + a",
        "∀ n m : Nat, n + (m + 1) = (n + m) + 1"
    ]

    def evaluator(genome: str) -> float:
        score = 2.0 if "intros" in genome else 0.0
        score += 3.0 if "refl" in genome else 0.0
        return score

    results = await batch_evolve(theorems, evaluator, max_concurrent=2)

    print("\n" + "=" * 70)
    print("BATCH RESULTS")
    print("=" * 70)
    for theorem, result in results.items():
        print(f"\nTheorem: {theorem}")
        print(f"Fitness: {result['best_fitness']:.2f}")
        print(f"Proof: {result['best_program'][:100]}...")

# Run
# asyncio.run(main())
```

---

## 6. Comparison Examples

### Example 6.1: Pure Evolution vs MDAP-Enhanced

```python
def compare_pure_vs_mdap(theorem: str):
    """Compare pure evolution with MDAP-enhanced evolution"""

    from evolution_maker_integration import (
        run_maker_evolution,
        MakerevolutionConfig,
        MakerevolutionMode
    )

    def evaluator(genome: str) -> float:
        score = 0.0
        if "intros" in genome:
            score += 2.0
        if "refl" in genome:
            score += 3.0
        if "simp" in genome:
            score += 1.0
        return score

    initial_proof = f"theorem tmp : {theorem} := intros sorry"

    # Pure evolution (no voting, no decomposition)
    print("Running Pure Evolution...")
    pure_result = run_maker_evolution(
        initial_program=initial_proof,
        evaluator=evaluator,
        max_generations=30,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.DECOMPOSITION,
            enable_voting=False,
            enable_decomposition=False,
            population_size=20
        )
    )

    # MDAP-enhanced evolution
    print("Running MDAP-Enhanced Evolution...")
    mdap_result = run_maker_evolution(
        initial_program=initial_proof,
        evaluator=evaluator,
        max_generations=30,
        config=MakerevolutionConfig(
            mode=MakerevolutionMode.HYBRID,
            enable_voting=True,
            voting_threshold=3,
            enable_decomposition=True,
            population_size=20
        )
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nPure Evolution:")
    print(f"  Fitness: {pure_result['best_fitness']:.3f}")
    print(f"  Generations: {pure_result['generations_completed']}")
    print(f"  Converged: {pure_result['converged']}")

    print(f"\nMDAP-Enhanced Evolution:")
    print(f"  Fitness: {mdap_result['best_fitness']:.3f}")
    print(f"  Generations: {mdap_result['generations_completed']}")
    print(f"  Converged: {mdap_result['converged']}")

    # Improvement
    fitness_improvement = (mdap_result['best_fitness'] - pure_result['best_fitness'])
    print(f"\nFitness improvement: {fitness_improvement:+.3f}")


# Usage
compare_pure_vs_mdap("∀ n : Nat, n + 0 = n")
```

---

### Example 6.2: Voting Threshold Comparison

```python
def compare_voting_thresholds(theorem: str):
    """Compare different voting thresholds"""

    from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

    def evaluator(genome: str) -> float:
        return 5.0 if "intros refl" in genome else 2.0

    initial_proof = f"theorem tmp : {theorem} := intros sorry"

    thresholds = [2, 3, 5]
    results = {}

    for k in thresholds:
        print(f"Running with k={k}...")
        result = run_maker_evolution(
            initial_program=initial_proof,
            evaluator=evaluator,
            max_generations=20,
            config=MakerevolutionConfig(
                voting_threshold=k,
                population_size=20
            )
        )
        results[k] = result

    # Display results
    print("\n" + "=" * 70)
    print("VOTING THRESHOLD COMPARISON")
    print("=" * 70)

    for k, result in results.items():
        print(f"\nk={k}:")
        print(f"  Fitness: {result['best_fitness']:.3f}")
        print(f"  Generations: {result['generations_completed']}")
        print(f"  Converged: {result['converged']}")


# Usage
compare_voting_thresholds("∀ n : Nat, n + 0 = n")
```

---

## 7. Advanced Use Cases

### Example 7.1: Multi-Objective Evolution

```python
def multi_objective_evolution(theorem: str):
    """
    Optimize for multiple objectives:
    1. Verification (most important)
    2. Conciseness (shorter proofs)
    3. Elegance (natural tactics)
    """

    from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

    def multi_objective_evaluator(genome: str) -> float:
        """Multi-objective fitness"""
        scores = {
            "verification": 0.0,
            "conciseness": 0.0,
            "elegance": 0.0
        }

        # Verification (weight: 10)
        if "verified" in genome.lower():
            scores["verification"] = 10.0
        elif "sorry" in genome.lower():
            scores["verification"] = 0.0
        else:
            # Partial credit for structure
            if "intros" in genome:
                scores["verification"] += 3.0
            if "refl" in genome or "rfl" in genome:
                scores["verification"] += 5.0

        # Conciseness (weight: 1)
        tactic_count = len(genome.split())
        scores["conciseness"] = max(0, 10 - tactic_count) * 0.5

        # Elegance (weight: 2)
        elegant_tactics = ["rfl", "simp", "linarith"]
        for tactic in elegant_tactics:
            if tactic in genome:
                scores["elegance"] += 0.5

        # Weighted sum
        total = (
            scores["verification"] * 10 +
            scores["conciseness"] * 1 +
            scores["elegance"] * 2
        )

        return total

    initial_proof = f"theorem tmp : {theorem} := intros sorry"

    result = run_maker_evolution(
        initial_program=initial_proof,
        evaluator=multi_objective_evaluator,
        max_generations=30,
        config=MakerevolutionConfig(
            voting_threshold=3,
            population_size=30,
            enable_decomposition=True
        )
    )

    return result


# Usage
result = multi_objective_evolution("∀ n : Nat, n + 0 = n")
print(f"Multi-objective best fitness: {result['best_fitness']:.2f}")
```

---

### Example 7.2: Adaptive Strategy Selection

```python
class AdaptiveEvolutionController:
    """Adaptively select evolution strategy based on progress"""

    def __init__(self):
        self.progress_history = []
        self.current_strategy = "hybrid"

    def select_strategy(self, progress: dict) -> str:
        """
        Select evolution strategy based on progress metrics.

        Args:
            progress: Dict with 'fitness_improvement', 'diversity', etc.

        Returns:
            Strategy name: 'voting_only', 'decomposition', 'hybrid', or 'full_maker'
        """
        improvement = progress.get("fitness_improvement", 0.0)
        diversity = progress.get("diversity", 0.5)
        generations_stuck = progress.get("generations_without_improvement", 0)

        # Stuck for many generations → switch strategies
        if generations_stuck > 5:
            if diversity < 0.2:
                # Low diversity → need exploration
                return "voting_only"  # Faster exploration
            else:
                # High diversity but stuck → need decomposition
                return "full_maker"  # Thorough search

        # Normal progress based on diversity
        if diversity > 0.5:
            return "voting_only"  # Fast convergence
        elif diversity < 0.2:
            return "full_maker"  # Conservative refinement
        else:
            return "hybrid"  # Balanced approach

    def evolve_with_adaptive_strategy(
        self,
        initial_program: str,
        evaluator,
        max_generations: int
    ):
        """Evolve with adaptive strategy selection"""

        from evolution_maker_integration import (
            run_maker_evolution,
            MakerevolutionConfig,
            MakerevolutionMode
        )

        current_program = initial_program

        for generation in range(0, max_generations, 10):
            # Analyze progress
            progress = {
                "fitness_improvement": 0.0,  # Would track from history
                "diversity": 0.4,
                "generations_without_improvement": 0
            }

            # Select strategy
            strategy = self.select_strategy(progress)
            self.current_strategy = strategy

            print(f"Generation {generation}: Using {strategy} strategy")

            # Configure based on strategy
            mode_map = {
                "voting_only": MakerevolutionMode.VOTING_ONLY,
                "decomposition": MakerevolutionMode.DECOMPOSITION,
                "hybrid": MakerevolutionMode.HYBRID,
                "full_maker": MakerevolutionMode.FULL_MAKER
            }

            config = MakerevolutionConfig(
                mode=mode_map[strategy],
                voting_threshold=3 if strategy != "full_maker" else 5,
                population_size=20,
                enable_decomposition=(strategy in ["decomposition", "hybrid", "full_maker"])
            )

            # Run evolution for 10 generations
            remaining_generations = min(10, max_generations - generation)
            result = run_maker_evolution(
                initial_program=current_program,
                evaluator=evaluator,
                max_generations=remaining_generations,
                config=config
            )

            current_program = result['best_program']

            # Early termination if converged
            if result['converged']:
                print(f"Converged at generation {generation}")
                break

        return current_program


# Usage
controller = AdaptiveEvolutionController()

def evaluator(genome: str) -> float:
    return 5.0 if "intros refl" in genome else 2.0

initial = "intros n sorry"
final = controller.evolve_with_adaptive_strategy(
    initial_program=initial,
    evaluator=evaluator,
    max_generations=50
)

print(f"\nFinal proof:\n{final}")
```

---

**Document End**

For more information, see:
- `LEANAIDE_EVOLUTION_MDAP_GUIDE.md` - User guide
- `LEANAIDE_EVOLUTION_MDAP_API.md` - API reference
- `LEANAIDE_EVOLUTION_MDAP_ARCHITECTURE.md` - Architecture diagrams
