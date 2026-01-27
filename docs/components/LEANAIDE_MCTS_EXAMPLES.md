# LeanAide MCTS Examples

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Custom Rollout Policies](#custom-rollout-policies)
3. [Parallel MCTS](#parallel-mcts)
4. [MCTS in Workflow](#mcts-in-workflow)
5. [Hybrid MCTS + Evolution](#hybrid-mcts--evolution)
6. [Domain-Specific MCTS](#domain-specific-mcts)
7. [Performance Tuning](#performance-tuning)
8. [Advanced Examples](#advanced-examples)

---

## Basic Usage

### Example 1: Simple MCTS Search

```python
from leanaide_mcts import LeanProofMCTS, ProofContext

# Create initial proof context
context = ProofContext(
    goal="∀ n : Nat, n + 0 = n",
    hypotheses=[],
    available_lemmas=["Nat.add_zero"],
    depth=0
)

# Initialize MCTS with default parameters
mcts = LeanProofMCTS(
    exploration_constant=1.414,  # Balanced exploration
    simulations=1000,             # Number of simulations
    rollout_depth=7               # Rollout depth
)

# Run search
best_sequence, root = mcts.search(context)

# Display results
print("Best proof sequence:")
for i, action in enumerate(best_sequence, 1):
    print(f"  {i}. {action.tactic.name}")

# Get statistics
stats = mcts.get_statistics()
print(f"\nSearch time: {stats['average_time']:.2f}s")
print(f"Proof found: {len(best_sequence) > 0}")
```

### Example 2: Quick MCTS Search

```python
# Fast mode for simple theorems
mcts = LeanProofMCTS(
    exploration_constant=0.5,   # Exploit known tactics
    simulations=100,            # Quick search
    rollout_depth=3             # Shallow rollouts
)

best_sequence, root = mcts.search(context)
```

### Example 3: Thorough MCTS Search

```python
# Thorough mode for complex theorems
mcts = LeanProofMCTS(
    exploration_constant=2.0,   # Explore more
    simulations=10000,          # Many simulations
    rollout_depth=15,           # Deep rollouts
    dirichlet_alpha=0.5,        # Wider exploration
    dirichlet_epsilon=0.5       # More noise
)

best_sequence, root = mcts.search(context)
```

---

## Custom Rollout Policies

### Example 4: Random Rollout

```python
from leanaide_mcts import MCTS

# Use random rollout policy
mcts = MCTS(
    exploration_constant=1.414,
    rollout_depth=10
)

# Random rollout is default
def random_action_generator(context):
    """Generate random actions"""
    return [
        TacticAction(
            tactic=Tactic(name=t),
            context=context
        )
        for t in ["intro", "apply", "simp", "rw"]
    ]

best_sequence, root = mcts.search(context, random_action_generator, evaluator)
```

### Example 5: Heuristic-Guided Rollout

```python
def heuristic_action_generator(context):
    """Generate actions with heuristic ordering"""
    tactics = []

    # Prefer safe tactics first
    for tactic_name in ["intro", "simp", "assumption"]:
        if is_applicable(context, tactic_name):
            tactics.append(TacticAction(
                tactic=Tactic(name=tactic_name, is_safe=True),
                context=context,
                estimated_value=0.8
            ))

    # Then other tactics
    for tactic_name in ["apply", "rw", "cases"]:
        if is_applicable(context, tactic_name):
            tactics.append(TacticAction(
                tactic=Tactic(name=tactic_name),
                context=context,
                estimated_value=0.5
            ))

    return tactics

# Use heuristic rollout
mcts = MCTS(rollout_depth=10)
best_sequence, root = mcts.search(
    context,
    heuristic_action_generator,
    evaluator
)
```

### Example 6: Success-Rate-Based Rollout

```python
def success_rate_action_generator(context):
    """Generate actions weighted by historical success rates"""
    tactics = get_applicable_tactics(context)

    # Sort by success rate
    tactics.sort(key=lambda t: t.success_rate, reverse=True)

    actions = []
    for tactic in tactics:
        actions.append(TacticAction(
            tactic=tactic,
            context=context,
            estimated_value=tactic.success_rate,
            prior_probability=tactic.success_rate
        ))

    return actions

# Use success-rate-guided rollout
mcts = MCTS(rollout_depth=10)
best_sequence, root = mcts.search(
    context,
    success_rate_action_generator,
    evaluator
)
```

### Example 7: Custom Evaluation Function

```python
def custom_evaluator(context, lean_client=None):
    """Custom state evaluation"""

    # Terminal state
    if not context.goal:
        return 1.0

    # Base value
    value = 0.5

    # Depth penalty (prefer shorter proofs)
    value -= 0.01 * context.depth

    # Bonus for hypotheses (more resources)
    value += 0.02 * len(context.hypotheses)

    # Bonus for available lemmas
    value += 0.01 * len(context.available_lemmas)

    # Penalty for very complex goals
    if len(context.goal) > 200:
        value -= 0.1

    # Bonus for specific patterns
    if "Nat" in context.goal and "add" in context.goal:
        value += 0.05  # Arithmetic is easier

    return max(0.0, min(1.0, value))

# Use custom evaluator
mcts = LeanProofMCTS()
# Override internal evaluation
best_sequence, root = mcts.search(context)
```

---

## Parallel MCTS

### Example 8: Basic Parallel MCTS

```python
import concurrent.futures
from leanaide_mcts import LeanProofMCTS

def run_single_mcts(context, seed):
    """Run single MCTS with random seed"""
    import random
    random.seed(seed)

    mcts = LeanProofMCTS(simulations=500)
    return mcts.search(context)

def parallel_mcts_search(context, num_workers=4):
    """Run MCTS in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            future = executor.submit(run_single_mcts, context, i)
            futures.append(future)

        results = [f.result() for f in futures]

    # Select best result
    best_result = max(results, key=lambda r: len(r[0]))

    return best_result

# Use parallel MCTS
best_sequence, root = parallel_mcts_search(context, num_workers=4)
```

### Example 9: Parallel Root Parallelization

```python
class ParallelMCTS:
    """Parallel MCTS with root parallelization"""

    def __init__(self, num_workers=4, simulations_per_worker=250):
        self.num_workers = num_workers
        self.simulations_per_worker = simulations_per_worker

    def search(self, context):
        """Search with parallel workers"""
        with concurrent.futures.ProcessPoolExecutor(self.num_workers) as executor:
            futures = []
            for _ in range(self.num_workers):
                future = executor.submit(
                    self._worker_search,
                    context,
                    self.simulations_per_worker
                )
                futures.append(future)

            results = [f.result() for f in futures]

        # Merge trees (simplified)
        # In practice, would need more sophisticated merging
        best = max(results, key=lambda r: r[0].visit_count)
        return best

    @staticmethod
    def _worker_search(context, simulations):
        """Worker process"""
        mcts = LeanProofMCTS(simulations=simulations)
        return mcts.search(context)

# Use parallel MCTS
parallel_mcts = ParallelMCTS(num_workers=4, simulations_per_worker=250)
best_sequence, root = parallel_mcts.search(context)
```

---

## MCTS in Workflow

### Example 10: MCTS in Decomposition Workflow

```python
from leanaide_mcts import LeanProofMCTS

def decompose_theorem_with_mcts(theorem):
    """Decompose theorem using MCTS"""

    # Stage 1: MCTS for high-level strategy
    context = ProofContext(
        goal=theorem,
        hypotheses=[],
        available_lemmas=get_available_lemmas()
    )

    mcts = LeanProofMCTS(
        simulations=500,
        exploration_constant=1.5
    )

    strategy, root = mcts.search(context)

    # Extract sub-goals from strategy
    subgoals = extract_subgoals_from_strategy(strategy)

    return subgoals

def solve_subgoal_with_mcts(subgoal):
    """Solve sub-goal using MCTS"""

    context = ProofContext(
        goal=subgoal.statement,
        hypotheses=subgoal.hypotheses,
        available_lemmas=subgoal.lemmas
    )

    mcts = LeanProofMCTS(
        simulations=1000,
        exploration_constant=1.414
    )

    proof, root = mcts.search(context)
    return proof

def synthesize_with_mcts(subproofs):
    """Synthesize final proof from sub-proofs"""

    context = ProofContext(
        goal="synthesize",
        available_lemmas=[p for p in subproofs]
    )

    mcts = LeanProofMCTS(simulations=500)
    final_proof, root = mcts.search(context)

    return final_proof

# Full workflow
theorem = "∀ n m : Nat, n + m = m + n"
subgoals = decompose_theorem_with_mcts(theorem)

subproofs = []
for subgoal in subgoals:
    proof = solve_subgoal_with_mcts(subgoal)
    subproofs.append(proof)

final_proof = synthesize_with_mcts(subproofs)
```

### Example 11: MCTS for Proof Refinement

```python
def refine_proof_with_mcts(initial_proof):
    """Refine proof using MCTS"""

    # Analyze initial proof
    issues = analyze_proof(initial_proof)

    if not issues:
        return initial_proof  # Already good

    # Use MCTS to find better alternative for problematic section
    context = ProofContext(
        goal=issues[0].goal,
        hypotheses=issues[0].hypotheses,
        depth=issues[0].depth
    )

    mcts = LeanProofMCTS(
        simulations=2000,
        exploration_constant=1.0,  # More focused
        rollout_depth=5            # Local refinement
    )

    refined_section, root = mcts.search(context)

    # Replace problematic section
    refined_proof = replace_section(initial_proof, issues[0], refined_section)

    return refined_proof
```

---

## Hybrid MCTS + Evolution

### Example 12: MCTS + Evolution Hybrid

```python
class HybridMCTSEvolution:
    """Hybrid MCTS and evolutionary search"""

    def __init__(self):
        self.mcts = LeanProofMCTS(simulations=500)
        self.population_size = 10
        self.generations = 5

    def search(self, context):
        """Hybrid search strategy"""

        # Phase 1: MCTS for initial proof
        print("Phase 1: MCTS search")
        mcts_proof, mcts_tree = self.mcts.search(context)

        # Phase 2: Initialize population with MCTS result
        population = [mcts_proof]
        for _ in range(self.population_size - 1):
            # Variations of MCTS proof
            variant = vary_proof(mcts_proof)
            population.append(variant)

        # Phase 3: Evolutionary refinement
        print("Phase 2: Evolutionary refinement")
        for gen in range(self.generations):
            population = self.evolve_population(population, context)

        # Phase 4: MCTS polish on best evolved proof
        print("Phase 3: MCTS polish")
        best_evolved = max(population, key=evaluate_proof_quality)

        # Create context from evolved proof
        polished_context = ProofContext(
            goal=best_evolved[0].context.goal,
            hypotheses=best_evolved[0].context.hypotheses
        )

        final_proof, _ = self.mcts.search(polished_context)

        return final_proof

    def evolve_population(self, population, context):
        """Evolve population using genetic operators"""
        # Evaluate fitness
        fitness_scores = [evaluate_proof_quality(p, context) for p in population]

        # Selection
        selected = select_top_population(population, fitness_scores, top_k=5)

        # Crossover
        offspring = []
        for i in range(len(population) - len(selected)):
            parent1, parent2 = random.sample(selected, 2)
            child = crossover_proofs(parent1, parent2)
            offspring.append(child)

        # Mutation
        for individual in offspring:
            if random.random() < 0.3:  # Mutation rate
                mutate_proof(individual)

        return selected + offspring

# Use hybrid search
hybrid = HybridMCTSEvolution()
final_proof = hybrid.search(context)
```

### Example 13: Alternating MCTS and Evolution

```python
def alternating_search(context, rounds=3):
    """Alternate between MCTS and evolution"""

    current_best = None

    for round_num in range(rounds):
        print(f"Round {round_num + 1}")

        # MCTS phase
        mcts = LeanProofMCTS(simulations=1000)
        mcts_proof, _ = mcts.search(context)

        # Evolution phase
        if current_best:
            population = [current_best, mcts_proof]
        else:
            population = [mcts_proof]

        # Evolve
        for _ in range(5):
            population = evolve_population(population, context)

        current_best = max(population, key=evaluate_proof_quality)

        print(f"  Best quality: {evaluate_proof_quality(current_best, context):.3f}")

    return current_best
```

---

## Domain-Specific MCTS

### Example 14: Arithmetic Proofs

```python
class ArithmeticMCTS(LeanProofMCTS):
    """MCTS specialized for arithmetic theorems"""

    def __init__(self):
        super().__init__(
            exploration_constant=1.414,
            simulations=1500,
            rollout_depth=8
        )

        # Arithmetic-specific tactics
        self.arithmetic_tactics = [
            Tactic(name="linarith", category="arithmetic", is_safe=True),
            Tactic(name="ring", category="algebra", is_safe=True),
            Tactic(name="norm_num", category="arithmetic", is_safe=True),
            Tactic(name="aesop", category="automated"),
        ]

    def _generate_actions(self, context):
        """Generate arithmetic-specific actions"""
        actions = super()._generate_actions(context)

        # Prioritize arithmetic tactics
        arithmetic_actions = [
            a for a in actions
            if a.tactic.category in ["arithmetic", "algebra"]
        ]

        # Give them higher priors
        for action in arithmetic_actions:
            action.prior_probability *= 2.0

        return actions

    def _evaluate_state(self, context, lean_client=None):
        """Arithmetic-specific evaluation"""
        if not context.goal:
            return 1.0

        value = 0.5

        # Bonus for arithmetic in goal
        if any(op in context.goal for op in ["+", "-", "*", "/", "="]):
            value += 0.1

        # Bonus for Num type
        if "Nat" in context.goal or "Int" in context.goal:
            value += 0.1

        return min(1.0, value)

# Use for arithmetic theorems
arithmetic_mcts = ArithmeticMCTS()
proof, root = arithmetic_mcts.search(arithmetic_context)
```

### Example 15: Inductive Proofs

```python
class InductiveMCTS(LeanProofMCTS):
    """MCTS specialized for inductive proofs"""

    def __init__(self):
        super().__init__(
            exploration_constant=1.6,  # More exploration
            simulations=2000,
            rollout_depth=12
        )

    def _generate_actions(self, context):
        """Generate inductive-specific actions"""
        actions = super()._generate_actions(context)

        # Prioritize induction tactics
        for action in actions:
            if action.tactic.name in ["induction", "cases", "constructor"]:
                action.prior_probability *= 3.0
                action.estimated_value += 0.2

        return actions

    def _evaluate_state(self, context, lean_client=None):
        """Inductive-specific evaluation"""
        if not context.goal:
            return 1.0

        value = 0.5

        # Bonus for inductive types
        if any(typ in context.goal for typ in ["Nat", "List", "Tree"]):
            value += 0.15

        # Bonus for induction in context
        if any("induction" in h for h in context.hypotheses):
            value += 0.1

        return min(1.0, value)

# Use for inductive theorems
inductive_mcts = InductiveMCTS()
proof, root = inductive_mcts.search(inductive_context)
```

---

## Performance Tuning

### Example 16: Transposition Table

```python
from leanaide_mcts import LeanProofMCTS

class TranspositionMCTS(LeanProofMCTS):
    """MCTS with transposition table"""

    def __init__(self):
        super().__init__()
        self.transposition_table = {}
        self.table_hits = 0
        self.table_misses = 0

    def search(self, initial_context, lean_client=None):
        """Search with transposition table"""
        root = MCTSNode(state=initial_context)

        for _ in range(self.simulations):
            node = self.mcts.select(root)

            # Check transposition table
            state_hash = hash(node.state)
            if state_hash in self.transposition_table:
                self.table_hits += 1
                value = self.transposition_table[state_hash]
            else:
                self.table_misses += 1
                # Expand, simulate, backpropagate
                actions = self._generate_actions(node.state)
                child = self.mcts.expand(node, actions)
                value = self.mcts.simulate(
                    child,
                    self._generate_actions,
                    lambda ctx: self._evaluate_state(ctx, lean_client)
                )

                # Cache result
                if len(self.transposition_table) < 10000:
                    self.transposition_table[state_hash] = value

            self.mcts.backpropagate(node, value)

        best_sequence = self._extract_best_sequence(root)
        return best_sequence, root

    def get_cache_stats(self):
        """Get cache statistics"""
        total = self.table_hits + self.table_misses
        hit_rate = self.table_hits / total if total > 0 else 0
        return {
            "hits": self.table_hits,
            "misses": self.table_misses,
            "hit_rate": hit_rate,
            "table_size": len(self.transposition_table)
        }

# Use transposition MCTS
trans_mcts = TranspositionMCTS()
proof, root = trans_mcts.search(context)

stats = trans_mcts.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### Example 17: Adaptive Simulations

```python
class AdaptiveMCTS(LeanProofMCTS):
    """MCTS with adaptive simulation count"""

    def __init__(self, min_sim=100, max_sim=5000):
        super().__init__(simulations=min_sim)
        self.min_sim = min_sim
        self.max_sim = max_sim
        self.convergence_threshold = 0.01

    def search(self, initial_context, lean_client=None):
        """Search with adaptive stopping"""
        root = MCTSNode(state=initial_context)

        prev_value = 0.0
        iteration = 0

        while iteration < self.max_sim:
            # Run simulation
            node = self.mcts.select(root)
            actions = self._generate_actions(node.state)
            child = self.mcts.expand(node, actions)
            value = self.mcts.simulate(
                child,
                self._generate_actions,
                lambda ctx: self._evaluate_state(ctx, lean_client)
            )
            self.mcts.backpropagate(child, value)

            iteration += 1

            # Check convergence
            current_value = root.average_value
            if iteration > self.min_sim:
                if abs(current_value - prev_value) < self.convergence_threshold:
                    print(f"Converged after {iteration} iterations")
                    break

            prev_value = current_value

        print(f"Final: {iteration} iterations, value={current_value:.3f}")
        best_sequence = self._extract_best_sequence(root)
        return best_sequence, root

# Use adaptive MCTS
adaptive_mcts = AdaptiveMCTS(min_sim=500, max_sim=5000)
proof, root = adaptive_mcts.search(context)
```

---

## Advanced Examples

### Example 18: MCTS with Learned Policy

```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """Neural network policy for MCTS"""

    def __init__(self, input_size, hidden_size, num_tactics):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.hidden = nnTransformerEncoder(...)
        self.policy_head = nn.Linear(hidden_size, num_tactics)

    def forward(self, state_features):
        """Generate action probabilities"""
        encoded = self.encoder(state_features)
        hidden = self.hidden(encoded)
        logits = self.policy_head(hidden)
        return torch.softmax(logits, dim=-1)

def encode_proof_context(context):
    """Encode proof context as tensor"""
    # Simplified encoding
    features = []

    # Goal length
    features.append(len(context.goal) / 1000.0)

    # Number of hypotheses
    features.append(len(context.hypotheses) / 10.0)

    # Number of lemmas
    features.append(len(context.available_lemmas) / 50.0)

    # Depth
    features.append(context.depth / 20.0)

    return torch.tensor(features)

def search_with_learned_policy(context, policy_net):
    """MCTS with learned policy"""
    mcts = LeanProofMCTS(simulations=1000)
    root = MCTSNode(state=context)

    for _ in range(mcts.simulations):
        node = mcts.mcts.select(root)

        # Get policy probabilities
        state_features = encode_proof_context(node.state)
        policy_probs = policy_net(state_features)

        # Map to actions
        actions = mcts._generate_actions(node.state)
        action_probs = {
            actions[i].action_id: policy_probs[i].item()
            for i in range(len(actions))
        }

        # Use policy in expansion
        child = mcts.mcts.expand(node, actions, action_probs)
        value = mcts.mcts.simulate(
            child,
            mcts._generate_actions,
            lambda ctx: mcts._evaluate_state(ctx)
        )
        mcts.mcts.backpropagate(child, value)

    best_sequence = mcts._extract_best_sequence(root)
    return best_sequence, root
```

### Example 19: Interactive MCTS

```python
class InteractiveMCTS(LeanProofMCTS):
    """Interactive MCTS with human guidance"""

    def search(self, initial_context, lean_client=None, guidance_callback=None):
        """Search with human guidance"""
        root = MCTSNode(state=initial_context)

        for iteration in range(self.simulations):
            node = self.mcts.select(root)
            actions = self._generate_actions(node.state)

            # Ask for human guidance every 100 iterations
            if guidance_callback and iteration % 100 == 0:
                print(f"\nIteration {iteration}")
                print(f"Current goal: {node.state.goal}")
                print(f"Available actions: {[a.tactic.name for a in actions[:5]]}")

                # Get guidance
                guidance = guidance_callback(node.state, actions)

                if guidance:
                    # Prioritize guided actions
                    for action in actions:
                        if action.tactic.name in guidance:
                            action.prior_probability *= 5.0

            child = self.mcts.expand(node, actions)
            value = self.mcts.simulate(
                child,
                self._generate_actions,
                lambda ctx: self._evaluate_state(ctx, lean_client)
            )
            self.mcts.backpropagate(child, value)

        best_sequence = self._extract_best_sequence(root)
        return best_sequence, root

def human_guidance(state, actions):
    """Callback for human guidance"""
    print("\nEnter preferred tactics (comma-separated), or press Enter to skip:")
    user_input = input("> ").strip()

    if user_input:
        return [t.strip() for t in user_input.split(",")]
    return None

# Use interactive MCTS
interactive_mcts = InteractiveMCTS(simulations=500)
proof, root = interactive_mcts.search(context, guidance_callback=human_guidance)
```

### Example 20: Multi-Objective MCTS

```python
class MultiObjectiveMCTS(LeanProofMCTS):
    """MCTS optimizing multiple objectives"""

    def __init__(self):
        super().__init__()
        self.objectives = {
            "correctness": 0.7,    # Most important
            "length": 0.2,         # Prefer shorter proofs
            "elegance": 0.1        # Prefer elegant proofs
        }

    def _evaluate_state(self, context, lean_client=None):
        """Multi-objective evaluation"""
        if not context.goal:
            return {obj: 1.0 for obj in self.objectives}

        scores = {
            "correctness": self._evaluate_correctness(context),
            "length": self._evaluate_length(context),
            "elegance": self._evaluate_elegance(context)
        }

        return scores

    def _evaluate_correctness(self, context):
        """Evaluate proof correctness"""
        value = 0.5

        # Check if goal is simplifying
        if len(context.goal) < 100:
            value += 0.2

        # Check for contradictions
        if "False" in context.goal:
            value -= 0.3

        return max(0.0, min(1.0, value))

    def _evaluate_length(self, context):
        """Evaluate proof length (prefer shorter)"""
        return max(0.0, 1.0 - context.depth / 50.0)

    def _evaluate_elegance(self, context):
        """Evaluate proof elegance"""
        value = 0.5

        # Prefer using high-level tactics
        if any(h in context.hypotheses for h in context.available_lemmas):
            value += 0.1

        # Penalize very low-level tactics
        low_level = ["rw", "simp", "apply"]
        if all(t in low_level for t in context.available_lemmas[:5]):
            value -= 0.1

        return max(0.0, min(1.0, value))

    def get_combined_score(self, scores):
        """Combine multiple objectives"""
        return sum(
            self.objectives[obj] * scores[obj]
            for obj in self.objectives
        )

# Use multi-objective MCTS
multi_mcts = MultiObjectiveMCTS()
proof, root = multi_mcts.search(context)

# Get scores
final_scores = multi_mcts._evaluate_state(proof[-1].context)
combined = multi_mcts.get_combined_score(final_scores)
print(f"Combined score: {combined:.3f}")
print(f"Breakdown: {final_scores}")
```

---

## Conclusion

These examples demonstrate the versatility of MCTS for Lean 4 theorem proving:

- **Basic examples** show simple usage
- **Custom rollouts** demonstrate domain knowledge integration
- **Parallel MCTS** shows performance optimization
- **Workflow integration** shows practical usage
- **Hybrid approaches** combine multiple strategies
- **Domain-specific** shows specialization
- **Performance tuning** shows optimization
- **Advanced examples** show cutting-edge techniques

Adapt these examples to your specific use case and requirements!

---

*Last Updated: 2025-12-30*
*Version: 1.0.0*
