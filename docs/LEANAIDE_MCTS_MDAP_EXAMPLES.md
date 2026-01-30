# LeanAide MCTS-MDAP Examples

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Custom Agent Configurations](#custom-agent-configurations)
3. [Custom Voting Strategies](#custom-voting-strategies)
4. [Workflow Integration](#workflow-integration)
5. [Performance Tuning](#performance-tuning)
6. [Comparison with Pure Approaches](#comparison-with-pure-approaches)
7. [Advanced Examples](#advanced-examples)

---

## Basic Usage

### Example 1: Simple MDAP-MCTS Search

```python
from leanaide_mcts import (
    MCTSConfig,
    ProofState,
    search_proof_with_mcts
)
from mdap_engine import MDAPConfig, MDAPOrchestrator
from workflow_structures import Team, ModelConfig
import os

# Create agent team
team = Team(
    team_id="basic_team",
    name="Basic Theorem Proving Team",
    members=[
        ModelConfig(
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=750
        )
    ]
)

# Configure MCTS
mcts_config = MCTSConfig(
    max_iterations=1000,
    time_budget=60.0,
    c_param=1.414,
    rollout_depth=50
)

# Configure MDAP
mdap_config = MDAPConfig(
    k_min=2,
    k_max=5,
    max_votes_per_step=20
)

# Create initial proof state
initial_state = ProofState(
    goals=["forall (a b : Nat), a + b = b + a"],
    context=[],
    depth=0
)

# Search with MDAP-MCTS
from test_leanaide_mcts_mdap import search_with_mdap_mcts

result = search_with_mdap_mcts(
    initial_state,
    mcts_config,
    mdap_config,
    team
)

# Check results
if result.success:
    print(f"Proof found in {result.search_iterations} iterations!")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Win rate: {result.win_rate:.2%}")
else:
    print("Proof not found")
    print(f"Best partial result: {result.win_rate:.2%} complete")
```

### Example 2: MAKER-Enhanced Simulation

```python
from leanaide_mcts import MCTSConfig, ProofState
from mdap_maker_complete import MAKEREngine
from workflow_structures import Team, ModelConfig
import os

# Create team
team = Team(
    team_id="maker_team",
    name="MAKER Team",
    members=[
        ModelConfig(
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
            temperature=0.0
        ),
        ModelConfig(
            model_id="claude-3-opus",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            api_base="https://api.anthropic.com/v1",
            temperature=0.0
        )
    ]
)

# Create MAKER engine
maker_engine = MAKEREngine(
    team=team,
    k_ahead=3,
    max_token_length=750,
    max_steps=100,
    enable_first_to_ahead=True,
    enable_red_flagging=True
)

# Configure MCTS
mcts_config = MCTSConfig(
    max_iterations=500,
    time_budget=120.0,
    rollout_depth=30
)

# Create state
state = ProofState(
    goals=["forall (a b c : Nat), a + (b + c) = (a + b) + c"],
    context=[],
    depth=0
)

# Search with MAKER-enhanced simulation
from test_leanaide_mcts_mdap import search_with_maker_mcts

result = search_with_maker_mcts(
    state,
    mcts_config,
    maker_engine
)

print(f"Success: {result.success}")
print(f"Iterations: {result.search_iterations}")
print(f"Proof quality: {result.confidence:.2%}")
```

### Example 3: Pure MCTS (Baseline)

```python
from leanaide_mcts import MCTSConfig, ProofState, search_proof_with_mcts

# Configure pure MCTS
config = MCTSConfig(
    max_iterations=2000,
    time_budget=60.0,
    c_param=1.414,
    rollout_depth=100,
    parallel_simulations=4
)

# Create state
state = ProofState(
    goals=["forall (n : Nat), n + 0 = n"],
    context=[],
    depth=0
)

# Search
result = search_proof_with_mcts(state, config)

print(f"Proof found: {result.success}")
print(f"Iterations: {result.search_iterations}")
print(f"Tree depth: {result.tree_depth}")
```

---

## Custom Agent Configurations

### Example 4: Specialized Agents

```python
from workflow_structures import ModelConfig, Team

# Create specialized agents with different roles
team = Team(
    team_id="specialized_team",
    name="Specialized Theorem Proving Team",
    members=[
        # Induction specialist
        ModelConfig(
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=750,
            problem_type_specialization=[
                "induction",
                "natural_number_proof"
            ],
            performance_metrics={
                "success_rate": 0.85,
                "avg_proof_length": 15
            }
        ),

        # Algebra specialist
        ModelConfig(
            model_id="claude-3-opus",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            api_base="https://api.anthropic.com/v1",
            temperature=0.0,
            max_tokens=750,
            problem_type_specialization=[
                "algebraic_manipulation",
                "rewriting",
                "simplification"
            ],
            performance_metrics={
                "success_rate": 0.82,
                "avg_proof_length": 12
            }
        ),

        # Tactic selection specialist
        ModelConfig(
            model_id="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            api_base="https://generativelanguage.googleapis.com/v1",
            temperature=0.1,
            max_tokens=750,
            problem_type_specialization=[
                "tactic_selection",
                "lemma_application"
            ],
            performance_metrics={
                "success_rate": 0.78,
                "avg_proof_length": 18
            }
        )
    ]
)

# Use specialized team
mdap_config = MDAPConfig(
    k_min=2,
    k_max=5
)

# AgentSelector will automatically weight agents by specialization
from mdap_engine import MDAPOrchestrator
orchestrator = MDAPOrchestrator(team, mdap_config)

# When solving an induction problem, the induction specialist
# will be selected more often
```

### Example 5: Temperature Variations

```python
from workflow_structures import ModelConfig, Team

# Team with different temperatures for diversity
team = Team(
    team_id="diverse_team",
    name="Diverse Temperature Team",
    members=[
        # Low temperature - consistent, focused
        ModelConfig(
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=750
        ),

        # Medium temperature - balanced
        ModelConfig(
            model_id="claude-3-opus",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            api_base="https://api.anthropic.com/v1",
            temperature=0.3,
            max_tokens=750
        ),

        # Higher temperature - creative, exploratory
        ModelConfig(
            model_id="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            api_base="https://generativelanguage.googleapis.com/v1",
            temperature=0.5,
            max_tokens=750
        )
    ]
)

# Voting will aggregate diverse perspectives
```

### Example 6: Performance-Based Weighting

```python
from mdap_engine import AgentSelector

# Custom agent selector with performance weighting
class PerformanceAgentSelector(AgentSelector):
    """Agent selector that weights by performance metrics."""

    def select(self, step, additional_weight=None):
        members = self.team.members
        if not members:
            raise ValueError("Team has no members")

        weights = []
        for member in members:
            weight = 1.0

            # Task type specialization
            if step.task_type and member.problem_type_specialization:
                if step.task_type in member.problem_type_specialization:
                    weight += 2.0

            # Performance metrics
            if member.performance_metrics:
                success_rate = member.performance_metrics.get("success_rate", 0.5)
                weight *= success_rate

                # Average proof length (prefer shorter proofs)
                avg_length = member.performance_metrics.get("avg_proof_length", 20)
                weight *= (20.0 / max(5, avg_length))

            # Additional custom weight
            if additional_weight:
                weight *= additional_weight.get(member.model_id, 1.0)

            weights.append(weight)

        # Select by weighted probability
        total = sum(weights)
        if total <= 0:
            return members[0]

        pick = self.rng.uniform(0, total)
        cumulative = 0.0
        for member, weight in zip(members, weights):
            cumulative += weight
            if cumulative >= pick:
                return member

        return members[-1]

# Use custom selector
selector = PerformanceAgentSelector(team, rng=random.Random(42))
best_agent = selector.select(MDAPStep(step_id="test", prompt="test", task_type="induction"))
```

---

## Custom Voting Strategies

### Example 7: Weighted Voting

```python
from typing import Dict, List, Tuple

def weighted_voting(
    agent_responses: List[Tuple[str, any]],
    agent_weights: Dict[str, float]
) -> Tuple[any, float]:
    """
    Perform weighted voting on agent responses.

    Args:
        agent_responses: List of (agent_id, response) tuples
        agent_weights: Weight for each agent_id

    Returns:
        (winner, confidence) tuple
    """
    # Count weighted votes
    votes = {}
    for agent_id, response in agent_responses:
        weight = agent_weights.get(agent_id, 1.0)
        response_key = str(response)
        votes[response_key] = votes.get(response_key, 0.0) + weight

    # Find winner
    winner_key = max(votes, key=votes.get)
    winner = eval(winner_key)  # Convert back to object

    # Calculate confidence
    total_weight = sum(votes.values())
    confidence = votes[winner_key] / total_weight if total_weight > 0 else 0.0

    return winner, confidence

# Usage
agent_responses = [
    ("gpt-4", {"action": "intros"}),
    ("gpt-4", {"action": "intros"}),
    ("claude-3-opus", {"action": "apply"}),
    ("gemini-pro", {"action": "intros"})
]

agent_weights = {
    "gpt-4": 1.5,  # Higher weight for GPT-4
    "claude-3-opus": 1.2,
    "gemini-pro": 1.0
}

winner, confidence = weighted_voting(agent_responses, agent_weights)
print(f"Winner: {winner}, Confidence: {confidence:.2%}")
```

### Example 8: Bayesian Voting

```python
def bayesian_voting(
    prior_beliefs: Dict[str, float],
    new_evidence: List[str],
    confidence: float = 0.9
) -> Tuple[str, float]:
    """
    Perform Bayesian voting by combining priors with evidence.

    Args:
        prior_beliefs: Prior belief for each action
        new_evidence: New votes from agents
        confidence: Confidence in new evidence

    Returns:
        (winner, posterior_belief) tuple
    """
    posterior = prior_beliefs.copy()

    # Update with new evidence
    for action in new_evidence:
        if action not in posterior:
            posterior[action] = 0.5  # Neutral prior

        # Bayesian update
        posterior[action] = (
            confidence * 1.0 +  # Evidence
            (1 - confidence) * posterior[action]  # Prior
        )

    # Find winner
    winner = max(posterior, key=posterior.get)
    posterior_belief = posterior[winner]

    return winner, posterior_belief

# Usage
priors = {
    "intros": 0.6,
    "apply": 0.3,
    "rw": 0.1
}

new_votes = ["intros", "intros", "apply", "intros"]

winner, belief = bayesian_voting(priors, new_votes, confidence=0.85)
print(f"Winner: {winner}, Belief: {belief:.2%}")
```

### Example 9: Adaptive k-Value

```python
def adaptive_k_value(
    step_number: int,
    total_steps: int,
    difficulty: str = "medium"
) -> int:
    """
    Compute adaptive k-value based on search progress.

    Args:
        step_number: Current step number
        total_steps: Total expected steps
        difficulty: Problem difficulty (easy, medium, hard)

    Returns:
        Recommended k-value
    """
    progress = step_number / max(1, total_steps)

    # Base k by difficulty
    if difficulty == "easy":
        base_k = 2
    elif difficulty == "medium":
        base_k = 3
    else:  # hard
        base_k = 5

    # Early phase: Lower k for exploration
    if progress < 0.2:
        return max(1, base_k - 1)

    # Middle phase: Standard k
    elif progress < 0.8:
        return base_k

    # Late phase: Higher k for consensus
    else:
        return base_k + 1

# Usage in search
for step in range(100):
    k = adaptive_k_value(step, 100, difficulty="medium")
    # Use k for voting
```

---

## Workflow Integration

### Example 10: Decomposition Workflow

```python
from mdap_maker_complete import RecursiveMAKERSolver

def solve_with_decomposition(
    theorem_statement: str,
    context: Dict[str, any],
    team: Team,
    mcts_config: MCTSConfig
) -> Tuple[any, any]:
    """
    Solve theorem by decomposing into subproblems.

    Workflow:
    1. Decompose main theorem
    2. Solve each subproblem with MCTS
    3. Compose final solution
    """
    # Step 1: Create MAKER solver for decomposition
    maker_config = {
        "max_depth": 5,
        "k_ahead": 3,
        "num_candidates": 5
    }
    maker_solver = RecursiveMAKERSolver(team, **maker_config)

    # Step 2: Decompose
    decomposition = maker_solver._decompose(
        theorem_statement,
        context,
        depth=0
    )

    if decomposition.is_atomic:
        # Atomic problem - solve directly with MCTS
        state = ProofState(goals=[theorem_statement], context=context.get("hypotheses", []))
        result = search_proof_with_mcts(state, mcts_config)
        return result.best_proof, result

    # Step 3: Solve subproblems
    subproblem1 = decomposition.subtask1
    subproblem2 = decomposition.subtask2

    print(f"Solving subproblem 1: {subproblem1['task'][:50]}...")
    proof1, result1 = solve_with_decomposition(
        subproblem1["task"],
        {**context, **subproblem1.get("context", {})},
        team,
        mcts_config
    )

    print(f"Solving subproblem 2: {subproblem2['task'][:50]}...")
    proof2, result2 = solve_with_decomposition(
        subproblem2["task"],
        {**context, **subproblem2.get("context", {})},
        team,
        mcts_config
    )

    # Step 4: Compose solutions
    composition_function = decomposition.composition_function
    final_proof = compose_proofs(
        theorem_statement,
        proof1,
        proof2,
        composition_function
    )

    return final_proof, (result1, result2)

def compose_proofs(theorem: str, proof1: any, proof2: any, composition: str) -> any:
    """Compose two subproofs using composition function."""
    # Implementation depends on proof representation
    return {
        "theorem": theorem,
        "proof1": proof1,
        "proof2": proof2,
        "composition": composition
    }

# Usage
theorem = "forall (a b c : Nat), a + (b + c) = (a + b) + c"
context = {"domain": "natural_numbers", "operation": "addition"}

final_proof, results = solve_with_decomposition(theorem, context, team, mcts_config)
```

### Example 11: Stage 3A Integration

```python
def stage_3a_mdap_mcts_integration(
    problem: Dict[str, any],
    team: Team,
    mcts_config: MCTSConfig,
    mdap_config: MDAPConfig
) -> any:
    """
    Stage 3A: Use MDAP-MCTS for tactic selection.

    Integrates with decomposition workflow for refined tactic selection.
    """
    # Extract problem information
    goal = problem["goal"]
    context = problem.get("context", [])

    # Create proof state
    state = ProofState(goals=[goal], context=context, depth=0)

    # Use MDAP-MCTS for search
    result = search_with_mdap_mcts(
        state,
        mcts_config,
        mdap_config,
        team,
        progress_callback=lambda i: print(f"Iteration {i}")
    )

    # Return result with metadata
    return {
        "success": result.success,
        "proof": result.best_proof,
        "iterations": result.search_iterations,
        "time": result.time_elapsed,
        "confidence": result.confidence
    }

# Usage in workflow
problem = {
    "goal": "forall (n : Nat), n + 0 = n",
    "context": [],
    "difficulty": "easy"
}

result = stage_3a_mdap_mcts_integration(problem, team, mcts_config, mdap_config)
```

### Example 12: Stage 3B Refinement

```python
def stage_3b_refinement(
    initial_proof: any,
    goal: str,
    team: Team,
    mcts_config: MCTSConfig
) -> any:
    """
    Stage 3B: Refine proof using MCTS.

    Takes an initial proof and refines it using MCTS search.
    """
    # Evaluate initial proof
    initial_quality = evaluate_proof_quality(initial_proof)
    print(f"Initial proof quality: {initial_quality:.2%}")

    if initial_quality > 0.9:
        print("Proof already high quality, skipping refinement")
        return initial_proof

    # Identify weak points
    weak_steps = identify_weak_steps(initial_proof)
    print(f"Found {len(weak_steps)} weak steps to refine")

    # Refine each weak step
    refined_proof = initial_proof
    for step_idx in weak_steps:
        print(f"Refining step {step_idx}...")

        # Create subproblem for refinement
        subproblem = create_refinement_subproblem(
            refined_proof,
            step_idx,
            goal
        )

        # Solve with MCTS
        state = ProofState(
            goals=[subproblem["goal"]],
            context=subproblem["context"],
            depth=step_idx
        )

        result = search_proof_with_mcts(state, mcts_config)

        if result.success:
            refined_proof = apply_refinement(
                refined_proof,
                step_idx,
                result.best_proof
            )

    # Evaluate final proof
    final_quality = evaluate_proof_quality(refined_proof)
    print(f"Final proof quality: {final_quality:.2%}")

    return refined_proof

def evaluate_proof_quality(proof: any) -> float:
    """Evaluate proof quality (0-1)."""
    # Implementation depends on proof representation
    # Factors: length, clarity, elegance, correctness
    return 0.85  # Placeholder

def identify_weak_steps(proof: any) -> List[int]:
    """Identify weak steps in proof."""
    # Implementation depends on proof representation
    return [1, 3, 5]  # Placeholder

def create_refinement_subproblem(proof: any, step_idx: int, goal: str) -> Dict:
    """Create subproblem for refining a specific step."""
    return {
        "goal": goal,
        "context": [],
        "step_to_refine": step_idx
    }

def apply_refinement(proof: any, step_idx: int, refinement: any) -> any:
    """Apply refinement to proof."""
    # Implementation depends on proof representation
    return proof  # Placeholder
```

---

## Performance Tuning

### Example 13: Parallel Search

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_mdap_mcts_search(
    initial_state: ProofState,
    mcts_config: MCTSConfig,
    mdap_config: MDAPConfig,
    team: Team,
    num_parallel: int = 4
) -> MCTSResult:
    """
    Run multiple MDAP-MCTS searches in parallel and return best result.

    Args:
        initial_state: Starting proof state
        mcts_config: MCTS configuration
        mdap_config: MDAP configuration
        team: Agent team
        num_parallel: Number of parallel searches

    Returns:
        Best MCTSResult from all searches
    """
    results = []

    def single_search(seed: int) -> MCTSResult:
        """Run single search with specific seed."""
        config = MCTSConfig(
            **mcts_config.__dict__,
            seed=seed
        )
        return search_with_mdap_mcts(
            initial_state,
            config,
            mdap_config,
            team
        )

    # Run searches in parallel
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [
            executor.submit(single_search, i * 1000)
            for i in range(num_parallel)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Search completed: success={result.success}, win_rate={result.win_rate:.2%}")

    # Return best result
    best_result = max(results, key=lambda r: r.win_rate)
    print(f"\nBest result: success={best_result.success}, win_rate={best_result.win_rate:.2%}")

    return best_result

# Usage
result = parallel_mdap_mcts_search(
    state,
    mcts_config,
    mdap_config,
    team,
    num_parallel=4
)
```

### Example 14: Progressive Deepening

```python
def progressive_deepening_search(
    initial_state: ProofState,
    base_config: MCTSConfig,
    mdap_config: MDAPConfig,
    team: Team,
    phases: List[Tuple[int, float]] = [(500, 10.0), (1000, 30.0), (2000, 60.0)]
) -> MCTSResult:
    """
    Perform progressive deepening search.

    Starts with quick search, progressively deepening if needed.

    Args:
        initial_state: Starting proof state
        base_config: Base MCTS configuration
        mdap_config: MDAP configuration
        team: Agent team
        phases: List of (iterations, time_limit) for each phase

    Returns:
        MCTSResult from first successful phase or best overall
    """
    for phase_num, (iterations, time_limit) in enumerate(phases):
        print(f"\nPhase {phase_num + 1}: {iterations} iterations, {time_limit}s time limit")

        config = MCTSConfig(
            **base_config.__dict__,
            max_iterations=iterations,
            time_budget=time_limit
        )

        result = search_with_mdap_mcts(
            initial_state,
            config,
            mdap_config,
            team
        )

        print(f"Result: success={result.success}, win_rate={result.win_rate:.2%}")

        # If proof found, return early
        if result.success:
            print("Proof found! Returning early.")
            return result

        # If win rate is high enough, continue to next phase
        if result.win_rate > 0.7:
            print(f"High win rate ({result.win_rate:.2%}), continuing to next phase...")
            continue

        print("Low win rate, stopping early")
        return result

    # Return best result from final phase
    return result

# Usage
result = progressive_deepening_search(
    state,
    mcts_config,
    mdap_config,
    team,
    phases=[(500, 10.0), (1000, 30.0), (2000, 60.0)]
)
```

### Example 15: Adaptive Configuration

```python
def adaptive_search(
    initial_state: ProofState,
    initial_config: MCTSConfig,
    mdap_config: MDAPConfig,
    team: Team,
    adjustment_threshold: float = 0.3
) -> MCTSResult:
    """
    Adaptively adjust configuration based on search progress.

    Monitors convergence and adjusts parameters if needed.
    """
    config = initial_config
    result = search_with_mdap_mcts(
        initial_state,
        config,
        mdap_config,
        team
    )

    # If win rate is very low, increase exploration
    if result.win_rate < adjustment_threshold:
        print(f"Low win rate ({result.win_rate:.2%}), increasing exploration")

        config = MCTSConfig(
            **config.__dict__,
            c_param=config.c_param * 1.5,  # More exploration
            max_iterations=int(config.max_iterations * 1.5)
        )

        result = search_with_mdap_mcts(
            initial_state,
            config,
            mdap_config,
            team
        )

    # If convergence is slow, adjust voting
    if result.win_rate > 0.5 and result.win_rate < 0.7:
        print(f"Slow convergence, adjusting voting parameters")

        mdap_config = MDAPConfig(
            **mdap_config.__dict__,
            k_min=mdap_config.k_min - 1,  # Lower k for faster convergence
            k_max=mdap_config.k_max - 1
        )

        result = search_with_mdap_mcts(
            initial_state,
            config,
            mdap_config,
            team
        )

    return result
```

---

## Comparison with Pure Approaches

### Example 16: MCTS vs MDAP-MCTS Comparison

```python
def compare_approaches(
    initial_state: ProofState,
    team: Team,
    num_runs: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compare pure MCTS vs MDAP-MCTS performance.

    Runs multiple searches and aggregates statistics.

    Args:
        initial_state: Starting proof state
        team: Agent team
        num_runs: Number of runs for each approach

    Returns:
        Comparison statistics
    """
    results = {
        "pure_mcts": {"successes": 0, "avg_time": 0.0, "avg_iterations": 0.0},
        "mdap_mcts": {"successes": 0, "avg_time": 0.0, "avg_iterations": 0.0}
    }

    # Pure MCTS runs
    print("Running pure MCTS...")
    for i in range(num_runs):
        config = MCTSConfig(
            max_iterations=1000,
            time_budget=30.0,
            seed=i * 100
        )

        result = search_proof_with_mcts(initial_state, config)

        results["pure_mcts"]["successes"] += int(result.success)
        results["pure_mcts"]["avg_time"] += result.time_elapsed
        results["pure_mcts"]["avg_iterations"] += result.search_iterations

    # MDAP-MCTS runs
    print("Running MDAP-MCTS...")
    mdap_config = MDAPConfig(k_min=2, k_max=5)
    for i in range(num_runs):
        config = MCTSConfig(
            max_iterations=1000,
            time_budget=30.0,
            seed=i * 100
        )

        result = search_with_mdap_mcts(
            initial_state,
            config,
            mdap_config,
            team
        )

        results["mdap_mcts"]["successes"] += int(result.success)
        results["mdap_mcts"]["avg_time"] += result.time_elapsed
        results["mdap_mcts"]["avg_iterations"] += result.search_iterations

    # Compute averages
    for approach in results:
        results[approach]["avg_time"] /= num_runs
        results[approach]["avg_iterations"] /= num_runs
        results[approach]["success_rate"] = results[approach]["successes"] / num_runs

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    for approach, stats in results.items():
        print(f"\n{approach.upper()}:")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Avg Time: {stats['avg_time']:.2f}s")
        print(f"  Avg Iterations: {stats['avg_iterations']:.0f}")

    return results

# Usage
comparison = compare_approaches(state, team, num_runs=10)
```

### Example 17: MAKER vs MCTS vs MDAP-MCTS

```python
def compare_all_approaches(
    theorem: str,
    context: List[str],
    team: Team
) -> None:
    """
    Compare all three approaches: MAKER, MCTS, MDAP-MCTS.
    """
    state = ProofState(goals=[theorem], context=context)

    print("Testing MAKER (decomposition)...")
    maker_engine = MAKEREngine(team, k_ahead=3, max_steps=100)
    # Run MAKER (simplified)
    start = time.time()
    # maker_result = maker_engine.solve(...)
    maker_time = time.time() - start
    # print(f"MAKER: success=..., time={maker_time:.2f}s")

    print("\nTesting pure MCTS...")
    config = MCTSConfig(max_iterations=1000, time_budget=60.0)
    start = time.time()
    mcts_result = search_proof_with_mcts(state, config)
    mcts_time = time.time() - start
    print(f"MCTS: success={mcts_result.success}, time={mcts_time:.2f}s")

    print("\nTesting MDAP-MCTS...")
    mdap_config = MDAPConfig(k_min=2, k_max=5)
    start = time.time()
    mdap_result = search_with_mdap_mcts(state, config, mdap_config, team)
    mdap_time = time.time() - start
    print(f"MDAP-MCTS: success={mdap_result.success}, time={mdap_time:.2f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pure MCTS:      success={mcts_result.success}, time={mcts_time:.2f}s")
    print(f"MDAP-MCTS:      success={mdap_result.success}, time={mdap_time:.2f}s")
    # print(f"MAKER:          success=..., time={maker_time:.2f}s")

# Usage
compare_all_approaches(
    "forall (a b : Nat), a + b = b + a",
    [],
    team
)
```

---

## Advanced Examples

### Example 18: Custom Reward Function

```python
def multi_objective_search(
    initial_state: ProofState,
    team: Team,
    objectives: Dict[str, float] = None
) -> MCTSResult:
    """
    MCTS search with custom multi-objective reward function.

    Objectives:
    - success: Complete proof (weight 0.5)
    - elegance: Proof elegance (weight 0.3)
    - brevity: Short proof length (weight 0.2)
    """
    if objectives is None:
        objectives = {
            "success": 0.5,
            "elegance": 0.3,
            "brevity": 0.2
        }

    # Custom reward function
    def custom_reward(node: 'MCTSNode') -> float:
        reward = 0.0

        # Success reward
        if node.state.is_complete:
            reward += objectives["success"] * 1.0

        # Elegance reward (based on tactic diversity)
        if hasattr(node, "tactics_sequence"):
            diversity = len(set(t.name for t in node.tactics_sequence))
            max_diversity = len(node.tactics_sequence) if node.tactics_sequence else 1
            elegance = diversity / max(max_diversity, 1)
            reward += objectives["elegance"] * elegance

        # Brevity reward (shorter proofs better)
        if hasattr(node, "depth"):
            brevity = 1.0 / (1.0 + node.depth * 0.1)
            reward += objectives["brevity"] * brevity

        return reward

    # Configure MCTS with custom reward
    config = MCTSConfig(
        max_iterations=1000,
        time_budget=60.0,
        reward_function=custom_reward  # Custom reward
    )

    # Run search
    result = search_proof_with_mcts(initial_state, config)

    return result
```

### Example 19: Interactive Proof Assistant

```python
def interactive_proof_assistant(
    theorem: str,
    team: Team
) -> None:
    """
    Interactive proof assistant using MDAP-MCTS.

    Allows user to guide search by suggesting tactics or approving agent suggestions.
    """
    state = ProofState(goals=[theorem], context=[], depth=0)

    print(f"Proving: {theorem}")
    print("=" * 60)

    while not state.is_complete:
        print(f"\nCurrent goal: {state.goals[0] if state.goals else 'None'}")
        print(f"Context: {', '.join(state.context[:3])}...")

        # Get agent suggestions with voting
        print("\nAgent suggestions:")
        suggestions = get_agent_suggestions(state, team)
        for i, (tactic, votes, confidence) in enumerate(suggestions, 1):
            print(f"  {i}. {tactic} (votes: {votes}, confidence: {confidence:.0%})")

        # Get user input
        print("\nOptions:")
        print("  1-3: Select tactic")
        print("  h: Get hint")
        print("  a: Auto-complete with MCTS")
        print("  q: Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            print("Quitting...")
            break
        elif choice == 'a':
            print("\nRunning MCTS auto-complete...")
            config = MCTSConfig(max_iterations=500, time_budget=30.0)
            mdap_config = MDAPConfig(k_min=2, k_max=5)
            result = search_with_mdap_mcts(state, config, mdap_config, team)
            if result.success:
                print("Proof found!")
                print(result.best_proof)
            break
        elif choice == 'h':
            print(f"\nHint: Try {suggestions[0][0]}")
        elif choice in ['1', '2', '3']:
            idx = int(choice) - 1
            tactic = suggestions[idx][0]
            print(f"\nApplying: {tactic}")
            # Apply tactic and update state
            # state = apply_tactic(state, tactic)
        else:
            print("Invalid choice")

def get_agent_suggestions(state: ProofState, team: Team) -> List[Tuple[str, int, float]]:
    """Get agent suggestions with voting."""
    # Mock implementation
    return [
        ("intros", 5, 0.83),
        ("apply Nat.add_comm", 3, 0.50),
        ("induction n", 2, 0.33)
    ]

# Usage
interactive_proof_assistant("forall (a b : Nat), a + b = b + a", team)
```

### Example 20: Batch Theorem Proving

```python
def batch_prove_theorems(
    theorems: List[str],
    team: Team,
    output_file: str = "proofs.json"
) -> Dict[str, any]:
    """
    Prove multiple theorems using MDAP-MCTS.

    Args:
        theorems: List of theorem statements
        team: Agent team
        output_file: Output file for results

    Returns:
        Dictionary mapping theorems to results
    """
    results = {}

    mcts_config = MCTSConfig(
        max_iterations=1000,
        time_budget=60.0
    )
    mdap_config = MDAPConfig(
        k_min=2,
        k_max=5
    )

    for i, theorem in enumerate(theorems, 1):
        print(f"\n[{i}/{len(theorems)}] Proving: {theorem[:60]}...")

        state = ProofState(goals=[theorem], context=[], depth=0)

        try:
            result = search_with_mdap_mcts(
                state,
                mcts_config,
                mdap_config,
                team
            )

            results[theorem] = {
                "success": result.success,
                "iterations": result.search_iterations,
                "time": result.time_elapsed,
                "win_rate": result.win_rate,
                "proof": str(result.best_proof) if result.best_proof else None
            }

            print(f"  Result: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"  Time: {result.time_elapsed:.2f}s")
            print(f"  Win rate: {result.win_rate:.2%}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[theorem] = {"success": False, "error": str(e)}

    # Save results
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successes = sum(1 for r in results.values() if r.get("success", False))
    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE: {successes}/{len(theorems)} theorems proved")
    print(f"Results saved to: {output_file}")

    return results

# Usage
theorems = [
    "forall (n : Nat), n + 0 = n",
    "forall (a b : Nat), a + b = b + a",
    "forall (a b c : Nat), a + (b + c) = (a + b) + c",
    "forall (n : Nat), 0 + n = n",
    "forall (a b : Nat), a * b = b * a"
]

batch_results = batch_prove_theorems(theorems, team)
```

---

## Quick Reference Summary

### Common Patterns

| Task | Approach | Code |
|------|----------|------|
| Simple proof | Pure MCTS | `search_proof_with_mcts(state, config)` |
| Medium proof | MCTS + MDAP | `search_with_mdap_mcts(state, mcts_config, mdap_config, team)` |
| Complex proof | MAKER + MCTS | `search_with_maker_mcts(state, mcts_config, maker_engine)` |
| Decomposition | Recursive MAKER | `RecursiveMAKERSolver(team).solve(task)` |
| Batch proving | Loop with MCTS | `batch_prove_theorems(theorems, team)` |

### Configuration Templates

```python
# Quick search (10s)
config = MCTSConfig(max_iterations=500, time_budget=10.0)

# Standard search (60s)
config = MCTSConfig(max_iterations=1000, time_budget=60.0)

# Deep search (5min)
config = MCTSConfig(max_iterations=5000, time_budget=300.0)

# MDAP voting
mdap_config = MDAPConfig(k_min=2, k_max=5)

# MAKER engine
maker_engine = MAKEREngine(team, k_ahead=3, max_steps=100)
```

For complete API reference, see `LEANAIDE_MCTS_MDAP_API.md`.
For comprehensive guide, see `LEANAIDE_MCTS_MDAP_GUIDE.md`.
