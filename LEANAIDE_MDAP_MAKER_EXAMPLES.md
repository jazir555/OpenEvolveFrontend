# LeanAide MDAP/MAKER Integration - Examples

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide MDAP/MAKER Integration

---

## Table of Contents

1. [Basic MDAP Usage](#1-basic-mdap-usage)
2. [Basic MAKER Usage](#2-basic-maker-usage)
3. [Hybrid MDAP+MAKER](#3-hybrid-mdapmaker)
4. [Custom Agent Configuration](#4-custom-agent-configuration)
5. [Custom Voting Strategies](#5-custom-voting-strategies)
6. [Workflow Integration Examples](#6-workflow-integration-examples)
7. [Advanced Examples](#7-advanced-examples)
8. [Real-World Theorem Examples](#8-real-world-theorem-examples)

---

## 1. Basic MDAP Usage

### Example 1.1: Simple MDAP Proof Generation

```python
import asyncio
from mdap_engine import (
    MDAPOrchestrator, MDAPConfig, MDAPTask, MDAPStep
)
from workflow_structures import ModelConfig

async def basic_mdap_example():
    """Generate proof using basic MDAP"""

    # Configuration
    config = MDAPConfig(
        k_min=3,
        k_max=5,
        timeout_seconds=60
    )

    model_config = ModelConfig(
        provider="openai",
        model="gpt-4o",
        api_key="your-api-key"
    )

    # Create orchestrator
    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=model_config
    )

    # Define proof task
    step = MDAPStep(
        step_id="proof_step",
        prompt="Prove: ∀ n : Nat, n + 0 = n",
        task_type="theorem_proving",
        temperature_override=0.1
    )

    task = MDAPTask(
        task_id="add_zero",
        description="Prove addition with zero",
        steps=[step]
    )

    # Execute
    result = await orchestrator.run_task_async(task)

    # Check result
    if result.success:
        print("SUCCESS!")
        print(f"Proof:\n{result.proof}")
        print(f"\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
    else:
        print("FAILED")
        for error in result.errors:
            print(f"  Error: {error}")

asyncio.run(basic_mdap_example())
```

**Expected Output:**
```
SUCCESS!
Proof:
theorem add_zero (n : Nat) : n + 0 = n := by
  cases n
  . rfl
  . simp [Nat.add_succ]

Metrics:
  total_steps: 1
  total_votes: 5
  red_flags: 0
  duration_seconds: 12.3
```

### Example 1.2: Multi-Step MDAP Task

```python
async def multi_step_mdap_example():
    """Generate proof with multiple steps"""

    config = MDAPConfig(k_min=3, k_max=5)
    model_config = ModelConfig(
        provider="openai",
        model="gpt-4o",
        api_key="your-api-key"
    )

    orchestrator = MDAPOrchestrator(config, model_config)

    # Step 1: Analyze theorem
    analyze_step = MDAPStep(
        step_id="analyze",
        prompt="Analyze the theorem structure: ∀ a b : Nat, a + b = b + a",
        task_type="analysis"
    )

    # Step 2: Generate proof
    prove_step = MDAPStep(
        step_id="prove",
        prompt="Generate the proof for commutativity of addition",
        task_type="theorem_proving"
    )

    # Step 3: Verify proof
    verify_step = MDAPStep(
        step_id="verify",
        prompt="Verify the generated proof",
        task_type="verification"
    )

    task = MDAPTask(
        task_id="add_comm",
        description="Prove addition commutativity",
        steps=[analyze_step, prove_step, verify_step]
    )

    result = await orchestrator.run_task_async(task)

    # Print step results
    for step_id, step_result in result.step_results.items():
        print(f"\nStep {step_id}:")
        print(f"  Status: {step_result.status}")
        print(f"  Votes: {step_result.vote_result.votes}")
        print(f"  Retries: {step_result.retries}")

asyncio.run(multi_step_mdap_example())
```

### Example 1.3: MDAP with Caching

```python
async def mdap_with_caching_example():
    """Use MDAP with response caching"""

    config = MDAPConfig(
        k_min=3,
        k_max=5,
        cache_ttl_seconds=3600,  # Cache for 1 hour
        cache_max_size=1000
    )

    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key="your-api-key"
        ),
        cache_enabled=True
    )

    # First call (will hit API)
    result1 = await orchestrator.run_task_async(task)

    # Second call with same task (will use cache)
    result2 = await orchestrator.run_task_async(task)

    # Second call should be faster
    print(f"First call: {result1.metrics['duration_seconds']:.2f}s")
    print(f"Second call: {result2.metrics['duration_seconds']:.2f}s")
```

---

## 2. Basic MAKER Usage

### Example 2.1: Sequential MAKER

```python
from maker_workflow_integration import (
    MAKERWorkflowConfig, MAKERMode, build_maker_config_from_workflow
)
from workflow_structures import WorkflowState, SubProblem, Team

async def sequential_maker_example():
    """Generate proof using sequential MAKER"""

    # Configure workflow
    state = WorkflowState()
    state.maker_enabled = True
    state.maker_config = {
        "maker_mode": "sequential",
        "maker_k_ahead": 3
    }

    # Create sub-problem
    sub_problem = SubProblem(
        id="mul_one",
        title="Prove multiplication by one",
        description="Prove: ∀ n : Nat, n * 1 = n",
        estimated_effort=5
    )

    # Build MAKER config
    config = build_maker_config_from_workflow(state, sub_problem)

    print(f"MAKER Mode: {config.mode}")
    print(f"K-Ahead: {config.k_ahead}")

    # Execute MAKER (simplified - actual execution requires full workflow)
    # solution = await generate_solution_with_maker_v2(...)

asyncio.run(sequential_maker_example())
```

### Example 2.2: Recursive MAKER

```python
async def recursive_maker_example():
    """Generate proof using recursive MAKER"""

    state = WorkflowState()
    state.maker_enabled = True
    state.maker_config = {
        "maker_mode": "recursive",
        "maker_k_ahead": 3,
        "maker_max_depth": 5
    }

    # Complex theorem that benefits from decomposition
    sub_problem = SubProblem(
        id="complex_theorem",
        title="Complex theorem",
        description="Prove: ∀ a b c : Nat, (a + b) * c = a * c + b * c",
        estimated_effort=20  # Large effort → recursive decomposition
    )

    config = build_maker_config_from_workflow(state, sub_problem)

    print(f"Using recursive MAKER with max depth {config.max_depth}")

asyncio.run(recursive_maker_example())
```

### Example 2.3: Parallel MAKER

```python
async def parallel_maker_example():
    """Generate proof using parallel MAKER"""

    state = WorkflowState()
    state.maker_enabled = True
    state.maker_config = {
        "maker_mode": "parallel",
        "maker_k_ahead": 3
    }

    sub_problem = SubProblem(
        id="parallel_theorem",
        title="Theorem with multiple approaches",
        description="Prove: ∀ n : Nat, 2 * n = n + n",
        estimated_effort=10
    )

    config = build_maker_config_from_workflow(state, sub_problem)

    print(f"Using parallel MAKER mode")

asyncio.run(parallel_maker_example())
```

---

## 3. Hybrid MDAP+MAKER

### Example 3.1: ROMA-MDAP-MAKER Integration

```python
from roma_mdap_maker_engine import (
    ROMAMDAPMakerEngine, ROMAMDAPMakerConfig
)

async def romamdap_maker_example():
    """Generate proof using ROMA-MDAP-MAKER integration"""

    config = ROMAMDAPMakerConfig(
        # ROMA settings
        roma_max_depth_solving=2,

        # MDAP settings
        mdap_enabled=True,
        mdap_k_ahead=3,

        # Integration
        apply_maker_to_roma_atomic=True,
        enable_hierarchical_voting=True,

        # Provider
        provider="openai",
        model="gpt-4o-mini",
        api_key="your-api-key"
    )

    engine = ROMAMDAPMakerEngine(config)

    result = await engine.solve_with_romamdap(
        theorem="theorem distributive_add_mul (a b c : Nat) : (a + b) * c = a * c + b * c",
        context="""
import Mathlib.Data.Nat.Basic
-- We have addition and multiplication properties available
"""
    )

    if result["success"]:
        print("SUCCESS!")
        print(f"Proof:\n{result['proof']}")
        print(f"\nDecomposition depth: {result['roma_depth']}")
        print(f"MDAP samples: {result['mdap_samples']}")
    else:
        print(f"FAILED: {result['error']}")

asyncio.run(romamdap_maker_example())
```

### Example 3.2: Hierarchical Voting

```python
async def hierarchical_voting_example():
    """Use hierarchical voting across decomposition levels"""

    config = ROMAMDAPMakerConfig(
        mdap_k_ahead=3,
        enable_hierarchical_voting=True,
        enable_adaptive_k=True
    )

    engine = ROMAMDAPMakerEngine(config)

    # Decompose theorem
    decomposition = engine.decompose_theorem(
        "theorem mul_assoc (a b c : Nat) : (a * b) * c = a * (b * c)"
    )

    print("Decomposition structure:")
    print(decomposition)

    # Solve with hierarchical voting
    result = await engine.solve_with_romamdap(
        theorem="theorem mul_assoc (a b c : Nat) : (a * b) * c = a * (b * c)"
    )

    print(f"\nHierarchical voting results:")
    for level, votes in result["hierarchical_votes"].items():
        print(f"  Level {level}: {votes}")

asyncio.run(hierarchical_voting_example())
```

### Example 3.3: Adaptive K-Selection

```python
async def adaptive_k_example():
    """Use adaptive k-selection based on difficulty"""

    config = ROMAMDAPMakerConfig(
        mdap_k_ahead=3,
        enable_adaptive_k=True
    )

    engine = ROMAMDAPMakerEngine(config)

    # Easy theorem → low k
    easy_result = await engine.solve_with_romamdap(
        theorem="theorem refl_nat (n : Nat) : n = n"
    )
    print(f"Easy theorem used k={easy_result['adaptive_k']}")

    # Hard theorem → high k
    hard_result = await engine.solve_with_romamdap(
        theorem="theorem complex_theorem (a b c d : Nat) : ... (complex expression)"
    )
    print(f"Hard theorem used k={hard_result['adaptive_k']}")

asyncio.run(adaptive_k_example())
```

---

## 4. Custom Agent Configuration

### Example 4.1: Domain-Specific Agents

```python
from mdap_engine import AgentConfig

async def custom_agents_example():
    """Use custom agents for specific domains"""

    # Algebraic theorem → use algebraic agents
    algebraic_agents = [
        AgentConfig(
            name="ring_specialist",
            system_prompt="You specialize in ring tactics (ring, linarith)",
            preferred_tactics=["ring", "linarith", "omega"]
        ),
        AgentConfig(
            name="simp_specialist",
            system_prompt="You specialize in simplification",
            preferred_tactics=["simp", "simp_all"]
        )
    ]

    # Inductive theorem → use inductive agents
    inductive_agents = [
        AgentConfig(
            name="induction_expert",
            system_prompt="You specialize in induction proofs",
            preferred_tactics=["induction'", "cases'"]
        )
    ]

    print(f"Configured {len(algebraic_agents)} algebraic agents")
    print(f"Configured {len(inductive_agents)} inductive agents")

asyncio.run(custom_agents_example())
```

### Example 4.2: Temperature Tuning per Agent

```python
async def temperature_tuning_example():
    """Use different temperatures for different agents"""

    agents = {
        "deterministic": AgentConfig(
            name="deterministic_prover",
            temperature=0.0,  # Very deterministic
            max_tokens=500
        ),
        "exploratory": AgentConfig(
            name="exploratory_prover",
            temperature=0.3,  # Some exploration
            max_tokens=750
        ),
        "creative": AgentConfig(
            name="creative_prover",
            temperature=0.7,  # More creative approaches
            max_tokens=1000
        )
    }

    for agent_type, agent in agents.items():
        print(f"{agent_type}: T={agent.temperature}, max_tokens={agent.max_tokens}")

asyncio.run(temperature_tuning_example())
```

### Example 4.3: Agent Selection Based on Theorem

```python
def select_agents_for_theorem(theorem: str) -> list:
    """Select appropriate agents based on theorem characteristics"""

    agents = []

    # Check for inductive structure
    if "∀" in theorem and ("Nat" in theorem or "List" in theorem):
        agents.append("inductive")

    # Check for algebraic operations
    if any(op in theorem for op in ["+", "-", "*", "/", "^"]):
        agents.append("algebraic")

    # Check for logical structure
    if any(logic in theorem for logic in ["→", "¬", "∧", "∨"]):
        agents.append("indirect")

    # Always include constructive
    agents.append("constructive")

    return agents

# Example usage
theorem1 = "∀ n : Nat, n + 0 = n"
theorem2 = "theorem not_not {p : Prop} : ¬¬p → p"
theorem3 = "∀ a b c : Nat, a * (b + c) = a * b + a * c"

print(f"Theorem 1 agents: {select_agents_for_theorem(theorem1)}")
print(f"Theorem 2 agents: {select_agents_for_theorem(theorem2)}")
print(f"Theorem 3 agents: {select_agents_for_theorem(theorem3)}")
```

---

## 5. Custom Voting Strategies

### Example 5.1: First-K-Ahead Voting

```python
from roma_mdap_maker_engine import AdaptiveKSelector

async def first_k_ahead_example():
    """Use first-K-ahead voting"""

    selector = AdaptiveKSelector(
        k_min=2,
        k_max=8,
        confidence_threshold=0.8
    )

    # Simulate voting
    votes = {
        "proof_a": 2,
        "proof_b": 1,
        "proof_c": 0
    }

    # Check if should stop early
    k_ahead = 3
    should_stop = selector.should_stop_early(votes, k_ahead)

    print(f"Votes so far: {votes}")
    print(f"Should stop (k_ahead={k_ahead}): {should_stop}")

    # proof_a has 2 votes (< k_ahead=3), so continue voting

asyncio.run(first_k_ahead_example())
```

### Example 5.2: Confidence-Weighted Voting

```python
def confidence_weighted_voting(candidates):
    """Vote with confidence weighting"""

    weighted_votes = {}

    for candidate in candidates:
        key = candidate["proof"]
        confidence = candidate.get("confidence", 0.5)

        if key not in weighted_votes:
            weighted_votes[key] = 0.0

        weighted_votes[key] += confidence

    # Select winner
    winner = max(weighted_votes, key=weighted_votes.get)

    return winner, weighted_votes

# Example usage
candidates = [
    {"proof": "proof_a", "confidence": 0.9},
    {"proof": "proof_a", "confidence": 0.85},  # Also proof_a
    {"proof": "proof_b", "confidence": 0.7},
    {"proof": "proof_c", "confidence": 0.6}
]

winner, votes = confidence_weighted_voting(candidates)

print(f"Winner: {winner}")
print(f"Weighted votes: {votes}")
# proof_a: 1.75
# proof_b: 0.7
# proof_c: 0.6
```

### Example 5.3: Quality-Weighted Voting

```python
def quality_score(proof):
    """Calculate quality score for proof"""
    score = 0

    if proof.get("verified"):
        score += 100

    # Prefer shorter proofs
    num_tactics = len(proof.get("tactics", []))
    if num_tactics > 0:
        score += 100 / num_tactics

    # Elegance bonus
    score += proof.get("elegance", 0) * 10

    return score

def quality_weighted_voting(candidates):
    """Vote with quality weighting"""

    weighted_votes = {}

    for candidate in candidates:
        key = candidate["proof"]
        quality = quality_score(candidate)
        votes = candidate.get("votes", 1)

        if key not in weighted_votes:
            weighted_votes[key] = 0.0

        weighted_votes[key] += quality * votes

    winner = max(weighted_votes, key=weighted_votes.get)

    return winner, weighted_votes

# Example usage
candidates = [
    {
        "proof": "proof_a",
        "votes": 3,
        "verified": True,
        "tactics": ["simp", "rfl"],
        "elegance": 0.8
    },
    {
        "proof": "proof_b",
        "votes": 2,
        "verified": True,
        "tactics": ["rw", "apply", "exact", "simp"],
        "elegance": 0.6
    }
]

winner, votes = quality_weighted_voting(candidates)

print(f"Winner: {winner}")
print(f"Quality-weighted votes: {votes}")
```

---

## 6. Workflow Integration Examples

### Example 6.1: Decomposition Workflow with MDAP

```python
from workflow_structures import WorkflowState, SubProblem, Team

async def decomposition_workflow_example():
    """Integrate MDAP with decomposition workflow"""

    # Create workflow state
    state = WorkflowState()
    state.mdap_config = {
        "k_min": 3,
        "k_max": 5,
        "timeout_seconds": 60
    }

    # Create sub-problems from decomposition
    sub_problems = [
        SubProblem(
            id="lemma_1",
            title="Base case for induction",
            description="Prove: 0 + n = n",
            estimated_effort=3
        ),
        SubProblem(
            id="lemma_2",
            title="Inductive step",
            description="Prove: (succ m) + n = succ (m + n)",
            estimated_effort=5
        ),
        SubProblem(
            id="main",
            title="Main theorem",
            description="Prove: ∀ m n : Nat, m + n = n + m",
            dependencies=["lemma_1", "lemma_2"],
            estimated_effort=8
        )
    ]

    # Solve each sub-problem with MDAP
    for sub_problem in sub_problems:
        print(f"\nSolving {sub_problem.id}: {sub_problem.title}")

        # Would use MDAP orchestrator here
        # result = await solve_subproblem_with_mdap(sub_problem, state)

        print(f"  Effort: {sub_problem.estimated_effort}")
        print(f"  Dependencies: {sub_problem.dependencies}")

asyncio.run(decomposition_workflow_example())
```

### Example 6.2: Stage 3A MDAP Integration

```python
async def stage_3a_mdap_example():
    """Stage 3A: Initial sub-problem solving with MDAP"""

    state = WorkflowState()
    state.mdap_config = {
        "k_min": 3,
        "k_max": 5
    }

    # Sub-problem from decomposition
    sub_problem = SubProblem(
        id="sub_1",
        title="Prove addition is associative",
        description="∀ a b c : Nat, (a + b) + c = a + (b + c)",
        estimated_effort=12
    )

    # Solve with MDAP
    config = MDAPConfig(
        k_min=state.mdap_config["k_min"],
        k_max=state.mdap_config["k_max"]
    )

    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key="your-api-key"
        )
    )

    # Create task
    step = MDAPStep(
        step_id="prove_associativity",
        prompt=f"Prove: {sub_problem.description}"
    )

    task = MDAPTask(
        task_id=sub_problem.id,
        description=sub_problem.title,
        steps=[step]
    )

    result = await orchestrator.run_task_async(task)

    if result.success:
        print(f"✓ Solved {sub_problem.id}")
        print(f"Proof: {result.proof}")
    else:
        print(f"✗ Failed {sub_problem.id}")

asyncio.run(stage_3a_mdap_example())
```

### Example 6.3: Stage 3B MDAP Refinement

```python
async def stage_3b_refinement_example():
    """Stage 3B: Refine solutions with additional MDAP rounds"""

    # Initial solution from Stage 3A
    initial_proof = """
theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  sorry
"""

    print("Initial proof (incomplete):")
    print(initial_proof)

    # Refine with MDAP
    config = MDAPConfig(
        k_min=5,  # More agents for refinement
        k_max=8
    )

    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key="your-api-key"
        )
    )

    # Refinement task
    step = MDAPStep(
        step_id="refine_proof",
        prompt=f"""
Complete and refine this proof:
{initial_proof}

Replace 'sorry' with actual tactics.
""",
        task_type="proof_refinement"
    )

    task = MDAPTask(
        task_id="refine_add_assoc",
        description="Refine associativity proof",
        steps=[step],
        max_retries=3
    )

    result = await orchestrator.run_task_async(task)

    if result.success:
        print("\nRefined proof:")
        print(result.proof)
    else:
        print("\nRefinement failed")

asyncio.run(stage_3b_refinement_example())
```

---

## 7. Advanced Examples

### Example 7.1: Custom Red-Flagging Rules

```python
from mdap_engine import RedFlagRules, RedFlagger

# Custom red-flagging rules
custom_rules = RedFlagRules(
    max_tokens=500,  # Strict limit
    min_confidence=0.7,  # High confidence required
    blocked_patterns=[
        "sorry",  # No placeholder proofs
        "admit",  # No admitted proofs
        "TODO"  # No incomplete proofs
    ],
    require_schema_match=True
)

flagger = RedFlagger(custom_rules)

# Test responses
test_responses = [
    "theorem test : True := by sorry",  # Should be flagged
    "theorem test : True := by trivial",  # Should pass
    '{"proof": "...", "confidence": 0.5}',  # Should be flagged (low confidence)
]

for response in test_responses:
    is_flagged, reasons = flagger.is_flagged(response, {}, None)
    print(f"Response: {response[:50]}...")
    print(f"  Flagged: {is_flagged}")
    print(f"  Reasons: {reasons}\n")
```

### Example 7.2: Batch Processing with MDAP

```python
async def batch_mdap_example():
    """Process multiple theorems with MDAP"""

    theorems = [
        "theorem add_zero (n : Nat) : n + 0 = n",
        "theorem zero_add (n : Nat) : 0 + n = n",
        "theorem add_comm (a b : Nat) : a + b = b + a",
        "theorem mul_one (n : Nat) : n * 1 = n",
        "theorem one_mul (n : Nat) : 1 * n = n"
    ]

    config = MDAPConfig(k_min=3, k_max=5)
    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key="your-api-key"
        )
    )

    results = {}

    for theorem in theorems:
        task = MDAPTask(
            task_id=f"task_{len(results)}",
            description=theorem,
            steps=[
                MDAPStep(
                    step_id="prove",
                    prompt=f"Prove: {theorem}"
                )
            ]
        )

        result = await orchestrator.run_task_async(task)
        results[theorem] = result.success

        status = "✓" if result.success else "✗"
        print(f"{status} {theorem}")

    # Summary
    success_rate = sum(results.values()) / len(results)
    print(f"\nSuccess rate: {success_rate:.1%}")

asyncio.run(batch_mdap_example())
```

### Example 7.3: Progress Tracking

```python
class ProgressTracker:
    """Track progress of MDAP execution"""

    def __init__(self):
        self.step_times = {}
        self.step_statuses = {}
        self.total_votes = 0
        self.total_red_flags = 0

    def track_step(self, step_id: str, result: MDAPStepResult):
        """Track step result"""
        self.step_statuses[step_id] = result.status
        self.total_votes += result.vote_result.attempts
        self.total_red_flags += result.vote_result.red_flags

    def report(self):
        """Generate progress report"""
        print("\n=== Progress Report ===")
        print(f"Steps completed: {len(self.step_statuses)}")
        print(f"Total votes: {self.total_votes}")
        print(f"Total red flags: {self.total_red_flags}")

        for step_id, status in self.step_statuses.items():
            print(f"  {step_id}: {status}")

# Usage
async def tracked_mdap_example():
    tracker = ProgressTracker()

    # Run MDAP task
    result = await orchestrator.run_task_async(task)

    # Track results
    for step_id, step_result in result.step_results.items():
        tracker.track_step(step_id, step_result)

    # Generate report
    tracker.report()

asyncio.run(tracked_mdap_example())
```

---

## 8. Real-World Theorem Examples

### Example 8.1: Mathlib Theorem

```python
async def mathlib_example():
    """Prove Mathlib-style theorem"""

    context = """
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

-- Available theorems:
-- Nat.add_zero, Nat.add_succ, Nat.zero_add
-- Nat.mul_one, Nat.one_mul
"""

    theorem_statement = """
theorem add_three (n : Nat) : n + 3 = n + 1 + 2 := by
  sorry
"""

    config = MDAPConfig(k_min=3, k_max=5)
    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key="your-api-key"
        )
    )

    step = MDAPStep(
        step_id="prove_add_three",
        prompt=f"""
Complete this Mathlib theorem:

Context:
{context}

Theorem:
{theorem_statement}

Replace 'sorry' with actual proof.
"""
    )

    task = MDAPTask(
        task_id="add_three",
        description="Prove addition with 3",
        steps=[step]
    )

    result = await orchestrator.run_task_async(task)

    if result.success:
        print("Proof:")
        print(result.proof)
    else:
        print("Failed to generate proof")

asyncio.run(mathlib_example())
```

### Example 8.2: Inductive Theorem

```python
async def induction_example():
    """Prove theorem requiring induction"""

    theorem = """
theorem sum_n (n : Nat) : 2 * n = n + n := by
  sorry
"""

    # Use inductive agents
    config = MDAPConfig(k_min=3, k_max=5)
    orchestrator = MDAPOrchestrator(
        config=config,
        model_config=ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key="your-api-key"
        )
    )

    step = MDAPStep(
        step_id="induction_proof",
        prompt=f"""
Prove this theorem using induction:

{theorem}

Hint: Use induction on n.
""",
        task_type="inductive_proof"
    )

    task = MDAPTask(
        task_id="sum_n",
        description="Prove 2*n = n+n by induction",
        steps=[step]
    )

    result = await orchestrator.run_task_async(task)

    if result.success:
        print("Inductive proof:")
        print(result.proof)

asyncio.run(induction_example())
```

### Example 8.3: Complex Algebraic Theorem

```python
async def complex_algebraic_example():
    """Prove complex algebraic theorem"""

    theorem = """
theorem pow_two_add (a b : Nat) :
    (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  sorry
"""

    # Use ROMA-MDAP-MAKER for complex theorem
    config = ROMAMDAPMakerConfig(
        roma_max_depth_solving=2,
        mdap_k_ahead=3,
        apply_maker_to_roma_atomic=True
    )

    engine = ROMAMDAPMakerEngine(
        config=config,
        api_key="your-api-key"
    )

    result = await engine.solve_with_romadm ap(
        theorem=theorem,
        context="""
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.Group.Basic
-- We have exponentiation and algebraic properties
"""
    )

    if result["success"]:
        print("Complex proof:")
        print(result["proof"])
        print(f"\nDecomposition used {result['roma_depth']} levels")
    else:
        print(f"Failed: {result['error']}")

asyncio.run(complex_algebraic_example())
```

---

**Document End**

For more information, see:
- `LEANAIDE_MDAP_MAKER_GUIDE.md` - Complete usage guide
- `LEANAIDE_MDAP_MAKER_API.md` - API reference
- `LEANAIDE_MDAP_ARCHITECTURE.md` - Architecture diagrams
