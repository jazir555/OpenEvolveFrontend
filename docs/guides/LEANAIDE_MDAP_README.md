# LeanAide MDAP Integration

A comprehensive Multi-Stage Agent Pipeline (MDAP) integration for Lean 4 proof generation, providing multi-agent, voting-based theorem proving with various strategies.

## Overview

This integration combines:
- **MDAP (Multi-Stage Agent Pipeline)**: Hierarchical task decomposition and execution
- **MAKER (Maximal Agentic decomposition + first-to-ahead-by-K Error correction)**: Zero-error voting
- **LeanAide**: Lean 4 proof generation and verification
- **Multiple Strategies**: Evolution, MCTS, Adversarial, Self-play, Direct translation

## Features

### Multi-Strategy Proof Generation
- **EvolutionaryAgent**: Genetic algorithm-based proof search
- **MCTSAgent**: Monte Carlo Tree Search for proof exploration
- **AdversarialAgent**: Red-blue team adversarial proof generation
- **SelfPlayAgent**: Reinforcement learning through self-play
- **DirectAgent**: Direct LLM translation to Lean

### Voting-Based Aggregation
- **First-K-Ahead**: Winner must be K votes ahead of runner-up
- **Majority**: Simple majority voting
- **Weighted**: Confidence-weighted aggregation
- **Threshold**: Select proofs above confidence threshold

### Quality Assurance
- **Red-Flagging**: Filter out invalid/low-quality proofs
- **Verification**: Lean server proof verification
- **Confidence Scoring**: Quality estimation for each proof
- **Schema Validation**: Ensure output format correctness

### Advanced Features
- **Hierarchical Decomposition**: Break complex theorems into sub-problems
- **Checkpointing**: Save/resume long-running tasks
- **Adaptive Agent Selection**: Choose best strategies per theorem
- **Domain Specialization**: Agent expertise by mathematical domain
- **Caching**: Avoid redundant computations

## Quick Start

```python
from leanaide_mdap import (
    LeanMDAPOrchestrator,
    LeanMDAPTask,
    ProofStrategy,
    LeanDomain,
    create_lean_mdap_config
)

# Create configuration
config = create_lean_mdap_config(
    available_agents=['evolution', 'mcts', 'direct'],
    default_parallel_agents=3,
    voting_strategy='first_k_ahead',
    k_ahead_threshold=3
)

# Initialize orchestrator
orchestrator = LeanMDAPOrchestrator(config=config)

# Create task
task = LeanMDAPTask(
    task_id='prove_addition_commutativity',
    description='Prove addition commutativity',
    theorem_statement='theorem add_comm (a b : Nat) : a + b = b + a',
    domain=LeanDomain.ALGEBRA
)

# Create execution plan
strategies = [ProofStrategy.EVOLUTION, ProofStrategy.MCTS, ProofStrategy.DIRECT]
task.create_default_steps(strategies, parallel=True)

# Execute (requires OPENAI_API_KEY)
result = orchestrator.orchestrate_proof_generation(task)

# Access results
if result.success:
    print(f'Best proof: {result.best_proof.lean_code}')
    print(f'Confidence: {result.best_proof.confidence:.2%}')
    print(f'Verified: {result.best_proof.verification_status}')
```

## API Reference

### Main Classes

#### LeanMDAPConfig
Configuration for Lean MDAP pipeline.

**Key Parameters:**
- `available_agents: List[str]` - Agent types to use
- `default_parallel_agents: int` - Number of parallel agents
- `voting_strategy: VotingStrategy` - Voting method
- `k_ahead_threshold: int` - K-ahead threshold
- `enable_red_flagging: bool` - Enable red-flagging
- `enable_checkpointing: bool` - Enable checkpointing

#### LeanMDAPOrchestrator
Main orchestration engine.

**Methods:**
- `orchestrate_proof_generation(task: LeanMDAPTask) -> LeanMDAPResult`
- `execute_hierarchical(task: LeanMDAPTask) -> LeanProof`
- `get_metrics() -> Dict`

#### LeanMDAPTask
Multi-step proof generation task.

**Methods:**
- `create_default_steps(strategies: List[ProofStrategy], parallel: bool) -> None`
- `get_execution_plan() -> List[LeanMDAPStep]`

#### LeanProof
Container for Lean 4 proof.

**Attributes:**
- `theorem_name: str` - Theorem name
- `lean_code: str` - Lean 4 code
- `confidence: float` - Confidence (0.0-1.0)
- `strategy_used: ProofStrategy` - Generating strategy
- `verification_status: bool` - Verification status
- `tactics_used: List[str]` - Tactics used

#### LeanMDAPResult
Result of proof generation.

**Attributes:**
- `success: bool` - Success status
- `best_proof: LeanProof` - Best proof
- `all_proofs: List[LeanProof]` - All candidates
- `voting_statistics: Dict` - Voting stats
- `red_flags: Dict` - Red-flag analysis

### Enums

#### ProofStrategy
- `EVOLUTION` - Genetic algorithm
- `MCTS` - Monte Carlo Tree Search
- `ADVERSARIAL` - Red-blue team
- `SELF_PLAY` - Reinforcement learning
- `DIRECT` - Direct translation

#### LeanDomain
- `ALGEBRA`, `ANALYSIS`, `LOGIC`, `CATEGORY_THEORY`
- `TOPOLOGY`, `NUMBER_THEORY`, `COMBINATORICS`, `GEOMETRY`, `GENERAL`

#### VotingStrategy
- `FIRST_K_AHEAD` - First-K-ahead-by-K
- `MAJORITY` - Simple majority
- `WEIGHTED` - Confidence-weighted
- `THRESHOLD` - Confidence threshold

## Usage Examples

### Example 1: Simple Proof
```python
config = create_lean_mdap_config(
    available_agents=['direct'],
    default_parallel_agents=1
)

orchestrator = LeanMDAPOrchestrator(config=config)

task = LeanMDAPTask(
    task_id='simple',
    theorem_statement='theorem trivial : True',
    domain=LeanDomain.LOGIC
)

result = orchestrator.orchestrate_proof_generation(task)
```

### Example 2: Multi-Strategy Parallel
```python
config = create_lean_mdap_config(
    available_agents=['evolution', 'mcts', 'adversarial', 'direct'],
    default_parallel_agents=4,
    voting_strategy='first_k_ahead',
    k_ahead_threshold=2
)

orchestrator = LeanMDAPOrchestrator(config=config)

task = LeanMDAPTask(
    task_id='parallel',
    theorem_statement='theorem mul_comm (a b : Nat) : a * b = b * a',
    domain=LeanDomain.ALGEBRA
)

strategies = [
    ProofStrategy.EVOLUTION,
    ProofStrategy.MCTS,
    ProofStrategy.ADVERSARIAL,
    ProofStrategy.DIRECT
]
task.create_default_steps(strategies, parallel=True)

result = orchestrator.orchestrate_proof_generation(task)
```

### Example 3: Domain-Specific
```python
config = create_lean_mdap_config(
    enable_domain_specialization=True,
    domain_agent_mapping={
        LeanDomain.ALGEBRA: ['evolution', 'mcts'],
        LeanDomain.LOGIC: ['adversarial', 'direct']
    }
)

orchestrator = LeanMDAPOrchestrator(config=config)
```

### Example 4: Custom Voting
```python
config = create_lean_mdap_config(
    voting_strategy='weighted',
    enable_red_flagging=True,
    max_proof_length=500,
    min_confidence=0.3,
    blocked_patterns=['sorry', 'admit']
)

result = orchestrator.orchestrate_proof_generation(task)
print(f'Red flags: {result.red_flags}')
```

### Example 5: Checkpointing
```python
config = create_lean_mdap_config(
    enable_checkpointing=True,
    checkpoint_dir='./checkpoints'
)

orchestrator = LeanMDAPOrchestrator(config=config)

# Long-running task - can resume if interrupted
result = orchestrator.orchestrate_proof_generation(task)
```

## Testing

```bash
# Run full test suite
python test_leanaide_mdap.py

# Run specific tests
python test_leanaide_mdap.py TestMDAPStepConfiguration
python test_leanaide_mdap.py TestRedFlagging

# Run demo
python leanaide_mdap_demo.py
```

## Architecture

### Pipeline Flow
```
Input Theorem
    ↓
Task Creation
    ↓
Agent Selection
    ↓
Parallel Execution (Multiple Agents)
    ├─ EvolutionaryAgent
    ├─ MCTSAgent
    ├─ AdversarialAgent
    ├─ SelfPlayAgent
    └─ DirectAgent
    ↓
Proof Generation
    ↓
Red-Flagging
    ↓
Voting Aggregation
    ↓
Best Proof Selection
    ↓
Verification
    ↓
Output
```

## Configuration

Full configuration example:

```python
config = LeanMDAPConfig(
    # Agents
    available_agents=['evolution', 'mcts', 'adversarial', 'self_play', 'direct'],
    default_parallel_agents=4,
    max_parallel_agents=8,

    # Voting
    voting_strategy=VotingStrategy.FIRST_K_AHEAD,
    k_ahead_threshold=3,
    min_confidence_threshold=0.5,

    # Red-flagging
    enable_red_flagging=True,
    max_proof_length=1000,
    min_confidence=0.2,
    require_verification=True,
    blocked_patterns=['TODO', 'FIXME'],

    # Execution
    timeout_seconds=300,
    max_retries=3,
    enable_checkpointing=True,

    # Strategy-specific
    evolution_population_size=20,
    evolution_max_generations=10,
    mcts_simulations=100,
    adversarial_rounds=5,

    # LeanAide
    leanaide_host='localhost',
    leanaide_port=7654,
    verification_timeout=60
)
```

## Error Handling

```python
try:
    result = orchestrator.orchestrate_proof_generation(task)
    if result.success:
        print('Success!')
    else:
        print(f'Failed: {result.error}')
except Exception as e:
    print(f'Error: {e}')
```

## Files

- `leanaide_mdap.py` - Main implementation
- `test_leanaide_mdap.py` - Test suite
- `leanaide_mdap_demo.py` - Usage examples
- `LEANAIDE_MDAP_README.md` - This file

## Dependencies

- `mdap_engine` - Core MDAP functionality
- `workflow_structures` - Team and model configuration
- `llm_utils` - LLM integration
- Optional: Lean 4 server for verification

## License

Same as OpenEvolve project.
