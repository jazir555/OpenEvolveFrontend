# Lean 4 Self-Play System

A comprehensive self-play system for automated Lean 4 theorem proving and continuous proof improvement, inspired by PSV (Propose-Solve-Verify) and AlphaZero-style self-play.

## Overview

This system enables Lean 4 proofs to improve through automated self-play, where:
- **Prover Agent**: Generates proof strategies and tactics
- **Verifier Agent**: Verifies proofs using Lean 4 formal verification
- **Experience Replay**: Stores and learns from successful and failed proof attempts
- **Continuous Improvement**: Agent improves over time through self-play iterations

## Key Features

- **Self-Play Learning**: Automated proof improvement through self-play
- **Experience Replay**: Prioritized replay buffer for efficient learning
- **Strategy Selection**: Intelligent proof strategy selection with exploration
- **Lean 4 Integration**: Full integration with LeanAide for proof verification
- **Performance Tracking**: Comprehensive metrics and monitoring
- **Async/Await**: Modern Python async for efficient parallelization
- **Checkpoint System**: Save and resume training progress

## Architecture

```
LeanSelfPlayEngine
├── LeanProofAgent (Prover + Verifier)
│   ├── Policy Network (Strategy Selection)
│   ├── Value Network (Proof Quality Estimation)
│   └── LLM Integration (Tactic Generation)
├── Lean4Verifier (Formal Verification)
├── LeanProofExperienceBuffer (Replay Memory)
└── LeanSelfPlayGame (Single Episode)
```

## Installation

### Requirements

```bash
pip install httpx numpy pytest
```

### Optional Dependencies

For full LLM integration:
```bash
pip install openai anthropic
```

For testing:
```bash
pip install pytest pytest-asyncio
```

## Quick Start

### Basic Usage

```python
import asyncio
from leanaide_selfplay import LeanSelfPlayEngine

async def main():
    # Initialize self-play engine
    engine = LeanSelfPlayEngine(
        leanaide_url="http://localhost:7654",
        buffer_capacity=1000
    )

    try:
        # Run self-play for a theorem
        theorem = "∀ n : Nat, n + 0 = n"
        best_proof = await engine.run_self_play(
            theorem=theorem,
            games=10
        )

        print(f"Best proof found: {best_proof.is_valid}")
        print(f"Proof tactics: {best_proof.tactic_count}")

        # Train from experiences
        metrics = await engine.train_from_buffer(
            batch_size=32,
            iterations=10
        )

        print(f"Success rate: {metrics.success_rate:.2%}")

    finally:
        await engine.close()

asyncio.run(main())
```

### Batch Self-Play

```python
async def batch_training():
    engine = LeanSelfPlayEngine()

    try:
        # Train on multiple theorems
        theorems = [
            "∀ n : Nat, n + 0 = n",
            "∀ a b : Nat, a + b = b + a",
            "∀ n : Nat, 2 * n = n + n"
        ]

        results = await engine.run_batch_self_play(
            theorems=theorems,
            games_per_theorem=5
        )

        # Train and save checkpoint
        await engine.train_from_buffer(batch_size=16, iterations=20)
        engine.save_checkpoint("training_checkpoint.json")

        # Get progress
        progress = engine.get_training_progress()
        print(f"Improvement: {progress['improvement']['relative']:.2%}")

    finally:
        await engine.close()

asyncio.run(batch_training())
```

### Custom Proof Strategies

```python
from leanaide_selfplay import LeanProofStrategy

# Define custom strategy
custom_strategy = LeanProofStrategy(
    name="induction_strategy",
    tactic_sequence=["induction", "case", "simp"],
    description="Proof by induction",
    适用领域=["combinatorics", "algebra"],
    success_rate=0.6
)

# Add to agent
engine.agent.known_strategies.append(custom_strategy)

# Self-play will now use custom strategy
await engine.run_self_play("∀ n : Nat, n + 0 = n", games=5)
```

## Core Components

### 1. LeanSelfPlayEngine

Main orchestration engine for self-play training.

```python
engine = LeanSelfPlayEngine(
    leanaide_url="http://localhost:7654",
    buffer_capacity=10000,
    max_concurrent_games=4
)

# Run self-play
proof = await engine.run_self_play(theorem, games=10)

# Train from buffer
metrics = await engine.train_from_buffer(batch_size=32, iterations=10)

# Get progress
progress = engine.get_training_progress()

# Save/load checkpoints
engine.save_checkpoint("checkpoint.json")
engine.load_checkpoint("checkpoint.json")
```

### 2. LeanProofAgent

Agent that generates and evaluates proofs.

```python
from leanaide_selfplay import LeanProofAgent, LeanTheorem

agent = LeanProofAgent(
    agent_id="prover",
    llm_config={"model": "gpt-4"},
    verifier=verifier,
    exploration_rate=0.3
)

# Select strategy
strategy = await agent.select_proof_strategy(theorem, training=True)

# Generate proof
proof = await agent.generate_proof(theorem, strategy)

# Evaluate quality
value = await agent.evaluate_proof(proof)
```

### 3. Lean4Verifier

Interface to Lean 4 theorem prover.

```python
from leanaide_selfplay import Lean4Verifier

verifier = Lean4Verifier(
    leanaide_url="http://localhost:7654",
    timeout=300
)

# Verify proof
status, output, error = await verifier.verify_proof(theorem, proof)

# Status is one of:
# - ProofStatus.VERIFIED
# - ProofStatus.FAILED
# - ProofStatus.PARTIAL
# - ProofStatus.TIMEOUT
```

### 4. LeanProofExperienceBuffer

Prioritized experience replay buffer.

```python
from leanaide_selfplay import LeanProofExperienceBuffer

buffer = LeanProofExperienceBuffer(
    capacity=10000,
    prioritized=True
)

# Add experience
buffer.add(experience)

# Sample batch
batch = buffer.sample(batch_size=32, beta=0.4)

# Get statistics
stats = buffer.get_statistics()

# Save/load
buffer.save("buffer.json")
buffer.load("buffer.json")
```

### 5. LeanSelfPlayGame

Single self-play game episode.

```python
from leanaide_selfplay import LeanSelfPlayGame

game = LeanSelfPlayGame(theorem, agent, verifier)

# Play game
experience = await game.play()

# Experience contains:
# - theorem
# - proof
# - reward
# - strategy_used
# - value_estimate
```

## Data Structures

### LeanTheorem

```python
theorem = LeanTheorem(
    id="theorem_1",
    statement="∀ n : Nat, n + 0 = n",
    lean_code="theorem test : ∀ n : Nat, n + 0 = n := by",
    difficulty=ProofDifficulty.EASY,
    domain="algebra",
    dependencies=["Nat.add_zero"]
)
```

### LeanProof

```python
proof = LeanProof(
    theorem_id="theorem_1",
    tactics=[
        LeanTactic(name="intro", args=["n"]),
        LeanTactic(name="rw", args=["Nat.add_zero"]),
        LeanTactic(name="rfl")
    ],
    lean_code="intro n\nrw [Nat.add_zero]\nrfl",
    status=ProofStatus.VERIFIED,
    confidence=0.9
)
```

### LeanProofExperience

```python
experience = LeanProofExperience(
    theorem=theorem,
    proof=proof,
    reward=1.0,
    strategy_used="direct_proof",
    value_estimate=0.9,
    policy_output={"direct_proof": 0.8, "induction": 0.2}
)
```

## Self-Play Dynamics

### Exploration vs Exploitation

```python
# During training:
exploration_rate = 0.3  # 30% exploration, 70% exploitation

if random.random() < exploration_rate:
    # Explore: Try random strategy
    strategy = random.choice(strategies)
else:
    # Exploit: Use best known strategy
    strategy = select_best_strategy(theorem)
```

### Reward Shaping

```python
reward = (
    base_reward  # 1.0 for verified, 0.5 for partial
    - length_penalty  # -0.01 per tactic
    - time_penalty  # -0.001 per second
    + elegance_bonus  # +0.1 for tactic diversity
    + confidence_bonus  # +0.1 * confidence
    + difficulty_bonus  # Up to +0.5 for hard theorems
)
```

### Difficulty Levels

```python
ProofDifficulty.TRIVIAL  # Basic properties
ProofDifficulty.EASY     # Simple theorems
ProofDifficulty.MEDIUM    # Intermediate
ProofDifficulty.HARD     # Complex proofs
ProofDifficulty.EXPERT   # Very difficult
ProofDifficulty.RESEARCH # Open problems
```

## Training Loop

```python
async def training_loop():
    engine = LeanSelfPlayEngine()

    try:
        for epoch in range(num_epochs):
            # Self-play phase
            for batch in theorem_batches:
                await engine.run_batch_self_play(
                    theorems=batch,
                    games_per_theorem=games_per_batch
                )

            # Training phase
            metrics = await engine.train_from_buffer(
                batch_size=32,
                iterations=training_iterations
            )

            # Logging
            print(f"Epoch {epoch}: {metrics.success_rate:.2%} success")

            # Checkpoint
            if epoch % checkpoint_interval == 0:
                engine.save_checkpoint(f"checkpoint_{epoch}.json")

    finally:
        await engine.close()
```

## Performance Metrics

### Success Rate

Percentage of proof attempts that verify successfully.

```python
metrics = engine.get_training_progress()
success_rate = metrics["success_rate"]
```

### Average Reward

Mean reward across all experiences.

```python
avg_reward = metrics["avg_reward"]
```

### Proof Length

Average number of tactics in proofs.

```python
avg_length = metrics["avg_proof_length"]
```

### Improvement

Relative and absolute improvement over training.

```python
improvement = metrics["improvement"]
relative_improvement = improvement["relative"]  # Percentage
absolute_improvement = improvement["absolute"]  # Absolute
```

## Integration with LeanAide

The system integrates with LeanAide server for Lean 4 verification:

1. **Start LeanAide server**:
   ```bash
   cd LeanAide
   python leanaide_server.py
   ```

2. **Configure self-play engine**:
   ```python
   engine = LeanSelfPlayEngine(
       leanaide_url="http://localhost:7654"
   )
   ```

3. **Run self-play**:
   ```python
   await engine.run_self_play(theorem, games=10)
   ```

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_leanaide_selfplay.py -v

# Run specific test class
pytest test_leanaide_selfplay.py::TestLeanProofExperienceBuffer -v

# Run with coverage
pytest test_leanaide_selfplay.py --cov=leanaide_selfplay --cov-report=html
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Stress test and benchmarks
- **Example Usage Tests**: Demonstrate usage patterns

## Advanced Features

### Custom Reward Function

```python
class CustomEngine(LeanSelfPlayEngine):
    def _calculate_reward(self, proof):
        # Custom reward calculation
        reward = base_reward

        # Add custom bonuses/penalties
        reward += custom_bonus(proof)
        reward -= custom_penalty(proof)

        return reward
```

### Curriculum Learning

```python
# Start with easy theorems
easy_theorems = load_theorems(difficulty="easy")
await engine.run_batch_self_play(easy_theorems)

# Progress to medium
medium_theorems = load_theorems(difficulty="medium")
await engine.run_batch_self_play(medium_theorems)

# Finally hard
hard_theorems = load_theorems(difficulty="hard")
await engine.run_batch_self_play(hard_theorems)
```

### Parallel Self-Play

```python
# Run multiple games in parallel
import asyncio

async def parallel_self_play():
    engine = LeanSelfPlayEngine(max_concurrent_games=8)

    theorems = [f"theorem_{i}" for i in range(100)]

    # Automatically parallelized
    await engine.run_batch_self_play(
        theorems=theorems,
        games_per_theorem=5
    )
```

## Monitoring and Visualization

### Training Metrics

```python
# Get metrics history
for metrics in engine.metrics_history:
    print(f"Iteration {metrics.iteration}:")
    print(f"  Success rate: {metrics.success_rate:.2%}")
    print(f"  Avg reward: {metrics.avg_reward:.3f}")
    print(f"  Buffer size: {metrics.buffer_size}")
```

### Performance Tracking

```python
# Agent performance history
for record in agent.performance_history:
    print(f"Theorem: {record['theorem_id']}")
    print(f"  Strategy: {record['strategy_used']}")
    print(f"  Success: {record['success']}")
    print(f"  Reward: {record['reward']:.3f}")
```

## Troubleshooting

### LeanAide Connection Issues

```python
# Verify LeanAide is running
import httpx

async def check_leanaide():
    try:
        response = await httpx.AsyncClient().get("http://localhost:7654/health")
        print("LeanAide is running")
    except Exception as e:
        print(f"Cannot connect to LeanAide: {e}")
```

### Buffer Management

```python
# Check buffer statistics
stats = engine.buffer.get_statistics()
print(f"Buffer size: {stats['size']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average reward: {stats['avg_reward']:.3f}")
```

### Performance Optimization

```python
# Increase parallelization
engine = LeanSelfPlayEngine(max_concurrent_games=8)

# Adjust buffer size
engine = LeanSelfPlayEngine(buffer_capacity=10000)

# Reduce verification timeout
verifier = Lean4Verifier(timeout=60)
```

## Examples

See the `test_leanaide_selfplay.py` file for comprehensive examples:

- Basic self-play workflow
- Training loops
- Custom strategies
- Performance tests
- Concurrent games

## References

This implementation is inspired by:

1. **PSV (Propose-Solve-Verify)**: Self-play through formal verification
   - Paper: "Propose, Solve, Verify: Self-Play Through Formal Verification"
   - Uses formal verification for reliable reward signals

2. **AlphaZero**: Self-play reinforcement learning
   - Paper: "Mastering Chess and Shogi by Self-Play"
   - MCTS + neural network approach

3. **Lean 4**: Theorem prover and programming language
   - Documentation: https://leanprover.github.io/

4. **LeanAide**: LLM-assisted Lean 4 proving
   - GitHub: [LeanAide repository]

## Future Enhancements

- [ ] Neural network integration for policy and value functions
- [ ] Monte Carlo Tree Search (MCTS) for tactic search
- [ ] Multi-agent self-play (competitive proving)
- [ ] Curriculum learning automation
- [ ] Transfer learning between domains
- [ ] Distributed training support
- [ ] Web dashboard for monitoring

## License

This project is part of OpenEvolve framework.

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- Code follows PEP 8 style
- Documentation is updated
- Examples are provided

## Support

For issues, questions, or contributions, please refer to the main OpenEvolve documentation.
