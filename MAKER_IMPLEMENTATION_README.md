# MAKER Complete Implementation

Complete implementation of the MAKER framework from the paper:

**"Solving a Million-Step LLM Task with Zero Errors"** (arXiv:2511.09030)

Authors: Giuseppe Paolo, Olivier Francon, Roberto Dailey, Conor F. Hayes, Hormoz Shahrzad, Xin Qiu, Babak Hodjat, Risto Miikkulainen

## Overview

MAKER (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) is the first system that successfully solves tasks with over one million LLM steps with zero errors.

### Key Innovations

1. **Maximal Agentic Decomposition (MAD)**: Breaking tasks into minimal subtasks where each agent performs only one step
2. **First-to-ahead-by-k Voting**: Efficient error correction through statistical voting
3. **Red-flagging**: Discarding responses that show signs of unreliability
4. **Recursive Decomposition**: General-purpose task decomposition (Algorithm 4)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAKER Integration Bridge                     │
│                  (maker_integration_bridge.py)                  │
└────────────┬────────────────────────────────────────────────────┘
             │
     ┌───────┴───────┬───────────────┬──────────────┐
     ▼               ▼               ▼              ▼
┌─────────┐    ┌──────────┐    ┌─────────┐   ┌──────────┐
│Algorithm│    │Algorithm │    │Algorithm│   │Algorithm │
│   1     │    │    2     │    │    3    │   │    4     │
│generate │    │do_voting │    │get_vote │    │recursive │
│solution │    │          │    │         │    │  solve   │
└─────────┘    └──────────┘    └─────────┘   └──────────┘
     │               │               │              │
     └───────────────┴───────────────┴──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Core MAKER     │
                    │  Components     │
                    │ (mdap_maker_    │
                    │  complete.py)   │
                    └─────────────────┘
```

## Files

1. **mdap_maker_complete.py** - Core MAKER implementation
   - Algorithm 1: generate_solution
   - Algorithm 2: do_voting (first-to-ahead-by-k)
   - Algorithm 3: get_vote (with red-flagging)
   - Algorithm 4: Recursive multi-agent solve

2. **maker_integration_bridge.py** - Unified integration layer
   - MAKERIntegrationBridge class
   - Convenience functions
   - Examples: Towers of Hanoi, multiplication

3. **demo_maker_complete.py** - Comprehensive demos
   - Towers of Hanoi (paper example)
   - Multi-digit multiplication
   - Custom task decomposition
   - Voting mechanism tests

## Installation

No additional dependencies required beyond the OpenEvolve stack:

```bash
# Already installed in your environment
- llm_utils
- workflow_structures
- Python 3.10+
```

## Quick Start

### Example 1: Towers of Hanoi (Canonical Example)

The paper's example: solving Towers of Hanoi with 20 disks = 1,048,575 steps with zero errors.

```python
from maker_integration_bridge import solve_towers_of_hanoi

# Solve with 20 disks (use smaller number for testing)
result = solve_towers_of_hanoi(num_disks=5, k_ahead=3)

print(f"Success: {result['success']}")
print(f"Steps: {result['metrics'].total_steps}")
print(f"Votes: {result['metrics'].total_votes}")
```

### Example 2: General Task Solving

Use recursive decomposition for any task:

```python
from maker_integration_bridge import solve_with_maker

result = solve_with_maker(
    task="Explain the causes of the American Civil War",
    mode="recursive",
    k_ahead=3,
    max_depth=4
)

print(f"Solution: {result['result']}")
```

### Example 3: Multi-Digit Multiplication

From Appendix F of the paper:

```python
from maker_integration_bridge import solve_multiplication

result = solve_multiplication(num1=123, num2=456, k_ahead=3)

print(f"Product: {result['result']}")
```

## Configuration

### Basic Configuration

```python
from maker_integration_bridge import create_maker_config, MAKERIntegrationBridge

config = create_maker_config(
    mode="recursive",           # Execution mode
    k_ahead=3,                  # Voting threshold
    max_depth=5,                # Max recursion depth
    enable_red_flagging=True,   # Enable red-flagging
    max_token_length=750        # Max response length
)

bridge = MAKERIntegrationBridge(config, team)
```

### Execution Modes

1. **Sequential** (Algorithm 1)
   - For tasks with predetermined step sequences
   - Example: Towers of Hanoi, known algorithms
   - Use: `mode="sequential"`

2. **Recursive** (Algorithm 4)
   - For general-purpose problem solving
   - Automatically decomposes tasks
   - Use: `mode="recursive"` (default)

3. **Hybrid**
   - ROMA decomposition + MAKER voting
   - For complex hierarchical tasks
   - Use: `mode="hybrid"`

### Voting Parameters

- **k_ahead** (int, default=3): Voting threshold
  - Higher k = more reliability, more cost
  - Paper used k=3 for 20-disk Hanoi

- **num_candidates** (int, default=5): N = 2k - 1
  - Number of samples per voting round
  - More candidates = better error correction

- **enable_first_to_ahead** (bool, default=True)
  - True: first-to-ahead-by-k (from paper)
  - False: first-to-k (simpler variant)

### Red-Flagging Parameters

From the paper, Section 3.3:

- **enable_red_flagging** (bool, default=True)
  - Discards unreliable responses
  - Critical for reducing correlated errors

- **max_token_length** (int, default=750)
  - Responses longer than this are flagged
  - Paper found error rate increases after ~700 tokens

- **max_characters** (int, optional, default=6000)
  - Character limit for responses

## Algorithms

### Algorithm 1: generate_solution

Main orchestration for sequential tasks:

```python
from maker_integration_bridge import MAKERIntegrationBridge, create_maker_config

config = create_maker_config(mode="sequential")
bridge = MAKERIntegrationBridge(config, team)

# Solve sequential task
actions, final_state, metrics = bridge.engine.generate_solution(
    initial_state=initial_state,
    prompt_template=lambda s: f"Current state: {s}",
    system_prompt="You are a task execution agent",
    stop_condition=lambda s: is_done(s)
)
```

**Time Complexity**: O(s × k) where s = steps, k = voting threshold
**Space Complexity**: O(1) - only keeps current state

### Algorithm 2: do_voting

First-to-ahead-by-k voting mechanism:

```python
from mdap_maker_complete import VotingEngine, VoteCollector

collector = VoteCollector(max_token_length=750)
engine = VotingEngine(collector, enable_first_to_ahead=True)

winner, votes, metrics = engine.do_voting(
    prompt="What is 7 × 8?",
    system_prompt="You are a helpful assistant",
    agents=team.members,
    k=3
)
```

**Guarantee**: With probability 1 - ε, selects correct answer

### Algorithm 3: get_vote

Vote collection with red-flagging:

```python
from mdap_maker_complete import VoteCollector

collector = VoteCollector(max_token_length=750)

action, state, raw_text = collector.get_vote(
    prompt="Solve this step",
    system_prompt="You are a specialized agent",
    agent=model_config,
    expected_schema={"type": "object"}
)
```

**Red Flags**:
- Response too long (> max_token_length)
- Malformed (fails schema validation)
- Empty response

### Algorithm 4: Recursive Multi-Agent Solve

General-purpose decomposition:

```python
from mdap_maker_complete import RecursiveMAKERSolver

solver = RecursiveMAKERSolver(
    team=team,
    max_depth=5,
    k_ahead=3
)

solution, metrics = solver.solve(
    task="Break down the problem of... ",
    context={"requirements": [...]}
)
```

**Key Features**:
1. DECOMPOSE(x): Break task into 2 subtasks with voting
2. ATOMIC(x): Solve minimal tasks directly with voting
3. SOLVE(x, d): Recursive solve with depth limit

## Scaling Laws

From the paper, Section 3.2:

### Probability of Success

For maximal decomposition (m=1):

$$P_{full} = \left(1 + \frac{1-p}{p}\right)^k^{-\frac{s}{m}}$$

Where:
- p = per-step success rate
- k = voting threshold
- s = total steps
- m = steps per subtask (1 for maximal decomposition)

### Expected Cost

$$E[cost] = \Theta\left(p^{-1} c s \ln s\right)$$

For maximal decomposition, cost grows **log-linearly** with steps!

### Practical Implications

| Steps | k (p=0.99) | k (p=0.95) | Expected Cost |
|-------|------------|------------|---------------|
| 100   | 2          | 3          | Low           |
| 1,000 | 2          | 4          | Medium        |
| 10,000| 3          | 5          | Medium-High   |
| 1M    | 3          | 8          | High          |

## Performance

### From the Paper

**Towers of Hanoi (20 disks = 1,048,575 steps)**:
- Model: gpt-4.1-mini (non-reasoning)
- k_ahead: 3
- Result: **Zero errors** ✓
- Cost: ~$3,500

**Key Finding**: Small non-reasoning models with MAKER outperform large reasoning models without MAKER.

### Expected Performance in This Implementation

```python
# Estimate cost and k for your task
from mdap_maker_complete import RecursiveMAKERSolver

# For a task with 10,000 expected steps
solver = RecursiveMAKERSolver(team, max_depth=5)

# Theoretical minimum k for 95% success rate
k_min = solver._compute_min_k(
    steps=10000,
    per_step_success_rate=0.99,
    target_probability=0.95
)
# k_min ≈ 3

# Expected cost
cost = solver._estimate_cost(
    steps=10000,
    k=k_min,
    per_step_success_rate=0.99,
    cost_per_call=0.001
)
```

## Use Cases

### 1. Long-Sequential Tasks

Tasks with many dependent steps where any error breaks the chain:

- Algorithm execution
- Multi-step procedures
- Proof verification
- Code generation (large programs)

### 2. Complex Decomposition

Tasks requiring intelligent breakdown:

- Research planning
- System design
- Project management
- Analysis tasks

### 3. High-Reliability Requirements

Tasks where 99.9% isn't good enough:

- Safety-critical systems
- Financial calculations
- Medical diagnosis support
- Legal reasoning

## Integration with Existing Components

### With MDAP Engine

```python
from mdap_engine import MDAPOrchestrator, MDAPConfig
from mdap_maker_complete import MAKEREngine

# Use MAKER voting within MDAP
mdap_config = MDAPConfig(k_min=3, k_max=8)
mdap = MDAPOrchestrator(team, mdap_config)
```

### With ROMA Decomposition

```python
from maker_integration_bridge import solve_with_maker

# ROMA provides hierarchy, MAKER provides voting
result = solve_with_maker(
    task="Complex task requiring decomposition",
    mode="hybrid",
    enable_roma=True
)
```

### With Sovereign Decomposition

```python
from decomposition_engine import DecompositionEngine
from maker_integration_bridge import MAKERIntegrationBridge

# Use Sovereign for initial decomposition, MAKER for execution
decomposer = DecompositionEngine()
plan = decomposer.decompose(problem)

# Execute each sub-problem with MAKER voting
bridge = MAKERIntegrationBridge(config, team)
for sub_problem in plan.sub_problems:
    result = bridge.solve(sub_problem.description)
```

## Testing

### Run All Demos

```bash
# Quick demo (5 disks)
python demo_maker_complete.py

# Full demo suite
python demo_maker_complete.py --all

# Test specific components
python demo_maker_complete.py --test-voting
python demo_maker_complete.py --test-redflag
```

### Test Custom Task

```bash
python demo_maker_complete.py --example custom --task "Your task here"
```

## Troubleshooting

### Issue: Voting never converges

**Symptoms**: Max rounds reached, no winner

**Solutions**:
1. Increase `k_ahead` (lower threshold)
2. Check prompt quality
3. Try different temperature settings
4. Verify schema is correct

### Issue: High red-flag rate

**Symptoms**: Many responses being discarded

**Solutions**:
1. Increase `max_token_length`
2. Check for schema issues
3. Review system prompt clarity
4. Try simpler task decomposition

### Issue: Slow execution

**Symptoms**: Taking too long

**Solutions**:
1. Reduce `k_ahead` (trade-off: less reliability)
2. Reduce `num_candidates`
3. Decrease `max_depth`
4. Use faster model (trade-off: lower per-step success rate)

### Issue: Out of memory

**Symptoms**: Memory error on large tasks

**Solutions**:
1. Reduce `cache_max_size`
2. Disable caching: `enable_caching=False`
3. Break task into smaller chunks
4. Use `max_steps` to limit sequential execution

## Advanced Usage

### Custom Parsers

Define custom response parsing:

```python
def my_parser(raw_text: str) -> Tuple[Any, Any]:
    # Extract action
    action = extract_action(raw_text)

    # Extract state
    state = extract_state(raw_text)

    return action, state

result = bridge.solve(
    task="...",
    parser=my_parser
)
```

### Custom Stop Conditions

```python
def my_stop_condition(state) -> bool:
    # Check if we should stop
    return state.get("done", False)

result = bridge.solve(
    task="...",
    stop_condition=my_stop_condition
)
```

### Progress Monitoring

```python
def my_progress_callback(step: int, state: Any):
    print(f"Step {step}: {state}")

result = bridge.solve(
    task="...",
    progress_callback=my_progress_callback
)
```

## References

1. **Paper**: "Solving a Million-Step LLM Task with Zero Errors"
   - arXiv:2511.09030
   - https://arxiv.org/abs/2511.09030

2. **Towers of Hanoi Benchmark**
   - Prior work on LLM reasoning limits
   - Demonstrates catastrophic failure without MAKER

3. **Error Correction Theory**
   - Sequential Probability Ratio Test (SPRT)
   - Gambler's Ruin problem
   - Voting/ensembling techniques

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{meyerson2025maker,
  title={Solving a Million-Step LLM Task with Zero Errors},
  author={Paolo, Giuseppe and Francon, Olivier and Dailey, Roberto and Hayes, Conor F. and Shahrzad, Hormoz and Qiu, Xin and Hodjat, Babak and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2511.09030},
  year={2025}
}
```

## License

This implementation follows the same license as the OpenEvolve project.

## Contributing

To extend MAKER with new features:

1. **New Voting Strategies**: Extend `VotingEngine` class
2. **New Red Flags**: Extend `VoteCollector._has_red_flags()`
3. **New Decomposition**: Extend `RecursiveMAKERSolver._decompose()`
4. **New Execution Modes**: Add to `MAKERIntegrationBridge`

## Contact

For questions or issues:
- Open an issue on the OpenEvolve repository
- Consult the paper for theoretical details
- Check the demo files for usage examples

---

**Status**: ✓ Complete Implementation
**Last Updated**: 2025-12-30
**Paper Version**: arXiv:2511.09030v1
