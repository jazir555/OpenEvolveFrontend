# Evolutionary MCTS - Quick Reference Guide

## Quick Start

```python
from mcts_evolutionary_nodes import (
    EvolutionaryMCTS,
    ProofContext,
    create_evolutionary_mcts,
    ActionSequence,
    create_action_sequence_from_tactics
)

# 1. Create context
context = ProofContext(
    theorem="forall a b, a + b = b + a",
    goals=["prove equality"],
    hypotheses=[],
    available_tactics=["intros", "simp", "rw", "apply"]
)

# 2. Create evolutionary MCTS
emcts = create_evolutionary_mcts(
    population_size=20,      # Population at each node
    evolution_generations=5, # Generations per simulation
    mcts_simulations=100     # MCTS iterations
)

# 3. Search
result = await emcts.search(context)

# 4. Get result
if result.success:
    print(result.best_proof.lean_code)
```

---

## Key Classes

### EvolutionaryMCTS
**Main algorithm class**
```python
emcts = EvolutionaryMCTS(
    population_size=20,
    evolution_generations=5,
    exploration_constant=1.414,
    mcts_simulations=100,
    mutation_rate=0.1,
    crossover_rate=0.7,
    elite_count=2
)
```

### ActionSequence
**Genome for evolution**
```python
sequence = ActionSequence(
    actions=[Tactic("intros"), Tactic("simp")],
    fitness=0.8,
    generation=5
)
```

### EvolutionaryNode
**MCTS node with population**
```python
node = EvolutionaryNode(
    state=proof_state,
    population_size=20,
    mutation_rate=0.1,
    crossover_rate=0.7,
    elite_count=2
)
```

---

## Operators

### Crossover
```python
from mcts_evolutionary_nodes import SequenceCrossover

crossover = SequenceCrossover(context_aware=True)
child1, child2 = crossover.context_aware_crossover(
    parent1, parent2, context
)
```

### Mutation
```python
from mcts_evolutionary_nodes import SequenceMutation

mutation = SequenceMutation()
mutated = mutation.adaptive_mutation(
    sequence,
    mutation_rate=0.1,
    available_tactics=["intros", "simp", "rw"]
)
```

### Selection
```python
from mcts_evolutionary_nodes import SequenceSelection

selection = SequenceSelection()
parent = selection.tournament_selection(
    population,
    tournament_size=3
)
```

### Evaluation
```python
from mcts_evolutionary_nodes import SequenceEvaluator

evaluator = SequenceEvaluator()
fitness = evaluator.evaluate(sequence, context)
```

---

## Advanced Features

### Adaptive Control
```python
from mcts_evolutionary_nodes import AdaptiveEvolutionController

controller = AdaptiveEvolutionController()

# Should we evolve?
if controller.should_evolve_at_node(node, depth=5):
    # How many generations?
    generations = controller.get_evolution_generations(node, depth=5)
    # What population size?
    pop_size = controller.get_population_size(node, depth=5)
```

### Distributed Search
```python
from mcts_evolutionary_nodes import DistributedEvolutionaryMCTS

distributed = DistributedEvolutionaryMCTS(
    base_mcts=emcts,
    max_workers=4
)
result = await distributed.distributed_search(context)
```

### LeanAide Verification
```python
from leanaide_client import LeanAideClient
from mcts_evolutionary_nodes import EvolutionaryMCTSWithLeanAide

client = LeanAideClient()
emcts = EvolutionaryMCTSWithLeanAide(
    leanaide_client=client,
    population_size=20
)
result = await emcts.search_with_verification(theorem, context)
```

### Caching
```python
from mcts_evolutionary_nodes import EvolutionaryNodeCache

cache = EvolutionaryNodeCache(max_size=1000)
node = cache.get_or_compute(
    state_hash="abc123",
    compute_fn=lambda: EvolutionaryNode(state)
)
```

---

## Configuration Guide

### Population Size
- **Small (10-15):** Fast, good for shallow trees
- **Medium (20-30):** Balanced, good default
- **Large (50+):** Thorough, good for hard theorems

### Evolution Generations
- **Low (1-3):** Fast exploration
- **Medium (5-10):** Balanced convergence
- **High (15+):** Thorough evolution

### Mutation Rate
- **Low (0.05):** Exploitation-focused
- **Medium (0.1):** Balanced
- **High (0.2+):** Exploration-focused

### Crossover Rate
- **Low (0.5):** More mutation
- **Medium (0.7):** Balanced (recommended)
- **High (0.9):** More crossover

### Elite Count
- **1-2:** Minimal elitism
- **3-5:** Moderate elitism
- **5+:** High elitism (may reduce diversity)

---

## Performance Tuning

### Speed vs Quality

| Goal | Population | Generations | Simulations |
|------|-----------|-------------|-------------|
| Fast | 10 | 3 | 50 |
| Balanced | 20 | 5 | 100 |
| Quality | 50 | 10 | 500 |

### Memory Constraints

If memory is limited:
1. Reduce population_size
2. Reduce mcts_simulations
3. Enable EvolutionaryNodeCache
4. Use distributed evolution with fewer workers

### Time Constraints

If time is limited:
1. Reduce evolution_generations
2. Reduce mcts_simulations
3. Use AdaptiveEvolutionController
4. Enable early termination

---

## Common Patterns

### Pattern 1: Simple Proof Search
```python
emcts = create_evolutionary_mcts(
    population_size=20,
    evolution_generations=5,
    mcts_simulations=100
)
result = await emcts.search(context)
```

### Pattern 2: High-Quality Proof
```python
emcts = create_evolutionary_mcts(
    population_size=50,
    evolution_generations=10,
    mcts_simulations=500,
    mutation_rate=0.05
)
result = await emcts.search(context)
```

### Pattern 3: Fast Proof Search
```python
emcts = create_evolutionary_mcts(
    population_size=10,
    evolution_generations=3,
    mcts_simulations=50
)
result = await emcts.search(context)
```

### Pattern 4: Parallel Search
```python
distributed = DistributedEvolutionaryMCTS(
    base_mcts=create_evolutionary_mcts(population_size=20),
    max_workers=4
)
result = await distributed.distributed_search(context)
```

### Pattern 5: Verified Search
```python
emcts = EvolutionaryMCTSWithLeanAide(
    leanaide_client=LeanAideClient(),
    population_size=30,
    evolution_generations=7
)
result = await emcts.search_with_verification(theorem, context)
```

---

## Troubleshooting

### Problem: Slow convergence
**Solution:**
- Increase mutation_rate
- Reduce population_size
- Check if population is converged

### Problem: Poor quality proofs
**Solution:**
- Increase evolution_generations
- Increase population_size
- Enable LeanAide verification

### Problem: Out of memory
**Solution:**
- Reduce population_size
- Enable EvolutionaryNodeCache
- Use distributed evolution

### Problem: Not finding proofs
**Solution:**
- Increase mcts_simulations
- Check available_tactics
- Try different theorem formulation

---

## API Reference

### EvolutionaryMCTS.search()
```python
async def search(
    initial_context: ProofContext,
    leanaide_client: Optional[LeanAideClient] = None
) -> MCTSResult
```

### EvolutionaryNode.update_population()
```python
def update_population(
    self,
    population: List[ActionSequence]
) -> None
```

### ActionSequence.to_string()
```python
def to_string(self) -> str:
    """Convert to Lean code"""
```

### SequenceEvaluator.evaluate()
```python
def evaluate(
    self,
    sequence: ActionSequence,
    context: ProofContext
) -> float:
    """Return fitness (0-1)"""
```

---

## Example Output

```
================================================================================
EVOLUTIONARY MCTS RESULTS
================================================================================

Success: True
Time: 2.34s
Nodes visited: 45
Tree depth: 12
Win rate: 0.9500

Best Proof:
--------------
intros
simp
rw [add_comm]
linarith
--------------

Statistics:
- Total evolutions: 100
- Total evaluations: 2000
- Avg population size: 20
- Convergence rate: 0.85
```

---

## Tips & Best Practices

1. **Start with default parameters**, then tune
2. **Use context-aware crossover** for better offspring
3. **Enable caching** for repeated searches
4. **Monitor convergence** to detect stagnation
5. **Use parallel search** for large theorems
6. **Verify proofs** with LeanAide when possible
7. **Adjust population size** based on theorem difficulty
8. **Use adaptive control** for efficient resource usage

---

## File Locations

- **Implementation:** `mcts_evolutionary_nodes.py` (2,200 lines)
- **Tests:** `test_evolutionary_mcts.py`
- **Documentation:** `EVOLUTIONARY_MCTS_IMPLEMENTATION.md`

---

## Related Modules

- `leanaide_mcts.py` - Basic MCTS implementation
- `leanaide_evolution.py` - Evolutionary proof generation
- `leanaide_client.py` - LeanAide API client
- `lean4_integration.py` - Lean 4 integration

---

## Contact & Support

For questions or issues:
1. Check the test file for examples
2. Read the implementation documentation
3. Examine the inline code comments
4. Run tests to verify installation

---

**Last Updated:** 2025-12-30
**Version:** 1.0.0
**Status:** Production Ready ✅
