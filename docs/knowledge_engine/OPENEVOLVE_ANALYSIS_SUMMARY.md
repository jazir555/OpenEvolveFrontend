# OpenEvolve Evolutionary Algorithm Analysis - Executive Summary

## QUICK REFERENCE

### What OpenEvolve Actually Is

**Algorithm Type:** Quality-Diversity Evolution (MAP-Elites + Island Model)

**NOT:**
- ❌ Traditional Genetic Algorithm (GA)
- ❌ NSGA-II or SPEA2
- ❌ Standard neuroevolution
- ❌ Simple hill climbing

**IS:**
- ✅ **MAP-Elites** (Multi-Dimensional Archive of Phenotypic Elites)
- ✅ **Island-based parallel evolution** (5+ islands)
- ✅ **LLM-driven mutation** (not random/crossover)
- ✅ **Behavioral space exploration** (not just fitness)
- ✅ **Steady-state evolution** (not generational)

### Core Components (3 Pillars)

```
┌─────────────────────────────────────────────────┐
│         MAP-ELITES ARCHIVE                       │
│  Behavioral Space → Feature Grid → Best Programs │
│  (e.g., [complexity, diversity] → 10×10 grid)   │
└─────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────┐
│         ISLAND MODEL                             │
│  5 separate populations                          │
│  Independent evolution                           │
│  Periodic migration (every 50 generations)       │
└─────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────┐
│         LLM ENSEMBLE                             │
│  Multiple models with weights                    │
│  Intelligent mutation via prompts               │
│  Diff-based or full rewrite                      │
└─────────────────────────────────────────────────┘
```

### Evolutionary Operators

**SELECTION (Multi-Strategy):**
- **Exploration (20%):** Random from current island
- **Exploitation (70%):** Elite from archive
- **Random (10%):** Any program in population

**No tournament selection. No roulette wheel.**

**MUTATION (LLM-Driven):**
- Parent + inspirations → Prompt → LLM → Child code
- **Diff-based:** Small targeted changes (default)
- **Full rewrite:** Complete replacement (risky)

**No uniform mutation. No Gaussian mutation.**

**CROSSOVER (Inspiration-Based):**
- **No code recombination** (no traditional crossover)
- **Learns from examples** via prompt context
- Inspirations = best programs from island + diverse cells

**SURVIVAL (Elitist Archive):**
- **MAP-Elites grid:** One best per behavioral cell
- **Archive:** All elite programs
- **Population limit:** Remove worst when exceeded
- **Steady-state:** Immediate survival competition

**No generational replacement.**

### Key Parameters (51 Actively Used)

| Category | Parameters | Impact |
|----------|-----------|--------|
| **Core** | 8 | Iterations, early stopping, reproducibility |
| **Database** | 13 | Population, islands, MAP-Elites, selection ratios |
| **LLM** | 10 | Temperature, tokens, models, ensemble |
| **Prompt** | 6 | Examples, artifacts, stochasticity |
| **Evaluator** | 7 | Timeout, cascade, parallel, feedback |
| **Trace** | 7 | Logging format, code inclusion |

**Documented vs Actual:**
- Claimed: 272+ parameters
- Realistically used: ~51 parameters
- Remaining: Deprecated, placeholder, or domain-specific

### Performance Characteristics

**Time Complexity:**
- Per iteration: O(P + I + E) where P=population, I=inspirations, E=evaluation
- Per generation: O(N × (P + I + E)) where N=iterations
- **Linear** in iterations (not quadratic like traditional GA)
- **Efficient** sampling via hash maps

**Memory Usage:**
- Per program: ~17-162 KB (depends on artifacts)
- Population=1000: ~200 MB total

**Convergence Patterns:**
1. **Exploration (20%):** Rapid behavioral space coverage
2. **Exploitation (60%):** Improving cell occupants
3. **Convergence (20%):** Diminishing returns

### Domain Suitability

**EXCELLENT FOR:**
- ✅ Scientific experiment design
- ✅ Engineering optimization
- ✅ Algorithm discovery
- ✅ Multi-modal problems
- ✅ Problems with behavioral diversity

**GOOD FOR:**
- ⚠️ Finance (with validation)
- ⚠️ Web optimization
- ⚠️ Code optimization

**POOR FOR:**
- ❌ Pharma (use specialized tools)
- ❌ Real-time systems (too slow)
- ❌ Simple regression (use SGD/Adam)
- ❌ Molecular evolution (use GA with SMILES)

**CAUTION:**
- ⚠️⚠️⚠️ Trading (HIGH overfitting risk)

### Comparison: OpenEvolve vs LoongFlow PES

| Aspect | OpenEvolve | LoongFlow PES |
|--------|-----------|---------------|
| **Target** | Code/Algorithms | Prompts |
| **Algorithm** | MAP-Elites + Islands | Standard GA (likely) |
| **Diversity** | Behavioral space | Implicit |
| **Selection** | Archival + Multi-strategy | Tournament |
| **Mutation** | LLM-driven diffs | LLM rewrites |
| **Parameters** | ~51 active | ~15-20 (estimated) |
| **Best For** | Algorithm discovery | Prompt optimization |

**Complementary:**
- Use LoongFlow to evolve prompts for OpenEvolve evaluators
- Use OpenEvolve to generate code for LoongFlow workflows

### Gauntlet Integration

**Multi-Round Evaluation:**
```
Stage 1 (Quick): 3 test cases, 30s → Reject if score < 0.5
Stage 2 (Medium): 10 test cases, 60s → Reject if score < 0.75
Stage 3 (Full): All test cases, 300s → Final score
```

**Overhead:** 3-30x per iteration (mitigated by cascade evaluation)

**Adversarial Mode:**
- Red team generates attacks
- Blue team tests robustness
- Co-evolutionary arms race

### Critical Success Factors

**For Good Performance:**
1. ✅ **Right feature dimensions** (must match problem structure)
2. ✅ **Fast but accurate evaluation** (cascade evaluation)
3. ✅ **Balanced exploration/exploitation** (20/70/10 ratio)
4. ✅ **Sufficient iterations** (1000+ for complex problems)
5. ✅ **Good LLM choice** (model affects mutation quality)

**Common Pitfalls:**
1. ❌ **Overfitting** to training data (especially in finance/trading)
2. ❌ **Poor evaluation** (too fast = inaccurate, too slow = waste)
3. ❌ **Wrong features** (behavioral space doesn't match problem)
4. ❌ **Impatience** (stopping before convergence)
5. ❌ **Single island** (loses diversity benefits)

### Code Example: Function Optimization

```python
from openevolve import run_evolution

# Initial function
initial = """
def optimize_portfolio(returns, risk_tolerance):
    n = len(returns)
    return [1.0/n] * n  # Equal weight
"""

# Evaluator
def evaluator(path):
    module = load_module(path)
    # Test on historical data
    returns = test_on_data(module, historical_data)
    sharpe = np.mean(returns) / np.std(returns)
    return {"sharpe_ratio": sharpe, "combined_score": sharpe}

# Evolve
result = run_evolution(
    initial_program=initial,
    evaluator=evaluator,
    config={
        "max_iterations": 100,
        "database": {
            "feature_dimensions": ["risk", "return"],
            "population_size": 500
        }
    }
)

print(f"Best sharpe: {result.best_score}")
print(f"Evolved code:\n{result.best_code}")
```

### Knowledge Engine Integration

```python
# Get domain expertise
guidance = knowledge_engine.query(
    "What parameters matter for optimizing "
    "sorting algorithms? What tradeoffs exist?"
)

# Use guidance for configuration
config = Config()
config.feature_dimensions = guidance["behavioral_dimensions"]
config.exploration_ratio = guidance["exploration_level"]

# Run knowledge-guided evolution
result = await run_evolution(
    initial_program=sorting_algo,
    evaluator=benchmark_sort,
    config=config
)
```

### File Locations

**Core Engine:**
- `openevolve/openevolve/controller.py` - Main orchestration (Lines 59-432)
- `openevolve/openevolve/database.py` - MAP-Elites + islands (Lines 100-2000+)
- `openevolve/openevolve/iteration.py` - Single iteration (Lines 32-168)
- `openevolve/openevolve/config.py` - All parameters (Lines 14-400)

**Examples:**
- `openevolve/examples/algotune/` - Algorithm optimization
- `openevolve/examples/circle_packing/` - State-of-the-art results
- `openevolve/examples/symbolic_regression/` - Math discovery

**Integration:**
- `BubbleLab/services/openevolve-api/core/evolution.py` - BubbleLab adapter
- `knowledge_engine/integrations/openevolve_integration.py` - KE integration

### Key Takeaways

1. **OpenEvolve is Quality-Diversity EA**, not traditional GA
2. **MAP-Elites** is the core innovation (behavioral diversity)
3. **Island model** prevents premature convergence
4. **LLM-driven mutation** enables intelligent search
5. **51 parameters** actively used (not 272)
6. **Excellent for algorithm discovery** and engineering
7. **Caution in finance/trading** (overfitting risk)
8. **Can integrate with Knowledge Engine** for parameter guidance
9. **Gauntlets provide robust evaluation** via multi-round testing
10. **Complementary to LoongFlow PES** (different strengths)

---

**Full Report:** `OPENEVOLVE_EVOLUTIONARY_ALGORITHM_FORENSIC_ANALYSIS.md`
