# LeanAide Decomposition Integration Guide

## Overview

This document describes the integration of LeanAide evolutionary proof generation capabilities into the OpenEvolve decomposition engine. The enhanced system can now detect mathematical problems, decompose them into Lean 4 formalizable components, and suggest evolutionary proof generation strategies.

## Architecture

```
User Input (Problem)
        ↓
┌───────────────────────────────────────┐
│  LeanMathematicalDetector              │
│  - Detect mathematical problems       │
│  - Classify problem type               │
│  - Identify domain                     │
│  - Estimate difficulty                 │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  Problem Routing                       │
│  - Mathematical → LeanAide path        │
│  - General → Standard decomposition    │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  LeanDecomposer (LeanAide)             │
│  - Extract mathematical components     │
│  - Identify dependencies               │
│  - Estimate complexity                 │
│  - Generate Lean code stubs            │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  LeanSubProblemDecomposer              │
│  - Create Lean-friendly sub-problems   │
│  - Add evolutionary metadata           │
│  - Generate verification tickets       │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  EvolutionaryStrategySuggestor         │
│  - Analyze proof difficulty            │
│  - Suggest evolutionary approach       │
│  - Configure evolution parameters      │
└───────────────────────────────────────┘
        ↓
Enhanced Sub-Problems with:
- Mathematical metadata
- Lean code stubs
- Evolutionary configuration
- CrewAI tickets
- ROMA integration hooks
```

## Key Components

### 1. LeanMathematicalDetector

Detects and classifies mathematical problems.

**Key Methods:**
- `detect_mathematical_problem()`: Main detection method
- `_is_mathematical()`: Checks for mathematical content
- `_classify_problem_type()`: Classifies theorem, lemma, definition, etc.
- `_identify_domain()`: Identifies mathematical domain (algebra, analysis, etc.)
- `_estimate_proof_difficulty()`: Estimates proof difficulty (1-10)
- `_estimate_formalization_complexity()`: Estimates Lean formalization complexity
- `_suggest_evolutionary_strategy()`: Recommends evolutionary approach

**Detection Criteria:**
- Proof keywords: "prove", "theorem", "lemma", "show", etc.
- Mathematical symbols: ∀, ∃, →, ∈, ⊂, etc.
- Domain keywords: "group", "limit", "topology", "prime", etc.
- Mathematical notation: LaTeX patterns, etc.

### 2. LeanEnhancedDecompositionEngine

Main engine that extends `DecompositionEngine` with LeanAide integration.

**Key Methods:**
- `decompose_with_leanaide()`: Main decomposition with Lean routing
- `_decompose_mathematical_problem()`: Lean-specific decomposition
- `detect_and_route()`: Automatic routing based on problem type

**Features:**
- Automatic mathematical problem detection
- Seamless routing to LeanAide or standard decomposition
- Lean-friendly sub-problem generation
- Evolutionary configuration integration

### 3. LeanSubProblemDecomposer

Creates Lean-formalizable sub-problems.

**Key Methods:**
- `decompose_mathematical_subproblem()`: Decompose into Lean components
- `_decompose_with_leanaide()`: Use LeanAide decomposition engine
- `_decompose_heuristic()`: Fallback heuristic decomposition

**Outputs:**
- `LeanEnhancedSubProblem` objects with:
  - Base sub-problem data
  - Mathematical metadata
  - Lean code stubs
  - Evolutionary configuration
  - Verification tickets

### 4. EvolutionaryStrategySuggestor

Recommends evolutionary proof generation strategies.

**Strategies:**
- **Standard Evolution**: Genetic algorithm for moderate difficulty
- **Adversarial Evolution**: Red team vs blue team for complex proofs
- **Self-Play**: Reinforcement learning style for interactive problems
- **Hill Climbing**: Iterative refinement for simple improvements
- **Simulated Annealing**: Temperature-based search for local optima
- **Hybrid Evolutionary**: Multi-phase approach for very difficult problems

**Configuration:**
- Population size: 20-50 (based on complexity)
- Max generations: 30-100 (based on difficulty)
- Mutation rate: 0.1-0.3 (based on complexity)
- Crossover rate: 0.8 (default)
- Selection: Tournament (default)
- Crossover: Uniform (default)

## Integration Points

### 1. OpenEvolve Client Integration

The system integrates with the OpenEvolve client for LLM-based analysis:

```python
from openevolve_client import OpenEvolveClient

# Initialize with LeanAide enabled
client = OpenEvolveClient()
engine = LeanEnhancedDecompositionEngine(
    enable_lean_detection=True,
    enable_evolution=True
)
```

### 2. ROMA Integration

ROMA (Recursive Optimizable Modular Architecture) integration for recursive decomposition:

```yaml
roma_integration:
  enabled: true
  max_recursion_depth: 3
  min_complexity_for_recursion: 7
  knowledge_extraction:
    enabled: true
    max_similar_proofs: 5
```

**Usage:**
- Automatic recursive decomposition for complex components
- Knowledge extraction from similar proofs
- Pattern learning from previous formalizations

### 3. CrewAI Integration

CrewAI ticket creation for tracking Lean formalization:

```python
# Tickets auto-created for Lean sub-problems
ticket = {
    "type": "lean_formalization",
    "component_id": component.component_id,
    "priority": calculate_priority(component),
    "metadata": {
        "lean_code": component.lean_code,
        "complexity": component.complexity,
        "domain": component.domain
    }
}
```

### 4. LeanAide Evolution Integration

Direct integration with LeanAide evolutionary proof generator:

```python
from leanaide_evolution import LeanProofEvolutionEngine

# Configure evolution based on problem metadata
config = generate_evolutionary_config(math_metadata)
evolution_engine = LeanProofEvolutionEngine(
    theorem=problem_statement,
    **config
)

# Run evolution
result = await evolution_engine.evolve()
```

## Configuration

The system is configured via `decomposition_config_lean.yaml`:

```yaml
leanaide:
  enabled: true
  client:
    server_url: "http://localhost:7654"
    timeout: 300

evolutionary:
  enabled: true
  min_difficulty: 7
  default_params:
    population_size: 20
    max_generations: 50
    mutation_rate: 0.1

mathematical_domains:
  complexity_multipliers:
    logic: 1.3
    analysis: 1.5
    algebra: 1.2
```

## Usage Examples

### Example 1: Basic Mathematical Problem Detection

```python
from decomposition_engine_lean_enhanced import (
    detect_and_route_mathematical_problem,
    LeanEnhancedDecompositionEngine
)
from sovereign_data_models import ProblemDefinition

# Create problem
problem = ProblemDefinition(
    title="Infinite Primes",
    description="Prove that there are infinitely many prime numbers.",
    # ... other fields
)

# Detect and decompose
plan, math_metadata = await detect_and_route_mathematical_problem(problem)

if math_metadata.is_mathematical:
    print(f"Domain: {math_metadata.domain}")
    print(f"Difficulty: {math_metadata.proof_difficulty}/10")
    print(f"Evolutionary Strategy: {math_metadata.recommended_evolutionary_strategy}")
```

### Example 2: Lean-Enhanced Decomposition

```python
# Create engine
engine = LeanEnhancedDecompositionEngine(
    enable_lean_detection=True,
    enable_evolution=True
)

# Decompose with LeanAide routing
plan = await engine.decompose_with_leanaide(problem)

# Access enhanced sub-problems
for sub_problem in plan.sub_problems:
    if sub_problem.metadata.get("lean_formalization"):
        print(f"Lean Component: {sub_problem.title}")
        print(f"  Domain: {sub_problem.mathematical_domain}")
        print(f"  Complexity: {sub_problem.complexity_score.overall_complexity}/10")
```

### Example 3: Evolutionary Configuration

```python
from decomposition_engine_lean_enhanced import generate_evolutionary_config

# Generate evolutionary config
config = await generate_evolutionary_config(math_metadata)

# Apply to LeanAide evolution engine
from leanaide_evolution import LeanProofEvolutionEngine

evolution_engine = LeanProofEvolutionEngine(
    theorem=problem_description,
    **config
)

result = await evolution_engine.evolve()
```

### Example 4: Integration with Workflow

```python
from workflow_structures import SubProblem
from decomposition_engine_lean_enhanced import LeanEnhancedSubProblem

# Convert to workflow sub-problem
lean_enhanced_sp = LeanEnhancedSubProblem(
    base_subproblem=workflow_subproblem,
    mathematical_metadata=math_metadata,
    lean_code_stub=lean_code,
    evolutionary_config=evolution_config
)

# Get standard SubProblem for workflow
workflow_sp = lean_enhanced_sp.to_subproblem()
```

## Mathematical Domains

The system supports the following mathematical domains:

| Domain | Complexity | Typical Tactics | Example Problems |
|--------|-----------|-----------------|------------------|
| **Algebra** | 1.2x | simp, rw, ring, linarith | Group theory, ring theory, linear algebra |
| **Analysis** | 1.5x | continuity, tendsto, filter | Limits, derivatives, integrals |
| **Topology** | 1.3x | isOpen, isClosed, compactness | Topological spaces, continuity |
| **Number Theory** | 1.0x | nat_dvd, prime, simp | Primes, divisibility, modular arithmetic |
| **Combinatorics** | 1.1x | card, finset, simp | Graphs, counting, bijections |
| **Geometry** | 1.2x | angle, distance, simp | Triangles, circles, polygons |
| **Logic** | 1.3x | apply, exact, intro | Propositional logic, quantifiers |
| **Set Theory** | 1.4x | set, mem, apply | Sets, functions, cardinality |

## Evolutionary Strategies

### Strategy Selection Guide

| Proof Difficulty | Formalization Complexity | Recommended Strategy |
|------------------|-------------------------|---------------------|
| 1-4 | 1-4 | None (direct proof) |
| 5-6 | 5-6 | Standard Evolution |
| 7-8 | 5-6 | Standard Evolution / Self-Play |
| 7-8 | 7-8 | Hybrid Evolutionary |
| 9-10 | 5-6 | Adversarial Evolution |
| 9-10 | 7-10 | Adversarial Evolution / Hybrid |

### Strategy Details

**Standard Evolution:**
- Best for: Moderate difficulty proofs (5-7/10)
- Population: 20-30
- Generations: 50
- Mutation: 0.1
- Selection: Tournament

**Adversarial Evolution:**
- Best for: Very difficult proofs (9-10/10)
- Red team: Generates proof attempts
- Blue team: Generates counter-examples
- Population: 30 (split 15/15)
- Generations: 75

**Self-Play:**
- Best for: Interactive domains (logic, set theory)
- Episodes: 100
- Opponent pool: 20
- Win threshold: 0.7

**Hill Climbing:**
- Best for: Refining near-complete proofs
- Population: 10
- Generations: 30
- Step size: 0.5

**Simulated Annealing:**
- Best for: Escaping local optima
- Temperature: 100.0 → 0.1
- Cooling rate: 0.95
- Population: 15

**Hybrid Evolutionary:**
- Best for: Very complex multi-stage proofs
- Phase 1: Standard evolution (30 generations)
- Phase 2: Adversarial (20 generations)
- Phase 3: Hill climbing (10 generations)

## Proof Difficulty Estimation

The system estimates proof difficulty based on:

1. **Problem Type** (base difficulty):
   - Definition formalization: 3
   - Exercise solution: 4
   - Lemma proof: 5
   - Theorem proof: 6
   - Construction problem: 7
   - Conjecture investigation: 9

2. **Length Modifiers**:
   - > 1000 chars: +2
   - > 500 chars: +1

3. **Complexity Keywords**:
   - "infinite", "uncountable": +1
   - "transfinite", "axiom of choice": +2
   - "non-constructive": +1

4. **Advanced Techniques**:
   - Induction, recursion: +1
   - Diagonal argument: +2
   - Compactness, contradiction: +1

## Formalization Complexity Estimation

The system estimates Lean formalization complexity based on:

1. **Domain Base Scores**:
   - Analysis: 9 (very complex definitions)
   - Set Theory: 8 (foundational complexity)
   - Topology: 8 (abstract structures)
   - Logic: 7 (formal systems)
   - Algebra: 7 (algebraic structures)
   - Geometry: 6 (geometric reasoning)
   - Combinatorics: 6 (finite structures)
   - Number Theory: 6 (arithmetic)

2. **Problem Type Modifiers**:
   - Definition: -2 (definitions are easier than proofs)
   - Theorem proof: +1
   - Conjecture: +3 (very hard to formalize)

3. **Dependency Overhead**:
   - +0.2 per dependency

## Sub-Problem Enhancement

Lean-enhanced sub-problems include:

### Base Fields (from SubProblem):
- id, parent_id, title, description
- type, complexity_score
- dependencies, success_criteria
- priority, estimated_effort

### Lean-Specific Fields:
- `mathematical_metadata`: MathematicalProblemMetadata
- `lean_code_stub`: Generated Lean 4 code
- `evolutionary_config`: Evolution strategy configuration
- `verification_ticket`: CrewAI ticket ID
- `formalization_status`: Current formalization status

### Metadata:
```python
{
    "lean_formalization": True,
    "mathematical_type": "theorem",
    "proof_difficulty": 7,
    "formalization_complexity": 8,
    "evolutionary_strategy": "adversarial_evolution",
    "domain": "number_theory",
    "suggested_tactics": ["simp", "rw", "linarith"],
    "imports": ["Mathlib.Data.Nat.Prime"]
}
```

## Testing and Validation

### Unit Tests

```python
# Test mathematical detection
def test_mathematical_detection():
    detector = LeanMathematicalDetector()
    metadata = detector.detect_mathematical_problem(
        "Prove that there are infinitely many primes."
    )
    assert metadata.is_mathematical
    assert metadata.domain == MathematicalDomain.NUMBER_THEORY
    assert metadata.proof_difficulty >= 6
```

### Integration Tests

```python
# Test end-to-end decomposition
async def test_lean_decomposition():
    engine = LeanEnhancedDecompositionEngine()
    plan = await engine.decompose_with_leanaide(problem)
    assert len(plan.sub_problems) > 0
    assert plan.metadata["lean_decomposition"] == True
```

### Validation Tests

```python
# Test evolutionary config generation
async def test_evolutionary_config():
    config = await generate_evolutionary_config(math_metadata)
    assert config["enable_evolution"] == True
    assert config["strategy_type"] in [
        "standard_evolution",
        "adversarial_evolution",
        "self_play"
    ]
```

## Performance Considerations

### Caching

The system uses aggressive caching to avoid redundant computations:

- Decomposition results cached by problem hash
- Lean code generation cached by component
- Dependency analysis cached by component set

### Parallel Processing

- Component extraction: Parallel
- Dependency analysis: Sequential (requires global view)
- Complexity estimation: Parallel
- Lean code generation: Parallel

### Timeouts

Default timeouts (configurable):

- Decomposition: 60s
- Component extraction: 30s
- Dependency analysis: 30s
- Complexity estimation: 10s

## Troubleshooting

### Issue: Mathematical problems not detected

**Solution:**
- Check detection thresholds in config
- Lower `confidence_threshold`
- Add domain-specific keywords

### Issue: LeanAide decomposition fails

**Solution:**
- System falls back to heuristic decomposition automatically
- Check LeanAide server is running
- Verify server URL in config

### Issue: Evolutionary config not generated

**Solution:**
- Verify `enable_evolution: true` in config
- Check problem difficulty exceeds thresholds
- Review strategy selection logic

### Issue: Lean code generation fails

**Solution:**
- Lean code stubs are optional (set `include_code_stubs: false`)
- Check LeanAide client configuration
- Verify imports are available in Mathlib

## Future Enhancements

1. **Interactive Proof Development**
   - Real-time proof feedback
   - Incremental formalization
   - User-guided decomposition

2. **Learning from Previous Proofs**
   - Pattern extraction from successful proofs
   - Automatic tactic selection
   - Proof strategy recommendation

3. **Collaborative Formalization**
   - Multi-user proof development
   - Distributed evolution
   - Proof merging

4. **Advanced Automation**
   - Auto-tactic selection
   - Proof repair
   - Counter-example generation

## References

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [LeanAide Integration Guide](./leanaide_integration.md)
- [ROMA Architecture](./ROMA_ARCHITECTURE.md)
- [CrewAI Ticket System](./crewai_integration.md)
