# I_mech: Quick Start Guide

**For:** Implementation Team
**Week:** 31 (Implementation)
**Status:** Ready to Code

---

## Overview

I_mech detects mechanistic isomorphisms between domains and transfers solutions with >80% success.

**One-Minute Summary:**
- Extract FDGs (Functional Dependency Graphs) from domains
- Compare using WL algorithm + VF2 + causal analysis
- Score similarity (0-1 scale)
- Generate Lean 4 proof (optional)
- Transfer solution if similarity > 0.7

---

## Installation

```bash
# Navigate to project
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Create research directories
mkdir -p rese/imech/{core,algorithms,transfer,lean4,theories,utils}
mkdir -p rese/tests/{benchmarks}

# Install dependencies
pip install networkx>=3.0 numpy dowhy>=0.11 pgmpy pytest

# Install Lean 4 (for proof verification)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

---

## Basic Usage

```python
from rese.imech import IMech
from rese.imech.core.domain import Domain

# Initialize I_mech
imech = IMech(
    use_exact_isomorphism=False,  # Use VF2 for exact matching
    enable_proofs=True,           # Generate Lean 4 proofs
    cache_enabled=True            # Cache results
)

# Create domains
domain1 = Domain(
    id="steam_engine",
    name="Steam Engine",
    description="External combustion using steam expansion",
    formal_constraints=["PV=nRT", "W = ∫P dV"],
    historical_data=steam_engine_data
)

domain2 = Domain(
    id="ic_engine",
    name="Internal Combustion Engine",
    description="Internal combustion using fuel explosion",
    formal_constraints=["PV=nRT", "W = ∫P dV"],
    historical_data=ic_engine_data
)

# Compare domains
result = imech.compare(domain1, domain2)

# Check results
if result.is_above_threshold(0.7):
    print(f"Similarity: {result.total_score:.2f}")
    print(f"Mapping: {result.node_mapping}")
    print(f"Proof verified: {result.proof_verified}")
    print(f"Transferred solution: {result.transferred_solution}")
else:
    print("Not sufficiently similar for transfer")
```

---

## Core Classes

### 1. FunctionalDependencyGraph

```python
from rese.imech.core.fdg import (
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType
)

# Create FDG
fdg = FunctionalDependencyGraph()

# Add nodes
node = Node(
    id="pressure",
    variable="P",
    constraint_type="continuous",
    metadata={"unit": "Pa", "range": (0, 1e6)}
)
fdg.add_node(node)

# Add edges
edge = Edge(
    source="temperature",
    target="pressure",
    edge_type=EdgeType.CAUSAL,
    weight=1.0
)
fdg.add_edge(edge)

# Extract causal subgraph
causal_graph = fdg.get_causal_subgraph()

# Detect feedback loops
loops = fdg.get_feedback_loops()
```

### 2. IsomorphismDetector

```python
from rese.imech.core.isomorphism import IsomorphismDetector

detector = IsomorphismDetector(use_exact=True)

# Detect similarity
score, mapping = detector.detect_similarity(fdg1, fdg2)

print(f"Structural similarity: {score:.2f}")
print(f"Node mapping: {mapping}")
```

### 3. IMech (Main Interface)

```python
from rese.imech import IMech

imech = IMech()
result = imech.compare(domain1, domain2)

# Access results
result.total_score          # Overall similarity (0-1)
result.structural_score     # Graph isomorphism
result.causal_score         # Causal mechanism
result.semantic_score       # Label matching
result.intervention_score   # Interventional equivalence
result.node_mapping         # Isomorphism mapping
result.proof                # Lean 4 proof
result.transferred_solution # Transferred solution
```

---

## Week 31 Implementation Checklist

### Day 1-2: Data Structures (rese/imech/core/)
- [ ] `fdg.py`: FunctionalDependencyGraph class
- [ ] `result.py`: SimilarityResult class
- [ ] `domain.py`: Domain class
- [ ] Tests: `test_fdg.py`

**Key Methods:**
```python
class FunctionalDependencyGraph:
    def add_node(node: Node) -> None
    def add_edge(edge: Edge) -> None
    def get_causal_subgraph() -> nx.DiGraph
    def get_feedback_loops() -> List[List[str]]
```

### Day 3-4: Isomorphism (rese/imech/core/algorithms/)
- [ ] `weisfeiler_lehman.py`: WL color refinement
- [ ] `vf2.py`: VF2 exact isomorphism
- [ ] `subgraph.py`: Subgraph isomorphism
- [ ] Tests: `test_isomorphism.py`

**Key Methods:**
```python
def weisfeiler_lehman(G1, G2, max_iter=10) -> float:
    """Compute structural similarity score"""

def vf2_isomorphism(G1, G2) -> Optional[Dict]:
    """Find exact isomorphism mapping"""

def subgraph_isomorphism(G1, G2) -> Tuple[Dict, float]:
    """Find best subgraph isomorphism"""
```

### Day 5: Causality (rese/imech/core/)
- [ ] `causality.py`: CausalSimilarityAnalyzer
- [ ] Integration with DoWhy
- [ ] Tests: `test_causality.py`

**Key Methods:**
```python
class CausalSimilarityAnalyzer:
    def analyze(fdg1, fdg2, mapping) -> float:
        """Compute causal similarity score"""

    def compare_interventions(fdg1, fdg2, mapping) -> float:
        """Compare intervention responses"""
```

### Day 6: Scoring & Transfer (rese/imech/core/)
- [ ] `scoring.py`: SimilarityScorer
- [ ] `transfer/mapper.py`: SolutionMapper
- [ ] `transfer/validator.py`: SolutionValidator
- [ ] Tests: `test_transfer.py`

**Key Methods:**
```python
class SimilarityScorer:
    def compute_total_score(struct, causal, semantic, intervention) -> float:
        """Combine scores with weights"""

class SolutionMapper:
    def transfer(solution, mapping, domain1, domain2) -> Solution:
        """Transfer solution between domains"""

class SolutionValidator:
    def validate(solution, domain, tolerance=0.1) -> ValidationResult:
        """Check if solution satisfies constraints"""
```

### Day 7: Proofs & Integration (rese/imech/lean4/)
- [ ] `lean4/generator.py`: ProofGenerator
- [ ] `lean4/verifier.py`: ProofVerifier
- [ ] `__init__.py`: IMech main class
- [ ] Integration with Stage 4
- [ ] Tests: `test_proof.py`, `test_integration.py`

**Key Methods:**
```python
class ProofGenerator:
    def generate(fdg1, fdg2, mapping) -> str:
        """Generate Lean 4 proof script"""

    def verify(proof_script) -> Tuple[bool, str]:
        """Verify proof using Lean 4"""

class IMech:
    def compare(domain1, domain2) -> SimilarityResult:
        """Main interface: compare two domains"""
```

---

## Testing

### Run Unit Tests
```bash
# All tests
pytest rese/tests/

# Specific module
pytest rese/tests/test_fdg.py

# With coverage
pytest --cov=rese/imech rese/tests/

# With benchmarking
pytest -m benchmark rese/tests/benchmarks/
```

### Run Integration Tests
```bash
pytest rese/tests/test_integration.py -v
```

### Quick Validation
```python
# Simple test case
from rese.tests.utils import create_test_domain

domain1 = create_test_domain(
    constraints=["x + y = 10"],
    solution={"x": 5, "y": 5}
)

domain2 = create_test_domain(
    constraints=["a + b = 10"],
    solution=None  # Expect transfer
)

imech = IMech()
result = imech.compare(domain1, domain2)

assert result.total_score > 0.7  # Should detect isomorphism
assert result.transferred_solution is not None
```

---

## Configuration

Edit `rese/imech/config.py`:

```python
IMechConfig = {
    # Scoring weights
    'weight_structural': 0.3,
    'weight_causal': 0.3,
    'weight_semantic': 0.2,
    'weight_intervention': 0.2,

    # Thresholds
    'structural_threshold': 0.3,
    'mechanistic_threshold': 0.7,

    # Performance
    'cache_enabled': True,
    'max_wl_iterations': 10,

    # Proofs
    'enable_proofs': True,
    'proof_threshold': 0.7,
}
```

---

## Common Issues

### Issue 1: DoWhy Import Error
```bash
# Fix: Install causal-learn
pip install causal-learn
```

### Issue 2: Lean 4 Not Found
```bash
# Fix: Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.profile
```

### Issue 3: Slow Performance
```python
# Fix: Enable caching
imech = IMech(cache_enabled=True)

# Fix: Disable exact isomorphism for large graphs
imech = IMech(use_exact_isomorphism=False)
```

### Issue 4: Proof Verification Failing
```python
# Fix: Disable proofs temporarily
imech = IMech(enable_proofs=False)

# Or adjust threshold
result = imech.compare(domain1, domain2)
if result.total_score > 0.8:  # Higher threshold for proofs
    proof = imech.proof_generator.generate(...)
```

---

## Performance Targets

| Domain Size | Target Time |
|-------------|-------------|
| Small (<100 nodes) | < 2s |
| Medium (100-1000) | < 20s |
| Large (>1000) | < 120s |

**Profiling:**
```bash
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumtime').print_stats(20)"
```

---

## Debugging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

imech = IMech()
result = imech.compare(domain1, domain2)
```

### Visualize FDGs
```python
import matplotlib.pyplot as plt
import networkx as nx

G = domain.fdg.graph
nx.draw(G, with_labels=True)
plt.savefig("fdg.png")
```

### Inspect Mapping
```python
result = imech.compare(domain1, domain2)

print("Node mapping:")
for src, tgt in result.node_mapping.items():
    print(f"  {src} -> {tgt}")

print("\nScores:")
print(f"  Structural: {result.structural_score:.2f}")
print(f"  Causal: {result.causal_score:.2f}")
print(f"  Semantic: {result.semantic_score:.2f}")
print(f"  Intervention: {result.intervention_score:.2f}")
print(f"  Total: {result.total_score:.2f}")
```

---

## Integration with Stage 4

```python
from rese.stage4.isomorphic_mapping import IsomorphicMappingStage

stage4 = IsomorphicMappingStage()

# Find analogous solution
solution = stage4.find_analogous_solution(target_domain)

if solution:
    print(f"Found analogous solution: {solution}")
else:
    print("No analogous solution found")
```

**Stage 4 Flow:**
1. Ψ₂ (Ontology) → Quick semantic filter
2. I_mech → Detailed mechanistic analysis
3. Combine → Final similarity score
4. Transfer → Solution if score > threshold

---

## Next Steps After Week 31

1. **Week 32**: Validation (see imech_validation_strategy.md)
   - Prepare benchmark datasets
   - Run validation experiments
   - Collect metrics
   - Iterate if needed

2. **Week 33-34**: Integration
   - Deploy to staging
   - User acceptance testing
   - Performance optimization
   - Documentation finalization

3. **Week 35+**: Production
   - Deploy to production
   - Monitor performance
   - Collect feedback
   - Plan extensions

---

## Support

**Documentation:**
- `imech_isomorphism_research.md` - Theoretical foundation
- `imech_algorithm_design.md` - Algorithm details
- `imech_implementation_plan.md` - Complete implementation plan
- `imech_validation_strategy.md` - Validation methodology
- `IMECH_SUMMARY.md` - Comprehensive summary

**Team:**
- Agent G3 (I_mech Specialist) - Lead designer
- Implementation Team - Week 31 coding
- Validation Team - Week 32 testing

**Contact:** See OpenEvolve project documentation

---

**Ready to implement! Start with Day 1-2: Data Structures.**
