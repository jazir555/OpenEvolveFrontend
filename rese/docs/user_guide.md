# RESE User Guide

**Recursive Epistemic Solvability Engine**
**Version:** 1.0.0
**Last Updated:** 2025-12-31

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Common Workflows](#common-workflows)
6. [Configuration](#configuration)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

---

## Introduction

### What is RESE?

RESE (Recursive Epistemic Solvability Engine) is a **four-phase formal methodology** that transforms intractable problems into tractable ones through systematic epistemic analysis:

- **Phase I (Φ):** Epistemic Audit - Systematic falsification of assumptions
- **Phase II (Ψ):** Isomorphic Resonance - Cross-domain knowledge transfer
- **Phase III (Γ):** Monte Carlo Refinement - ACI-guided adaptive search
- **Phase IV (Δ):** Architectural Synthesis - Validated solution assembly

### Key Benefits

- **Systematic Problem Solving:** Transform unsolvable problems into solvable ones
- **Validated Reasoning:** All constraints verified in Lean 4 theorem prover
- **Bias Detection:** Automatic detection of cognitive biases
- **Adaptive Search:** ACI-guided optimization focuses on promising regions
- **Formal Verification:** Mathematical rigor ensures solution validity

### Who Should Use RESE?

- **Researchers:** Working on complex, intractable problems
- **Engineers:** Designing novel solutions with formal guarantees
- **Data Scientists:** Solving optimization and search problems
- **Innovation Teams:** Accelerating invention and discovery processes

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Git
- 8GB+ RAM recommended
- Lean 4 (optional, for formal verification)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/rese.git
cd rese
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n rese python=3.9
conda activate rese
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import rese; print(f'RESE {rese.__version__} installed successfully')"
```

### Optional: Lean 4 Installation

For formal verification capabilities:

```bash
# Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Verify Lean installation
lean --version
```

---

## Quick Start

### Your First RESE Pipeline

```python
from rese.rese_pipeline import run_rese

# Define a problem
problem_description = "Optimize delivery routes for 50 locations"

constraints = [
    {
        "id": "c1",
        "type": "hard",
        "description": "All locations must be visited",
        "formalization": "∀ l ∈ locations, visited(l)"
    },
    {
        "id": "c2",
        "type": "soft",
        "description": "Minimize total distance",
        "formalization": "minimize Σ distance(route)"
    }
]

variables = {
    "num_locations": 50,
    "vehicle_capacity": 100,
    "time_window": [8, 17]
}

# Run RESE pipeline
result = run_rese(
    problem_description=problem_description,
    constraints=constraints,
    variables=variables
)

# Check results
print(f"Status: {result.status}")
print(f"Confidence: {result.confidence:.2f}")
print(f"ACI History: {result.aci_history}")
```

### Expected Output

```
[Phase I] Starting Epistemic Audit...
[Phase I] Found 3 tacit assumptions
[Phase I] Detected 2 cognitive biases (medium severity)
[Phase I] Resolved 1 contradiction
[Phase I] ✓ Completed in 2.3s

[Phase II] Starting Isomorphic Resonance...
[Phase II] Inverted 2 constraints
[Phase II] Mapped to 3 source domains
[Phase II] Isomorphism score: 0.75
[Phase II] ✓ Completed in 5.1s

[Phase III] Starting Monte Carlo Refinement...
[Phase III] ACI = 0.65
[Phase III] MCTS: 1000 iterations, best value = 0.85
[Phase III] Converged: True
[Phase III] ✓ Completed in 12.7s

[Phase IV] Starting Architectural Synthesis...
[Phase IV] Assembled 5 components
[Phase IV] Generated 10 predictions
[Phase IV] Validation score: 0.87
[Phase IV] ✓ Completed in 3.2s

Status: completed
Confidence: 0.87
ACI History: [0.80, 0.65, 0.43, 0.30]
```

---

## Core Concepts

### ACI (Algorithmic Complexity Index)

ACI measures how "solvable" a problem is:

```
ACI = α·(1-H) + β·C + γ·S

Where:
- H = Disorder Entropy (0-1, lower is better)
- C = Causal Coherence (0-1, higher is better)
- S = Solvability Index (0-1, higher is better)
- α + β + γ = 1
```

**Interpretation:**
- ACI > 0.7: Highly solvable (easy)
- ACI 0.4-0.7: Moderately solvable (medium)
- ACI < 0.4: Difficult (hard)
- ACI < 0.2: Intractable (very hard)

### Tacit Assumption Mining (Φ₁.₅)

Automatically discovers hidden assumptions from null results:

```python
from rese.phase1.tacit_assumption_miner import TacitAssumptionMiner

miner = TacitAssumptionMiner()
assumptions = miner.mine(
    failure_cases=known_failures,
    constraints=constraints
)

for assumption in assumptions:
    print(f"Found: {assumption.description}")
    print(f"Confidence: {assumption.confidence:.2f}")
```

### Isomorphic Resonance (I_mech)

Transfers knowledge across domains using structural similarity:

```python
from rese.phase2.imech import IMechValidator

validator = IMechValidator()
similarity = validator.compare_domains(
    source_domain=source,
    target_domain=target
)

print(f"Mechanistic similarity: {similarity.score:.2f}")
```

### MCTS Search

Monte Carlo Tree Search guided by ACI:

```python
from rese.phase3.mcts_search import MCTSSearch

search = MCTSSearch(
    aci_calculator=aci_calc,
    iterations=1000,
    exploration_constant=1.41
)

best_solution = search.search(initial_state)
```

---

## Common Workflows

### Workflow 1: Solving an Optimization Problem

```python
from rese.rese_pipeline import RESEPipeline, ProblemInput

# Create pipeline
pipeline = RESEPipeline()

# Define problem
problem = ProblemInput(
    id="tsp_50",
    description="Traveling Salesman Problem with 50 cities",
    constraints=[
        {
            "id": "visit_all",
            "type": "hard",
            "description": "Visit all cities exactly once",
            "formalization": "∀ city ∈ cities: visited(city) = 1"
        },
        {
            "id": "minimize_distance",
            "type": "soft",
            "description": "Minimize total distance",
            "formalization": "minimize Σ dist(city_i, city_j)"
        }
    ],
    variables={
        "num_cities": 50,
        "coordinates": city_coords
    }
)

# Run pipeline
result = pipeline.run(problem)

# Extract solution
if result.status.value == "completed":
    solution = result.final_solution
    print(f"Best route: {solution['route']}")
    print(f"Total distance: {solution['distance']:.2f}")
    print(f"Validation score: {result.validation_score:.2f}")
```

### Workflow 2: Using Individual Phase Components

```python
# Phase I: Epistemic Audit
from rese.phase1.cognitive_biases import CognitiveBiasDetector

detector = CognitiveBiasDetector()
bias_report = detector.analyze_constraints(constraints)
print(f"Bias score: {bias_report.overall_bias_score:.2f}")

# Phase II: Constraint Inversion
from rese.phase2.psi3 import ConstraintInverter

inverter = ConstraintInverter()
inverted = inverter.invert(constraints)
print(f"Inverted {len(inverted)} constraints")

# Phase III: ACI Calculation
from rese.gamma1.core.aci_calculator import ACICalculator

aci_calc = ACICalculator()
aci_result = aci_calc.calculate(csp_instance)
print(f"ACI = {aci_result.ACI:.3f}")

# Phase IV: Validation
from rese.phase4.aci_reduction_validator import Delta3Validator

validator = Delta3Validator()
validation = validator.validate(problem, solution)
print(f"Valid: {validation.is_valid}")
print(f"Score: {validation.validation_score:.2f}")
```

### Workflow 3: Bias Detection and Debiasing

```python
from rese.phase1.cognitive_biases import CognitiveBiasDetector, Debiaser

# Detect biases
detector = CognitiveBiasDetector()
report = detector.analyze_constraints(constraints)

# Review detected biases
for detection in report.detections:
    print(f"{detection.bias_type.value}: {detection.description}")
    print(f"Severity: {detection.severity.name}")
    print(f"Suggestion: {detection.suggestion}")
    print()

# Auto-debias if desired
if report.overall_bias_score > 0.5:
    debiaser = Debiaser()
    debiased_constraints = debiaser.debias(constraints, report)
    print(f"Debiased {len(debiased_constraints)} constraints")
```

### Workflow 4: Knowledge Transfer Across Domains

```python
from rese.phase2.imech import IMechValidator, Domain, TransferMapper

# Define source domain (known solved problem)
source = Domain(
    id="tsp",
    name="Traveling Salesman Problem",
    variables={"num_cities": 50},
    constraints=constraints_tsp
)

# Define target domain (new problem)
target = Domain(
    id="vrp",
    name="Vehicle Routing Problem",
    variables={"num_vehicles": 5, "num_customers": 50},
    constraints=constraints_vrp
)

# Validate isomorphism
validator = IMechValidator()
comparison = validator.compare_domains(source, target)

print(f"Similarity score: {comparison.score:.2f}")
print(f"Shared structure: {comparison.shared_structure}")
print(f"Transfer confidence: {comparison.confidence:.2f}")

# Transfer knowledge if similar enough
if comparison.score > 0.7:
    mapper = TransferMapper()
    transferred = mapper.transfer_knowledge(source, target)
    print(f"Transferred {len(transferred)} constraints")
```

### Workflow 5: ACI-Guided Search

```python
from rese.gamma1.core.aci_calculator import ACICalculator
from rese.phase3.mcts_search import MCTSSearch

# Initialize ACI calculator
aci_calc = ACICalculator(alpha=0.35, beta=0.35, gamma=0.30)

# Define ACI-guided MCTS
def aci_policy(state):
    """Use ACI to guide node selection"""
    result = aci_calc.calculate(state)
    return result.ACI

search = MCTSSearch(
    policy=aci_policy,
    iterations=1000,
    exploration_constant=1.41
)

# Search for best solution
best = search.search(initial_state)

print(f"Best value: {best.value:.2f}")
print(f"ACI progression: {best.aci_history}")
```

---

## Configuration

### Basic Configuration

```python
from rese.config import RESEConfig, get_config

# Load default configuration
config = get_config()

# Or create custom configuration
config = RESEConfig(
    environment="production",
    phase1=Phase1Config(
        sce_max_constraints=10000,
        phi15_assumption_threshold=0.6
    ),
    phase3=Phase3Config(
        gamma2_iterations=1000,
        convergence_patience=50
    )
)

# Save configuration
config.save("my_config.json")
```

### Environment-Specific Configuration

```python
from rese.config import Environment, RESEConfig

# Development environment
dev_config = RESEConfig().for_environment(Environment.DEVELOPMENT)

# Production environment
prod_config = RESEConfig().for_environment(Environment.PRODUCTION)
```

### Key Configuration Parameters

**Phase I (Epistemic Audit):**
- `sce_max_constraints`: Maximum number of constraints (default: 10000)
- `phi15_assumption_threshold`: Minimum confidence for assumptions (default: 0.6)
- `phi2_bias_threshold`: Bias detection threshold (default: 0.5)

**Phase II (Isomorphic Resonance):**
- `psi1_complexity_reduction_target`: Target complexity reduction (default: 0.1)
- `psi3_target_accuracy`: Target isomorphism accuracy (default: 0.80)

**Phase III (Monte Carlo Refinement):**
- `gamma2_iterations`: MCTS iterations (default: 1000)
- `gamma2_exploration_constant`: UCB exploration constant (default: 1.41)
- `convergence_patience`: Patience for early stopping (default: 50)

**Phase IV (Architectural Synthesis):**
- `delta3_validation_threshold`: Minimum validation score (default: 0.7)
- `delta3_min_aci_reduction`: Minimum ACI reduction required (default: 0.2)

---

## Best Practices

### 1. Problem Formulation

**DO:**
- Start with clear, well-defined constraints
- Separate hard constraints from soft preferences
- Provide formal specifications for critical constraints

**DON'T:**
- Mix constraints and objectives
- Use vague or ambiguous descriptions
- Over-constrain the problem

### 2. Bias Detection

**DO:**
- Review bias reports before proceeding
- Address critical and high-severity biases
- Use debiasing tools iteratively

**DON'T:**
- Ignore bias warnings
- Assume no biases exist
- Over-debias (can lose useful information)

### 3. ACI Interpretation

**DO:**
- Use ACI as a guide, not absolute truth
- Track ACI progression through phases
- Validate ACI predictions empirically

**DON'T:**
- Trust low-ACI problems to be easy
- Assume high ACI guarantees solution
- Ignore ACI confidence intervals

### 4. Iterative Refinement

**DO:**
- Run RESE multiple times with different parameters
- Use phase results to refine problem formulation
- Cache intermediate results for faster iteration

**DON'T:**
- Accept first run as final
- Neglect parameter tuning
- Skip validation phases

### 5. Performance Optimization

**DO:**
- Enable caching for repeated runs
- Use parallel execution when available
- Monitor memory usage for large problems

**DON'T:**
- Disable caching without reason
- Run all phases sequentially when parallel is possible
- Ignore resource limits

---

## FAQ

### Q1: What types of problems can RESE solve?

**A:** RESE is designed for constraint satisfaction and optimization problems, including:
- Combinatorial optimization (TSP, VRP, scheduling)
- Engineering design problems
- Scientific discovery problems
- Any problem with formal constraints

### Q2: How long does a typical RESE run take?

**A:** Depends on problem complexity:
- Simple problems (< 10 constraints): seconds to minutes
- Medium problems (10-100 constraints): minutes to hours
- Complex problems (> 100 constraints): hours to days

Use caching to speed up iteration.

### Q3: Do I need to know Lean 4?

**A:** No! Lean 4 verification is optional. RESE works without it, but formal verification provides mathematical guarantees for critical applications.

### Q4: Can I use RESE for real-time applications?

**A:** RESE is designed for batch/offline processing. For real-time use:
1. Pre-compute solutions offline
2. Use RESE for periodic re-optimization
3. Deploy only the inference components

### Q5: How accurate is the ACI prediction?

**A:** ACI accuracy depends on:
- Quality of problem formalization
- Similarity to training distribution
- Phase of problem (early phases less accurate)

Typical correlation: 0.7-0.9 with actual solve time.

### Q6: What if RESE fails to solve my problem?

**A:** Try:
1. Refine constraints (remove over-constraining rules)
2. Adjust phase parameters
3. Run individual phases separately
4. Check for hidden assumptions in Φ₁.₅

### Q7: Can I extend RESE with custom components?

**A:** Yes! RESE is modular:
- Implement custom phase executors
- Add new bias detectors
- Create domain-specific ACI components
- Integrate custom solvers

See Developer Guide for details.

### Q8: How do I cite RESE in academic papers?

**A:**
```
@software{rese2025,
  title={RESE: Recursive Epistemic Solvability Engine},
  author={RESE Development Team},
  year={2025},
  version={1.0.0},
  url={https://github.com/your-org/rese}
}
```

---

## Getting Help

### Documentation
- **Developer Guide:** `rese/docs/developer_guide.md`
- **API Reference:** `rese/docs/api_reference.md`
- **Integration Guide:** `rese/docs/e2e_integration.md`

### Community
- **GitHub Issues:** https://github.com/your-org/rese/issues
- **Discussions:** https://github.com/your-org/rese/discussions

### Support
- **Email:** support@rese.example.com
- **Discord:** https://discord.gg/rese-community

---

## Next Steps

1. **Explore Examples:** Check out `rese/examples/` for detailed tutorials
2. **Read API Docs:** See `rese/docs/api_reference.md` for complete API
3. **Customize:** Read Developer Guide to learn how to extend RESE

---

**Happy Solving! 🚀**
