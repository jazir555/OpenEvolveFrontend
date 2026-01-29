<<<<<<< HEAD
# RESE Integration Guide

## Table of Contents

1. [Overview](#overview)
2. [Integration Architecture](#integration-architecture)
3. [Stage-by-Stage Integration](#stage-by-stage-integration)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Common Pitfalls](#common-pitfalls)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Topics](#advanced-topics)

---

## Overview

### What is RESE Integration?

RESE (Recursive Epistemic Solvability Engine) enhances the End-to-End Invention System by providing **quantified reasoning validation** at every stage. This guide explains how to integrate RESE components into each E2E stage.

### Integration Philosophy

**Traditional E2E (without RESE):**
```
Stage 1 → Stage 2 → Stage 3 → ... → Stage 9
[Qualitative] [Heuristic] [Best-effort]
```

**E2E + RESE:**
```
Stage 1+RESE → Stage 2+RESE → Stage 3+RESE → ... → Stage 9+RESE
[Validated]  [Quantified]  [Statistical]      [Tracked]
```

### Key Benefits

1. **Quantified Confidence**: Every stage produces statistical confidence metrics
2. **ACI Tracking**: Algorithmic Complexity Index tracked through entire pipeline
3. **Error Elimination**: Φ₁.₅ discovers hidden assumptions before they cause failures
4. **Solution Transfer**: I_mech enables reliable analogy-based solution transfer
5. **Validation**: Δ₃ guarantees ≥20% ACI reduction

---

## Integration Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    E2E + RESE INTEGRATION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Stage 1 │    │ Stage 2 │    │ Stage 3 │    │ Stage 4 │       │
│  │ Prompt  │    │ KG      │    │ Soln    │    │ Formal  │       │
│  │ + SCE   │    │ + I_mech│    │ + MCTS  │    │ + Δ₃    │       │
│  │ + Φ₁.₅  │    │ + Ψ₂    │    │ + Γ₂    │    │         │       │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘       │
│       │              │              │              │              │
│       ▼              ▼              ▼              ▼              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Stage 5 │    │ Stage 6 │    │ Stage 7 │    │ Stage 8 │       │
│  │ Red     │    │ Know    │    │ SOP     │    │ Lab     │       │
│  │ + ACI   │    │ + Φ₁.₅  │    │ + Δ₁    │    │ + Γ₁    │       │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘       │
│       │              │              │              │              │
│       ▼              ▼              ▼              ▼              │
│  ┌─────────┐                                                     │
│  │ Stage 9 │◀───── RESE ACI Tracking Across All Stages          │
│  │ Monitor │                                                     │
│  │ + Γ₁    │                                                     │
│  └─────────┘                                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### RESE Wrapper Pattern

Each stage has a RESE wrapper that:

1. **Pre-processes**: Runs RESE analysis on stage input
2. **Enhances**: Adds RESE capabilities to stage execution
3. **Post-processes**: Validates output with RESE metrics
4. **Tracks**: Updates ACI history and confidence scores

---

## Stage-by-Stage Integration

### Stage 1: Prompt Analysis + RESE

**Location:** `rese/integrations/stage1.py`

#### RESE Components

- **Φ₁: Symbolic Constraint Engine** - Formalizes constraints
- **Φ₁.₅: Tacit Assumption Miner** - Discovers hidden requirements
- **Φ₂: Cognitive Bias Detector** - Identifies biases in prompt

#### Integration Code

```python
from rese.integrations.stage1 import Stage1RESEAnalyzer

# Initialize analyzer
analyzer = Stage1RESEAnalyzer()

# Analyze prompt
result = analyzer.analyze_prompt(
    prompt_text="Design a room-temperature superconductor",
    domain="materials_science"
)

# Access RESE enhancements
print(f"Constraints: {len(result.constraints)}")
print(f"Assumptions: {len(result.assumptions)}")
print(f"Bias Score: {result.bias_score}")
```

#### Data Flow

```
Input Prompt
    │
    ▼
┌───────────────────┐
│ Φ₁: SCE           │ → Formal constraints
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Φ₁.₅: Assumption  │ → Hidden assumptions
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Φ₂: Bias Detection│ → Bias report
└───────────────────┘
    │
    ▼
Enhanced Prompt (with constraints, assumptions, debiasing suggestions)
```

#### Example Output

```python
{
    'original_prompt': "Design a room-temperature superconductor",
    'constraints': [
        {'id': 'c1', 'type': 'hard', 'formalization': 'Tc >= 293 K'},
        {'id': 'c2', 'type': 'hard', 'formalization': 'critical_field >= 1 T'}
    ],
    'assumptions': [
        {'description': 'Material must be solid at room temperature', 'confidence': 0.95},
        {'description': 'Manufacturing cost must be feasible', 'confidence': 0.87},
        {'description': 'Material must be chemically stable', 'confidence': 0.92}
    ],
    'bias_report': {
        'overall_bias_score': 0.23,
        'detections': [
            {'type': 'confirmation_bias', 'severity': 'low'}
        ]
    },
    'refined_prompt': "Design a room-temperature superconductor (Tc >= 293 K) with chemical stability, feasible manufacturing cost, and critical field >= 1 T"
}
```

---

### Stage 2: Knowledge Graph + RESE

**Location:** `rese/integrations/stage2.py`

#### RESE Components

- **Ψ₂: Semantic Ontology Mapper** - Finds similar domains
- **I_mech: Mechanistic Isomorphism** - Validates analogies

#### Integration Code

```python
from rese.integrations.stage2 import Stage2RESEMapper

mapper = Stage2RESEMapper()

# Map to similar domains
mappings = mapper.map_to_domains(
    problem_description=stage1_output['refined_prompt'],
    knowledge_graph=kg,
    similarity_threshold=0.7
)

# Validate isomorphisms
for mapping in mappings:
    isomorphism = mapper.validate_isomorphism(
        source=mapping['source_domain'],
        target=mapping['target_domain']
    )

    if isomorphism.score > 0.8:
        print(f"Validated: {mapping['target_domain']} (score: {isomorphism.score})")
        print(f"Transfer confidence: {isomorphism.confidence}")
```

#### Data Flow

```
Stage 1 Output (Refined Prompt)
    │
    ▼
┌───────────────────┐
│ Ψ₂: Semantic Map  │ → Candidate domains
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ I_mech: Validate  │ → Isomorphic domains
└───────────────────┘
    │
    ▼
Validated Solution Transfer Candidates
```

---

### Stage 3: Solution Generation + RESE

**Location:** `rese/integrations/stage3.py`

#### RESE Components

- **Γ₂: ACI-Guided MCTS** - Optimizes with ACI as reward
- **N_max: Convergence Controller** - Detects when to stop search

#### Integration Code

```python
from rese.integrations.stage3 import Stage3RESEGenerator

generator = Stage3RESEGenerator()

# Generate solution with ACI guidance
solution = generator.generate(
    problem=stage2_output['validated_problem'],
    constraints=stage1_output['constraints'],
    mcts_iterations=5000,
    aci_guided=True
)

print(f"Solution: {solution.variables}")
print(f"ACI: {solution.aci}")
print(f"Converged: {solution.converged}")
```

#### MCTS Configuration

```python
# Configure MCTS for RESE
config = {
    'exploration_constant': 1.41,      # UCB C parameter
    'max_iterations': 5000,            # N_max
    'playout_depth': 100,              # Simulation depth
    'aci_guided': True,                # Use ACI for exploration
    'parallel_agents': 4,              # Parallel search
    'convergence_patience': 50,        # Stop if no improvement
    'convergence_min_delta': 0.001     # Minimum improvement
}
```

---

### Stage 4: Mathematical Formalization + RESE

**Location:** `rese/integrations/stage4.py`

#### RESE Components

- **Δ₃: ACI Reduction Validator** - Validates solution quality
- **Lean 4 Integration** - Requires formal proofs for critical steps

#### Integration Code

```python
from rese.integrations.stage4 import Stage4RESEFormalizer

formalizer = Stage4RESEFormalizer()

# Formalize solution with validation
formalization = formalizer.formalize(
    solution=stage3_output['solution'],
    constraints=stage1_output['constraints'],
    require_proof=True
)

# Validate ACI reduction
validation = formalizer.validate_aci_reduction(
    baseline_aci=stage1_output['initial_aci'],
    final_aci=formalization['aci'],
    min_reduction=0.2
)

if validation.is_valid:
    print(f"Formalization validated (score: {validation.score})")
    print(f"ACI reduction: {validation.aci_reduction * 100:.1f}%")
```

---

### Stage 5: Red Team Analysis + RESE

**Location:** `rese/integrations/stage5.py`

#### RESE Components

- **ACI Quantification** - Measures residual uncertainty
- **Φ₁.₅ Re-run** - Checks for newly introduced assumptions

#### Integration Code

```python
from rese.integrations.stage5 import Stage5RESEAnalyzer

analyzer = Stage5RESEAnalyzer()

# Red team analysis with ACI
red_team_result = analyzer.analyze(
    solution=stage4_output['formalization'],
    attack_vectors=['constraint_violation', 'assumption_failure', 'optimization_gap']
)

print(f"Residual ACI: {red_team_result.aci}")
print(f"Vulnerabilities Found: {len(red_team_result.vulnerabilities)}")
print(f"Mitigated: {red_team_result.vulnerabilities_mitigated}")
```

---

### Stage 6: Knowledge Extraction + RESE

**Location:** `rese/integrations/stage6.py`

#### RESE Components

- **Φ₁.₅ Feedback** - Feeds discovered assumptions back to database
- **I_mech Pattern Mining** - Extracts isomorphic patterns

#### Integration Code

```python
from rese.integrations.stage6 import Stage6RESEExtractor

extractor = Stage6RESEExtractor()

# Extract knowledge with RESE
knowledge = extractor.extract(
    execution_result=stage5_output,
    mine_assumptions=True,
    mine_isomorphisms=True
)

# Knowledge added to database:
# - Failed assumptions (for future Φ₁.₅ runs)
# - Validated isomorphisms (for future I_mech runs)
# - ACI reduction patterns (for prediction)
```

---

### Stage 7: SOP Generation + RESE

**Location:** `rese/integrations/stage7.py`

#### RESE Components

- **Δ₁: Architecture Assembly** - Assembles turnkey components
- **Confidence Annotation** - Adds confidence metrics to SOPs

#### Integration Code

```python
from rese.integrations.stage7 import Stage7RESEGenerator

generator = Stage7RESEGenerator()

# Generate SOP with confidence
sop = generator.generate_sop(
    solution=stage6_output['validated_solution'],
    confidence_threshold=0.7,
    include_contingencies=True
)

# SOP includes:
# - Step-by-step procedures
# - Confidence scores for each step
# - Contingency procedures for low-confidence steps
# - ACI monitoring checkpoints
```

---

### Stage 8: Lab Execution + RESE

**Location:** `rese/integrations/stage8.py`

#### RESE Components

- **Γ₁: Real-Time ACI Monitoring** - Tracks uncertainty during execution
- **Predictive Models** - Predicts quality issues before they occur

#### Integration Code

```python
from rese.integrations.stage8 import Stage8RESEMonitor

monitor = Stage8RESEMonitor()

# Monitor execution with ACI
monitor.start_monitoring(experiment_id='exp_001')

# During execution
monitor.record_step(
    step_name='heat_furnace',
    variables={'temperature': 847, 'time': 120},
    aci_before=0.15,
    aci_after=0.12
)

# Get real-time ACI
current_aci = monitor.get_current_aci()
if current_aci > 0.2:
    print("Warning: ACI increasing - potential issue")
```

---

### Stage 9: Continuous Monitoring + RESE

**Location:** `rese/integrations/stage9.py`

#### RESE Components

- **Γ₁: ACI Tracking** - Long-term ACI monitoring
- **Trend Analysis** - Detects ACI degradation

#### Integration Code

```python
from rese.integrations.stage9 import Stage9RESEMonitor

monitor = Stage9RESEMonitor()

# Continuous ACI tracking
monitor.track_system(
    system_id='system_001',
    metrics=['performance', 'reliability', 'quality'],
    aci_threshold=0.15
)

# Detect degradation
if monitor.detect_aci_degradation(window='7d'):
    print("Warning: System ACI degrading")
    print(f"Current ACI: {monitor.get_current_aci()}")
    print(f"Baseline ACI: {monitor.get_baseline_aci()}")
```

---

## Data Flow

### Complete RESE-Enhanced Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    E2E+RESE DATA FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STAGE 1: Prompt Analysis                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: "Design RT superconductor"                        │    │
│  │ ↓                                                         │    │
│  │ Φ₁ (SCE): Extract constraints → [Tc≥293K, Hc≥1T]         │    │
│  │ ↓                                                         │    │
│  │ Φ₁.₅: Mine assumptions → [stability, cost, manufactur.] │    │
│  │ ↓                                                         │    │
│  │ Φ₂: Detect biases → [confirmation_bias: 0.3]            │    │
│  │ ↓                                                         │    │
│  │ Output: Refined prompt + constraints + assumptions       │    │
│  │ ACI: 0.85 → 0.72 (15% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGE 2: Knowledge Graph                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: Refined prompt + constraints                      │    │
│  │ ↓                                                         │    │
│  │ Ψ₂: Map domains → [circuit_design: 0.82, ...]           │    │
│  │ ↓                                                         │    │
│  │ I_mech: Validate isomorphism → circuit: VALIDATED (0.89)│    │
│  │ ↓                                                         │    │
│  │ Output: Validated isomorphic problem + transfer solution │    │
│  │ ACI: 0.72 → 0.55 (24% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGE 3: Solution Generation                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: Isomorphic problem + transferred solution         │    │
│  │ ↓                                                         │    │
│  │ Γ₂: ACI-guided MCTS → Optimize for low ACI               │    │
│  │ ↓                                                         │    │
│  │ N_max: Convergence detection → Converged @ iter 847     │    │
│  │ ↓                                                         │    │
│  │ Output: Optimized solution + variables                   │    │
│  │ ACI: 0.55 → 0.28 (49% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGE 4: Mathematical Formalization                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: Optimized solution                                 │    │
│  │ ↓                                                         │    │
│  │ Lean 4: Generate proofs → Critical steps verified        │    │
│  │ ↓                                                         │    │
│  │ Δ₃: Validate ACI reduction → 67% total reduction ✓      │    │
│  │ ↓                                                         │    │
│  │ Output: Formalized solution + proofs                     │    │
│  │ ACI: 0.28 → 0.15 (46% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGES 5-9: Red Team → Knowledge → SOP → Lab → Monitor        │
│  (All with RESE ACI tracking and validation)                   │
│                                                                  │
│  FINAL: TRL-9 system with ACI < 0.2 (82% total reduction)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Complete RESE Configuration for E2E

```python
from rese.config import RESEConfig

# Configure RESE for E2E integration
config = RESEConfig()

# Stage 1: Prompt Analysis
config.phase1.phi15_enabled = True
config.phase1.phi15_assumption_threshold = 0.6
config.phase1.phi2_enabled = True
config.phase1.phi2_bias_threshold = 0.5

# Stage 2: Knowledge Graph
config.phase2.psi2_similarity_threshold = 0.7
config.phase2.psi3_target_accuracy = 0.80
config.phase2.imech_algorithm = "weisfeiler_lehman"

# Stage 3: Solution Generation
config.phase3.gamma2_iterations = 5000
config.phase3.gamma2_aci_guided = True
config.phase3.convergence_enabled = True

# Stage 4: Formalization
config.phase4.delta3_min_aci_reduction = 0.2
config.phase4.delta3_validation_threshold = 0.7

# Pipeline
config.pipeline.enable_caching = True
config.pipeline.checkpoint_interval = 300

# Monitoring
config.monitoring.enable_metrics = True
config.monitoring.alert_threshold_aci = 0.2

# Save configuration
config.save('data/e2e_rese_config.json')
```

---

## Best Practices

### 1. Gradual Integration

**Phase 1 (Week 1-2):** Integrate Φ₁ and Φ₁.₅ into Stage 1 only
```python
# Start with just Stage 1
from rese.integrations.stage1 import Stage1RESEAnalyzer
analyzer = Stage1RESEAnalyzer()
```

**Phase 2 (Week 3-4):** Add Stage 2 (Ψ₂, I_mech)
**Phase 3 (Week 5-6):** Add Stage 3 (Γ₂, N_max)
**Phase 4 (Week 7-8):** Add Stage 4 (Δ₃)
**Phase 5 (Week 9+):** Add remaining stages

### 2. ACI Threshold Management

Set appropriate ACI thresholds for each stage:

```python
STAGE_ACI_THRESHOLDS = {
    'stage1': 0.75,   # After constraint extraction
    'stage2': 0.60,   # After isomorphism validation
    'stage3': 0.35,   # After MCTS optimization
    'stage4': 0.20,   # After formalization (final target)
    'stage5': 0.20,   # After red team (maintain)
    'stage6': 0.20,   # After knowledge extraction
    'stage7': 0.20,   # In SOP
    'stage8': 0.25,   # During lab (allow some variance)
    'stage9': 0.20    # During monitoring
}
```

### 3. Assumption Validation

Always validate Φ₁.₅ assumptions with domain experts:

```python
assumptions = analyzer.mine_assumptions(prompt)

# Get human validation
for assumption in assumptions:
    if assumption['confidence'] < 0.9:
        print(f"REVIEW: {assumption['description']}")
        validation = input("Valid? (y/n/q): ")
        if validation == 'n':
            # Reject assumption
            assumption['rejected'] = True
```

### 4. Isomorphism Verification

Don't blindly trust I_mech scores:

```python
# High similarity but different domains
if similarity.score > 0.8:
    # Verify with human expert
    print(f"Potential isomorphism: {target_domain}")
    print(f"Score: {similarity.score}")
    expert_validation = input("Validate transfer? (y/n): ")

    if expert_validation == 'y':
        # Proceed with transfer
        transfer_solution(source, target)
```

### 5. ACI Monitoring

Monitor ACI continuously:

```python
# Set up ACI alerts
if current_aci > STAGE_ACI_THRESHOLDS[current_stage] * 1.2:
    print(f"WARNING: ACI {current_aci:.2f} exceeds threshold")
    print("Investigate immediately")
```

---

## Common Pitfalls

### Pitfall 1: Over-Reliance on Automation

**Problem:** Trusting RESE outputs without verification

**Solution:**
```python
# ALWAYS validate critical assumptions
if assumption['confidence'] < 0.95:
    expert_review_required = True
```

### Pitfall 2: Ignoring ACI Trends

**Problem:** Looking only at final ACI, not trend

**Solution:**
```python
# Check ACI trend
if aci_history[-1] > aci_history[-2]:
    print("WARNING: ACI increasing - solution degrading!")
```

### Pitfall 3: Premature Isomorphism Acceptance

**Problem:** Accepting isomorphisms with score just above threshold

**Solution:**
```python
# Require safety margin for isomorphisms
if similarity.score < 0.85:  # Even though threshold is 0.8
    print("Score too close to threshold - manual review required")
```

### Pitfall 4: Insufficient MCTS Iterations

**Problem:** Stopping MCTS too early

**Solution:**
```python
# Use adaptive iterations
if not converged and iterations < max_iterations:
    print("Not converged - continuing search...")
```

### Pitfall 5: Configuration Drift

**Problem**: Different stages using different RESE configs

**Solution:**
```python
# Use single config source
config = get_config()
# All stages use same config
```

---

## Troubleshooting

### Issue: Stage 1 Φ₁.₅ Finds Too Many Assumptions

**Symptom:** 100+ assumptions found

**Diagnosis:**
```python
# Check confidence distribution
confidences = [a['confidence'] for a in assumptions]
print(f"Mean confidence: {np.mean(confidences)}")
```

**Solution:**
```python
# Increase threshold
config.phase1.phi15_assumption_threshold = 0.7  # Was 0.6
```

---

### Issue: Stage 2 I_mech No Isomorphisms Found

**Symptom:** Zero domains above similarity threshold

**Diagnosis:**
```python
# Check maximum similarity
max_sim = max(m['similarity'] for m in mappings)
print(f"Max similarity: {max_sim}")
```

**Solution:**
```python
if max_sim < 0.7 and max_sim > 0.6:
    # Lower threshold slightly
    config.phase2.psi2_similarity_threshold = 0.65
```

---

### Issue: Stage 3 MCTS Does Not Converge

**Symptom:** MCTS runs to max_iterations without converging

**Diagnosis:**
```python
# Check ACI improvement
improvement = aci_history[0] - aci_history[-1]
print(f"ACI improvement: {improvement}")
```

**Solution:**
```python
if improvement < 0.1:
    # Increase iterations or check problem formulation
    config.phase3.gamma2_iterations = 10000
```

---

### Issue: Stage 4 Δ₃ Validation Fails

**Symptom:** ACI reduction < 20%

**Diagnosis:**
```python
reduction = (baseline_aci - final_aci) / baseline_aci
print(f"ACI reduction: {reduction * 100:.1f}%")
```

**Solution:**
```python
if reduction < 0.2:
    # Re-run Stage 3 with more iterations
    # Or check if problem is over-constrained
```

---

## Performance Optimization

### Caching Strategy

```python
# Enable caching for development
config.pipeline.enable_caching = True
config.pipeline.cache_ttl_seconds = 3600

# Disable for production (fresh analysis each time)
config.pipeline.enable_caching = False
```

### Parallel Processing

```python
# Parallel MCTS agents
config.phase3.gamma2_parallel_agents = 8  # Use 8 cores

# Parallel isomorphism checking
config.phase2.psi3_parallel_isomorphism_check = True
```

### Memory Management

```python
# Limit memory usage
config.pipeline.max_memory_gb = 32.0

# Clear cache periodically
if len(cache) > 1000:
    cache.clear()
```

---

## Advanced Topics

### Custom ACI Calculation

```python
from gamma1.core.aci_calculator import ACICalculator

class CustomACICalculator(ACICalculator):
    def calculate(self, constraints, solution):
        # Custom ACI formula
        base_aci = super().calculate(constraints, solution)
        custom_factor = self._calculate_custom_factor(solution)
        return base_aci * custom_factor

# Use custom calculator
config.phase3.gamma2_aci_calculator = CustomACICalculator()
```

### Custom Isomorphism Algorithm

```python
from phase2.imech.algorithms import WeisfeilerLehman

class CustomIsomorphismAlgorithm(WeisfeilerLehman):
    def compare(self, graph1, graph2):
        # Custom comparison logic
        similarity = super().compare(graph1, graph2)
        # Adjust based on domain knowledge
        return self._adjust_similarity(similarity)

# Use custom algorithm
config.phase2.imech_algorithm = 'custom'
```

### Stage Skipping

```python
# Skip Stage 2 if no similar domains needed
if skip_kg_stage:
    result = pipeline.run(problem, phases=['phase1', 'phase3', 'phase4'])
```

---

**Integration Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Integration Team
=======
# RESE Integration Guide

## Table of Contents

1. [Overview](#overview)
2. [Integration Architecture](#integration-architecture)
3. [Stage-by-Stage Integration](#stage-by-stage-integration)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Common Pitfalls](#common-pitfalls)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Topics](#advanced-topics)

---

## Overview

### What is RESE Integration?

RESE (Recursive Epistemic Solvability Engine) enhances the End-to-End Invention System by providing **quantified reasoning validation** at every stage. This guide explains how to integrate RESE components into each E2E stage.

### Integration Philosophy

**Traditional E2E (without RESE):**
```
Stage 1 → Stage 2 → Stage 3 → ... → Stage 9
[Qualitative] [Heuristic] [Best-effort]
```

**E2E + RESE:**
```
Stage 1+RESE → Stage 2+RESE → Stage 3+RESE → ... → Stage 9+RESE
[Validated]  [Quantified]  [Statistical]      [Tracked]
```

### Key Benefits

1. **Quantified Confidence**: Every stage produces statistical confidence metrics
2. **ACI Tracking**: Algorithmic Complexity Index tracked through entire pipeline
3. **Error Elimination**: Φ₁.₅ discovers hidden assumptions before they cause failures
4. **Solution Transfer**: I_mech enables reliable analogy-based solution transfer
5. **Validation**: Δ₃ guarantees ≥20% ACI reduction

---

## Integration Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    E2E + RESE INTEGRATION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Stage 1 │    │ Stage 2 │    │ Stage 3 │    │ Stage 4 │       │
│  │ Prompt  │    │ KG      │    │ Soln    │    │ Formal  │       │
│  │ + SCE   │    │ + I_mech│    │ + MCTS  │    │ + Δ₃    │       │
│  │ + Φ₁.₅  │    │ + Ψ₂    │    │ + Γ₂    │    │         │       │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘       │
│       │              │              │              │              │
│       ▼              ▼              ▼              ▼              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ Stage 5 │    │ Stage 6 │    │ Stage 7 │    │ Stage 8 │       │
│  │ Red     │    │ Know    │    │ SOP     │    │ Lab     │       │
│  │ + ACI   │    │ + Φ₁.₅  │    │ + Δ₁    │    │ + Γ₁    │       │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘       │
│       │              │              │              │              │
│       ▼              ▼              ▼              ▼              │
│  ┌─────────┐                                                     │
│  │ Stage 9 │◀───── RESE ACI Tracking Across All Stages          │
│  │ Monitor │                                                     │
│  │ + Γ₁    │                                                     │
│  └─────────┘                                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### RESE Wrapper Pattern

Each stage has a RESE wrapper that:

1. **Pre-processes**: Runs RESE analysis on stage input
2. **Enhances**: Adds RESE capabilities to stage execution
3. **Post-processes**: Validates output with RESE metrics
4. **Tracks**: Updates ACI history and confidence scores

---

## Stage-by-Stage Integration

### Stage 1: Prompt Analysis + RESE

**Location:** `rese/integrations/stage1.py`

#### RESE Components

- **Φ₁: Symbolic Constraint Engine** - Formalizes constraints
- **Φ₁.₅: Tacit Assumption Miner** - Discovers hidden requirements
- **Φ₂: Cognitive Bias Detector** - Identifies biases in prompt

#### Integration Code

```python
from rese.integrations.stage1 import Stage1RESEAnalyzer

# Initialize analyzer
analyzer = Stage1RESEAnalyzer()

# Analyze prompt
result = analyzer.analyze_prompt(
    prompt_text="Design a room-temperature superconductor",
    domain="materials_science"
)

# Access RESE enhancements
print(f"Constraints: {len(result.constraints)}")
print(f"Assumptions: {len(result.assumptions)}")
print(f"Bias Score: {result.bias_score}")
```

#### Data Flow

```
Input Prompt
    │
    ▼
┌───────────────────┐
│ Φ₁: SCE           │ → Formal constraints
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Φ₁.₅: Assumption  │ → Hidden assumptions
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Φ₂: Bias Detection│ → Bias report
└───────────────────┘
    │
    ▼
Enhanced Prompt (with constraints, assumptions, debiasing suggestions)
```

#### Example Output

```python
{
    'original_prompt': "Design a room-temperature superconductor",
    'constraints': [
        {'id': 'c1', 'type': 'hard', 'formalization': 'Tc >= 293 K'},
        {'id': 'c2', 'type': 'hard', 'formalization': 'critical_field >= 1 T'}
    ],
    'assumptions': [
        {'description': 'Material must be solid at room temperature', 'confidence': 0.95},
        {'description': 'Manufacturing cost must be feasible', 'confidence': 0.87},
        {'description': 'Material must be chemically stable', 'confidence': 0.92}
    ],
    'bias_report': {
        'overall_bias_score': 0.23,
        'detections': [
            {'type': 'confirmation_bias', 'severity': 'low'}
        ]
    },
    'refined_prompt': "Design a room-temperature superconductor (Tc >= 293 K) with chemical stability, feasible manufacturing cost, and critical field >= 1 T"
}
```

---

### Stage 2: Knowledge Graph + RESE

**Location:** `rese/integrations/stage2.py`

#### RESE Components

- **Ψ₂: Semantic Ontology Mapper** - Finds similar domains
- **I_mech: Mechanistic Isomorphism** - Validates analogies

#### Integration Code

```python
from rese.integrations.stage2 import Stage2RESEMapper

mapper = Stage2RESEMapper()

# Map to similar domains
mappings = mapper.map_to_domains(
    problem_description=stage1_output['refined_prompt'],
    knowledge_graph=kg,
    similarity_threshold=0.7
)

# Validate isomorphisms
for mapping in mappings:
    isomorphism = mapper.validate_isomorphism(
        source=mapping['source_domain'],
        target=mapping['target_domain']
    )

    if isomorphism.score > 0.8:
        print(f"Validated: {mapping['target_domain']} (score: {isomorphism.score})")
        print(f"Transfer confidence: {isomorphism.confidence}")
```

#### Data Flow

```
Stage 1 Output (Refined Prompt)
    │
    ▼
┌───────────────────┐
│ Ψ₂: Semantic Map  │ → Candidate domains
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ I_mech: Validate  │ → Isomorphic domains
└───────────────────┘
    │
    ▼
Validated Solution Transfer Candidates
```

---

### Stage 3: Solution Generation + RESE

**Location:** `rese/integrations/stage3.py`

#### RESE Components

- **Γ₂: ACI-Guided MCTS** - Optimizes with ACI as reward
- **N_max: Convergence Controller** - Detects when to stop search

#### Integration Code

```python
from rese.integrations.stage3 import Stage3RESEGenerator

generator = Stage3RESEGenerator()

# Generate solution with ACI guidance
solution = generator.generate(
    problem=stage2_output['validated_problem'],
    constraints=stage1_output['constraints'],
    mcts_iterations=5000,
    aci_guided=True
)

print(f"Solution: {solution.variables}")
print(f"ACI: {solution.aci}")
print(f"Converged: {solution.converged}")
```

#### MCTS Configuration

```python
# Configure MCTS for RESE
config = {
    'exploration_constant': 1.41,      # UCB C parameter
    'max_iterations': 5000,            # N_max
    'playout_depth': 100,              # Simulation depth
    'aci_guided': True,                # Use ACI for exploration
    'parallel_agents': 4,              # Parallel search
    'convergence_patience': 50,        # Stop if no improvement
    'convergence_min_delta': 0.001     # Minimum improvement
}
```

---

### Stage 4: Mathematical Formalization + RESE

**Location:** `rese/integrations/stage4.py`

#### RESE Components

- **Δ₃: ACI Reduction Validator** - Validates solution quality
- **Lean 4 Integration** - Requires formal proofs for critical steps

#### Integration Code

```python
from rese.integrations.stage4 import Stage4RESEFormalizer

formalizer = Stage4RESEFormalizer()

# Formalize solution with validation
formalization = formalizer.formalize(
    solution=stage3_output['solution'],
    constraints=stage1_output['constraints'],
    require_proof=True
)

# Validate ACI reduction
validation = formalizer.validate_aci_reduction(
    baseline_aci=stage1_output['initial_aci'],
    final_aci=formalization['aci'],
    min_reduction=0.2
)

if validation.is_valid:
    print(f"Formalization validated (score: {validation.score})")
    print(f"ACI reduction: {validation.aci_reduction * 100:.1f}%")
```

---

### Stage 5: Red Team Analysis + RESE

**Location:** `rese/integrations/stage5.py`

#### RESE Components

- **ACI Quantification** - Measures residual uncertainty
- **Φ₁.₅ Re-run** - Checks for newly introduced assumptions

#### Integration Code

```python
from rese.integrations.stage5 import Stage5RESEAnalyzer

analyzer = Stage5RESEAnalyzer()

# Red team analysis with ACI
red_team_result = analyzer.analyze(
    solution=stage4_output['formalization'],
    attack_vectors=['constraint_violation', 'assumption_failure', 'optimization_gap']
)

print(f"Residual ACI: {red_team_result.aci}")
print(f"Vulnerabilities Found: {len(red_team_result.vulnerabilities)}")
print(f"Mitigated: {red_team_result.vulnerabilities_mitigated}")
```

---

### Stage 6: Knowledge Extraction + RESE

**Location:** `rese/integrations/stage6.py`

#### RESE Components

- **Φ₁.₅ Feedback** - Feeds discovered assumptions back to database
- **I_mech Pattern Mining** - Extracts isomorphic patterns

#### Integration Code

```python
from rese.integrations.stage6 import Stage6RESEExtractor

extractor = Stage6RESEExtractor()

# Extract knowledge with RESE
knowledge = extractor.extract(
    execution_result=stage5_output,
    mine_assumptions=True,
    mine_isomorphisms=True
)

# Knowledge added to database:
# - Failed assumptions (for future Φ₁.₅ runs)
# - Validated isomorphisms (for future I_mech runs)
# - ACI reduction patterns (for prediction)
```

---

### Stage 7: SOP Generation + RESE

**Location:** `rese/integrations/stage7.py`

#### RESE Components

- **Δ₁: Architecture Assembly** - Assembles turnkey components
- **Confidence Annotation** - Adds confidence metrics to SOPs

#### Integration Code

```python
from rese.integrations.stage7 import Stage7RESEGenerator

generator = Stage7RESEGenerator()

# Generate SOP with confidence
sop = generator.generate_sop(
    solution=stage6_output['validated_solution'],
    confidence_threshold=0.7,
    include_contingencies=True
)

# SOP includes:
# - Step-by-step procedures
# - Confidence scores for each step
# - Contingency procedures for low-confidence steps
# - ACI monitoring checkpoints
```

---

### Stage 8: Lab Execution + RESE

**Location:** `rese/integrations/stage8.py`

#### RESE Components

- **Γ₁: Real-Time ACI Monitoring** - Tracks uncertainty during execution
- **Predictive Models** - Predicts quality issues before they occur

#### Integration Code

```python
from rese.integrations.stage8 import Stage8RESEMonitor

monitor = Stage8RESEMonitor()

# Monitor execution with ACI
monitor.start_monitoring(experiment_id='exp_001')

# During execution
monitor.record_step(
    step_name='heat_furnace',
    variables={'temperature': 847, 'time': 120},
    aci_before=0.15,
    aci_after=0.12
)

# Get real-time ACI
current_aci = monitor.get_current_aci()
if current_aci > 0.2:
    print("Warning: ACI increasing - potential issue")
```

---

### Stage 9: Continuous Monitoring + RESE

**Location:** `rese/integrations/stage9.py`

#### RESE Components

- **Γ₁: ACI Tracking** - Long-term ACI monitoring
- **Trend Analysis** - Detects ACI degradation

#### Integration Code

```python
from rese.integrations.stage9 import Stage9RESEMonitor

monitor = Stage9RESEMonitor()

# Continuous ACI tracking
monitor.track_system(
    system_id='system_001',
    metrics=['performance', 'reliability', 'quality'],
    aci_threshold=0.15
)

# Detect degradation
if monitor.detect_aci_degradation(window='7d'):
    print("Warning: System ACI degrading")
    print(f"Current ACI: {monitor.get_current_aci()}")
    print(f"Baseline ACI: {monitor.get_baseline_aci()}")
```

---

## Data Flow

### Complete RESE-Enhanced Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    E2E+RESE DATA FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STAGE 1: Prompt Analysis                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: "Design RT superconductor"                        │    │
│  │ ↓                                                         │    │
│  │ Φ₁ (SCE): Extract constraints → [Tc≥293K, Hc≥1T]         │    │
│  │ ↓                                                         │    │
│  │ Φ₁.₅: Mine assumptions → [stability, cost, manufactur.] │    │
│  │ ↓                                                         │    │
│  │ Φ₂: Detect biases → [confirmation_bias: 0.3]            │    │
│  │ ↓                                                         │    │
│  │ Output: Refined prompt + constraints + assumptions       │    │
│  │ ACI: 0.85 → 0.72 (15% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGE 2: Knowledge Graph                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: Refined prompt + constraints                      │    │
│  │ ↓                                                         │    │
│  │ Ψ₂: Map domains → [circuit_design: 0.82, ...]           │    │
│  │ ↓                                                         │    │
│  │ I_mech: Validate isomorphism → circuit: VALIDATED (0.89)│    │
│  │ ↓                                                         │    │
│  │ Output: Validated isomorphic problem + transfer solution │    │
│  │ ACI: 0.72 → 0.55 (24% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGE 3: Solution Generation                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: Isomorphic problem + transferred solution         │    │
│  │ ↓                                                         │    │
│  │ Γ₂: ACI-guided MCTS → Optimize for low ACI               │    │
│  │ ↓                                                         │    │
│  │ N_max: Convergence detection → Converged @ iter 847     │    │
│  │ ↓                                                         │    │
│  │ Output: Optimized solution + variables                   │    │
│  │ ACI: 0.55 → 0.28 (49% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGE 4: Mathematical Formalization                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input: Optimized solution                                 │    │
│  │ ↓                                                         │    │
│  │ Lean 4: Generate proofs → Critical steps verified        │    │
│  │ ↓                                                         │    │
│  │ Δ₃: Validate ACI reduction → 67% total reduction ✓      │    │
│  │ ↓                                                         │    │
│  │ Output: Formalized solution + proofs                     │    │
│  │ ACI: 0.28 → 0.15 (46% reduction)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  STAGES 5-9: Red Team → Knowledge → SOP → Lab → Monitor        │
│  (All with RESE ACI tracking and validation)                   │
│                                                                  │
│  FINAL: TRL-9 system with ACI < 0.2 (82% total reduction)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Complete RESE Configuration for E2E

```python
from rese.config import RESEConfig

# Configure RESE for E2E integration
config = RESEConfig()

# Stage 1: Prompt Analysis
config.phase1.phi15_enabled = True
config.phase1.phi15_assumption_threshold = 0.6
config.phase1.phi2_enabled = True
config.phase1.phi2_bias_threshold = 0.5

# Stage 2: Knowledge Graph
config.phase2.psi2_similarity_threshold = 0.7
config.phase2.psi3_target_accuracy = 0.80
config.phase2.imech_algorithm = "weisfeiler_lehman"

# Stage 3: Solution Generation
config.phase3.gamma2_iterations = 5000
config.phase3.gamma2_aci_guided = True
config.phase3.convergence_enabled = True

# Stage 4: Formalization
config.phase4.delta3_min_aci_reduction = 0.2
config.phase4.delta3_validation_threshold = 0.7

# Pipeline
config.pipeline.enable_caching = True
config.pipeline.checkpoint_interval = 300

# Monitoring
config.monitoring.enable_metrics = True
config.monitoring.alert_threshold_aci = 0.2

# Save configuration
config.save('data/e2e_rese_config.json')
```

---

## Best Practices

### 1. Gradual Integration

**Phase 1 (Week 1-2):** Integrate Φ₁ and Φ₁.₅ into Stage 1 only
```python
# Start with just Stage 1
from rese.integrations.stage1 import Stage1RESEAnalyzer
analyzer = Stage1RESEAnalyzer()
```

**Phase 2 (Week 3-4):** Add Stage 2 (Ψ₂, I_mech)
**Phase 3 (Week 5-6):** Add Stage 3 (Γ₂, N_max)
**Phase 4 (Week 7-8):** Add Stage 4 (Δ₃)
**Phase 5 (Week 9+):** Add remaining stages

### 2. ACI Threshold Management

Set appropriate ACI thresholds for each stage:

```python
STAGE_ACI_THRESHOLDS = {
    'stage1': 0.75,   # After constraint extraction
    'stage2': 0.60,   # After isomorphism validation
    'stage3': 0.35,   # After MCTS optimization
    'stage4': 0.20,   # After formalization (final target)
    'stage5': 0.20,   # After red team (maintain)
    'stage6': 0.20,   # After knowledge extraction
    'stage7': 0.20,   # In SOP
    'stage8': 0.25,   # During lab (allow some variance)
    'stage9': 0.20    # During monitoring
}
```

### 3. Assumption Validation

Always validate Φ₁.₅ assumptions with domain experts:

```python
assumptions = analyzer.mine_assumptions(prompt)

# Get human validation
for assumption in assumptions:
    if assumption['confidence'] < 0.9:
        print(f"REVIEW: {assumption['description']}")
        validation = input("Valid? (y/n/q): ")
        if validation == 'n':
            # Reject assumption
            assumption['rejected'] = True
```

### 4. Isomorphism Verification

Don't blindly trust I_mech scores:

```python
# High similarity but different domains
if similarity.score > 0.8:
    # Verify with human expert
    print(f"Potential isomorphism: {target_domain}")
    print(f"Score: {similarity.score}")
    expert_validation = input("Validate transfer? (y/n): ")

    if expert_validation == 'y':
        # Proceed with transfer
        transfer_solution(source, target)
```

### 5. ACI Monitoring

Monitor ACI continuously:

```python
# Set up ACI alerts
if current_aci > STAGE_ACI_THRESHOLDS[current_stage] * 1.2:
    print(f"WARNING: ACI {current_aci:.2f} exceeds threshold")
    print("Investigate immediately")
```

---

## Common Pitfalls

### Pitfall 1: Over-Reliance on Automation

**Problem:** Trusting RESE outputs without verification

**Solution:**
```python
# ALWAYS validate critical assumptions
if assumption['confidence'] < 0.95:
    expert_review_required = True
```

### Pitfall 2: Ignoring ACI Trends

**Problem:** Looking only at final ACI, not trend

**Solution:**
```python
# Check ACI trend
if aci_history[-1] > aci_history[-2]:
    print("WARNING: ACI increasing - solution degrading!")
```

### Pitfall 3: Premature Isomorphism Acceptance

**Problem:** Accepting isomorphisms with score just above threshold

**Solution:**
```python
# Require safety margin for isomorphisms
if similarity.score < 0.85:  # Even though threshold is 0.8
    print("Score too close to threshold - manual review required")
```

### Pitfall 4: Insufficient MCTS Iterations

**Problem:** Stopping MCTS too early

**Solution:**
```python
# Use adaptive iterations
if not converged and iterations < max_iterations:
    print("Not converged - continuing search...")
```

### Pitfall 5: Configuration Drift

**Problem**: Different stages using different RESE configs

**Solution:**
```python
# Use single config source
config = get_config()
# All stages use same config
```

---

## Troubleshooting

### Issue: Stage 1 Φ₁.₅ Finds Too Many Assumptions

**Symptom:** 100+ assumptions found

**Diagnosis:**
```python
# Check confidence distribution
confidences = [a['confidence'] for a in assumptions]
print(f"Mean confidence: {np.mean(confidences)}")
```

**Solution:**
```python
# Increase threshold
config.phase1.phi15_assumption_threshold = 0.7  # Was 0.6
```

---

### Issue: Stage 2 I_mech No Isomorphisms Found

**Symptom:** Zero domains above similarity threshold

**Diagnosis:**
```python
# Check maximum similarity
max_sim = max(m['similarity'] for m in mappings)
print(f"Max similarity: {max_sim}")
```

**Solution:**
```python
if max_sim < 0.7 and max_sim > 0.6:
    # Lower threshold slightly
    config.phase2.psi2_similarity_threshold = 0.65
```

---

### Issue: Stage 3 MCTS Does Not Converge

**Symptom:** MCTS runs to max_iterations without converging

**Diagnosis:**
```python
# Check ACI improvement
improvement = aci_history[0] - aci_history[-1]
print(f"ACI improvement: {improvement}")
```

**Solution:**
```python
if improvement < 0.1:
    # Increase iterations or check problem formulation
    config.phase3.gamma2_iterations = 10000
```

---

### Issue: Stage 4 Δ₃ Validation Fails

**Symptom:** ACI reduction < 20%

**Diagnosis:**
```python
reduction = (baseline_aci - final_aci) / baseline_aci
print(f"ACI reduction: {reduction * 100:.1f}%")
```

**Solution:**
```python
if reduction < 0.2:
    # Re-run Stage 3 with more iterations
    # Or check if problem is over-constrained
```

---

## Performance Optimization

### Caching Strategy

```python
# Enable caching for development
config.pipeline.enable_caching = True
config.pipeline.cache_ttl_seconds = 3600

# Disable for production (fresh analysis each time)
config.pipeline.enable_caching = False
```

### Parallel Processing

```python
# Parallel MCTS agents
config.phase3.gamma2_parallel_agents = 8  # Use 8 cores

# Parallel isomorphism checking
config.phase2.psi3_parallel_isomorphism_check = True
```

### Memory Management

```python
# Limit memory usage
config.pipeline.max_memory_gb = 32.0

# Clear cache periodically
if len(cache) > 1000:
    cache.clear()
```

---

## Advanced Topics

### Custom ACI Calculation

```python
from gamma1.core.aci_calculator import ACICalculator

class CustomACICalculator(ACICalculator):
    def calculate(self, constraints, solution):
        # Custom ACI formula
        base_aci = super().calculate(constraints, solution)
        custom_factor = self._calculate_custom_factor(solution)
        return base_aci * custom_factor

# Use custom calculator
config.phase3.gamma2_aci_calculator = CustomACICalculator()
```

### Custom Isomorphism Algorithm

```python
from phase2.imech.algorithms import WeisfeilerLehman

class CustomIsomorphismAlgorithm(WeisfeilerLehman):
    def compare(self, graph1, graph2):
        # Custom comparison logic
        similarity = super().compare(graph1, graph2)
        # Adjust based on domain knowledge
        return self._adjust_similarity(similarity)

# Use custom algorithm
config.phase2.imech_algorithm = 'custom'
```

### Stage Skipping

```python
# Skip Stage 2 if no similar domains needed
if skip_kg_stage:
    result = pipeline.run(problem, phases=['phase1', 'phase3', 'phase4'])
```

---

**Integration Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Integration Team
>>>>>>> 1cb9c5e35 (update)
