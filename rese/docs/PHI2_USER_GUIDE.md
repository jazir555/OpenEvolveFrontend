# Φ₂ Metacognitive Debiasing System - User Guide

**Agent**: B2 (Φ₂ Specialist)
**Date**: 2025-12-31
**Status**: ✅ Complete
**Version**: 1.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Bias Detection](#bias-detection)
4. [Debiasing Strategies](#debiasing-strategies)
5. [SCE Integration](#sce-integration)
6. [Stage 5 Integration](#stage-5-integration)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## 1. Introduction

The **Φ₂ Metacognitive Debiasing System** is a critical component of the RESE framework's Phase I (Epistemic Audit). It systematically identifies and mitigates cognitive biases in problem formulation, constraint specification, and solution generation.

### What is Φ₂?

Φ₂ provides:
- **12 cognitive bias detectors** covering major bias categories
- **6 debiasing strategies** to mitigate detected biases
- **Real-time bias monitoring** during solution generation
- **Seamless integration** with the Symbolic Constraint Engine (SCE)
- **Comprehensive reporting** with actionable recommendations

### Why Use Φ₂?

Even with formal logic and assumption mining, AI systems are vulnerable to cognitive biases that can:
- Lead to false positive constraint validations
- Cause missed edge cases due to confirmation bias
- Overweight familiar solutions (availability bias)
- Anchor on initial formulations too early
- Reject valid alternatives due to sunk cost fallacy

Φ₂ acts as a "second thoughts" layer to catch and correct these biases.

---

## 2. Quick Start

### Installation

Φ₂ is part of the RESE framework. No additional installation is required.

```python
# Import Φ₂ components
from rese.phase1.cognitive_biases import (
    CognitiveBiasDetector,
    DebiasingStrategy
)
from rese.phase1.phi2_integration import (
    SCEPhi2Integrator,
    Stage5Phi2Monitor,
    IntegrationConfig
)
```

### Basic Usage

#### 1. Standalone Bias Detection

```python
from rese.phase1.cognitive_biases import CognitiveBiasDetector
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

# Create detector
detector = CognitiveBiasDetector()

# Create some constraints
constraints = [
    Constraint(
        id="c1",
        type=ConstraintType.HARD,
        description="The system will certainly achieve 100% accuracy",
        formalization="accuracy = 1.0",
        source="user_prompt"
    ),
    Constraint(
        id="c2",
        type=ConstraintType.HARD,
        description="This is clearly the best approach",
        formalization="best = current",
        source="expert"
    )
]

# Analyze for bias
report = detector.analyze_constraints(constraints)

# View results
print(f"Overall bias score: {report.overall_bias_score:.2f}")
print(f"Total detections: {report.total_detections}")
for rec in report.recommendations:
    print(f"  - {rec}")
```

#### 2. SCE Integration

```python
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine
from rese.phase1.phi2_integration import SCEPhi2Integrator, IntegrationConfig

# Create SCE and integrator
sce = SymbolicConstraintEngine()
config = IntegrationConfig(
    auto_check_on_add=True,
    bias_threshold=0.4
)
integrator = SCEPhi2Integrator(sce, config)

# Add constraints (automatically checked for bias)
constraint = Constraint(
    id="auto_1",
    type=ConstraintType.HARD,
    description="This will certainly work perfectly",
    formalization="perfect = true",
    source="user"
)
sce.add_constraint(constraint)  # Bias checked automatically!

# Get debiased suggestions
suggestions = integrator.suggest_debiased_formulation("auto_1")
for suggestion in suggestions:
    print(f"  {suggestion}")
```

#### 3. Stage 5 Monitoring

```python
from rese.phase1.phi2_integration import Stage5Phi2Monitor

# Create monitor
monitor = Stage5Phi2Monitor()

# Monitor generation steps
reasoning_steps = [
    "We will definitely achieve the optimal solution",
    "This approach is clearly superior",
    "The outcome is guaranteed"
]

for i, reasoning in enumerate(reasoning_steps, 1):
    report = monitor.monitor_generation_step(i, reasoning)

    # Check if intervention needed
    if monitor.should_intervene(i-1):
        print(f"[⚠️] Step {i}: Intervention recommended")
        alternatives = monitor.generate_debiased_alternatives(reasoning)
        print(f"  Alternatives: {alternatives[0]}")
```

---

## 3. Bias Detection

### Supported Biases

Φ₂ detects **12 cognitive biases**:

#### Evidence Evaluation Biases

1. **Confirmation Bias**
   - Tendency to seek confirming evidence
   - One-sided hypothesis testing
   - Absolutist language ("clearly", "obviously")

2. **Availability Bias**
   - Overweighting familiar information
   - Skewed source distribution
   - Neglect of unfamiliar domains

3. **Anchoring Bias**
   - Over-reliance on initial formulations
   - Star topology in constraint dependencies
   - Low solution variance

#### Decision Quality Biases

4. **Sunk Cost Fallacy**
   - Persisting with failing approaches
   - "We've already invested" reasoning
   - Path dependency

5. **Framing Effects**
   - Loss vs. gain framing
   - Emotional language
   - Preference reversals

6. **Overconfidence Effect**
   - Point estimates without uncertainty
   - High confidence with low justification
   - Absence of "may", "might", "approximately"

#### Social Reasoning Biases

7. **Dunning-Kruger Effect**
   - High confidence with low complexity
   - Lack of acknowledged limitations
   - Poor self-assessment

8. **Authority Bias**
   - Excessive deference to authority
   - Appeals to expertise over evidence
   - Source-based weighting

#### Pattern Recognition Biases

9. **Clustering Illusion**
   - Seeing patterns in randomness
   - Causal claims without statistics
   - Over-interpreting correlations

10. **Texas Sharpshooter Fallacy**
    - Post-hoc pattern selection
    - Cherry-picking data
    - Narrative fallacy

#### Causal Reasoning Biases

11. **Causal Oversimplification**
    - Single-cause explanations
    - Neglect of interactions
    - Linear assumptions

12. **Illusion of Control**
    - Deterministic language for stochastic processes
    - Underestimating external factors
    - No acknowledgment of uncertainty

### Detection API

```python
# Analyze all biases
report = detector.analyze_constraints(constraints)

# Analyze specific biases only
report = detector.analyze_constraints(
    constraints,
    bias_types=[BiasType.CONFIRMATION, BiasType.OVERCONFIDENCE]
)

# Get detection details
for detection in report.detections:
    print(f"Bias: {detection.bias_type.value}")
    print(f"Severity: {detection.severity.name}")
    print(f"Confidence: {detection.confidence:.2f}")
    print(f"Evidence: {detection.evidence}")
    print(f"Suggestion: {detection.suggestion}")
```

---

## 4. Debiasing Strategies

### Available Strategies

Φ₂ provides **6 debiasing strategies**:

#### 1. Consider-the-Opposite

Generate the opposite of a constraint to challenge assumptions.

```python
opposite = DebiasingStrategy.consider_the_opposite(constraint)
print(opposite)
# Output: "Consider opposite: 'The system will NOT achieve...'"
```

#### 2. Devil's Advocate

Generate challenges to a constraint's assumptions.

```python
challenges = DebiasingStrategy.devils_advocate(constraint)
for challenge in challenges:
    print(f"  - {challenge}")
# Output:
#   - Challenge: What if this is based on false assumptions?
#   - Challenge: Is this truly necessary?
#   - Challenge: What evidence supports this beyond the source?
```

#### 3. Pre-Mortem Analysis

Identify potential failure modes before implementation.

```python
failure_modes = DebiasingStrategy.pre_mortem_analysis(
    constraints,
    solution="proposed_solution"
)
for mode in failure_modes:
    print(f"  - {mode}")
```

#### 4. Red Teaming

Adversarially test for flaws (via integration monitoring).

```python
# Automatic in Stage 5 monitoring
# See Section 6 for details
```

#### 5. Reference Class Forecasting

Adjust predictions based on similar historical cases.

```python
# Coming in v2.0
# Will use historical constraint outcomes
```

#### 6. Forced Reformulation

Generate alternative formulations from different angles.

```python
reforms = DebiasingStrategy.forced_reformulation(constraint)
for reform in reforms:
    print(f"  - {reform}")
# Output:
#   - Original: We must maximize performance
#   - Positive frame: We must achieve performance
#   - Negative frame: We must avoid underperformance
```

---

## 5. SCE Integration

### Automatic Bias Checking

```python
# Configure integration
config = IntegrationConfig(
    auto_check_on_add=True,      # Check when adding constraints
    auto_check_on_conflict=True, # Check when conflicts detected
    bias_threshold=0.4,          # Alert threshold
    max_bias_score=0.7,          # Intervention threshold
    log_all_detections=True,     # Log all detections
    log_path="phi2_logs.jsonl"   # Log file path
)

# Create integrator
integrator = SCEPhi2Integrator(sce, config)

# Add constraints - automatically checked!
sce.add_constraint(biased_constraint)
# Output:
# [Φ₂ WARNING] High bias detected (0.85) in constraint 'biased_constraint'
#   Top issues: ...
```

### Manual Bias Checking

```python
# Check specific constraint
report = integrator.check_constraint_bias(constraint)

# Check all constraints
report = integrator.check_all_constraints()

# Get biased constraints above severity threshold
biased = integrator.get_biased_constraints(
    min_severity=Severity.MEDIUM
)
for constraint_id, detections in biased.items():
    print(f"{constraint_id}: {len(detections)} biases")
```

### Debiasing Constraints

```python
# Get debiased formulation suggestions
suggestions = integrator.suggest_debiased_formulation(constraint_id)

# Apply suggestion (manual step)
for suggestion in suggestions:
    if "consider opposite" in suggestion.lower():
        # Create opposite constraint
        debiased = create_opposite_constraint(constraint)
        sce.add_constraint(debiased)
```

### Statistics

```python
# Get integration statistics
stats = integrator.get_integration_statistics()
print(f"Constraints analyzed: {stats['sce_constraints_analyzed']}")
print(f"Reports generated: {stats['bias_reports_generated']}")
print(f"Average bias score: {stats['average_bias_score']:.2f}")
```

---

## 6. Stage 5 Integration

### Real-Time Monitoring

```python
# Create monitor
config = IntegrationConfig(
    real_time_monitoring=True,
    max_bias_score=0.6
)
monitor = Stage5Phi2Monitor(config)

# Monitor generation steps
for step, reasoning in enumerate(generation_steps):
    report = monitor.monitor_generation_step(
        step=step,
        reasoning=reasoning,
        constraints=current_constraints  # optional
    )

    # Check if intervention needed
    if monitor.should_intervene(step):
        # Get recommendations
        recs = monitor.get_step_recommendations(step)
        print(f"Intervention needed: {recs}")

        # Generate alternatives
        alts = monitor.generate_debiased_alternatives(reasoning)
        print(f"Alternatives: {alts}")
```

### Bias Trajectory

```python
# Get bias scores over time
trajectory = monitor.get_bias_trajectory()

for step, score in enumerate(trajectory, 1):
    print(f"Step {step}: {score:.2f}")

# Visualize (if matplotlib available)
import matplotlib.pyplot as plt
plt.plot(trajectory)
plt.xlabel('Generation Step')
plt.ylabel('Bias Score')
plt.title('Bias Trajectory')
plt.show()
```

### Monitoring Statistics

```python
stats = monitor.get_monitoring_statistics()
print(f"Steps monitored: {stats['total_steps_monitored']}")
print(f"Average bias: {stats['average_bias_score']:.2f}")
print(f"Interventions: {stats['interventions_recommended']}")
print(f"Critical biases: {stats['total_critical_biases']}")
```

---

## 7. API Reference

### CognitiveBiasDetector

```python
class CognitiveBiasDetector:
    """Main bias detection class"""

    def analyze_constraints(
        self,
        constraints: List[Constraint],
        bias_types: Optional[List[BiasType]] = None
    ) -> BiasReport:
        """Analyze constraints for cognitive biases"""

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
```

### BiasReport

```python
@dataclass
class BiasReport:
    """Comprehensive bias analysis report"""
    total_detections: int
    detections_by_type: Dict[BiasType, int]
    detections_by_severity: Dict[Severity, int]
    overall_bias_score: float  # [0, 1]
    recommendations: List[str]
    detections: List[BiasDetection]
```

### BiasDetection

```python
@dataclass
class BiasDetection:
    """A detected cognitive bias"""
    bias_type: BiasType
    severity: Severity  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float   # [0, 1]
    description: str
    evidence: Dict[str, str]
    suggestion: str
    affected_elements: List[str]
```

### SCEPhi2Integrator

```python
class SCEPhi2Integrator:
    """Integrates Φ₂ with Symbolic Constraint Engine"""

    def __init__(
        self,
        sce: SymbolicConstraintEngine,
        config: Optional[IntegrationConfig] = None
    ):

    def check_constraint_bias(
        self,
        constraint: Constraint
    ) -> BiasReport:

    def check_all_constraints(self) -> BiasReport:

    def get_biased_constraints(
        self,
        min_severity: Severity = Severity.MEDIUM
    ) -> Dict[str, List[BiasDetection]]:

    def suggest_debiased_formulation(
        self,
        constraint_id: str
    ) -> List[str]:

    def get_integration_statistics(self) -> Dict:
```

### Stage5Phi2Monitor

```python
class Stage5Phi2Monitor:
    """Monitors solution generation for bias"""

    def monitor_generation_step(
        self,
        step: int,
        reasoning: str,
        constraints: Optional[List[Constraint]] = None
    ) -> BiasReport:

    def should_intervene(self, current_step: int) -> bool:

    def get_bias_trajectory(self) -> List[float]:

    def generate_debiased_alternatives(
        self,
        reasoning: str
    ) -> List[str]:

    def get_monitoring_statistics(self) -> Dict:
```

---

## 8. Best Practices

### 1. Set Appropriate Thresholds

```python
# For high-stakes decisions (medical, safety-critical)
config = IntegrationConfig(
    bias_threshold=0.3,    # Lower = more sensitive
    max_bias_score=0.5     # Lower = stricter
)

# For exploratory research
config = IntegrationConfig(
    bias_threshold=0.6,    # Higher = less sensitive
    max_bias_score=0.8     # Higher = more permissive
)
```

### 2. Use Iterative Debiasing

```python
# Iteration 1: Initial detection
report = detector.analyze_constraints(constraints)

# Iteration 2: Apply debiasing
debiased = apply_debiasing(constraints, report)

# Iteration 3: Verify reduction
new_report = detector.analyze_constraints(debiased)
assert new_report.overall_bias_score < report.overall_bias_score
```

### 3. Combine Multiple Strategies

```python
# Don't rely on a single debiasing method
strategies = [
    DebiasingStrategy.consider_the_opposite,
    DebiasingStrategy.devils_advocate,
    DebiasingStrategy.forced_reformulation
]

all_suggestions = []
for strategy in strategies:
    all_suggestions.extend(strategy(constraint))

# Synthesize best approach
final = synthesize_suggestions(all_suggestions)
```

### 4. Log and Review

```python
# Enable logging for retrospective analysis
config = IntegrationConfig(
    log_all_detections=True,
    log_path="phi2_analysis.jsonl"
)

# Periodically review logs
import json
with open("phi2_analysis.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        analyze_bias_pattern(entry)
```

### 5. Human-in-the-Loop

```python
# Φ₂ detects, human decides
report = detector.analyze_constraints(constraints)

if report.overall_bias_score > 0.7:
    print("High bias detected - human review required")
    display_detections_to_user(report.detections)
    user_decision = get_user_input()
    # Apply based on human decision
```

---

## 9. Examples

### Example 1: Debiasing Problem Formulation

```python
from rese.phase1.cognitive_biases import (
    CognitiveBiasDetector,
    DebiasingStrategy,
    BiasType
)
from rese.core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType
)

# Original biased formulation
biased = Constraint(
    id="original",
    type=ConstraintType.HARD,
    description="We must achieve perfect accuracy because we've " \
                "already invested so much in this approach",
    formalization="accuracy = 1.0",
    source="existing_work"
)

# Detect biases
detector = CognitiveBiasDetector()
report = detector.analyze_constraints([biased])

print(f"Bias score: {report.overall_bias_score:.2f}")
print(f"Detections: {report.total_detections}")
# Output: Bias score: 0.85, Detections: 4
# Biases: sunk_cost, overconfidence, illusion_of_control, ...

# Apply debiasing
print("\n=== Debiasing Strategies ===")

# 1. Consider the opposite
print("\n1. Consider the Opposite:")
print(DebiasingStrategy.consider_the_opposite(biased))

# 2. Devil's advocate
print("\n2. Devil's Advocate Challenges:")
for challenge in DebiasingStrategy.devils_advocate(biased):
    print(f"  {challenge}")

# 3. Forced reformulation
print("\n3. Alternative Formulations:")
for reform in DebiasingStrategy.forced_reformulation(biased):
    print(f"  {reform}")

# 4. Create debiased version
debiased = Constraint(
    id="debiased",
    type=ConstraintType.HARD,
    description="The system should maintain accuracy above 95% " \
                "based on empirical validation",
    formalization="accuracy >= 0.95",
    source="requirements"
)

# Verify reduction
new_report = detector.analyze_constraints([debiased])
print(f"\n=== Result ===")
print(f"Original bias score: {report.overall_bias_score:.2f}")
print(f"Debiased score: {new_report.overall_bias_score:.2f}")
print(f"Reduction: {(1 - new_report.overall_bias_score/report.overall_bias_score)*100:.1f}%")
```

### Example 2: Real-Time Monitoring

```python
from rese.phase1.phi2_integration import Stage5Phi2Monitor

# Create monitor
monitor = Stage5Phi2Monitor()

# Simulate biased generation
biased_reasoning = [
    "This will certainly work",
    "Clearly the best approach",
    "Guaranteed success"
]

unbiased_reasoning = [
    "This approach may work with approximately 80% probability",
    "Consider alternative strategies as well",
    "Success depends on external factors"
]

print("=== Biased Generation ===")
for i, reasoning in enumerate(biased_reasoning):
    monitor.monitor_generation_step(i, reasoning)

stats = monitor.get_monitoring_statistics()
print(f"Average bias: {stats['average_bias_score']:.2f}")
print(f"Interventions: {stats['interventions_recommended']}")

# Reset
monitor = Stage5Phi2Monitor()

print("\n=== Debiasing Intervention ===")
for i, reasoning in enumerate(biased_reasoning):
    report = monitor.monitor_generation_step(i, reasoning)

    if monitor.should_intervene(i):
        # Generate debiased alternative
        alternative = unbiased_reasoning[i]
        print(f"Step {i}: Debiasing applied")
        print(f"  Original: {reasoning}")
        print(f"  Debiasing: {alternative}")

        # Monitor debiased version
        monitor.monitor_generation_step(
            i + len(biased_reasoning),
            alternative
        )

# Check improvement
final_stats = monitor.get_monitoring_statistics()
print(f"\nFinal average bias: {final_stats['average_bias_score']:.2f}")
```

### Example 3: SCE Integration Workflow

```python
from rese.core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)
from rese.phase1.phi2_integration import (
    SCEPhi2Integrator,
    IntegrationConfig
)

# Setup
sce = SymbolicConstraintEngine()
config = IntegrationConfig(
    auto_check_on_add=True,
    bias_threshold=0.4
)
integrator = SCEPhi2Integrator(sce, config)

# Formulate problem (with automatic bias checking)
constraints = [
    Constraint(
        id="req_1",
        type=ConstraintType.HARD,
        description="The system will certainly achieve perfect results",
        formalization="perfect = true",
        source="stakeholder"
    ),
    Constraint(
        id="req_2",
        type=ConstraintType.HARD,
        description="Response time must be zero",
        formalization="response_time = 0",
        source="stakeholder"
    )
]

print("=== Adding Constraints (with bias checking) ===")
for c in constraints:
    print(f"\nAdding: {c.id}")
    sce.add_constraint(c)
    # Automatically checked and logged!

# Review bias report
print("\n=== Bias Analysis ===")
report = integrator.check_all_constraints()
print(f"Overall bias score: {report.overall_bias_score:.2f}")
print(f"Biased constraints: {len(integrator.get_biased_constraints())}")

# Get suggestions for each biased constraint
print("\n=== Debiasing Suggestions ===")
for constraint_id in ["req_1", "req_2"]:
    print(f"\n{constraint_id}:")
    suggestions = integrator.suggest_debiased_formulation(constraint_id)
    for suggestion in suggestions[:3]:
        print(f"  {suggestion}")

# Create debiased versions
print("\n=== Creating Debiasing Versions ===")
debiased_req1 = Constraint(
    id="req_1_debiased",
    type=ConstraintType.HARD,
    description="The system should maintain accuracy above 95%",
    formalization="accuracy >= 0.95",
    source="validated_requirements"
)

debiased_req2 = Constraint(
    id="req_2_debiased",
    type=ConstraintType.HARD,
    description="Response time should be under 100ms",
    formalization="response_time < 100ms",
    source="validated_requirements"
)

sce.add_constraint(debiased_req1)
sce.add_constraint(debiased_req2)

# Verify improvement
final_report = integrator.check_all_constraints()
print(f"\n=== Final Bias Score ===")
print(f"Original: {report.overall_bias_score:.2f}")
print(f"Final: {final_report.overall_bias_score:.2f}")
print(f"Improvement: {(1 - final_report.overall_bias_score/report.overall_bias_score)*100:.1f}%")
```

---

## 10. Troubleshooting

### Issue: Too Many False Positives

**Problem**: Φ₂ flags unbiased content as biased.

**Solution**:
```python
# Increase thresholds
config = IntegrationConfig(
    bias_threshold=0.6,  # Was 0.4
    max_bias_score=0.8   # Was 0.7
)
```

### Issue: Missing Real Biases

**Problem**: Φ₂ fails to detect actual biases.

**Solution**:
```python
# Decrease thresholds
config = IntegrationConfig(
    bias_threshold=0.3,  # Was 0.4
    max_bias_score=0.5   # Was 0.7
)

# Focus on specific high-risk biases
report = detector.analyze_constraints(
    constraints,
    bias_types=[BiasType.CONFIRMATION, BiasType.OVERCONFIDENCE]
)
```

### Issue: Performance Too Slow

**Problem**: Bias detection slows down the system.

**Solution**:
```python
# Disable auto-checking
config = IntegrationConfig(
    auto_check_on_add=False,
    auto_check_on_conflict=False
)

# Check periodically instead
if step % 10 == 0:  # Every 10 steps
    report = integrator.check_all_constraints()
```

### Issue: Logs Too Large

**Problem**: Bias detection logs grow too large.

**Solution**:
```python
# Disable logging
config = IntegrationConfig(
    log_all_detections=False
)

# Or log only high-severity detections
# (Custom implementation required)
```

---

## 11. Future Enhancements

Planned for v2.0:
- Machine learning-based bias detection
- Cross-domain bias pattern recognition
- Automated debiasing application
- Lean 4 formal verification of debiasing strategies
- Integration with Φ₁.₅ (Tacit Assumption Mining)

---

## 12. Support and Contributing

For issues, questions, or contributions:
- Documentation: `rese/docs/phi2_research.md`
- Tests: `rese/tests/phase1/test_cognitive_biases.py`
- Integration Tests: `rese/tests/phase1/test_phi2_integration.py`
- Source: `rese/phase1/cognitive_biases.py`

---

**Φ₂: Enabling clearer, more objective reasoning in the RESE framework.**

*Last Updated: 2025-12-31*
*Version: 1.0*
