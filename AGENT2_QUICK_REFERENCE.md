# Agent 2 Error Analysis and Adversarial Testing - Quick Reference

## Overview

This quick reference guide shows how to use the Phase 2 (Error Analysis and Adversarial Testing) components.

## Module 1: Uncertainty Propagation

### Basic Usage

```python
from uncertainty_propagation import (
    UncertaintyPropagator,
    enumerate_all_errors,
    propagate_uncertainties
)
import numpy as np

# Initialize propagator
propagator = UncertaintyPropagator(random_seed=42)

# Define equipment specifications
equipment_specs = [
    {
        'name': 'Precision Scale',
        'accuracy': 0.001,      # ±0.001g
        'precision': 0.0005,    # ±0.0005g
        'tolerance': 0.002,     # ±0.002g
        'failure_rate': 0.0001  # 0.01% failure rate
    },
    {
        'name': 'Thermometer',
        'accuracy': 0.5,        # ±0.5°C
        'precision': 0.1,       # ±0.1°C
        'tolerance': 1.0,       # ±1.0°C
        'failure_rate': 0.001   # 0.1% failure rate
    }
]

# Enumerate equipment errors
equipment_errors = propagator.enumerate_equipment_errors(equipment_specs)
print(f"Found {len(equipment_errors)} equipment error sources")

# Define material specifications
material_specs = [
    {
        'name': 'Chemical Reagent',
        'property_variations': {
            'purity': 0.01,        # ±1% purity
            'concentration': 0.02  # ±2% concentration
        },
        'impurity_level': 0.001,   # 0.1% impurities
        'batch_variation': 0.005   # ±0.5% batch-to-batch
    }
]

# Enumerate material errors
material_errors = propagator.enumerate_material_errors(material_specs)

# Define measurement specifications
measurement_specs = [
    {
        'name': 'Length Measurement',
        'resolution': 0.001,  # 1mm resolution
        'uncertainty': 0.005,  # ±5mm uncertainty
        'bias': 0.0
    }
]

# Enumerate measurement errors
measurement_errors = propagator.enumerate_measurement_errors(measurement_specs)

# Combine all errors
all_errors = equipment_errors + material_errors + measurement_errors

# Define model function (how errors affect the outcome)
def my_model(error_values):
    """
    Model function that maps error sources to output.

    Args:
        error_values: Array of error values (one per error source)

    Returns:
        Single output value (e.g., success metric)
    """
    # Simple example: weighted sum
    weights = np.array([e.probability_of_occurrence for e in all_errors])
    return np.sum(error_values * weights)

# Run Monte Carlo propagation
result = propagator.monte_carlo_propagation(
    error_sources=all_errors,
    model_function=my_model,
    n_samples=10000,
    target_value=0.0,
    tolerance=1.0
)

# Print results
print(f"\nMonte Carlo Results:")
print(f"Mean: {result.mean:.4f}")
print(f"Std Dev: {result.std:.4f}")
print(f"5th Percentile: {result.percentile_5:.4f}")
print(f"95th Percentile: {result.percentile_95:.4f}")
print(f"95% CI: ({result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f})")
print(f"Probability of Success: {result.probability_of_success:.2%}")

print(f"\nCritical Error Sources (by sensitivity):")
for name, sensitivity in result.critical_error_sources[:5]:
    print(f"  {name}: {sensitivity:.4f}")
```

### Calculating Failure Probability

```python
# Calculate probability of failure
failure_prob = propagator.calculate_failure_probability(
    error_sources=all_errors,
    model_function=my_model,
    failure_threshold=2.0,  # Values > 2.0 are failures
    n_samples=10000
)

print(f"Probability of Failure: {failure_prob:.2%}")
print(f"Probability of Success: {1 - failure_prob:.2%}")
```

### Enumerating Errors from Specifications

```python
# Convenience function to enumerate all errors
all_errors = enumerate_all_errors(
    equipment_specs=equipment_specs,
    material_specs=material_specs,
    measurement_specs=measurement_specs
)

print(f"Total error sources: {len(all_errors)}")

# Inspect individual error sources
for error in all_errors:
    print(f"\n{error.name}:")
    print(f"  Category: {error.category.value}")
    print(f"  Description: {error.description}")
    print(f"  Distribution: {error.distribution.value}")
    print(f"  Nominal Value: {error.nominal_value}")
    print(f"  Tolerance: ±{error.tolerance}")
    print(f"  Probability: {error.probability_of_occurrence:.2%}")
    print(f"  Impact: {error.impact_severity}")
    print(f"  Mitigation: {error.mitigation_strategy}")
```

## Module 2: Red Team Testing

### Basic Red Team Assessment

```python
from red_team import RedTeam

# Initialize red team
red_team = RedTeam()

# Define invention plan content
invention_plan = """
# Invention Plan: Novel Material Synthesis

## Steps:
1. Prepare precursor materials (99.9% purity required)
2. Heat to 500°C for 2 hours
3. Apply pressure of 5 GPa
4. Cool at rate of 10°C/min
5. Characterize product with XRD

## Equipment:
- High-temperature furnace (±5°C accuracy)
- Hydraulic press (±0.1 GPa accuracy)
- X-ray diffractometer
"""

# Perform red team assessment
assessment = red_team.assess_content(
    content=invention_plan,
    content_type="protocol",
    attack_modes=[
        "security scan",
        "edge case exploration",
        "assumption challenge",
        "compliance check",
        "logic verification"
    ]
)

# Print findings
print(f"Red Team found {len(assessment.findings)} vulnerabilities\n")

for i, finding in enumerate(assessment.findings, 1):
    print(f"{i}. [{finding.severity.value.upper()}] {finding.category.value}")
    print(f"   {finding.title}")
    print(f"   {finding.description}")
    if finding.suggested_fix:
        print(f"   Suggested: {finding.suggested_fix}")
    print()
```

### Red Team with Different Attack Strategies

```python
# Systematic attack (comprehensive)
assessment_systematic = red_team.assess_content(
    content=invention_plan,
    content_type="protocol",
    attack_mode="systematic"
)

# Random attack (diverse)
assessment_random = red_team.assess_content(
    content=invention_plan,
    content_type="protocol",
    attack_mode="random"
)

# Focused attack (specific vulnerability types)
assessment_focused = red_team.assess_content(
    content=invention_plan,
    content_type="protocol",
    attack_mode="focused",
    focus_areas=["edge cases", "assumptions", "validation"]
)

# Deep dive attack (thorough analysis)
assessment_deep = red_team.assess_content(
    content=invention_plan,
    content_type="protocol",
    attack_mode="deep_dive"
)
```

## Module 3: Blue Team Defense

### Basic Blue Team Fix Application

```python
from blue_team import BlueTeam, FixPriority
from red_team import RedTeam

# Initialize teams
red_team = RedTeam()
blue_team = BlueTeam(red_team=red_team)

# Get red team assessment
red_assessment = red_team.assess_content(invention_plan, "protocol")

# Apply blue team fixes
blue_assessment = blue_team.fix_content_from_red_team(
    content=invention_plan,
    red_team_assessment=red_assessment,
    content_type="protocol",
    strategy="comprehensive"
)

# Print results
print(f"Original Content Length: {len(blue_assessment.original_content)}")
print(f"Fixed Content Length: {len(blue_assessment.fixed_content)}")
print(f"Improvement Score: {blue_assessment.overall_improvement_score:.2f}%")
print(f"\nFixes Applied: {len(blue_assessment.applied_fixes)}\n")

for i, fix in enumerate(blue_assessment.applied_fixes, 1):
    print(f"{i}. {fix.fix_suggestion.fix_description}")
    print(f"   Type: {fix.fix_suggestion.fix_type.value}")
    print(f"   Priority: {fix.fix_suggestion.priority.value}")
    print(f"   Status: {fix.fix_status}")
    print(f"   Effectiveness: {fix.effectiveness_score:.1f}/100")
    print()

print(f"\nAdditional Fix Suggestions: {len(blue_assessment.fix_suggestions)}\n")

for i, suggestion in enumerate(blue_assessment.fix_suggestions[:5], 1):
    print(f"{i}. {suggestion.fix_description}")
    print(f"   Priority: {suggestion.priority.value}")
    if suggestion.testing_approach:
        print(f"   Testing: {suggestion.testing_approach}")
```

### Different Blue Team Strategies

```python
# Comprehensive: Fix everything
blue_comprehensive = blue_team.fix_content_from_red_team(
    invention_plan, red_assessment, "protocol",
    strategy="comprehensive"
)

# Defensive: Security-focused only
blue_defensive = blue_team.fix_content_from_red_team(
    invention_plan, red_assessment, "protocol",
    strategy="defensive"
)

# Minimal: Essential fixes only
blue_minimal = blue_team.fix_content_from_red_team(
    invention_plan, red_assessment, "protocol",
    strategy="minimal"
)

# Targeted: High-priority only
blue_targeted = blue_team.fix_content_from_red_team(
    invention_plan, red_assessment, "protocol",
    strategy="targeted"
)
```

## Module 4: Integrated Agent 2 Workflow

### Complete Error Analysis and Adversarial Testing

```python
from end_to_end_invention_planner_agent2 import InventionPlannerAgent2
from dataclasses import dataclass

# Define invention goal
@dataclass
class InventionGoal:
    target: str = "Novel material synthesis"
    domain: str = "materials_science"
    key_requirements: list = None
    complexity_score: float = 0.7
    success_definition: str = "Material forms with desired crystal structure"
    constraints: list = None

# Initialize Agent 2
agent2 = InventionPlannerAgent2()

# Define decomposition
decomposition = {
    'steps': [
        {'description': 'Prepare precursor materials', 'duration': '1 hour'},
        {'description': 'Heat to 500°C for 2 hours', 'duration': '2 hours'},
        {'description': 'Apply pressure of 5 GPa', 'duration': '1 hour'},
        {'description': 'Cool at rate of 10°C/min', 'duration': '3 hours'},
        {'description': 'Characterize with XRD', 'duration': '2 hours'}
    ]
}

# Define knowledge base
knowledge = [
    "High-pressure synthesis requires specialized equipment",
    "Temperature uniformity is critical for crystal growth",
    "Cooling rate affects crystal structure formation"
]

# Run error analysis (Task 2.1)
errors = await agent2.analyze_error_sources(
    goal=InventionGoal(),
    decomposition=decomposition,
    knowledge=knowledge,
    error_source_class=ErrorSource  # From main planner
)

print(f"Error Analysis Complete: {len(errors)} error sources identified")

# Run adversarial testing (Task 2.2 & 2.3)
red_findings, blue_fixes = await agent2.red_blue_team_test(
    goal=InventionGoal(),
    decomposition=decomposition,
    errors=errors,
    error_source_class=ErrorSource
)

print(f"\nAdversarial Testing Complete:")
print(f"  Red Team: {len(red_findings)} vulnerabilities found")
print(f"  Blue Team: {len(blue_fixes)} fixes applied")
```

## Error Source Categories

### Equipment Specification Errors
- Accuracy errors (systematic deviation from true value)
- Precision errors (repeatability)
- Equipment failures (random failures based on MTBF)

### Material Property Errors
- Property variations (purity, concentration, dimensions)
- Impurities (contaminants)
- Batch-to-batch variations

### Measurement Errors
- Resolution errors (quantization)
- Uncertainty (measurement precision)
- Bias errors (systematic offset)

### Environmental Errors
- Temperature variations
- Humidity effects
- Pressure fluctuations
- Vibration interference

### Human Errors
- Procedural mistakes
- Interpretation errors
- Timing errors
- Communication errors

### Systematic Errors
- Calibration drift
- Equipment aging
- Environmental drift
- Methodological biases

## Probability Distributions

### Normal Distribution
```python
# For: Accuracy, precision, most measurement errors
ErrorSource(
    name="measurement_error",
    distribution=ProbabilityDistribution.NORMAL,
    distribution_params={'mean': 0.0, 'std': 0.1},
    nominal_value=0.0,
    tolerance=0.3
)
```

### Uniform Distribution
```python
# For: Resolution errors, quantization
ErrorSource(
    name="quantization_error",
    distribution=ProbabilityDistribution.UNIFORM,
    distribution_params={},
    nominal_value=0.0,
    tolerance=0.5
)
```

### Log-Normal Distribution
```python
# For: Impurities, concentrations
ErrorSource(
    name="impurity_level",
    distribution=ProbabilityDistribution.LOGNORMAL,
    distribution_params={'mean': 0.001, 'sigma': 0.0005},
    nominal_value=0.0,
    tolerance=0.002
)
```

### Exponential Distribution
```python
# For: Equipment failures, time-to-failure
ErrorSource(
    name="equipment_failure",
    distribution=ProbabilityDistribution.EXPONENTIAL,
    distribution_params={'scale': 1000},  # Mean time to failure
    nominal_value=0.0,
    tolerance=1.0
)
```

## Tips and Best Practices

1. **Use Real Specifications**: Always use actual equipment tolerances and specifications from manufacturers.

2. **Monte Carlo Sample Size**:
   - Quick analysis: 1,000-5,000 samples
   - Standard analysis: 10,000 samples
   - High precision: 50,000+ samples

3. **Error Source Consolidation**: The system automatically removes duplicate errors, but provide unique descriptions to help consolidation.

4. **Sensitivity Analysis**: Critical errors identified by sensitivity analysis should be prioritized for mitigation.

5. **Red/Blue Team Iteration**: Run multiple red/blue team cycles for thorough testing:
   - First pass: Identify major vulnerabilities
   - Apply blue team fixes
   - Second pass: Check for remaining issues
   - Iterate until no critical findings

6. **Blue Team Strategy Selection**:
   - Use "comprehensive" for final production plans
   - Use "minimal" for rapid prototyping
   - Use "defensive" for safety-critical systems
   - Use "targeted" for focused improvements

## Troubleshooting

### Import Errors
```python
# If imports fail, check that these files exist:
# - uncertainty_propagation.py
# - red_team.py
# - blue_team.py
# - sovereign_problem_analyzer.py
```

### Monte Carlo Slow Performance
```python
# Reduce sample size for faster execution
result = propagator.monte_carlo_propagation(
    errors, model_function, n_samples=1000  # Instead of 10000
)
```

### Red Team Not Finding Issues
```python
# Try different attack modes
assessment = red_team.assess_content(
    content,
    content_type="protocol",
    attack_modes=["deep_dive", "edge case exploration"]
)
```

### Blue Team Not Applying Fixes
```python
# Check that red findings are in correct format
# Ensure findings have severity, category, and description
print(f"Red findings: {red_findings}")  # Inspect before passing to blue team
```

## Further Reading

- `uncertainty_propagation.py`: Full documentation of error propagation methods
- `red_team.py`: Red team implementation with attack strategies
- `blue_team.py`: Blue team implementation with fix strategies
- `AGENT2_COMPLETION_REPORT.md`: Detailed implementation report
