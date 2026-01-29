<<<<<<< HEAD
# SOP Template-Based Evolution Guide

**System**: Facet-Specific SOP Evolution using OpenEvolve Ensemble
**Status**: ✅ Complete and Ready for Use

---

## Overview

The template-based evolution system provides **specialized handling for each SOP facet** (section), with:
- **Facet-specific extraction**: Isolate each Part (0-6) automatically
- **Targeted red team attacks**: Different vulnerabilities for each section
- **Specialized blue team strategies**: Context-appropriate fix approaches
- **Custom validators**: Section-specific sanity checks
- **Tailored evaluation criteria**: Weighted by facet importance

---

## Available Facets

| Facet | Part | Description | Key Concerns |
|-------|------|-------------|--------------|
| `environmental` | 0 | Environmental Conditions | Temperature, humidity, pressure, vibration control |
| `equipment` | 1 | Equipment Specifications | Magnetic field, UV curing, thermal stage |
| `materials` | 2 | Materials | Resins, nanoclusters, liquid crystals |
| `execution` | 3 | Execution Protocols | 4 phases, timing, dependencies |
| `quality` | 4 | Quality Control | Acceptance criteria, documentation |
| `safety` | 5 | Safety Protocols | Emergency procedures, PPE, training |
| `validation` | 6 | Validation & Scalability | Scaling laws, batch specifications |

---

## Quick Start

### Method 1: Evolve Entire SOP (All Facets)

```bash
# Evolve all 7 facets at once
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2.txt \
    --ensemble-size 7

# Output:
# SOP_v16.2.txt - Complete evolved SOP
# SOP_v16.2.txt.facet_metadata.json - Detailed results for each facet
```

**Expected Runtime**: ~45 minutes (7 facets × ~6 min each)

---

### Method 2: Evolve Specific Facets Only

```bash
# Evolve only critical facets (e.g., safety and equipment)
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_critical.txt \
    --facets safety equipment

# Evolve execution and quality facets
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_process.txt \
    --facets execution quality
```

**Use Cases**:
- **Rapid iteration**: Focus on high-priority facets first
- **Time constraints**: Evolve most critical sections only
- **Specific concerns**: Address known problem areas

---

### Method 3: Evolve Single Facet

```bash
# Evolve just environmental conditions
python evolve_sop_facets.py \
    --input SOP.txt \
    --output part0_environmental_evolved.txt \
    --facet environmental \
    --single

# Output: Only the evolved Part 0 (not entire SOP)
```

**Use Cases**:
- **Deep dive**: Focus all ensemble power on one section
- **Testing**: Validate evolution on single facet before full run
- **Documentation**: Extract evolved facet for review

---

## Detailed Facet Breakdown

### Part 0: Environmental Conditions

**Red Team Attacks** (6 types):
1. `unrealistic_tolerance` - Temperature/humidity recovery times unrealistic
2. `missing_contingency` - No protocol for environmental excursions
3. `insufficient_monitoring` - Monitoring frequency too low
4. `seasonal_variation` - Seasonal effects not considered
5. `thermal_inertia` - Facility thermal mass not accounted for
6. `hvac_capacity_limit` - HVAC capacity insufficient for load

**Validators** (3 checks):
- ✓ Temperature specs within 15-30°C range
- ✓ Humidity specs within 20-60% range
- ✓ Monitoring frequency ≤ 240 minutes

**Example Evolution**:
```
Original: "Recovery within 15 minutes"
Issue: Commercial HVAC with 100 m² lab requires 30-45 min for 0.4°C step
Evolved: "Recovery within 45 minutes"
```

---

### Part 1: Equipment Specifications

**Red Team Attacks** (6 types):
1. `measurement_uncertainty` - Verification tolerance exceeds equipment capability
2. `equipment_compatibility` - Components incompatible
3. `calibration_traceability` - NIST traceability broken
4. `power_requirement_mismatch` - Power supply insufficient
5. `cooling_capacity_insufficient` - Thermal management inadequate
6. `interlock_ambiguity` - Safety interlock logic unclear

**Validators** (3 checks):
- ✓ Magnetic tolerance achievable with specified equipment
- ✓ UV power density < 100 mW/cm²
- ✓ Thermal range < 100°C (single stage)

**Example Evolution**:
```
Original: "0.500 T ± 0.001 T verified by Lake Shore 425 Hall probe"
Issue: Hall probe accuracy ±3.5 mT, but spec requires ±1 mT
Evolved: "0.500 T ± 0.0005 T verified by Metrolab PT2025 NMR gaussmeter"
```

---

### Part 2: Materials

**Red Team Attacks** (6 types):
1. `chemical_instability` - Materials degrade during process
2. `shelf_life_unrealistic` - Shelf life too long at room temperature
3. `purity_unachievable` - 5-6 nines purity not commercially available
4. `mixing_incompatibility` - Components phase-separate
5. `contamination_risk` - No contamination controls
6. `supply_chain_risk` - Single-source components

**Validators** (3 checks):
- ✓ Purity < 99.999% (5 nines)
- ✓ Shelf life ≤ 365 days at room temp or refrigerated specified
- ✓ Mixing equipment specified

**Example Evolution**:
```
Original: "Shelf life: 30 days at 4°C"
Issue: No specification for room temperature storage
Evolved: "Shelf life: 30 days at 4°C, 7 days at 25°C"
```

---

### Part 3: Execution Protocols

**Red Team Attacks** (6 types):
1. `timing_conflict` - Phase timing overlaps or conflicts
2. `sequential_dependency` - Dependencies unclear
3. `phase_transition_risk` - No verification between phases
4. `measurement_timing` - Measurements at wrong times
5. `equilibrium_time_insufficient` - Systems not at equilibrium
6. `thermal_gradient_issue` - Temperature gradients unaddressed

**Validators** (3 checks):
- ✓ Total time < 48 hours
- ✓ Phase dependencies explicit
- ✓ ≥ 10 verification points

**Example Evolution**:
```
Original: "Phase 1: 720 minutes exact"
Issue: No verification that assembly completed before proceeding
Evolved: "Phase 1: 720 minutes exact
         Verification: At 720 min, confirm > 90% nanoclusters within ± 5 µm before proceeding to Phase 2"
```

---

### Part 4: Quality Control

**Red Team Attacks** (5 types):
1. `unverifiable_criteria` - Criteria can't be measured
2. `missing_acceptance_test` - No test for acceptance
3. `ambiguous_pass_fail` - Pass/fail unclear
4. `insufficient_sampling` - Sample size too small
5. `statistical_weakness` - No statistical basis

**Validators** (3 checks):
- ✓ ≥ 5 numeric acceptance criteria (≥, ≤, etc.)
- ✓ Sample size specified (N, n=)
- ✓ ≥ 5 logging requirements

**Example Evolution**:
```
Original: "Phase 1 acceptance: Visual inspection shows assembly"
Issue: Subjective, not quantifiable
Evolved: "Phase 1 acceptance: ≥ 90% of nanoclusters within ± 5 µm of CAD coordinates by confocal microscopy (5 predetermined locations)"
```

---

### Part 5: Safety Protocols

**Red Team Attacks** (6 types):
1. `missing_emergency_procedure` - No procedure for specific emergency
2. `insufficient_training` - Training requirements inadequate
3. `unsafe_equipment` - Equipment itself hazardous
4. `chemical_exposure_risk` - Chemical handling not addressed
5. `magnetic_hazard` - Magnetic field hazards not mitigated
6. `uv_radiation_risk` - UV exposure not controlled

**Validators** (3 checks):
- ✓ All 5 emergency procedures present (stop, evacuation, first aid, fire, chemical)
- ✓ ≥ 3 training requirements
- ✓ PPE specified (goggles, gloves, lab coat, shoes)

**Example Evolution**:
```
Original: "UV protective goggles worn"
Issue: No optical density specification
Evolved: "UV protective goggles worn (optical density > 4 at 405 nm, ANSI Z87.1 compliant)"
```

---

### Part 6: Validation and Scalability

**Red Team Attacks** (5 types):
1. `scaling_law_invalid` - Scaling laws not physically justified
2. `volume_limit_unspecified` - Maximum volume not specified
3. `batch_size_inconsistency` - Batch size inconsistent with scaling
4. `cost_unrealistic` - Cost estimates unrealistic
5. `yield_unrealistic` - Yield targets too high

**Validators** (3 checks):
- ✓ Scaling law specified (time ∝ volume^x)
- ✓ ≥ 2 batch sizes specified
- ✓ Yield < 99%

**Example Evolution**:
```
Original: "5 L batch (100×): Phase 1 time = 12 × 100^(1/3) ≈ 55.7 hours"
Issue: Magnetic gradient decays exponentially with depth, not captured by V^(1/3)
Evolved: "5 L batch (100×):
         Phase 1 time: 55.7 hours (diffusion-limited)
         ADD: Magnetic gradient correction factor: 1.8× for depth penetration
         Total Phase 1 time: 55.7 × 1.8 ≈ 100.3 hours"
```

---

## Python API Usage

### Evolve Single Facet

```python
from sop_templates import SOPTemplateRegistry, SOPFacet

# Initialize
registry = SOPTemplateRegistry(api_key="your-api-key")

# Evolve environmental conditions only
result = registry.evolve_facet(
    sop_content=open("SOP.txt").read(),
    facet=SOPFacet.ENVIRONMENTAL,
    num_models=7
)

print(f"Vulnerabilities found: {result['vulnerabilities_found']}")
print(f"Quality score: {result['quality_score']:.3f}")
print(f"Evolved facet:\n{result['evolved_content']}")
```

---

### Evolve Multiple Facets

```python
# Evolve safety and equipment only
facets = [SOPFacet.SAFETY, SOPFacet.EQUIPMENT]

results = registry.evolve_entire_sop(
    sop_content=open("SOP.txt").read(),
    facets_to_evolve=facets,
    num_models=7
)

for facet_name, facet_result in results["facets"].items():
    if facet_result["status"] == "EVOLVED":
        print(f"{facet_name}: {facet_result['quality_score']:.3f}")

# Save evolved SOP
with open("SOP_safety_equipment_evolved.txt", 'w') as f:
    f.write(results["evolved_sop"])
```

---

### Convenience Functions

```python
from sop_templates import (
    evolve_environmental_conditions,
    evolve_equipment_specifications,
    evolve_safety_protocols
)

# Evolve Part 0 (environmental)
env_result = evolve_environmental_conditions(
    sop_content=open("SOP.txt").read(),
    api_key="your-api-key"
)

# Evolve Part 1 (equipment)
eq_result = evolve_equipment_specifications(
    sop_content=open("SOP.txt").read(),
    api_key="your-api-key"
)

# Evolve Part 5 (safety)
safety_result = evolve_safety_protocols(
    sop_content=open("SOP.txt").read(),
    api_key="your-api-key"
)
```

---

## Advanced Usage

### Custom Validation

```python
from sop_templates import SOPTemplateRegistry, SOPFacet

registry = SOPTemplateRegistry(api_key="your-api-key")

# Add custom validator for environmental facet
def validate_temp_range(content: str) -> bool:
    """Custom validator: ensure temp range ≤ 10°C"""
    import re
    range_pattern = r'(\d+\.?\d*)\s*°?C\s*to\s*(\d+\.?\d*)\s*°?C'
    matches = re.findall(range_pattern, content)

    for min_str, max_str in matches:
        min_temp = float(min_str)
        max_temp = float(max_str)
        if (max_temp - min_temp) > 10:
            return False
    return True

# Get template and add validator
env_template = registry.templates[SOPFacet.ENVIRONMENTAL]
env_template.facet_specific_validators.append(validate_temp_range)

# Evolve with custom validator
result = registry.evolve_facet(
    sop_content=open("SOP.txt").read(),
    facet=SOPFacet.ENVIRONMENTAL
)

# Check validation results
for vr in result["validation_results"]:
    print(f"{vr['validator']}: {'PASS' if vr['passed'] else 'FAIL'}")
```

---

### Iterative Facet Evolution

```python
"""Evolve facets iteratively until all validators pass"""

from sop_templates import SOPTemplateRegistry, SOPFacet

registry = SOPTemplateRegistry(api_key="your-api-key")

sop_content = open("SOP.txt").read()
max_iterations = 5

for iteration in range(max_iterations):
    print(f"\nIteration {iteration + 1}")

    # Evolve safety facet
    result = registry.evolve_facet(
        sop_content=sop_content,
        facet=SOPFacet.SAFETY,
        num_models=7
    )

    # Check if all validators passed
    if result["all_validators_passed"]:
        print("✓ All validators passed!")
        break

    # Show failed validators
    print("Validators that failed:")
    for vr in result["validation_results"]:
        if not vr["passed"]:
            print(f"  ✗ {vr['validator']}")

    # Update SOP with evolved facet and continue
    sop_content = sop_content.replace(
        result["original_content"],
        result["evolved_content"]
    )

    print(f"Quality score: {result['quality_score']:.3f}")
    print("Continuing to next iteration...")

print(f"\nFinal safety facet after {iteration + 1} iterations:")
print(sop_content)
```

---

## Output Structure

### Facet Metadata JSON

```json
{
  "facet": "Part 1: Equipment Specifications",
  "status": "EVOLVED",
  "vulnerabilities_found": 6,
  "fixes_applied": 6,
  "quality_score": 0.94,
  "consensus_reached": true,
  "all_validators_passed": true,
  "validation_results": [
    {
      "validator": "_validate_magnetic_specs",
      "passed": true
    },
    {
      "validator": "_validate_uv_specs",
      "passed": true
    },
    {
      "validator": "_validate_thermal_specs",
      "passed": true
    }
  ],
  "original_content": "...",
  "evolved_content": "..."
}
```

### Complete SOP Metadata JSON

```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "num_models": 7,
  "overall_status": "SUCCESS",
  "total_vulnerabilities_found": 28,
  "total_fixes_applied": 28,
  "facets": {
    "Part 0: Environmental Conditions": {
      "status": "EVOLVED",
      "vulnerabilities_found": 4,
      "fixes_applied": 4,
      "quality_score": 0.92
    },
    "Part 1: Equipment Specifications": {
      "status": "EVOLVED",
      "vulnerabilities_found": 6,
      "fixes_applied": 6,
      "quality_score": 0.94
    },
    ...
  },
  "evolved_sop": "..."
}
```

---

## Performance Metrics

| Facet | Red Team Time | Blue Team Time | Evaluator Time | Total Time |
|-------|---------------|----------------|----------------|------------|
| Environmental | ~4 min | ~5 min | ~4 min | ~13 min |
| Equipment | ~5 min | ~6 min | ~5 min | ~16 min |
| Materials | ~4 min | ~5 min | ~4 min | ~13 min |
| Execution | ~6 min | ~8 min | ~6 min | ~20 min |
| Quality | ~4 min | ~5 min | ~4 min | ~13 min |
| Safety | ~5 min | ~7 min | ~5 min | ~17 min |
| Validation | ~4 min | ~5 min | ~4 min | ~13 min |
| **Total (All 7)** | **~32 min** | **~41 min** | **~32 min** | **~105 min** |

---

## Best Practices

### 1. Start with Critical Facets

```bash
# First iteration: Focus on safety and equipment (highest risk)
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_safety_first.txt \
    --facets safety equipment
```

### 2. Validate Each Facet Individually

```bash
# Evolve and test one facet at a time
for facet in environmental equipment materials execution quality safety validation; do
    echo "Evolving $facet..."
    python evolve_sop_facets.py \
        --input SOP.txt \
        --output SOP_${facet}.txt \
        --facet $facet \
        --single

    # Review evolved facet
    echo "Review SOP_${facet}.txt before proceeding..."
    read -p "Continue? (y/n) " && [[ $REPLY == y ]] || break
done
```

### 3. Use Larger Ensemble for Critical Facets

```bash
# Safety facet gets more ensemble members
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_safety_high_confidence.txt \
    --facet safety \
    --single \
    --ensemble-size 11  # More models = higher confidence
```

### 4. Compare Facet Evolution Strategies

```python
"""Compare different blue team strategies for same facet"""

from sop_templates import SOPTemplateRegistry, SOPFacet
from blue_team import BlueTeamStrategy

registry = SOPTemplateRegistry(api_key="your-api-key")
sop_content = open("SOP.txt").read()

strategies = [
    BlueTeamStrategy.MINIMAL,
    BlueTeamStrategy.DEFENSIVE,
    BlueTeamStrategy.COMPREHENSIVE
]

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy.value}")
    print(f"{'='*70}")

    # Modify template strategy
    template = registry.templates[SOPFacet.SAFETY]
    template.blue_team_strategy = strategy

    # Evolve
    result = registry.evolve_facet(
        sop_content=sop_content,
        facet=SOPFacet.SAFETY,
        num_models=5  # Faster for comparison
    )

    print(f"Quality Score: {result['quality_score']:.3f}")
    print(f"Fixes Applied: {result['fixes_applied']}")
```

---

## Troubleshooting

### Issue: "Could not extract facet from SOP"

**Cause**: Facet section not found in SOP

**Solution**:
```bash
# Check if facet exists in SOP
grep -i "PART 0" SOP.txt  # Should find "PART 0 — ENVIRONMENTAL"

# If using different naming, modify extractor in sop_templates.py
```

### Issue: "All validators failed"

**Cause**: Evolved facet violates constraints

**Solution**:
```bash
# Try with larger ensemble (more diverse fixes)
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2.txt \
    --facets safety \
    --ensemble-size 11

# Or try different blue team strategy
# Modify sop_templates.py template to use COMPREHENSIVE instead of DEFENSIVE
```

### Issue: "Quality score < 0.70"

**Cause**: Facet has significant issues

**Solution**:
```bash
# Run iterative evolution
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_iter2.txt \
    --facets safety \
    --ensemble-size 9

# Then repeat
python evolve_sop_facets.py \
    --input SOP_v16.2_iter2.txt \
    --output SOP_v16.2_iter3.txt \
    --facets safety \
    --ensemble-size 9
```

---

## Summary

The template-based evolution system provides:

✅ **Facet-specific handling** - Different attacks/strategies for each section
✅ **Automated extraction** - No manual section isolation needed
✅ **Custom validators** - Section-specific sanity checks
✅ **Flexible evolution** - Evolve all facets, some, or one
✅ **Detailed metadata** - Track evolution per facet
✅ **Iterative refinement** - Improve problematic facets until validators pass

**This is the most precise, controlled method for evolving complex technical SOPs.**

---

**End of Guide**
=======
# SOP Template-Based Evolution Guide

**System**: Facet-Specific SOP Evolution using OpenEvolve Ensemble
**Status**: ✅ Complete and Ready for Use

---

## Overview

The template-based evolution system provides **specialized handling for each SOP facet** (section), with:
- **Facet-specific extraction**: Isolate each Part (0-6) automatically
- **Targeted red team attacks**: Different vulnerabilities for each section
- **Specialized blue team strategies**: Context-appropriate fix approaches
- **Custom validators**: Section-specific sanity checks
- **Tailored evaluation criteria**: Weighted by facet importance

---

## Available Facets

| Facet | Part | Description | Key Concerns |
|-------|------|-------------|--------------|
| `environmental` | 0 | Environmental Conditions | Temperature, humidity, pressure, vibration control |
| `equipment` | 1 | Equipment Specifications | Magnetic field, UV curing, thermal stage |
| `materials` | 2 | Materials | Resins, nanoclusters, liquid crystals |
| `execution` | 3 | Execution Protocols | 4 phases, timing, dependencies |
| `quality` | 4 | Quality Control | Acceptance criteria, documentation |
| `safety` | 5 | Safety Protocols | Emergency procedures, PPE, training |
| `validation` | 6 | Validation & Scalability | Scaling laws, batch specifications |

---

## Quick Start

### Method 1: Evolve Entire SOP (All Facets)

```bash
# Evolve all 7 facets at once
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2.txt \
    --ensemble-size 7

# Output:
# SOP_v16.2.txt - Complete evolved SOP
# SOP_v16.2.txt.facet_metadata.json - Detailed results for each facet
```

**Expected Runtime**: ~45 minutes (7 facets × ~6 min each)

---

### Method 2: Evolve Specific Facets Only

```bash
# Evolve only critical facets (e.g., safety and equipment)
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_critical.txt \
    --facets safety equipment

# Evolve execution and quality facets
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_process.txt \
    --facets execution quality
```

**Use Cases**:
- **Rapid iteration**: Focus on high-priority facets first
- **Time constraints**: Evolve most critical sections only
- **Specific concerns**: Address known problem areas

---

### Method 3: Evolve Single Facet

```bash
# Evolve just environmental conditions
python evolve_sop_facets.py \
    --input SOP.txt \
    --output part0_environmental_evolved.txt \
    --facet environmental \
    --single

# Output: Only the evolved Part 0 (not entire SOP)
```

**Use Cases**:
- **Deep dive**: Focus all ensemble power on one section
- **Testing**: Validate evolution on single facet before full run
- **Documentation**: Extract evolved facet for review

---

## Detailed Facet Breakdown

### Part 0: Environmental Conditions

**Red Team Attacks** (6 types):
1. `unrealistic_tolerance` - Temperature/humidity recovery times unrealistic
2. `missing_contingency` - No protocol for environmental excursions
3. `insufficient_monitoring` - Monitoring frequency too low
4. `seasonal_variation` - Seasonal effects not considered
5. `thermal_inertia` - Facility thermal mass not accounted for
6. `hvac_capacity_limit` - HVAC capacity insufficient for load

**Validators** (3 checks):
- ✓ Temperature specs within 15-30°C range
- ✓ Humidity specs within 20-60% range
- ✓ Monitoring frequency ≤ 240 minutes

**Example Evolution**:
```
Original: "Recovery within 15 minutes"
Issue: Commercial HVAC with 100 m² lab requires 30-45 min for 0.4°C step
Evolved: "Recovery within 45 minutes"
```

---

### Part 1: Equipment Specifications

**Red Team Attacks** (6 types):
1. `measurement_uncertainty` - Verification tolerance exceeds equipment capability
2. `equipment_compatibility` - Components incompatible
3. `calibration_traceability` - NIST traceability broken
4. `power_requirement_mismatch` - Power supply insufficient
5. `cooling_capacity_insufficient` - Thermal management inadequate
6. `interlock_ambiguity` - Safety interlock logic unclear

**Validators** (3 checks):
- ✓ Magnetic tolerance achievable with specified equipment
- ✓ UV power density < 100 mW/cm²
- ✓ Thermal range < 100°C (single stage)

**Example Evolution**:
```
Original: "0.500 T ± 0.001 T verified by Lake Shore 425 Hall probe"
Issue: Hall probe accuracy ±3.5 mT, but spec requires ±1 mT
Evolved: "0.500 T ± 0.0005 T verified by Metrolab PT2025 NMR gaussmeter"
```

---

### Part 2: Materials

**Red Team Attacks** (6 types):
1. `chemical_instability` - Materials degrade during process
2. `shelf_life_unrealistic` - Shelf life too long at room temperature
3. `purity_unachievable` - 5-6 nines purity not commercially available
4. `mixing_incompatibility` - Components phase-separate
5. `contamination_risk` - No contamination controls
6. `supply_chain_risk` - Single-source components

**Validators** (3 checks):
- ✓ Purity < 99.999% (5 nines)
- ✓ Shelf life ≤ 365 days at room temp or refrigerated specified
- ✓ Mixing equipment specified

**Example Evolution**:
```
Original: "Shelf life: 30 days at 4°C"
Issue: No specification for room temperature storage
Evolved: "Shelf life: 30 days at 4°C, 7 days at 25°C"
```

---

### Part 3: Execution Protocols

**Red Team Attacks** (6 types):
1. `timing_conflict` - Phase timing overlaps or conflicts
2. `sequential_dependency` - Dependencies unclear
3. `phase_transition_risk` - No verification between phases
4. `measurement_timing` - Measurements at wrong times
5. `equilibrium_time_insufficient` - Systems not at equilibrium
6. `thermal_gradient_issue` - Temperature gradients unaddressed

**Validators** (3 checks):
- ✓ Total time < 48 hours
- ✓ Phase dependencies explicit
- ✓ ≥ 10 verification points

**Example Evolution**:
```
Original: "Phase 1: 720 minutes exact"
Issue: No verification that assembly completed before proceeding
Evolved: "Phase 1: 720 minutes exact
         Verification: At 720 min, confirm > 90% nanoclusters within ± 5 µm before proceeding to Phase 2"
```

---

### Part 4: Quality Control

**Red Team Attacks** (5 types):
1. `unverifiable_criteria` - Criteria can't be measured
2. `missing_acceptance_test` - No test for acceptance
3. `ambiguous_pass_fail` - Pass/fail unclear
4. `insufficient_sampling` - Sample size too small
5. `statistical_weakness` - No statistical basis

**Validators** (3 checks):
- ✓ ≥ 5 numeric acceptance criteria (≥, ≤, etc.)
- ✓ Sample size specified (N, n=)
- ✓ ≥ 5 logging requirements

**Example Evolution**:
```
Original: "Phase 1 acceptance: Visual inspection shows assembly"
Issue: Subjective, not quantifiable
Evolved: "Phase 1 acceptance: ≥ 90% of nanoclusters within ± 5 µm of CAD coordinates by confocal microscopy (5 predetermined locations)"
```

---

### Part 5: Safety Protocols

**Red Team Attacks** (6 types):
1. `missing_emergency_procedure` - No procedure for specific emergency
2. `insufficient_training` - Training requirements inadequate
3. `unsafe_equipment` - Equipment itself hazardous
4. `chemical_exposure_risk` - Chemical handling not addressed
5. `magnetic_hazard` - Magnetic field hazards not mitigated
6. `uv_radiation_risk` - UV exposure not controlled

**Validators** (3 checks):
- ✓ All 5 emergency procedures present (stop, evacuation, first aid, fire, chemical)
- ✓ ≥ 3 training requirements
- ✓ PPE specified (goggles, gloves, lab coat, shoes)

**Example Evolution**:
```
Original: "UV protective goggles worn"
Issue: No optical density specification
Evolved: "UV protective goggles worn (optical density > 4 at 405 nm, ANSI Z87.1 compliant)"
```

---

### Part 6: Validation and Scalability

**Red Team Attacks** (5 types):
1. `scaling_law_invalid` - Scaling laws not physically justified
2. `volume_limit_unspecified` - Maximum volume not specified
3. `batch_size_inconsistency` - Batch size inconsistent with scaling
4. `cost_unrealistic` - Cost estimates unrealistic
5. `yield_unrealistic` - Yield targets too high

**Validators** (3 checks):
- ✓ Scaling law specified (time ∝ volume^x)
- ✓ ≥ 2 batch sizes specified
- ✓ Yield < 99%

**Example Evolution**:
```
Original: "5 L batch (100×): Phase 1 time = 12 × 100^(1/3) ≈ 55.7 hours"
Issue: Magnetic gradient decays exponentially with depth, not captured by V^(1/3)
Evolved: "5 L batch (100×):
         Phase 1 time: 55.7 hours (diffusion-limited)
         ADD: Magnetic gradient correction factor: 1.8× for depth penetration
         Total Phase 1 time: 55.7 × 1.8 ≈ 100.3 hours"
```

---

## Python API Usage

### Evolve Single Facet

```python
from sop_templates import SOPTemplateRegistry, SOPFacet

# Initialize
registry = SOPTemplateRegistry(api_key="your-api-key")

# Evolve environmental conditions only
result = registry.evolve_facet(
    sop_content=open("SOP.txt").read(),
    facet=SOPFacet.ENVIRONMENTAL,
    num_models=7
)

print(f"Vulnerabilities found: {result['vulnerabilities_found']}")
print(f"Quality score: {result['quality_score']:.3f}")
print(f"Evolved facet:\n{result['evolved_content']}")
```

---

### Evolve Multiple Facets

```python
# Evolve safety and equipment only
facets = [SOPFacet.SAFETY, SOPFacet.EQUIPMENT]

results = registry.evolve_entire_sop(
    sop_content=open("SOP.txt").read(),
    facets_to_evolve=facets,
    num_models=7
)

for facet_name, facet_result in results["facets"].items():
    if facet_result["status"] == "EVOLVED":
        print(f"{facet_name}: {facet_result['quality_score']:.3f}")

# Save evolved SOP
with open("SOP_safety_equipment_evolved.txt", 'w') as f:
    f.write(results["evolved_sop"])
```

---

### Convenience Functions

```python
from sop_templates import (
    evolve_environmental_conditions,
    evolve_equipment_specifications,
    evolve_safety_protocols
)

# Evolve Part 0 (environmental)
env_result = evolve_environmental_conditions(
    sop_content=open("SOP.txt").read(),
    api_key="your-api-key"
)

# Evolve Part 1 (equipment)
eq_result = evolve_equipment_specifications(
    sop_content=open("SOP.txt").read(),
    api_key="your-api-key"
)

# Evolve Part 5 (safety)
safety_result = evolve_safety_protocols(
    sop_content=open("SOP.txt").read(),
    api_key="your-api-key"
)
```

---

## Advanced Usage

### Custom Validation

```python
from sop_templates import SOPTemplateRegistry, SOPFacet

registry = SOPTemplateRegistry(api_key="your-api-key")

# Add custom validator for environmental facet
def validate_temp_range(content: str) -> bool:
    """Custom validator: ensure temp range ≤ 10°C"""
    import re
    range_pattern = r'(\d+\.?\d*)\s*°?C\s*to\s*(\d+\.?\d*)\s*°?C'
    matches = re.findall(range_pattern, content)

    for min_str, max_str in matches:
        min_temp = float(min_str)
        max_temp = float(max_str)
        if (max_temp - min_temp) > 10:
            return False
    return True

# Get template and add validator
env_template = registry.templates[SOPFacet.ENVIRONMENTAL]
env_template.facet_specific_validators.append(validate_temp_range)

# Evolve with custom validator
result = registry.evolve_facet(
    sop_content=open("SOP.txt").read(),
    facet=SOPFacet.ENVIRONMENTAL
)

# Check validation results
for vr in result["validation_results"]:
    print(f"{vr['validator']}: {'PASS' if vr['passed'] else 'FAIL'}")
```

---

### Iterative Facet Evolution

```python
"""Evolve facets iteratively until all validators pass"""

from sop_templates import SOPTemplateRegistry, SOPFacet

registry = SOPTemplateRegistry(api_key="your-api-key")

sop_content = open("SOP.txt").read()
max_iterations = 5

for iteration in range(max_iterations):
    print(f"\nIteration {iteration + 1}")

    # Evolve safety facet
    result = registry.evolve_facet(
        sop_content=sop_content,
        facet=SOPFacet.SAFETY,
        num_models=7
    )

    # Check if all validators passed
    if result["all_validators_passed"]:
        print("✓ All validators passed!")
        break

    # Show failed validators
    print("Validators that failed:")
    for vr in result["validation_results"]:
        if not vr["passed"]:
            print(f"  ✗ {vr['validator']}")

    # Update SOP with evolved facet and continue
    sop_content = sop_content.replace(
        result["original_content"],
        result["evolved_content"]
    )

    print(f"Quality score: {result['quality_score']:.3f}")
    print("Continuing to next iteration...")

print(f"\nFinal safety facet after {iteration + 1} iterations:")
print(sop_content)
```

---

## Output Structure

### Facet Metadata JSON

```json
{
  "facet": "Part 1: Equipment Specifications",
  "status": "EVOLVED",
  "vulnerabilities_found": 6,
  "fixes_applied": 6,
  "quality_score": 0.94,
  "consensus_reached": true,
  "all_validators_passed": true,
  "validation_results": [
    {
      "validator": "_validate_magnetic_specs",
      "passed": true
    },
    {
      "validator": "_validate_uv_specs",
      "passed": true
    },
    {
      "validator": "_validate_thermal_specs",
      "passed": true
    }
  ],
  "original_content": "...",
  "evolved_content": "..."
}
```

### Complete SOP Metadata JSON

```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "num_models": 7,
  "overall_status": "SUCCESS",
  "total_vulnerabilities_found": 28,
  "total_fixes_applied": 28,
  "facets": {
    "Part 0: Environmental Conditions": {
      "status": "EVOLVED",
      "vulnerabilities_found": 4,
      "fixes_applied": 4,
      "quality_score": 0.92
    },
    "Part 1: Equipment Specifications": {
      "status": "EVOLVED",
      "vulnerabilities_found": 6,
      "fixes_applied": 6,
      "quality_score": 0.94
    },
    ...
  },
  "evolved_sop": "..."
}
```

---

## Performance Metrics

| Facet | Red Team Time | Blue Team Time | Evaluator Time | Total Time |
|-------|---------------|----------------|----------------|------------|
| Environmental | ~4 min | ~5 min | ~4 min | ~13 min |
| Equipment | ~5 min | ~6 min | ~5 min | ~16 min |
| Materials | ~4 min | ~5 min | ~4 min | ~13 min |
| Execution | ~6 min | ~8 min | ~6 min | ~20 min |
| Quality | ~4 min | ~5 min | ~4 min | ~13 min |
| Safety | ~5 min | ~7 min | ~5 min | ~17 min |
| Validation | ~4 min | ~5 min | ~4 min | ~13 min |
| **Total (All 7)** | **~32 min** | **~41 min** | **~32 min** | **~105 min** |

---

## Best Practices

### 1. Start with Critical Facets

```bash
# First iteration: Focus on safety and equipment (highest risk)
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_safety_first.txt \
    --facets safety equipment
```

### 2. Validate Each Facet Individually

```bash
# Evolve and test one facet at a time
for facet in environmental equipment materials execution quality safety validation; do
    echo "Evolving $facet..."
    python evolve_sop_facets.py \
        --input SOP.txt \
        --output SOP_${facet}.txt \
        --facet $facet \
        --single

    # Review evolved facet
    echo "Review SOP_${facet}.txt before proceeding..."
    read -p "Continue? (y/n) " && [[ $REPLY == y ]] || break
done
```

### 3. Use Larger Ensemble for Critical Facets

```bash
# Safety facet gets more ensemble members
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_safety_high_confidence.txt \
    --facet safety \
    --single \
    --ensemble-size 11  # More models = higher confidence
```

### 4. Compare Facet Evolution Strategies

```python
"""Compare different blue team strategies for same facet"""

from sop_templates import SOPTemplateRegistry, SOPFacet
from blue_team import BlueTeamStrategy

registry = SOPTemplateRegistry(api_key="your-api-key")
sop_content = open("SOP.txt").read()

strategies = [
    BlueTeamStrategy.MINIMAL,
    BlueTeamStrategy.DEFENSIVE,
    BlueTeamStrategy.COMPREHENSIVE
]

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy.value}")
    print(f"{'='*70}")

    # Modify template strategy
    template = registry.templates[SOPFacet.SAFETY]
    template.blue_team_strategy = strategy

    # Evolve
    result = registry.evolve_facet(
        sop_content=sop_content,
        facet=SOPFacet.SAFETY,
        num_models=5  # Faster for comparison
    )

    print(f"Quality Score: {result['quality_score']:.3f}")
    print(f"Fixes Applied: {result['fixes_applied']}")
```

---

## Troubleshooting

### Issue: "Could not extract facet from SOP"

**Cause**: Facet section not found in SOP

**Solution**:
```bash
# Check if facet exists in SOP
grep -i "PART 0" SOP.txt  # Should find "PART 0 — ENVIRONMENTAL"

# If using different naming, modify extractor in sop_templates.py
```

### Issue: "All validators failed"

**Cause**: Evolved facet violates constraints

**Solution**:
```bash
# Try with larger ensemble (more diverse fixes)
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2.txt \
    --facets safety \
    --ensemble-size 11

# Or try different blue team strategy
# Modify sop_templates.py template to use COMPREHENSIVE instead of DEFENSIVE
```

### Issue: "Quality score < 0.70"

**Cause**: Facet has significant issues

**Solution**:
```bash
# Run iterative evolution
python evolve_sop_facets.py \
    --input SOP.txt \
    --output SOP_v16.2_iter2.txt \
    --facets safety \
    --ensemble-size 9

# Then repeat
python evolve_sop_facets.py \
    --input SOP_v16.2_iter2.txt \
    --output SOP_v16.2_iter3.txt \
    --facets safety \
    --ensemble-size 9
```

---

## Summary

The template-based evolution system provides:

✅ **Facet-specific handling** - Different attacks/strategies for each section
✅ **Automated extraction** - No manual section isolation needed
✅ **Custom validators** - Section-specific sanity checks
✅ **Flexible evolution** - Evolve all facets, some, or one
✅ **Detailed metadata** - Track evolution per facet
✅ **Iterative refinement** - Improve problematic facets until validators pass

**This is the most precise, controlled method for evolving complex technical SOPs.**

---

**End of Guide**
>>>>>>> 1cb9c5e35 (update)
