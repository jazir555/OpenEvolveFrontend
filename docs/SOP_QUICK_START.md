<<<<<<< HEAD
# SOP Evolution Quick Start Guide

## Overview

Use the integrated Red Team / Blue Team / Evaluator ensemble system to evolve and improve technical Standard Operating Procedures.

**Files Created**:
1. `SOP_EVOLUTION_FRAMEWORK.md` - Complete framework documentation
2. `evolve_sop.py` - Automated evolution script

---

## Quick Start (3 Methods)

### Method 1: Command-Line Script (Recommended)

```bash
# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Run evolution
python evolve_sop.py \
    --input SOP.txt \
    --output SOP_v16.2.txt \
    --iterations 3 \
    --threshold 0.90 \
    --ensemble-size 7 \
    --save-intermediate

# Output:
# SOP_v16.2.txt - Evolved SOP
# SOP_v16.2.txt.metadata.json - Evolution history
# SOP_v16.2.txt.iter1, iter2, ... - Intermediate versions (if --save-intermediate)
```

---

### Method 2: Python API

```python
from evolve_sop import SOPEvolver

# Initialize evolver
evolver = SOPEvolver(
    api_key="your-api-key",
    red_team_models=["gpt-4o", "claude-3-opus"],
    blue_team_models=["gpt-4o", "claude-3-opus"],
    evaluator_models=["gpt-4o", "claude-3-opus", "gemini-ultra"],
    num_ensemble_models=7
)

# Run evolution
results = evolver.evolve_sop(
    input_sop_path="SOP.txt",
    output_sop_path="SOP_v16.2.txt",
    max_iterations=3,
    quality_threshold=0.90,
    save_intermediate=True
)

print(f"Final quality score: {results['iterations'][-1]['quality_score']}")
```

---

### Method 3: Direct Team Integration

```python
from red_team import RedTeam
from blue_team import BlueTeam, BlueTeamStrategy
from evaluator_team import EvaluatorTeam

# Initialize teams
red_team = RedTeam()
blue_team = BlueTeam()
evaluator = EvaluatorTeam()

# Step 1: Red Team Analysis
print("Analyzing SOP for vulnerabilities...")
red_result = red_team.analyze_with_ensemble(
    content=open("SOP.txt").read(),
    content_type="technical_sop",
    api_key="your-api-key",
    num_models=7,
    attack_types=[
        "unrealistic_tolerance",
        "safety_vulnerability",
        "scaling_limitation"
    ]
)
print(f"Found {len(red_result.vulnerabilities)} vulnerabilities")

# Step 2: Blue Team Fixes
print("Generating fixes...")
from red_team import IssueFinding, IssueCategory, SeverityLevel

issues = [
    IssueFinding(
        title=vuln.get('title'),
        description=vuln.get('description'),
        severity=SeverityLevel.HIGH,
        category=IssueCategory.LOGICAL_ERROR,
        confidence=0.8
    )
    for vuln in red_result.vulnerabilities
]

blue_result = blue_team.generate_solutions_with_ensemble(
    issues=issues,
    content=open("SOP.txt").read(),
    content_type="technical_sop",
    api_key="your-api-key",
    num_models=7,
    strategy=BlueTeamStrategy.COMPREHENSIVE
)

# Save fixed SOP
with open("SOP_fixed.txt", 'w') as f:
    f.write(blue_result.fixed_content)

# Step 3: Evaluation
print("Evaluating quality...")
eval_result = evaluator.evaluate_with_ensemble(
    content=open("SOP_fixed.txt").read(),
    content_type="technical_sop",
    api_key="your-api-key",
    num_models=9
)

print(f"Quality score: {eval_result.consensus_score:.3f}")
print(f"Consensus reached: {eval_result.consensus_reached}")
print(f"Verdict: {eval_result.final_verdict}")
```

---

## Understanding the Output

### Quality Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.95 - 1.00 | Excellent | Deploy immediately |
| 0.90 - 0.95 | Very Good | Deploy with monitoring |
| 0.80 - 0.90 | Good | Consider another iteration |
| 0.70 - 0.80 | Fair | Requires fixes |
| < 0.70 | Poor | Significant issues |

### Consensus Score Breakdown

```
Consensus: 0.94
  ├─ Physical Realizability: 0.95
  ├─ Safety: 0.98
  ├─ Verifiability: 0.92
  ├─ Operational Clarity: 0.91
  └─ Scalability: 0.94
```

### Vulnerability Categories

1. **unrealistic_tolerance** - Tolerances too tight for real equipment
2. **missing_contingency** - Unhandled failure modes
3. **contradictory_requirement** - Conflicting specifications
4. **safety_vulnerability** - Dangerous edge cases
5. **scaling_limitation** - Issues blocking scale-up
6. **measurement_uncertainty** - Unverified measurement claims
7. **timing_conflict** - Impossible temporal constraints

---

## Expected Evolution Trajectory for SOP.txt

### Iteration 1: v16.1 → v16.1.1
**Vulnerabilities Found**: 4
1. Temperature recovery time (15 min → 45 min)
2. Magnetic field verification tolerance
3. UV interlock ambiguity
4. Gradient non-uniformity at scale

**Fixes Applied**: 4
**Quality Score**: 0.69 → 0.94 (+0.26)
**Status**: ✓ PASSED threshold (0.90)

**Expected Outcome**: Single iteration sufficient to reach 0.90 threshold

---

## Customizing Evolution

### Adjusting Ensemble Size

```python
# Small (fast, less diverse)
num_models=3  # ~3 minutes per iteration

# Medium (balanced)
num_models=7  # ~7 minutes per iteration (default)

# Large (thorough, more diverse)
num_models=11  # ~15 minutes per iteration
```

### Adjusting Attack Types

```python
# Comprehensive (all categories)
attack_types=[
    "unrealistic_tolerance",
    "missing_contingency",
    "contradictory_requirement",
    "safety_vulnerability",
    "scaling_limitation",
    "measurement_uncertainty",
    "timing_conflict",
    "material_degradation",
    "human_error_prone",
    "calibration_drift"
]

# Focused (safety-critical only)
attack_types=[
    "safety_vulnerability",
    "contradictory_requirement",
    "missing_contingency"
]
```

### Adjusting Fix Strategy

```python
# Minimal changes (conservative)
strategy=BlueTeamStrategy.MINIMAL

# Safety-focused (defense in depth)
strategy=BlueTeamStrategy.DEFENSIVE

# Comprehensive improvements (default)
strategy=BlueTeamStrategy.COMPREHENSIVE
```

---

## Troubleshooting

### Issue: "OpenEvolve not available"
**Solution**: Ensure OpenEvolve is installed
```bash
cd openevolve
pip install -e .
```

### Issue: Low quality score (< 0.70)
**Solution**: Increase iterations or ensemble size
```bash
python evolve_sop.py --input SOP.txt --output out.txt --iterations 5 --ensemble-size 11
```

### Issue: No vulnerabilities found
**Possible causes**:
1. SOP is already excellent (verify by checking quality score)
2. Attack types not comprehensive enough
3. Ensemble size too small (increase to 11+)

### Issue: Fix introduces new vulnerabilities
**Solution**: Run additional iteration (ensemble will catch it)
```bash
python evolve_sop.py --input SOP_v16.2.txt --output SOP_v16.3.txt --iterations 2
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Quality score ≥ 0.90
- [ ] All safety vulnerabilities resolved
- [ ] Backward compatibility verified
- [ ] Cost impact assessed
- [ ] Stakeholder approval obtained
- [ ] Rollback plan documented

### Deployment Steps

1. **Shadow Mode** (7 days)
   ```bash
   # Run both old and new SOP in parallel
   python compare_sop_versions.py --old v16.1 --new v16.2 --days 7
   ```

2. **Staging** (30 days)
   ```bash
   python deploy_to_staging.py --sop v16.2 --monitor
   ```

3. **Production**
   ```bash
   python deploy_to_production.py --sop v16.2 --monitor-days 30
   ```

---

## Best Practices

1. **Start with small ensemble** (3-5 models) for rapid prototyping
2. **Increase ensemble size** (7-11 models) for production-ready evolution
3. **Use comprehensive attack types** for initial evolution
4. **Focus on safety-critical** attacks for subsequent iterations
5. **Always save intermediate versions** for rollback capability
6. **Review metadata.json** to understand evolution history
7. **Validate physically** by running actual experiments if possible

---

## Next Steps

1. **Run first evolution**:
   ```bash
   python evolve_sop.py --input SOP.txt --output SOP_v16.2.txt
   ```

2. **Review results**:
   ```bash
   cat SOP_v16.2.txt.metadata.json | python -m json.tool
   ```

3. **Compare versions**:
   ```bash
   diff -u SOP.txt SOP_v16.2.txt | head -100
   ```

4. **Deploy if quality ≥ 0.90**

5. **Continuous improvement**: Run evolution monthly with production feedback

---

## Questions?

Refer to detailed framework documentation: `SOP_EVOLUTION_FRAMEWORK.md`

---

**End of Quick Start Guide**
=======
# SOP Evolution Quick Start Guide

## Overview

Use the integrated Red Team / Blue Team / Evaluator ensemble system to evolve and improve technical Standard Operating Procedures.

**Files Created**:
1. `SOP_EVOLUTION_FRAMEWORK.md` - Complete framework documentation
2. `evolve_sop.py` - Automated evolution script

---

## Quick Start (3 Methods)

### Method 1: Command-Line Script (Recommended)

```bash
# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Run evolution
python evolve_sop.py \
    --input SOP.txt \
    --output SOP_v16.2.txt \
    --iterations 3 \
    --threshold 0.90 \
    --ensemble-size 7 \
    --save-intermediate

# Output:
# SOP_v16.2.txt - Evolved SOP
# SOP_v16.2.txt.metadata.json - Evolution history
# SOP_v16.2.txt.iter1, iter2, ... - Intermediate versions (if --save-intermediate)
```

---

### Method 2: Python API

```python
from evolve_sop import SOPEvolver

# Initialize evolver
evolver = SOPEvolver(
    api_key="your-api-key",
    red_team_models=["gpt-4o", "claude-3-opus"],
    blue_team_models=["gpt-4o", "claude-3-opus"],
    evaluator_models=["gpt-4o", "claude-3-opus", "gemini-ultra"],
    num_ensemble_models=7
)

# Run evolution
results = evolver.evolve_sop(
    input_sop_path="SOP.txt",
    output_sop_path="SOP_v16.2.txt",
    max_iterations=3,
    quality_threshold=0.90,
    save_intermediate=True
)

print(f"Final quality score: {results['iterations'][-1]['quality_score']}")
```

---

### Method 3: Direct Team Integration

```python
from red_team import RedTeam
from blue_team import BlueTeam, BlueTeamStrategy
from evaluator_team import EvaluatorTeam

# Initialize teams
red_team = RedTeam()
blue_team = BlueTeam()
evaluator = EvaluatorTeam()

# Step 1: Red Team Analysis
print("Analyzing SOP for vulnerabilities...")
red_result = red_team.analyze_with_ensemble(
    content=open("SOP.txt").read(),
    content_type="technical_sop",
    api_key="your-api-key",
    num_models=7,
    attack_types=[
        "unrealistic_tolerance",
        "safety_vulnerability",
        "scaling_limitation"
    ]
)
print(f"Found {len(red_result.vulnerabilities)} vulnerabilities")

# Step 2: Blue Team Fixes
print("Generating fixes...")
from red_team import IssueFinding, IssueCategory, SeverityLevel

issues = [
    IssueFinding(
        title=vuln.get('title'),
        description=vuln.get('description'),
        severity=SeverityLevel.HIGH,
        category=IssueCategory.LOGICAL_ERROR,
        confidence=0.8
    )
    for vuln in red_result.vulnerabilities
]

blue_result = blue_team.generate_solutions_with_ensemble(
    issues=issues,
    content=open("SOP.txt").read(),
    content_type="technical_sop",
    api_key="your-api-key",
    num_models=7,
    strategy=BlueTeamStrategy.COMPREHENSIVE
)

# Save fixed SOP
with open("SOP_fixed.txt", 'w') as f:
    f.write(blue_result.fixed_content)

# Step 3: Evaluation
print("Evaluating quality...")
eval_result = evaluator.evaluate_with_ensemble(
    content=open("SOP_fixed.txt").read(),
    content_type="technical_sop",
    api_key="your-api-key",
    num_models=9
)

print(f"Quality score: {eval_result.consensus_score:.3f}")
print(f"Consensus reached: {eval_result.consensus_reached}")
print(f"Verdict: {eval_result.final_verdict}")
```

---

## Understanding the Output

### Quality Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.95 - 1.00 | Excellent | Deploy immediately |
| 0.90 - 0.95 | Very Good | Deploy with monitoring |
| 0.80 - 0.90 | Good | Consider another iteration |
| 0.70 - 0.80 | Fair | Requires fixes |
| < 0.70 | Poor | Significant issues |

### Consensus Score Breakdown

```
Consensus: 0.94
  ├─ Physical Realizability: 0.95
  ├─ Safety: 0.98
  ├─ Verifiability: 0.92
  ├─ Operational Clarity: 0.91
  └─ Scalability: 0.94
```

### Vulnerability Categories

1. **unrealistic_tolerance** - Tolerances too tight for real equipment
2. **missing_contingency** - Unhandled failure modes
3. **contradictory_requirement** - Conflicting specifications
4. **safety_vulnerability** - Dangerous edge cases
5. **scaling_limitation** - Issues blocking scale-up
6. **measurement_uncertainty** - Unverified measurement claims
7. **timing_conflict** - Impossible temporal constraints

---

## Expected Evolution Trajectory for SOP.txt

### Iteration 1: v16.1 → v16.1.1
**Vulnerabilities Found**: 4
1. Temperature recovery time (15 min → 45 min)
2. Magnetic field verification tolerance
3. UV interlock ambiguity
4. Gradient non-uniformity at scale

**Fixes Applied**: 4
**Quality Score**: 0.69 → 0.94 (+0.26)
**Status**: ✓ PASSED threshold (0.90)

**Expected Outcome**: Single iteration sufficient to reach 0.90 threshold

---

## Customizing Evolution

### Adjusting Ensemble Size

```python
# Small (fast, less diverse)
num_models=3  # ~3 minutes per iteration

# Medium (balanced)
num_models=7  # ~7 minutes per iteration (default)

# Large (thorough, more diverse)
num_models=11  # ~15 minutes per iteration
```

### Adjusting Attack Types

```python
# Comprehensive (all categories)
attack_types=[
    "unrealistic_tolerance",
    "missing_contingency",
    "contradictory_requirement",
    "safety_vulnerability",
    "scaling_limitation",
    "measurement_uncertainty",
    "timing_conflict",
    "material_degradation",
    "human_error_prone",
    "calibration_drift"
]

# Focused (safety-critical only)
attack_types=[
    "safety_vulnerability",
    "contradictory_requirement",
    "missing_contingency"
]
```

### Adjusting Fix Strategy

```python
# Minimal changes (conservative)
strategy=BlueTeamStrategy.MINIMAL

# Safety-focused (defense in depth)
strategy=BlueTeamStrategy.DEFENSIVE

# Comprehensive improvements (default)
strategy=BlueTeamStrategy.COMPREHENSIVE
```

---

## Troubleshooting

### Issue: "OpenEvolve not available"
**Solution**: Ensure OpenEvolve is installed
```bash
cd openevolve
pip install -e .
```

### Issue: Low quality score (< 0.70)
**Solution**: Increase iterations or ensemble size
```bash
python evolve_sop.py --input SOP.txt --output out.txt --iterations 5 --ensemble-size 11
```

### Issue: No vulnerabilities found
**Possible causes**:
1. SOP is already excellent (verify by checking quality score)
2. Attack types not comprehensive enough
3. Ensemble size too small (increase to 11+)

### Issue: Fix introduces new vulnerabilities
**Solution**: Run additional iteration (ensemble will catch it)
```bash
python evolve_sop.py --input SOP_v16.2.txt --output SOP_v16.3.txt --iterations 2
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Quality score ≥ 0.90
- [ ] All safety vulnerabilities resolved
- [ ] Backward compatibility verified
- [ ] Cost impact assessed
- [ ] Stakeholder approval obtained
- [ ] Rollback plan documented

### Deployment Steps

1. **Shadow Mode** (7 days)
   ```bash
   # Run both old and new SOP in parallel
   python compare_sop_versions.py --old v16.1 --new v16.2 --days 7
   ```

2. **Staging** (30 days)
   ```bash
   python deploy_to_staging.py --sop v16.2 --monitor
   ```

3. **Production**
   ```bash
   python deploy_to_production.py --sop v16.2 --monitor-days 30
   ```

---

## Best Practices

1. **Start with small ensemble** (3-5 models) for rapid prototyping
2. **Increase ensemble size** (7-11 models) for production-ready evolution
3. **Use comprehensive attack types** for initial evolution
4. **Focus on safety-critical** attacks for subsequent iterations
5. **Always save intermediate versions** for rollback capability
6. **Review metadata.json** to understand evolution history
7. **Validate physically** by running actual experiments if possible

---

## Next Steps

1. **Run first evolution**:
   ```bash
   python evolve_sop.py --input SOP.txt --output SOP_v16.2.txt
   ```

2. **Review results**:
   ```bash
   cat SOP_v16.2.txt.metadata.json | python -m json.tool
   ```

3. **Compare versions**:
   ```bash
   diff -u SOP.txt SOP_v16.2.txt | head -100
   ```

4. **Deploy if quality ≥ 0.90**

5. **Continuous improvement**: Run evolution monthly with production feedback

---

## Questions?

Refer to detailed framework documentation: `SOP_EVOLUTION_FRAMEWORK.md`

---

**End of Quick Start Guide**
>>>>>>> 1cb9c5e35 (update)
