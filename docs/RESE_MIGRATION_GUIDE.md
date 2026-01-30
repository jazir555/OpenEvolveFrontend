<<<<<<< HEAD
# RESE Migration Guide

## Table of Contents

1. [Overview](#overview)
2. [Migration Strategy](#migration-strategy)
3. [Pre-Migration Checklist](#pre-migration-checklist)
4. [Migration Steps](#migration-steps)
5. [Code Changes Required](#code-changes-required)
6. [Testing](#testing)
7. [Rollback Plan](#rollback-plan)
8. [Post-Migration](#post-migration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What is This Migration?

This guide helps you migrate from the current **End-to-End (E2E) Invention System** to the **RESE-enhanced E2E System**.

### Why Migrate?

**Benefits of RESE-Enhanced E2E:**

1. **Quantified Confidence**: Every stage produces statistical confidence metrics
2. **ACI Tracking**: Algorithmic Complexity Index tracked across entire pipeline
3. **Hidden Assumption Discovery**: Φ₁.₅ finds assumptions before they cause failures
4. **Reliable Solution Transfer**: I_mech validates analogies with 80%+ accuracy
5. **Guaranteed Quality**: Δ₃ ensures ≥20% ACI reduction before execution

### Migration Scope

**Affected Components:**
- Stage 1: Prompt Analysis (add SCE, Φ₁.₅, Φ₂)
- Stage 2: Knowledge Graph (add Ψ₂, I_mech)
- Stage 3: Solution Generation (add Γ₂, N_max)
- Stage 4: Formalization (add Δ₃)
- Stage 5: Red Team (add ACI quantification)
- Stage 6: Knowledge Extraction (add Φ₁.₅ feedback)
- Stage 7: SOP Generation (add Δ₁)
- Stage 8: Lab Execution (add Γ₁ monitoring)
- Stage 9: Monitoring (add ACI tracking)

**Backwards Compatibility:** ✅ Yes - RESE can be gradually integrated

---

## Migration Strategy

### Option 1: Big Bang Migration (Not Recommended)

Migrate all stages at once to RESE-enhanced versions.

**Pros:**
- Immediate benefits
- Single migration effort

**Cons:**
- High risk
- Difficult to debug issues
- Long downtime

**Not recommended unless you have a fully isolated staging environment.**

---

### Option 2: Gradual Migration (Recommended)

Migrate one stage at a time, with thorough testing at each step.

**Pros:**
- Low risk
- Easy to roll back
- Learn and adjust at each step
- Minimal downtime

**Cons:**
- Longer migration timeline
- Need to maintain two systems temporarily

**Recommended timeline:**
- Week 1-2: Stage 1 (Prompt Analysis)
- Week 3-4: Stage 2 (Knowledge Graph)
- Week 5-6: Stage 3 (Solution Generation)
- Week 7-8: Stage 4 (Formalization)
- Week 9-10: Stages 5-9 (remaining stages)

---

### Option 3: A/B Testing (Advanced)

Run both old and RESE-enhanced systems in parallel, compare results.

**Pros:**
- Direct performance comparison
- Risk-free (old system still handles traffic)
- Data-driven decision making

**Cons:**
- Requires 2x infrastructure
- More complex setup
- Need result comparison framework

---

## Pre-Migration Checklist

### 1. Environment Preparation

- [ ] Create staging environment
- [ ] Backup current E2E system
- [ ] Document current performance metrics (baseline)
- [ ] Set up monitoring and logging
- [ ] Prepare rollback procedure

### 2. Dependencies

- [ ] Python 3.9+ installed
- [ ] RESE dependencies installed:
  ```bash
  pip install numpy fastapi uvicorn pydantic networkx scipy
  ```
- [ ] Optional but recommended:
  ```bash
  pip install psutil numba
  ```

### 3. Configuration

- [ ] Create RESE configuration file
- [ ] Set appropriate ACI thresholds for each stage
- [ ] Configure caching
- [ ] Set up monitoring

### 4. Team Preparation

- [ ] Team trained on RESE concepts
- [ ] Documentation reviewed
- [ ] Support plan in place
- [ ] Communication plan prepared

---

## Migration Steps

### Phase 1: Stage 1 Migration (Prompt Analysis + RESE)

**Goal:** Add SCE (Φ₁), Φ₁.₅, and Φ₂ to Stage 1

**Steps:**

1. **Install RESE Stage 1 components:**
   ```python
   from rese.integrations.stage1 import Stage1RESEAnalyzer
   ```

2. **Update Stage 1 code:**

   **Before:**
   ```python
   # Old E2E Stage 1
   def analyze_prompt(prompt_text):
       # Simple constraint extraction
       constraints = extract_constraints(prompt_text)
       return constraints
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 1
   from rese.integrations.stage1 import Stage1RESEAnalyzer

   def analyze_prompt(prompt_text):
       analyzer = Stage1RESEAnalyzer()
       result = analyzer.analyze_prompt(
           prompt_text=prompt_text,
           domain="invention"
       )

       # Access RESE-enhanced results
       constraints = result.constraints
       assumptions = result.assumptions
       bias_report = result.bias_report

       return {
           'constraints': constraints,
           'assumptions': assumptions,
           'bias_report': bias_report,
           'refined_prompt': result.refined_prompt
       }
   ```

3. **Test Stage 1:**
   ```python
   # Test with sample prompt
   result = analyze_prompt("Design a room-temperature superconductor")

   assert 'constraints' in result
   assert 'assumptions' in result
   assert len(result['assumptions']) > 0  # Φ₁.₅ should find assumptions
   assert result['bias_report']['overall_bias_score'] < 0.5  # Check for bias
   ```

4. **Validate Results:**
   - Compare constraint quality (should be better)
   - Review assumptions (should reveal hidden requirements)
   - Check bias score (should flag issues)

5. **Deploy to Staging:**
   ```bash
   # Deploy to staging environment
   ./deploy.sh staging
   ```

6. **Monitor in Staging:**
   - Check ACI reduction (target: 15% in Stage 1)
   - Review assumption quality
   - Measure performance impact

7. **Deploy to Production** (if staging passes):
   ```bash
   # Deploy to production
   ./deploy.sh production
   ```

---

### Phase 2: Stage 2 Migration (Knowledge Graph + RESE)

**Goal:** Add Ψ₂ and I_mech to Stage 2

**Steps:**

1. **Update Stage 2 code:**

   **Before:**
   ```python
   # Old E2E Stage 2
   def map_to_knowledge_graph(prompt_constraints):
       # Simple keyword matching
       matches = kg.search(prompt_constraints)
       return matches
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 2
   from rese.integrations.stage2 import Stage2RESEMapper

   def map_to_knowledge_graph(prompt_constraints):
       mapper = Stage2RESEMapper()

       # Semantic matching
       mappings = mapper.map_to_domains(
           problem_description=prompt_constraints['refined_prompt'],
           knowledge_graph=kg,
           similarity_threshold=0.7
       )

       # Validate isomorphisms
       for mapping in mappings:
           isomorphism = mapper.validate_isomorphism(
               source=mapping['source_domain'],
               target=mapping['target_domain']
           )
           mapping['isomorphism'] = isomorphism

       return mappings
   ```

2. **Test Stage 2:**
   ```python
   # Test isomorphism detection
   mappings = map_to_knowledge_graph(stage1_result)

   assert len(mappings) > 0
   assert any(m['similarity'] > 0.7 for m in mappings)

   # Check validated isomorphisms
   validated = [m for m in mappings if m['isomorphism'].score > 0.8]
   assert len(validated) >= 1  # At least one high-confidence match
   ```

3. **Deploy and Monitor:**
   - Check isomorphism accuracy (target: >80%)
   - Review transfer success rate
   - Monitor performance

---

### Phase 3: Stage 3 Migration (Solution Generation + RESE)

**Goal:** Add Γ₂ (MCTS) and N_max to Stage 3

**Steps:**

1. **Update Stage 3 code:**

   **Before:**
   ```python
   # Old E2E Stage 3
   def generate_solution(problem_definition):
       # Heuristic search
       solution = heuristic_search(problem_definition)
       return solution
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 3
   from rese.integrations.stage3 import Stage3RESEGenerator

   def generate_solution(problem_definition):
       generator = Stage3RESEGenerator()

       solution = generator.generate(
           problem=problem_definition,
           constraints=problem_definition['constraints'],
           mcts_iterations=5000,
           aci_guided=True
       )

       return {
           'variables': solution.variables,
           'aci': solution.aci,
           'confidence': solution.confidence,
           'converged': solution.converged
       }
   ```

2. **Test Stage 3:**
   ```python
   # Test MCTS optimization
   solution = generate_solution(stage2_result)

   assert solution['converged'] == True
   assert solution['aci'] < 0.35  # Target ACI after Stage 3
   assert solution['confidence'] > 0.7
   ```

3. **Deploy and Monitor:**
   - Check convergence rate (target: >95%)
   - Monitor ACI reduction (target: 49% in Stage 3)
   - Measure solution quality improvement

---

### Phase 4: Stage 4 Migration (Formalization + RESE)

**Goal:** Add Δ₃ validation to Stage 4

**Steps:**

1. **Update Stage 4 code:**

   **Before:**
   ```python
   # Old E2E Stage 4
   def formalize_solution(solution):
       # Generate Lean 4 proofs
       proofs = generate_proofs(solution)
       return proofs
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 4
   from rese.integrations.stage4 import Stage4RESEFormalizer

   def formalize_solution(solution):
       formalizer = Stage4RESEFormalizer()

       formalization = formalizer.formalize(
           solution=solution,
           constraints=solution['constraints'],
           require_proof=True
       )

       # Validate ACI reduction
       validation = formalizer.validate_aci_reduction(
           baseline_aci=solution['initial_aci'],
           final_aci=formalization['aci'],
           min_reduction=0.2
       )

       return {
           'formalization': formalization,
           'validation': validation
       }
   ```

2. **Test Stage 4:**
   ```python
   # Test validation
   result = formalize_solution(stage3_result)

   assert result['validation'].is_valid == True
   assert result['validation'].aci_reduction >= 0.2
   assert result['validation'].validation_score >= 0.7
   ```

---

### Phase 5: Migrate Remaining Stages (5-9)

Repeat similar process for Stages 5-9:

- **Stage 5:** Add ACI quantification to red team analysis
- **Stage 6:** Add Φ₁.₅ feedback to knowledge extraction
- **Stage 7:** Add Δ₁ architecture assembly to SOP generation
- **Stage 8:** Add Γ₁ real-time ACI monitoring to lab execution
- **Stage 9:** Add ACI trend analysis to monitoring

---

## Code Changes Required

### 1. Import Changes

**Add to all stage files:**
```python
# Add RESE logging
from rese.monitoring import MonitoringSystem

# Add RESE config
from rese.config import get_config
config = get_config()

# Add ACI tracking
from rese.gamma1.core.aci_calculator import ACICalculator
aci_calculator = ACICalculator()
```

### 2. Data Structure Changes

**Stage outputs must include:**
```python
# All stages must track ACI
output = {
    # ... existing fields ...
    'aci': current_aci,  # Add this
    'confidence': confidence,  # Add this
    'metadata': {
        'rese_version': '1.0.0',
        'aci_history': aci_history,
        'validation_score': validation_score
    }
}
```

### 3. Error Handling Changes

**Add RESE exception handling:**
```python
from rese.rese_pipeline import PhaseExecutionError, ValidationError

try:
    result = run_rese_stage()
except PhaseExecutionError as e:
    # Log RESE-specific error
    logger.error(f"RESE phase failed: {e}")
    # Handle partial results
    if e.partial_result:
        return e.partial_result
except ValidationError as e:
    # Fix validation issues
    logger.warning(f"Validation failed: {e}")
    return retry_with_fixed_input(e.validation_report)
```

### 4. Monitoring Changes

**Add RESE metrics to monitoring:**
```python
# Track ACI in monitoring system
monitoring.record_aci(
    stage=current_stage,
    aci_value=current_aci,
    timestamp=datetime.now()
)

# Track validation scores
monitoring.record_validation(
    stage=current_stage,
    score=validation_score,
    confidence=confidence
)
```

---

## Testing

### Unit Tests

```python
# Test each RESE component
def test_stage1_rese():
    analyzer = Stage1RESEAnalyzer()
    result = analyzer.analyze_prompt("test prompt")

    assert result.constraints is not None
    assert len(result.assumptions) > 0
    assert result.bias_report is not None
```

### Integration Tests

```python
# Test full RESE pipeline
def test_full_rese_pipeline():
    result = run_rese_pipeline_with_rese()

    assert result.status == 'completed'
    assert result.aci_history[-1] < 0.2  # Target ACI
    assert result.validation_score >= 0.7
```

### Performance Tests

```python
# Compare old vs RESE performance
def test_performance_comparison():
    # Old system
    old_time = time.time()
    old_result = run_old_e2e()
    old_elapsed = time.time() - old_time

    # RESE system
    new_time = time.time()
    new_result = run_rese_e2e()
    new_elapsed = time.time() - new_time

    # RESE can be slower but better
    print(f"Old: {old_elapsed:.2f}s, Quality: {old_result.quality}")
    print(f"New: {new_elapsed:.2f}s, Quality: {new_result.quality}")

    # Quality should improve significantly
    assert new_result.quality > old_result.quality * 1.5
```

### A/B Tests

```python
# Run both systems in parallel
old_results = []
new_results = []

for i in range(100):
    problem = generate_test_problem()

    old_result = run_old_e2e(problem)
    new_result = run_rese_e2e(problem)

    old_results.append(old_result)
    new_results.append(new_result)

# Compare
old_quality = mean(r.quality for r in old_results)
new_quality = mean(r.quality for r in new_results)

print(f"Old quality: {old_quality:.2f}")
print(f"New quality: {new_quality:.2f}")
print(f"Improvement: {(new_quality/old_quality - 1)*100:.1f}%")
```

---

## Rollback Plan

### When to Rollback

- Critical bugs in production
- Performance degradation >50%
- Data corruption
- Validation score < 0.5

### Rollback Procedure

**Step 1: Immediate Rollback (if critical)**
```bash
# Switch to old system
./rollback_to_old.sh

# Verify rollback
curl http://localhost:8000/health
# Should show old version
```

**Step 2: Analyze Issue**
```bash
# Check logs
tail -f logs/rese.log

# Identify root cause
# Document issue
```

**Step 3: Fix and Re-test**
```bash
# Fix issue in staging
# Test thoroughly
# Re-deploy
```

### Rollback Verification

```python
# Verify old system is working
def verify_rollback():
    result = run_old_e2e(test_problem)

    assert result is not None
    assert result.status == 'completed'
    assert result.quality >= baseline_quality

    print("✓ Rollback successful")
```

---

## Post-Migration

### 1. Monitor Performance

**Key metrics to track:**
```python
# ACI reduction
aci_reduction = (initial_aci - final_aci) / initial_aci
assert aci_reduction >= 0.2

# Validation score
assert validation_score >= 0.7

# Solution quality
assert solution_quality >= baseline * 1.5

# Performance (should be acceptable)
assert execution_time < max_acceptable_time
```

### 2. Collect Feedback

**From users:**
- Solution quality improvement?
- Performance impact?
- Any issues?

**From system:**
- Error rates
- ACI trends
- Validation scores

### 3. Optimize

**Based on monitoring data:**
```python
# Tune thresholds
if validation_score > 0.9:
    # Can reduce iterations for speed
    config.phase3.gamma2_iterations = 3000  # Was 5000

if aci_reduction < 0.25:
    # Need more thorough search
    config.phase3.gamma2_iterations = 10000  # Was 5000
```

### 4. Document

**Update documentation:**
- Record migration decisions
- Document any customizations
- Update runbooks
- Train team on RESE-enhanced system

---

## Troubleshooting

### Issue 1: Stage 1 Finds Too Many Assumptions

**Symptom:** Φ₁.₅ returns 100+ assumptions

**Solution:**
```python
# Increase threshold
config.phase1.phi15_assumption_threshold = 0.7  # Was 0.6
```

---

### Issue 2: Stage 2 No Isomorphisms Found

**Symptom:** Zero domains above threshold

**Solution:**
```python
# Lower threshold slightly
config.phase2.psi2_similarity_threshold = 0.65  # Was 0.7

# Or check if domain is too novel
if max_similarity < 0.6:
    print("Domain too novel - skip isomorphism transfer")
```

---

### Issue 3: Stage 3 MCTS Does Not Converge

**Symptom:** MCTS runs to max_iterations

**Solution:**
```python
# Increase iterations
config.phase3.gamma2_iterations = 10000

# Or relax convergence criteria
config.phase3.convergence_patience = 100  # Was 50
```

---

### Issue 4: Stage 4 Validation Fails

**Symptom:** ACI reduction < 20%

**Solution:**
```python
# Check if problem is over-constrained
if reduction < 0.2:
    print("Warning: ACI reduction below target")
    print("Consider:")
    print("1. Re-running Stage 3 with more iterations")
    print("2. Relaxing some constraints")
    print("3. Checking problem formulation")
```

---

### Issue 5: Performance Degradation

**Symptom:** RESE system much slower than old

**Solution:**
```python
# Enable caching
config.pipeline.enable_caching = True

# Reduce iterations
config.phase3.gamma2_iterations = 3000  # Was 5000

# Use parallel processing
config.phase3.gamma2_parallel_agents = 8

# Profile and optimize bottlenecks
python -m cProfile -o profile.stats my_script.py
```

---

## Migration Timeline Example

**Week 1-2: Stage 1**
- Day 1-2: Development and testing
- Day 3-4: Staging deployment
- Day 5: Production deployment
- Day 6-7: Monitoring and adjustment

**Week 3-4: Stage 2**
- Similar pattern

**Week 5-6: Stage 3**
- Similar pattern

**Week 7-8: Stage 4**
- Similar pattern

**Week 9-10: Stages 5-9**
- Can be done faster as pattern is established

**Week 11-12: Optimization and Documentation**

**Total: 12 weeks for full migration**

---

## Success Criteria

Migration is successful when:

1. **All stages deployed** with RESE enhancements
2. **ACI reduction ≥20%** across all problems
3. **Validation score ≥70%** for all solutions
4. **Performance impact <50%** slowdown acceptable
5. **User satisfaction** improved (measured by surveys)
6. **Zero critical bugs** in production for 30 days

---

**Migration Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team

**Need Help?**
- Review troubleshooting section
- Check RESE_USER_GUIDE.md for concepts
- See RESE_INTEGRATION_GUIDE.md for detailed integration steps
- Open GitHub issue for specific problems
=======
# RESE Migration Guide

## Table of Contents

1. [Overview](#overview)
2. [Migration Strategy](#migration-strategy)
3. [Pre-Migration Checklist](#pre-migration-checklist)
4. [Migration Steps](#migration-steps)
5. [Code Changes Required](#code-changes-required)
6. [Testing](#testing)
7. [Rollback Plan](#rollback-plan)
8. [Post-Migration](#post-migration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What is This Migration?

This guide helps you migrate from the current **End-to-End (E2E) Invention System** to the **RESE-enhanced E2E System**.

### Why Migrate?

**Benefits of RESE-Enhanced E2E:**

1. **Quantified Confidence**: Every stage produces statistical confidence metrics
2. **ACI Tracking**: Algorithmic Complexity Index tracked across entire pipeline
3. **Hidden Assumption Discovery**: Φ₁.₅ finds assumptions before they cause failures
4. **Reliable Solution Transfer**: I_mech validates analogies with 80%+ accuracy
5. **Guaranteed Quality**: Δ₃ ensures ≥20% ACI reduction before execution

### Migration Scope

**Affected Components:**
- Stage 1: Prompt Analysis (add SCE, Φ₁.₅, Φ₂)
- Stage 2: Knowledge Graph (add Ψ₂, I_mech)
- Stage 3: Solution Generation (add Γ₂, N_max)
- Stage 4: Formalization (add Δ₃)
- Stage 5: Red Team (add ACI quantification)
- Stage 6: Knowledge Extraction (add Φ₁.₅ feedback)
- Stage 7: SOP Generation (add Δ₁)
- Stage 8: Lab Execution (add Γ₁ monitoring)
- Stage 9: Monitoring (add ACI tracking)

**Backwards Compatibility:** ✅ Yes - RESE can be gradually integrated

---

## Migration Strategy

### Option 1: Big Bang Migration (Not Recommended)

Migrate all stages at once to RESE-enhanced versions.

**Pros:**
- Immediate benefits
- Single migration effort

**Cons:**
- High risk
- Difficult to debug issues
- Long downtime

**Not recommended unless you have a fully isolated staging environment.**

---

### Option 2: Gradual Migration (Recommended)

Migrate one stage at a time, with thorough testing at each step.

**Pros:**
- Low risk
- Easy to roll back
- Learn and adjust at each step
- Minimal downtime

**Cons:**
- Longer migration timeline
- Need to maintain two systems temporarily

**Recommended timeline:**
- Week 1-2: Stage 1 (Prompt Analysis)
- Week 3-4: Stage 2 (Knowledge Graph)
- Week 5-6: Stage 3 (Solution Generation)
- Week 7-8: Stage 4 (Formalization)
- Week 9-10: Stages 5-9 (remaining stages)

---

### Option 3: A/B Testing (Advanced)

Run both old and RESE-enhanced systems in parallel, compare results.

**Pros:**
- Direct performance comparison
- Risk-free (old system still handles traffic)
- Data-driven decision making

**Cons:**
- Requires 2x infrastructure
- More complex setup
- Need result comparison framework

---

## Pre-Migration Checklist

### 1. Environment Preparation

- [ ] Create staging environment
- [ ] Backup current E2E system
- [ ] Document current performance metrics (baseline)
- [ ] Set up monitoring and logging
- [ ] Prepare rollback procedure

### 2. Dependencies

- [ ] Python 3.9+ installed
- [ ] RESE dependencies installed:
  ```bash
  pip install numpy fastapi uvicorn pydantic networkx scipy
  ```
- [ ] Optional but recommended:
  ```bash
  pip install psutil numba
  ```

### 3. Configuration

- [ ] Create RESE configuration file
- [ ] Set appropriate ACI thresholds for each stage
- [ ] Configure caching
- [ ] Set up monitoring

### 4. Team Preparation

- [ ] Team trained on RESE concepts
- [ ] Documentation reviewed
- [ ] Support plan in place
- [ ] Communication plan prepared

---

## Migration Steps

### Phase 1: Stage 1 Migration (Prompt Analysis + RESE)

**Goal:** Add SCE (Φ₁), Φ₁.₅, and Φ₂ to Stage 1

**Steps:**

1. **Install RESE Stage 1 components:**
   ```python
   from rese.integrations.stage1 import Stage1RESEAnalyzer
   ```

2. **Update Stage 1 code:**

   **Before:**
   ```python
   # Old E2E Stage 1
   def analyze_prompt(prompt_text):
       # Simple constraint extraction
       constraints = extract_constraints(prompt_text)
       return constraints
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 1
   from rese.integrations.stage1 import Stage1RESEAnalyzer

   def analyze_prompt(prompt_text):
       analyzer = Stage1RESEAnalyzer()
       result = analyzer.analyze_prompt(
           prompt_text=prompt_text,
           domain="invention"
       )

       # Access RESE-enhanced results
       constraints = result.constraints
       assumptions = result.assumptions
       bias_report = result.bias_report

       return {
           'constraints': constraints,
           'assumptions': assumptions,
           'bias_report': bias_report,
           'refined_prompt': result.refined_prompt
       }
   ```

3. **Test Stage 1:**
   ```python
   # Test with sample prompt
   result = analyze_prompt("Design a room-temperature superconductor")

   assert 'constraints' in result
   assert 'assumptions' in result
   assert len(result['assumptions']) > 0  # Φ₁.₅ should find assumptions
   assert result['bias_report']['overall_bias_score'] < 0.5  # Check for bias
   ```

4. **Validate Results:**
   - Compare constraint quality (should be better)
   - Review assumptions (should reveal hidden requirements)
   - Check bias score (should flag issues)

5. **Deploy to Staging:**
   ```bash
   # Deploy to staging environment
   ./deploy.sh staging
   ```

6. **Monitor in Staging:**
   - Check ACI reduction (target: 15% in Stage 1)
   - Review assumption quality
   - Measure performance impact

7. **Deploy to Production** (if staging passes):
   ```bash
   # Deploy to production
   ./deploy.sh production
   ```

---

### Phase 2: Stage 2 Migration (Knowledge Graph + RESE)

**Goal:** Add Ψ₂ and I_mech to Stage 2

**Steps:**

1. **Update Stage 2 code:**

   **Before:**
   ```python
   # Old E2E Stage 2
   def map_to_knowledge_graph(prompt_constraints):
       # Simple keyword matching
       matches = kg.search(prompt_constraints)
       return matches
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 2
   from rese.integrations.stage2 import Stage2RESEMapper

   def map_to_knowledge_graph(prompt_constraints):
       mapper = Stage2RESEMapper()

       # Semantic matching
       mappings = mapper.map_to_domains(
           problem_description=prompt_constraints['refined_prompt'],
           knowledge_graph=kg,
           similarity_threshold=0.7
       )

       # Validate isomorphisms
       for mapping in mappings:
           isomorphism = mapper.validate_isomorphism(
               source=mapping['source_domain'],
               target=mapping['target_domain']
           )
           mapping['isomorphism'] = isomorphism

       return mappings
   ```

2. **Test Stage 2:**
   ```python
   # Test isomorphism detection
   mappings = map_to_knowledge_graph(stage1_result)

   assert len(mappings) > 0
   assert any(m['similarity'] > 0.7 for m in mappings)

   # Check validated isomorphisms
   validated = [m for m in mappings if m['isomorphism'].score > 0.8]
   assert len(validated) >= 1  # At least one high-confidence match
   ```

3. **Deploy and Monitor:**
   - Check isomorphism accuracy (target: >80%)
   - Review transfer success rate
   - Monitor performance

---

### Phase 3: Stage 3 Migration (Solution Generation + RESE)

**Goal:** Add Γ₂ (MCTS) and N_max to Stage 3

**Steps:**

1. **Update Stage 3 code:**

   **Before:**
   ```python
   # Old E2E Stage 3
   def generate_solution(problem_definition):
       # Heuristic search
       solution = heuristic_search(problem_definition)
       return solution
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 3
   from rese.integrations.stage3 import Stage3RESEGenerator

   def generate_solution(problem_definition):
       generator = Stage3RESEGenerator()

       solution = generator.generate(
           problem=problem_definition,
           constraints=problem_definition['constraints'],
           mcts_iterations=5000,
           aci_guided=True
       )

       return {
           'variables': solution.variables,
           'aci': solution.aci,
           'confidence': solution.confidence,
           'converged': solution.converged
       }
   ```

2. **Test Stage 3:**
   ```python
   # Test MCTS optimization
   solution = generate_solution(stage2_result)

   assert solution['converged'] == True
   assert solution['aci'] < 0.35  # Target ACI after Stage 3
   assert solution['confidence'] > 0.7
   ```

3. **Deploy and Monitor:**
   - Check convergence rate (target: >95%)
   - Monitor ACI reduction (target: 49% in Stage 3)
   - Measure solution quality improvement

---

### Phase 4: Stage 4 Migration (Formalization + RESE)

**Goal:** Add Δ₃ validation to Stage 4

**Steps:**

1. **Update Stage 4 code:**

   **Before:**
   ```python
   # Old E2E Stage 4
   def formalize_solution(solution):
       # Generate Lean 4 proofs
       proofs = generate_proofs(solution)
       return proofs
   ```

   **After:**
   ```python
   # RESE-enhanced Stage 4
   from rese.integrations.stage4 import Stage4RESEFormalizer

   def formalize_solution(solution):
       formalizer = Stage4RESEFormalizer()

       formalization = formalizer.formalize(
           solution=solution,
           constraints=solution['constraints'],
           require_proof=True
       )

       # Validate ACI reduction
       validation = formalizer.validate_aci_reduction(
           baseline_aci=solution['initial_aci'],
           final_aci=formalization['aci'],
           min_reduction=0.2
       )

       return {
           'formalization': formalization,
           'validation': validation
       }
   ```

2. **Test Stage 4:**
   ```python
   # Test validation
   result = formalize_solution(stage3_result)

   assert result['validation'].is_valid == True
   assert result['validation'].aci_reduction >= 0.2
   assert result['validation'].validation_score >= 0.7
   ```

---

### Phase 5: Migrate Remaining Stages (5-9)

Repeat similar process for Stages 5-9:

- **Stage 5:** Add ACI quantification to red team analysis
- **Stage 6:** Add Φ₁.₅ feedback to knowledge extraction
- **Stage 7:** Add Δ₁ architecture assembly to SOP generation
- **Stage 8:** Add Γ₁ real-time ACI monitoring to lab execution
- **Stage 9:** Add ACI trend analysis to monitoring

---

## Code Changes Required

### 1. Import Changes

**Add to all stage files:**
```python
# Add RESE logging
from rese.monitoring import MonitoringSystem

# Add RESE config
from rese.config import get_config
config = get_config()

# Add ACI tracking
from rese.gamma1.core.aci_calculator import ACICalculator
aci_calculator = ACICalculator()
```

### 2. Data Structure Changes

**Stage outputs must include:**
```python
# All stages must track ACI
output = {
    # ... existing fields ...
    'aci': current_aci,  # Add this
    'confidence': confidence,  # Add this
    'metadata': {
        'rese_version': '1.0.0',
        'aci_history': aci_history,
        'validation_score': validation_score
    }
}
```

### 3. Error Handling Changes

**Add RESE exception handling:**
```python
from rese.rese_pipeline import PhaseExecutionError, ValidationError

try:
    result = run_rese_stage()
except PhaseExecutionError as e:
    # Log RESE-specific error
    logger.error(f"RESE phase failed: {e}")
    # Handle partial results
    if e.partial_result:
        return e.partial_result
except ValidationError as e:
    # Fix validation issues
    logger.warning(f"Validation failed: {e}")
    return retry_with_fixed_input(e.validation_report)
```

### 4. Monitoring Changes

**Add RESE metrics to monitoring:**
```python
# Track ACI in monitoring system
monitoring.record_aci(
    stage=current_stage,
    aci_value=current_aci,
    timestamp=datetime.now()
)

# Track validation scores
monitoring.record_validation(
    stage=current_stage,
    score=validation_score,
    confidence=confidence
)
```

---

## Testing

### Unit Tests

```python
# Test each RESE component
def test_stage1_rese():
    analyzer = Stage1RESEAnalyzer()
    result = analyzer.analyze_prompt("test prompt")

    assert result.constraints is not None
    assert len(result.assumptions) > 0
    assert result.bias_report is not None
```

### Integration Tests

```python
# Test full RESE pipeline
def test_full_rese_pipeline():
    result = run_rese_pipeline_with_rese()

    assert result.status == 'completed'
    assert result.aci_history[-1] < 0.2  # Target ACI
    assert result.validation_score >= 0.7
```

### Performance Tests

```python
# Compare old vs RESE performance
def test_performance_comparison():
    # Old system
    old_time = time.time()
    old_result = run_old_e2e()
    old_elapsed = time.time() - old_time

    # RESE system
    new_time = time.time()
    new_result = run_rese_e2e()
    new_elapsed = time.time() - new_time

    # RESE can be slower but better
    print(f"Old: {old_elapsed:.2f}s, Quality: {old_result.quality}")
    print(f"New: {new_elapsed:.2f}s, Quality: {new_result.quality}")

    # Quality should improve significantly
    assert new_result.quality > old_result.quality * 1.5
```

### A/B Tests

```python
# Run both systems in parallel
old_results = []
new_results = []

for i in range(100):
    problem = generate_test_problem()

    old_result = run_old_e2e(problem)
    new_result = run_rese_e2e(problem)

    old_results.append(old_result)
    new_results.append(new_result)

# Compare
old_quality = mean(r.quality for r in old_results)
new_quality = mean(r.quality for r in new_results)

print(f"Old quality: {old_quality:.2f}")
print(f"New quality: {new_quality:.2f}")
print(f"Improvement: {(new_quality/old_quality - 1)*100:.1f}%")
```

---

## Rollback Plan

### When to Rollback

- Critical bugs in production
- Performance degradation >50%
- Data corruption
- Validation score < 0.5

### Rollback Procedure

**Step 1: Immediate Rollback (if critical)**
```bash
# Switch to old system
./rollback_to_old.sh

# Verify rollback
curl http://localhost:8000/health
# Should show old version
```

**Step 2: Analyze Issue**
```bash
# Check logs
tail -f logs/rese.log

# Identify root cause
# Document issue
```

**Step 3: Fix and Re-test**
```bash
# Fix issue in staging
# Test thoroughly
# Re-deploy
```

### Rollback Verification

```python
# Verify old system is working
def verify_rollback():
    result = run_old_e2e(test_problem)

    assert result is not None
    assert result.status == 'completed'
    assert result.quality >= baseline_quality

    print("✓ Rollback successful")
```

---

## Post-Migration

### 1. Monitor Performance

**Key metrics to track:**
```python
# ACI reduction
aci_reduction = (initial_aci - final_aci) / initial_aci
assert aci_reduction >= 0.2

# Validation score
assert validation_score >= 0.7

# Solution quality
assert solution_quality >= baseline * 1.5

# Performance (should be acceptable)
assert execution_time < max_acceptable_time
```

### 2. Collect Feedback

**From users:**
- Solution quality improvement?
- Performance impact?
- Any issues?

**From system:**
- Error rates
- ACI trends
- Validation scores

### 3. Optimize

**Based on monitoring data:**
```python
# Tune thresholds
if validation_score > 0.9:
    # Can reduce iterations for speed
    config.phase3.gamma2_iterations = 3000  # Was 5000

if aci_reduction < 0.25:
    # Need more thorough search
    config.phase3.gamma2_iterations = 10000  # Was 5000
```

### 4. Document

**Update documentation:**
- Record migration decisions
- Document any customizations
- Update runbooks
- Train team on RESE-enhanced system

---

## Troubleshooting

### Issue 1: Stage 1 Finds Too Many Assumptions

**Symptom:** Φ₁.₅ returns 100+ assumptions

**Solution:**
```python
# Increase threshold
config.phase1.phi15_assumption_threshold = 0.7  # Was 0.6
```

---

### Issue 2: Stage 2 No Isomorphisms Found

**Symptom:** Zero domains above threshold

**Solution:**
```python
# Lower threshold slightly
config.phase2.psi2_similarity_threshold = 0.65  # Was 0.7

# Or check if domain is too novel
if max_similarity < 0.6:
    print("Domain too novel - skip isomorphism transfer")
```

---

### Issue 3: Stage 3 MCTS Does Not Converge

**Symptom:** MCTS runs to max_iterations

**Solution:**
```python
# Increase iterations
config.phase3.gamma2_iterations = 10000

# Or relax convergence criteria
config.phase3.convergence_patience = 100  # Was 50
```

---

### Issue 4: Stage 4 Validation Fails

**Symptom:** ACI reduction < 20%

**Solution:**
```python
# Check if problem is over-constrained
if reduction < 0.2:
    print("Warning: ACI reduction below target")
    print("Consider:")
    print("1. Re-running Stage 3 with more iterations")
    print("2. Relaxing some constraints")
    print("3. Checking problem formulation")
```

---

### Issue 5: Performance Degradation

**Symptom:** RESE system much slower than old

**Solution:**
```python
# Enable caching
config.pipeline.enable_caching = True

# Reduce iterations
config.phase3.gamma2_iterations = 3000  # Was 5000

# Use parallel processing
config.phase3.gamma2_parallel_agents = 8

# Profile and optimize bottlenecks
python -m cProfile -o profile.stats my_script.py
```

---

## Migration Timeline Example

**Week 1-2: Stage 1**
- Day 1-2: Development and testing
- Day 3-4: Staging deployment
- Day 5: Production deployment
- Day 6-7: Monitoring and adjustment

**Week 3-4: Stage 2**
- Similar pattern

**Week 5-6: Stage 3**
- Similar pattern

**Week 7-8: Stage 4**
- Similar pattern

**Week 9-10: Stages 5-9**
- Can be done faster as pattern is established

**Week 11-12: Optimization and Documentation**

**Total: 12 weeks for full migration**

---

## Success Criteria

Migration is successful when:

1. **All stages deployed** with RESE enhancements
2. **ACI reduction ≥20%** across all problems
3. **Validation score ≥70%** for all solutions
4. **Performance impact <50%** slowdown acceptable
5. **User satisfaction** improved (measured by surveys)
6. **Zero critical bugs** in production for 30 days

---

**Migration Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team

**Need Help?**
- Review troubleshooting section
- Check RESE_USER_GUIDE.md for concepts
- See RESE_INTEGRATION_GUIDE.md for detailed integration steps
- Open GitHub issue for specific problems
>>>>>>> 1cb9c5e35 (update)
