# RESE Z3 Integration Plan

**Date:** 2026-02-04
**Status:** Ready for Implementation
**Objective:** Integrate z3prover into all RESE components requiring formal verification and constraint solving

---

## Executive Summary

The RESE framework currently uses **inefficient heuristic-based methods** for constraint solving, contradiction detection, and formal verification. This plan leverages the **existing production-ready z3prover integration** (17,000+ lines of code) to replace these methods with formal SMT solving.

**Current State:**
- RESE uses naive O(n²) pairwise contradiction detection
- Text-based constraint manipulation (not formal logic)
- Statistical validation without constraint verification
- Heuristic-based pattern matching

**Target State:**
- Z3 SMT solver for O(n log n) contradiction detection
- Formal first-order logic constraint representation
- SMT-based hypothesis validation
- Provable constraint satisfaction

---

## Existing Z3 Infrastructure

### Available Components

**1. Core Z3 Integration** (Root Level)
- `z3prover_integration.py` (983 lines) - Core Python API wrapper
- `z3prover_advanced.py` (1,199 lines) - Optimization, arrays, bit-vectors
- `z3_config.yaml` (318 lines) - Complete configuration
- Total: **17,000+ lines** of production Python integration code

**2. Z3-LeanAide Bridge**
- `z3_leanaide_bridge.py` (1,005 lines) - Bidirectional translation
- Cross-verification with multiple strategies
- SMT-LIB ↔ Lean 4 translation

**3. BubbleLab Integration**
- `z3_constraint_solving_node.py` (690+ lines)
- `z3_theorem_proving_node.py` (690+ lines)
- Z3 service bubble (TypeScript, port 7655)
- API routes with full OpenAPI schema

**4. LeanAide Integration**
- Core project: `core-projects/LeanAide/`
- Adapter: `glue/adapters/leanaide-adapter/`
- Cross-validation: Z3 + Lean 4 theorem proving

---

## RESE Integration Points

### Priority 1: CRITICAL - SCE Contradiction Detection

**Location:** `glue/adapters/rese-sce/src/sce_bridge.py`

**Current Implementation (Lines 370-446):**
```python
# Naive O(n²) pairwise checking
for i in range(len(constraints)):
    for j in range(i + 1, len(constraints)):
        contradiction = self._check_pairwise_contradiction(c1, c2)
```

**Z3 Integration:**
```python
# Replace with Z3 SMT solving
from z3prover_integration import Z3ProverIntegration

class SymbolicConstraintEngine:
    def __init__(self):
        self.z3 = Z3ProverIntegration()

    def detect_contradictions(self, constraints, correlation_id):
        # 1. Encode constraints as Z3 formulas
        z3_constraints = [self._encode_to_z3(c) for c in constraints]

        # 2. Check satisfiability
        result = self.z3.solve_constraints(
            smtlib2=z3_constraints,
            timeout=5000,
            correlation_id=correlation_id
        )

        # 3. If UNSAT, extract unsat core (minimal contradiction set)
        if result.status == 'UNSAT':
            contradictions = self._extract_unsat_core(result.unsat_core)

        return contradictions
```

**Benefits:**
- O(n²) → O(n log n) complexity
- Formal proofs of contradiction
- Minimal contradiction sets (unsat_core)
- Handles transitive contradictions

---

### Priority 2: HIGH - Phase I Constraint Formalization

**Location:** `glue/adapters/rese-phase1/src/phase1_executor.py`

**Current Implementation (Lines 896-980):**
```python
# Text-based constraint extraction
def _invert_constraint(self, constraint: str) -> str:
    inversions = {
        'impossible': 'possible',
        'cannot': 'can',
        # Missing: quantifier negation, De Morgan's laws
    }
```

**Z3 Integration:**
```python
from z3prover_integration import Z3ProverIntegration

class ConstraintHardener:
    def __init__(self):
        self.z3 = Z3ProverIntegration()

    def harden_constraints(self, problem_description, correlation_id):
        # 1. Extract constraints (existing logic)
        constraints = self._extract_constraints(problem_description)

        # 2. Encode as Z3 formulas
        z3_formulas = []
        for constraint in constraints:
            # Parse to first-order logic
            formula = self._parse_to_fol(constraint)
            z3_formulas.append(formula)

        # 3. Invert using Z3.Not() with proper quantifier handling
        inverted = []
        for formula in z3_formulas:
            # Z3 handles: ¬(∃x. P(x)) → ∀x. ¬P(x)
            inv_formula = self.z3.invert_constraint(formula)
            inverted.append(inv_formula)

        # 4. Verify inverted constraints are satisfiable
        result = self.z3.solve_constraints(
            smtlib2=inverted,
            timeout=5000,
            correlation_id=correlation_id
        )

        return inverted, result
```

**Benefits:**
- Formal first-order logic representation
- Proper quantifier negation
- Constraint simplification via Z3
- Satisfiability verification

---

### Priority 3: HIGH - Phase III MCTS Constraint Satisfaction

**Location:** `glue/adapters/rese-phase3/src/phase3_executor.py`

**Current Implementation (Lines 929-1077):**
```python
# MCTS search without constraint checking
def execute_search(self, root_hypothesis, correlation_id):
    while not converged:
        # Selection, expansion, simulation
        # Missing: Constraint checking during expansion
```

**Z3 Integration:**
```python
from z3prover_integration import Z3ProverIntegration

class MCTSSearchExecutor:
    def __init__(self):
        self.z3 = Z3ProverIntegration()

    def execute_search(self, root_hypothesis, correlation_id):
        while not converged:
            # 1. Select node (existing UCB1 logic)
            node = self._select_node()

            # 2. Before expanding, check constraint satisfaction
            if not self._is_constraint_satisfiable(node, correlation_id):
                # Prune this branch
                continue

            # 3. Expand node
            children = self._expand_node(node)

            # 4. For each child, verify hypothesis satisfies constraints
            valid_children = []
            for child in children:
                if self._verify_hypothesis_satisfies_constraints(child, correlation_id):
                    valid_children.append(child)

            # 5. Continue with valid children only
            # ... rest of MCTS logic

    def _is_constraint_satisfiable(self, node, correlation_id):
        # Encode current path as Z3 constraints
        path_constraints = self._encode_path_to_z3(node)

        # Check satisfiability
        result = self.z3.solve_constraints(
            smtlib2=path_constraints,
            timeout=1000,  # Fast check for MCTS
            correlation_id=correlation_id
        )

        return result.status == 'SAT'

    def _verify_hypothesis_satisfies_constraints(self, hypothesis, correlation_id):
        # Encode: hypothesis ∧ all_constraints
        formula = self._encode_hypothesis_with_constraints(hypothesis)

        # Verify satisfiable
        result = self.z3.solve_constraints(
            smtlib2=formula,
            timeout=1000,
            correlation_id=correlation_id
        )

        return result.status == 'SAT'
```

**Benefits:**
- Prune invalid MCTS branches early (10-100x speedup)
- Constraint-guided hypothesis generation
- Only explore valid regions of search space
- Formal verification of hypotheses

---

### Priority 4: MEDIUM - Phase II Isomorphism Verification

**Location:** `glue/adapters/rese-phase2/src/phase2_executor.py`

**Current Implementation (Lines 297-392):**
```python
# Structural overlap only
def compute_imech_score(self, source_fdg, target_fdg):
    fdg_overlap = self.compute_fdg_overlap(source_fdg, target_fdg)
    # Missing: Formal verification of behavioral equivalence
```

**Z3 Integration:**
```python
from z3prover_integration import Z3ProverIntegration

class CrossDomainMapper:
    def __init__(self):
        self.z3 = Z3ProverIntegration()

    def verify_isomorphism(self, fdg1, fdg2, correlation_id):
        # 1. Compute structural overlap (existing logic)
        structural_score = self.compute_fdg_overlap(fdg1, fdg2)

        # 2. If structural score > threshold, verify with Z3
        if structural_score > 0.7:
            # Encode: "∀ inputs. behavior(fdg1) ≡ behavior(fdg2)"
            equivalence_formula = self._encode_behavioral_equivalence(fdg1, fdg2)

            # Prove equivalence
            result = self.z3.prove_theorem(
                smtlib2=equivalence_formula,
                timeout=10000,
                correlation_id=correlation_id
            )

            if result.status == 'VALID':
                return True, structural_score, result.proof

        return False, structural_score, None
```

**Benefits:**
- Formal proof of behavioral equivalence
- Not just structural similarity
- Machine-verified isomorphism

---

### Priority 5: MEDIUM - LLTL Contradiction Detection

**Location:** `glue/adapters/rese-lltl/src/lltl_adapter.py`

**Current Implementation (Lines 378-421):**
```python
# Naive O(n²) DITO
def detect_contradictions(self, constraints):
    # Uses naive pairwise comparison
```

**Z3 Integration:**
```python
from z3prover_integration import Z3ProverIntegration

class LogicToLossTranslator:
    def __init__(self):
        self.z3 = Z3ProverIntegration()

    def detect_contradictions(self, formal_commitments, correlation_id):
        # 1. Convert formal commitments to Z3 formulas
        z3_formulas = [fc.to_z3_formula() for fc in formal_commitments]

        # 2. Use Z3 to detect contradictions (O(n log n))
        result = self.z3.solve_constraints(
            smtlib2=z3.Formula.And(*z3_formulas),
            timeout=5000,
            correlation_id=correlation_id
        )

        # 3. If UNSAT, extract unsat core
        if result.status == 'UNSAT':
            contradictory = self._extract_unsat_core(result.unsat_core)
            return contradictory

        return []
```

**Benefits:**
- Replace naive DITO with Z3 optimization
- O(n²) → O(n log n) contradiction detection
- Minimal contradiction sets via unsat_core

---

## Implementation Strategy

### Phase 1: Quick Wins (Week 1-2)

**1. SCE Contradiction Detection**
- File: `glue/adapters/rese-sce/src/sce_bridge.py`
- Lines: 370-446, 448-490
- Changes:
  - Add Z3 import and initialization
  - Replace `detect_contradictions()` with Z3 version
  - Replace `_check_pairwise_contradiction()` with Z3 encoding
  - Add `_encode_to_z3()` method
  - Add `_extract_unsat_core()` method
- Testing: Run existing SCE tests, verify performance improvement

**2. Phase I Constraint Inversion**
- File: `glue/adapters/rese-phase1/src/phase1_executor.py`
- Lines: 896-980
- Changes:
  - Add Z3 import to `ConstraintHardener`
  - Replace `_invert_constraint()` with Z3.Not()
  - Add `_parse_to_fol()` method
  - Add satisfiability check
- Testing: Verify proper quantifier handling

**3. LLTL Contradiction Detection**
- File: `glue/adapters/rese-lltl/src/lltl_adapter.py`
- Lines: 378-421
- Changes:
  - Add Z3 import
  - Replace `detect_contradictions()` with Z3 version
  - Add formula conversion methods
- Testing: Verify DITO optimization

### Phase 2: Core Enhancements (Week 3-4)

**4. Phase III MCTS Constraint Satisfaction**
- File: `glue/adapters/rese-phase3/src/phase3_executor.py`
- Lines: 929-1077
- Changes:
  - Add Z3 import to `MCTSSearchExecutor`
  - Add `_is_constraint_satisfiable()` method
  - Add `_verify_hypothesis_satisfies_constraints()` method
  - Add constraint checking before node expansion
  - Add pruning logic for unsat branches
- Testing: Benchmark MCTS with/without Z3 pruning

**5. Phase I Constraint Hardening**
- File: `glue/adapters/rese-phase1/src/phase1_executor.py`
- Lines: 909-959
- Changes:
  - Encode constraints as Z3 formulas in `harden_constraints()`
  - Add Z3-based constraint validation
  - Extract implications using Z3
- Testing: Verify constraint quality improvement

### Phase 3: Advanced Features (Week 5-6)

**6. Phase II Isomorphism Verification**
- File: `glue/adapters/rese-phase2/src/phase2_executor.py`
- Lines: 297-392
- Changes:
  - Add Z3 import to `CrossDomainMapper`
  - Implement behavioral equivalence encoding
  - Add formal isomorphism proofs
- Testing: Verify I_mech improvement

**7. Hypothesis Validation Enhancement**
- File: `glue/adapters/rese-phase3/src/phase3_executor.py`
- Lines: 557-753
- Changes:
  - Add Z3 verification to `HypothesisValidator`
  - Prove hypothesis satisfies all constraints
  - Add SMT-based validation
- Testing: Verify hypothesis quality

---

## CLAUDE.md Compliance

All Z3 integrations will follow CLAUDE.md principles:

✅ **Law of Air Gap**
- No direct imports from `core-projects/z3prover/`
- Use existing root-level Z3 integration (glue layer)

✅ **Law of Runtime Truth**
- Run probe scripts before relying on Z3
- Verify Z3 API availability
- Use existing `glue/adapters/z3-adapter/probes/check_api.sh`

✅ **Law of Configuration Explicitness**
- All Z3 config via environment variables
- Use existing `z3_config.yaml`
- Crash if Z3 not available (fail-fast)

✅ **Law of Idempotency**
- Z3 solving is deterministic (same input → same output)
- Safe to run multiple times

✅ **Circuit Breaker Pattern**
- Use existing Z3 circuit breaker (5 failures → open)
- Exponential backoff retry
- Graceful degradation when Z3 unavailable

✅ **Structured Logging**
- JSON lines with correlation_id
- Include Z3 solving metrics
- Track solving time and memory usage

---

## Configuration

### Environment Variables

```bash
# Z3 Configuration (using existing infrastructure)
Z3_CONFIG_PATH=./z3_config.yaml
Z3_ENABLED=true
Z3_TIMEOUT=5000
Z3_MAX_MEMORY_MB=4096
Z3_NUM_THREADS=4

# RESE-Specific Z3 Settings
RESE_Z3_SCE_ENABLED=true
RESE_Z3_PHASE1_ENABLED=true
RESE_Z3_PHASE2_ENABLED=true
RESE_Z3_PHASE3_ENABLED=true
RESE_Z3_LLTL_ENABLED=true

# Z3-LeanAide Bridge (optional, for cross-verification)
LEANAIDE_ENABLED=true
LEANAIDE_PORT=7654
Z3_LEANAIDE_BRIDGE_ENABLED=true
BRIDGE_DEFAULT_STRATEGY=adaptive
```

### Dependencies

**Required (Already in Root):**
- `z3-solver` >= 4.12.0
- Existing `z3prover_integration.py`
- Existing `z3prover_advanced.py`
- Existing `z3_leanaide_bridge.py`

**RESE-Specific:**
- No new dependencies required
- Leverage existing infrastructure

---

## Testing Strategy

### Unit Tests

For each integration point:
1. Test Z3 encoding correctness
2. Test SMT solving returns expected results
3. Test unsat core extraction
4. Test constraint inversion
5. Test error handling (timeout, unsat, etc.)

### Integration Tests

1. **SCE Integration**:
   - Verify contradiction detection with Z3
   - Compare results: naive vs Z3
   - Benchmark: O(n²) vs O(n log n)

2. **Phase I Integration**:
   - Verify constraint hardening with Z3
   - Test proper quantifier negation
   - Verify satisfiability checking

3. **Phase III Integration**:
   - Verify MCTS pruning with Z3
   - Benchmark: nodes pruned, search time reduction
   - Test constraint satisfaction checking

4. **End-to-End**:
   - Run full RESE pipeline with Z3 enabled
   - Verify all phases use Z3 correctly
   - Track performance improvements

### Performance Benchmarks

**SCE Contradiction Detection:**
- Baseline: 100 constraints → ~10,000 operations (O(n²))
- With Z3: 100 constraints → ~700 operations (O(n log n))
- Expected: **10-100x improvement**

**Phase III MCTS:**
- Baseline: Explores invalid branches
- With Z3: Prunes invalid branches early
- Expected: **10-100x speedup**

---

## Success Criteria

### Phase 1 Success Criteria
- [ ] SCE contradiction detection uses Z3
- [ ] Performance improvement >10x
- [ ] All existing tests pass
- [ ] Unsat core extraction works

### Phase 2 Success Criteria
- [ ] Phase I constraint inversion uses Z3.Not()
- [ ] Phase III MCTS prunes invalid branches
- [ ] MCTS speedup >10x
- [ ] All integration tests pass

### Phase 3 Success Criteria
- [ ] Phase II isomorphism verified with Z3
- [ ] Hypothesis validation uses SMT proofs
- [ ] End-to-end pipeline functional
- [ ] Performance benchmarks met

---

## Rollout Strategy

### Week 1: SCE Integration
1. Modify `sce_bridge.py`
2. Add Z3 encoding methods
3. Test with existing test suite
4. Benchmark performance
5. Deploy if tests pass

### Week 2: Phase I & LLTL Integration
1. Modify Phase I constraint hardening
2. Modify LLTL contradiction detection
3. Test with existing test suites
4. Integration tests
5. Deploy if tests pass

### Week 3-4: Phase III Integration
1. Modify MCTS executor
2. Add constraint satisfaction checking
3. Add pruning logic
4. Benchmark MCTS performance
5. Deploy if significant improvement

### Week 5-6: Phase II & Validation
1. Modify Phase II isomorphism verification
2. Enhance hypothesis validation
3. End-to-end testing
4. Performance tuning
5. Deploy if all tests pass

---

## Risk Mitigation

### Technical Risks

**Risk 1: Z3 Performance**
- **Mitigation**: Set appropriate timeouts (5-10s)
- **Fallback**: Graceful degradation to naive methods
- **Monitoring**: Track solving time, fail fast if too slow

**Risk 2: Z3 Unavailability**
- **Mitigation**: Circuit breaker pattern
- **Fallback**: Use existing naive methods
- **Monitoring**: Health checks every 30s

**Risk 3: Incorrect Encoding**
- **Mitigation**: Extensive testing
- **Validation**: Cross-check with existing results
- **Rollback**: Quick revert to existing code

### Operational Risks

**Risk 4: Breaking Changes**
- **Mitigation:** Feature flags (environment variables)
- **Rollback:** Keep existing code as fallback
- **Testing**: Comprehensive test suite

**Risk 5: Resource Consumption**
- **Mitigation**: Memory limits, timeouts
- **Monitoring**: Track Z3 memory usage
- **Throttling**: Rate limit if needed

---

## Next Steps

**Immediate Actions:**
1. ✅ Review existing Z3 integration (17,000+ lines)
2. ✅ Identify RESE integration points (5 components)
3. ✅ Create implementation plan
4. ⏭️ **Deploy agents to implement Z3 integrations**

**Implementation Order:**
1. **SCE** contradiction detection (CRITICAL)
2. **Phase I** constraint hardening (HIGH)
3. **LLTL** contradiction detection (HIGH)
4. **Phase III** MCTS constraint satisfaction (HIGH)
5. **Phase II** isomorphism verification (MEDIUM)

**After Implementation:**
- Run comprehensive tests
- Benchmark performance
- Deploy to production
- Monitor metrics
- Iterate based on results

---

**End of Z3 Integration Plan**

*Prepared:* 2026-02-04
*Status:* Ready for Implementation
*Next:* Deploy agents to implement Z3 integrations
