<<<<<<< HEAD
# RESE Framework Phase 1 & Phase 2 Debug & Validation Report

**Generated**: 2025-12-31
**Target Modules**:
- Phase 1: Φ₁.₅ Tacit Assumption Miner, Φ₂ Metacognitive Debiasing
- Phase 2: Ψ₃ Constraint Inversion, Ψ₂ Ontology Mapping, I_mech Isomorphism Validator
**Author**: Claude Code (RESE Testing/QA Agent)

---

## Executive Summary

This report documents comprehensive debugging and validation of the RESE framework Phase 1 and Phase 2 modules. Testing methodology includes unit tests, integration tests, performance benchmarks, and validation of KEY INNOVATIONS.

### Key Findings

| Module | Status | Tests Run | Passed | Failed | Performance |
|--------|--------|-----------|--------|--------|-------------|
| Φ₁.₅ Tacit Assumption Miner | 🟢 Active | 150+ | Pending | Pending | Target: <10s for 1K failures |
| Φ₂ Metacognitive Debiasing | 🟢 Active | 100+ | Pending | Pending | N/A |
| Ψ₃ Constraint Inversion | 🟢 Active | 150+ | Pending | Pending | Target: 6.6x average reduction |
| Ψ₂ Ontology Mapping | 🟡 Partial | 50+ | Pending | Pending | N/A |
| I_mech Isomorphism Validator | 🟢 Active | 80+ | Pending | Pending | Target: <30s for domain pairs |

---

## Test Structure

### Phase 1 Tests

#### 1. Φ₁.₅ Tacit Assumption Miner (`test_phi15.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_phi15.py`

**Test Classes** (618 lines):
- `TestDataStructures`: NullResult, TacitAssumption creation & serialization
- `TestFailurePreprocessor`: Feature extraction, keyword extraction, time-to-failure
- `TestAnomalyDetector`: Isolation forest-based anomaly detection
- `TestFailureClusterer`: DBSCAN clustering of failures
- `TestAssumptionGenerator`: Generating assumptions from clusters
- `TestConfidenceScorer`: Multi-factor confidence scoring
- `TestParadigmShiftDetector`: Crisis detection from assumption patterns
- `TestPhi15Engine`: Main engine orchestration
- `TestIntegration`: End-to-end pipeline, SCE constraint conversion

**Key Test Coverage**:
- Data structure serialization/deserialization
- Feature vector extraction (5-dimensional)
- Anomaly detection with contamination parameter
- Clustering with silhouette scoring
- Assumption generation with pattern types
- Confidence scoring based on cluster quality
- Paradigm shift detection with configurable threshold
- Full pipeline integration

#### 2. Φ₂ Metacognitive Debiasing (`test_cognitive_biases.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\phase1\test_cognitive_biases.py`

**Test Classes** (501 lines):
- `TestBiasDetection`: Confirmation bias, overconfidence, anchoring, availability, authority
- `TestDebiasingStrategies`: Consider-the-opposite, devil's advocate, pre-mortem, reformulation
- `TestBiasDetectionAccuracy`: Illusion of control, framing effect, availability bias
- `TestEdgeCases`: Empty constraints, single constraint, long descriptions, special characters
- `TestRecommendationGeneration`: High/low bias recommendations

**Bias Types Tested**:
- Confirmation Bias
- Overconfidence
- Anchoring Bias
- Sunk Cost Fallacy
- Availability Heuristic
- Authority Bias
- Illusion of Control
- Framing Effect

#### 3. Φ₂ Integration (`test_phi2_integration.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\phase1\test_phi2_integration.py`

**Test Classes** (449 lines):
- `TestSCEIntegration`: Integration with Symbolic Constraint Engine
- `TestStage5Integration`: Real-time monitoring during solution generation
- `TestEndToEndWorkflows`: Complete constraint-to-solution workflows
- `TestErrorHandling`: Nonexistent constraints, invalid steps

### Phase 2 Tests

#### 4. I_mech Isomorphism Validator (`test_validator.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_imech\test_validator.py`

**Test Classes** (201 lines):
- `TestIMechValidator`: Identical domains, different domains, solution transfer, caching
- `TestCompareDomains`: Convenience function testing

**Key Capabilities Tested**:
- Structural similarity scoring
- Node mapping computation
- Solution transfer validation
- Result caching
- Finding analogous domains

#### 5. I_mech Integration (`test_integration.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_imech\test_integration.py`

**Test Classes** (326 lines):
- `TestFullPipeline`: Isomorphic domains, partial isomorphism, solution transfer
- `TestHistoricalAnalogies`: Edison's light bulb, steam→combustion engine, telegraph→telephone
- `TestPerformance`: Small graphs (10 nodes), medium graphs (50 nodes)

**Historical Analogies Tested**:
- Candle → Electric Light Bulb
- Steam Engine → Internal Combustion Engine
- Telegraph → Telephone

#### 6. Ψ₃ Constraint Inversion (`test_constraint_inverter.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\phase2\psi3\tests\unit\test_constraint_inverter.py`

**Test Classes** (583 lines):
- `TestConstraint`: Creation, hashing, equality, complexity estimation
- `TestExpression`: Variable, constant, boolean ops, arithmetic ops, nested expressions
- `TestSyntacticPreprocessing`: Duplicate removal, subsumption removal, redundancy estimation
- `TestDependencyAnalysis`: Graph construction, redundant constraints, implications, SCCs
- `TestMinimalCover`: Minimal cover generation, component solving
- `TestIntegration`: Full pipeline on hierarchical/independent constraints

**4-Stage Pipeline**:
1. Syntactic Preprocessing (40 tests)
2. Dependency Analysis (30 tests)
3. Minimal Cover Generation (30 tests)
4. Verification & Output (20 tests)

### Integration Tests

#### 7. Phase 1 Integration (`test_phase1_integration.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_integration\test_phase1_integration.py`

**Test Classes** (619 lines):
- `TestPhi15EndToEnd`: Systematic/diverse failure patterns, top-k assumptions, paradigm shift
- `TestPhi15ComponentIntegration`: Preprocessor→detector→clusterer→generator→scorer
- `TestPhi15DataFlow`: Serialization roundtrips, feature vector consistency
- `TestPhi15Performance`: Large dataset (500 failures), incremental processing
- `TestPhi15Validation`: Accuracy validation (target >70%)

**Performance Benchmarks**:
- Systematic pipeline: <30 seconds
- Large dataset (500): <60 seconds
- Accuracy threshold: >70%

---

## Running the Tests

### Command Line Execution

```bash
# Navigate to project root
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Run all Phase 1 tests
pytest rese/tests/phase1/ -v

# Run all I_mech tests
pytest rese/tests/test_imech/ -v

# Run integration tests
pytest rese/tests/test_integration/test_phase1_integration.py -v
pytest rese/tests/test_integration/test_phase2_integration.py -v

# Run with coverage
pytest rese/tests/phase1/ rese/tests/test_imech/ --cov=rese --cov-report=html

# Run specific test file
pytest rese/tests/test_phi15.py -v

# Run specific test class
pytest rese/tests/test_phi15.py::TestPhi15Engine -v

# Run specific test
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_engine_initialization -v

# Run with performance benchmarks
pytest rese/tests/ -m performance -v

# Run with debug output
pytest rese/tests/phase1/ -v -s --tb=long
```

### Test Markers

```bash
# Run only unit tests
pytest rese/tests/ -m unit -v

# Run only integration tests
pytest rese/tests/ -m integration -v

# Run only performance tests
pytest rese/tests/ -m performance -v

# Run only Phase 1 tests
pytest rese/tests/ -m phase1 -v

# Run only Phase 2 tests
pytest rese/tests/ -m phase2 -v

# Skip slow tests
pytest rese/tests/ -m "not slow" -v
```

---

## Debugging Workflow

### Step 1: Run All Tests

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Run Phase 1 tests
pytest rese/tests/phase1/ -v --tb=short 2>&1 | tee phase1_test_results.txt

# Run I_mech tests
pytest rese/tests/test_imech/ -v --tb=short 2>&1 | tee imech_test_results.txt

# Run integration tests
pytest rese/tests/test_integration/ -v --tb=short 2>&1 | tee integration_test_results.txt
```

### Step 2: Identify Failing Tests

Check output for:
- FAILED markers
- Error messages
- Stack traces
- Assertion failures

### Step 3: Debug Individual Failures

```bash
# Run with full traceback
pytest <test_file>::<test_class>::<test_method> -v -s --tb=long

# Run with debugger
pytest <test_file>::<test_class>::<test_method> -v --pdb

# Run with logging
pytest <test_file>::<test_class>::<test_method> -v --log-cli-level=DEBUG
```

### Step 4: Fix Issues

Common issues to check:
1. **Import errors**: Verify paths in `sys.path`
2. **Missing dependencies**: Install required packages
3. **Data type mismatches**: Check expected vs actual types
4. **Algorithm correctness**: Verify logic implementation
5. **Performance issues**: Profile slow code sections

### Step 5: Re-run Tests

```bash
# Re-run specific test
pytest <test_file>::<test_class>::<test_method> -v

# Re-run all tests in file
pytest <test_file> -v

# Run until first failure
pytest -x
```

---

## Known Issues & Solutions

### Issue 1: Module Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'phase1.tacit_assumption_miner'`

**Solution**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue 2: Missing Test Fixtures

**Symptom**: `Fixture 'sample_null_result' not found`

**Solution**: Ensure `conftest.py` is in the tests directory and contains the fixture definitions.

### Issue 3: Z3 Solver Not Available

**Symptom**: Tests skip with "Z3 not available"

**Solution**:
```bash
pip install z3-solver
```

Or skip Z3-dependent tests:
```bash
pytest -m "not z3"
```

### Issue 4: NumPy Version Incompatibility

**Symptom**: `AttributeError: module 'numpy' has no attribute 'float'`

**Solution**:
```bash
pip install numpy==1.24.0
```

---

## Performance Benchmarks

### Φ₁.₅ Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| 1K failures processing | <10s | `test_large_dataset_performance` |
| Assumption accuracy | >70% | `test_phi15_accuracy_validation` |
| Pipeline overhead | <5s | `test_complete_pipeline_*` |

### Ψ₃ Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Average reduction ratio | 6.6x | `test_full_pipeline_hierarchical` |
| Hierarchical constraints | 10x+ | `test_full_pipeline_hierarchical` |
| Independent constraints | 1x (no reduction) | `test_full_pipeline_independent` |

### I_mech Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Small graphs (10 nodes) | <5s | `test_small_graphs_performance` |
| Medium graphs (50 nodes) | <30s | `test_medium_graphs_performance` |
| Isomorphism detection | >0.7 score | `test_full_pipeline_isomorphic_domains` |

---

## Validation Criteria

### KEY INNOVATIONS Validation

#### Φ₁.₅: Tacit Assumption Miner
- **Innovation**: Mining hidden assumptions from failure patterns
- **Target**: >70% accuracy on labeled assumption data
- **Validation**: `validate_phi15_accuracy(predictions, ground_truth, min_accuracy=0.70)`

#### Φ₂: Metacognitive Debiasing
- **Innovation**: Debiasing constraints with cognitive science
- **Target**: Detect all 8+ bias types with >50% confidence
- **Validation**: Bias detection coverage across all bias types

#### Ψ₃: Constraint Inversion
- **Innovation**: 6.6x average constraint reduction
- **Target**: Achieve 6.6x reduction on hierarchical constraints
- **Validation**: `reduction_ratio >= 6.6` on hierarchical datasets

#### Ψ₂: Ontology Mapping
- **Innovation**: Cross-domain knowledge graph alignment
- **Target**: >80% precision on entity mapping
- **Validation**: Entity mapping precision and recall

#### I_mech: Isomorphism Validator
- **Innovation**: Transfer solutions across mechanistically similar domains
- **Target**: >80% successful transfer on isomorphic domains
- **Validation**: `validate_transfer_success()` on known analogies

---

## Test Coverage

### Phase 1 Coverage

| Component | Lines Covered | % | Tests |
|-----------|--------------|---|-------|
| Φ₁.₅ Core | TBD | - | 150+ |
| Φ₂ Detection | TBD | - | 100+ |
| Integration | TBD | - | 50+ |

### Phase 2 Coverage

| Component | Lines Covered | % | Tests |
|-----------|--------------|---|-------|
| Ψ₃ Pipeline | TBD | - | 150+ |
| Ψ₂ Mapping | TBD | - | 50+ |
| I_mech Core | TBD | - | 80+ |
| Integration | TBD | - | 40+ |

---

## Debug Report Template

When reporting bugs, use this template:

```markdown
## Bug Report

**Module**: [Φ₁.₅ / Φ₂ / Ψ₃ / Ψ₂ / I_mech]
**Test File**: [test_file.py]
**Test Method**: [test_method]
**Priority**: [High / Medium / Low]

### Description
[Brief description of the issue]

### Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Error Message
```
[Error traceback]
```

### Environment
- Python Version: [3.9+]
- NumPy Version: [1.24+]
- OS: [Windows/Linux/Mac]

### Fix Status
- [ ] Identified
- [ ] Fixed
- [ ] Tested
- [ ] Verified
```

---

## Next Steps

### Immediate Actions

1. **Run all tests** and capture results
2. **Document all failures** with error messages
3. **Categorize issues** by severity and module
4. **Fix critical bugs** first
5. **Re-run tests** to verify fixes
6. **Validate performance** against benchmarks
7. **Generate final report** with all metrics

### Test Execution Checklist

- [ ] Run Phase 1 unit tests
- [ ] Run Phase 1 integration tests
- [ ] Run I_mech unit tests
- [ ] Run I_mech integration tests
- [ ] Run Ψ₃ unit tests
- [ ] Run Ψ₂ tests (if available)
- [ ] Run full integration tests
- [ ] Run performance benchmarks
- [ ] Validate KEY INNOVATIONS
- [ ] Generate coverage report
- [ ] Document all bugs found
- [ ] Verify all fixes

---

## Appendix: Test Data

### Sample Null Results

Generated by `mock_failure_generator` in `conftest.py`:

```python
{
    "attempt_id": "test_000",
    "timestamp": <datetime>,
    "problem_type": "optimization",
    "approach_type": "deterministic",
    "constraints": ["c1", "c2", "c3"],
    "error_type": "OPTIMIZATION_FAILED",
    "error_message": "Solver failed to converge",
    "state": {"iteration": 100, "objective": -999},
    "iteration": 100,
    "resources_used": {"cpu": 50.0, "memory": 100.0},
    "metadata": {"test": True}
}
```

### Sample Constraints

Generated by `mock_constraint_generator` in `conftest.py`:

```python
{
    "id": "constraint_0",
    "type": "inequality",
    "variables": ["x0", "x1", "x2"],
    "expression": "complex_expr_3_vars_3",
    "priority": 5,
    "verified": True,
    "verification_time": 1.5,
    "tightness": 0.75
}
```

### Sample Domains (I_mech)

```python
Domain(
    id="control_system",
    name="Control System",
    description="Feedback control system with gain and damping",
    formal_constraints=["gain * error > 0", "damping >= 0 and damping <= 1"],
    natural_language_constraints=["Gain must be positive", "Damping in [0,1]"],
    fdg=<FunctionalDependencyGraph>
)
```

---

## Contact & Support

For questions or issues with testing:
- **Testing Framework**: Based on pytest 7.0+
- **Test Utilities**: `rese/tests/test_utils.py`
- **Fixtures**: `rese/tests/conftest.py`
- **Documentation**: See individual module READMEs

---

**Status**: 🟡 Testing In Progress
**Last Updated**: 2025-12-31
**Version**: 1.0.0
=======
# RESE Framework Phase 1 & Phase 2 Debug & Validation Report

**Generated**: 2025-12-31
**Target Modules**:
- Phase 1: Φ₁.₅ Tacit Assumption Miner, Φ₂ Metacognitive Debiasing
- Phase 2: Ψ₃ Constraint Inversion, Ψ₂ Ontology Mapping, I_mech Isomorphism Validator
**Author**: Claude Code (RESE Testing/QA Agent)

---

## Executive Summary

This report documents comprehensive debugging and validation of the RESE framework Phase 1 and Phase 2 modules. Testing methodology includes unit tests, integration tests, performance benchmarks, and validation of KEY INNOVATIONS.

### Key Findings

| Module | Status | Tests Run | Passed | Failed | Performance |
|--------|--------|-----------|--------|--------|-------------|
| Φ₁.₅ Tacit Assumption Miner | 🟢 Active | 150+ | Pending | Pending | Target: <10s for 1K failures |
| Φ₂ Metacognitive Debiasing | 🟢 Active | 100+ | Pending | Pending | N/A |
| Ψ₃ Constraint Inversion | 🟢 Active | 150+ | Pending | Pending | Target: 6.6x average reduction |
| Ψ₂ Ontology Mapping | 🟡 Partial | 50+ | Pending | Pending | N/A |
| I_mech Isomorphism Validator | 🟢 Active | 80+ | Pending | Pending | Target: <30s for domain pairs |

---

## Test Structure

### Phase 1 Tests

#### 1. Φ₁.₅ Tacit Assumption Miner (`test_phi15.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_phi15.py`

**Test Classes** (618 lines):
- `TestDataStructures`: NullResult, TacitAssumption creation & serialization
- `TestFailurePreprocessor`: Feature extraction, keyword extraction, time-to-failure
- `TestAnomalyDetector`: Isolation forest-based anomaly detection
- `TestFailureClusterer`: DBSCAN clustering of failures
- `TestAssumptionGenerator`: Generating assumptions from clusters
- `TestConfidenceScorer`: Multi-factor confidence scoring
- `TestParadigmShiftDetector`: Crisis detection from assumption patterns
- `TestPhi15Engine`: Main engine orchestration
- `TestIntegration`: End-to-end pipeline, SCE constraint conversion

**Key Test Coverage**:
- Data structure serialization/deserialization
- Feature vector extraction (5-dimensional)
- Anomaly detection with contamination parameter
- Clustering with silhouette scoring
- Assumption generation with pattern types
- Confidence scoring based on cluster quality
- Paradigm shift detection with configurable threshold
- Full pipeline integration

#### 2. Φ₂ Metacognitive Debiasing (`test_cognitive_biases.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\phase1\test_cognitive_biases.py`

**Test Classes** (501 lines):
- `TestBiasDetection`: Confirmation bias, overconfidence, anchoring, availability, authority
- `TestDebiasingStrategies`: Consider-the-opposite, devil's advocate, pre-mortem, reformulation
- `TestBiasDetectionAccuracy`: Illusion of control, framing effect, availability bias
- `TestEdgeCases`: Empty constraints, single constraint, long descriptions, special characters
- `TestRecommendationGeneration`: High/low bias recommendations

**Bias Types Tested**:
- Confirmation Bias
- Overconfidence
- Anchoring Bias
- Sunk Cost Fallacy
- Availability Heuristic
- Authority Bias
- Illusion of Control
- Framing Effect

#### 3. Φ₂ Integration (`test_phi2_integration.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\phase1\test_phi2_integration.py`

**Test Classes** (449 lines):
- `TestSCEIntegration`: Integration with Symbolic Constraint Engine
- `TestStage5Integration`: Real-time monitoring during solution generation
- `TestEndToEndWorkflows`: Complete constraint-to-solution workflows
- `TestErrorHandling`: Nonexistent constraints, invalid steps

### Phase 2 Tests

#### 4. I_mech Isomorphism Validator (`test_validator.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_imech\test_validator.py`

**Test Classes** (201 lines):
- `TestIMechValidator`: Identical domains, different domains, solution transfer, caching
- `TestCompareDomains`: Convenience function testing

**Key Capabilities Tested**:
- Structural similarity scoring
- Node mapping computation
- Solution transfer validation
- Result caching
- Finding analogous domains

#### 5. I_mech Integration (`test_integration.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_imech\test_integration.py`

**Test Classes** (326 lines):
- `TestFullPipeline`: Isomorphic domains, partial isomorphism, solution transfer
- `TestHistoricalAnalogies`: Edison's light bulb, steam→combustion engine, telegraph→telephone
- `TestPerformance`: Small graphs (10 nodes), medium graphs (50 nodes)

**Historical Analogies Tested**:
- Candle → Electric Light Bulb
- Steam Engine → Internal Combustion Engine
- Telegraph → Telephone

#### 6. Ψ₃ Constraint Inversion (`test_constraint_inverter.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\phase2\psi3\tests\unit\test_constraint_inverter.py`

**Test Classes** (583 lines):
- `TestConstraint`: Creation, hashing, equality, complexity estimation
- `TestExpression`: Variable, constant, boolean ops, arithmetic ops, nested expressions
- `TestSyntacticPreprocessing`: Duplicate removal, subsumption removal, redundancy estimation
- `TestDependencyAnalysis`: Graph construction, redundant constraints, implications, SCCs
- `TestMinimalCover`: Minimal cover generation, component solving
- `TestIntegration`: Full pipeline on hierarchical/independent constraints

**4-Stage Pipeline**:
1. Syntactic Preprocessing (40 tests)
2. Dependency Analysis (30 tests)
3. Minimal Cover Generation (30 tests)
4. Verification & Output (20 tests)

### Integration Tests

#### 7. Phase 1 Integration (`test_phase1_integration.py`)
**Location**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_integration\test_phase1_integration.py`

**Test Classes** (619 lines):
- `TestPhi15EndToEnd`: Systematic/diverse failure patterns, top-k assumptions, paradigm shift
- `TestPhi15ComponentIntegration`: Preprocessor→detector→clusterer→generator→scorer
- `TestPhi15DataFlow`: Serialization roundtrips, feature vector consistency
- `TestPhi15Performance`: Large dataset (500 failures), incremental processing
- `TestPhi15Validation`: Accuracy validation (target >70%)

**Performance Benchmarks**:
- Systematic pipeline: <30 seconds
- Large dataset (500): <60 seconds
- Accuracy threshold: >70%

---

## Running the Tests

### Command Line Execution

```bash
# Navigate to project root
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Run all Phase 1 tests
pytest rese/tests/phase1/ -v

# Run all I_mech tests
pytest rese/tests/test_imech/ -v

# Run integration tests
pytest rese/tests/test_integration/test_phase1_integration.py -v
pytest rese/tests/test_integration/test_phase2_integration.py -v

# Run with coverage
pytest rese/tests/phase1/ rese/tests/test_imech/ --cov=rese --cov-report=html

# Run specific test file
pytest rese/tests/test_phi15.py -v

# Run specific test class
pytest rese/tests/test_phi15.py::TestPhi15Engine -v

# Run specific test
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_engine_initialization -v

# Run with performance benchmarks
pytest rese/tests/ -m performance -v

# Run with debug output
pytest rese/tests/phase1/ -v -s --tb=long
```

### Test Markers

```bash
# Run only unit tests
pytest rese/tests/ -m unit -v

# Run only integration tests
pytest rese/tests/ -m integration -v

# Run only performance tests
pytest rese/tests/ -m performance -v

# Run only Phase 1 tests
pytest rese/tests/ -m phase1 -v

# Run only Phase 2 tests
pytest rese/tests/ -m phase2 -v

# Skip slow tests
pytest rese/tests/ -m "not slow" -v
```

---

## Debugging Workflow

### Step 1: Run All Tests

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Run Phase 1 tests
pytest rese/tests/phase1/ -v --tb=short 2>&1 | tee phase1_test_results.txt

# Run I_mech tests
pytest rese/tests/test_imech/ -v --tb=short 2>&1 | tee imech_test_results.txt

# Run integration tests
pytest rese/tests/test_integration/ -v --tb=short 2>&1 | tee integration_test_results.txt
```

### Step 2: Identify Failing Tests

Check output for:
- FAILED markers
- Error messages
- Stack traces
- Assertion failures

### Step 3: Debug Individual Failures

```bash
# Run with full traceback
pytest <test_file>::<test_class>::<test_method> -v -s --tb=long

# Run with debugger
pytest <test_file>::<test_class>::<test_method> -v --pdb

# Run with logging
pytest <test_file>::<test_class>::<test_method> -v --log-cli-level=DEBUG
```

### Step 4: Fix Issues

Common issues to check:
1. **Import errors**: Verify paths in `sys.path`
2. **Missing dependencies**: Install required packages
3. **Data type mismatches**: Check expected vs actual types
4. **Algorithm correctness**: Verify logic implementation
5. **Performance issues**: Profile slow code sections

### Step 5: Re-run Tests

```bash
# Re-run specific test
pytest <test_file>::<test_class>::<test_method> -v

# Re-run all tests in file
pytest <test_file> -v

# Run until first failure
pytest -x
```

---

## Known Issues & Solutions

### Issue 1: Module Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'phase1.tacit_assumption_miner'`

**Solution**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue 2: Missing Test Fixtures

**Symptom**: `Fixture 'sample_null_result' not found`

**Solution**: Ensure `conftest.py` is in the tests directory and contains the fixture definitions.

### Issue 3: Z3 Solver Not Available

**Symptom**: Tests skip with "Z3 not available"

**Solution**:
```bash
pip install z3-solver
```

Or skip Z3-dependent tests:
```bash
pytest -m "not z3"
```

### Issue 4: NumPy Version Incompatibility

**Symptom**: `AttributeError: module 'numpy' has no attribute 'float'`

**Solution**:
```bash
pip install numpy==1.24.0
```

---

## Performance Benchmarks

### Φ₁.₅ Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| 1K failures processing | <10s | `test_large_dataset_performance` |
| Assumption accuracy | >70% | `test_phi15_accuracy_validation` |
| Pipeline overhead | <5s | `test_complete_pipeline_*` |

### Ψ₃ Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Average reduction ratio | 6.6x | `test_full_pipeline_hierarchical` |
| Hierarchical constraints | 10x+ | `test_full_pipeline_hierarchical` |
| Independent constraints | 1x (no reduction) | `test_full_pipeline_independent` |

### I_mech Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Small graphs (10 nodes) | <5s | `test_small_graphs_performance` |
| Medium graphs (50 nodes) | <30s | `test_medium_graphs_performance` |
| Isomorphism detection | >0.7 score | `test_full_pipeline_isomorphic_domains` |

---

## Validation Criteria

### KEY INNOVATIONS Validation

#### Φ₁.₅: Tacit Assumption Miner
- **Innovation**: Mining hidden assumptions from failure patterns
- **Target**: >70% accuracy on labeled assumption data
- **Validation**: `validate_phi15_accuracy(predictions, ground_truth, min_accuracy=0.70)`

#### Φ₂: Metacognitive Debiasing
- **Innovation**: Debiasing constraints with cognitive science
- **Target**: Detect all 8+ bias types with >50% confidence
- **Validation**: Bias detection coverage across all bias types

#### Ψ₃: Constraint Inversion
- **Innovation**: 6.6x average constraint reduction
- **Target**: Achieve 6.6x reduction on hierarchical constraints
- **Validation**: `reduction_ratio >= 6.6` on hierarchical datasets

#### Ψ₂: Ontology Mapping
- **Innovation**: Cross-domain knowledge graph alignment
- **Target**: >80% precision on entity mapping
- **Validation**: Entity mapping precision and recall

#### I_mech: Isomorphism Validator
- **Innovation**: Transfer solutions across mechanistically similar domains
- **Target**: >80% successful transfer on isomorphic domains
- **Validation**: `validate_transfer_success()` on known analogies

---

## Test Coverage

### Phase 1 Coverage

| Component | Lines Covered | % | Tests |
|-----------|--------------|---|-------|
| Φ₁.₅ Core | TBD | - | 150+ |
| Φ₂ Detection | TBD | - | 100+ |
| Integration | TBD | - | 50+ |

### Phase 2 Coverage

| Component | Lines Covered | % | Tests |
|-----------|--------------|---|-------|
| Ψ₃ Pipeline | TBD | - | 150+ |
| Ψ₂ Mapping | TBD | - | 50+ |
| I_mech Core | TBD | - | 80+ |
| Integration | TBD | - | 40+ |

---

## Debug Report Template

When reporting bugs, use this template:

```markdown
## Bug Report

**Module**: [Φ₁.₅ / Φ₂ / Ψ₃ / Ψ₂ / I_mech]
**Test File**: [test_file.py]
**Test Method**: [test_method]
**Priority**: [High / Medium / Low]

### Description
[Brief description of the issue]

### Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Error Message
```
[Error traceback]
```

### Environment
- Python Version: [3.9+]
- NumPy Version: [1.24+]
- OS: [Windows/Linux/Mac]

### Fix Status
- [ ] Identified
- [ ] Fixed
- [ ] Tested
- [ ] Verified
```

---

## Next Steps

### Immediate Actions

1. **Run all tests** and capture results
2. **Document all failures** with error messages
3. **Categorize issues** by severity and module
4. **Fix critical bugs** first
5. **Re-run tests** to verify fixes
6. **Validate performance** against benchmarks
7. **Generate final report** with all metrics

### Test Execution Checklist

- [ ] Run Phase 1 unit tests
- [ ] Run Phase 1 integration tests
- [ ] Run I_mech unit tests
- [ ] Run I_mech integration tests
- [ ] Run Ψ₃ unit tests
- [ ] Run Ψ₂ tests (if available)
- [ ] Run full integration tests
- [ ] Run performance benchmarks
- [ ] Validate KEY INNOVATIONS
- [ ] Generate coverage report
- [ ] Document all bugs found
- [ ] Verify all fixes

---

## Appendix: Test Data

### Sample Null Results

Generated by `mock_failure_generator` in `conftest.py`:

```python
{
    "attempt_id": "test_000",
    "timestamp": <datetime>,
    "problem_type": "optimization",
    "approach_type": "deterministic",
    "constraints": ["c1", "c2", "c3"],
    "error_type": "OPTIMIZATION_FAILED",
    "error_message": "Solver failed to converge",
    "state": {"iteration": 100, "objective": -999},
    "iteration": 100,
    "resources_used": {"cpu": 50.0, "memory": 100.0},
    "metadata": {"test": True}
}
```

### Sample Constraints

Generated by `mock_constraint_generator` in `conftest.py`:

```python
{
    "id": "constraint_0",
    "type": "inequality",
    "variables": ["x0", "x1", "x2"],
    "expression": "complex_expr_3_vars_3",
    "priority": 5,
    "verified": True,
    "verification_time": 1.5,
    "tightness": 0.75
}
```

### Sample Domains (I_mech)

```python
Domain(
    id="control_system",
    name="Control System",
    description="Feedback control system with gain and damping",
    formal_constraints=["gain * error > 0", "damping >= 0 and damping <= 1"],
    natural_language_constraints=["Gain must be positive", "Damping in [0,1]"],
    fdg=<FunctionalDependencyGraph>
)
```

---

## Contact & Support

For questions or issues with testing:
- **Testing Framework**: Based on pytest 7.0+
- **Test Utilities**: `rese/tests/test_utils.py`
- **Fixtures**: `rese/tests/conftest.py`
- **Documentation**: See individual module READMEs

---

**Status**: 🟡 Testing In Progress
**Last Updated**: 2025-12-31
**Version**: 1.0.0
>>>>>>> 1cb9c5e35 (update)
