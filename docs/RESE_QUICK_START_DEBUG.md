<<<<<<< HEAD
# RESE Framework Quick Start Debugging Guide

**Last Updated**: 2025-12-31
**Purpose**: Rapid debugging and validation of Phase 1 & Phase 2 modules

---

## Quick Start (5 Minutes)

### 1. Run All Tests

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Run the comprehensive test suite
python run_rese_tests.py --phase all --verbose

# Or run with pytest directly
pytest rese/tests/phase1/ rese/tests/test_imech/ -v --tb=short
```

### 2. Check Results

Results will be saved to: `rese_test_results.json`

### 3. Read This Guide

Follow the sections below for detailed debugging steps.

---

## Module-Specific Debugging

### Φ₁.₅ Tacit Assumption Miner

**Location**: `rese/phase1/tacit_assumption_miner.py`
**Tests**: `rese/tests/test_phi15.py`

#### Common Issues

**Issue 1**: Import Error
```python
ModuleNotFoundError: No module named 'phase1.tacit_assumption_miner'
```
**Fix**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Issue 2**: Insufficient Data for Clustering
```
DBSCAN requires min_samples > 1
```
**Fix**: Use at least 30 null results for testing

**Issue 3**: Feature Vector Dimension Mismatch
```
ValueError: shapes (N, 5) != (M, 5)
```
**Fix**: Ensure all feature vectors have same dimension (5)

#### Debug Commands

```bash
# Run Φ₁.₅ tests only
pytest rese/tests/test_phi15.py -v

# Run specific test class
pytest rese/tests/test_phi15.py::TestPhi15Engine -v

# Run with verbose output
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_process_null_results -v -s

# Run with debugger
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_process_null_results --pdb
```

#### Validate Performance

```bash
# Run performance test
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v

# Check if <10s target met for 1K failures
```

---

### Φ₂ Metacognitive Debiasing

**Location**: `rese/phase1/cognitive_biases.py`
**Tests**: `rese/tests/phase1/test_cognitive_biases.py`

#### Common Issues

**Issue 1**: Bias Detection Not Triggering
```python
# Expected: detections > 0
# Actual: detections == 0
```
**Fix**: Use strongly biased phrases:
```python
# Good (high bias):
"This will certainly always work perfectly"
"We must maximize performance"

# Bad (low bias):
"The system should maintain accuracy"
"Consider using approximation"
```

**Issue 2**: Missing Debiasing Strategies
```
AttributeError: 'DebiasingStrategy' object has no attribute 'method_name'
```
**Fix**: Ensure all strategy methods are implemented

#### Debug Commands

```bash
# Run Φ₂ tests
pytest rese/tests/phase1/test_cognitive_biases.py -v

# Test specific bias
pytest rese/tests/phase1/test_cognitive_biases.py::TestBiasDetection::test_confirmation_bias_detection -v

# Test debiasing strategies
pytest rese/tests/phase1/test_cognitive_biases.py::TestDebiasingStrategies -v
```

#### Validate Bias Detection

Create test constraints:
```python
from rese.phase1.cognitive_biases import CognitiveBiasDetector
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

detector = CognitiveBiasDetector()

# High bias constraint
biased = Constraint(
    id="biased_1",
    type=ConstraintType.HARD,
    description="This will certainly achieve perfect results",
    formalization="perfect = true",
    source="expert"
)

report = detector.analyze_constraints([biased])
print(f"Detections: {report.total_detections}")
print(f"Bias Score: {report.overall_bias_score}")
```

---

### I_mech Isomorphism Validator

**Location**: `rese/phase2/imech/isomorphism_validator.py`
**Tests**: `rese/tests/test_imech/`

#### Common Issues

**Issue 1**: Domain Creation Error
```python
TypeError: __init__() missing required arguments
```
**Fix**: Ensure all required fields provided:
```python
from rese.phase2.imech import Domain, FunctionalDependencyGraph, Node, Edge

domain = Domain(
    id="test_domain",
    name="Test Domain",
    description="Test description",
    formal_constraints=["c1", "c2"],
    natural_language_constraints=["constraint 1", "constraint 2"]
)

# Must have FDG
fdg = FunctionalDependencyGraph()
fdg.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
domain.fdg = fdg
```

**Issue 2**: Similarity Score Always 0
```python
result.total_score == 0.0
```
**Fix**: Ensure FDG has nodes and edges:
```python
fdg.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL))
```

#### Debug Commands

```bash
# Run I_mech validator tests
pytest rese/tests/test_imech/test_validator.py -v

# Run integration tests
pytest rese/tests/test_imech/test_integration.py -v

# Run performance tests
pytest rese/tests/test_imech/test_integration.py::TestPerformance -v
```

#### Validate Isomorphism Detection

```python
from rese.phase2.imech import IMechValidator, Domain, FunctionalDependencyGraph, Node, Edge

validator = IMechValidator()

# Create two isomorphic domains
domain1 = create_test_domain(prefix="a")
domain2 = create_test_domain(prefix="b")

result = validator.compare(domain1, domain2)

print(f"Structural Score: {result.structural_score}")
print(f"Total Score: {result.total_score}")
print(f"Node Mapping: {result.node_mapping}")

# Should detect isomorphism (score > 0.7)
assert result.structural_score > 0.7
```

---

### Ψ₃ Constraint Inversion

**Location**: `rese/phase2/psi3/src/core/constraint_inverter.py`
**Tests**: `rese/phase2/psi3/tests/unit/test_constraint_inverter.py`

#### Common Issues

**Issue 1**: Z3 Solver Not Available
```
ImportError: No module named 'z3'
```
**Fix**:
```bash
pip install z3-solver
```

**Issue 2**: Constraint Creation Errors
```python
TypeError: '>Gt' object is not callable
```
**Fix**: Import correctly:
```python
from core.expression import Gt, Var, Const
from core.constraint import Constraint, ConstraintType, Metadata

constraint = Constraint(
    id=1,
    expr=Gt(Var("x"), Const(5)),  # Not Gt()()
    type=ConstraintType.ARITH,
    vars=frozenset(["x"]),
    metadata=Metadata(source="test")
)
```

**Issue 3**: Reduction Not Working
```python
result.final_size == result.original_size  # Expected reduction
```
**Fix**: Create hierarchical constraints:
```python
# Good (hierarchical, will reduce):
constraints = [
    Constraint(id=1, expr=Gt(Var("x"), Const(0))),
    Constraint(id=2, expr=Gt(Var("x"), Const(5))),
    Constraint(id=3, expr=Gt(Var("x"), Const(10))),
]

# Bad (independent, won't reduce):
constraints = [
    Constraint(id=1, expr=Gt(Var("x"), Const(0))),
    Constraint(id=2, expr=Gt(Var("y"), Const(0))),
    Constraint(id=3, expr=Gt(Var("z"), Const(0))),
]
```

#### Debug Commands

```bash
# Run Ψ₃ tests
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py -v

# Run specific stage
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py::TestSyntacticPreprocessing -v

# Run integration
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py::TestIntegration -v
```

#### Validate Constraint Reduction

```python
from rese.phase2.psi3.src.core.constraint_inverter import ConstraintInverter, PSI3Config
from rese.phase2.psi3.src.core.expression import Gt, Var, Const
from rese.phase2.psi3.src.core.constraint import Constraint, ConstraintType, Metadata
from rese.phase2.psi3.src.solvers.sat_wrapper import SATInterface

# Create hierarchical constraints
constraints = [
    Constraint(id=1, expr=Gt(Var("x"), Const(0)), type=ConstraintType.ARITH,
               vars=frozenset(["x"]), metadata=Metadata(source="test")),
    Constraint(id=2, expr=Gt(Var("x"), Const(5)), type=ConstraintType.ARITH,
               vars=frozenset(["x"]), metadata=Metadata(source="test")),
    Constraint(id=3, expr=Gt(Var("x"), Const(10)), type=ConstraintType.ARITH,
               vars=frozenset(["x"]), metadata=Metadata(source="test")),
]

# Run inverter
config = PSI3Config(mode="fast", verify=False, verbose=True)
inverter = ConstraintInverter(config)
result = inverter.reduce_constraints(constraints, timeout=30.0)

print(f"Original: {result.original_size}")
print(f"Final: {result.final_size}")
print(f"Reduction: {result.reduction_ratio}x")

# Should achieve 6.6x+ reduction
assert result.reduction_ratio >= 6.6
```

---

### Ψ₂ Ontology Mapping

**Location**: `rese/phase2/ontology_mapper.py`
**Tests**: `rese/tests/test_ontology_mapper/`

#### Common Issues

**Issue 1**: Missing Embeddings
```
ValueError: No embeddings found
```
**Fix**: Install sentence-transformers:
```bash
pip install sentence-transformers
```

**Issue 2**: Graph Database Not Available
```
ConnectionError: Cannot connect to Neo4j
```
**Fix**: Use in-memory mode or mock KG:
```python
from rese.phase2.ontology_components.kg_validator import KGValidator

validator = KGValidator(use_in_memory=True)
```

#### Debug Commands

```bash
# Run Ψ₂ tests
pytest rese/tests/test_ontology_mapper/test_ontology_mapper.py -v

# Run integration
pytest rese/tests/test_ontology_mapper/test_integration.py -v
```

---

## Integration Testing

### Phase 1 Integration

```bash
# Run full Phase 1 integration
pytest rese/tests/test_integration/test_phase1_integration.py -v

# Test end-to-end pipeline
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15EndToEnd -v

# Test component integration
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15ComponentIntegration -v

# Test performance
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v
```

### Full Pipeline Integration

```bash
# Run complete integration
pytest rese/tests/test_integration/test_full_pipeline.py -v

# Test all phases together
pytest rese/tests/test_integration/ -v -m integration
```

---

## Performance Validation

### Run Performance Benchmarks

```bash
# Using test runner
python run_rese_tests.py --phase all --coverage

# Or with pytest markers
pytest rese/tests/ -m performance -v

# Phase 1 performance
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v

# I_mech performance
pytest rese/tests/test_imech/test_integration.py::TestPerformance -v
```

### Check Results

Performance targets:
- Φ₁.₅ (1K failures): <10s
- Ψ₃ (reduction): 6.6x
- I_mech (10 nodes): <5s
- I_mech (50 nodes): <30s

---

## Common Debugging Patterns

### Pattern 1: Test Fixture Issues

**Problem**: `Fixture 'xyz' not found`

**Solution**:
```python
# Check conftest.py has the fixture
# rese/tests/conftest.py

@pytest.fixture
def xyz():
    return X()
```

### Pattern 2: Path Issues

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use absolute paths
data_file = project_root / "data" / "test_data.json"
```

### Pattern 3: Import Order Issues

**Problem**: `ImportError: cannot import name 'X' from 'Y'`

**Solution**:
```python
# Ensure correct import order
# 1. Standard library
import sys
from pathlib import Path

# 2. Third-party
import pytest
import numpy as np

# 3. Local (with path setup)
sys.path.insert(0, str(Path(__file__).parent.parent))
from module import X
```

### Pattern 4: Data Type Mismatches

**Problem**: `TypeError: expected type X, got Y`

**Solution**:
```python
# Check expected types
from typing import List, Dict

def process_data(data: List[Dict]) -> None:
    """Process list of dictionaries"""
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")

    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict, got {type(item)}")
```

---

## Quick Fixes

### Fix 1: Reset Test Database

```bash
# Delete test databases
rm -rf rese/tests/test_databases/*.db

# Tests will recreate on next run
```

### Fix 2: Clear Test Cache

```bash
# Clear pytest cache
pytest --cache-clear

# Or manually
rm -rf .pytest_cache
```

### Fix 3: Reinstall Dependencies

```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# Or specific packages
pip install --force-reinstall pytest numpy scipy z3-solver
```

### Fix 4: Update Python Path

```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

---

## Getting Help

### Check Test Logs

```bash
# Run with verbose logging
pytest rese/tests/ -v -s --log-cli-level=DEBUG

# Save output to file
pytest rese/tests/ -v 2>&1 | tee test_output.txt
```

### Run with Debugger

```bash
# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Drop into debugger on error
pytest --pdb --tb=long
```

### Generate Coverage Report

```bash
# Generate HTML coverage
pytest rese/tests/ --cov=rese --cov-report=html

# Open in browser
open htmlcov/index.html  # Mac
start htmlcov/index.html # Windows
xdg-open htmlcov/index.html # Linux
```

---

## Next Steps

1. ✅ Run `python run_rese_tests.py --phase all --verbose`
2. ✅ Review `rese_test_results.json`
3. ✅ Document any failures in bug tracking template
4. ✅ Fix critical bugs first
5. ✅ Re-run tests to verify fixes
6. ✅ Validate performance against targets
7. ✅ Generate final report

---

## Summary

**Files Created**:
1. `RESE_PHASE_DEBUG_REPORT.md` - Comprehensive testing documentation
2. `run_rese_tests.py` - Automated test runner
3. `RESE_BUG_TRACKING_TEMPLATE.md` - Bug tracking and validation
4. `RESE_QUICK_START_DEBUG.md` - This quick start guide

**Test Structure**:
- Phase 1: 150+ tests (Φ₁.₅, Φ₂)
- Phase 2: 150+ tests (I_mech, Ψ₃, Ψ₂)
- Integration: 50+ tests
- Total: 350+ tests

**Status**: 🟡 Ready for testing
**Estimated Runtime**: 10-30 minutes for full suite

---

**Start Testing Now**:
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python run_rese_tests.py --phase all --verbose
```

Good luck! 🚀
=======
# RESE Framework Quick Start Debugging Guide

**Last Updated**: 2025-12-31
**Purpose**: Rapid debugging and validation of Phase 1 & Phase 2 modules

---

## Quick Start (5 Minutes)

### 1. Run All Tests

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Run the comprehensive test suite
python run_rese_tests.py --phase all --verbose

# Or run with pytest directly
pytest rese/tests/phase1/ rese/tests/test_imech/ -v --tb=short
```

### 2. Check Results

Results will be saved to: `rese_test_results.json`

### 3. Read This Guide

Follow the sections below for detailed debugging steps.

---

## Module-Specific Debugging

### Φ₁.₅ Tacit Assumption Miner

**Location**: `rese/phase1/tacit_assumption_miner.py`
**Tests**: `rese/tests/test_phi15.py`

#### Common Issues

**Issue 1**: Import Error
```python
ModuleNotFoundError: No module named 'phase1.tacit_assumption_miner'
```
**Fix**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Issue 2**: Insufficient Data for Clustering
```
DBSCAN requires min_samples > 1
```
**Fix**: Use at least 30 null results for testing

**Issue 3**: Feature Vector Dimension Mismatch
```
ValueError: shapes (N, 5) != (M, 5)
```
**Fix**: Ensure all feature vectors have same dimension (5)

#### Debug Commands

```bash
# Run Φ₁.₅ tests only
pytest rese/tests/test_phi15.py -v

# Run specific test class
pytest rese/tests/test_phi15.py::TestPhi15Engine -v

# Run with verbose output
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_process_null_results -v -s

# Run with debugger
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_process_null_results --pdb
```

#### Validate Performance

```bash
# Run performance test
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v

# Check if <10s target met for 1K failures
```

---

### Φ₂ Metacognitive Debiasing

**Location**: `rese/phase1/cognitive_biases.py`
**Tests**: `rese/tests/phase1/test_cognitive_biases.py`

#### Common Issues

**Issue 1**: Bias Detection Not Triggering
```python
# Expected: detections > 0
# Actual: detections == 0
```
**Fix**: Use strongly biased phrases:
```python
# Good (high bias):
"This will certainly always work perfectly"
"We must maximize performance"

# Bad (low bias):
"The system should maintain accuracy"
"Consider using approximation"
```

**Issue 2**: Missing Debiasing Strategies
```
AttributeError: 'DebiasingStrategy' object has no attribute 'method_name'
```
**Fix**: Ensure all strategy methods are implemented

#### Debug Commands

```bash
# Run Φ₂ tests
pytest rese/tests/phase1/test_cognitive_biases.py -v

# Test specific bias
pytest rese/tests/phase1/test_cognitive_biases.py::TestBiasDetection::test_confirmation_bias_detection -v

# Test debiasing strategies
pytest rese/tests/phase1/test_cognitive_biases.py::TestDebiasingStrategies -v
```

#### Validate Bias Detection

Create test constraints:
```python
from rese.phase1.cognitive_biases import CognitiveBiasDetector
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

detector = CognitiveBiasDetector()

# High bias constraint
biased = Constraint(
    id="biased_1",
    type=ConstraintType.HARD,
    description="This will certainly achieve perfect results",
    formalization="perfect = true",
    source="expert"
)

report = detector.analyze_constraints([biased])
print(f"Detections: {report.total_detections}")
print(f"Bias Score: {report.overall_bias_score}")
```

---

### I_mech Isomorphism Validator

**Location**: `rese/phase2/imech/isomorphism_validator.py`
**Tests**: `rese/tests/test_imech/`

#### Common Issues

**Issue 1**: Domain Creation Error
```python
TypeError: __init__() missing required arguments
```
**Fix**: Ensure all required fields provided:
```python
from rese.phase2.imech import Domain, FunctionalDependencyGraph, Node, Edge

domain = Domain(
    id="test_domain",
    name="Test Domain",
    description="Test description",
    formal_constraints=["c1", "c2"],
    natural_language_constraints=["constraint 1", "constraint 2"]
)

# Must have FDG
fdg = FunctionalDependencyGraph()
fdg.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
domain.fdg = fdg
```

**Issue 2**: Similarity Score Always 0
```python
result.total_score == 0.0
```
**Fix**: Ensure FDG has nodes and edges:
```python
fdg.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL))
```

#### Debug Commands

```bash
# Run I_mech validator tests
pytest rese/tests/test_imech/test_validator.py -v

# Run integration tests
pytest rese/tests/test_imech/test_integration.py -v

# Run performance tests
pytest rese/tests/test_imech/test_integration.py::TestPerformance -v
```

#### Validate Isomorphism Detection

```python
from rese.phase2.imech import IMechValidator, Domain, FunctionalDependencyGraph, Node, Edge

validator = IMechValidator()

# Create two isomorphic domains
domain1 = create_test_domain(prefix="a")
domain2 = create_test_domain(prefix="b")

result = validator.compare(domain1, domain2)

print(f"Structural Score: {result.structural_score}")
print(f"Total Score: {result.total_score}")
print(f"Node Mapping: {result.node_mapping}")

# Should detect isomorphism (score > 0.7)
assert result.structural_score > 0.7
```

---

### Ψ₃ Constraint Inversion

**Location**: `rese/phase2/psi3/src/core/constraint_inverter.py`
**Tests**: `rese/phase2/psi3/tests/unit/test_constraint_inverter.py`

#### Common Issues

**Issue 1**: Z3 Solver Not Available
```
ImportError: No module named 'z3'
```
**Fix**:
```bash
pip install z3-solver
```

**Issue 2**: Constraint Creation Errors
```python
TypeError: '>Gt' object is not callable
```
**Fix**: Import correctly:
```python
from core.expression import Gt, Var, Const
from core.constraint import Constraint, ConstraintType, Metadata

constraint = Constraint(
    id=1,
    expr=Gt(Var("x"), Const(5)),  # Not Gt()()
    type=ConstraintType.ARITH,
    vars=frozenset(["x"]),
    metadata=Metadata(source="test")
)
```

**Issue 3**: Reduction Not Working
```python
result.final_size == result.original_size  # Expected reduction
```
**Fix**: Create hierarchical constraints:
```python
# Good (hierarchical, will reduce):
constraints = [
    Constraint(id=1, expr=Gt(Var("x"), Const(0))),
    Constraint(id=2, expr=Gt(Var("x"), Const(5))),
    Constraint(id=3, expr=Gt(Var("x"), Const(10))),
]

# Bad (independent, won't reduce):
constraints = [
    Constraint(id=1, expr=Gt(Var("x"), Const(0))),
    Constraint(id=2, expr=Gt(Var("y"), Const(0))),
    Constraint(id=3, expr=Gt(Var("z"), Const(0))),
]
```

#### Debug Commands

```bash
# Run Ψ₃ tests
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py -v

# Run specific stage
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py::TestSyntacticPreprocessing -v

# Run integration
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py::TestIntegration -v
```

#### Validate Constraint Reduction

```python
from rese.phase2.psi3.src.core.constraint_inverter import ConstraintInverter, PSI3Config
from rese.phase2.psi3.src.core.expression import Gt, Var, Const
from rese.phase2.psi3.src.core.constraint import Constraint, ConstraintType, Metadata
from rese.phase2.psi3.src.solvers.sat_wrapper import SATInterface

# Create hierarchical constraints
constraints = [
    Constraint(id=1, expr=Gt(Var("x"), Const(0)), type=ConstraintType.ARITH,
               vars=frozenset(["x"]), metadata=Metadata(source="test")),
    Constraint(id=2, expr=Gt(Var("x"), Const(5)), type=ConstraintType.ARITH,
               vars=frozenset(["x"]), metadata=Metadata(source="test")),
    Constraint(id=3, expr=Gt(Var("x"), Const(10)), type=ConstraintType.ARITH,
               vars=frozenset(["x"]), metadata=Metadata(source="test")),
]

# Run inverter
config = PSI3Config(mode="fast", verify=False, verbose=True)
inverter = ConstraintInverter(config)
result = inverter.reduce_constraints(constraints, timeout=30.0)

print(f"Original: {result.original_size}")
print(f"Final: {result.final_size}")
print(f"Reduction: {result.reduction_ratio}x")

# Should achieve 6.6x+ reduction
assert result.reduction_ratio >= 6.6
```

---

### Ψ₂ Ontology Mapping

**Location**: `rese/phase2/ontology_mapper.py`
**Tests**: `rese/tests/test_ontology_mapper/`

#### Common Issues

**Issue 1**: Missing Embeddings
```
ValueError: No embeddings found
```
**Fix**: Install sentence-transformers:
```bash
pip install sentence-transformers
```

**Issue 2**: Graph Database Not Available
```
ConnectionError: Cannot connect to Neo4j
```
**Fix**: Use in-memory mode or mock KG:
```python
from rese.phase2.ontology_components.kg_validator import KGValidator

validator = KGValidator(use_in_memory=True)
```

#### Debug Commands

```bash
# Run Ψ₂ tests
pytest rese/tests/test_ontology_mapper/test_ontology_mapper.py -v

# Run integration
pytest rese/tests/test_ontology_mapper/test_integration.py -v
```

---

## Integration Testing

### Phase 1 Integration

```bash
# Run full Phase 1 integration
pytest rese/tests/test_integration/test_phase1_integration.py -v

# Test end-to-end pipeline
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15EndToEnd -v

# Test component integration
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15ComponentIntegration -v

# Test performance
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v
```

### Full Pipeline Integration

```bash
# Run complete integration
pytest rese/tests/test_integration/test_full_pipeline.py -v

# Test all phases together
pytest rese/tests/test_integration/ -v -m integration
```

---

## Performance Validation

### Run Performance Benchmarks

```bash
# Using test runner
python run_rese_tests.py --phase all --coverage

# Or with pytest markers
pytest rese/tests/ -m performance -v

# Phase 1 performance
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v

# I_mech performance
pytest rese/tests/test_imech/test_integration.py::TestPerformance -v
```

### Check Results

Performance targets:
- Φ₁.₅ (1K failures): <10s
- Ψ₃ (reduction): 6.6x
- I_mech (10 nodes): <5s
- I_mech (50 nodes): <30s

---

## Common Debugging Patterns

### Pattern 1: Test Fixture Issues

**Problem**: `Fixture 'xyz' not found`

**Solution**:
```python
# Check conftest.py has the fixture
# rese/tests/conftest.py

@pytest.fixture
def xyz():
    return X()
```

### Pattern 2: Path Issues

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use absolute paths
data_file = project_root / "data" / "test_data.json"
```

### Pattern 3: Import Order Issues

**Problem**: `ImportError: cannot import name 'X' from 'Y'`

**Solution**:
```python
# Ensure correct import order
# 1. Standard library
import sys
from pathlib import Path

# 2. Third-party
import pytest
import numpy as np

# 3. Local (with path setup)
sys.path.insert(0, str(Path(__file__).parent.parent))
from module import X
```

### Pattern 4: Data Type Mismatches

**Problem**: `TypeError: expected type X, got Y`

**Solution**:
```python
# Check expected types
from typing import List, Dict

def process_data(data: List[Dict]) -> None:
    """Process list of dictionaries"""
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")

    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict, got {type(item)}")
```

---

## Quick Fixes

### Fix 1: Reset Test Database

```bash
# Delete test databases
rm -rf rese/tests/test_databases/*.db

# Tests will recreate on next run
```

### Fix 2: Clear Test Cache

```bash
# Clear pytest cache
pytest --cache-clear

# Or manually
rm -rf .pytest_cache
```

### Fix 3: Reinstall Dependencies

```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# Or specific packages
pip install --force-reinstall pytest numpy scipy z3-solver
```

### Fix 4: Update Python Path

```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

---

## Getting Help

### Check Test Logs

```bash
# Run with verbose logging
pytest rese/tests/ -v -s --log-cli-level=DEBUG

# Save output to file
pytest rese/tests/ -v 2>&1 | tee test_output.txt
```

### Run with Debugger

```bash
# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Drop into debugger on error
pytest --pdb --tb=long
```

### Generate Coverage Report

```bash
# Generate HTML coverage
pytest rese/tests/ --cov=rese --cov-report=html

# Open in browser
open htmlcov/index.html  # Mac
start htmlcov/index.html # Windows
xdg-open htmlcov/index.html # Linux
```

---

## Next Steps

1. ✅ Run `python run_rese_tests.py --phase all --verbose`
2. ✅ Review `rese_test_results.json`
3. ✅ Document any failures in bug tracking template
4. ✅ Fix critical bugs first
5. ✅ Re-run tests to verify fixes
6. ✅ Validate performance against targets
7. ✅ Generate final report

---

## Summary

**Files Created**:
1. `RESE_PHASE_DEBUG_REPORT.md` - Comprehensive testing documentation
2. `run_rese_tests.py` - Automated test runner
3. `RESE_BUG_TRACKING_TEMPLATE.md` - Bug tracking and validation
4. `RESE_QUICK_START_DEBUG.md` - This quick start guide

**Test Structure**:
- Phase 1: 150+ tests (Φ₁.₅, Φ₂)
- Phase 2: 150+ tests (I_mech, Ψ₃, Ψ₂)
- Integration: 50+ tests
- Total: 350+ tests

**Status**: 🟡 Ready for testing
**Estimated Runtime**: 10-30 minutes for full suite

---

**Start Testing Now**:
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python run_rese_tests.py --phase all --verbose
```

Good luck! 🚀
>>>>>>> 1cb9c5e35 (update)
