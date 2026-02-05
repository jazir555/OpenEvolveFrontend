#!/bin/bash
# ============================================================================
# RESE Phase IV Output Probe
# ============================================================================
#
# Runtime verification script for Phase IV output generation components.
#
# Following CLAUDE.md Law of Runtime Truth:
# "You generally do not trust the documentation. You trust execution."
#
# This script probes:
# 1. Directory structure exists
# 2. Schema imports work
# 3. OutputGenerator can be instantiated
# 4. PredictiveValidator can be instantiated
# 5. ResultVerifier can be instantiated
# 6. JSON output generation works
# 7. Markdown output generation works
# 8. Predictive validation works
# 9. Result verification works
# 10. Full integration pipeline works
#
# Usage:
#   cd glue/adapters/rese-phase4
#   bash probes/check_phase4_output.sh
#
# Author: RESE Team
# Created: 2026-02-04
# Phase: IV - Architectural Synthesis and Validation
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for tests
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Helper functions
pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

info() {
    echo -e "${YELLOW}INFO${NC}: $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCHEMAS_DIR="$(cd "$PROJ_ROOT/../.." && pwd)/glue/schemas"

echo "============================================================================"
echo "RESE Phase IV Output Probe"
echo "============================================================================"
echo ""
info "Project Root: $PROJ_ROOT"
info "Phase IV Directory: $SCRIPT_DIR/.."
info "Schemas Directory: $SCHEMAS_DIR"
echo ""

# ============================================================================
# TEST 1: Directory Structure
# ============================================================================

echo "Test 1: Checking directory structure..."

test_dir() {
    local dir="$1"
    if [ -d "$dir" ]; then
        pass "Directory exists: $dir"
        return 0
    else
        fail "Directory missing: $dir"
        return 1
    fi
}

test_dir "$SCRIPT_DIR/../src"
test_dir "$SCRIPT_DIR/../tests"
test_dir "$SCHEMAS_DIR"
test_dir "$PROJ_ROOT/../lib"

echo ""

# ============================================================================
# TEST 2: Source Files Exist
# ============================================================================

echo "Test 2: Checking source files..."

test_file() {
    local file="$1"
    if [ -f "$file" ]; then
        pass "File exists: $file"
        return 0
    else
        fail "File missing: $file"
        return 1
    fi
}

test_file "$SCRIPT_DIR/../src/output_generator.py"
test_file "$SCRIPT_DIR/../src/predictive_validator.py"
test_file "$SCRIPT_DIR/../src/result_verifier.py"
test_file "$SCRIPT_DIR/../src/phase4_executor.py"
test_file "$SCRIPT_DIR/../src/adapter.py"

echo ""

# ============================================================================
# TEST 3: Test Files Exist
# ============================================================================

echo "Test 3: Checking test files..."

test_file "$SCRIPT_DIR/../tests/test_output_generator.py"
test_file "$SCRIPT_DIR/../tests/test_predictive_validator.py"
test_file "$SCRIPT_DIR/../tests/test_phase4_integration.py"

echo ""

# ============================================================================
# TEST 4: Schema Imports
# ============================================================================

echo "Test 4: Testing schema imports..."

cd "$SCRIPT_DIR/../src"

python3 -c "
import sys
sys.path.insert(0, '$SCHEMAS_DIR')
from rese_phase4_schemas import (
    ArchitectureAssembly,
    ParadigmShift,
    SynthesizedKnowledge,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
    ParadigmShiftType,
    IntegrationStrategy,
)
print('✓ Schema imports successful')
" 2>&1

if [ $? -eq 0 ]; then
    pass "Schema imports"
else
    fail "Schema imports"
fi

echo ""

# ============================================================================
# TEST 5: OutputGenerator Instantiation
# ============================================================================

echo "Test 5: Testing OutputGenerator instantiation..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from output_generator import OutputGenerator, OutputFormat, StructuredLogger
from rese_phase4_schemas import Phase4Config, ValidationLevel

config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
    min_confidence_threshold=0.7,
    correlation_id='probe-test-123',
)

generator = OutputGenerator(config)
print('✓ OutputGenerator instantiated successfully')
print(f'  - Config: {config.to_dict()}')
print(f'  - Logger: {type(generator.logger).__name__}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "OutputGenerator instantiation"
else
    fail "OutputGenerator instantiation"
fi

echo ""

# ============================================================================
# TEST 6: PredictiveValidator Instantiation
# ============================================================================

echo "Test 6: Testing PredictiveValidator instantiation..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from predictive_validator import PredictiveValidator, StatisticalTest
from rese_phase4_schemas import Phase4Config, ValidationLevel

config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
    min_confidence_threshold=0.7,
    correlation_id='probe-test-123',
)

validator = PredictiveValidator(config, test_type=StatisticalTest.WILCOXON)
print('✓ PredictiveValidator instantiated successfully')
print(f'  - Test type: {validator.test_type.value}')
print(f'  - Significance level: {validator.significance_level}')
print(f'  - Min effect size: {validator.min_effect_size}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "PredictiveValidator instantiation"
else
    fail "PredictiveValidator instantiation"
fi

echo ""

# ============================================================================
# TEST 7: ResultVerifier Instantiation
# ============================================================================

echo "Test 7: Testing ResultVerifier instantiation..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from result_verifier import ResultVerifier
from rese_phase4_schemas import Phase4Config, ValidationLevel

config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
    min_confidence_threshold=0.7,
    correlation_id='probe-test-123',
)

verifier = ResultVerifier(config)
print('✓ ResultVerifier instantiated successfully')
print(f'  - Number of checks: {len(verifier.checks)}')
for check in verifier.checks:
    print(f'    - {check.__class__.__name__}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "ResultVerifier instantiation"
else
    fail "ResultVerifier instantiation"
fi

echo ""

# ============================================================================
# TEST 8: JSON Output Generation
# ============================================================================

echo "Test 8: Testing JSON output generation..."

python3 -c "
import sys
import json
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from output_generator import OutputGenerator, OutputFormat
from rese_phase4_schemas import (
    ArchitectureAssembly,
    SynthesizedKnowledge,
    ParadigmShift,
    ParadigmShiftType,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
)
from datetime import datetime, timezone

# Create sample assembly
config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
    min_confidence_threshold=0.7,
)

shift = ParadigmShift(
    shift_type=ParadigmShiftType.STRUCTURAL,
    description='Test structural shift',
    confidence=0.85,
    validation_status='validated',
)

knowledge = SynthesizedKnowledge(
    knowledge_type='test',
    paradigm_shifts=[shift],
    confidence=0.82,
    completeness=0.9,
    consistency=0.88,
)

assembly = ArchitectureAssembly(
    synthesized_knowledge=knowledge,
    paradigm_shifts=[shift],
    validation_results=[
        {'validation_type': 'completeness', 'passed': True},
        {'validation_type': 'aci_reduction', 'passed': True, 'aci_reduction': 0.35},
    ],
    final_architecture={'architecture_id': 'test-arch-1'},
    aci_reduction_achieved=0.35,
    confidence=0.82,
    validation_level=ValidationLevel.STANDARD,
    status=AssemblyStatus.VALIDATED,
)

# Generate JSON output
generator = OutputGenerator(config)
result = generator.generate(assembly, OutputFormat.JSON)

# Validate output
assert 'formatted_output' in result
assert 'metrics' in result
assert 'validation_summary' in result
assert 'predictions' in result
assert result['formatted_output']['format'] == 'json'
assert result['metrics']['aci_reduction_achieved'] == 0.35
assert result['metrics']['overall_confidence'] == 0.82

print('✓ JSON output generation successful')
print(f'  - ACI reduction: {result[\"metrics\"][\"aci_reduction_achieved\"]:.2%}')
print(f'  - Overall confidence: {result[\"metrics\"][\"overall_confidence\"]:.2%}')
print(f'  - Validation passed: {result[\"metrics\"][\"validation_passed\"]}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "JSON output generation"
else
    fail "JSON output generation"
fi

echo ""

# ============================================================================
# TEST 9: Predictive Validation
# ============================================================================

echo "Test 9: Testing predictive validation..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from predictive_validator import PredictiveValidator, StatisticalTest
from rese_phase4_schemas import (
    ArchitectureAssembly,
    SynthesizedKnowledge,
    ParadigmShift,
    ParadigmShiftType,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
)

# Create sample assembly
config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
)

shift = ParadigmShift(
    shift_type=ParadigmShiftType.STRUCTURAL,
    description='Test shift',
    confidence=0.85,
    validation_status='validated',
)

knowledge = SynthesizedKnowledge(
    knowledge_type='test',
    paradigm_shifts=[shift],
    confidence=0.82,
    completeness=0.9,
    consistency=0.88,
)

assembly = ArchitectureAssembly(
    synthesized_knowledge=knowledge,
    paradigm_shifts=[shift],
    aci_reduction_achieved=0.35,
    confidence=0.82,
    validation_level=ValidationLevel.STANDARD,
    status=AssemblyStatus.VALIDATED,
)

# Run predictive validation
validator = PredictiveValidator(config, test_type=StatisticalTest.WILCOXON)
incumbent_aci = [0.85, 0.82, 0.88, 0.90, 0.87, 0.83, 0.86, 0.89, 0.84, 0.88]
new_aci = [0.55, 0.52, 0.58, 0.50, 0.56, 0.53, 0.54, 0.51, 0.57, 0.52]

result = validator.validate(assembly, incumbent_aci, new_aci)

# Validate result
assert result.is_valid is True
assert result.aci_reduction > 0
assert result.incumbent_aci > result.new_aci
assert result.statistical_significance['is_significant'] is True

print('✓ Predictive validation successful')
print(f'  - Valid: {result.is_valid}')
print(f'  - ACI reduction: {result.aci_reduction:.2%}')
print(f'  - Effect size: {result.effect_size:.2f}')
print(f'  - Significant: {result.statistical_significance[\"is_significant\"]}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "Predictive validation"
else
    fail "Predictive validation"
fi

echo ""

# ============================================================================
# TEST 10: Result Verification
# ============================================================================

echo "Test 10: Testing result verification..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from result_verifier import ResultVerifier
from rese_phase4_schemas import (
    ArchitectureAssembly,
    SynthesizedKnowledge,
    ParadigmShift,
    ParadigmShiftType,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
)

# Create sample assembly
config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
)

shift = ParadigmShift(
    shift_type=ParadigmShiftType.STRUCTURAL,
    description='Test shift',
    confidence=0.85,
    validation_status='validated',
)

knowledge = SynthesizedKnowledge(
    knowledge_type='test',
    paradigm_shifts=[shift],
    confidence=0.82,
    completeness=0.9,
    consistency=0.88,
)

assembly = ArchitectureAssembly(
    synthesized_knowledge=knowledge,
    paradigm_shifts=[shift],
    validation_results=[
        {'validation_type': 'completeness', 'passed': True},
        {'validation_type': 'aci_reduction', 'passed': True, 'aci_reduction': 0.35},
    ],
    final_architecture={'architecture_id': 'test-arch-1'},
    aci_reduction_achieved=0.35,
    confidence=0.82,
    validation_level=ValidationLevel.STANDARD,
    status=AssemblyStatus.VALIDATED,
)

# Run verification
verifier = ResultVerifier(config)
result = verifier.verify(assembly)

# Validate result
assert result.verification_id is not None
assert result.is_valid is True
assert len(result.results) > 0
assert result.checks_passed > 0

print('✓ Result verification successful')
print(f'  - Valid: {result.is_valid}')
print(f'  - Checks passed: {result.checks_passed}')
print(f'  - Checks failed: {result.checks_failed}')
print(f'  - Total checks: {result.checks_passed + result.checks_failed}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "Result verification"
else
    fail "Result verification"
fi

echo ""

# ============================================================================
# TEST 11: Full Integration Pipeline
# ============================================================================

echo "Test 11: Testing full integration pipeline..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from phase4_executor import ArchitectureAssemblyExecutor
from output_generator import OutputGenerator, OutputFormat
from predictive_validator import PredictiveValidator, StatisticalTest
from result_verifier import ResultVerifier
from rese_phase4_schemas import Phase4Config, ValidationLevel

# Create configuration
config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
    min_confidence_threshold=0.7,
)

# Create executor
executor = ArchitectureAssemblyExecutor(config)

# Create sample phase results
phase1_result = {
    'audit_id': 'audit-001',
    'constraints': [{'constraint_id': 'c1', 'type': 'equation'}],
    'contradictions': [],
    'confidence': 0.85,
    'validation_status': 'validated',
}

phase2_result = {
    'mapping_id': 'map-001',
    'isomorphisms': [],
    'confidence': 0.78,
}

phase3_result = {
    'refinement_id': 'ref-001',
    'aci_reduction': 0.35,
    'validated_hypotheses': [],
    'confidence': 0.82,
}

phase1_patterns = [
    {
        'pattern_id': 'p1-1',
        'type': 'structural',
        'description': 'Structural pattern',
        'confidence': 0.85,
    }
]

# Execute assembly
assembly = executor.execute(
    phase1_result=phase1_result,
    phase2_result=phase2_result,
    phase3_result=phase3_result,
    phase1_patterns=phase1_patterns,
    phase2_patterns=[],
    phase3_patterns=[],
)

print('✓ Assembly execution successful')

# Generate output
output_gen = OutputGenerator(config)
output = output_gen.generate(assembly, OutputFormat.JSON)
print('✓ Output generation successful')

# Validate predictions
validator = PredictiveValidator(config)
incumbent_aci = [0.85, 0.82, 0.88, 0.90, 0.87]
new_aci = [0.55, 0.52, 0.58, 0.50, 0.56]
pred_result = validator.validate(assembly, incumbent_aci, new_aci)
print('✓ Predictive validation successful')

# Verify results
verifier = ResultVerifier(config)
verify_result = verifier.verify(assembly)
print('✓ Result verification successful')

print(f'  - Assembly ID: {assembly.assembly_id}')
print(f'  - Assembly status: {assembly.status.value}')
print(f'  - ACI reduction: {assembly.aci_reduction_achieved:.2%}')
print(f'  - Overall confidence: {assembly.confidence:.2%}')
print(f'  - Paradigm shifts: {len(assembly.paradigm_shifts)}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "Full integration pipeline"
else
    fail "Full integration pipeline"
fi

echo ""

# ============================================================================
# TEST 12: Health Check
# ============================================================================

echo "Test 12: Testing health check..."

python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '$PROJ_ROOT/schemas')

from adapter import Phase4Adapter
from rese_phase4_schemas import Phase4Config, ValidationLevel

config = Phase4Config(
    assembly_timeout_ms=25000,
    validation_level=ValidationLevel.STANDARD,
)

adapter = Phase4Adapter(config)
health = adapter.health_check()

assert health['status'] == 'healthy'
assert health['circuit_breaker_state'] == 'closed'
assert 'timestamp' in health

print('✓ Health check successful')
print(f'  - Status: {health[\"status\"]}')
print(f'  - Circuit breaker: {health[\"circuit_breaker_state\"]}')
print(f'  - Failure count: {health[\"failure_count\"]}')
" 2>&1

if [ $? -eq 0 ]; then
    pass "Health check"
else
    fail "Health check"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"
echo ""
echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo ""
    echo "Phase IV Output Generation is ready for use."
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failures above."
    exit 1
fi
