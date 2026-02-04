#!/bin/bash
###############################################################################
# RESE Phase IV: Architecture Assembly - Probe Script
#
# Following CLAUDE.md §4.1: The Probe (Discovery)
#
# This script verifies that Phase IV executor and adapter are functional.
# It tests:
# 1. Schema imports work correctly
# 2. Executor can be instantiated
# 3. Adapter can be instantiated
# 4. Simple assembly operation completes
# 5. Health check endpoint works
#
# Before implementing Phase IV features, this probe must succeed.
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Log functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test result tracker
test_result() {
    if [ $1 -eq 0 ]; then
        log_info "✓ $2"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "✗ $2"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
SCHEMAS_DIR="$SCRIPT_DIR/../../../schemas"

log_info "=== RESE Phase IV Probe Script ==="
log_info "Script directory: $SCRIPT_DIR"
log_info "Source directory: $SRC_DIR"
log_info "Schemas directory: $SCHEMAS_DIR"

###############################################################################
# Test 1: Check directory structure
###############################################################################

log_info "Test 1: Checking directory structure..."

[ -d "$SRC_DIR" ]
test_result $? "Source directory exists"

[ -f "$SRC_DIR/phase4_executor.py" ]
test_result $? "Executor file exists"

[ -f "$SRC_DIR/adapter.py" ]
test_result $? "Adapter file exists"

[ -f "$SCHEMAS_DIR/rese_phase4_schemas.py" ]
test_result $? "Schemas file exists"

###############################################################################
# Test 2: Check Python environment
###############################################################################

log_info "Test 2: Checking Python environment..."

python3 --version &>/dev/null
test_result $? "Python 3 is available"

# Check if we can import schemas (basic syntax check)
log_info "Checking schema syntax..."
python3 -c "
import sys
sys.path.insert(0, '$SCHEMAS_DIR')
try:
    import rese_phase4_schemas
    print('Schema import: OK')
    exit(0)
except Exception as e:
    print(f'Schema import failed: {e}')
    exit(1)
" 2>&1
test_result $? "Schema imports work"

###############################################################################
# Test 3: Test executor instantiation
###############################################################################

log_info "Test 3: Testing executor instantiation..."

python3 -c "
import sys
import os
sys.path.insert(0, '$SCHEMAS_DIR')
sys.path.insert(0, '$SRC_DIR')

# Set minimal environment variables for testing
os.environ['PHASE4_ASSEMBLY_TIMEOUT_MS'] = '25000'
os.environ['PHASE4_VALIDATION_LEVEL'] = 'standard'

try:
    from phase4_executor import ArchitectureAssemblyExecutor
    executor = ArchitectureAssemblyExecutor()
    print(f'Executor created: {type(executor).__name__}')
    exit(0)
except Exception as e:
    print(f'Executor creation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1
test_result $? "Executor can be instantiated"

###############################################################################
# Test 4: Test adapter instantiation
###############################################################################

log_info "Test 4: Testing adapter instantiation..."

python3 -c "
import sys
import os
sys.path.insert(0, '$SCHEMAS_DIR')
sys.path.insert(0, '$SRC_DIR')

os.environ['PHASE4_ASSEMBLY_TIMEOUT_MS'] = '25000'
os.environ['PHASE4_VALIDATION_LEVEL'] = 'standard'

try:
    from adapter import Phase4Adapter
    adapter = Phase4Adapter()
    print(f'Adapter created: {type(adapter).__name__}')

    # Test health check
    health = adapter.health_check()
    print(f'Health check: {health[\"status\"]}')
    exit(0)
except Exception as e:
    print(f'Adapter creation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1
test_result $? "Adapter can be instantiated and health check works"

###############################################################################
# Test 5: Test simple assembly operation
###############################################################################

log_info "Test 5: Testing simple assembly operation..."

python3 -c "
import sys
import os
sys.path.insert(0, '$SCHEMAS_DIR')
sys.path.insert(0, '$SRC_DIR')

os.environ['PHASE4_ASSEMBLY_TIMEOUT_MS'] = '25000'
os.environ['PHASE4_VALIDATION_LEVEL'] = 'basic'

try:
    from adapter import Phase4Adapter

    # Create adapter
    adapter = Phase4Adapter()

    # Create minimal request
    request = {
        'request_id': 'test-request-001',
        'phase1_patterns': [
            {
                'pattern_id': 'test-pattern-1',
                'type': 'structural',
                'description': 'Test pattern from Phase I',
                'confidence': 0.8,
            }
        ],
        'phase2_patterns': [
            {
                'pattern_id': 'test-pattern-2',
                'type': 'structural',
                'description': 'Test pattern from Phase II',
                'confidence': 0.75,
            }
        ],
    }

    # Execute assembly
    response = adapter.assemble_architecture(request)

    # Verify response
    assert 'assembly' in response, 'Response missing assembly'
    assert 'status' in response, 'Response missing status'
    assert response['status'] == 'success', f'Expected success, got {response[\"status\"]}'

    assembly = response['assembly']
    assert 'assembly_id' in assembly, 'Assembly missing ID'
    assert 'synthesized_knowledge' in assembly, 'Assembly missing knowledge'

    print(f'Assembly ID: {assembly[\"assembly_id\"]}')
    print(f'Knowledge ID: {assembly[\"synthesized_knowledge\"][\"knowledge_id\"]}')
    print(f'Status: {response[\"status\"]}')

    exit(0)
except Exception as e:
    print(f'Assembly operation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1
test_result $? "Simple assembly operation works"

###############################################################################
# Test 6: Test schema validation
###############################################################################

log_info "Test 6: Testing schema validation..."

python3 -c "
import sys
sys.path.insert(0, '$SCHEMAS_DIR')

try:
    from rese_phase4_schemas import (
        ArchitectureAssembly,
        ParadigmShift,
        SynthesizedKnowledge,
        Phase4Config,
        AssemblyStatus,
        ParadigmShiftType,
    )

    # Test ParadigmShift creation
    shift = ParadigmShift(
        description='Test paradigm shift',
        shift_type=ParadigmShiftType.STRUCTURAL,
        confidence=0.85,
    )
    assert shift.shift_id is not None
    assert shift.confidence == 0.85

    # Test SynthesizedKnowledge creation
    knowledge = SynthesizedKnowledge(
        description='Test synthesized knowledge',
        confidence=0.9,
    )
    assert knowledge.knowledge_id is not None

    # Test ArchitectureAssembly creation
    assembly = ArchitectureAssembly(
        synthesized_knowledge=knowledge,
        paradigm_shifts=[shift],
        confidence=0.88,
        status=AssemblyStatus.VALIDATED,
    )
    assert assembly.assembly_id is not None
    assert assembly.status == AssemblyStatus.VALIDATED

    # Test serialization
    assembly_dict = assembly.to_dict()
    assert 'assembly_id' in assembly_dict

    # Test deserialization
    assembly2 = ArchitectureAssembly.from_dict(assembly_dict)
    assert assembly2.assembly_id == assembly.assembly_id

    print('Schema validation: OK')
    exit(0)
except Exception as e:
    print(f'Schema validation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1
test_result $? "Schema validation works"

###############################################################################
# Summary
###############################################################################

echo ""
log_info "=== Probe Summary ==="
log_info "Tests Passed: $TESTS_PASSED"
log_info "Tests Failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    log_info "✓ All probes passed! Phase IV is ready for use."
    exit 0
else
    log_error "✗ Some probes failed. Phase IV needs fixes."
    exit 1
fi
