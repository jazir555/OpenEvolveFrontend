#!/bin/bash
# Probe script for RESE Phase II adapter
# Following CLAUDE.md: Law of Runtime Truth - verify before using

set -e

# Change to Frontend root directory
cd "$(dirname "$0")/../.." || exit 1

echo "=== RESE Phase II Adapter Probe ==="
echo "Testing Phase II: Isomorphic Mapping functionality..."
echo ""

# Detect Python command
PYTHON_CMD=""
if [ -f "/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe" ]; then
    PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"
elif command -v python3 &> /dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null 2>&1; then
    PYTHON_CMD="python"
fi

# Check if Python is available
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "PASS: Python available"

# Check if we can import the module
echo ""
echo "Testing module import..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, 'glue/adapters/rese-phase2/src')
sys.path.insert(0, 'glue/schemas')
try:
    from phase2_executor import IsomorphicMappingExecutor, create_executor
    print('PASS: Module import successful')
except Exception as e:
    print(f'FAIL: Import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Module import failed"
    exit 1
fi

# Test basic functionality
echo ""
echo "Testing basic Phase II execution..."
$PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, 'glue/adapters/rese-phase2/src')
sys.path.insert(0, 'glue/schemas')

# Set required env vars
os.environ['PHASE2_MAX_TARGET_DOMAINS'] = '10'
os.environ['PHASE2_IMECH_THRESHOLD'] = '0.7'
os.environ['PHASE2_PATTERN_THRESHOLD'] = '0.6'
os.environ['PHASE2_TIMEOUT_MS'] = '20000'
os.environ['PHASE2_MAX_MAPPINGS'] = '50'
os.environ['PHASE2_ENABLE_CONSTRAINT_INVERSION'] = 'true'
os.environ['PHASE2_SEARCH_DEPTH'] = '5'

try:
    from phase2_executor import create_executor
    from rese_schemas import Phase2Config

    # Create executor
    executor = create_executor()

    # Execute Phase II
    result = executor.execute_phase2(
        source_domain='physics',
        problem_description='Energy conservation problem',
        target_domains=['biology', 'economics'],
        constraints=['energy is conserved']
    )

    # Verify result
    assert result is not None, 'Result is None'
    assert result.source_domain == 'physics', 'Source domain mismatch'
    assert len(result.target_domains) == 2, 'Target domains count mismatch'

    print('PASS: Phase II execution successful')
    print(f'  - Source domain: {result.source_domain}')
    print(f'  - Target domains: {len(result.target_domains)}')
    print(f'  - Mappings found: {len(result.mappings_found)}')
    print(f'  - Patterns found: {len(result.cross_domain_patterns)}')
    print(f'  - Execution time: {result.execution_time_ms:.2f}ms')

except Exception as e:
    print(f'FAIL: Phase II execution failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Phase II execution failed"
    exit 1
fi

# Test adapter interface
echo ""
echo "Testing adapter interface..."
$PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, 'glue/adapters/rese-phase2/src')

# Set required env vars
os.environ['PHASE2_MAX_TARGET_DOMAINS'] = '10'
os.environ['PHASE2_IMECH_THRESHOLD'] = '0.7'
os.environ['PHASE2_PATTERN_THRESHOLD'] = '0.6'
os.environ['PHASE2_TIMEOUT_MS'] = '20000'
os.environ['PHASE2_MAX_MAPPINGS'] = '50'
os.environ['PHASE2_ENABLE_CONSTRAINT_INVERSION'] = 'true'
os.environ['PHASE2_SEARCH_DEPTH'] = '5'

try:
    from phase2_adapter import Phase2Adapter

    # Create adapter
    adapter = Phase2Adapter()

    # Test request
    request = {
        'source_domain': 'computer_science',
        'problem_description': 'Algorithm optimization problem',
        'target_domains': ['physics', 'biology']
    }

    # Execute
    result = adapter.execute_phase2(request)

    # Verify result structure
    assert 'result_id' in result, 'Missing result_id'
    assert 'source_domain' in result, 'Missing source_domain'
    assert 'mappings' in result, 'Missing mappings'
    assert 'summary' in result, 'Missing summary'

    print('PASS: Adapter interface successful')
    print(f'  - Result ID: {result[\"result_id\"]}')
    print(f'  - Mapping count: {result[\"summary\"][\"mapping_count\"]}')
    print(f'  - Best I_mech: {result[\"summary\"][\"best_imech_score\"]:.2f}')

except Exception as e:
    print(f'FAIL: Adapter interface failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Adapter interface failed"
    exit 1
fi

echo ""
echo "=== All Probe Tests Passed ==="
echo "Phase II adapter is ready for use"
exit 0
