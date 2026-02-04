#!/bin/bash
# Probe script for RESE Phase II adapter
# Following CLAUDE.md: Law of Runtime Truth - verify before using

set -e

echo "=== RESE Phase II Adapter Probe ==="
echo "Testing Phase II: Isomorphic Mapping functionality..."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "✓ Python available"

# Check if we can import the module
echo ""
echo "Testing module import..."
python -c "
import sys
sys.path.insert(0, '../src')
try:
    from phase2_executor import IsomorphicMappingExecutor, create_executor
    print('✓ Module import successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Module import failed"
    exit 1
fi

# Test basic functionality
echo ""
echo "Testing basic Phase II execution..."
python -c "
import sys
import os
sys.path.insert(0, '../src')

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

    print('✓ Phase II execution successful')
    print(f'  - Source domain: {result.source_domain}')
    print(f'  - Target domains: {len(result.target_domains)}')
    print(f'  - Mappings found: {len(result.mappings_found)}')
    print(f'  - Patterns found: {len(result.cross_domain_patterns)}')
    print(f'  - Execution time: {result.execution_time_ms:.2f}ms')

except Exception as e:
    print(f'✗ Phase II execution failed: {e}')
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
python -c "
import sys
import os
sys.path.insert(0, '../src')

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

    print('✓ Adapter interface successful')
    print(f'  - Result ID: {result[\"result_id\"]}')
    print(f'  - Mapping count: {result[\"summary\"][\"mapping_count\"]}')
    print(f'  - Best I_mech: {result[\"summary\"][\"best_imech_score\"]:.2f}')

except Exception as e:
    print(f'✗ Adapter interface failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Adapter interface failed"
    exit 1
fi

echo ""
echo "=== All Probe Tests Passed ✓ ==="
echo "Phase II adapter is ready for use"
exit 0
