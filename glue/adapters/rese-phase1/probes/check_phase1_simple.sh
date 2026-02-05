#!/bin/bash
# Simple Phase I probe - just check imports work
set -e

echo "Testing Phase I imports..."

# Change to Frontend root
cd "$(dirname "$0")/../../../.." || exit 1

# Set Python command
PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"

echo "Current directory: $(pwd)"
echo "Python: $PYTHON_CMD"

# Test executor import
echo ""
echo "Test 1: Import executor..."
if $PYTHON_CMD -c "import sys; sys.path.insert(0, 'glue/adapters/rese-phase1/src'); from phase1_executor import EpistemicAuditExecutor; print('  PASS: Executor imported')"; then
    echo "✓ Executor import: PASS"
else
    echo "✗ Executor import: FAIL"
    exit 1
fi

# Test adapter import
echo ""
echo "Test 2: Import adapter..."
if $PYTHON_CMD -c "import sys; sys.path.insert(0, 'glue/adapters/rese-phase1/src'); from phase1_adapter import Phase1Adapter; print('  PASS: Adapter imported')"; then
    echo "✓ Adapter import: PASS"
else
    echo "✗ Adapter import: FAIL"
    exit 1
fi

# Test config loading
echo ""
echo "Test 3: Load config..."
if $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import Phase1Config
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
config = Phase1Config.from_env()
print('  PASS: Config loaded')
"; then
    echo "✓ Config loading: PASS"
else
    echo "✗ Config loading: FAIL"
    exit 1
fi

# Test executor instantiation
echo ""
echo "Test 4: Create executor..."
if $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import EpistemicAuditExecutor, Phase1Config
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '5'
config = Phase1Config.from_env()
executor = EpistemicAuditExecutor(config=config)
print('  PASS: Executor created')
"; then
    echo "✓ Executor creation: PASS"
else
    echo "✗ Executor creation: FAIL"
    exit 1
fi

echo ""
echo "========================================="
echo "All Phase I probe tests PASSED!"
echo "========================================="
exit 0
