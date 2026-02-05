#!/bin/bash
# PROBE: check_dee.sh
# Purpose: Verify RESE Deep Exploration Engine is functional
# Part of CLAUDE.md Phase 1: The Probe (Discovery)

set -e

# Python command configuration
PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"

echo "==================================="
echo "RESE DEE Probe Script"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if DEE module exists
echo "Test 1: Checking if DEE module exists..."
if [ -f "glue/lib/rese_dee.py" ]; then
    echo -e "${GREEN}✓${NC} DEE module found"
else
    echo -e "${RED}✗${NC} DEE module not found"
    exit 1
fi

# Test 2: Check if schemas exist
echo ""
echo "Test 2: Checking if RESE schemas exist..."
if [ -f "glue/schemas/rese_schemas.py" ]; then
    echo -e "${GREEN}✓${NC} RESE schemas found"
else
    echo -e "${RED}✗${NC} RESE schemas not found"
    exit 1
fi

# Test 3: Check if required dependencies are available
echo ""
echo "Test 3: Checking dependencies..."
PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
try:
    import dataclasses
    import datetime
    import typing
    import uuid
    import json
    import logging
    print('All standard library dependencies available')
except ImportError as e:
    print(f'Missing dependency: {e}')
    sys.exit(1)
" || exit 1
echo -e "${GREEN}✓${NC} Dependencies available"

# Test 4: Import DEE module
echo ""
echo "Test 4: Importing DEE module..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from glue.lib.rese_dee import DeepExplorationEngine, HypothesisGenerator, PatternRecognizer, MCTSExplainer
print('DEE module imported successfully')
" || exit 1
echo -e "${GREEN}✓${NC} DEE module imported"

# Test 5: Import schemas
echo ""
echo "Test 5: Importing RESE schemas..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from glue.schemas.rese_schemas import Hypothesis, SearchTreeNode, Pattern, MCTSSearchResult, ExplorationConfig
print('RESE schemas imported successfully')
" || exit 1
echo -e "${GREEN}✓${NC} Schemas imported"

# Test 6: Create exploration configuration from environment
echo ""
echo "Test 6: Testing configuration from environment..."
export EXPLORATION_DEPTH=10
export MCTS_ITERATIONS=100
export EXPLORATION_TIMEOUT_MS=5000
$PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, '.')
from glue.schemas.rese_schemas import ExplorationConfig
config = ExplorationConfig.from_env()
assert config.exploration_depth == 10
assert config.mcts_iterations == 100
assert config.timeout_ms == 5000
print('Configuration loaded from environment')
" || exit 1
echo -e "${GREEN}✓${NC} Configuration from environment working"

# Test 7: Create a hypothesis
echo ""
echo "Test 7: Testing hypothesis creation..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from glue.schemas.rese_schemas import Hypothesis, HypothesisStatus
h = Hypothesis(
    statement='Test hypothesis',
    type='causal',
    domain='test_domain',
    confidence=0.7
)
assert h.hypothesis_id is not None
assert h.statement == 'Test hypothesis'
assert h.confidence == 0.7
print('Hypothesis created successfully')
" || exit 1
echo -e "${GREEN}✓${NC} Hypothesis creation working"

# Test 8: Test hypothesis idempotency (Law of Idempotency)
echo ""
echo "Test 8: Testing hypothesis idempotency..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from glue.schemas.rese_schemas import Hypothesis
h = Hypothesis(
    hypothesis_id='test-id-123',
    statement='Test hypothesis'
)
# Add evidence twice
h.update_evidence({'source': 'test1'}, is_supporting=True)
h.update_evidence({'source': 'test1'}, is_supporting=True)
# Should only have one evidence item (deduplicated)
assert len(h.evidence) == 1
print('Hypothesis idempotency working')
" || exit 1
echo -e "${GREEN}✓${NC} Hypothesis idempotency working"

# Test 9: Test MCTS node creation and UCB calculation
echo ""
echo "Test 9: Testing MCTS node operations..."
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from glue.schemas.rese_schemas import SearchTreeNode
node = SearchTreeNode(
    node_id='test-node',
    visit_count=10,
    value=5.0
)
ucb = node.calculate_ucb(total_visits=20, exploration_constant=1.414)
assert ucb > 0
print('MCTS node operations working')
" || exit 1
echo -e "${GREEN}✓${NC} MCTS node operations working"

# Test 10: Test simple exploration (minimal configuration)
echo ""
echo "Test 10: Testing simple DEE exploration..."
export EXPLORATION_DEPTH=3
export MCTS_ITERATIONS=10
export EXPLORATION_TIMEOUT_MS=10000
$PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, '.')
from glue.lib.rese_dee import DeepExplorationEngine
from glue.schemas.rese_schemas import ExplorationConfig
config = ExplorationConfig.from_env()
dee = DeepExplorationEngine(config=config)
print('Deep Exploration Engine initialized')
print(f'Config: depth={config.exploration_depth}, iterations={config.mcts_iterations}')
" || exit 1
echo -e "${GREEN}✓${NC} DEE initialization working"

# Summary
echo ""
echo "==================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "==================================="
echo ""
echo "DEE is ready for use."
echo "Next steps:"
echo "  1. Run full exploration tests"
echo "  2. Test with real problem statements"
echo "  3. Integrate with RESE pipeline"
