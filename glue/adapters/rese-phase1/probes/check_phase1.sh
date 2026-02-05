#!/bin/bash
# check_phase1.sh
# RESE Phase I Probe - Following "Law of Runtime Truth"
# We trust execution, not documentation.
#
# This probe validates that Phase I: Epistemic Audit can execute.
# Exit code: 0 = success, 1 = failure
# Output: Structured JSON to stdout

set -e

# Detect Python command
PYTHON_CMD=""
if [ -f "/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe" ]; then
    PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"
elif command -v python3 &> /dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null 2>&1; then
    PYTHON_CMD="py"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "{\"error\": \"Python not found\"}"
    exit 1
fi

# Configuration - MUST be run from FRONTEND_ROOT
cd "$(dirname "$0")/../../../.." || exit 1
FRONTEND_ROOT="$(pwd)"
PHASE1_DIR="${PHASE1_DIR:-${FRONTEND_ROOT}/glue/adapters/rese-phase1}"
RESE_ROOT="${RESE_ROOT_DIR:-${FRONTEND_ROOT}/rese}"

# Generate correlation ID
CORRELATION_ID=$($PYTHON_CMD -c "import uuid; print(str(uuid.uuid4()))")
TIMESTAMP=$($PYTHON_CMD -c "from datetime import datetime; print(datetime.utcnow().isoformat() + 'Z')")

# Initialize JSON output
echo "{"
echo "  \"probe_name\": \"check_phase1\","
echo "  \"probe_type\": \"phase1_verification\","
echo "  \"correlation_id\": \"$CORRELATION_ID\","
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"source_service\": \"phase1_probe\","
echo "  \"target_service\": \"rese_phase1\","
echo "  \"phase1_directory\": \"$PHASE1_DIR\","
echo "  \"checks\": {"

EXIT_CODE=0
DELIMITER=""

# Function to perform a check
perform_check() {
    local check_name="$1"
    local check_desc="$2"
    local check_command="$3"

    echo "$DELIMITER"
    echo "    \"$check_name\": {"

    if eval "$check_command" > /dev/null 2>&1; then
        echo "      \"status\": \"PASS\","
        echo "      \"description\": \"$check_desc\","
        echo "      \"message\": \"$check_name successful\""
        echo -n "    }"
        DELIMITER=","
    else
        echo "      \"status\": \"FAIL\","
        echo "      \"description\": \"$check_desc\","
        echo "      \"message\": \"$check_name failed\""
        echo -n "    }"
        DELIMITER=","
        EXIT_CODE=1
    fi
}

# Check 1: Phase I directory exists
perform_check "directory_exists" \
    "Phase I adapter directory exists" \
    "[ -d \"$PHASE1_DIR\" ] && cd \"$FRONTEND_ROOT\" "

# Check 2: Executor module exists
perform_check "executor_module_exists" \
    "Phase I executor module exists" \
    "[ -f \"$PHASE1_DIR/src/phase1_executor.py\" ]"

# Check 3: Adapter module exists
perform_check "adapter_module_exists" \
    "Phase I adapter module exists" \
    "[ -f \"$PHASE1_DIR/src/phase1_adapter.py\" ]"

# Check 4: Python can import executor
perform_check "executor_importable" \
    "Executor module can be imported" \
    "$PYTHON_CMD -c \"import sys; sys.path.insert(0, 'glue/adapters/rese-phase1/src'); from phase1_executor import EpistemicAuditExecutor\""

# Check 5: Python can import adapter
perform_check "adapter_importable" \
    "Adapter module can be imported" \
    "$PYTHON_CMD -c \"import sys; sys.path.insert(0, 'glue/adapters/rese-phase1/src'); from phase1_adapter import Phase1Adapter\""

# Check 6: Configuration can be loaded
perform_check "config_loadable" \
    "Phase I configuration can be loaded from environment" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import Phase1Config

# Set test environment variables
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'

config = Phase1Config.from_env()
assert config.TIMEOUT_MS == 15000
assert config.MAX_ASSUMPTIONS == 100
\""

# Check 7: Executor can be instantiated
perform_check "executor_instantiable" \
    "Executor can be instantiated" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import EpistemicAuditExecutor, Phase1Config

# Set test environment variables
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '5'

config = Phase1Config.from_env()
executor = EpistemicAuditExecutor(config=config)
assert executor is not None
\""

# Check 8: TacitAssumption dataclass works
perform_check "tacit_assumption_works" \
    "TacitAssumption dataclass can be created and serialized" \
    "$PYTHON_CMD -c \"
import sys
import json
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import TacitAssumption

assumption = TacitAssumption(
    id='test-id',
    description='Test assumption',
    source_pattern='Test pattern',
    confidence_score=0.8,
    supporting_evidence_count=10,
)

data = assumption.to_dict()
assert data['id'] == 'test-id'
assert data['confidence_score'] == 0.8

reconstructed = TacitAssumption.from_dict(data)
assert reconstructed.id == assumption.id
\""

# Check 9: ConstraintHardener works
perform_check "constraint_hardener_works" \
    "ConstraintHardener can extract constraints from problem description" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import ConstraintHardener, Phase1Config, StructuredLogger

config = Phase1Config.from_env()
logger = StructuredLogger('test')
hardener = ConstraintHardener(config=config, logger=logger)

constraints = hardener.harden_constraints(
    problem_description='This problem is impossible to solve due to limited resources.',
    correlation_id='test-correlation',
)

assert len(constraints) > 0
assert 'category' in constraints[0]
assert 'inverted_description' in constraints[0]
\""

# Check 10: AssumptionMiner works
perform_check "assumption_miner_works" \
    "AssumptionMiner can mine tacit assumptions from failure patterns" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import AssumptionMiner, Phase1Config, StructuredLogger

config = Phase1Config.from_env()
logger = StructuredLogger('test')
miner = AssumptionMiner(config=config, logger=logger)

patterns = [
    {
        'pattern_description': 'High failure rate in lattice defects',
        'failure_rate': 0.7,
        'data_points': 50,
    },
]

assumptions = miner.mine_assumptions(
    failure_patterns=patterns,
    correlation_id='test-correlation',
)

assert len(assumptions) > 0
assert assumptions[0].confidence_score >= 0.7
\""

# Check 11: Circuit breaker works
perform_check "circuit_breaker_works" \
    "Circuit breaker can detect failures and open" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import CircuitBreaker, CircuitBreakerState, StructuredLogger

logger = StructuredLogger('test')
cb = CircuitBreaker(threshold=2, timeout_ms=1000, logger=logger.logger)

assert cb.can_execute() == True
assert cb.get_stats()['state'] == 'closed'

# Record failures
cb.record_failure()
cb.record_failure()

# Should be open now
assert cb.get_stats()['state'] == 'open'
assert cb.can_execute() == False
\""

# Check 12: Dead letter queue works
perform_check "dlq_works" \
    "Dead letter queue can enqueue and dequeue items" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import DeadLetterQueue, StructuredLogger

logger = StructuredLogger('test')
dlq = DeadLetterQueue(max_size=10, logger=logger.logger)

item = {'test': 'data'}
dlq.enqueue(item)

assert dlq.size() == 1
retrieved = dlq.dequeue()
assert retrieved['test'] == 'data'
assert dlq.size() == 0
\""

# Check 13: Full audit end-to-end test
perform_check "full_audit_works" \
    "Full Phase I audit can be executed end-to-end" \
    "$PYTHON_CMD -c \"
import sys
import os
sys.path.insert(0, '$PHASE1_DIR/src')
from phase1_executor import EpistemicAuditExecutor, Phase1Config

# Set test environment variables
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '5'
os.environ['PHASE1_ENABLE_TACIT_MINING'] = 'true'
os.environ['PHASE1_ENABLE_RED_TEAM'] = 'true'

config = Phase1Config.from_env()
executor = EpistemicAuditExecutor(config=config)

problem = 'LENR thermal coefficient inconsistency shows 50% null results'
patterns = [
    {
        'pattern_description': 'Lattice defects cause irregular heat distribution',
        'failure_rate': 0.5,
        'data_points': 100,
    },
]

result = executor.perform_audit(
    problem_description=problem,
    failure_patterns=patterns,
    correlation_id='test-correlation',
)

assert result.phase == 'phase1_epistemic_audit'
assert result.audit_id is not None
assert len(result.tacit_assumptions) > 0
assert result.timestamp is not None

# Verify canonical format
result_dict = result.to_dict()
assert 'phase' in result_dict
assert 'audit_id' in result_dict
assert 'tacit_assumptions' in result_dict
assert 'contradictions' in result_dict
assert 'falsification_results' in result_dict
assert 'metrics' in result_dict
assert 'metadata' in result_dict
assert 'timestamp' in result_dict
\""

echo ""
echo "  },"
echo "  \"overall_status\": \"$([ $EXIT_CODE -eq 0 ] && echo 'PASS' || echo 'FAIL')\","
echo "  \"exit_code\": $EXIT_CODE,"
echo "  \"checks_passed\": $(echo "$CHECKS" | grep -c 'PASS' || echo 0),"
echo "  \"checks_total\": 13,"
echo "  \"note\": \"Phase I: Epistemic Audit executor is functional\""
echo "}"

exit $EXIT_CODE
