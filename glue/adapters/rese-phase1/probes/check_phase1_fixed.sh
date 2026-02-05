#!/bin/bash
# check_phase1_fixed.sh - RESE Phase I Probe
# Following "Law of Runtime Truth" - we trust execution, not documentation

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
    echo '{"error": "Python not found"}'
    exit 1
fi

# Change to Frontend root
cd "$(dirname "$0")/../../../.." || exit 1
FRONTEND_ROOT="$(pwd)"
PHASE1_DIR="${FRONTEND_ROOT}/glue/adapters/rese-phase1"

# Generate metadata
CORRELATION_ID=$($PYTHON_CMD -c "import uuid; print(str(uuid.uuid4()))")
TIMESTAMP=$($PYTHON_CMD -c "from datetime import datetime; print(datetime.utcnow().isoformat() + 'Z')")

# Initialize counters
TOTAL_CHECKS=13
PASSED_CHECKS=0
FAILED_CHECKS=0

echo "{"
echo "  \"probe_name\": \"check_phase1\","
echo "  \"probe_type\": \"phase1_verification\","
echo "  \"correlation_id\": \"$CORRELATION_ID\","
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"source_service\": \"phase1_probe\","
echo "  \"target_service\": \"rese_phase1\","
echo "  \"phase1_directory\": \"$PHASE1_DIR\","
echo "  \"checks\": {"

DELIMITER=""

# Function to run a check
run_check() {
    local name=$1
    local desc=$2
    local command=$3

    echo "$DELIMITER"
    echo "    \"$name\": {"
    echo "      \"description\": \"$desc\","

    if eval "$command" > /tmp/check_output.txt 2>&1; then
        echo "      \"status\": \"PASS\","
        echo "      \"message\": \"$name successful\""
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo "      \"status\": \"FAIL\","
        echo "      \"message\": \"$name failed\","
        echo "      \"error\": $(head -1 /tmp/check_output.txt | sed 's/"/\\"/g' | tr -d '\n')"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi

    echo -n "    }"
    DELimiter=","
}

# Check 1: Directory exists
run_check "directory_exists" \
    "Phase I adapter directory exists" \
    "[ -d '$PHASE1_DIR' ]"

# Check 2: Executor module exists
run_check "executor_module_exists" \
    "Phase I executor module exists" \
    "[ -f '$PHASE1_DIR/src/phase1_executor.py' ]"

# Check 3: Adapter module exists
run_check "adapter_module_exists" \
    "Phase I adapter module exists" \
    "[ -f '$PHASE1_DIR/src/phase1_adapter.py' ]"

# Check 4: Executor importable
run_check "executor_importable" \
    "Executor module can be imported" \
    "$PYTHON_CMD -c 'import sys; sys.path.insert(0, \"glue/adapters/rese-phase1/src\"); from phase1_executor import EpistemicAuditExecutor'"

# Check 5: Adapter importable
run_check "adapter_importable" \
    "Adapter module can be imported" \
    "$PYTHON_CMD -c 'import sys; sys.path.insert(0, \"glue/adapters/rese-phase1/src\"); from phase1_adapter import Phase1Adapter'"

# Check 6: Config loadable
run_check "config_loadable" \
    "Phase I configuration can be loaded" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import Phase1Config
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
config = Phase1Config.from_env()
assert config.TIMEOUT_MS == 15000
\""

# Check 7: Executor instantiable
run_check "executor_instantiable" \
    "Executor can be instantiated" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import EpistemicAuditExecutor, Phase1Config
os.environ['PHASE1_TIMEOUT_MS'] = '15000'
os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
os.environ['PHASE1_CIRCUIT_BREAKER_THRESHOLD'] = '5'
config = Phase1Config.from_env()
executor = EpistemicAuditExecutor(config=config)
assert executor is not None
\""

# Check 8: TacitAssumption dataclass
run_check "tacit_assumption_works" \
    "TacitAssumption dataclass works" \
    "$PYTHON_CMD -c \"
import sys, json
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
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
\""

# Check 9: ConstraintHardener
run_check "constraint_hardener_works" \
    "ConstraintHardener can extract constraints" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import ConstraintHardener, Phase1Config, StructuredLogger
config = Phase1Config.from_env()
logger = StructuredLogger('test')
hardener = ConstraintHardener(config=config, logger=logger)
constraints = hardener.harden_constraints(
    problem_description='This problem is impossible to solve due to limited resources.',
    correlation_id='test-correlation',
)
assert len(constraints) > 0
\""

# Check 10: AssumptionMiner
run_check "assumption_miner_works" \
    "AssumptionMiner can mine tacit assumptions" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
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
\""

# Check 11: Circuit breaker
run_check "circuit_breaker_works" \
    "Circuit breaker can detect failures" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import CircuitBreaker, StructuredLogger
logger = StructuredLogger('test')
cb = CircuitBreaker(threshold=2, timeout_ms=1000, structured_logger=logger.logger)
assert cb.can_execute() == True
cb.record_failure()
cb.record_failure()
assert cb.can_execute() == False
\""

# Check 12: Dead letter queue
run_check "dlq_works" \
    "Dead letter queue can enqueue/dequeue" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import DeadLetterQueue, StructuredLogger
logger = StructuredLogger('test')
dlq = DeadLetterQueue(max_size=10, structured_logger=logger.logger)
item = {'test': 'data'}
dlq.enqueue(item)
assert dlq.size() == 1
retrieved = dlq.dequeue()
assert retrieved['test'] == 'data'
\""

# Check 13: Full audit
run_check "full_audit_works" \
    "Full Phase I audit can execute" \
    "$PYTHON_CMD -c \"
import sys, os
sys.path.insert(0, 'glue/adapters/rese-phase1/src')
from phase1_executor import EpistemicAuditExecutor, Phase1Config
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
\""

echo ""
echo "  },"
echo "  \"overall_status\": \"$([ $FAILED_CHECKS -eq 0 ] && echo 'PASS' || echo 'FAIL')\","
echo "  \"exit_code\": $([ $FAILED_CHECKS -eq 0 ] && echo '0' || echo '1'),"
echo "  \"checks_passed\": $PASSED_CHECKS,"
echo "  \"checks_total\": $TOTAL_CHECKS,"
echo "  \"note\": \"Phase I: Epistemic Audit executor verification\""
echo "}"

# Clean up
rm -f /tmp/check_output.txt

exit $([ $FAILED_CHECKS -eq 0 ] && echo '0' || echo '1')
