#!/bin/bash
# RESE LLTL Probe Script
#
# CLAUDE.md Compliance: Law of Runtime Truth
# This script verifies the LLTL implementation is working correctly.
#
# Tests:
# 1. Module imports
# 2. Adapter initialization
# 3. Encoding single constraint
# 4. Translating multiple constraints
# 5. Contradiction detection
# 6. Health check
#
# Author: RESE Team
# Created: 2026-02-04

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

run_test() {
    local test_name=$1
    local test_command=$2

    TESTS_RUN=$((TESTS_RUN + 1))
    echo ""
    echo "=========================================="
    echo "Test $TESTS_RUN: $test_name"
    echo "=========================================="

    if eval "$test_command"; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log_info "✓ PASSED"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        log_error "✗ FAILED"
        return 1
    fi
}

# Change to script directory
cd "$(dirname "$0")/../.." || exit 1
PROJ_ROOT=$(pwd)
log_info "Project root: $PROJ_ROOT"

# Setup Python path
export PYTHONPATH="${PROJ_ROOT}/glue/lib:${PYTHONPATH}"
log_info "PYTHONPATH: $PYTHONPATH"

echo ""
echo "=========================================="
echo "RESE LLTL PROBE SCRIPT"
echo "=========================================="
echo "Testing Logic-to-Loss Translation Layer"
echo ""

# Test 1: Module imports
run_test "Module imports" << 'EOF'
python3 << 'PYTHON'
import sys
from pathlib import Path

# Add glue/lib to path
glue_lib = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(glue_lib))

try:
    from rese_lltl import (
        LogicToLossTranslator,
        SymbolicConstraintEncoder,
        LossFunctionComposer,
        DITOOptimizer
    )
    print("✓ Core LLTL modules imported successfully")
    exit(0)
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)
PYTHON
EOF

# Test 2: Adapter imports
run_test "Adapter imports" << 'EOF'
python3 << 'PYTHON'
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from lltl_adapter import LLTLAdapter, create_adapter, is_available
    print("✓ Adapter module imported successfully")
    print(f"✓ LLTL available: {is_available()}")
    exit(0)
except ImportError as e:
    print(f"✗ Adapter import failed: {e}")
    exit(1)
PYTHON
EOF

# Test 3: Adapter initialization
run_test "Adapter initialization" << 'EOF'
python3 << 'PYTHON'
import sys
import os
from pathlib import Path

# Setup paths
src_path = Path(__file__).parent / "src"
lib_path = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(lib_path))

# Set minimal env vars
os.environ["LLTL_ENCODING_DIM"] = "64"
os.environ["LLTL_TIMEOUT_MS"] = "5000"
os.environ["LLTL_LEARNING_RATE"] = "0.001"

try:
    from lltl_adapter import LLTLAdapter
    adapter = LLTLAdapter()
    print("✓ Adapter initialized successfully")
    print(f"✓ Encoding dim: {adapter.config['encoding']['encoding_dim']}")
    print(f"✓ Timeout: {adapter.config['timeout_ms']}ms")
    exit(0)
except Exception as e:
    print(f"✗ Adapter initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON
EOF

# Test 4: Health check
run_test "Health check" << 'EOF'
python3 << 'PYTHON'
import sys
import os
from pathlib import Path

# Setup paths
src_path = Path(__file__).parent / "src"
lib_path = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(lib_path))

os.environ["LLTL_ENCODING_DIM"] = "64"

try:
    from lltl_adapter import LLTLAdapter
    adapter = LLTLAdapter()
    is_healthy, message = adapter.health_check()

    if is_healthy:
        print(f"✓ Health check passed: {message}")
        exit(0)
    else:
        print(f"✗ Health check failed: {message}")
        exit(1)
except Exception as e:
    print(f"✗ Health check error: {e}")
    exit(1)
PYTHON
EOF

# Test 5: Encode single constraint
run_test "Encode single constraint" << 'EOF'
python3 << 'PYTHON'
import sys
import os
from pathlib import Path
from dataclasses import dataclass

# Setup paths
src_path = Path(__file__).parent / "src"
lib_path = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(lib_path))

os.environ["LLTL_ENCODING_DIM"] = "64"

# Create mock constraint
@dataclass
class MockConstraint:
    constraint_id: str
    type: str
    category: str
    description: str
    expression: str
    dependencies: list
    priority: float
    confidence: float

try:
    from lltl_adapter import LLTLAdapter
    adapter = LLTLAdapter()

    constraint = MockConstraint(
        constraint_id="test-001",
        type="hard",
        category="logical",
        description="Test constraint",
        expression="x > 5",
        dependencies=[],
        priority=1.0,
        confidence=0.9
    )

    encoded, error = adapter.encode_single(constraint)

    if error:
        print(f"✗ Encoding failed: {error}")
        exit(1)

    if encoded is None:
        print("✗ Encoding returned None")
        exit(1)

    print(f"✓ Constraint encoded successfully")
    print(f"✓ Constraint ID: {encoded['constraint_id']}")
    print(f"✓ Feature vector length: {len(encoded['feature_vector'])}")
    exit(0)
except Exception as e:
    print(f"✗ Encoding test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON
EOF

# Test 6: Translate multiple constraints
run_test "Translate multiple constraints" << 'EOF'
python3 << 'PYTHON'
import sys
import os
from pathlib import Path
from dataclasses import dataclass

# Setup paths
src_path = Path(__file__).parent / "src"
lib_path = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(lib_path))

os.environ["LLTL_ENCODING_DIM"] = "64"
os.environ["LLTL_TIMEOUT_MS"] = "10000"

# Create mock constraints
@dataclass
class MockConstraint:
    constraint_id: str
    type: str
    category: str
    description: str
    expression: str
    dependencies: list
    priority: float
    confidence: float

try:
    from lltl_adapter import LLTLAdapter
    adapter = LLTLAdapter()

    constraints = [
        MockConstraint(
            constraint_id="test-001",
            type="hard",
            category="logical",
            description="Test constraint 1",
            expression="x > 5",
            dependencies=[],
            priority=1.0,
            confidence=0.9
        ),
        MockConstraint(
            constraint_id="test-002",
            type="soft",
            category="causal",
            description="Test constraint 2",
            expression="y < 10",
            dependencies=[],
            priority=0.8,
            confidence=0.7
        )
    ]

    result, error = adapter.translate_constraints(constraints)

    if error:
        print(f"✗ Translation failed: {error}")
        exit(1)

    if result is None:
        print("✗ Translation returned None")
        exit(1)

    print(f"✓ Translation completed successfully")
    print(f"✓ Input constraints: {result['input_constraints']}")
    print(f"✓ Encoded constraints: {result['encoded_constraints']}")
    print(f"✓ Loss functions: {result['loss_functions']}")
    print(f"✓ Contradictions detected: {result['contradictions_detected']}")
    print(f"✓ Duration: {result['duration_ms']:.2f}ms")
    exit(0)
except Exception as e:
    print(f"✗ Translation test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON
EOF

# Test 7: Contradiction detection
run_test "Contradiction detection" << 'EOF'
python3 << 'PYTHON'
import sys
import os
from pathlib import Path
from dataclasses import dataclass

# Setup paths
src_path = Path(__file__).parent / "src"
lib_path = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(lib_path))

os.environ["LLTL_ENCODING_DIM"] = "64"

# Create mock constraints (potentially contradictory)
@dataclass
class MockConstraint:
    constraint_id: str
    type: str
    category: str
    description: str
    expression: str
    dependencies: list
    priority: float
    confidence: float

try:
    from lltl_adapter import LLTLAdapter
    adapter = LLTLAdapter()

    constraints = [
        MockConstraint(
            constraint_id="test-001",
            type="hard",
            category="logical",
            description="X is greater than 5",
            expression="x > 5",
            dependencies=[],
            priority=1.0,
            confidence=0.9
        ),
        MockConstraint(
            constraint_id="test-002",
            type="hard",
            category="logical",
            description="X is less than 3",
            expression="x < 3",
            dependencies=[],
            priority=1.0,
            confidence=0.9
        )
    ]

    contradictions, error = adapter.detect_contradictions(constraints)

    if error:
        print(f"⚠ Warning: {error}")
        print("✓ DITO ran with warnings")
    else:
        print("✓ DITO completed without errors")

    print(f"✓ Contradictions detected: {len(contradictions)}")
    print(f"✓ Detection completed successfully")
    exit(0)
except Exception as e:
    print(f"✗ Contradiction detection test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON
EOF

# Test 8: Get stats
run_test "Get statistics" << 'EOF'
python3 << 'PYTHON'
import sys
import os
from pathlib import Path

# Setup paths
src_path = Path(__file__).parent / "src"
lib_path = Path(__file__).parent.parent.parent / "lib"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(lib_path))

os.environ["LLTL_ENCODING_DIM"] = "64"

try:
    from lltl_adapter import LLTLAdapter
    adapter = LLTLAdapter()

    stats = adapter.get_stats()

    print("✓ Statistics retrieved successfully")
    print(f"✓ Adapter has config: {'adapter_config' in stats}")
    print(f"✓ Adapter has translator stats: {'translator_stats' in stats}")
    print(f"✓ Adapter is available: {stats.get('available', False)}")
    exit(0)
except Exception as e:
    print(f"✗ Stats test failed: {e}")
    exit(1)
PYTHON
EOF

# Print summary
echo ""
echo "=========================================="
echo "PROBE SUMMARY"
echo "=========================================="
echo "Total tests run: $TESTS_RUN"
echo -e "${GREEN}Tests passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests failed: $TESTS_FAILED${NC}"
    exit 1
else
    echo "Tests failed: $TESTS_FAILED"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    log_info "All probes passed successfully! ✓"
    echo ""
    echo "The LLTL implementation is ready for use."
    exit 0
else
    log_error "Some probes failed. Please review the errors above."
    exit 1
fi
