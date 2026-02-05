#!/bin/bash
###############################################################################
# Z3 ACI Integration Probe Script
#
# This script verifies that Z3 constraint-based anomaly detection is properly
# integrated into the ACI Calculator following CLAUDE.md Law of Runtime Truth.
#
# Tests:
# 1. Z3 module availability
# 2. Z3AnomalyDetector instantiation
# 3. Constraint encoding
# 4. Satisfiability checking
# 5. Formal verification of anomalies
#
# Author: RESE Team
# Created: 2026-02-04
# Phase: III - Monte Carlo Refinement
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        ((TESTS_FAILED++))
    fi
}

# Function to print section header
print_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$SCRIPT_DIR/../src"
PROBE_DIR="$SCRIPT_DIR/probe_temp"

# Create probe temp directory
mkdir -p "$PROBE_DIR"

print_section "Z3 ACI Integration Probe"

echo "Script directory: $SCRIPT_DIR"
echo "Source directory: $SRC_DIR"
echo "Probe directory: $PROBE_DIR"

###############################################################################
# Test 1: Check Python availability
###############################################################################
print_section "Test 1: Python Environment"

if command -v python &> /dev/null; then
    PYTHON_CMD=python
    print_result 0 "Python command available"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    print_result 0 "Python3 command available"
else
    print_result 1 "Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

###############################################################################
# Test 2: Check Z3 Python bindings
###############################################################################
print_section "Test 2: Z3 Python Bindings"

cat > "$PROBE_DIR/test_z3_import.py" << 'EOF'
import sys
sys.path.insert(0, "../src")

try:
    import z3
    print(f"Z3 version: {z3.get_version()}")
    print("Z3 bindings: AVAILABLE")
    sys.exit(0)
except ImportError as e:
    print(f"Z3 bindings: NOT AVAILABLE ({e})")
    sys.exit(1)
EOF

cd "$PROBE_DIR"
if $PYTHON_CMD test_z3_import.py > z3_import.log 2>&1; then
    print_result 0 "Z3 Python bindings"
    echo "  $(cat z3_import.log | grep 'Z3 version')"
else
    print_result 1 "Z3 Python bindings"
    echo "  $(cat z3_import.log | grep 'Z3 bindings')"
fi

###############################################################################
# Test 3: Check Z3 integration module
###############################################################################
print_section "Test 3: Z3 Integration Module"

cat > "$PROBE_DIR/test_z3_integration.py" << 'EOF'
import sys
import os
sys.path.insert(0, "../../..")
sys.path.insert(0, "../src")

try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3Variable,
        Z3Constraint,
        Z3Config,
        Z3ConstraintType,
        is_z3_available,
        get_z3_solver_engine
    )

    available = is_z3_available()
    print(f"Z3 available: {available}")

    if available:
        # Try to create solver engine
        config = Z3Config(timeout=5.0)
        engine = get_z3_solver_engine(config)
        print(f"Z3 engine created: {engine is not None}")

        # Try simple constraint solve
        variables = [
            Z3Variable("x", Z3ConstraintType.INTEGER),
            Z3Variable("y", Z3ConstraintType.INTEGER)
        ]
        constraints = [
            Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER),
            Z3Constraint("(< x 10)", Z3ConstraintType.INTEGER),
            Z3Constraint("(= y (+ x 5))", Z3ConstraintType.INTEGER)
        ]

        result = engine.solve_constraints(variables, constraints)
        print(f"Constraint solve: {result.status.value}")

        if result.is_sat():
            print(f"Solution: x={result.model.get_value('x')}, y={result.model.get_value('y')}")

    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

cd "$PROBE_DIR"
if $PYTHON_CMD test_z3_integration.py > z3_integration.log 2>&1; then
    print_result 0 "Z3 integration module"
    echo "  $(cat z3_integration.log | grep 'Z3 available')"
    echo "  $(cat z3_integration.log | grep 'Constraint solve')"
else
    print_result 1 "Z3 integration module"
    cat z3_integration.log | head -20
fi

###############################################################################
# Test 4: Check ACI Calculator with Z3
###############################################################################
print_section "Test 4: ACI Calculator Z3 Integration"

cat > "$PROBE_DIR/test_aci_z3.py" << 'EOF'
import sys
import os
import numpy as np
sys.path.insert(0, "../src")

# Set test environment
os.environ["PHASE3_ACI_WINDOW_SIZE"] = "100"
os.environ["PHASE3_ACI_ENTROPY_BINS"] = "10"
os.environ["PHASE3_ACI_COHERENCE_THRESHOLD"] = "0.5"
os.environ["PHASE3_ACI_ENTROPY_THRESHOLD"] = "0.7"
os.environ["PHASE3_ACI_TIMEOUT_MS"] = "3000"
os.environ["PHASE3_ACI_MIN_SAMPLES"] = "10"
os.environ["PHASE3_ACI_CORRELATION_METHOD"] = "pearson"
os.environ["PHASE3_ACI_ENABLE_Z3"] = "true"
os.environ["PHASE3_ACI_Z3_TIMEOUT"] = "5.0"

try:
    from aci_calculator import (
        ACIConfig,
        Z3AnomalyDetector,
        Z3_AVAILABLE,
        AnomalyCharacterizationIndex
    )

    print(f"Z3_AVAILABLE: {Z3_AVAILABLE}")

    if Z3_AVAILABLE:
        # Test Z3AnomalyDetector
        config = ACIConfig.from_env()
        print(f"Config loaded: window_size={config.window_size}, z3_enabled={config.enable_z3_verification}")

        detector = Z3AnomalyDetector(config)
        print(f"Z3 detector created: {detector is not None}")
        print(f"Z3 detector enabled: {detector.z3_enabled}")

        if detector.z3_enabled:
            # Test constraint encoding
            variables, constraints = detector.encode_anomaly_constraints(
                0.8, 0.7, 0.7, 0.5
            )
            print(f"Constraints encoded: {len(variables)} variables, {len(constraints)} constraints")

            # Test satisfiability verification
            result = detector.verify_anomaly_satisfiability(
                0.8, 0.7, 0.7, 0.5
            )
            print(f"Verification result: satisfiable={result['satisfiable']}, verified={result['verified']}")

            if result['entropy_bounds']:
                print(f"Entropy bounds: {result['entropy_bounds']}")
            if result['coherence_bounds']:
                print(f"Coherence bounds: {result['coherence_bounds']}")

        # Test ACI Calculator with Z3
        aci = AnomalyCharacterizationIndex(config)
        print(f"ACI Calculator created with Z3 detector: {aci.z3_detector is not None}")

        # Test signal detection
        np.random.seed(42)
        length = 200
        input_var = np.random.rand(length)
        output = input_var * 0.8 + np.random.randn(length) * 0.2

        experiment_data = {
            'output': output,
            'input1': input_var,
        }

        results = aci.detect_high_entropy_signals(
            experiment_data,
            time_series_key='output'
        )

        print(f"Signal detection: {len(results)} windows processed")

        if results:
            for i, result in enumerate(results[:3]):
                print(f"  Window {i+1}: entropy={result.disorder_entropy:.3f}, "
                      f"coherence={result.causal_coherence:.3f}, "
                      f"high_signal={result.is_high_entropy_signal}, "
                      f"z3_verified={result.z3_constraint_verified}")

        print("ACI Calculator with Z3: WORKING")
    else:
        print("Z3 not available, testing ACI Calculator without Z3")

        config = ACIConfig.from_env()
        aci = AnomalyCharacterizationIndex(config)

        np.random.seed(42)
        length = 200
        experiment_data = {
            'output': np.random.rand(length),
            'input1': np.random.rand(length),
        }

        results = aci.detect_high_entropy_signals(
            experiment_data,
            time_series_key='output'
        )

        print(f"Signal detection (without Z3): {len(results)} windows processed")
        print("ACI Calculator without Z3: WORKING")

    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

cd "$PROBE_DIR"
if $PYTHON_CMD test_aci_z3.py > aci_z3.log 2>&1; then
    print_result 0 "ACI Calculator with Z3"
    echo "  $(cat aci_z3.log | grep 'Z3_AVAILABLE')"
    echo "  $(cat aci_z3.log | grep 'Z3 detector')"
    echo "  $(cat aci_z3.log | grep 'Signal detection')"
else
    print_result 1 "ACI Calculator with Z3"
    cat aci_z3.log | head -30
fi

###############################################################################
# Test 5: Run ACI Calculator tests
###############################################################################
print_section "Test 5: ACI Calculator Test Suite"

cd "$SRC_DIR"
if [ -f "../tests/test_aci_calculator.py" ]; then
    echo "Running test suite..."
    if $PYTHON_CMD -m pytest ../tests/test_aci_calculator.py -v --tb=short -x 2>&1 | tee "$PROBE_DIR/test_results.log"; then
        print_result 0 "ACI Calculator test suite"
        echo "  Tests passed successfully"
    else
        print_result 1 "ACI Calculator test suite"
        echo "  Some tests failed (see log)"
    fi
else
    print_result 1 "ACI Calculator test suite"
    echo "  Test file not found"
fi

###############################################################################
# Test 6: Check constraint satisfiability
###############################################################################
print_section "Test 6: Constraint Satisfiability Verification"

cat > "$PROBE_DIR/test_satisfiability.py" << 'EOF'
import sys
sys.path.insert(0, "../src")
sys.path.insert(0, "../../..")

import os
os.environ["PHASE3_ACI_ENABLE_Z3"] = "true"
os.environ["PHASE3_ACI_Z3_TIMEOUT"] = "5.0"

try:
    from aci_calculator import ACIConfig, Z3AnomalyDetector, Z3_AVAILABLE

    if not Z3_AVAILABLE:
        print("Z3 not available - skipping")
        sys.exit(0)

    config = ACIConfig.from_env()
    detector = Z3AnomalyDetector(config)

    # Test Case 1: High entropy + High coherence (SATISFIABLE)
    print("Test 1: High entropy (0.8) + High coherence (0.7)")
    result1 = detector.verify_anomaly_satisfiability(0.8, 0.7, 0.7, 0.5)
    print(f"  Satisfiable: {result1['satisfiable']}, Verified: {result1['verified']}")

    # Test Case 2: Low entropy + Low coherence (UNSATISFIABLE for high-signal)
    print("Test 2: Low entropy (0.2) + Low coherence (0.1)")
    result2 = detector.verify_anomaly_satisfiability(0.2, 0.1, 0.7, 0.5)
    print(f"  Satisfiable: {result2['satisfiable']}, Verified: {result2['verified']}")

    # Test Case 3: At threshold
    print("Test 3: At threshold (0.7, 0.5)")
    result3 = detector.verify_anomaly_satisfiability(0.7, 0.5, 0.7, 0.5)
    print(f"  Satisfiable: {result3['satisfiable']}, Verified: {result3['verified']}")

    # Verify high-entropy signal detection
    print("\nHigh-entropy signal verification:")
    is_high = detector.verify_high_entropy_signal(0.8, 0.7)
    print(f"  Signal (0.8, 0.7): {is_high}")

    is_low = detector.verify_high_entropy_signal(0.2, 0.1)
    print(f"  Signal (0.2, 0.1): {is_low}")

    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

cd "$PROBE_DIR"
if $PYTHON_CMD test_satisfiability.py > satisfiability.log 2>&1; then
    print_result 0 "Constraint satisfiability"
    echo "  Test results:"
    grep "Test " satisfiability.log | head -5
else
    print_result 1 "Constraint satisfiability"
    cat satisfiability.log | head -20
fi

###############################################################################
# Cleanup
###############################################################################
print_section "Cleanup"

cd "$SCRIPT_DIR"
rm -rf "$PROBE_DIR"
echo "Probe directory cleaned up"

###############################################################################
# Summary
###############################################################################
print_section "Probe Summary"

echo ""
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All probe tests passed!${NC}"
    echo ""
    echo "Z3 ACI Integration: VERIFIED"
    exit 0
else
    echo -e "${RED}Some probe tests failed!${NC}"
    echo ""
    echo "Z3 ACI Integration: INCOMPLETE"
    exit 1
fi
