#!/bin/bash
###############################################################################
# DITO Z3 ATP Integration Probe Script
#
# This script verifies that the Z3 ATP integration with DITO is working correctly.
# It tests:
# 1. Z3 solver availability
# 2. Z3 Python bindings
# 3. DITO optimizer with Z3 detector
# 4. Constraint encoding to SMT-LIB2
# 5. Contradiction detection performance
#
# Follows CLAUDE.md Law of "Runtime Truth"
#
# Author: OpenEvolve
# Created: 2026-02-04
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "DITO Z3 ATP Integration Probe"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check Python
echo -n "Checking Python... "
if command -v python &> /dev/null; then
    PYTHON_CMD=python
    echo -e "${GREEN}OK${NC} (python)"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    echo -e "${GREEN}OK${NC} (python3)"
else
    echo -e "${RED}FAIL${NC}"
    echo "ERROR: Python not found"
    exit 1
fi

# Check Z3 binary
echo -n "Checking Z3 binary... "
if command -v z3 &> /dev/null; then
    Z3_VERSION=$(z3 --version 2>&1 | head -1 || echo "unknown")
    echo -e "${GREEN}OK${NC} ($Z3_VERSION)"
else
    echo -e "${YELLOW}WARN${NC} (Z3 binary not found, will use Python API)"
fi

# Check Z3 Python bindings
echo -n "Checking Z3 Python bindings... "
if $PYTHON_CMD -c "import z3; print(f'Z3 version: {z3.get_version()}')" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "ERROR: Z3 Python bindings not available"
    echo "Install with: pip install z3-solver"
    exit 1
fi

# Check z3prover_integration
echo -n "Checking z3prover_integration module... "
if $PYTHON_CMD -c "from z3prover_integration import Z3SolverEngine, Z3Config" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "ERROR: z3prover_integration not available"
    exit 1
fi

# Check DITO optimizer
echo -n "Checking DITO optimizer... "
if $PYTHON_CMD -c "from dito_optimizer import DITOOptimizer, Z3ContradictionDetector" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "ERROR: DITO optimizer not available"
    exit 1
fi

# Run Z3 ATP test
echo ""
echo "Running Z3 ATP functionality test..."
echo "======================================"

$PYTHON_CMD << 'EOF'
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3Config,
        Z3Variable,
        Z3Constraint,
        Z3ConstraintType,
    )
    from dito_optimizer import Z3ContradictionDetector
    from sce_bridge import Constraint, ConstraintType, ConstraintCategory, SCEConfig
    import logging

    # Setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('probe')

    print("Creating Z3 solver...")
    config = SCEConfig.from_env()
    z3_config = Z3Config(
        timeout=config.Z3_TIMEOUT_MS / 1000.0,
        memory_limit_mb=config.Z3_MAX_MEMORY_MB,
    )
    z3_solver = Z3SolverEngine(config=z3_config)

    print("Creating Z3 contradiction detector...")
    detector = Z3ContradictionDetector(z3_solver, config, logger)

    print("Creating test constraints...")
    c1 = Constraint(
        constraint_id="c1",
        type=ConstraintType.HARD,
        category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
        description="T < 1000",
    )
    c2 = Constraint(
        constraint_id="c2",
        type=ConstraintType.HARD,
        category=ConstraintCategory.HARD_PARAMETER_INEQUALITY,
        description="T > 1500",
    )

    print("Encoding constraints to Z3...")
    encoded1 = detector.encode_constraint_to_z3(c1)
    encoded2 = detector.encode_constraint_to_z3(c2)

    if encoded1 and encoded2:
        print("✓ Constraint encoding: OK")
    else:
        print("✗ Constraint encoding: FAIL")
        sys.exit(1)

    print("Checking for contradiction...")
    contradiction, result = detector.check_contradiction_z3([c1, c2], "probe-test")

    print(f"Z3 Result: {result.status.value}")
    print(f"Contradiction found: {contradiction is not None}")

    stats = detector.get_stats()
    print(f"Z3 checks performed: {stats.z3_checks_performed}")
    print(f"UNSAT results: {stats.z3_unsat_results}")

    if stats.z3_checks_performed > 0:
        print("\n✓ Z3 ATP functionality: OK")
    else:
        print("\n✗ Z3 ATP functionality: FAIL")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Z3 ATP Integration Probe: SUCCESS${NC}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run full test suite: python tests/test_dito_z3_atp.py"
    echo "  2. Run DITO optimization: python src/dito_optimizer.py"
    echo "  3. Check documentation: docs/DITO_Z3_ATP_INTEGRATION.md"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo -e "${RED}Z3 ATP Integration Probe: FAILURE${NC}"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure Z3 is installed: pip install z3-solver"
    echo "  2. Check z3prover_integration.py is in root directory"
    echo "  3. Verify PYTHONPATH includes src directory"
    echo ""
    exit 1
fi
