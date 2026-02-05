#!/bin/bash
###############################################################################
# Z3 API Probe Script
#
# Probes Z3 availability for Phase II behavioral equivalence verification.
#
# Following CLAUDE.md Law of Runtime Truth:
# - Verify Z3 is actually available before using it
# - Test both Python bindings and CLI
# - Test basic constraint solving
#
# Usage: ./check_z3_api.sh
#
# Exit codes:
#   0 - Z3 fully available (both Python and CLI)
#   1 - Z3 Python bindings only
#   2 - Z3 CLI only
#   3 - Z3 not available
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Z3 API Probe for RESE Phase II"
echo "=========================================="
echo ""

# Track availability
PYTHON_AVAILABLE=0
CLI_AVAILABLE=0

# Test 1: Check Python bindings
echo "Test 1: Checking Z3 Python bindings..."
if python3 -c "import z3; print('Z3 version:', z3.get_version())" 2>/dev/null; then
    PYTHON_AVAILABLE=1
    echo -e "${GREEN}✓${NC} Z3 Python bindings available"
    python3 -c "import z3; print('  Version:', z3.get_version())"
else
    echo -e "${YELLOW}✗${NC} Z3 Python bindings NOT available"
fi
echo ""

# Test 2: Check Z3 CLI
echo "Test 2: Checking Z3 CLI..."
if z3 --version 2>/dev/null; then
    CLI_AVAILABLE=1
    echo -e "${GREEN}✓${NC} Z3 CLI available"
else
    echo -e "${YELLOW}✗${NC} Z3 CLI NOT available"
fi
echo ""

# Test 3: Test basic constraint solving
echo "Test 3: Testing basic constraint solving..."

# Create test SMT-LIB file
TEST_FILE=$(mktemp --suffix=.smt2)
cat > "$TEST_FILE" << 'EOF'
; Simple satisfiability test
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(assert (< x 10))
(check-sat)
(get-model)
EOF

if z3 "$TEST_FILE" 2>/dev/null | grep -q "sat"; then
    echo -e "${GREEN}✓${NC} Z3 can solve simple constraints"
    echo "  Output:"
    z3 "$TEST_FILE" 2>/dev/null | head -5 | sed 's/^/  /'
else
    echo -e "${RED}✗${NC} Z3 constraint solving failed"
fi
rm -f "$TEST_FILE"
echo ""

# Test 4: Test theorem proving
echo "Test 4: Testing theorem proving..."

THEOREM_FILE=$(mktemp --suffix=.smt2)
cat > "$THEOREM_FILE" << 'EOF'
; Simple theorem: x > 0 implies x + 1 > 0
(set-logic LIA)
(declare-const x Int)
(assert (> x 0))
(assert (not (> (+ x 1) 0)))
(check-sat)
EOF

if z3 "$THEOREM_FILE" 2>/dev/null | grep -q "unsat"; then
    echo -e "${GREEN}✓${NC} Z3 can prove theorems"
    echo "  Theorem 'x > 0 → x + 1 > 0': PROVEN"
else
    echo -e "${YELLOW}✗${NC} Z3 theorem proving failed"
fi
rm -f "$THEOREM_FILE"
echo ""

# Test 5: Check for Z3-LeanAide bridge
echo "Test 5: Checking Z3-LeanAide bridge..."
if python3 -c "from z3_leanaide_bridge import Z3LeanAideBridge; print('Bridge available')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Z3-LeanAide bridge available"
else
    echo -e "${YELLOW}✗${NC} Z3-LeanAide bridge NOT available (optional)"
fi
echo ""

# Summary
echo "=========================================="
echo "Probe Summary"
echo "=========================================="
echo "Python bindings: $([ $PYTHON_AVAILABLE -eq 1 ] && echo 'Available' || echo 'Not available')"
echo "CLI:            $([ $CLI_AVAILABLE -eq 1 ] && echo 'Available' || echo 'Not available')"
echo ""

# Determine exit code
if [ $PYTHON_AVAILABLE -eq 1 ] && [ $CLI_AVAILABLE -eq 1 ]; then
    echo -e "${GREEN}Result: Z3 fully available (recommended)${NC}"
    echo ""
    echo "Recommended configuration:"
    echo "  export RESE_Z3_PHASE2_ENABLED=true"
    echo "  export Z3_TIMEOUT=10000"
    exit 0
elif [ $PYTHON_AVAILABLE -eq 1 ]; then
    echo -e "${YELLOW}Result: Z3 Python bindings only (usable)${NC}"
    echo ""
    echo "Configuration:"
    echo "  export RESE_Z3_PHASE2_ENABLED=true"
    echo "  export Z3_TIMEOUT=10000"
    exit 1
elif [ $CLI_AVAILABLE -eq 1 ]; then
    echo -e "${YELLOW}Result: Z3 CLI only (usable)${NC}"
    echo ""
    echo "Configuration:"
    echo "  export RESE_Z3_PHASE2_ENABLED=true"
    echo "  export Z3_TIMEOUT=10000"
    exit 2
else
    echo -e "${RED}Result: Z3 not available${NC}"
    echo ""
    echo "To install Z3:"
    echo "  pip install z3-solver"
    echo ""
    echo "Or download binary from:"
    echo "  https://github.com/Z3Prover/z3/releases"
    echo ""
    echo "Fallback configuration:"
    echo "  export RESE_Z3_PHASE2_ENABLED=false"
    exit 3
fi
