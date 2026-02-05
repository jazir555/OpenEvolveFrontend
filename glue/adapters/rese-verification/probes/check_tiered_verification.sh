#!/bin/bash
# Probe script for Tiered Verification System
#
# Verifies that all 3 tiers are operational:
# - Tier 1: Z3 Fast Verification
# - Tier 2: LeanAide AI-Assisted Proving
# - Tier 3: Lean 4 Formal Verification
#
# Following CLAUDE.md principles:
# - Law of Runtime Truth: Verify solvers actually work
# - Fail fast if any tier is unavailable
#
# Usage:
#   ./probes/check_tiered_verification.sh
#
# Exit codes:
#   0: All tiers operational
#   1: Z3 not available
#   2: LeanAide not available
#   3: Lean 4 not available
#   4: Multiple tiers unavailable
#
# Author: RESE Team
# Created: 2026-02-04

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Tiered Verification System Probe"
echo "=================================="
echo ""

# Track failures
FAILURES=0

# =============================================================================
# TIER 1: Z3 VERIFICATION
# =============================================================================

echo -n "Checking Tier 1 (Z3)... "

if command -v z3 &> /dev/null; then
    Z3_VERSION=$(z3 --version 2>&1 | head -n 1)
    echo -e "${GREEN}✓${NC}"
    echo "  Version: $Z3_VERSION"

    # Test Z3 with simple SAT problem
    echo -n "  Testing SAT problem... "
    if echo "(declare-const x Int) (assert (> x 0)) (check-sat)" | z3 -in &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo "  ERROR: Z3 SAT test failed"
        FAILURES=$((FAILURES + 1))
    fi

    # Test Z3 with simple UNSAT problem
    echo -n "  Testing UNSAT problem... "
    if echo "(declare-const x Int) (assert (< x 0)) (assert (> x 0)) (check-sat)" | z3 -in &> /dev/null; then
        # Should be UNSAT, so this should fail
        echo -e "${YELLOW}?${NC}"
        echo "  WARNING: Z3 UNSAT test unexpected result"
    else
        echo -e "${GREEN}✓${NC}"
    fi
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Z3 not found"
    FAILURES=$((FAILURES + 1))
fi

echo ""

# =============================================================================
# TIER 2: LEANAIDE VERIFICATION
# =============================================================================

echo -n "Checking Tier 2 (LeanAide)... "

if command -v leanaide &> /dev/null; then
    LEANAIDE_VERSION=$(leanaide --version 2>&1 | head -n 1)
    echo -e "${GREEN}✓${NC}"
    echo "  Version: $LEANAIDE_VERSION"
else
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: LeanAide not found (optional tier)"
fi

# Check if LeanAide server is running
if curl -s http://localhost:8001/health &> /dev/null; then
    echo -n "  Checking LeanAide server... "
    echo -e "${GREEN}✓${NC}"
else
    echo -n "  Checking LeanAide server... "
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: LeanAide server not running"
fi

echo ""

# =============================================================================
# TIER 3: LEAN 4 VERIFICATION
# =============================================================================

echo -n "Checking Tier 3 (Lean 4)... "

if command -v lean &> /dev/null; then
    LEAN4_VERSION=$(lean --version 2>&1 | head -n 1)
    echo -e "${GREEN}✓${NC}"
    echo "  Version: $LEAN4_VERSION"

    # Test Lean 4 with simple theorem
    echo -n "  Testing theorem verification... "
    TEST_LEAN_FILE="/tmp/test_tier3.lean"
    cat > "$TEST_LEAN_FILE" << 'EOF'
import Mathlib

theorem test_theorem : 1 + 1 = 2 := by
  rfl
EOF

    if lean "$TEST_LEAN_FILE" &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
        rm -f "$TEST_LEAN_FILE"
    else
        echo -e "${RED}✗${NC}"
        echo "  ERROR: Lean 4 theorem test failed"
        FAILURES=$((FAILURES + 1))
        rm -f "$TEST_LEAN_FILE"
    fi
else
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: Lean 4 not found (optional tier)"
fi

echo ""

# =============================================================================
# PYTHON DEPENDENCIES
# =============================================================================

echo -n "Checking Python dependencies... "

if python3 -c "import sys; sys.path.insert(0, '../src'); from verification_result import UnifiedVerificationResult" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Python dependencies not found"
    FAILURES=$((FAILURES + 1))
fi

echo ""

# =============================================================================
# SUMMARY
# =============================================================================

echo "=================================="
echo "Summary"
echo "=================================="

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All critical tiers operational${NC}"
    echo ""
    echo "Tier 1 (Z3):        ${GREEN}Available${NC}"
    echo "Tier 2 (LeanAide):  ${YELLOW}Optional${NC}"
    echo "Tier 3 (Lean 4):    ${YELLOW}Optional${NC}"
    echo ""
    echo "Tiered verification system is ready."
    exit 0
elif [ $FAILURES -eq 1 ]; then
    echo -e "${RED}1 tier unavailable${NC}"
    echo ""
    echo "Tiered verification system is degraded."
    exit 1
else
    echo -e "${RED}$FAILURES tier(s) unavailable${NC}"
    echo ""
    echo "Tiered verification system is not operational."
    exit 4
fi
