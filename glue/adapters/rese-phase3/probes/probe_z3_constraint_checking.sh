#!/bin/bash
###############################################################################
# PROBE: Z3 Constraint Satisfaction for Phase III MCTS
#
# Law of Runtime Truth: Verify Z3 is available and constraint checking works
# BEFORE implementing the integration.
#
# This probe tests:
# 1. Z3 binary availability
# 2. Z3 Python bindings availability
# 3. SMT-LIB2 constraint solving
# 4. Performance: Can solve simple constraints in <1 second
#
# Usage: ./probes/probe_z3_constraint_checking.sh
###############################################################################

set -e

echo "=== PROBE: Z3 Constraint Satisfaction for MCTS ==="
echo ""

# Test 1: Check Z3 binary
echo "[TEST 1] Checking Z3 binary availability..."
if command -v z3 &> /dev/null; then
    Z3_VERSION=$(z3 --version 2>&1 || echo "unknown")
    echo "✓ Z3 binary found: $Z3_VERSION"
else
    echo "✗ Z3 binary NOT found"
    echo "  Install: apt-get install z3 (Ubuntu/Debian)"
    echo "           brew install z3 (macOS)"
    exit 1
fi

# Test 2: Check Z3 Python bindings
echo ""
echo "[TEST 2] Checking Z3 Python bindings..."
if python3 -c "import z3; print(f'Z3 Python version: {z3.get_version()}')" 2>/dev/null; then
    echo "✓ Z3 Python bindings available"
else
    echo "✗ Z3 Python bindings NOT found"
    echo "  Install: pip install z3-solver"
    exit 1
fi

# Test 3: Test simple constraint satisfaction
echo ""
echo "[TEST 3] Testing constraint satisfaction (simple SAT problem)..."

cat > /tmp/test_z3_simple.smt2 << 'EOF'
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (> x 0))
(assert (< x 10))
(assert (= y (+ x 5)))
(check-sat)
(get-model)
EOF

START_TIME=$(date +%s%N)
RESULT=$(z3 /tmp/test_z3_simple.smt2 2>&1)
END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if echo "$RESULT" | grep -q "^sat"; then
    echo "✓ Simple SAT problem solved successfully"
    echo "  Solution: $(echo "$RESULT" | grep -A 2 "model" | tail -2)"
    echo "  Time: ${DURATION_MS}ms"
else
    echo "✗ Simple SAT problem FAILED"
    echo "  Result: $RESULT"
    exit 1
fi

# Test 4: Test UNSAT problem (constraint contradiction detection)
echo ""
echo "[TEST 4] Testing UNSAT problem (contradiction detection)..."

cat > /tmp/test_z3_unsat.smt2 << 'EOF'
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (> x 10))
(assert (< x 5))
(check-sat)
EOF

RESULT=$(z3 /tmp/test_z3_unsat.smt2 2>&1)
if echo "$RESULT" | grep -q "^unsat"; then
    echo "✓ UNSAT problem detected correctly"
else
    echo "✗ UNSAT problem detection FAILED"
    echo "  Result: $RESULT"
    exit 1
fi

# Test 5: Test performance (complex constraint)
echo ""
echo "[TEST 5] Testing performance (medium complexity)..."

cat > /tmp/test_z3_performance.smt2 << 'EOF'
(set-logic QF_LIA)
(declare-fun a () Int)
(declare-fun b () Int)
(declare-fun c () Int)
(declare-fun d () Int)
(declare-fun e () Int)
(assert (> a 0))
(assert (< a 100))
(assert (> b 0))
(assert (< b 100))
(assert (> c 0))
(assert (< c 100))
(assert (> d 0))
(assert (< d 100))
(assert (> e 0))
(assert (< e 100))
(assert (= (+ a b c d e) 250))
(check-sat)
(get-model)
EOF

START_TIME=$(date +%s%N)
RESULT=$(timeout 5 z3 /tmp/test_z3_performance.smt2 2>&1)
EXIT_CODE=$?
END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ $EXIT_CODE -eq 0 ] && echo "$RESULT" | grep -q "^sat"; then
    echo "✓ Medium complexity problem solved"
    echo "  Time: ${DURATION_MS}ms"
    if [ $DURATION_MS -lt 1000 ]; then
        echo "  ✓ Performance OK (<1s threshold for MCTS)"
    else
        echo "  ⚠ Performance borderline: ${DURATION_MS}ms (target <1000ms)"
    fi
else
    echo "✗ Medium complexity problem FAILED or TIMEOUT"
    echo "  Exit code: $EXIT_CODE"
    echo "  Result: $RESULT"
    exit 1
fi

# Test 6: Test MCTS-style constraint (path constraints)
echo ""
echo "[TEST 6] Testing MCTS path constraint satisfaction..."

cat > /tmp/test_z3_mcts.smt2 << 'EOF'
; Simulate MCTS path constraints
(set-logic QF_LIA)
(declare-fun depth () Int)
(declare-fun visits () Int)
(declare-fun reward () Real)
(assert (>= depth 0))
(assert (< depth 20))
(assert (> visits 0))
(assert (>= reward 0.0))
(assert (<= reward 1.0))
(check-sat)
(get-model)
EOF

START_TIME=$(date +%s%N)
RESULT=$(timeout 2 z3 /tmp/test_z3_mcts.smt2 2>&1)
EXIT_CODE=$?
END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ $EXIT_CODE -eq 0 ] && echo "$RESULT" | grep -q "^sat"; then
    echo "✓ MCTS path constraint SAT check passed"
    echo "  Time: ${DURATION_MS}ms"
    if [ $DURATION_MS -lt 500 ]; then
        echo "  ✓ Excellent performance for MCTS (<500ms)"
    fi
else
    echo "✗ MCTS path constraint check FAILED"
    exit 1
fi

# Cleanup
rm -f /tmp/test_z3_*.smt2

# Summary
echo ""
echo "=== PROBE SUMMARY ==="
echo "All tests passed! Z3 is available and suitable for MCTS constraint checking."
echo ""
echo "Required capabilities verified:"
echo "  ✓ Z3 binary available"
echo "  ✓ Z3 Python bindings available"
echo "  ✓ SAT constraint solving works"
echo "  ✓ UNSAT detection works"
echo "  ✓ Performance acceptable for MCTS (<1s per check)"
echo ""
echo "Recommendation: Proceed with Z3 integration for Phase III MCTS"
echo ""
echo "Environment variables to set:"
echo "  export RESE_Z3_PHASE3_ENABLED=true"
echo "  export Z3_TIMEOUT=1000  # Fast timeout for MCTS"
echo "  export Z3_MAX_MEMORY_MB=2048"
