#!/bin/bash
# verify_all.sh
# Verify all Lean 4 files in RESE project
#
# Usage: ./scripts/verify_all.sh

set -e  # Exit on error

RESE_LEAN_DIR="/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/lean4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESE_LEAN_DIR/verification_$TIMESTAMP.log"

echo "========================================"
echo "RESE Lean 4 Verification Script"
echo "========================================"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""

cd "$RESE_LEAN_DIR"

# Check Lean 4 is available
if ! command -v lean &> /dev/null; then
    echo "ERROR: lean command not found"
    echo "Please install Lean 4: https://leanprover.github.io/lean4/doc/setup.html"
    exit 1
fi

echo "Lean version: $(lean --version)"
echo ""

# Files to verify (in dependency order)
FILES=(
    "Basic.lean"
    "Constraint.lean"
    "Templates.lean"
    "TestCases.lean"
    "RESE.lean"
)

# Verify each file
echo "Verifying files..."
echo ""

TOTAL=0
PASSED=0
FAILED=0

for file in "${FILES[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo -n "Checking $file... "

    if [ ! -f "$file" ]; then
        echo "[FAIL] File not found"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run Lean 4
    if lean --make "$file" > /dev/null 2>&1; then
        echo "[OK]"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL]"
        FAILED=$((FAILED + 1))
        echo "  Error output:" >> "$LOG_FILE"
        lean --make "$file" >> "$LOG_FILE" 2>&1
    fi
done

echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo "Total files: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "[SUCCESS] All files verified successfully!"
    exit 0
else
    echo "[WARNING] Some files failed verification"
    echo "Check log: $LOG_FILE"
    exit 1
fi
