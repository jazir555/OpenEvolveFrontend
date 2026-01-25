#!/bin/bash
# Script to verify all Lean 4 files in RESE

echo "===== RESE Lean 4 Verification Report ====="
echo ""
echo "Generated: $(date)"
echo ""

LEAN_DIR="/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/lean4"
cd "$LEAN_DIR" || exit 1

echo "## 1. Files Found"
echo "==============="
ls -1 *.lean 2>/dev/null | grep -v lakefile
echo ""

echo "## 2. Checking Each File"
echo "======================="
for file in Basic.lean Constraint.lean Templates.lean TestCases.lean RESE.lean; do
    if [ -f "$file" ]; then
        echo "### Checking $file ..."
        lean --check "$file" 2>&1 | head -20
        echo ""
    fi
done

echo "## 3. Theorem Inventory"
echo "===================="
for file in Basic.lean Constraint.lean Templates.lean TestCases.lean RESE.lean; do
    if [ -f "$file" ]; then
        echo "### $file"
        grep -E "^(theorem|example|def)" "$file" | head -20
        echo ""
    fi
done

echo "## 4. Admitted Proofs (sorry)"
echo "============================"
for file in Basic.lean Constraint.lean Templates.lean TestCases.lean RESE.lean; do
    if [ -f "$file" ]; then
        count=$(grep -c "sorry" "$file" 2>/dev/null || echo "0")
        if [ "$count" -gt 0 ]; then
            echo "$file: $count admitted proofs"
        fi
    done
echo ""

echo "## 5. Import Dependencies"
echo "========================"
grep -h "^import" *.lean | sort -u
