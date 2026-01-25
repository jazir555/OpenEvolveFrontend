# Syntax Error Fix Report

## Status: Partial Fix

**Successfully Fixed:** 2 files
**Require Manual Review:** 10 files

---

## ✅ Successfully Fixed (2 files)

1. **simple_verify_implementation.py**
   - Issue: Missing except block at line 77
   - Fix: Added `except Exception as e: raise` block
   - Status: VERIFIED - Compiles successfully

2. **sovereign_gauntlets.py**
   - Issue: Empty except block at line 451
   - Fix: Added `raise` statement to except block
   - Status: VERIFIED - Compiles successfully

---

## ❌ Require Manual Review/Deletion (10 files)

### Severely Corrupted Files (DELETE or RECREATE)

These files have no newlines and are completely malformed. They appear to be incomplete or corrupted copies.

1. **ace_mcp_tools_FIXED.py**
   - Issue: Entire file is on one line (no newlines)
   - Line count: 1 line (should be ~260+ lines)
   - Recommendation: **DELETE** - File appears to be corrupted copy
   - Action: Use original `ace_mcp_tools.py` instead

2. **demo_mcts_mdap.py**
   - Issue: Multiple f-string backslash errors
   - Lines: 137, 155, 167, 239, 314, 335, 349, 410, 423, 444, 494, 495, 581, 604
   - Recommendation: **MANUAL FIX** required
   - Fix: Replace `\n` in f-strings with `\\n` or use separate strings

3. **leanaide_mdap_demo.py**
   - Issue: Unterminated string literals
   - Lines: 44, 45, 56, 66, 81, 99
   - Recommendation: **MANUAL FIX** required
   - Fix: Add closing quotes to unterminated strings

### Missing Except Blocks (4 files)

4. **adversarial_adapter.py**
   - Issue: Expected 'except' or 'finally' block at line 355
   - Status: Could not auto-fix - manual review needed
   - Fix: Add except block after try block

5. **bubblelabs_evolution_integration.py**
   - Issue: Expected 'except' or 'finally' block at line 449
   - Status: Could not auto-fix - manual review needed
   - Fix: Add except block after try block

### Await Outside Async (2 files)

6. **adversarial_error_handling.py**
   - Issue: 'await' outside async function at line 778
   - Status: Could not auto-fix - manual review needed
   - Fix: Either make function async or remove await

7. **hybrid_error_handling.py**
   - Issue: 'await' outside async function at line 297
   - Status: Could not auto-fix - manual review needed
   - Fix: Either make function async or remove await

### Generic Invalid Syntax (3 files)

8. **leanaide_sop_integration.py**
   - Issue: Invalid syntax at line 162
   - Status: Could not auto-fix - manual review needed

9. **openevolve_leanaide_bridge.py**
   - Issue: Invalid syntax at line 483
   - Status: Could not auto-fix - manual review needed

10. **workflow_stage_functions.py**
    - Issue: Unterminated string literal at line 90
    - Status: No auto-fix needed but still has syntax error
    - Fix: Check for unclosed triple quotes or multiline strings

---

## Recommendations

### Option 1: Delete Corrupted Files
```bash
# Delete files that are beyond repair
rm ace_mcp_tools_FIXED.py
# Use ace_mcp_tools.py instead (if it exists)
```

### Option 2: Manual Fixes Required
For the remaining files, you have two options:

1. **Fix manually** - Open each file and fix the specific syntax errors
2. **Delete unused files** - If these are test/demo files that aren't used

### Option 3: Use a Python Formatter
```bash
# Use black to auto-format Python files
pip install black
black --line-length=120 *.py

# Or use autopep8
pip install autopep8
autopep8 --in-place --aggressive *.py
```

---

## Next Steps

1. **Delete ace_mcp_tools_FIXED.py** (corrupted, use original)
2. **Fix demo files manually** or delete if unused
3. **Fix integration files** (adversarial_adapter, bubblelabs_evolution_integration)
4. **Run auto-fix for other issues** (bare except, pickle usage, etc.)

---

## Files That Can Proceed to Auto-Fix

✅ The remaining 585 files (597 - 12 syntax errors) can now be processed by the auto-fix script for:
- Bare except clauses
- Hardcoded /tmp paths
- Pickle usage documentation

---

**Generated:** 2026-01-20
**Backups created:** .syntax_backup files for modified files
**Status:** 2/12 fixed, 10 require manual review
