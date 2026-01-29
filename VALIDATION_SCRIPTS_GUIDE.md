# Migration Validation Scripts - Quick Guide

This guide explains how to use the 4 validation scripts created for verifying the OpenEvolve import migration.

---

## Scripts Overview

### 1. `test_import_functionality.py`
**Purpose:** Test that openevolve_imports module works correctly

**What it checks:**
- All imports can be loaded without errors
- Availability flags are properly set (booleans)
- API classes are accessible (EvolutionAPI, AdversarialAPI, etc.)
- Convenience functions work correctly
- Error handling works as expected

**Usage:**
```bash
python test_import_functionality.py
```

**Sample Output:**
```
================================================================================
OPENEVOLVE_IMPORTS COMPREHENSIVE TEST SUITE
================================================================================

Test 1: Importing openevolve_imports...
  ✓ All imports successful

Test 2: Checking availability flags...
  ✓ EVOLUTION_AVAILABLE = True
  ✓ ADVERSARIAL_AVAILABLE = True
  ✓ PARAMETER_MANAGER_AVAILABLE = True

...

✓ ALL TESTS PASSED - openevolve_imports is working correctly!
```

---

### 2. `validate_batch1_imports.py`
**Purpose:** Verify Batch 1 import replacements work correctly

**What it checks:**
- Files properly import from openevolve_imports
- No old try/except import patterns remain
- Import availability flags are used correctly
- Detects files that still need migration

**Usage:**
```bash
python validate_batch1_imports.py
```

**Sample Output:**
```
================================================================================
BATCH 1 IMPORT VALIDATION REPORT
================================================================================

Total files checked: 35
Valid files: 11
Files using openevolve_imports: 2
Files with old patterns: 23

--------------------------------------------------------------------------------
DETAILED RESULTS:
--------------------------------------------------------------------------------

test_adversarial_comprehensive.py
  Status: ✓ VALID
  Uses openevolve_imports: ✗
  Has old patterns: ✓
  Has availability checks: ✗
  Lines: 326
  Warnings:
    - File doesn't use openevolve_imports or old patterns
```

---

### 3. `validate_syntax.py`
**Purpose:** Check Python syntax of all updated files

**What it checks:**
- Python syntax is valid
- Files can be compiled without errors
- No indentation errors
- No token errors

**Usage:**
```bash
# Validate all Python files
python validate_syntax.py

# Validate specific patterns
python validate_syntax.py "test_*.py"
python validate_syntax.py "*_integration.py"

# Show verbose output
python validate_syntax.py --verbose

# Specify directory
python validate_syntax.py -d /path/to/code
```

**Sample Output:**
```
Found 146 file(s) to validate
================================================================================
PYTHON SYNTAX VALIDATION REPORT
================================================================================

Total files checked: 146
Valid files: 140
Can compile: 143
Invalid files: 6

--------------------------------------------------------------------------------
FAILED FILES:
--------------------------------------------------------------------------------

✗ test_ace_edge_cases.py
  Path: test_ace_edge_cases.py
  Lines: 496
  Can compile: ✗
  Can parse: ✗
  Errors:
    - Compilation error: SyntaxError: unterminated string literal (line 265)
```

---

### 4. `migration_report.py`
**Purpose:** Generate migration progress report

**What it generates:**
- Executive summary with total files, completion percentage
- Batch-by-batch progress breakdown
- Detailed file status tables
- Progress bars for visual tracking
- Recommendations and next steps

**Usage:**
```bash
# Generate report with default filename (MIGRATION_REPORT.md)
python migration_report.py

# Specify custom output file
python migration_report.py MY_REPORT.md

# View the report
cat MIGRATION_REPORT.md
```

**Sample Output:**
```
# Migration Progress Report

**Generated:** 2026-01-03 17:36:55

---

## Executive Summary

- **Total Files:** 20
- **Completed:** 0 (0.0%)
- **In Progress:** 0
- **Total Lines:** 7,507

### Overall Progress
░░░░░░░░░░ 0%

---

## Batch 1: Import Replacements
...
```

---

## Typical Workflow

### Step 1: Initial Assessment
```bash
# Test that openevolve_imports module works
python test_import_functionality.py

# Generate initial migration report
python migration_report.py INITIAL_REPORT.md
```

### Step 2: Apply Migration Changes
Make your code changes to replace old import patterns with openevolve_imports.

### Step 3: Validate Changes
```bash
# Check syntax of modified files
python validate_syntax.py "test_*.py"

# Validate import replacements
python validate_batch1_imports.py
```

### Step 4: Generate Updated Report
```bash
# Generate final report
python migration_report.py FINAL_REPORT.md
```

---

## Exit Codes

All scripts return appropriate exit codes for CI/CD integration:

- **Exit code 0:** All checks passed
- **Exit code 1:** Some checks failed

**Usage in CI/CD:**
```bash
python test_import_functionality.py && echo "Imports OK" || echo "Imports FAILED"
python validate_syntax.py && echo "Syntax OK" || echo "Syntax FAILED"
```

---

## Troubleshooting

### Unicode Encoding Issues (Windows)
If you see encoding errors on Windows, the scripts automatically handle UTF-8 encoding. If you still have issues:

```bash
# Set Python to use UTF-8
set PYTHONIOENCODING=utf-8
python test_import_functionality.py
```

### Import Errors
If `openevolve_imports` module cannot be imported:

```bash
# Ensure you're in the correct directory
cd /path/to/openevolve/frontend

# Verify the module exists
ls openevolve_imports.py

# Test basic import
python -c "import openevolve_imports; print('OK')"
```

### Large File Sets
When validating many files, the scripts show progress dots:

```bash
python validate_syntax.py "*.py"
..................................  # Progress dots

Found 146 file(s) to validate
```

---

## Integration with Development Workflow

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running migration validation..."

python test_import_functionality.py
if [ $? -ne 0 ]; then
    echo "Import functionality test failed"
    exit 1
fi

python validate_syntax.py
if [ $? -ne 0 ]; then
    echo "Syntax validation failed"
    exit 1
fi

echo "All validation checks passed"
```

### GitHub Actions
```yaml
name: Validate Migration

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Test imports
        run: python test_import_functionality.py

      - name: Validate syntax
        run: python validate_syntax.py

      - name: Generate report
        run: python migration_report.py

      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: migration-report
          path: MIGRATION_REPORT.md
```

---

## Customization

### Modifying Batch Definitions
Edit `migration_report.py` to change which files are tracked:

```python
BATCHES = {
    'batch1': {
        'name': 'Batch 1: Import Replacements',
        'description': 'Replace all try/except import patterns with openevolve_imports',
        'files': [
            'test_adversarial_comprehensive.py',
            # Add your files here
        ],
    },
}
```

### Adding New Validation Patterns
Edit `validate_batch1_imports.py` to add new patterns:

```python
OLD_PATTERNS = [
    r'try:\s*from\s+evolution\s+import',
    # Add your patterns here
]
```

---

## Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_import_functionality.py` | Test openevolve_imports works | First, after installing |
| `validate_batch1_imports.py` | Check import replacements | After making changes |
| `validate_syntax.py` | Check Python syntax | Before committing |
| `migration_report.py` | Generate progress report | Before/after migration |

---

**Last Updated:** 2026-01-03
