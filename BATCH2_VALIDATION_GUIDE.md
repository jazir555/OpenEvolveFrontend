# Batch 2 Validation Scripts - Usage Guide

This document explains how to use the three validation scripts created for Batch 2 adapter integration.

## Scripts Overview

### 1. `validate_batch2_adapters.py`
Validates that Batch 2 files correctly use evolution_adapter and adversarial_adapter instead of direct module calls.

**Usage:**
```bash
python validate_batch2_adapters.py
```

**What it checks:**
- Import statements from `evolution_adapter` and `adversarial_adapter`
- Adapter instantiation patterns
- Direct imports from evolution/adversarial modules
- Direct calls to evolution/adversarial modules
- Provides a summary of adapter adoption rate

**Exit codes:**
- 0: All checks passed
- 1: Issues found

### 2. `test_adapter_functionality.py`
Tests that adapters work correctly with evolution and adversarial modules.

**Usage:**
```bash
python test_adapter_functionality.py
```

**What it tests:**
- EvolutionAdapter import and instantiation
- AdversarialAdapter import and instantiation
- Module interoperability
- Basic performance timing

**Exit codes:**
- 0: All tests passed
- 1: Some tests failed

### 3. `compare_before_after.py`
Compares performance and results between old direct calls and new adapter calls.

**Usage:**
```bash
python compare_before_after.py
```

**What it compares:**
- Direct calls to evolution/adversarial modules vs adapter calls
- Performance overhead measurement
- Memory usage comparison
- Provides recommendations based on results

## Running All Scripts

To run all validation scripts in sequence:

```bash
echo "Running Batch 2 validation..."
python validate_batch2_adapters.py
echo "Testing adapter functionality..."
python test_adapter_functionality.py
echo "Comparing performance..."
python compare_before_after.py
echo "Validation complete!"
```

## Expected Output Examples

### Validation Script Output
```
VALIDATING BATCH 2 ADAPTER INTEGRATION...
============================================================
[FAIL] app.py: No adapter usage detected
   [INFO] Consider adding adapter usage

[OK] integrated_workflow.py: Uses adapters
   Patterns: create_adversarial_adapter(, create_evolution_adapter(

[FAIL] evolution.py: No adapter usage detected
   [INFO] Consider adding adapter usage

============================================================
SUMMARY
============================================================
Total files checked: 10
Files using adapters: 1
Files with issues: 0
Adapter adoption rate: 10.0%
```

### Test Script Output
```
STARTING ADAPTER FUNCTIONALITY TESTS
============================================================

========================================
TESTING EVOLUTION ADAPTER...
  [OK] Successfully imported create_evolution_adapter
  [OK] Successfully created EvolutionAdapter
  [OK] Adapter has 'run' method
  [OK] Adapter has 'analyze' method
[SUCCESS] EvolutionAdapter works correctly

[... additional test output ...]

TEST RESULTS SUMMARY
============================================================
EvolutionAdapter    : [OK] PASS
AdversarialAdapter  : [OK] PASS
Interoperability    : [OK] PASS

Overall: 3/3 tests passed
[SUCCESS] All adapter tests passed!
```

### Comparison Script Output
```
BEFORE vs AFTER ADAPTER PERFORMANCE COMPARISON
======================================================================

This script compares performance between:
  - Direct calls to evolution/adversarial modules
  - Adapter calls through evolution_adapter/adversarial_adapter

RUNNING COMPREHENSIVE PERFORMANCE COMPARISON
======================================================================

TESTING WITH 2 ITERATIONS/ROUNDS:
--------------------------------------------------
  Evolution:
    Direct call:    0.000123s
    Adapter call:   0.000145s
    Overhead:       17.89%

  Adversarial:
    Direct call:    0.000098s
    Adapter call:   0.000112s
    Overhead:       14.29%

[... additional comparison output ...]

RECOMMENDATIONS:
  [OK] Adapter performance is good (under 25% overhead)
  [OK] Adapters provide better modularity and maintainability
  [OK] Adapters enable better error handling and logging
  [OK] Adapters facilitate easier future integration changes
```

## Troubleshooting

### Common Issues

1. **Import Errors**: If adapters are not found, ensure they exist in the current directory
2. **Configuration Errors**: Some adapters may require specific configuration
3. **Unicode Issues**: Scripts use ASCII-compatible output to avoid encoding problems

### Getting Help

- Check script output for specific error messages
- Ensure all required dependencies are installed
- Verify adapter files exist in the expected location

## Integration with CI/CD

These scripts can be integrated into your CI/CD pipeline:

```yaml
# Example CI configuration
steps:
  - name: Validate Batch 2 Adapters
    run: python validate_batch2_adapters.py

  - name: Test Adapter Functionality
    run: python test_adapter_functionality.py

  - name: Performance Comparison
    run: python compare_before_after.py
```

## Best Practices

1. Run validation scripts before committing changes
2. Monitor adapter adoption rate over time
3. Use performance comparison to identify optimization opportunities
4. Keep adapter usage consistent across the codebase