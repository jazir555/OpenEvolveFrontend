#!/bin/bash
# Run all validation scripts and generate comprehensive report

echo "=========================================================================="
echo "OPENEVOLVE MIGRATION VALIDATION - COMPREHENSIVE TEST SUITE"
echo "=========================================================================="
echo ""

# Create results directory
mkdir -p validation_results

# Track overall success
OVERALL_SUCCESS=true

# Test 1: Import functionality
echo "=========================================================================="
echo "TEST 1: Import Functionality"
echo "=========================================================================="
python test_import_functionality.py > validation_results/test1_import_functionality.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ PASSED - Results saved to validation_results/test1_import_functionality.log"
    echo ""
else
    echo "✗ FAILED - Check validation_results/test1_import_functionality.log"
    echo ""
    OVERALL_SUCCESS=false
fi

# Test 2: Batch1 imports validation
echo "=========================================================================="
echo "TEST 2: Batch 1 Import Validation"
echo "=========================================================================="
python validate_batch1_imports.py > validation_results/test2_batch1_validation.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ PASSED - Results saved to validation_results/test2_batch1_validation.log"
    echo ""
else
    echo "✗ FAILED - Check validation_results/test2_batch1_validation.log"
    echo ""
    OVERALL_SUCCESS=false
fi

# Test 3: Syntax validation
echo "=========================================================================="
echo "TEST 3: Python Syntax Validation (sample files)"
echo "=========================================================================="
python validate_syntax.py "test_*.py" > validation_results/test3_syntax_validation.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ PASSED - Results saved to validation_results/test3_syntax_validation.log"
    echo ""
else
    echo "✗ FAILED - Check validation_results/test3_syntax_validation.log"
    echo ""
    OVERALL_SUCCESS=false
fi

# Test 4: Migration report
echo "=========================================================================="
echo "TEST 4: Migration Progress Report"
echo "=========================================================================="
python migration_report.py validation_results/MIGRATION_REPORT.md > validation_results/test4_migration_report.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ PASSED - Report saved to validation_results/MIGRATION_REPORT.md"
    echo ""
else
    echo "✗ FAILED - Check validation_results/test4_migration_report.log"
    echo ""
    OVERALL_SUCCESS=false
fi

# Summary
echo "=========================================================================="
echo "VALIDATION TEST SUMMARY"
echo "=========================================================================="
echo ""
echo "Results directory: validation_results/"
echo ""
echo "Test files generated:"
echo "  - test1_import_functionality.log"
echo "  - test2_batch1_validation.log"
echo "  - test3_syntax_validation.log"
echo "  - test4_migration_report.log"
echo "  - MIGRATION_REPORT.md"
echo ""

if [ "$OVERALL_SUCCESS" = true ]; then
    echo "✓ ALL VALIDATION TESTS PASSED"
    echo "=========================================================================="
    exit 0
else
    echo "✗ SOME VALIDATION TESTS FAILED - Review logs above"
    echo "=========================================================================="
    exit 1
fi
