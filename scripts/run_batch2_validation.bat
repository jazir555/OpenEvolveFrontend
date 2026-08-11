@echo off
REM Batch 2 Validation Scripts Runner
REM This script runs all Batch 2 validation scripts

echo ================================================
echo BATCH 2 VALIDATION SUITE
echo ================================================
echo.

echo [1/3] Running Batch 2 Adapter Validation...
python validate_batch2_adapters.py
echo Validation complete.

echo.
echo [2/3] Running Adapter Functionality Tests...
python test_adapter_functionality.py
echo Testing complete.

echo.
echo [3/3] Running Performance Comparison...
python compare_before_after.py
echo Comparison complete.

echo.
echo ================================================
echo ALL BATCH 2 VALIDATIONS COMPLETED
echo ================================================

pause