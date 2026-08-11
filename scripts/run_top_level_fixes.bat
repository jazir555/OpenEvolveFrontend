@echo off
REM ========================================
REM Top-Level Security Auto-Fix Runner
REM ========================================

echo.
echo ========================================
echo Top-Level Security Auto-Fix Tool
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo This tool will fix security issues in top-level Python files only.
echo.
echo Issues to fix:
echo   - 130 bare except clauses
echo   - 11 hardcoded /tmp paths
echo   - Documents 51 pickle usages
echo   - Attempts to fix 12 syntax errors
echo.

echo Step 1: DRY RUN - Preview changes without applying
echo ---------------------------------------------------
python auto_fix_top_level.py --dry-run --verbose
echo.

echo.
echo Step 2: Review the dry-run output above
echo.
echo If you want to apply the fixes, run:
echo   python auto_fix_top_level.py --verbose
echo.

pause
