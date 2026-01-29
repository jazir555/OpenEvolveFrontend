@echo off
REM ========================================
REM OpenEvolve-BubbleLab Security Fix Runner
REM ========================================

echo.
echo ========================================
echo OpenEvolve-BubbleLab Security Fix Tool
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo Step 1: Analyzing security issues (dry-run mode)...
echo ---------------------------------------------------
python auto_fix_security.py --dry-run --analyze-only
echo.

echo Step 2: Generating manual fix report...
echo ---------------------------------------------------
python fix_manual_security_issues.py --target-dir . --generate-patches
echo.

echo Step 3: Ready to apply automatic fixes
echo ---------------------------------------------------
echo.
echo The following files are ready:
echo   1. security_analysis_*.json - Detailed analysis
echo   2. MANUAL_SECURITY_FIXES_*.md - Manual fix instructions
echo   3. security_patches/ - Individual patch files
echo.
echo To apply automatic fixes:
echo   python auto_fix_security.py --verbose
echo.
echo To do a dry-run first:
echo   python auto_fix_security.py --dry-run --verbose
echo.

pause
