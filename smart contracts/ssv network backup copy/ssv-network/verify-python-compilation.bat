@echo off
REM Python POC Compilation Verification Script (Windows)
REM This script verifies that all Python POCs compile successfully

echo ==========================================
echo Python POC Compilation Verification
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "scripts" (
    echo Error: Please run this script from the ssv-network directory
    exit /b 1
)

echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
python --version
echo.

echo Step 2: Compiling Python POC files...
echo.

set "all_passed=true"

REM Compile POC 1
echo Checking: scripts/poc1_single_cluster_actual_protocol.py
python -m py_compile scripts/poc1_single_cluster_actual_protocol.py 2>&1
if %errorlevel% equ 0 (
    echo   Compiles successfully
) else (
    echo   Compilation failed
    set "all_passed=false"
)
echo.

REM Compile POC 2
echo Checking: scripts/poc2_multi_cluster_actual_protocol.py
python -m py_compile scripts/poc2_multi_cluster_actual_protocol.py 2>&1
if %errorlevel% equ 0 (
    echo   Compiles successfully
) else (
    echo   Compilation failed
    set "all_passed=false"
)
echo.

REM Compile POC 3
echo Checking: scripts/poc3_liquidation_griefing_actual_protocol.py
python -m py_compile scripts/poc3_liquidation_griefing_actual_protocol.py 2>&1
if %errorlevel% equ 0 (
    echo   Compiles successfully
) else (
    echo   Compilation failed
    set "all_passed=false"
)
echo.

REM Compile POC 4
echo Checking: scripts/poc4_dao_sybil_actual_protocol.py
python -m py_compile scripts/poc4_dao_sybil_actual_protocol.py 2>&1
if %errorlevel% equ 0 (
    echo   Compiles successfully
) else (
    echo   Compilation failed
    set "all_passed=false"
)
echo.

REM Compile POC 5
echo Checking: scripts/poc5_operator_sybil_actual_protocol.py
python -m py_compile scripts/poc5_operator_sybil_actual_protocol.py 2>&1
if %errorlevel% equ 0 (
    echo   Compiles successfully
) else (
    echo   Compilation failed
    set "all_passed=false"
)
echo.

echo ==========================================
if "%all_passed%"=="true" (
    echo ALL PYTHON POCS COMPILE SUCCESSFULLY
    echo ==========================================
    echo.
    echo All Python POC files are syntactically correct.
    echo They use web3.py to interact with actual SSV Network
    echo protocol via local Hardhat fork.
    echo.
    echo To run Python POCs:
    echo   1. Start Hardhat node: npx hardhat node --fork MAINNET_RPC
    echo   2. Run POC: python scripts/poc1_single_cluster_actual_protocol.py
    exit /b 0
) else (
    echo SOME PYTHON POCS FAILED TO COMPILE
    echo ==========================================
    exit /b 1
)
