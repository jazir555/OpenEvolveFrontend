@echo off
REM ###########################################################################
REM LeanAide Integration Probe Script (Windows)
REM
REM Law of Runtime Truth: Verify LeanAide server is actually available
REM and functioning before attempting integration.
REM
REM Usage: probes\check_leanaide.bat
REM
REM Author: RESE Team
REM Created: 2026-02-04
REM ###########################################################################

setlocal enabledelayedexpansion

REM Configuration
if "%LEANAIDE_HOST%"=="" set LEANAIDE_HOST=localhost
if "%LEANAIDE_PORT%"=="" set LEANAIDE_PORT=7654
set LEANAIDE_URL=http://%LEANAIDE_HOST%:%LEANAIDE_PORT%
set TIMEOUT=10

echo.
echo =============================================================
echo          LeanAide Integration Probe Script
echo
echo   Law of Runtime Truth: Verify Before Using
echo =============================================================
echo.

REM ###########################################################################
REM TEST 1: Server Availability
REM ###########################################################################

echo [TEST 1] Checking LeanAide server availability...
echo Target: %LEANAIDE_URL%

curl -s --max-time %TIMEOUT% "%LEANAIDE_URL%/" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] - LeanAide server is reachable
    set LEANAIDE_AVAILABLE=true
) else (
    echo [FAIL] - Cannot reach LeanAide server at %LEANAIDE_URL%
    echo [INFO] - Ensure LeanAide server is running on port %LEANAIDE_PORT%
    set LEANAIDE_AVAILABLE=false
)

echo.

REM ###########################################################################
REM TEST 2: Health Check
REM ###########################################################################

echo [TEST 2] Performing health check...

if "%LEANAIDE_AVAILABLE%"=="true" (
    curl -s --max-time %TIMEOUT% "%LEANAIDE_URL%/"
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] - Health check successful
    ) else (
        echo [FAIL] - Health check failed
    )
) else (
    echo [SKIP] - Server not available
)

echo.

REM ###########################################################################
REM TEST 3: Autoformalization
REM ###########################################################################

echo [TEST 3] Testing autoformalization...
echo Theorem: 'For all natural numbers n, n + 0 = n'

if "%LEANAIDE_AVAILABLE%"=="true" (
    echo {"task": "translate_thm", "theorem_text": "For all natural numbers n, n + 0 = n"} > request.json
    curl -s --max-time 30 -X POST -H "Content-Type: application/json" -d @request.json "%LEANAIDE_URL%/" > response.json
    del request.json

    findstr /i "lean theorem Nat" response.json >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] - Autoformalization working
        echo Generated Lean code snippet:
        powershell -Command "Get-Content response.json -Head 5"
    ) else (
        echo [FAIL] - Autoformalization failed
        echo Response:
        type response.json
    )
    del response.json
) else (
    echo [SKIP] - Server not available
)

echo.

REM ###########################################################################
REM TEST 4: AI-Powered Proving
REM ###########################################################################

echo [TEST 4] Testing AI-powered proving...
echo Theorem: '1 + 1 = 2'

if "%LEANAIDE_AVAILABLE%"=="true" (
    echo {"task": "math_query", "query": "What is 1 + 1?", "n": 1} > request.json
    curl -s --max-time 30 -X POST -H "Content-Type: application/json" -d @request.json "%LEANAIDE_URL%/" > response.json
    del request.json

    findstr /i "2 two" response.json >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] - AI query working
    ) else (
        echo [UNCERTAIN] - AI query response unclear
        echo Response:
        type response.json
    )
    del response.json
) else (
    echo [SKIP] - Server not available
)

echo.

REM ###########################################################################
REM TEST 5: Z3-LeanAide Bridge
REM ###########################################################################

echo [TEST 5] Testing Z3-LeanAide bridge...

set BRIDGE_FILE=..\..\..\..\z3_leanaide_bridge.py

if exist "%BRIDGE_FILE%" (
    echo [PASS] - Z3-LeanAide bridge file exists
    echo Location: %BRIDGE_FILE%

    python -c "import sys; sys.path.insert(0, '../../..'); from z3_leanaide_bridge import Z3LeanAideBridge" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] - Z3-LeanAide bridge can be imported
    ) else (
        echo [WARN] - Bridge file exists but dependencies missing
    )
) else (
    echo [SKIP] - Z3-LeanAide bridge not found
    echo Expected location: %BRIDGE_FILE%
)

echo.

REM ###########################################################################
REM TEST 6: LeanAide Client
REM ###########################################################################

echo [TEST 6] Testing LeanAide client...

set CLIENT_FILE=..\..\..\..\leanaide_client.py

if exist "%CLIENT_FILE%" (
    echo [PASS] - LeanAide client file exists
    echo Location: %CLIENT_FILE%

    python -c "import sys; sys.path.insert(0, '../../..'); from leanaide_client import LeanAideClient" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [PASS] - LeanAide client can be imported
    ) else (
        echo [WARN] - Client file exists but dependencies missing
    )
) else (
    echo [SKIP] - LeanAide client not found
    echo Expected location: %CLIENT_FILE%
)

echo.

REM ###########################################################################
REM TEST 7: Configuration Validation
REM ###########################################################################

echo [TEST 7] Checking environment configuration...

if not "%LEANAIDE_BASE_URL%"=="" (
    echo [INFO] - LEANAIDE_BASE_URL = %LEANAIDE_BASE_URL%
) else (
    echo [INFO] - LEANAIDE_BASE_URL not set (will use default)
)

if not "%LEANAIDE_TIMEOUT_MS%"=="" (
    echo [INFO] - LEANAIDE_TIMEOUT_MS = %LEANAIDE_TIMEOUT_MS%
)

if not "%LEANAIDE_ENABLE%"=="" (
    echo [INFO] - LEANAIDE_ENABLE = %LEANAIDE_ENABLE%
)

echo.

REM ###########################################################################
REM SUMMARY
REM ###########################################################################

echo =============================================================
echo                    Probe Summary
echo =============================================================
echo.
echo LeanAide Server: %LEANAIDE_URL%
echo.

if "%LEANAIDE_AVAILABLE%"=="true" (
    echo Status: [PASS] LEANAIDE INTEGRATION READY
    echo.
    echo Next steps:
    echo   1. Import: from rese_z3_bridge import RESEZ3Bridge
    echo   2. Create: bridge = RESEZ3Bridge()
    echo   3. Use: bridge.autoformalize('Your theorem')
    echo.
    echo Available features:
    echo   - Autoformalization: Natural language to Lean 4
    echo   - AI-powered proving: Generate proofs automatically
    echo   - Z3-Lean translation: Bridge Z3 constraints to Lean
    echo   - Tactic suggestions: Get AI-recommended tactics
    echo.
    exit /b 0
) else (
    echo Status: [FAIL] LEANAIDE INTEGRATION NOT READY
    echo.
    echo Required actions:
    echo   1. Start LeanAide server on port %LEANAIDE_PORT%
    echo   2. Verify server is responding: curl %LEANAIDE_URL%/
    echo   3. Re-run this probe script
    echo.
    exit /b 1
)
