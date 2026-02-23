@echo off
REM =============================================================================
REM OpenEvolve Smoke Test Script (Windows)
REM License: Apache 2.0
REM Description: Run basic smoke tests after deployment
REM =============================================================================

setlocal EnableDelayedExpansion

set "BASE_URL=http://localhost:8080"
set "TIMEOUT=30"
set "TESTS_PASSED=0"
set "TESTS_FAILED=0"

echo ===============================================================================
echo                     OpenEvolve Smoke Tests
echo ===============================================================================
echo.
echo Base URL: %BASE_URL%
echo Timeout: %TIMEOUT%s
echo.
echo ===============================================================================
echo.

cd /d "%~dp0.."

REM Test 1: Health Endpoint
echo Test 1: Health Endpoint
curl -s -o nul -w "HTTP %%{http_code}\n" %BASE_URL%/health --max-time %TIMEOUT%
if %errorlevel% equ 0 (
    echo [PASS]
    set /a TESTS_PASSED+=1
) else (
    echo [FAIL]
    set /a TESTS_FAILED+=1
)
echo.

REM Test 2: API Readiness
echo Test 2: API Readiness
curl -s -o nul -w "HTTP %%{http_code}\n" %BASE_URL%/api/v1/ready --max-time %TIMEOUT%
if %errorlevel% equ 0 (
    echo [PASS]
    set /a TESTS_PASSED+=1
) else (
    echo [FAIL]
    set /a TESTS_FAILED+=1
)
echo.

REM Test 3: Event Bus
echo Test 3: Event Bus Connection
curl -s -o nul -w "HTTP %%{http_code}\n" %BASE_URL%/api/v1/status/eventbus --max-time %TIMEOUT%
if %errorlevel% equ 0 (
    echo [PASS]
    set /a TESTS_PASSED+=1
) else (
    echo [FAIL]
    set /a TESTS_FAILED+=1
)
echo.

REM Test 4: Workflow Engine
echo Test 4: Workflow Engine
curl -s -o nul -w "HTTP %%{http_code}\n" %BASE_URL%/api/v1/workflows/health --max-time %TIMEOUT%
if %errorlevel% equ 0 (
    echo [PASS]
    set /a TESTS_PASSED+=1
) else (
    echo [FAIL]
    set /a TESTS_FAILED+=1
)
echo.

REM Summary
echo ===============================================================================
echo Smoke Test Summary
echo ===============================================================================
set /a TOTAL=%TESTS_PASSED%+%TESTS_FAILED%
echo Total Tests:  %TOTAL%
echo Passed:       %TESTS_PASSED%
echo Failed:       %TESTS_FAILED%
echo ===============================================================================
echo.

if %TESTS_FAILED% equ 0 (
    echo All smoke tests passed!
) else (
    echo Some smoke tests failed!
)

echo.
pause
