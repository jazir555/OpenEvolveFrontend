@echo off
REM =============================================================================
REM OpenEvolve Validation Script (Windows)
REM License: Apache 2.0
REM Description: Validates service health and configuration
REM =============================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "LOG_DIR=%PROJECT_ROOT%\logs"

REM Create logs directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "REPORT_FILE=%LOG_DIR%\validation-report-%date:~10,4%%date:~4,2%%date:~7,2%.md"

echo ===============================================================================
echo                     OpenEvolve Health Validation
echo ===============================================================================
echo.

cd /d "%PROJECT_ROOT%"

echo Checking services...
docker compose ps
echo.

echo Checking health endpoints...
echo.

REM Check OpenEvolve API
curl -s -o nul -w "OpenEvolve API: %%{http_code}\n" http://localhost:8080/health --max-time 5
curl -s -o nul -w "Orchestrator: %%{http_code}\n" http://localhost:8080/health --max-time 5
curl -s -o nul -w "Prometheus: %%{http_code}\n" http://localhost:9090/-/healthy --max-time 5

echo.
echo Validation complete.
echo.
pause
