@echo off
REM =============================================================================
REM OpenEvolve Health Check Script (Windows)
REM License: Apache 2.0
REM Description: Check health status of all services
REM =============================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

echo ===============================================================================
echo                     OpenEvolve Health Check
echo ===============================================================================
echo.

cd /d "%PROJECT_ROOT%"

echo Service Status:
echo.
echo Service                        Status          Response
echo ------------------------------ --------------- --------------------

REM Check OpenEvolve API
curl -s -o nul -w "openevolve-app          " http://localhost:8080/health --max-time 3 2>nul
if %errorlevel% equ 0 (
    echo healthy          HTTP 200
) else (
    echo unreachable      Connection failed
)

REM Check Prometheus
curl -s -o nul -w "openevolve-prometheus    " http://localhost:9090/-/healthy --max-time 3 2>nul
if %errorlevel% equ 0 (
    echo healthy          HTTP 200
) else (
    echo unreachable      Connection failed
)

REM Check Grafana
curl -s -o nul -w "openevolve-grafana       " http://localhost:3000/api/health --max-time 3 2>nul
if %errorlevel% equ 0 (
    echo healthy          HTTP 200
) else (
    echo unreachable      Connection failed
)

echo.
echo Docker Containers:
docker compose ps
echo.

pause
