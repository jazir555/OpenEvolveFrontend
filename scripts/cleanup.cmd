@echo off
REM =============================================================================
REM OpenEvolve Cleanup Script (Windows)
REM License: Apache 2.0
REM Description: Stop services and clean up
REM =============================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

echo ===============================================================================
echo                     OpenEvolve Cleanup
echo ===============================================================================
echo.

cd /d "%PROJECT_ROOT%"

echo Step 1: Stopping all services...
docker compose down
echo Services stopped.
echo.

echo Step 2: Removing containers...
docker compose rm -f -v 2>nul
echo Containers removed.
echo.

echo Step 3: Cleaning build artifacts...
for /d /r "%PROJECT_ROOT%\glue" %%d in (node_modules) do @if exist "%%d" rd /s /q "%%d"
for /d /r "%PROJECT_ROOT%\glue" %%d in (dist) do @if exist "%%d" rd /s /q "%%d"
echo Build artifacts cleaned.
echo.

echo Cleanup complete!
echo.
echo To restart services, run: scripts\quick-start.cmd
echo.
pause
