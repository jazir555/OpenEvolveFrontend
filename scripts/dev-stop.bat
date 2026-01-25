@echo off
REM =============================================================================
REM OpenEvolve Development Environment Stop Script (Windows)
REM =============================================================================
REM This script stops all infrastructure services gracefully.
REM
REM Usage:
REM   scripts\dev-stop.bat [--volumes]
REM
REM Options:
REM   --volumes    Also remove named volumes (WARNING: This deletes data!)
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuration
set COMPOSE_FILE=docker-compose.infrastructure.yml
set ENV_FILE=.env.infrastructure
set COMPOSE_PROJECT_NAME=openevolve

REM Parse arguments
set REMOVE_VOLUMES=false

for %%i in (%*) do (
    if "%%i"=="--volumes" set REMOVE_VOLUMES=true
)

REM Main execution
:main
    echo.
    echo [INFO] Stopping OpenEvolve Infrastructure Services
    echo.

    REM Determine compose command
    docker compose version >nul 2>&1
    if errorlevel 1 (
        set COMPOSE_CMD=docker-compose
    ) else (
        set COMPOSE_CMD=docker compose
    )

    set COMPOSE_ARGS=-f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% --env-file %ENV_FILE%

    if "%REMOVE_VOLUMES%"=="true" (
        echo [WARNING] This will delete all data in the volumes!
        set /p confirm="Are you sure? (yes/no): "
        if not "!confirm!"=="yes" (
            echo [INFO] Aborted
            exit /b 0
        )
        set COMPOSE_ARGS=%COMPOSE_ARGS% -v
    )

    %COMPOSE_CMD% %COMPOSE_ARGS% down

    if "%REMOVE_VOLUMES%"=="true" (
        echo [SUCCESS] Services stopped and volumes removed
    ) else (
        echo [SUCCESS] Services stopped (volumes preserved)
    )

    echo.
    echo [INFO] To start services again, run: scripts\dev-start.bat
    echo.

goto :eof

call :main
