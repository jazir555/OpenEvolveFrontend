@echo off
REM =============================================================================
REM OpenEvolve Development Environment Startup Script (Windows)
REM =============================================================================
REM This script starts all infrastructure services in the correct order with
REM health checks to ensure everything is ready before proceeding.
REM
REM Usage:
REM   scripts\dev-start.bat [--with-tools] [--skip-health-check]
REM
REM Options:
REM   --with-tools    Start optional management UIs (pgAdmin, Redis Commander)
REM   --skip-health-check  Skip health checks (not recommended)
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuration
set COMPOSE_FILE=docker-compose.infrastructure.yml
set ENV_FILE=.env.infrastructure
set COMPOSE_PROJECT_NAME=openevolve

REM Parse arguments
set WITH_TOOLS=false
set SKIP_HEALTH_CHECK=false

for %%i in (%*) do (
    if "%%i"=="--with-tools" set WITH_TOOLS=true
    if "%%i"=="--skip-health-check" set SKIP_HEALTH_CHECK=true
)

REM Functions (simulated with labels and subroutines)
:main
    echo.
    echo [INFO] OpenEvolve Infrastructure Setup
    echo.

    call :check_docker
    if errorlevel 1 exit /b 1

    call :check_env_file
    if errorlevel 1 exit /b 1

    call :start_services
    if errorlevel 1 exit /b 1

    if "%SKIP_HEALTH_CHECK%"=="false" (
        timeout /t 5 /nobreak >nul
        call :run_health_checks
        if errorlevel 1 exit /b 1
    )

    call :print_service_info

    echo [SUCCESS] Infrastructure setup complete!
    goto :eof

REM Check Docker installation
:check_docker
    echo [INFO] Checking Docker installation...
    docker --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker is not installed. Please install Docker Desktop first.
        exit /b 1
    )

    docker compose version >nul 2>&1
    if errorlevel 1 (
        docker-compose --version >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
            exit /b 1
        )
        set COMPOSE_CMD=docker-compose
    ) else (
        set COMPOSE_CMD=docker compose
    )

    echo [SUCCESS] Docker is installed
    goto :eof

REM Check environment file
:check_env_file
    echo [INFO] Checking environment configuration...

    if not exist "%ENV_FILE%" (
        echo [WARNING] Environment file not found: %ENV_FILE%
        echo [INFO] Creating from template...

        if exist ".env.infrastructure.example" (
            copy .env.infrastructure.example "%ENV_FILE%" >nul
            echo [WARNING] Please edit %ENV_FILE% with your configuration before running again
            exit /b 1
        ) else (
            echo [ERROR] Template file not found: .env.infrastructure.example
            exit /b 1
        )
    )

    echo [SUCCESS] Environment file found
    goto :eof

REM Start services
:start_services
    echo [INFO] Starting infrastructure services...

    set COMPOSE_ARGS=-f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% --env-file %ENV_FILE%

    if "%WITH_TOOLS%"=="true" (
        set COMPOSE_ARGS=%COMPOSE_ARGS% --profile tools
        echo [INFO] Including management UI tools (pgAdmin, Redis Commander)
    )

    %COMPOSE_CMD% %COMPOSE_ARGS% up -d
    if errorlevel 1 (
        echo [ERROR] Failed to start services
        exit /b 1
    )

    echo [SUCCESS] Services started
    goto :eof

REM Wait for service (simulated health check)
:wait_for_service
    set SERVICE_NAME=%1
    set SERVICE_HOST=%2
    set SERVICE_PORT=%3
    set MAX_ATTEMPTS=30
    set ATTEMPT=0

    echo [INFO] Waiting for %SERVICE_NAME% to be ready...

    :wait_loop
    if !ATTEMPT! geq %MAX_ATTEMPTS% (
        echo.
        echo [ERROR] %SERVICE_NAME% failed to start within expected time
        exit /b 1
    )

    REM Try to connect to the port (using PowerShell)
    powershell -Command "Test-NetConnection -ComputerName %SERVICE_HOST% -Port %SERVICE_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue" >nul 2>&1
    if errorlevel 1 (
        set /a ATTEMPT+=1
        echo|set /p="."
        timeout /t 2 /nobreak >nul
        goto wait_loop
    )

    echo.
    echo [SUCCESS] %SERVICE_NAME% is ready
    goto :eof

REM Run health checks
:run_health_checks
    if "%SKIP_HEALTH_CHECK%"=="true" (
        echo [WARNING] Skipping health checks
        goto :eof
    )

    echo [INFO] Running health checks...

    call :wait_for_service "PostgreSQL" "localhost" "5432"
    call :wait_for_service "Qdrant" "localhost" "6333"
    call :wait_for_service "Redis" "localhost" "6379"

    echo [SUCCESS] All services are healthy
    goto :eof

REM Print service information
:print_service_info
    echo.
    echo ============================================================================
    echo  OpenEvolve Infrastructure Services
    echo ============================================================================
    echo.
    echo PostgreSQL:
    echo   Host: localhost
    echo   Port: 5432
    echo   Database: openevolve
    echo   User: openevolve
    echo   Connection String: postgresql://openevolve:changeme@localhost:5432/openevolve
    echo.
    echo Qdrant Vector Database:
    echo   HTTP API: http://localhost:6333
    echo   gRPC API: localhost:6334
    echo   Dashboard: http://localhost:6333/dashboard
    echo.
    echo Redis:
    echo   Host: localhost
    echo   Port: 6379
    echo   URL: redis://localhost:6379
    echo.

    if "%WITH_TOOLS%"=="true" (
        echo Management Tools:
        echo   pgAdmin: http://localhost:5050
        echo     Email: admin@openevolve.local
        echo     Password: (see .env.infrastructure)
        echo   Redis Commander: http://localhost:8081
        echo.
    )

    echo ============================================================================
    echo.
    echo Useful Commands:
    echo   View logs: docker compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% logs -f
    echo   Stop services: docker compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% down
    echo   Restart services: docker compose -f %COMPOSE_FILE% -p %COMPOSE_PROJECT_NAME% restart
    echo.
    goto :eof

REM Run main
call :main
