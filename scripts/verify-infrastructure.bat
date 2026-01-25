@echo off
REM =============================================================================
REM OpenEvolve Infrastructure Verification Script (Windows)
REM =============================================================================
REM This script verifies that all infrastructure services are running and
REM accessible. Run this after starting services to confirm everything works.
REM
REM Usage:
REM   scripts\verify-infrastructure.bat
REM =============================================================================

setlocal enabledelayedexpansion

set TOTAL_CHECKS=0
set PASSED_CHECKS=0
set FAILED_CHECKS=0

:main
    echo.
    echo ============================================================================
    echo  OpenEvolve Infrastructure Verification
    echo ============================================================================
    echo.

    REM Check Docker
    echo [INFO] Checking Docker installation...
    docker --version >nul 2>&1
    if errorlevel 1 (
        echo [X] Docker is not installed
        set /a FAILED_CHECKS+=1
    ) else (
        echo [✓] Docker is installed
        docker --version
        set /a PASSED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1

    echo.

    REM Check PostgreSQL
    echo [INFO] Checking PostgreSQL container...
    docker ps | findstr openevolve-postgres >nul 2>&1
    if errorlevel 1 (
        echo [X] PostgreSQL container is not running
        set /a FAILED_CHECKS+=1
    ) else (
        echo [✓] PostgreSQL container is running

        docker exec openevolve-postgres pg_isready -U openevolve -d openevolve >nul 2>&1
        if errorlevel 1 (
            echo [X] PostgreSQL is not accepting connections
            set /a FAILED_CHECKS+=1
        ) else (
            echo [✓] PostgreSQL is accepting connections
            set /a PASSED_CHECKS+=1
        )
        set /a TOTAL_CHECKS+=1
        set /a PASSED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1

    echo.

    REM Check Qdrant
    echo [INFO] Checking Qdrant container...
    docker ps | findstr openevolve-qdrant >nul 2>&1
    if errorlevel 1 (
        echo [X] Qdrant container is not running
        set /a FAILED_CHECKS+=1
    ) else (
        echo [✓] Qdrant container is running

        REM Try to access Qdrant using PowerShell
        powershell -Command "Test-NetConnection -ComputerName localhost -Port 6333 -InformationLevel Quiet -WarningAction SilentlyContinue" >nul 2>&1
        if errorlevel 1 (
            echo [X] Qdrant HTTP API is not accessible
            set /a FAILED_CHECKS+=1
        ) else (
            echo [✓] Qdrant HTTP API is accessible
            set /a PASSED_CHECKS+=1
        )
        set /a TOTAL_CHECKS+=1
        set /a PASSED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1

    echo.

    REM Check Redis
    echo [INFO] Checking Redis container...
    docker ps | findstr openevolve-redis >nul 2>&1
    if errorlevel 1 (
        echo [X] Redis container is not running
        set /a FAILED_CHECKS+=1
    ) else (
        echo [✓] Redis container is running

        docker exec openevolve-redis redis-cli ping | findstr PONG >nul 2>&1
        if errorlevel 1 (
            echo [X] Redis is not responding
            set /a FAILED_CHECKS+=1
        ) else (
            echo [✓] Redis is responding to PING
            set /a PASSED_CHECKS+=1
        )
        set /a TOTAL_CHECKS+=1
        set /a PASSED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1

    echo.
    echo ============================================================================
    echo  Summary
    echo ============================================================================
    echo.
    echo Total checks: %TOTAL_CHECKS%
    echo Passed: %PASSED_CHECKS%
    echo Failed: %FAILED_CHECKS%
    echo.

    if %FAILED_CHECKS%==0 (
        echo [✓] All checks passed! Infrastructure is ready.
        echo.
        echo Next steps:
        echo   1. Configure your application to use the services
        echo   2. See docs\INFRASTRUCTURE_SETUP.md for connection details
        echo.
    ) else (
        echo [X] Some checks failed. Please review the errors above.
        echo.
        echo Troubleshooting:
        echo   1. Check container logs: docker logs ^<container-name^>
        echo   2. Restart services: scripts\dev-stop.bat ^&^& scripts\dev-start.bat
        echo   3. See docs\INFRASTRUCTURE_SETUP.md for troubleshooting guide
        echo.
    )

goto :eof

call :main
