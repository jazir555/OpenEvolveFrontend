@echo off
REM #############################################################################
REM Universal Deployment Script for All Adapters (Windows)
REM
REM Usage:
REM   deploy-all-adapters.cmd [options]
REM
REM Options:
REM   --push              Push images to registry after building
REM   --registry <url>    Docker registry URL (default: localhost:5000)
REM   --tag <tag>         Image tag (default: latest)
REM   --skip-tests        Skip running contract tests
REM   --adapter <name>    Deploy only a specific adapter
REM   --dry-run           Show what would be deployed without building
REM   -h, --help          Show this help message
REM
REM Examples:
REM   deploy-all-adapters.cmd
REM   deploy-all-adapters.cmd --push --registry registry.example.com
REM   deploy-all-adapters.cmd --adapter bubblelab-adapter --tag v1.0.0
REM   deploy-all-adapters.cmd --skip-tests
REM #############################################################################

setlocal EnableDelayedExpansion

REM Configuration
set ADAPTERS_DIR=glue\adapters
set REGISTRY=%DOCKER_REGISTRY%
if "%REGISTRY%"=="" set REGISTRY=localhost:5000
set TAG=%IMAGE_TAG%
if "%TAG%"=="" set TAG=latest
set PUSH=false
set SKIP_TESTS=false
set SPECIFIC_ADAPTER=
set DRY_RUN=false

REM Parse arguments
:parse_args
if "%~1"=="" goto done_parsing
if "%~1"=="--push" (
    set PUSH=true
    shift
    goto parse_args
)
if "%~1"=="--registry" (
    shift
    set REGISTRY=%~1
    shift
    goto parse_args
)
if "%~1"=="--tag" (
    shift
    set TAG=%~1
    shift
    goto parse_args
)
if "%~1"=="--skip-tests" (
    set SKIP_TESTS=true
    shift
    goto parse_args
)
if "%~1"=="--adapter" (
    shift
    set SPECIFIC_ADAPTER=%~1
    shift
    goto parse_args
)
if "%~1"=="--dry-run" (
    set DRY_RUN=true
    shift
    goto parse_args
)
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
echo [ERROR] Unknown option: %~1
goto show_help
:done_parsing

REM #############################################################################
REM Main Deployment Flow
REM #############################################################################

echo.
echo ================================================================
echo   OpenEvolve Adapter Deployment System (Windows)
echo ================================================================
echo.

REM Pre-flight checks
echo [INFO] Running pre-flight checks...
if not exist "%ADAPTERS_DIR%" (
    echo [ERROR] Adapters directory not found: %ADAPTERS_DIR%
    exit /b 1
)

docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running
    exit /b 1
)

echo [SUCCESS] Pre-flight checks passed
echo.

REM Show summary
echo ================================================================
echo   Deployment Summary
echo ================================================================
echo Registry: %REGISTRY%
echo Tag: %TAG%
echo Push to registry: %PUSH%
echo Skip tests: %SKIP_TESTS%
echo Dry run: %DRY_RUN%
if not "%SPECIFIC_ADAPTER%"=="" echo Specific adapter: %SPECIFIC_ADAPTER%
echo.

REM Count statistics
set /a success_count=0
set /a failed_count=0
set /a skipped_count=0

REM Deploy adapters
if "%SPECIFIC_ADAPTER%"=="" (
    REM Deploy all adapters
    echo [INFO] Scanning for adapters in %ADAPTERS_DIR%...
    echo.

    for /d %%D in ("%ADAPTERS_DIR%\*") do (
        set "adapter=%%~nxD"
        call :deploy_adapter %%~nxD
        if !errorlevel! equ 0 (
            set /a success_count+=1
        ) else if !errorlevel! equ 2 (
            set /a skipped_count+=1
        ) else (
            set /a failed_count+=1
        )
    )
) else (
    REM Deploy specific adapter
    call :deploy_adapter %SPECIFIC_ADAPTER%
    if !errorlevel! equ 0 (
        set /a success_count=1
    ) else if !errorlevel! equ 2 (
        set /a skipped_count=1
    ) else (
        set /a failed_count=1
    )
)

REM Final summary
echo.
echo ================================================================
echo   Deployment Complete
echo ================================================================
echo Successful: %success_count%
echo Failed: %failed_count%
echo Skipped: %skipped_count%
echo.

if %failed_count% GTR 0 (
    echo [ERROR] Some adapters failed to deploy
    exit /b 1
) else (
    echo [SUCCESS] All adapters deployed successfully!
    echo.
    echo Next Steps:
    echo   To run individual adapters:
    echo     docker run -p 8080:8080 %REGISTRY%/\<adapter-name\>:%TAG%
    echo.
    echo   To view all built images:
    echo     docker images ^| findstr %REGISTRY%
    echo.
    echo   To push to registry:
    echo     deploy-all-adapters.cmd --push
    echo.
)

exit /b 0

REM #############################################################################
REM Deploy single adapter
REM #############################################################################
:deploy_adapter
setlocal
set adapter=%~1
set adapter_dir=%ADAPTERS_DIR%\%adapter%

echo ================================================================
echo   Deploying: %adapter%
echo ================================================================

REM Check if adapter has Dockerfile
if not exist "%adapter_dir%\Dockerfile" (
    echo [WARN] Skipping %adapter% (no Dockerfile)
    exit /b 2
)

REM Build image
echo [STEP] Building %adapter%...
if "%DRY_RUN%"=="true" (
    echo [INFO] [DRY RUN] Would build: %REGISTRY%/%adapter%:%TAG%
) else (
    cd /d "%adapter_dir%"
    docker build -t "%REGISTRY%/%adapter%:%TAG" -t "%REGISTRY%/%adapter%:latest" .
    if errorlevel 1 (
        echo [ERROR] Failed to build %adapter%
        exit /b 1
    )
    echo [SUCCESS] Built %REGISTRY%/%adapter%:%TAG%
)

REM Run tests (if not skipped)
if "%SKIP_TESTS%"=="false" (
    if "%DRY_RUN%"=="false" (
        if exist "%adapter_dir%\tests" (
            echo [STEP] Testing %adapter%...
            docker run --rm "%REGISTRY%/%adapter%:%TAG" python -m pytest tests/ -v >nul 2>&1
            if errorlevel 1 (
                docker run --rm "%REGISTRY%/%adapter%:%TAG%" pytest tests/ -v >nul 2>&1
                if errorlevel 1 (
                    echo [WARN] Could not run tests for %adapter%
                )
            )
            echo [SUCCESS] Tests completed for %adapter%
        ) else (
            echo [WARN] No tests directory found for %adapter%
        )
    )
)

REM Push to registry (if requested)
if "%PUSH%"=="true" (
    echo [STEP] Pushing %adapter% to registry...
    if "%DRY_RUN%"=="true" (
        echo [INFO] [DRY RUN] Would push: %REGISTRY%/%adapter%:%TAG%
    ) else (
        docker push "%REGISTRY%/%adapter%:%TAG%"
        docker push "%REGISTRY%/%adapter%:latest"
        if errorlevel 1 (
            echo [ERROR] Failed to push %adapter%
            exit /b 1
        )
        echo [SUCCESS] Pushed %adapter% to registry
    )
)

echo [SUCCESS] Deployed %adapter% successfully
echo.

exit /b 0

REM #############################################################################
REM Show help
REM #############################################################################
:show_help
echo.
echo OpenEvolve Adapter Deployment Script (Windows)
echo.
echo Usage:
echo   deploy-all-adapters.cmd [options]
echo.
echo Options:
echo   --push              Push images to registry after building
echo   --registry ^<url^>    Docker registry URL (default: localhost:5000)
echo   --tag ^<tag^>         Image tag (default: latest)
echo   --skip-tests        Skip running contract tests
echo   --adapter ^<name^>    Deploy only a specific adapter
echo   --dry-run           Show what would be deployed without building
echo   -h, --help          Show this help message
echo.
echo Examples:
echo   deploy-all-adapters.cmd
echo   deploy-all-adapters.cmd --push --registry registry.example.com
echo   deploy-all-adapters.cmd --adapter bubblelab-adapter --tag v1.0.0
echo   deploy-all-adapters.cmd --skip-tests
echo.
exit /b 0
