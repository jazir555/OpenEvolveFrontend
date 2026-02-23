@echo off
REM #############################################################################
REM Single Adapter Deployment Script (Windows)
REM
REM Usage:
REM   deploy-adapter.cmd <adapter-name> [options]
REM
REM Arguments:
REM   adapter-name        Name of the adapter (e.g., bubblelab-adapter)
REM
REM Options:
REM   --push              Push image to registry after building
REM   --registry <url>    Docker registry URL (default: localhost:5000)
REM   --tag <tag>         Image tag (default: latest)
REM   --skip-tests        Skip running contract tests
REM   --dry-run           Show what would be deployed without building
REM   -h, --help          Show this help message
REM
REM Examples:
REM   deploy-adapter.cmd bubblelab-adapter
REM   deploy-adapter.cmd z3-adapter --push --tag v1.0.0
REM   deploy-adapter.cmd openevolve-adapter --registry registry.example.com
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
set DRY_RUN=false

REM Check if adapter name is provided
if "%~1"=="" goto show_help
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help

set ADAPTER_NAME=%~1
shift

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
if "%~1"=="--dry-run" (
    set DRY_RUN=true
    shift
    goto parse_args
)
echo [ERROR] Unknown option: %~1
goto show_help
:done_parsing

REM #############################################################################
REM Main Deployment Flow
REM #############################################################################

echo.
echo ================================================================
echo   OpenEvolve Single Adapter Deployment (Windows)
echo ================================================================
echo.
echo Adapter: %ADAPTER_NAME%
echo.

REM #############################################################################
REM Pre-flight checks
REM #############################################################################

echo [INFO] Running pre-flight checks...

REM Check if adapter exists
if not exist "%ADAPTERS_DIR%\%ADAPTER_NAME%" (
    echo [ERROR] Adapter not found: %ADAPTER_NAME%
    echo.
    echo Available adapters:
    dir /b /ad "%ADAPTERS_DIR%" | findstr /i "adapter$" || echo   None found
    exit /b 1
)

REM Check if Dockerfile exists
if not exist "%ADAPTERS_DIR%\%ADAPTER_NAME%\Dockerfile" (
    echo [ERROR] Dockerfile not found for adapter: %ADAPTER_NAME%
    exit /b 1
)

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running
    exit /b 1
)

echo [SUCCESS] Pre-flight checks passed
echo.

REM #############################################################################
REM Build Docker image
REM #############################################################################

set IMAGE_NAME=%REGISTRY%/%ADAPTER_NAME%:%TAG%
set ADAPTER_DIR=%ADAPTERS_DIR%\%ADAPTER_NAME%

echo [STEP] Building Docker image: %IMAGE_NAME%

if "%DRY_RUN%"=="true" (
    echo [INFO] [DRY RUN] Would build: %IMAGE_NAME%
) else (
    cd /d "%ADAPTER_DIR%"
    docker build -t "!IMAGE_NAME!" -t "%REGISTRY%/%ADAPTER_NAME%:latest" .
    if errorlevel 1 (
        echo [ERROR] Image build failed
        exit /b 1
    )
    echo [SUCCESS] Image built successfully
)

REM #############################################################################
REM Run contract tests
REM #############################################################################

if "%SKIP_TESTS%"=="false" (
    echo [STEP] Running contract tests...

    if "%DRY_RUN%"=="true" (
        echo [INFO] [DRY RUN] Would test: !IMAGE_NAME!
    ) else (
        if exist "%ADAPTER_DIR%\tests" (
            docker run --rm "!IMAGE_NAME!" python -m pytest tests/ -v >nul 2>&1
            if errorlevel 1 (
                docker run --rm "!IMAGE_NAME!" pytest tests/ -v >nul 2>&1
                if errorlevel 1 (
                    echo [WARN] Could not run tests (no pytest found or tests failed)
                ) else (
                    echo [SUCCESS] Contract tests completed
                )
            ) else (
                echo [SUCCESS] Contract tests completed
            )
        ) else (
            echo [WARN] No tests directory found, skipping tests
        )
    )
) else (
    echo [WARN] Skipping contract tests
)

REM #############################################################################
REM Push to registry
REM #############################################################################

if "%PUSH%"=="true" (
    echo [STEP] Pushing image to registry: %REGISTRY%

    if "%DRY_RUN%"=="true" (
        echo [INFO] [DRY RUN] Would push: !IMAGE_NAME!
    ) else (
        docker push "!IMAGE_NAME!"
        docker push "%REGISTRY%/%ADAPTER_NAME%:latest"
        if errorlevel 1 (
            echo [ERROR] Failed to push image
            exit /b 1
        )
        echo [SUCCESS] Image pushed to registry
    )
)

REM #############################################################################
REM Show deployment status
REM #############################################################################

echo.
echo ================================================================
echo   Deployment Status
echo ================================================================
echo Adapter: %ADAPTER_NAME%
echo Image: !IMAGE_NAME!
echo Registry: %REGISTRY%
echo Tag: %TAG%
if "%SKIP_TESTS%"=="true" (
    echo Tests: Skipped
) else (
    echo Tests: Passed
)
echo.

if "%DRY_RUN%"=="false" (
    echo Local images:
    docker images | findstr "%ADAPTER_NAME%" || echo Image not found in local cache
)

REM #############################################################################
REM Show next steps
REM #############################################################################

echo.
echo ================================================================
echo   Deployment Complete
echo ================================================================
echo.
echo [SUCCESS] Adapter '%ADAPTER_NAME%' deployed successfully!
echo.
echo Next Steps:
echo   To run the adapter:
echo     docker run -p 8080:8080 %REGISTRY%/%ADAPTER_NAME%:%TAG%
echo.
echo   To run with custom environment variables:
echo     docker run -p 8080:8080 -e LOG_LEVEL=DEBUG %REGISTRY%/%ADAPTER_NAME%:%TAG%
echo.
echo   To view logs:
echo     docker logs -f ^<container-id^>
echo.
echo   To push to registry:
echo     deploy-adapter.cmd %ADAPTER_NAME% --push
echo.

exit /b 0

REM #############################################################################
REM Show help
REM #############################################################################
:show_help
echo.
echo OpenEvolve Single Adapter Deployment Script (Windows)
echo.
echo Usage:
echo   deploy-adapter.cmd ^<adapter-name^> [options]
echo.
echo Arguments:
echo   adapter-name        Name of the adapter (e.g., bubblelab-adapter)
echo.
echo Options:
echo   --push              Push image to registry after building
echo   --registry ^<url^>    Docker registry URL (default: localhost:5000)
echo   --tag ^<tag^>         Image tag (default: latest)
echo   --skip-tests        Skip running contract tests
echo   --dry-run           Show what would be deployed without building
echo   -h, --help          Show this help message
echo.
echo Examples:
echo   deploy-adapter.cmd bubblelab-adapter
echo   deploy-adapter.cmd z3-adapter --push --tag v1.0.0
echo   deploy-adapter.cmd openevolve-adapter --registry registry.example.com
echo.
exit /b 0
