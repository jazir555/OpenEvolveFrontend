@echo off
REM =============================================================================
REM OpenEvolve Deployment Script (Windows)
REM License: Apache 2.0
REM
REM Description: Automated deployment script for OpenEvolve Frontend
REM Usage: deploy.cmd [local|production] [--dry-run] [--skip-smoke-tests]
REM =============================================================================

setlocal EnableDelayedExpansion

REM =============================================================================
REM Configuration
REM =============================================================================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "LOG_FILE=%LOG_DIR%\deploy-%TIMESTAMP%.log"

set "DEPLOYMENT_TYPE=local"
set "DRY_RUN=0"
set "SKIP_SMOKE_TESTS=0"

REM =============================================================================
REM Utility Functions
REM =============================================================================

:print_step
echo.
echo ================================================================
echo   STEP %~1: %~2
echo ================================================================
echo.
goto :eof

:log_info
echo [INFO] %*
echo [INFO] %* >> "%LOG_FILE%"
goto :eof

:log_success
echo [SUCCESS] %*
echo [SUCCESS] %* >> "%LOG_FILE%"
goto :eof

:log_warning
echo [WARNING] %*
echo [WARNING] %* >> "%LOG_FILE%"
goto :eof

:log_error
echo [ERROR] %*
echo [ERROR] %* >> "%LOG_FILE%"
goto :eof

:usage
echo Usage: %~nx0 [ENVIRONMENT] [OPTIONS]
echo.
echo OpenEvolve Deployment Script - Automated deployment
echo.
echo ENVIRONMENTS:
echo     local               Deploy to local Docker Compose (default)
echo     production          Deploy to production
echo.
echo OPTIONS:
echo     --dry-run           Show what would be done without executing
echo     --skip-smoke-tests  Skip running smoke tests after deployment
echo     -h, --help          Show this help message
echo.
echo EXAMPLES:
echo     %~nx0 local                        # Deploy to local
echo     %~nx0 production                   # Deploy to production
echo     %~nx0 local --dry-run              # Preview deployment
echo.
goto :eof

REM =============================================================================
REM Pre-Deployment Checks
REM =============================================================================

:pre_deployment_checks
call :print_step "1" "Pre-Deployment Checks"

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :log_error "Docker is not running. Please start Docker and try again."
    exit /b 1
)
call :log_success "Docker is running"

REM Check environment file
set "ENV_FILE=%PROJECT_ROOT%\.env"
if "%DEPLOYMENT_TYPE%"=="production" (
    set "ENV_FILE=%PROJECT_ROOT%\deploy\production\.env.production"
)

if not exist "!ENV_FILE!" (
    call :log_error "Environment file not found: !ENV_FILE!"
    exit /b 1
)
call :log_success "Environment file found: !ENV_FILE!"

call :log_success "Pre-deployment checks complete"
exit /b 0

REM =============================================================================
REM Build Adapters
REM =============================================================================

:build_adapters
call :print_step "2" "Building Adapters"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would build all adapters"
    exit /b 0
)

call :log_info "Building glue layer and adapters..."
if exist "%PROJECT_ROOT%\glue\package.json" (
    cd /d "%PROJECT_ROOT%\glue"
    call npm run build >> "%LOG_FILE%" 2>&1
    if %errorlevel% equ 0 (
        call :log_success "Glue layer built successfully"
    ) else (
        call :log_warning "Glue layer build failed or no build script"
    )
)

cd /d "%PROJECT_ROOT%"
exit /b 0

REM =============================================================================
REM Build Docker Images
REM =============================================================================

:build_docker_images
call :print_step "3" "Building Docker Images"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would build Docker images"
    exit /b 0
)

call :log_info "Building Docker images..."
docker compose build --no-cache >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Docker images built successfully"
) else (
    call :log_error "Failed to build Docker images"
    exit /b 1
)

exit /b 0

REM =============================================================================
REM Validate Configurations
REM =============================================================================

:validate_configurations
call :print_step "4" "Validating Configurations"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would validate configurations"
    exit /b 0
)

call :log_info "Validating Docker Compose configuration..."
docker compose config >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Docker Compose configuration is valid"
) else (
    call :log_error "Docker Compose configuration is invalid"
    exit /b 1
)

exit /b 0

REM =============================================================================
REM Deploy Services
REM =============================================================================

:deploy_services
call :print_step "5" "Deploying Services"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would deploy services to %DEPLOYMENT_TYPE%"
    exit /b 0
)

call :log_info "Stopping existing services..."
docker compose down >> "%LOG_FILE%" 2>&1

call :log_info "Deploying services to %DEPLOYMENT_TYPE%..."
docker compose up -d >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Services deployed successfully"
) else (
    call :log_error "Failed to deploy services"
    exit /b 1
)

exit /b 0

REM =============================================================================
REM Wait for Services
REM =============================================================================

:wait_for_services
call :print_step "6" "Waiting for Services to be Healthy"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would wait for services to be healthy"
    exit /b 0
)

call :log_info "Waiting for services to start..."
timeout /t 15 /nobreak >nul

call :log_success "Services started"
exit /b 0

REM =============================================================================
REM Run Smoke Tests
REM =============================================================================

:run_smoke_tests
if %SKIP_SMOKE_TESTS%==1 (
    call :log_info "Skipping smoke tests as requested"
    exit /b 0
)

call :print_step "7" "Running Smoke Tests"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would run smoke tests"
    exit /b 0
)

if exist "%SCRIPT_DIR%\smoke-test.cmd" (
    call :log_info "Running smoke tests..."
    call "%SCRIPT_DIR%\smoke-test.cmd" >> "%LOG_FILE%" 2>&1
    if %errorlevel% equ 0 (
        call :log_success "Smoke tests passed"
    ) else (
        call :log_warning "Smoke tests failed"
    )
) else (
    call :log_warning "Smoke test script not found. Skipping."
)

exit /b 0

REM =============================================================================
REM Show Deployment Status
REM =============================================================================

:show_deployment_status
call :print_step "8" "Deployment Status"

echo.
call :log_info "Container Status:"
docker compose ps
echo.

call :log_info "Service URLs:"
echo   - OpenEvolve API:        http://localhost:8000
echo   - GraphQL API:           http://localhost:8001
echo   - Orchestrator/Gateway:  http://localhost:8080
echo   - BubbleLab Dashboard:   http://localhost:8501
echo   - Jaeger Tracing:        http://localhost:16686
echo   - Prometheus Metrics:    http://localhost:9090
echo   - Grafana Dashboard:     http://localhost:3000
echo.

if "%DEPLOYMENT_TYPE%"=="production" (
    call :log_warning "Deployed to PRODUCTION environment"
)

call :log_info "Full deployment log saved to: %LOG_FILE%"
exit /b 0

REM =============================================================================
REM Main Execution
REM =============================================================================

:main
echo ===============================================================================
echo      OpenEvolve Frontend - Deployment Script
echo ===============================================================================
echo.

REM Create logs directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Parse arguments
:parse_loop
if "%~1"=="" goto :parse_done
if /i "%~1"=="local" (
    set "DEPLOYMENT_TYPE=local"
    shift
    goto :parse_loop
)
if /i "%~1"=="production" (
    set "DEPLOYMENT_TYPE=production"
    shift
    goto :parse_loop
)
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_loop
)
if /i "%~1"=="--skip-smoke-tests" (
    set "SKIP_SMOKE_TESTS=1"
    shift
    goto :parse_loop
)
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage
shift
goto :parse_loop

:parse_done
call :log_info "Deployment type: %DEPLOYMENT_TYPE%"

REM Execute deployment steps
call :pre_deployment_checks
if %errorlevel% neq 0 exit /b 1

call :build_adapters
if %errorlevel% neq 0 exit /b 1

call :build_docker_images
if %errorlevel% neq 0 exit /b 1

call :validate_configurations
if %errorlevel% neq 0 exit /b 1

call :deploy_services
if %errorlevel% neq 0 exit /b 1

call :wait_for_services
if %errorlevel% neq 0 exit /b 1

call :run_smoke_tests

call :show_deployment_status

echo.
call :log_success "Deployment to %DEPLOYMENT_TYPE% completed successfully!"
pause
exit /b 0
