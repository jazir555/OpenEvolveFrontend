@echo off
REM =============================================================================
REM OpenEvolve Quick Start Script (Windows)
REM License: Apache 2.0
REM
REM Description: Automated setup and deployment script for OpenEvolve Frontend
REM Usage: quick-start.cmd [--dry-run] [--skip-tests] [--env-file FILE]
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
set "LOG_FILE=%LOG_DIR%\quick-start-%TIMESTAMP%.log"

set "DRY_RUN=0"
set "SKIP_TESTS=0"
set "ENV_FILE=%PROJECT_ROOT%\.env"

REM =============================================================================
REM Utility Functions
REM =============================================================================

:log
set "LEVEL=%~1"
shift
set "MESSAGE=%*"
set "TIMESTAMP_LOG=%date% %time%"
echo %TIMESTAMP_LOG% [%LEVEL%] %MESSAGE%
echo %TIMESTAMP_LOG% [%LEVEL%] %MESSAGE% >> "%LOG_FILE%"
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

:print_step
echo.
echo ================================================================
echo   STEP %~1
echo ================================================================
echo.
goto :eof

:usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo OpenEvolve Quick Start Script - Automated setup and deployment
echo.
echo OPTIONS:
echo     --dry-run           Show what would be done without executing
echo     --skip-tests        Skip running tests
echo     --env-file FILE     Use specific environment file (default: .env)
echo     -h, --help          Show this help message
echo.
echo EXAMPLES:
echo     %~nx0                              # Standard quick start
echo     %~nx0 --dry-run                    # Preview actions without executing
echo     %~nx0 --skip-tests                 # Skip tests
echo.
goto :eof

REM =============================================================================
REM Prerequisites Check
REM =============================================================================

:check_prerequisites
call :print_step "1: Checking Prerequisites"

set "MISSING_DEPS="

REM Check Node.js
where node >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('node -v') do set "NODE_VERSION=%%i"
    call :log_success "Node.js found: !NODE_VERSION!"
) else (
    call :log_error "Node.js not found"
    set "MISSING_DEPS=!MISSING_DEPS! node"
)

REM Check npm
where npm >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('npm -v') do set "NPM_VERSION=%%i"
    call :log_success "npm found: !NPM_VERSION!"
) else (
    call :log_error "npm not found"
    set "MISSING_DEPS=!MISSING_DEPS! npm"
)

REM Check Docker
where docker >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=1,2,3" %%a in ('docker --version') do set "DOCKER_VERSION=%%c"
    set "DOCKER_VERSION=!DOCKER_VERSION:,=!"
    call :log_success "Docker found: !DOCKER_VERSION!"
) else (
    call :log_error "Docker not found"
    set "MISSING_DEPS=!MISSING_DEPS! docker"
)

REM Check Docker Compose
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('docker compose version --short 2^>nul') do set "COMPOSE_VERSION=%%i"
    call :log_success "Docker Compose found: !COMPOSE_VERSION!"
) else (
    call :log_error "Docker Compose not found"
    set "MISSING_DEPS=!MISSING_DEPS! docker-compose"
)

REM Check Python (optional)
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    call :log_success "Python found: !PYTHON_VERSION!"
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('python3 --version 2^>^&1') do set "PYTHON_VERSION=%%i"
        call :log_success "Python found: !PYTHON_VERSION!"
    ) else (
        call :log_warning "Python not found (optional but recommended)"
    )
)

REM Check git
where git >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('git --version') do set "GIT_VERSION=%%i"
    call :log_success "Git found: !GIT_VERSION!"
) else (
    call :log_warning "Git not found (optional)"
)

if not "!MISSING_DEPS!"=="" (
    call :log_error "Missing required dependencies:!MISSING_DEPS!"
    call :log_info "Please install missing dependencies and try again"
    exit /b 1
)

call :log_success "All required prerequisites are installed"
exit /b 0

REM =============================================================================
REM Environment Validation
REM =============================================================================

:validate_environment
call :print_step "2: Validating Environment"

REM Check if .env file exists
if not exist "%ENV_FILE%" (
    call :log_warning "Environment file not found: %ENV_FILE%"

    if exist "%PROJECT_ROOT%\.env.example" (
        call :log_info "Creating .env from .env.example..."
        if %DRY_RUN%==0 (
            copy "%PROJECT_ROOT%\.env.example" "%ENV_FILE%" >nul
            call :log_warning "Please edit %ENV_FILE% and set appropriate values"
            call :log_warning "Especially set SECRET_KEY to a secure random value!"
        )
    ) else (
        call :log_error ".env.example not found. Cannot create environment file"
        exit /b 1
    )
)

REM Load environment variables
call :log_info "Validating environment variables..."

REM Read SECRET_KEY from .env
set "SECRET_KEY="
for /f "tokens=1,2 delims==" %%a in ('type "%ENV_FILE%" ^| findstr /b "SECRET_KEY="') do (
    set "SECRET_KEY=%%b"
)

REM Check if SECRET_KEY is still the default
if "!SECRET_KEY!"=="changeme-in-production-generate-random-string" (
    call :log_warning "SECRET_KEY is set to default value. This is unsafe for production!"
    set /p "GENERATE_KEY=Generate a new SECRET_KEY? (y/n): "
    if /i "!GENERATE_KEY!"=="y" (
        if %DRY_RUN%==0 (
            REM Generate random key using PowerShell
            for /f "tokens=*" %%i in ('powershell -Command " -join ((48..57) + (97..102) | Get-Random -Count 64 | %%{[char]$_})"') do set "NEW_KEY=%%i"
            powershell -Command "(Get-Content '%ENV_FILE%') -replace '^SECRET_KEY=.*', 'SECRET_KEY=!NEW_KEY!' | Set-Content '%ENV_FILE%'"
            call :log_success "Generated new SECRET_KEY"
        )
    )
)

call :log_success "Environment validation complete"
exit /b 0

REM =============================================================================
REM Install Dependencies
REM =============================================================================

:install_dependencies
call :print_step "3: Installing Dependencies"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would install npm dependencies"
    exit /b 0
)

call :log_info "Installing npm dependencies..."
call npm install --legacy-peer-deps >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Dependencies installed successfully"
) else (
    call :log_error "Failed to install dependencies"
    exit /b 1
)
exit /b 0

REM =============================================================================
REM Run Tests
REM =============================================================================

:run_tests
if %SKIP_TESTS%==1 (
    call :log_warning "Skipping tests as requested"
    exit /b 0
)

call :print_step "4: Running Tests"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would run test suite"
    exit /b 0
)

call :log_info "Running TypeScript type checking..."
call npm run typecheck >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Type checking passed"
) else (
    call :log_warning "Type checking failed (continuing anyway)"
)

call :log_info "Running lint checks..."
call npm run lint >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Lint checks passed"
) else (
    call :log_warning "Lint checks failed (continuing anyway)"
)

call :log_info "Running unit tests..."
call npm run test >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Unit tests passed"
) else (
    call :log_warning "Unit tests failed (continuing anyway)"
)

exit /b 0

REM =============================================================================
REM Build Everything
REM =============================================================================

:build_everything
call :print_step "5: Building Everything"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would build TypeScript code and Docker images"
    exit /b 0
)

call :log_info "Building TypeScript code..."
call npm run build >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "TypeScript build successful"
) else (
    call :log_warning "TypeScript build failed or no build script found"
)

call :log_info "Building Docker images..."
docker compose build >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Docker images built successfully"
) else (
    call :log_error "Failed to build Docker images"
    exit /b 1
)

exit /b 0

REM =============================================================================
REM Start Services
REM =============================================================================

:start_services
call :print_step "6: Starting Services"

cd /d "%PROJECT_ROOT%"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would start Docker Compose services"
    exit /b 0
)

call :log_info "Starting Docker Compose services..."
docker compose up -d >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_success "Services started successfully"
) else (
    call :log_error "Failed to start services"
    exit /b 1
)

exit /b 0

REM =============================================================================
REM Verify Health
REM =============================================================================

:verify_health
call :print_step "7: Verifying Service Health"

if %DRY_RUN%==1 (
    call :log_info "[DRY-RUN] Would verify service health"
    exit /b 0
)

call :log_info "Waiting for services to become healthy..."
timeout /t 10 /nobreak >nul

set "UNHEALTHY_COUNT=0"

REM Check OpenEvolve API
call :log_info "Checking openevolve-app at http://localhost:8080/health..."
curl -f -s -o nul --max-time 5 http://localhost:8080/health >nul 2>&1
if %errorlevel% equ 0 (
    call :log_success "openevolve-app is healthy"
) else (
    call :log_error "openevolve-app is not responding"
    set /a "UNHEALTHY_COUNT+=1"
)

REM Check Prometheus
call :log_info "Checking openevolve-prometheus at http://localhost:9090/-/healthy..."
curl -f -s -o nul --max-time 5 http://localhost:9090/-/healthy >nul 2>&1
if %errorlevel% equ 0 (
    call :log_success "openevolve-prometheus is healthy"
) else (
    call :log_error "openevolve-prometheus is not responding"
    set /a "UNHEALTHY_COUNT+=1"
)

REM Check Grafana
call :log_info "Checking openevolve-grafana at http://localhost:3000/api/health..."
curl -f -s -o nul --max-time 5 http://localhost:3000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    call :log_success "openevolve-grafana is healthy"
) else (
    call :log_warning "openevolve-grafana is not responding (may still be starting)"
)

if %UNHEALTHY_COUNT% gtr 0 (
    call :log_warning "Some services are not healthy"
    call :log_info "Check logs with: docker compose logs -f"
    exit /b 1
)

call :log_success "All services are healthy"
exit /b 0

REM =============================================================================
REM Show Next Steps
REM =============================================================================

:show_next_steps
call :print_step "8: Next Steps"

echo.
echo [SUCCESS] OpenEvolve Frontend is now running!
echo.
echo Service URLs:
echo   - OpenEvolve API:        http://localhost:8000
echo   - GraphQL API:           http://localhost:8001
echo   - Orchestrator/Gateway:  http://localhost:8080
echo   - BubbleLab Dashboard:   http://localhost:8501
echo   - Jaeger Tracing:        http://localhost:16686
echo   - Prometheus Metrics:    http://localhost:9090
echo   - Grafana Dashboard:     http://localhost:3000 (admin/admin)
echo.
echo Useful Commands:
echo   - View logs:             docker compose logs -f
echo   - Stop services:         docker compose down
echo   - Restart services:      docker compose restart
echo   - Check health:          scripts\health-check.cmd
echo   - Run smoke tests:       scripts\smoke-test.cmd
echo   - Validate deployment:   scripts\validate.cmd
echo.
echo Documentation:
echo   - Project README:        %PROJECT_ROOT%\README.md
echo   - Scripts README:        %PROJECT_ROOT%\scripts\README.md
echo.
echo Important:
echo   - Check logs for any warnings or errors
echo   - Update SECRET_KEY in .env for production use
echo   - Review and configure environment variables
echo   - Set up authentication for external access
echo.
call :log_info "Full log saved to: %LOG_FILE%"
exit /b 0

REM =============================================================================
REM Main Execution
REM =============================================================================

:main
echo ===============================================================================
echo      OpenEvolve Frontend - Quick Start Script
echo ===============================================================================
echo.

REM Create logs directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Parse arguments
:parse_loop
if "%~1"=="" goto :parse_done
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_loop
)
if /i "%~1"=="--skip-tests" (
    set "SKIP_TESTS=1"
    shift
    goto :parse_loop
)
if /i "%~1"=="--env-file" (
    set "ENV_FILE=%~2"
    shift
    shift
    goto :parse_loop
)
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage
shift
goto :parse_loop

:parse_done
REM Execute steps
call :check_prerequisites
if %errorlevel% neq 0 exit /b 1

call :validate_environment
if %errorlevel% neq 0 exit /b 1

call :install_dependencies
if %errorlevel% neq 0 exit /b 1

call :run_tests
if %errorlevel% neq 0 exit /b 1

call :build_everything
if %errorlevel% neq 0 exit /b 1

call :start_services
if %errorlevel% neq 0 exit /b 1

call :verify_health
if %errorlevel% neq 0 exit /b 1

call :show_next_steps

echo.
call :log_success "Quick start completed successfully!"
pause
exit /b 0
