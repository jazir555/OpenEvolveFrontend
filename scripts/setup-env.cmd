@echo off
REM Set up environment variables for Docker Compose
REM Following CLAUDE.md - Law of Configuration Explicitness

setlocal enabledelayedexpansion

set ENV_FILE=.env
set INFRA_EXAMPLE=infra\.env.example
set LOONGFLOW_EXAMPLE=infra\.env.loongflow.example

echo ================================================
echo OpenEvolve Environment Setup
echo ================================================
echo.

REM Check if .env exists
if exist %ENV_FILE% (
    echo ⚠️  .env file already exists
    set /p OVERWRITE="Overwrite? (y/N): "

    if /i not "!OVERWRITE!"=="y" (
        echo Exiting...
        exit /b 0
    )

    echo Backing up existing .env to .env.backup
    copy %ENV_FILE% .env.backup >nul
)

REM Determine which environment to set up
echo.
echo Which environment do you want to set up?
echo 1^) Full OpenEvolve Federation ^(all adapters^)
echo 2^) LoongFlow Core only
echo 3^) Minimal development setup
echo.
set /p CHOICE="Choose (1-3): "

if "%CHOICE%"=="1" (
    echo 📋 Setting up Full OpenEvolve Federation environment...
    if exist %INFRA_EXAMPLE% (
        copy %INFRA_EXAMPLE% %ENV_FILE% >nul
        echo ✅ Copied from %INFRA_EXAMPLE%
    ) else (
        echo ⚠️  Example file not found, creating minimal .env
        call :create_minimal_env
    )
) else if "%CHOICE%"=="2" (
    echo 📋 Setting up LoongFlow Core environment...
    if exist %LOONGFLOW_EXAMPLE% (
        copy %LOONGFLOW_EXAMPLE% %ENV_FILE% >nul
        echo ✅ Copied from %LOONGFLOW_EXAMPLE%
    ) else (
        echo ⚠️  Example file not found, creating minimal .env
        call :create_minimal_env
    )
) else if "%CHOICE%"=="3" (
    echo 📝 Creating minimal development environment...
    call :create_minimal_env
) else (
    echo ❌ Invalid choice
    exit /b 1
)

echo.
echo ✅ Environment file created: %ENV_FILE%
echo.
echo ⚠️  IMPORTANT: Review .env and update values before starting services!
echo.
echo Required variables to set:
echo   - API keys ^(OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.^)
echo   - Service URLs ^(if not using defaults^)
echo   - Database credentials ^(NEO4J_PASSWORD, etc.^)
echo.
echo Next steps:
echo 1. Edit .env with your values:
echo    notepad .env
echo.
echo 2. Start infrastructure:
echo    docker-compose -f docker-compose.infrastructure.yml up -d
echo.
echo 3. Start services:
echo    docker-compose -f infra\docker-compose-all-adapters.yml up -d
echo.

exit /b 0

REM ============================================================================
REM Function to create minimal environment
REM ============================================================================
:create_minimal_env

REM Get current date/time in UTC
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -AsUTC"') do set UTC_DATE=%%i

(
    echo # OpenEvolve Environment Configuration
    echo # Generated: %UTC_DATE%
    echo #
    echo # Following CLAUDE.md - Law of Configuration Explicitness:
    echo # - All values must be explicitly configured
    echo # - NO magic defaults - services will crash if required values are missing
    echo # - All timestamps in UTC ^(Law of UTC^)
    echo.
    echo # =============================================================================
    echo # Infrastructure Configuration
    echo # =============================================================================
    echo.
    echo # Event Bus ^(Redis^)
    echo EVENT_BUS_URL=redis://event-bus:6379
    echo REDIS_PORT=6379
    echo.
    echo # Orchestrator
    echo ORCHESTRATOR_PORT=8080
    echo.
    echo # =============================================================================
    echo # Logging Configuration
    echo # =============================================================================
    echo.
    echo # Log level: DEBUG, INFO, WARNING, ERROR
    echo LOG_LEVEL=INFO
    echo.
    echo # Log format: json or text
    echo LOG_FORMAT=json
    echo.
    echo # =============================================================================
    echo # Timezone ^(Law of UTC^)
    echo # =============================================================================
    echo.
    echo # All services MUST use UTC ^(Law of UTC^)
    echo TZ=UTC
    echo.
    echo # =============================================================================
    echo # API Endpoints ^(Docker Service Names^)
    echo # =============================================================================
    echo.
    echo # Core Project APIs ^(internal Docker service names^)
    echo LOONGFLOW_API_URL=http://loongflow-core:8050
    echo OPENEVOLVE_API_URL=http://openevolve-core:8000
    echo BUBBLELAB_API_URL=http://bubblelab-core:8501
    echo RAGBITS_API_URL=http://ragbits-core:8000
    echo.
    echo # =============================================================================
    echo # Service Configuration
    echo # =============================================================================
    echo.
    echo # Timeouts ^(milliseconds^)
    echo TIMEOUT_MS=30000
    echo LOONGFLOW_TIMEOUT_MS=30000
    echo OPENEVOLVE_TIMEOUT_MS=30000
    echo BUBBLELAB_TIMEOUT_MS=30000
    echo.
    echo # Retries
    echo MAX_RETRIES=3
    echo.
    echo # =============================================================================
    echo # LLM Provider Configuration ^(REQUIRED^)
    echo # =============================================================================
    echo.
    echo # OpenAI API Key ^(REQUIRED for LoongFlow^)
    echo # LOONGFLOW_LLM_API_KEY=sk-your-openai-api-key-here
    echo # OPENAI_API_KEY=sk-your-openai-api-key-here
    echo.
    echo # Anthropic API Key ^(optional^)
    echo # ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
    echo.
    echo # Google API Key ^(optional^)
    echo # GOOGLE_API_KEY=your-google-api-key-here
    echo.
    echo # =============================================================================
    echo # Database Configuration
    echo # =============================================================================
    echo.
    echo # Neo4j Configuration ^(for Graphiti^)
    echo NEO4J_URI=bolt://neo4j:7687
    echo NEO4J_USER=neo4j
    echo NEO4J_PASSWORD=your-neo4j-password-here
    echo.
    echo # =============================================================================
    echo # Contract Testing
    echo # =============================================================================
    echo.
    echo # Skip contract tests on startup ^(useful for development^)
    echo SKIP_CONTRACT_TESTS=false
    echo.
    echo # =============================================================================
    echo # Development Settings
    echo # =============================================================================
    echo.
    echo # Enable debug mode ^(set to false in production^)
    echo DEBUG=false
) > %ENV_FILE%

exit /b 0
