@echo off
REM Validate required environment variables
REM Following CLAUDE.md - Law of Configuration Explicitness

setlocal enabledelayedexpansion

set ENV_FILE=.env

echo ================================================
echo OpenEvolve Environment Validation
echo ================================================
echo.

if not exist %ENV_FILE% (
    echo ❌ .env file not found
    echo.
    echo Run: scripts\setup-env.cmd
    exit /b 1
)

echo 🔍 Validating environment variables...
echo.

REM Load environment variables from .env file
for /f "usebackq tokens=1,2 delims==" %%i in ("%ENV_FILE%") do (
    REM Skip comments and empty lines
    echo %%i | findstr /r "^#" >nul
    if errorlevel 1 (
        if not "%%i"=="" (
            set %%i=%%j
        )
    )
)

REM Track validation results
set /a REQUIRED_ERRORS=0
set /a OPTIONAL_WARNINGS=0
set /a SUCCESS_COUNT=0

REM ============================================================================
REM Validation Functions
REM ============================================================================

:check_required
set VAR_NAME=%1
set DESCRIPTION=%2

REM Check if variable is set
setlocal enabledelayedexpansion
if "!%VAR_NAME%!"=="" (
    echo ❌ REQUIRED: %VAR_NAME%
    echo    %DESCRIPTION%
    set /a REQUIRED_ERRORS=%REQUIRED_ERRORS%+1
) else (
    REM Check if it's a placeholder
    set VALUE=!%VAR_NAME%!
    echo !VALUE! | findstr /i "your- changeme here" >nul
    if not errorlevel 1 (
        echo ⚠️  PLACEHOLDER: %VAR_NAME%
        echo    %DESCRIPTION%
        echo    Current value: !VALUE!
        set /a OPTIONAL_WARNINGS=%OPTIONAL_WARNINGS%+1
    ) else (
        echo ✅ %VAR_NAME%
        set /a SUCCESS_COUNT=%SUCCESS_COUNT%+1
    )
)
endlocal
exit /b 0

:check_optional
set VAR_NAME=%1
set DEFAULT_VALUE=%2

if "!%VAR_NAME%!"=="" (
    echo ℹ️  OPTIONAL ^(not set^): %VAR_NAME%
    echo    Will use default: %DEFAULT_VALUE%
) else (
    echo ✅ %VAR_NAME% = !%VAR_NAME%!
    set /a SUCCESS_COUNT=%SUCCESS_COUNT%+1
)
exit /b 0

REM ============================================================================
REM Check Infrastructure Configuration
REM ============================================================================
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Infrastructure Configuration
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :check_required EVENT_BUS_URL "Event bus connection URL"
call :check_required TZ "Timezone ^(should be UTC^)"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo API Endpoints
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :check_required LOONGFLOW_API_URL "LoongFlow Core API endpoint"
call :check_required OPENEVOLVE_API_URL "OpenEvolve API endpoint"
call :check_required BUBBLELAB_API_URL "BubbleLab API endpoint"
call :check_required RAGBITS_API_URL "RagBits API endpoint"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Service Configuration
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :check_optional TIMEOUT_MS 30000
call :check_optional MAX_RETRIES 3
call :check_optional LOG_LEVEL INFO
call :check_optional LOG_FORMAT json

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo LLM Provider Configuration
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :check_required LOONGFLOW_LLM_API_KEY "OpenAI API key for LoongFlow"
call :check_optional OPENAI_API_KEY "sk-..."
call :check_optional ANTHROPIC_API_KEY "sk-ant-..."
call :check_optional GOOGLE_API_KEY "..."

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Database Configuration
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :check_required NEO4J_URI "Neo4j connection URI"
call :check_required NEO4J_USER "Neo4j username"
call :check_required NEO4J_PASSWORD "Neo4j password"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Development Settings
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :check_optional DEBUG "false"
call :check_optional SKIP_CONTRACT_TESTS "false"

REM ============================================================================
REM Validation Summary
REM ============================================================================
echo.
echo ================================================
echo Validation Summary
echo ================================================
echo ✅ Valid variables: %SUCCESS_COUNT%
echo ⚠️  Placeholders: %OPTIONAL_WARNINGS%
echo ❌ Missing required: %REQUIRED_ERRORS%
echo.

if %REQUIRED_ERRORS% GTR 0 (
    echo ❌ VALIDATION FAILED
    echo.
    echo Required environment variables are missing!
    echo.
    echo Please set the required variables in .env before starting services.
    echo.
    echo Edit .env:
    echo    notepad .env
    echo.
    exit /b 1
) else if %OPTIONAL_WARNINGS% GTR 0 (
    echo ⚠️  VALIDATION PASSED WITH WARNINGS
    echo.
    echo Some variables contain placeholder values.
    echo Services may not start correctly without proper values.
    echo.
    echo Please review and update placeholder values in .env
    exit /b 0
) else (
    echo ✅ VALIDATION PASSED
    echo.
    echo All required environment variables are set!
    echo.
    echo You can now start services:
    echo.
    echo 1. Start infrastructure:
    echo    docker-compose -f docker-compose.infrastructure.yml up -d
    echo.
    echo 2. Start adapters:
    echo    docker-compose -f infra\docker-compose-all-adapters.yml up -d
    echo.
    echo 3. Check health:
    echo    scripts\health-check.cmd
    echo.
    exit /b 0
)
