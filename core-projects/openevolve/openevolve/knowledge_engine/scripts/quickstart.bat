@echo off
REM Neo4j Quick Start Script (Windows)
REM OpenEvolve Knowledge Engine - Phase 1.1.1

setlocal enabledelayedexpansion

echo ================================================
echo Neo4j Quick Start - OpenEvolve Knowledge Engine
echo ================================================
echo.

REM Check if Docker is available
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not available. Please install Docker Desktop.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
where docker-compose >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Compose is not available. Please install Docker Compose.
    pause
    exit /b 1
)

REM Set environment (default: dev)
set ENV=%1
if "%ENV%"=="" set ENV=dev

if "%ENV%"=="dev" (
    set ENV_FILE=knowledge_engine\config\neo4j.dev.env
    echo [INFO] Starting Neo4j in DEVELOPMENT mode...
) else if "%ENV%"=="prod" (
    set ENV_FILE=knowledge_engine\config\neo4j.prod.env
    echo [INFO] Starting Neo4j in PRODUCTION mode...
) else (
    echo [ERROR] Invalid environment. Use: quickstart.bat [dev^|prod]
    pause
    exit /b 1
)

echo.
echo [1/5] Creating data directories...
if not exist "data\neo4j\data" mkdir "data\neo4j\data"
if not exist "data\neo4j\logs" mkdir "data\neo4j\logs"
if not exist "data\neo4j\import" mkdir "data\neo4j\import"
if not exist "data\neo4j\plugins" mkdir "data\neo4j\plugins"
if not exist "data\neo4j\backups" mkdir "data\neo4j\backups"
echo [OK] Data directories created
echo.

echo [2/5] Checking for existing containers...
docker ps -a --filter "name=openevolve-neo4j" --format "{{.Names}}" | findstr "openevolve-neo4j" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Stopping existing Neo4j container...
    docker stop openevolve-neo4j >nul 2>&1
    docker rm openevolve-neo4j >nul 2>&1
    echo [OK] Existing container removed
) else (
    echo [OK] No existing container found
)
echo.

echo [3/5] Starting Neo4j container...
docker-compose -f docker-compose.neo4j.yml --env-file %ENV_FILE% up -d neo4j
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start Neo4j container
    pause
    exit /b 1
)
echo [OK] Neo4j container started
echo.

echo [4/5] Waiting for Neo4j to be ready...
set MAX_WAIT=60
set WAIT_TIME=0
set HEALTH_CHECK_INTERVAL=2

:wait_loop
if %WAIT_TIME% geq %MAX_WAIT% (
    echo [ERROR] Neo4j failed to start within %MAX_WAIT% seconds
    echo Check logs with: docker logs openevolve-neo4j
    pause
    exit /b 1
)

docker exec openevolve-neo4j wget -O /dev/null http://localhost:7474 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Neo4j is ready!
    goto :neo4j_ready
)

echo|set /p="."
timeout /t %HEALTH_CHECK_INTERVAL% >nul
set /a WAIT_TIME+=2
goto :wait_loop

:neo4j_ready
echo.

echo [5/5] Initializing database...
docker exec openevolve-neo4j cypher-shell -u neo4j -p openevolve2026 -f /scripts/init_neo4j.cypher >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Database initialized
) else (
    echo [WARNING] Database initialization had warnings. Check logs.
)
echo.

REM Summary
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Neo4j is now running and ready to use!
echo.
echo Connection Details:
echo   - Bolt URI:       bolt://localhost:7687
echo   - HTTP URI:       http://localhost:7474
echo   - Username:       neo4j
echo   - Password:       openevolve2026
echo.
echo Quick Commands:
echo   - Open Neo4j Browser:  start http://localhost:7474
echo   - Run health check:    knowledge_engine\scripts\health_check.bat
echo   - View logs:           docker logs -f openevolve-neo4j
echo   - Stop Neo4j:          docker stop openevolve-neo4j
echo.
echo For more information, see: knowledge_engine\docs\neo4j_setup.md
echo.

pause
