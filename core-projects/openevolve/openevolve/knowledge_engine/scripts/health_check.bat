@echo off
REM Neo4j Health Check Script (Windows)
REM OpenEvolve Knowledge Engine - Phase 1.1.1

setlocal enabledelayedexpansion

echo ================================================
echo Neo4j Health Check - OpenEvolve Knowledge Engine
echo ================================================
echo.

REM Configuration
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=openevolve2026
set NEO4J_HTTP_URI=http://localhost:7474

REM Check if curl is available
where curl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] curl is not available. Please install curl or use WSL.
    exit /b 1
)

REM Check HTTP endpoint
echo [1/6] Checking HTTP endpoint...
curl -s -o nul -w "%%{http_code}" %NEO4J_HTTP_URI% > temp_http_code.txt
set /p HTTP_CODE=<temp_http_code.txt
del temp_http_code.txt

if "%HTTP_CODE%"=="200" (
    echo [OK] HTTP endpoint is reachable (HTTP %HTTP_CODE%)
) else if "%HTTP_CODE%"=="302" (
    echo [OK] HTTP endpoint is reachable (HTTP %HTTP_CODE%)
) else (
    echo [FAIL] HTTP endpoint is not reachable (HTTP %HTTP_CODE%)
    goto :error
)
echo.

REM Check if cypher-shell is available
echo [2/6] Checking cypher-shell availability...
where cypher-shell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] cypher-shell is not available
    echo [INFO] Some checks will be skipped
    set CYPHER_AVAILABLE=0
) else (
    echo [OK] cypher-shell is available
    set CYPHER_AVAILABLE=1
)
echo.

REM Check Bolt connection
echo [3/6] Checking Bolt protocol connection...
if %CYPHER_AVAILABLE%==1 (
    echo RETURN 1 | cypher-shell -a %NEO4J_URI% -u %NEO4J_USER% -p %NEO4J_PASSWORD% >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [OK] Bolt protocol connection successful
    ) else (
        echo [FAIL] Bolt protocol connection failed
        goto :error
    )
) else (
    echo [SKIP] Cannot check Bolt (cypher-shell not available)
)
echo.

REM Check database version
echo [4/6] Checking database version...
if %CYPHER_AVAILABLE%==1 (
    for /f "tokens=*" %%i in ('echo CALL dbms.components^() YIELD versions RETURN versions[0^] as version ^| cypher-shell -a %NEO4J_URI% -u %NEO4J_USER% -p %NEO4J_PASSWORD% 2^>nul ^| findstr /R "^[0-9]"') do set VERSION=%%i
    if defined VERSION (
        echo [OK] Neo4j version: %VERSION%
    ) else (
        echo [WARNING] Could not determine database version
    )
) else (
    echo [SKIP] Cannot check version (cypher-shell not available)
)
echo.

REM Check database statistics
echo [5/6] Checking database statistics...
if %CYPHER_AVAILABLE%==1 (
    for /f "tokens=*" %%i in ('echo MATCH ^(n^) RETURN count^(n^) as count ^| cypher-shell -a %NEO4J_URI% -u %NEO4J_USER% -p %NEO4J_PASSWORD% 2^>nul ^| findstr /R "^[0-9]"') do set NODE_COUNT=%%i
    if defined NODE_COUNT (
        echo [OK] Database contains %NODE_COUNT% nodes
        if %NODE_COUNT%==0 (
            echo [WARNING] Database is empty. Run init_neo4j.cypher to initialize.
        )
    ) else (
        echo [WARNING] Could not retrieve node count
    )
) else (
    echo [SKIP] Cannot check statistics (cypher-shell not available)
)
echo.

REM Check container status
echo [6/6] Checking container status...
where docker >nul 2>&1
if %ERRORLEVEL% equ 0 (
    docker ps --filter "name=openevolve-neo4j" --format "{{.Status}}" | findstr "Up" >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [OK] Neo4j container is running
    ) else (
        echo [WARNING] Neo4j container is not running or not found
    )
) else (
    echo [SKIP] Docker is not available
)
echo.

REM Summary
echo ================================================
echo Health Check Summary
echo ================================================
echo [OK] Neo4j is reachable and operational!
echo.
echo Quick Commands:
echo   - Open Neo4j Browser:  open http://localhost:7474
echo   - View logs:           docker logs openevolve-neo4j
echo   - Stop Neo4j:          docker stop openevolve-neo4j
echo.
goto :end

:error
echo.
echo ================================================
echo Health Check Failed
echo ================================================
echo.
echo Please check the errors above and verify:
echo   - Neo4j container is running
echo   - Ports 7474 and 7687 are available
echo   - Credentials are correct
echo.
exit /b 1

:end
exit /b 0
