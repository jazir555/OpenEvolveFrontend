@echo off
REM ###########################################################################
REM OpenEvolve BubbleLabs Backend - Windows Startup Script
REM
REM This script starts the Python backend server with proper environment setup
REM ###########################################################################

setlocal enabledelayedexpansion

echo ========================================
echo OPENEVOLVE BUBBLELABS BACKEND STARTUP
echo ========================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
    echo.
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
    echo.
) else (
    echo [WARNING] No virtual environment found
    echo It's recommended to create one with:
    echo   python -m venv venv
    echo.
)

REM Check if requirements.txt exists
if exist "requirements.txt" (
    echo Checking dependencies...
    python -c "import fastapi" >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo Installing dependencies...
        pip install -q -r requirements.txt
        echo [OK] Dependencies installed
        echo.
    ) else (
        echo [OK] Dependencies already installed
        echo.
    )
) else (
    echo [WARNING] requirements.txt not found
    echo.
)

echo ========================================
echo STARTING SERVER
echo ========================================
echo.

REM Start the server
python start_server.py

REM Exit cleanup
set EXIT_CODE=%ERRORLEVEL%
echo.
if %EXIT_CODE% EQU 0 (
    echo [OK] Server stopped successfully
) else (
    echo [ERROR] Server exited with code %EXIT_CODE%
)

pause
