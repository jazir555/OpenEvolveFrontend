@echo off
cd /d "%~dp0"
echo Script started.
setlocal enabledelayedexpansion
echo Delayed expansion enabled.

set VENV_DIR=env
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set PIP_EXE=%VENV_DIR%\Scripts\pip.exe
echo Variables VENV_DIR, PYTHON_EXE, PIP_EXE defined.

:: --- Manual Python Command Setting ---
:: IMPORTANT: This path has been set based on your input.
set "PYTHON_COMMAND=C:\Users\mmeadow\AppData\Local\Programs\Python\Python311\python.exe"

echo Using Python command: !PYTHON_COMMAND!

:: Check if virtual environment exists and skip creation/installation if it does
if exist %VENV_DIR% (
    echo Virtual environment already exists. Skipping creation and package installation.
) else (
    echo Creating virtual environment with !PYTHON_COMMAND!...
    !PYTHON_COMMAND! -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo Activating virtual environment...
    call %VENV_DIR%\Scripts\activate.bat

    :: Clear pip cache
    echo Clearing pip cache...
    %PIP_EXE% cache purge >> pip_install_log.txt 2>&1
    if errorlevel 1 (
        echo WARNING: Failed to clear pip cache. This might not be critical. >> pip_install_log.txt
    )

    :: Set PYTHONIOENCODING for pip to handle UnicodeDecodeError (keep for robustness)
    set PYTHONIOENCODING=utf-8

    :: Install core scientific dependencies first
    echo Installing numpy and scipy...
    %PIP_EXE% install numpy scipy >> pip_install_log.txt 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to install numpy and scipy. Check pip_install_log.txt for details.
        pause
        exit /b 1
    )

    echo Installing required Python packages...
    %PIP_EXE% install -r requirements.txt > pip_install_log.txt 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to install Python packages from requirements.txt. Check pip_install_log.txt for details.
        pause
        exit /b 1
    )

    :: Install aleph-alpha-client separately
    echo Installing aleph-alpha-client...
    %PIP_EXE% install aleph-alpha-client==3.1.0 >> pip_install_log.txt 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to install aleph-alpha-client. Check pip_install_log.txt for details.
        pause
        exit /b 1
    )

    echo Installing OpenEvolve in editable mode...
    %PIP_EXE% install -e .\openevolve >> pip_install_log.txt 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to install OpenEvolve in editable mode. Check pip_install_log.txt for details.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

echo Creating desktop shortcut for OpenEvolve...
set TARGET_URL=http://localhost:8501
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%USERPROFILE%\Desktop\OpenEvolve.url\"); $s.TargetPath = \"%TARGET_URL%\"; $s.Save()"
echo Shortcut created on your desktop.

:: --- Configure Streamlit config.toml ---
:: Create .streamlit directory if it's not exist
if not exist ".streamlit" mkdir ".streamlit"

:: Write config.toml content
(
    echo [general]
    echo showEmailPrompt = false
    echo(
    echo [server]
    echo headless = true
    echo(
    echo [theme]
    echo primaryColor = "#a0a0a0"
) > ".streamlit\config.toml"

echo Streamlit config.toml configured.

echo Launching Streamlit application...
start "" %TARGET_URL%
%PYTHON_EXE% -m streamlit run main.py

pause