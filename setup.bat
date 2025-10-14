@echo off

set VENV_DIR=env
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set PIP_EXE=%VENV_DIR%\Scripts\pip.exe

:: Check if virtual environment exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

echo Installing required Python packages...
%PIP_EXE% install -r requirements.txt

echo Installing OpenEvolve in editable mode...
%PIP_EXE% install -e .\openevolve

echo Creating desktop shortcut for OpenEvolve...
set TARGET_URL=http://localhost:8501
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%USERPROFILE%\Desktop\OpenEvolve.url\"); $s.TargetPath = \"%TARGET_URL%\"; $s.Save()"
echo Shortcut created on your desktop.

:: --- Configure Streamlit config.toml ---
:: Create .streamlit directory if it doesn't exist
if not exist ".streamlit" mkdir ".streamlit"

:: Write config.toml content
(
    echo [general]
    echo showEmailPrompt = false
    echo.
    echo [server]
    echo headless = true
    echo.
    echo [theme]
    echo primaryColor = "#a0a0a0"
) > ".streamlit\config.toml"

echo Streamlit config.toml configured.

echo Launching Streamlit application...
start "" %TARGET_URL%
%PYTHON_EXE% -m streamlit run main.py
