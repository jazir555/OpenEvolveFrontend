@echo off

echo Installing required packages...
pip install -r requirements.txt

set SCRIPT_PATH=%~dp0
set TARGET_URL=http://localhost:8501

echo Creating desktop shortcut for OpenEvolve...

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%USERPROFILE%\Desktop\OpenEvolve.url\"); $s.TargetPath = \"%TARGET_URL%\"; $s.Save()"

echo Shortcut created on your desktop.

start "" %TARGET_URL%

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

streamlit run main.py
