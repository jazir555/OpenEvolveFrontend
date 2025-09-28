@echo off

echo Installing required packages...
pip install -r requirements.txt

set SCRIPT_PATH=%~dp0
set TARGET_URL=http://localhost:8501

echo Creating desktop shortcut for OpenEvolve...

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%USERPROFILE%\Desktop\OpenEvolve.url\"); $s.TargetPath = \"%TARGET_URL%\"; $s.Save()"

echo Shortcut created on your desktop.

start "" %TARGET_URL%

streamlit run main.py
