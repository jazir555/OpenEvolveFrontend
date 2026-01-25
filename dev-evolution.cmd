@echo off
setlocal
set ROOT=%~dp0

echo Starting BubbleLab API (http://localhost:3001)...
start "bubblelab-api" cmd /k "cd /d %ROOT%BubbleLab\\apps\\bubblelab-api && pnpm dev"

echo Starting OpenEvolve Gateway (http://localhost:8000)...
start "openevolve-gateway" cmd /k "cd /d %ROOT%api\\gateway && python main.py"

echo Starting BubbleLab Studio (http://localhost:3000)...
start "bubble-studio" cmd /k "cd /d %ROOT%BubbleLab\\apps\\bubble-studio && pnpm dev"

echo.
echo BubbleLab Studio: http://localhost:3000
echo BubbleLab API:    http://localhost:3001
echo OpenEvolve API:   http://localhost:8000
echo.
echo If the browser still cannot connect, check system proxy/firewall settings.
