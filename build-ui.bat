@echo off
setlocal

cd /d "%~dp0frontend"

if not exist "node_modules" (
  npm install --strict-ssl=false --registry=https://registry.npmjs.org/ --no-audit --no-fund
)

npm run build

echo.
echo Built UI:
echo %~dp0frontend\dist\index.html
echo.
echo Note: opening this HTML directly still needs the backend running for live model analysis.
pause
