@echo off
setlocal

cd /d "%~dp0"

if not exist "frontend\dist\index.html" (
  call "%~dp0build-ui.bat"
)

start "" "%~dp0frontend\dist\index.html"
