@echo off
setlocal

cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
  echo Creating backend virtual environment...
  py -m venv backend\.venv
)

if not exist "backend\.venv\Lib\site-packages\fastapi" (
  echo Installing backend dependencies...
  cd /d "%~dp0backend"
  .\.venv\Scripts\python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
  cd /d "%~dp0"
)

if not exist "backend\.venv\Lib\site-packages\torch" (
  echo Installing ML dependencies: PyTorch, Transformers, etc. ...
  cd /d "%~dp0backend"
  .\.venv\Scripts\python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-ml.txt
  cd /d "%~dp0"
)

if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd /d "%~dp0frontend"
  npm install --strict-ssl=false --registry=https://registry.npmjs.org/ --no-audit --no-fund
  cd /d "%~dp0"
)

start "LLM Mind Visualizer API" cmd /k "cd /d "%~dp0backend" && .\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "LLM Mind Visualizer UI" cmd /k "cd /d "%~dp0frontend" && npm run dev"

timeout /t 4 /nobreak > nul
start http://127.0.0.1:5173

echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.
echo This window can be closed. Keep the two server windows open while using the app.
pause
