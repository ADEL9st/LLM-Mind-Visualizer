#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# LLM Mind Visualizer — start script (Linux / macOS)
# ──────────────────────────────────────────────────────────────
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Backend virtual environment ──────────────────────────────
if [ ! -d "backend/.venv" ]; then
  echo "Creating backend virtual environment..."
  python3 -m venv backend/.venv
fi

source backend/.venv/bin/activate

if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing backend dependencies..."
  pip install -r backend/requirements.txt -q
fi

if ! python -c "import torch" 2>/dev/null; then
  echo "Installing ML dependencies (PyTorch, Transformers, etc.)..."
  pip install -r backend/requirements-ml.txt -q
fi

# ── Frontend dependencies ────────────────────────────────────
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  cd frontend && npm install --no-audit --no-fund -q && cd "$ROOT"
fi

# ── Launch servers ───────────────────────────────────────────
echo ""
echo "Starting backend on http://127.0.0.1:8000 ..."
cd "$ROOT/backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://127.0.0.1:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

sleep 3

# Open browser (best-effort)
if command -v xdg-open &>/dev/null; then
  xdg-open http://127.0.0.1:5173
elif command -v open &>/dev/null; then
  open http://127.0.0.1:5173
else
  echo "Open http://127.0.0.1:5173 in your browser."
fi

echo ""
echo "Backend:  http://127.0.0.1:8000  (PID $BACKEND_PID)"
echo "Frontend: http://127.0.0.1:5173  (PID $FRONTEND_PID)"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait for both; Ctrl+C kills them
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
