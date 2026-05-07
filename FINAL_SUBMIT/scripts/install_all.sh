#!/usr/bin/env bash
# install_all.sh — fresh-machine setup for SupplyMind final submit.
# Tested on Ubuntu 22.04 + RTX 4090; should work on Windows WSL2 + RTX 4080.

set -euo pipefail

echo "[1/6] Checking Python 3.11..."
python3.11 --version || { echo "Install Python 3.11 first"; exit 1; }

echo "[2/6] Creating venv..."
python3.11 -m venv .venv
source .venv/bin/activate

echo "[3/6] Installing pip deps..."
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "[4/6] Setting up .env..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "[!] Edit .env to fill in:"
  echo "    OPENROUTER_API_KEY (https://openrouter.ai/keys)"
  echo "    EIA_API_KEY        (https://www.eia.gov/opendata/register.php)"
  echo "    NASA_FIRMS_MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/)"
  echo "    GFW_API_TOKEN      (https://globalfishingwatch.org/our-apis/)"
fi

echo "[5/6] Pulling Ollama models (skip if Ollama not installed)..."
if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen2.5:14b || true
  ollama pull mistral-nemo || true
  ollama pull deepseek-r1 || true
  # Custom analyst v5 — built from Modelfile in repo
  if [ -f versions/v4_arcadia_live/features/Modelfile.analyst_v5 ]; then
    ollama create supplymind-analyst:v5 -f versions/v4_arcadia_live/features/Modelfile.analyst_v5 || true
  fi
else
  echo "[i] Ollama not installed; skipping LLM model pulls. Install: https://ollama.com"
fi

echo "[6/6] Building FAISS index for crisis library v2..."
if [ ! -f versions/v4_arcadia_live/scenarios/crisis_library_v2.faiss ]; then
  python -c "from versions.v4_arcadia_live.scenarios.library_v2_search import singleton; singleton('warm-up', k=1)"
fi

echo ""
echo "[done] Run with:  python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"
echo "       Open:      http://127.0.0.1:8000/demo/master"
