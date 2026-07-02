.PHONY: install demo benchmark benchmark-local video submit help test-master test-warroom

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000

# venv interpreter path differs by OS (POSIX activate is not portable to Windows).
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif

help:
	@echo "SupplyMind Makefile"
	@echo ""
	@echo "  make install          create .venv + install deps + .env template"
	@echo "  make demo             start FastAPI server, open master page"
	@echo "  make test-master      curl all 8 master-card health probes"
	@echo "  make test-warroom     POST a war-room scenario, print receipt sha256"
	@echo "  make benchmark        run the full benchmark harness (benchmark/run_full_benchmark.py)"
	@echo "  make benchmark-local  run receipt scripts (last step REQUIRES a local Ollama daemon)"
	@echo "  make video            OBS recording instructions"
	@echo "  make submit           tag the current release"

install:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env && echo "[i] Edit .env to add your keys"; fi
	@echo "[i] Activate the venv: (POSIX) source .venv/bin/activate  |  (Windows) .venv\\Scripts\\activate"

demo:
	@echo "[i] Starting server at http://$(HOST):$(PORT)/demo/master"
	$(PYTHON) -m uvicorn server.app:app --host $(HOST) --port $(PORT)

test-master:
	@curl -s -o /dev/null -w "/health                        %{http_code}\n" http://$(HOST):$(PORT)/health
	@curl -s -o /dev/null -w "/demo/master                   %{http_code}\n" http://$(HOST):$(PORT)/demo/master
	@curl -s -o /dev/null -w "/demo/hormuz-war-room/health   %{http_code}\n" http://$(HOST):$(PORT)/demo/hormuz-war-room/health
	@curl -s -o /dev/null -w "/demo/hormuz-war-room/ui       %{http_code}\n" http://$(HOST):$(PORT)/demo/hormuz-war-room/ui
	@curl -s -o /dev/null -w "/arena/health                  %{http_code}\n" http://$(HOST):$(PORT)/arena/health
	@curl -s -o /dev/null -w "/phoenix/status                %{http_code}\n" http://$(HOST):$(PORT)/phoenix/status
	@curl -s -o /dev/null -w "/replay/health                 %{http_code}\n" http://$(HOST):$(PORT)/replay/health
	@curl -s -o /dev/null -w "/live/health                   %{http_code}\n" http://$(HOST):$(PORT)/live/health

test-warroom:
	@curl -s -X POST http://$(HOST):$(PORT)/demo/hormuz-war-room \
	   -H 'Content-Type: application/json' \
	   -d '{"scenario_text":"Iran-Israel-US escalation restricts Hormuz","severity":0.85,"brent_price_usd_bbl":132,"duration_days":21,"enable_llm_judges":false,"include_recent_signals":false,"enable_openrouter_panel":false}' \
	   | $(PYTHON) -c "import json,sys; r=json.load(sys.stdin); print('elapsed', r['elapsed_s'], 's'); print('risk:', r['live_pipeline']['risk_level']); print('confidence:', r['confidence']['composite']); print('sha256:', r['receipt_sha256'])"

# Honest reproducibility harness — no local model daemon required.
benchmark:
	$(PYTHON) benchmark/run_full_benchmark.py
	@echo "[i] Results in benchmark/results/"

# Receipt scripts. The final step (ollama_v5_vs_frontier) REQUIRES a local
# Ollama daemon at 127.0.0.1:11434 and is NOT part of `make benchmark`.
benchmark-local:
	$(PYTHON) scripts/calibrate_conformal_from_harvest.py
	$(PYTHON) scripts/validate_ensemble_brent.py
	$(PYTHON) scripts/validate_war_room.py
	# bootstrap_leaderboard.py was DELETED (sorted-'paired' Wilcoxon on synthesized
	# samples -> fabricated significance, CLAIMS_LEDGER A1). The honest replacement is
	# real per-episode paired stats in scripts/pass27_killshot.py block B
	# (-> pass27_B_real_episodic_bootstrap.json); run that instead of this target's
	# deleted script.
	$(PYTHON) scripts/ollama_v5_vs_frontier.py   # requires local Ollama daemon
	@echo "[i] All receipts in tests/receipts/*.json"

video:
	@echo "Recording playbook: see FINAL_SUBMIT/DEMO_SCRIPT_90S.md"
	@echo "OBS preset: 1080p60 H.264 CRF 18, browser-only window capture"
	@echo "Pre-warm: hit /demo/master once 60s before recording"

submit:
	@if [ -n "$$(git status --porcelain)" ]; then \
	   echo "[!] uncommitted changes:"; git status --short; exit 1; \
	fi
	git tag -a v6.0-genesis -m "SupplyMind v6.0-genesis release"
	@echo "[i] Tagged v6.0-genesis. Push with: git push --tags"
