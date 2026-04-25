.PHONY: install demo benchmark video submit help test-master test-warroom

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000

help:
	@echo "SupplyMind Final-Submit Makefile"
	@echo ""
	@echo "  make install      install pip deps + .env template"
	@echo "  make demo         start FastAPI server, open master page"
	@echo "  make test-master  curl all 9 master-card health probes"
	@echo "  make test-warroom POST a war-room scenario, print receipt sha256"
	@echo "  make benchmark    run 8 reproducibility scripts, save receipts"
	@echo "  make video        OBS recording instructions"
	@echo "  make submit       final commit + tag"

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env && echo "[i] Edit .env to add your 4 keys"; fi

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

benchmark:
	$(PYTHON) scripts/calibrate_conformal_from_harvest.py
	$(PYTHON) scripts/validate_ensemble_brent.py
	$(PYTHON) scripts/validate_war_room.py
	$(PYTHON) scripts/bootstrap_leaderboard.py
	$(PYTHON) scripts/ollama_v5_vs_frontier.py
	@echo "[i] All receipts in tests/receipts/*.json"

video:
	@echo "Recording playbook: see FINAL_SUBMIT/DEMO_SCRIPT_90S.md"
	@echo "OBS preset: 1080p60 H.264 CRF 18, browser-only window capture"
	@echo "Pre-warm: hit /demo/master once 60s before recording"

submit:
	@if [ -n "$$(git status --porcelain)" ]; then \
	   echo "[!] uncommitted changes:"; git status --short; exit 1; \
	fi
	git tag -a v4.0-final-submit -m "SupplyMind final submit · 100% war-room backtest · 100% ensemble Brent · 0.9001 conformal"
	@echo "[i] Tagged v4.0-final-submit. Push with: git push --tags"
