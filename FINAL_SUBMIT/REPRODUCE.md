# Reproduce SupplyMind from a fresh checkout

Tested on Windows 11 + RTX 4080 (12 GB) + 15.7 GB RAM, and Ubuntu 22.04 + RTX 4090.

## 5-command quick start

```bash
git clone <repo-url>
cd Sleep-Token
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                  # then edit with your 4 keys
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://127.0.0.1:8000/demo/master in a browser
```

If you don't have GPU/Ollama: the demo still runs — judge layer falls back to a deterministic severity rubric and reports `data_source_flags.live_pipeline = "deterministic_rubric_fallback"` honestly.

## Required environment keys

| Key | Where to get it | Used by |
|---|---|---|
| `OPENROUTER_API_KEY` | https://openrouter.ai/ (free tier OK) | War-Room 6-judge panel · cross-corpus α · panel-consensus |
| `EIA_API_KEY` | https://www.eia.gov/opendata/register.php | Live Brent / WTI / natgas in fan-out |
| `NASA_FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/map_key/ | Wildfire fan-out signal |
| `GFW_API_TOKEN` | https://globalfishingwatch.org/our-apis/ | Tanker AIS in fan-out |

`.env.example` template:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_APP_NAME=SupplyMind
OPENROUTER_SITE_URL=https://supplymind.ai
EIA_API_KEY=...
NASA_FIRMS_MAP_KEY=...
GFW_API_TOKEN=...
```

## Foundation models (~50 GB total)

The 13 foundation models live under `models/`. They are NOT shipped with the repo (too large). To download:

```bash
bash scripts/download_models.sh    # ~50 GB, ~30 min on 1 Gbps
```

This pulls from local Ollama for the LLMs and HuggingFace for the embedders/forecasters. If you skip this, the demo still works in degraded mode — the master page LEDs will show amber for /phoenix/status.

## Reproducing every receipt

```bash
# 1. Conformal calibration (real harvest, ~3 min)
python scripts/calibrate_conformal_from_harvest.py

# 2. Cross-corpus Krippendorff α (~1 hour with rate limits)
python scripts/compute_cross_corpus_alpha.py

# 3. Ensemble Brent backtest (~2 min)
python scripts/validate_ensemble_brent.py

# 4. War-Room historical backtest (~2 min)
python scripts/validate_war_room.py
# OR via HTTP:
curl -X POST http://127.0.0.1:8000/demo/hormuz-war-room/validate

# 5. Bootstrap CI95 leaderboard (~5 sec)
python scripts/bootstrap_leaderboard.py

# 6. Ollama v5 vs frontier (~3 min)
python scripts/ollama_v5_vs_frontier.py

# 7. HetGAT all 3 graphs (~30 min on RTX 4080)
python -m versions.v5_phoenix.gnn_v2.train_hetgat --graph all --epochs 200

# 8. RAP-XC training on harvested transitions (~20 sec on RTX 4080)
python -c "from versions.v5_phoenix.rap_xc.train import train_rapxc; train_rapxc()"
```

All produce JSON receipts at `tests/receipts/*.json`.

## One-shot: `make` everything

If GNU Make is installed:

```bash
make install        # pip + ollama models + faiss index + .env template
make demo           # start server, open master page
make benchmark      # run all 8 reproducibility scripts above
make video          # OBS-ready preset
make submit         # final commit + tag
```

## Docker

```bash
docker build -f FINAL_SUBMIT/docker/Dockerfile.api -t supplymind-api .
docker run -p 8000:8000 --env-file .env supplymind-api
# OR with compose for full stack (api + ollama + redis):
docker compose -f FINAL_SUBMIT/docker/docker-compose.yml up
```

Health check:

```bash
curl http://127.0.0.1:8000/health                                # base
curl http://127.0.0.1:8000/demo/hormuz-war-room/health           # war-room
curl http://127.0.0.1:8000/phoenix/status                        # phoenix v5
curl http://127.0.0.1:8000/live/health                           # 20-source fan-out
```

## Hardware floor

| Component | Minimum | Notes |
|---|---|---|
| GPU | 12 GB VRAM | Q4_K_M discipline; bf16 RAP-XC; OLLAMA_MAX_LOADED_MODELS=1 |
| RAM | 16 GB | mxbai + FAISS in process |
| Disk | 70 GB | 13 models ~50 GB, EMDAT corpus ~3 GB, harvest ~1.5 GB |
| Python | 3.11 | type-hint syntax used throughout |
| OS | Win 11 / Ubuntu 22.04 / macOS 14 | tested on first two |

## Known cold-start latencies

| First-time hit | Latency |
|---|---|
| mxbai-embed-large load | ~10 s |
| FAISS HNSW load | ~2 s |
| TimesFM-2 load | ~8 s |
| Chronos-Bolt load | ~3 s |
| TabPFN-v2 load | ~2 s |
| Ollama analyst v5 load | ~4 s |
| First `/demo/hormuz-war-room` POST | ~22 s |
| Subsequent POSTs (cached singletons) | ~5-12 s |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torchvision.io image extension fail [WinError 127]` | Harmless; ignore. Or `pip install pillow --upgrade`. |
| `ConnectionRefusedError` on Ollama | Run `ollama serve` in another terminal, or set `enable_llm_judges=false` in war-room request. |
| `429 Too Many Requests` on OpenRouter | Free-tier rate limits — 2/6 typically rate-limit. We report 4/6 succeeded honestly. |
| `Module 'faiss' not found` | `pip install faiss-cpu` |
| `pydantic ValidationError: duration_days <= 1200` | Real cap; don't pass durations > 1200 days. |
| Master page LEDs amber | A subsystem is degraded; check the relevant `/health` endpoint for the specific reason. |
