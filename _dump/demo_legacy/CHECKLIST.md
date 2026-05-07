# SupplyMind v3.0-arcadia — Final Demo Checklist

Everything a judge would want, in one place. Updated 2026-04-18.

## 📺 What to watch / read / try

### 2-minute path (for the busiest judge)
1. **Landing page**: [`demo/LANDING_PAGE.md`](LANDING_PAGE.md) or HF Space README
2. **Pitch deck**: open [`demo/SupplyMind_pitch.html`](SupplyMind_pitch.html) in a browser (5 slides, landscape A4)
3. **3 curl demos** in the landing page (risk, forecast, RAG)
4. **Read-only demo transcript**: [`demo/DEMO_TRANSCRIPT.md`](DEMO_TRANSCRIPT.md) — all 8 scenes with exact commands (no video needed)
5. **Real-cited external credibility**: [`../EXTERNAL_CREDIBILITY.md`](../EXTERNAL_CREDIBILITY.md) — 10+ published quotes validating design choices
6. **Deploy playbook**: [`../DEPLOY_HF_SPACE.md`](../DEPLOY_HF_SPACE.md) — phoenix rebuild walkthrough

### 10-minute path
4. **Demo video** (when recorded): follow [`demo/DEMO_VIDEO_SCRIPT.md`](DEMO_VIDEO_SCRIPT.md) — 3-min OBS screencap
5. **Unified MODEL_CARD**: [`../MODEL_CARD.md`](../MODEL_CARD.md) — every benchmark, every honest finding with a world-class fix
6. **PyTorch engineering story**: [`../PYTORCH_STORY.md`](../PYTORCH_STORY.md) — 11 non-trivial items

### Deep-dive path
7. **Benchmarks vs public**: [`../BENCHMARKS_VS_PUBLIC.md`](../BENCHMARKS_VS_PUBLIC.md) — M5, MTEB, MuJoCo, LLM-as-judge comparison + $1M-ambition appendix
8. **Audit plan**: [`../AUDIT_PLAN.md`](../AUDIT_PLAN.md) — coverage matrix of every directive → every commit
9. **Final demo plan**: [`../FINAL_DEMO.md`](../FINAL_DEMO.md) — status table per-layer + killer-gap tracking
10. **Failure table**: [`../FAILURE_TABLE.md`](../FAILURE_TABLE.md) — every v1/v2 scar with v3 resolution
11. **R4 reproducibility challenge**: [`../challenges/R4_RUBRIC_CHALLENGE.md`](../challenges/R4_RUBRIC_CHALLENGE.md) — invitation to beat 75% accuracy / α=0.70

### Reproduce everything
```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
pip install -r requirements.txt
pytest tests/ -q        # 173 passing in 2m14s
uvicorn server.app:app  # serve on :8000
```

---

## 🎬 Asset inventory for GitHub Release

Run `bash scripts/release_assets.sh` (requires `gh` CLI) to auto-upload:

- **Plots** (`v3_arcadia/plots/**/*.png`):
  - `gethsemane/r6_gethsemane.png` — PPO vs random vs greedy bars with error bars
  - `gethsemane/learning_curves.png` — 3-task reward-vs-steps curves
  - `provider/r6_provider.png` — GNN v1 F1 on BFS task
  - `provider/r6_provider_v2.png` — GNN v2 arrival-time MAE
  - `aqua_regia/r6_aqua_regia.png` — conformal pooled coverage
  - `aqua_regia/r6_aqua_regia_v2.png` — per-horizon vs pooled
  - `euclidian/r6_euclidian.png` — 8,100-ep bootstrap CI95
  - `dangerous/r4v2_heatmap.png` — judge agreement heatmap
  - `dangerous/r4v2_ablation.png` — 2-judge vs 3-judge Pareto
  - `dangerous/r4v2_calibration.png` — ECE reliability diagrams
  - `dangerous/r4v2_confusion.png` — per-judge GT confusion matrices
  - `dangerous/r4v2_escalation.png` — escalation tier distribution
  - `dangerous/r4v2_latency.png` — judge latency box plot
  - `granite/r5_metrics.png` — 8 pipelines × 5 metrics
  - `granite/r5_hard_redemption.png` — reranker regime plot
  - `granite/r5_corpus.png` — corpus composition pie
  - `granite/r5_latency_vs_mrr.png` — Pareto
  - `granite/r5_per_query_heatmap.png` — per-query P@1
  - `past_self/r3_summary.png` — 8 FRED targets × 3 horizons
  - `gethsemane/r6_masking_ablation.png` — masked vs unmasked PPO isolated lift (+26.8%)

- **JSONs** (`v3_arcadia/results/*.json`):
  - `R1_VERIFIED.json` — 13-model verification
  - `R1_QWEN_VL_DOWNSTREAM.json` — VL downstream + hw constraint doc
  - `R2_CARAMEL.json`, `R2_BENEFIT_FIX.json`, `R2_SHAP_FAIRNESS_CALIBRATION.json`, `R2_TABPFN_BAGGING_DEMO.json`
  - `R3_PAST_SELF.json`, `R3_STACKING_V2.json`, `R3_STACKING_V3_POINTLEVEL.json`, `R3_TIMESFM_QUANTILE.json`
  - `R4_DANGEROUS.json`, `R4_DANGEROUS_V2.json`, `R4_DANGEROUS_V2_ABLATION.json`, `R4_DANGEROUS_V2_HUMAN_BASELINE.json`, `R4_DANGEROUS_V2_LIVE.json`, `R4_DANGEROUS_V2_REPORT.md`
  - `R5_GRANITE.json`, `R5_GRANITE_HARD.json`, `R5_GRANITE_REPORT.md`, `R5_MTEB_SUBSET.json`
  - `R6_GETHSEMANE.json`, `R6_GETHSEMANE_ONNX_EXPORT.json`, `R6_GETHSEMANE_MASKING_ABLATION.json`
  - `R6_EUCLIDIAN.json`, `R6_PROVIDER.json`, `R6_PROVIDER_V2.json`, `R6_AQUA_REGIA.json`, `R6_AQUA_REGIA_V2.json`

- **ONNX policies**: `v3_arcadia/checkpoints/gethsemane/ppo_*.onnx` (3 × 0.97 MB)

- **Docs**:
  - `MODEL_CARD.md`, `PYTORCH_STORY.md`, `BENCHMARKS_VS_PUBLIC.md`
  - `FINAL_DEMO.md`, `AUDIT_PLAN.md`, `FAILURE_TABLE.md`
  - `demo/PITCH_DECK.md`, `demo/SupplyMind_pitch.html`
  - `demo/DEMO_VIDEO_SCRIPT.md`, `demo/LANDING_PAGE.md`, `demo/social.md`
  - `challenges/R4_RUBRIC_CHALLENGE.md`

- **Colab**: `notebooks/04_v3_quickstart_colab.ipynb`

---

## 🧪 Verification commands a judge can run

```bash
# 1. Tests pass
pytest tests/ -q
# → 173 passed in 2m14s

# 2. OpenEnv compliance specifically
pytest tests/test_openenv_compliance.py -v
# → 19/19 pass

# 3. Server works
uvicorn server.app:app --port 8000 &
sleep 5
curl http://localhost:8000/health       # → {"ok": true, ...}
curl http://localhost:8000/tasks        # → 3 tasks listed
curl -X POST "http://localhost:8000/reset?task_id=easy_typhoon_response&seed=42"

# 4. Scripted baseline is deterministic
for i in {1..5}; do python scripted_agent.py --task easy_typhoon_response 2>&1 | grep "final_score"; done
# → all 5 runs should show identical scores

# 5. ONNX policy roundtrip
python -c "
import onnxruntime as ort
import numpy as np
s = ort.InferenceSession('v3_arcadia/checkpoints/gethsemane/ppo_easy_typhoon_response.onnx')
r = s.run(None, {'observation': np.random.randn(1, 408).astype(np.float32)})
print('ONNX inference:', r[0].shape)
"
```

---

## 📊 Headline numbers (verifiable from committed JSONs)

| Claim | Source | Verify with |
|---|---|---|
| 173 tests passing | `pytest tests/ -q` | `pytest tests/ -q` |
| 13 foundation models verified | `v3_arcadia/results/R1_VERIFIED.json` | `cat` it |
| 261,175 real data points | `DATA_SOURCES.md` + `MODEL_CARD.md §4` | follow citations |
| mxbai P@1 = 0.962 | `v3_arcadia/results/R5_GRANITE.json` | `jq '.pipelines.P2_mxbai_bi.p1'` |
| reranker +5pp on hard queries | `v3_arcadia/results/R5_GRANITE_HARD.json` | `jq '.reranker_lift_deltas'` |
| 2-judge α = 0.750 | `v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json` | `jq '.agreement_primary_panel'` |
| Cohen κ(Qwen, Mistral) = 0.747 | same | same |
| PPO vs baselines CI95 non-overlapping | `v3_arcadia/results/R6_EUCLIDIAN.json` | `jq '.tasks'` |
| GNN +48-64% vs MLP on arrival-time | `v3_arcadia/results/R6_PROVIDER_V2.json` | `jq '.graphs'` |
| Per-horizon conformal dev < 0.024 on oil@95% | `v3_arcadia/results/R6_AQUA_REGIA_V2.json` | `jq '.results.DCOILWTICO'` |

---

## ⚠️ Honest status of judge-facing actions (no spin)

Required manual actions (your token / your camera):

- [ ] **Record demo video** via `demo/DEMO_VIDEO_SCRIPT.md` (3 hours)
- [ ] **Push to HF Space** — `git push hf main --force-with-lease` or trigger `.github/workflows/deploy-hf-space.yml` (requires HF_TOKEN secret)
- [ ] **Deploy Streamlit dashboard** via Streamlit Community Cloud (link GitHub)
- [ ] **Deploy Damocles FastAPI** via Render or Fly.io free tier
- [ ] **Render pitch PDF** — open `demo/SupplyMind_pitch.html` in browser, Ctrl+P → Save as PDF
- [ ] **Populate GitHub Release** — `bash scripts/release_assets.sh` (requires `gh auth login`)
- [ ] **Post social thread** — copy from `demo/social.md`

All code artifacts ready for each action. You drive, I've built the rails.
