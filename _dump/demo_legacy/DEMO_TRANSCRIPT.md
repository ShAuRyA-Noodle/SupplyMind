# SupplyMind v3.0-arcadia — Demo Transcript

This is the **read-only substitute** for the demo video. If a judge can't watch (no sound, time pressure, bandwidth), every beat of the planned 3-minute screencast is transcribed below with exact commands, expected outputs, and on-screen captions. Follow top-to-bottom — each section is a scene.

Recorded walkthrough script: [`demo/DEMO_VIDEO_SCRIPT.md`](DEMO_VIDEO_SCRIPT.md).

---

## Scene 1 — The hook (0:00 – 0:15)

**On screen**: landing page [`demo/LANDING_PAGE.md`](LANDING_PAGE.md), scroll to headline.

**Narration**: "$184 billion lost to supply chain disruptions in 2023 alone — BCI Report. Every model you'll see in the next 3 minutes is trained on **real** data, verified on **public benchmarks**, and scored with **independent LLM judges**. Nothing synthetic."

**On-screen caption**: *"SupplyMind v3.0-arcadia · 261,175 real data points · 173 tests passing"*

---

## Scene 2 — Architecture (0:15 – 0:45)

**On screen**: [`MODEL_CARD.md`](../MODEL_CARD.md) §3 architecture diagram.

**Narration**: "Seven layers: Foundation model roster (R1) → tabular learners (R2 Caramel) → time-series ensemble (R3 Past Self) → dangerous-scenario judge panel (R4) → RAG stack (R5 Granite) → RL policies + GNNs + conformal bands (R6) → Arcadia deployment (R7). Every sub-block ships with a committed JSON."

**Shown files** (each flashed for ~2s):
- `v3_arcadia/results/R1_VERIFIED.json` — 13 models
- `v3_arcadia/results/R2_CARAMEL.json` — tabular AUC
- `v3_arcadia/results/R3_PAST_SELF.json` — 8 FRED targets × 3 horizons
- `v3_arcadia/results/R4_DANGEROUS_V2.json` — 26 scenarios × 4 LLMs
- `v3_arcadia/results/R5_GRANITE.json` — 8 RAG pipelines, P@1=0.962
- `v3_arcadia/results/R6_GETHSEMANE.json` — PPO with action masking

---

## Scene 3 — Live API demo (0:45 – 1:30)

**On screen**: terminal, split pane with browser at `http://localhost:8000/docs`.

```bash
$ uvicorn server.app:app --port 8000
INFO:     Uvicorn running on http://0.0.0.0:8000

$ curl -s http://localhost:8000/health
{"status":"ok","version":"v3.0-arcadia","timestamp":"2026-04-17T..."}

$ curl -s http://localhost:8000/tasks | jq '.[].task_id'
"easy_typhoon_response"
"medium_multi_front"
"hard_cascading_crisis"

$ curl -s -X POST "http://localhost:8000/reset?task_id=easy_typhoon_response&seed=42" \
  | jq '.situation_summary'
"Typhoon Koinu bearing NNW at 15 kt, projected landfall Kaohsiung in 36h.
Fab utilization 94%. Buffer inventory 4.2 weeks. Recommend pre-staging
shipments to Osaka backup route (+$2.1M, -28% delay risk)."
```

**Narration**: "One curl brings up a full Pydantic-typed environment observation, not JSON soup. Every field is schema-validated at the boundary."

---

## Scene 4 — Headline benchmarks (1:30 – 2:15)

**On screen**: [`BENCHMARKS_VS_PUBLIC.md`](../BENCHMARKS_VS_PUBLIC.md), scroll through 4 tables.

| Benchmark | SupplyMind | Public SOTA | Gap |
|---|---|---|---|
| mxbai P@1 (our retrieval corpus) | 0.962 | ~0.39 (NFCorpus) | +0.57 on in-domain |
| Reranker lift on hard queries | +5pp | literature: +3–8pp | in range |
| 2-judge Krippendorff α | 0.750 | ~0.80 (MT-Bench) | within 0.05 |
| PPO vs baselines CI95 | non-overlapping | — | sig. at p<0.05 |
| GNN arrival-time MAE vs MLP | −48% to −64% | literature: −30–50% | above range |
| Per-horizon conformal dev (oil 95%) | 0.024 | pooled: 0.112 | 4.7× tighter |

**Narration**: "Every number above is reproducible by re-running the committed script. Every script reads real data — no synthetic fallback paths."

---

## Scene 5 — Action-masking ablation (2:15 – 2:35)

**On screen**: `v3_arcadia/plots/gethsemane/r6_masking_ablation.png`.

**Narration**: "Isolated contribution of invalid-action masking: **+26.8% reward**, and invalid picks drop from **13.6 per episode to structurally zero**. This is exactly the range Huang et al. 2020 reported on the original paper."

---

## Scene 6 — Per-horizon conformal (2:35 – 2:50)

**On screen**: `v3_arcadia/plots/aqua_regia/r6_aqua_regia_v2.png`.

**Narration**: "Pooled split-conformal over-covers on heavy-tailed oil; per-horizon adapts to each horizon's own residual distribution. Deviation from target 95% coverage drops from 0.112 to 0.024 — a 4.7× improvement, following exactly Foygel Barber et al. 2022."

---

## Scene 7 — Reproducibility challenge (2:50 – 3:00)

**On screen**: [`challenges/R4_RUBRIC_CHALLENGE.md`](../challenges/R4_RUBRIC_CHALLENGE.md).

**Narration**: "We published our full 26-scenario rubric, our judge prompts, and our gold-label set. The challenge: beat our 2-judge α=0.750 with any setup you like. Code's open. Do it."

**Final caption**: *"GitHub: ShAuRyA-Noodle/Sleep-Token · HF Space: Shaurya-Noodle/Supplymind · License: Apache-2.0"*

---

## What a judge should do after reading this

1. **Skim** the [BENCHMARKS_VS_PUBLIC.md](../BENCHMARKS_VS_PUBLIC.md) tables — 3 min.
2. **Run** the 5 commands in Scene 3 locally — 2 min.
3. **Open** a plot PNG from `v3_arcadia/plots/**` — 30 s.
4. **Pick** any JSON from `v3_arcadia/results/` and `jq` a claim — 1 min.

That's a full defense in under 7 minutes without watching any video.
