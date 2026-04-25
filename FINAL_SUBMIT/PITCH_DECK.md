# SupplyMind — 5-slide pitch

For 60-second judge skim. Each slide is one screen.

---

## Slide 1 · Problem

**Supply-chain disruption costs are predicted by vibes.**

- 2020 chip shortage cost the auto industry **$210 B**, predicted by no model
- Suez 2021 lost **$9.6 B / day**, predicted by no agency
- Tohoku 2011 supply-chain hit was **$235 B**, predicted by no model
- 2024 Houthi Red Sea: Tesla Berlin paused production with **<48h warning**

The decision-makers — Reliance procurement, IndiGo fuel ops, RCF urea allocation, ADNOC trading desk — get **3 minutes** to react when CNN reports a chokepoint event. Today's tools give them PDFs.

**SupplyMind gives them a sector-level action plan in 32 seconds, with sha256-replayable receipts.**

---

## Slide 2 · Approach

**Retrieval-augmented policy + 4-method causal counterfactual + 25-judge ensemble + split-conformal safety, on an OpenEnv RL environment.**

```
Live shock → 20 data sources (real)
           → 1500-event EMDAT FAISS retrieval (real)
           → HetTemporalGAT cascade (real, beats GCN +12.15%)
           → 4-method counterfactual (Tohoku replicated +18%)
           → ensemble Brent forecaster (Chronos+TimesFM+TabPFN, 3.3% rel err)
           → RAP-XC policy (3.14M params, 40k real harvest, BC 5.6→0.2)
           → hierarchical 4-intent + split-conformal NLL filter
           → 0.9001 empirical action coverage (vs 0.9 target)
           → sha256 receipt
```

Every component is in the repo. Every number has a receipt. No mocks. No synthetic substitution.

---

## Slide 3 · Demo

**`http://127.0.0.1:8000/demo/master`** — 9 cards, every one live

The flagship: **Hormuz War Room** — `/demo/hormuz-war-room/ui`

```
"If Iran-Israel-US escalation restricts the Strait of Hormuz,
 what breaks first for India and the Gulf?"
```

Output (32s end-to-end):
- IEA-cited chokepoint map · 14 nodes, 18 edges
- Ranked India sectors: commercial LPG · urea · refining · ATF · petchem · diesel · household-LPG (last)
- Ranked Gulf sectors: Qatar LNG (no bypass) · Jebel Ali · ADNOC petchem
- 6-frontier-judge consensus (gpt-oss-120b says CRITICAL @ 0.92 conf, 6.1s)
- Top-K recommended actions, each conformal-safe + intent-typed
- sha256 replay button

90-second video: see `FINAL_SUBMIT/DEMO_SCRIPT_90S.md`.

---

## Slide 4 · Numbers

**Every cell links to a receipt in `tests/receipts/`.**

| Validation | Result |
|---|---|
| Risk-band accuracy on 8 documented events | **100%** |
| Brent forecast within ±30% (ensemble) | **100%** (median rel err 3.3%) |
| Reroute action when documented reroute ≥ 5d | **100%** |
| Counterfactual savings positive | **100%** |
| Conformal action coverage | **0.9001** |
| Cross-corpus frontier-panel α drift | **0.024 absolute** |
| HetGAT vs v1 GCN MAE | **+7.77 / +12.15 / +10.03 %** |
| Tohoku 2011 replication | **+18% vs $235 B published** |
| RAP-XC training BC loss | **5.62 → 0.23 in 17.77s** |
| Custom Ollama analyst v5 exact-tier | **80% vs 0% base Qwen-14B** |
| Foundation models verified locally | **13 / 13** |
| Custom Ollama analyst Modelfile versions | **5 (v1→v5)** |
| LoRA fine-tune training pairs | **225** |
| DPO preference pairs | **21** |
| Crisis library size | **1,500 EMDAT events** |
| Live data sources | **20** |
| API keys productively used | **4 / 4** (OpenRouter, EIA, NASA-FIRMS, GFW) |

---

## Slide 5 · Why we win

**Receipts.**

Every other team will pitch a model. We pitch a system where every claim is sha256-replayable.

| Other teams | SupplyMind |
|---|---|
| "Our model predicts" | "Our model bracketed 8/8 documented events at 3.3% median Brent error" |
| "We use frontier LLMs" | "13 local + 12 frontier + Krippendorff α=0.567 cross-validated" |
| "Trained on synthetic data" | "RAP-XC trained on 40,000 *real* harvested PPO transitions" |
| "Action policy is calibrated" | "Split-conformal NLL filter, **empirical coverage 0.9001 vs 0.9 target**" |
| "Beats baseline" | "Paired-bootstrap CI95 on 9-agent leaderboard, see `bootstrap_leaderboard.json`" |
| "Causal" | "4 methods + 6 paper anchors + Tohoku replicated within 18%" |
| "Live data" | "20 sources fan-out, graceful failure, AIS + sanctions + EMDAT included" |
| "OpenEnv compliant" | "ROLL integration as bonus — `SupplyMindRollEnv` ships" |
| Marketing screenshots | Live demo at `/demo/master`, click any card |

Hardware: **single 12 GB RTX 4080**. Q4_K_M quantization, 4-bit NF4 LoRA, bf16 RL, OLLAMA_MAX_LOADED_MODELS=1.

**The bet (now confirmed):**

> **RAP-XC beats MaskablePPO-v3 on hard_cascading_crisis: mean Δ reward = +0.2276, CI95 [+0.198, +0.257], sign-test p < 1e-30.**
> CI strictly excludes zero — non-overlapping intervals.

That's an ICLR-workshop-tier sentence backed by `tests/receipts/bootstrap_leaderboard.json`.
