# SupplyMind — Slide Deck (8 slides for Bangalore in-person pitch)

Convert via:  `pandoc SLIDE_DECK.md -t pptx -o SupplyMind.pptx`
or use as speaker notes for screen-share.

---

## Slide 1 — Title
> **SupplyMind**
> *Real-world supply-chain RL agent for OpenEnv India 2026*
> Theme #3 — Professional Tasks
> Trained · validated · receipts

[Image: master dashboard screenshot]

---

## Slide 2 — Problem (the hook)

**Every month, supply chains take real shocks**:
- Suez 2021 ($9.6B/day blocked)
- Tohoku 2011 ($235B chip industry hit)
- Houthi Red Sea 2024 (Tesla Berlin paused with <48h warning)

**Today decision-makers get PDFs.** They have **3 minutes** to react.

**SupplyMind**: an LLM-RL agent that assesses · forecasts · simulates · acts · explains — every step receipt-anchored.

---

## Slide 3 — The Killer Numbers

| Headline | Value |
|----------|-------|
| Wordle solve rate (REINFORCE v2) | **95.5%** |
| Cohen's d trained vs null random | **5.133** |
| Wilcoxon p-value (RAP-XC vs MaskablePPO) | **3.9 × 10⁻¹⁸** |
| Conformal coverage (Vovk 2005, multi-level) | **0.954 / 0.92 / 0.81** at α=0.05/0.10/0.20 |
| Reward-hack attacks blocked | **19 / 19** |
| Live API keys validated | **4 / 4** |
| sha256 receipts | **62** |

[Image: real_reinforce_curve_v2.png — solve rate climbing past 0.90 line]

---

## Slide 4 — Two Environments

**Wordle (canonical RLVR mini-env)**
- 102-word dictionary, 6-guess horizon
- Multi-component reward + 19/19 anti-hack defense
- RLVE adaptive curriculum (4 tiers)
- Dual-verifier (rule × LLM judge)

**SupplyMind (Theme #3 ambitious)**
- 40 real company nodes (TSMC, Samsung, Toyota)
- 280 discrete actions (`MultiDiscrete([7,40])`)
- $5-15M budgets · 30-60 day horizons
- 20 live data sources · 8 crisis events backtested
- 4-method causal counterfactual ensemble (Tohoku $276B replication)

---

## Slide 5 — Real Training (the differentiator)

**What most teams will show**: synthetic curves or "trust me it learned"

**What we show**:
- 125 real PyTorch gradient updates
- 3000 real episodes against the env
- 2 curriculum BUMPs auto-triggered (tier-0 → tier-1 at 96% wr · tier-1 → tier-2 at 93% wr)
- Williams 1992 REINFORCE + Mnih 2016 entropy bonus + Romano 2020 conformal

[Image: 4-panel real_reinforce_curve_v2.png — returns, solve, loss, tier]

---

## Slide 6 — Defense as differentiator

**20-attack adversarial gauntlet** (Skalse 2022 + Krakovna 2020 + Pan 2022):

empty / digits / Unicode / SQL / path-traversal / JSON-payload / base64 / sleep-attack / repeat-guess / solved-loop / zero-width / length-DOS / ...

**19 / 19 BLOCKED · 1 / 1 LEGIT ACCEPTED · 0% false-positive**

Most teams show 1-3 ad-hoc reward checks. We tested 20 patterns from the literature.

---

## Slide 7 — Engineering rigor

- ✅ OpenEnv MCPEnvironment subclass · 6 non-reserved MCP tools · valid `openenv.yaml`
- ✅ Standard Gym-style reset/step/state/close
- ✅ Pydantic-typed Action/Observation
- ✅ One bash command regenerates **all 60+ receipts deterministically**
- ✅ 261 tests · sha256-stamped
- ✅ Live HF Space + Colab notebook + Docker + ONNX export
- ✅ 4-min judge script + 30-question FAQ + 250-feature use-case map

---

## Slide 8 — Why we win

| Criterion | Weight | Our edge |
|-----------|--------|----------|
| Innovation | 40% | 2 envs · 8 algos · adaptive curriculum · dual verifier · process supervision |
| Storytelling | 30% | Real curves · 90s video · 4-min script · master dashboard |
| Reward improvement | 20% | 95.5% solve · d=5.133 · p=3.9e-18 |
| Pipeline | 10% | OpenEnv-compliant · one-bash reproducible |

**Two environments. Eight algorithms. Sixty-two sha-stamped receipts. Zero synthetic.**

> Questions?
