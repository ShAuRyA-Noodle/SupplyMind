# PRODUCT THESIS — SupplyMind

> Status: DRAFT by Fable orchestrator 2026-07-03. Owner: approve/adjust the persona + one-liner —
> that is the single decision only you should make. Everything else follows from it.
> Rebuild waves reference this file for priority (CLAUDE.md §12b).

## One-sentence product
**SupplyMind is a live supply-chain-risk decision copilot: it watches real-world signals
(shipping, commodities, disasters, news), tells a risk officer what is about to break, what to do
about it, and what it costs if they don't — every claim traceable to real data and a reproducible
simulation.**

## Persona (anchor for every feature decision)
**Priya — supply-chain risk officer at a large India-based manufacturer/importer** (Reliance-scale;
the Hormuz/Reliance material already in the repo fits her exactly). She owns a war room. At 8am she
needs: what changed overnight, how bad, what's my move, what's the downside if I do nothing. She
distrusts black boxes and will ask "how do you know that number?" — so every output must cite.

If a feature does not help Priya answer one of those four questions, it is appendix or cut.

## The golden path (the 4-minute judge demo — the ONE thing we execute flawlessly)
1. **Signal**: a real live event surfaces — GFW ship-density anomaly near Hormuz / a NewsAPI
   headline / a FRED Brent spike (all real APIs, keys in .env).
2. **Assessment**: the SupplyMind analyst (OpenRouter-hosted, v5 calibration prompt) returns a
   strict-JSON verdict — risk level + calibrated confidence + cited evidence. Calibration is
   Brier-scored on a holdout, not asserted.
3. **Decision**: the engine + agent recommend a concrete action (reroute / hedge / pre-buy /
   activate backup) with the action grounded in the real 408-dim observation.
4. **Counterfactual**: real methods (R6) estimate the $ impact of acting vs not — with a
   confidence interval, on real Tōhoku/Suez/Hormuz analogs.
5. **War room** renders it live, with a receipt link behind every number.

That is the story. Everything in the repo either serves this path, supports it, or steps aside.

## REBUILD_BACKLOG → spine mapping
| Tier | Items | Rationale |
|---|---|---|
| **CORE SPINE** (build first, must be flawless) | R6 real causal counterfactual · R10 real Brent backtest · R14 real judge panel *(key-blocked)* · analyst-v5 port + Brier A/B (P1.2) · live-signal layer (P1.7) | These ARE the golden path. |
| **SUPPORTING** (strengthen the spine's credibility) | R1/R2 real benchmark+leaderboard · R7 300+ gauntlet (safety story) · R11 real conformal (calibration story) · R17 RAG re-cook (evidence retrieval) · R5 real GAT (cascade viz) | Make the spine's claims falsifiable + defensible. |
| **APPENDIX** (real, but a side room — build if cheap) | R3 ablation · R4 Pareto · R8 DPO judge · R12 FedAvg · R13 cross-env transfer · R16 analytics dashboard | Impressive depth for judges who dig; not the headline. |
| **RETIRE?** (owner call) | R18 SAC-Discrete (QR-DQN already real) · R19 Damocles-as-separate-service (fold into main) | Low marginal value vs cost. |

## Open decisions for owner
1. **Persona**: confirm Priya/India-risk-officer, or name a different one (US importer? multinational
   CPG? logistics 3PL?). Changes which live feeds headline.
2. **RL positioning**: DECIDED BY EVIDENCE — the P1.3 benchmark decides whether RL leads or is
   framed as research frontier. Do not pre-commit.
3. **Retire list**: OK to retire R18 + fold R19? (my recommendation: yes.)

## Naming
Product = **SupplyMind** everywhere judge-facing. Arcadia/Phoenix/Damocles/Genesis = internal
version history only, live in versions/ + git tags. No judge-facing artifact uses a codename.
