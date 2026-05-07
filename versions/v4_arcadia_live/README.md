# SupplyMind v4.0-arcadia-live — Staging Directory

> "Rain will come, but we're ready. The signal is live."

This directory contains the **v4 arcadia-live** layer being built on top of v3.0-arcadia. Nothing here is committed to main until it is green, tested, and reviewed. Once a feature is complete it migrates into the main repo (server/, versions/v3_arcadia/, tests/).

## Directory Layout

| Dir | Purpose |
|---|---|
| `autoresearch/` | **Karpathy-style autonomous research loop** — `program.md` + agent-driven code mutation + fixed-budget runner + single-metric accept/reject + auto lab notebook |
| `realtime/` | **Live geopolitical ingestion** — NewsAPI + GDELT + USGS + MarineTraffic + FRED Brent crude polling; Hormuz / Iran / Israel / Red Sea focus |
| `scenarios/` | Real 2024-2026 crisis library (Iran-Israel, Hormuz, Red Sea Houthi, Taiwan Strait) with full citations |
| `features/` | 10 new unique features F1-F10 (Qwen-VL port imagery, multi-agent, conformal RL, Pareto carbon, provenance graph, receipts, etc.) |
| `deploy/` | HF Space v4 Dockerfile + deploy scripts + GitHub Actions updates |
| `docs/` | v4 docs: program.md for autoresearch, docs/v4/JUDGES.md, arxiv-style preprint, external quotes |
| `tests/` | v4 integration + unit tests |
| `receipts/` | F10 reproducibility receipt system — every headline number gets `.receipt` + `.reproduce.sh` |

## Phase Map

| Phase | Scope | Status |
|---|---|---|
| **Phase 0** | Foundation (this dir, .env hygiene, v4 plan) | done |
| **Phase L1** | Karpathy autoresearch deep integration (L1.1 — L1.5) | active |
| **Phase L2** | Live Hormuz demo (L2.1 — L2.5) | pending |
| **Phase G-Fix** | Gaps G2-G15 (HF deploy, Qwen-VL, multi-agent, DT v3, LoRA, SPOF, analyst, arxiv, CUDA, ensemble) | pending |
| **Phase L3** | 10 unique features F1-F10 | pending |
| **Phase L4** | Deploy + pitch + Colab (video deferred per user — recorded on Mac at end) | pending |
| **Phase L5** | Polish + docs/v4/JUDGES.md + external quotes | pending |
| **Final** | v4.0-arcadia-live tag + GitHub release | pending |

## Commit naming (Sleep Token v4 tracks, unused so far)

- **Rain** — Phase L1 Karpathy autoresearch
- **The Summoning** — Phase L2 Hormuz live demo
- **Vore** — Gap fixes batch 1 (G2, G3, G4)
- **Chokehold** — Gap fixes batch 2 (G6, G7, G8, G9)
- **DYWTYLM** — Feature batch (F1-F10)
- **Granite** (already used R5) → use **Ascensionism** — Phase L3 unique features
- **Arcadia II** — final v4.0-arcadia-live tag

## Hackathon context

- Finals: **April 25–26, 2026** (48-hour on-campus, Bangalore)
- Today: **2026-04-21** (4-5 days runway)
- Prize: $10K 1st / $10K 3rd / $4.55K 2nd / $2K 4-8 / $650 9-15
- Judged by Meta's global team. "Programmatic checks + LLM scoring."

## v3 → v4 diff principle

v3.0-arcadia is **frozen**. v4 adds on top. If any v4 feature breaks v3 tests, we roll back and fix before integrating. This directory is the sandbox that keeps the SOTA submission safe.
