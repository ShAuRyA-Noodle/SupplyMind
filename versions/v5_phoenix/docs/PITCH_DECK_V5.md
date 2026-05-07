# SupplyMind Phoenix v5 — Pitch Deck (8 slides)

For Meta PyTorch OpenEnv Hackathon Finals. Render via `pandoc ... -t beamer`
or paste into Keynote. Speaker notes in blockquotes.

---

## Slide 1 — Title

# SupplyMind
## OpenEnv-native supply-chain risk environment
### v5.0-phoenix-ascensionism · solo submission · 2026-04-25

> "We built an OpenEnv environment with a live geopolitical pipeline, a DPO-
> fine-tuned LLM judge, an arena where you can drop in your own PyTorch
> policy, and two upstream PRs to Meta and Alibaba. Let me show you."

---

## Slide 2 — The problem + the hook

**$184 B / year** in supply-chain disruptions (BCI 2023).
**Zero** public benchmarks for supply-chain RL.

SupplyMind fills the gap:
- 3 calibrated tasks on an OpenEnv-spec environment
- Real data everywhere (DataCo, NOAA, FRED, World Bank, SEC, Wikipedia — 261K points total)
- Trained SOTA agents + LLM judges + live geopolitical pipeline

> "When a real crisis happens — Hormuz, Suez, Red Sea — right now, supply-
> chain teams look at spreadsheets. We built a benchmarkable environment
> where agents make the decisions and we measure $ saved."

---

## Slide 3 — Headline numbers (live, one-bash-command each)

| Claim | Value |
|---|---|
| mxbai RAG P@1 | **0.9622** |
| Snowflake BEIR nDCG@10 | **0.971** |
| 2-judge Krippendorff α | **0.7499** |
| MaskablePPO masking lift | **+26.77 %** |
| GCN MAE reduction vs MLP | **−48 %** |
| Per-horizon conformal dev @ 95 % | **0.024** |
| v3+v4 tests passing | **249 / 249** |
| Autoresearch best experiment lift | **+0.051 CI95** |

20 receipts in `versions/v5_phoenix/receipts_v2/` — pick any 3, we run them live.

> "Every number on this slide has a 30-second receipt you can paste into
> your terminal. If it doesn't match, I fail."

---

## Slide 4 — The OpenEnv Arena

**Drop in your PyTorch policy, we benchmark it.**

```bash
curl -X POST http://localhost:8000/arena/run \
  -F "policy=@my_policy.pt" -F "episodes=50"
```

Returns bootstrap-CI95 reward on 3 tasks + ranking vs our 6 baselines
(MaskablePPO, RecurrentPPO, PPO, A2C, Random, Greedy from R6 Euclidian).

> "Judges spend their careers training agents. We wanted to let you try
> yours against ours. It's 90 seconds end-to-end."

---

## Slide 5 — Live Hormuz demo

```bash
curl -X POST /live/hormuz-closure -d '{"scenario_text": "Iran threatens
Hormuz closure; Brent $123/bbl..."}' | jq
```

Returns:
- Analog match: `hormuz_trump_cargo_ship_2026_04` @ 0.99 similarity
- risk = CRITICAL, 5 actions
- **Counterfactual: $X M no-action loss → $Y M with plan → $Z M saved (live)**
- 3-judge LLM panel output

> "This isn't a scripted demo — it's hitting real 2026 NewsAPI + FRED
> Brent. Let me run it live."

---

## Slide 6 — Karpathy autoresearch + DPO-fine-tuned judge

**Autoresearch** (v5 fixed): agent mutates `candidate_train.py`, runs 50 K
steps, evaluator decides via bootstrap CI95 lower.
- s1 bigger-net accepted (seed baseline)
- **s2 higher-entropy accepted (+0.051 CI95 lower over baseline)**
- s3/s4/s5 pending rerun (v4 crash bugs fixed here)

**ROLL-DPO-judge-v1**: Qwen-2.5-3B + LoRA r=8, DPO on 26 preference pairs.
Ships either via ROLL pipeline or `trl.DPOTrainer` fallback. Adapter
~20 MB, HF Hub shareable.

> "Real LLM post-training, not prompt engineering. The adapter is 20 MB
> and I'll upload it to HF Hub for you to download and test."

---

## Slide 7 — Open-source contributions

Three upstream ships:

1. **`meta-pytorch/openenv`** — SupplyMind as a reference env
2. **`alibaba/ROLL`** — SupplyMind as an agentic-RL training target
3. **`obra/superpowers-marketplace`** — `supplymind-skills` methodology pack

> "The hackathon page says 'code ships to Meta-backed projects.' We go one
> better — code ships to three different open-source ecosystems."

---

## Slide 8 — Ask + contact

- **This is top-3 material**: 60–75 % P(top-3) honest estimate.
- **Interview-ready**: the repo is the portfolio.
- **Ask me anything**:
    1. Upload your policy to the Arena
    2. Run any 3 receipts
    3. Watch the live Hormuz demo
    4. Point at any claim; I'll show you the code and receipt

**Contact**: https://github.com/ShAuRyA-Noodle/Sleep-Token
**Email**: (from README)

> *"Built solo. Three months. No compromises. Real data everywhere."*

---

## Speaker notes / contingency

- If asked "what would you do with $1M compute?": see docs/v3/BENCHMARKS_VS_PUBLIC.md §
  ambition appendix.
- If live demo fails: "We have three paths — live, replay, video. Pivoting to
  replay now." Show `?replay=1` endpoint or `DEMO_BACKUP_2026_04_24.mp4`.
- If asked "how is this different from Coding-Agent Bench / MiniGrid /
  MuJoCo": see `docs/v3/comparison.md`.
- If asked "what's your win probability": 60–75 % top 3 with plan executed,
  85-92 % top 10 locked, interview opportunity > 90 %.
