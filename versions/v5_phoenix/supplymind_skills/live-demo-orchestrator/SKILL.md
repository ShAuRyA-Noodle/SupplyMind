---
name: live-demo-orchestrator
description: Use when preparing, running, or recovering from a live demo that depends on external APIs, live data, or unpredictable hardware. Enforces offline replay fallbacks, pre-demo health checks, golden response captures, and a post-demo receipt.
---

# Live Demo Orchestrator

## The iron law

EVERY LIVE FEATURE HAS AN OFFLINE REPLAY. IF IT DOESN'T, IT'S NOT DEMO-READY.

## When to invoke

- Pre-demo checklist (night before a live presentation)
- During a demo (recovery if something breaks)
- Post-demo receipt generation

## When NOT to invoke

- Internal testing (tests are their own artifact)
- Offline-only demos (nothing to fall back to)

## Three phases

### Phase 1 — Pre-demo (night before)

```markdown
## Pre-demo checklist

- [ ] **Health check**: `curl -s localhost:8000/health | jq .status` returns "ok"
- [ ] **Golden response capture**: run the exact demo curl/CLI command with
      live APIs on, save output to `docs/HORMUZ_DEMO_GOLDEN.json`. This is
      your "what should happen" reference.
- [ ] **Replay cache freeze**: run any ingestor with `--once` flag, save all
      fetched records to `replay_cache_<YYYYMMDD>.json`. Add `--replay` flag
      to the live endpoint to serve from cache when network is down.
- [ ] **Ollama warm**: `ollama list | grep -E "qwen2.5:14b|mistral-nemo|deepseek-r1"`
      shows all three models loaded.
- [ ] **Mobile hotspot**: one spare hotspot, tested from the demo venue
      network, IP rotation confirmed.
- [ ] **API key rotation**: keys in `.env` rotated within last 24h, old keys
      revoked, `.env` not committed.
- [ ] **Backup video**: 90-sec recording of the exact demo running on known-good
      state. Named `DEMO_BACKUP_<date>.mp4`. If live fails, you pivot to video.
- [ ] **Environment lock**: pip-freeze current venv to `requirements.locked.txt`.
      No installs the morning of the demo.
- [ ] **Test regression**: `pytest -q` passes with the current commit hash.
- [ ] **Git state**: clean working tree, tag on the demo-day commit.
```

### Phase 2 — During demo

```markdown
## The 4-step recovery protocol

If something misbehaves mid-demo:

1. Acknowledge out loud: "the live signal looks off, let me switch to the
    captured replay." NEVER pretend nothing happened.
2. Pivot to `--replay` path: same curl, different backend, identical
    golden-response shape. The audience shouldn't notice the switch.
3. If replay also fails: pivot to the backup video. "Here's the demo
    we captured yesterday — same result, just not live."
4. After the demo: never edit the failing path in the demo machine while
    the audience is still around. Note the failure, fix it later.

## The one-line mental model for the judge

> "Three paths: live → replay → video. One of them always works."
```

### Phase 3 — Post-demo receipt

```markdown
## Receipt

Write `demo_receipt_<timestamp>.yaml`:

```yaml
demo_id: v5_hormuz_live_2026_04_25_10_30
scenario: "Iran threatens Hormuz closure; Brent $123/bbl"
path_used: live       # live | replay | video
live_path_ok: true
replay_path_ok: true
video_backup_ready: true
response_time_s: 2.8
risk_level_returned: CRITICAL
top_analog_id: hormuz_trump_cargo_ship_2026_04
top_analog_similarity: 0.99
counterfactual_loss_usd_no_action: 324000000
counterfactual_loss_usd_with_plan: 65000000
counterfactual_savings_pct: 80
judge_panel_size: 3
judge_consensus_level: 2  # 2 of 3 agree on CRITICAL
issues_encountered: []
audience_questions_flagged: []
```
```

## Anti-patterns

- Demoing a path that wasn't dry-run <24h ago
- Relying on venue Wi-Fi without a hotspot backup
- No backup video
- Ollama not warm ("first call takes 30s while the model loads" = dead on stage)
- Editing code the morning of the demo
- Unrotated API keys (rate limit == silent death mid-demo)
- Treating replay as a testing-only artifact instead of a demo-path first-class citizen

## The Phoenix v5 demo paths

Our specific commands, documented:

```bash
# Live
curl -X POST http://localhost:8000/live/hormuz-closure -d @scenario.json

# Replay
curl -X POST http://localhost:8000/live/hormuz-closure?replay=1 -d @scenario.json

# Video
open demo/DEMO_BACKUP_2026_04_24.mp4
```

All three should return the same JSON shape. If the replay path returns a
different shape, it's a bug — fix before demo day.
