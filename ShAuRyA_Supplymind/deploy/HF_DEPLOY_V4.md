# HF Space v4 Deploy Guide

> G2 + L4.2 — one-command deploy of v4.0-arcadia-live to Hugging Face Spaces.

## One-time setup

```bash
# Add HF remote (SSH or HTTPS)
git remote add hf https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
```

The HF Space secrets UI needs these **same** env var names as `.env`:
`FRED_API_KEY`, `NEWS_API_KEY`, `HF_TOKEN`, `NOAA_TOKEN` (WANDB optional).

## Deploy command (one-shot)

```bash
# Dry check first
pytest tests/ ShAuRyA_Supplymind/tests/ -q --tb=line

# If green, push to HF
git push hf main --force-with-lease

# Wait 5-8 min, then smoke test
curl https://shaurya-noodle-supplymind.hf.space/health
curl https://shaurya-noodle-supplymind.hf.space/live/health
curl -X POST https://shaurya-noodle-supplymind.hf.space/reset?task_id=easy_typhoon_response
```

## 7-item smoke checklist (copy-paste from FINAL_DEMO.md §7)

- [ ] `/health` returns 200
- [ ] `/tasks` lists 3 tasks
- [ ] `/reset?task_id=easy_typhoon_response` returns full Pydantic observation
- [ ] `/live/health` reports event store + ollama availability
- [ ] `/live/hormuz-closure` POST returns a structured risk assessment
- [ ] `/docs` renders Swagger UI
- [ ] GitHub Release tag `v4.0-arcadia-live` populated with plots + MODEL_CARD PDF

## HF Space constraints we respect

- `.gitignore` excludes **159 GB of models/** (only referenced in local inference).
- `.gitignore` excludes `.venv/`, `catboost_info/`, large `rl/checkpoints/*.pt`, embedding caches.
- `.gitignore` excludes v4-generated state (`events.db`, `library_embeddings.pkl`,
  `autoresearch/experiments/`, `autoresearch/state.json`).
- The `Dockerfile` at repo root is the one HF uses; `Dockerfile.damocles` + `Dockerfile.dashboard`
  are for local multi-service runs only.

## Expected deploy metrics

| Metric | Expected |
|--------|----------|
| Build time | 6-8 min |
| Container size | <2 GB (slim base, no models) |
| Cold start | 15-25s (pre-warm graphs on startup) |
| Memory | 2-3 GB steady |
| CPU-only inference | works for `/reset`, `/step`, `/grader`, `/live/*` |
| GPU inference | only needed if you want the Ollama LLM judges live |

## If HF deploy fails

1. **Size too big**: check `.gitignore` excludes working, `du -sh .` should be <500 MB.
2. **Build timeout**: simplify `Dockerfile` — pin fewer deps.
3. **Port mismatch**: ensure `app_port: 8000` in README frontmatter matches `Dockerfile` EXPOSE.
4. **Missing secrets**: HF Space Settings → Variables — add the 4 env vars.
5. **Repo type wrong**: must be `sdk: docker` (not gradio/streamlit).

See `DEPLOY_HF_SPACE.md` (root) for the full v3 deploy doc we inherit.
