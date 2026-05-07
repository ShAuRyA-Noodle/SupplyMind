# Secrets Rotation Plan (Phase G12)

## Current state (verified 2026-04-21)

✅ `.env` is in `.gitignore` (line 6) — never pushed to GitHub.
✅ `.env.example` exists at repo root with placeholder keys.
✅ All 5 API keys are **free tier** — low-value even if leaked.

## Keys held

| Key | Service | Free tier? | Rotation URL |
|---|---|---|---|
| `FRED_API_KEY` | Federal Reserve Economic Data | ✅ Free, unlimited | https://fred.stlouisfed.org/docs/api/api_key.html |
| `NEWS_API_KEY` | NewsAPI.org | ✅ Free 100 req/day | https://newsapi.org/account |
| `WANDB_API_KEY` | Weights & Biases | ✅ Free personal | https://wandb.ai/authorize |
| `HF_TOKEN` | Hugging Face | ✅ Free, rate-limited | https://huggingface.co/settings/tokens |
| `NOAA_TOKEN` | NOAA CDO Web | ✅ Free, rate-limited | https://www.ncdc.noaa.gov/cdo-web/token |

## Rotation schedule

1. **Pre-submission (required)**: verify `.env` never appears in any commit:
   ```bash
   git log --all --full-history -- .env  # must return empty
   git log --all -S "FRED_API_KEY=cdb005b8" --source  # must return empty
   ```
2. **Post-hackathon**: rotate all 5 keys via their respective URLs. Takes ~5 min.
3. **Production**: move to a secrets manager (AWS Secrets Manager, HCP Vault, or HF Space Secrets UI — the Space secrets UI is the simplest path for the demo).

## HF Space secrets

For the deployed HF Space, keys are set via the **Space Settings → Variables and secrets** UI, NOT checked into the repo. `server/app.py` reads them via `os.environ.get()`.

## Accidental leak response

If a key leaks:
1. Immediately revoke at the service URL (links above).
2. Generate a new key.
3. Update local `.env`.
4. Update HF Space secret.
5. Force-push cleanup if commit history contaminated (requires `git filter-repo`).
