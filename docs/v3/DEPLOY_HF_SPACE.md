# Phoenix HF Space deploy — complete walkthrough

Push the v3.0-arcadia release to `huggingface.co/spaces/Shaurya-Noodle/Supplymind` in one sitting. The user said they restarted the Space; this doc is the complete rebuild-from-ashes playbook.

**Expected time**: 15 minutes. **Requires**: your HF token.

---

## Option A — one-time manual push (fastest)

### 1. Get your HF token
1. Open https://huggingface.co/settings/tokens
2. Click "New token" → Role: **Write** → name it `supplymind-deploy` → create
3. Copy the token (starts with `hf_...`)

### 2. Configure local git (one-time)
```bash
# Save credentials so git doesn't ask on every push
git config --global credential.helper store

# Or use the huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token hf_xxxxxxxxxxxxxxxx
```

### 3. Add the HF Space remote
```bash
cd /path/to/Sleep-Token
git remote add hf https://huggingface.co/spaces/Shaurya-Noodle/Supplymind

# Or if already added, update it:
# git remote set-url hf https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
```

### 4. Push (force, since HF Space was reset)
```bash
git push hf main --force-with-lease
```

### 5. Wait for HF to rebuild (Docker build, ~5-8 min)
Visit https://huggingface.co/spaces/Shaurya-Noodle/Supplymind — you'll see the build log.

### 6. Verify
```bash
# Once build is green
curl -fsS https://shaurya-noodle-supplymind.hf.space/health
# → {"status": "ok", ...}

curl -X POST "https://shaurya-noodle-supplymind.hf.space/reset?task_id=easy_typhoon_response&seed=42"
# → SupplyMindObservation JSON
```

---

## Option B — automated via GitHub Action (set it and forget it)

### 1. Add HF_TOKEN as a GitHub secret
1. GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `HF_TOKEN`
4. Value: `hf_xxxxxxxxxxxxxxxx` (from step 1 above)
5. Save

### 2. Trigger the workflow
The workflow at `.github/workflows/deploy-hf-space.yml` is already committed. Trigger options:

- **Auto**: any push to `main` that touches server/, models.py, openenv.yaml, versions/v3_arcadia/, or top-level MD files will trigger deploy.
- **Manual**: GitHub repo → **Actions** tab → "Deploy to HuggingFace Space" → **Run workflow** → `main` branch → Run.

### 3. Watch it run
Takes ~3 min for git push + ~8 min for HF Docker rebuild.

---

## What gets deployed

The full repo minus large blobs. The `.gitignore` already excludes:
- `models/` (159 GB of GGUF/safetensors — HF Space would refuse anyway)
- `versions/v3_arcadia/checkpoints/granite/corpus_emb_*.npy` (regeneratable)

The Space runs the `Dockerfile` (multi-stage build for `server.app:app` on port 8000). Judges visiting the Space get:
- `/health` — smoke check
- `/tasks` — lists 3 tasks with descriptions
- `/reset?task_id=easy_typhoon_response&seed=42` — initial observation
- `/step` — POST action, get observation + reward + done
- `/grader` — final episode score
- `/docs` — Swagger UI (FastAPI auto-generated)
- `/redoc` — ReDoc rendering

---

## Known pitfalls

1. **HF Space rebuild fails on first push**: HF Spaces have a ~10 GB total repo-size limit. The `.gitignore` handles this; if your local checkout has stray large files (e.g. historical `models/` copies), run `git status --short` and make sure no untracked 1 GB+ files are in the push.

2. **Docker build timeout**: HF free-tier containers have a 1 CPU / 16 GB RAM limit during build. The `Dockerfile` is already slim-based; if build fails, check the log — usually it's a transient timeout, retry by pushing a no-op commit.

3. **Health check returns 500**: First request after deploy is a cold start; wait 30s and retry.

4. **Wrong repo type**: If the HF repo was accidentally created as a Model or Dataset instead of a Space, delete and recreate as Space with Docker SDK.

---

## After deploy, update the landing page

In the `demo/LANDING_PAGE.md` / `README.md` / `demo/PITCH_DECK.md`, replace any "HF deployment pending" notes with the verified live URL.

GitHub Actions can also auto-update these files if you want — see the `deploy-hf-space.yml` workflow's final step.

---

## After deploy, populate the GitHub Release

```bash
# Requires `gh` CLI authenticated (gh auth login)
bash scripts/release_assets.sh
```

This uploads all plots, JSONs, ONNX artifacts, and MD docs as Release assets at
https://github.com/ShAuRyA-Noodle/Sleep-Token/releases/tag/v3.0-arcadia

---

## Verification checklist (what judges should see)

- [ ] https://huggingface.co/spaces/Shaurya-Noodle/Supplymind loads without 404
- [ ] HF Space shows the v3 README-header (not the v2 content)
- [ ] `/health` returns 200
- [ ] `/tasks` lists 3 tasks with `easy_typhoon_response`, `medium_multi_front`, `hard_cascading_crisis`
- [ ] `/reset` with `seed=42` returns a full Pydantic observation with `situation_summary` and `compact_summary`
- [ ] `/docs` renders the Swagger UI
- [ ] GitHub Release page shows plots + MODEL_CARD + PITCH_DECK + ONNX + JSONs

Once all 7 are ✅, the Space is truly deployed and top-3 is in striking distance.
