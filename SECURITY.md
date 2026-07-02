# SECURITY

> World-class security pass for SupplyMind (Sleep-Token). Scope: full-repo + full git-history
> secret scan, `.env` hygiene, deserialization (pickle/`torch.load`) supply-chain risk,
> notebook/CI token handling, and a rotation + remediation plan.
> Produced 2026-07-02 (Wave 2 security-hardening, branch `p0/foundation`).
>
> **This file reproduces NO secret values.** Where a secret shape was searched for, only the
> search verdict is recorded. No live key material was found anywhere, so there are no
> fingerprints to print.

---

## 0. Bottom line (read this first)

**No live secret exists in the git history or the working tree.** Every secret-shaped pattern was
scanned across all 291 commits on all refs, across every object in the object database (including
dangling/unreachable objects), and across every tracked working-tree file. All hits were either
(a) placeholders, (b) key *names* with no values, or (c) sha256 receipt digests.

- The `# SCRUBBED` marker in `scripts/push_to_hf_space.py` is a **false alarm for git**: the file
  entered version control **already scrubbed** (`HF_TOKEN = os.environ.get('HF_TOKEN') or ''`). The
  real token — if it ever existed — lived only in the working tree / a chat log, never in a
  committed blob. **Nothing to purge from history.** (Rotation is still required — see §5, because
  the token was pasted into a conversation log.)
- `.env` is gitignored and **was never committed** (`git log --all -- .env` is empty).
- `versions/v5_phoenix/docs/PHOENIX_PUSH_REPORT.md §3.6` lists key **names only** — no values — so
  **no doc redaction and no history rewrite are needed** for it.

The residual security debt is **deserialization risk** (unsafe `pickle`/`torch.load` on a machine
running a torch version with a known RCE CVE) and **key rotation hygiene**. Both are enumerated
below as precise action items for the owning agents.

---

## 1. Threat summary

| Threat | Status | Where |
|---|---|---|
| Secret committed to git history | **NONE FOUND** | §2 |
| Secret in working tree (tracked files) | **NONE FOUND** | §3 |
| `.env` leaked / not ignored | **SAFE** (ignored, never committed) | §3 |
| Key values in docs / receipts | **NONE** (names + sha256 hashes only) | §3, §4 |
| Compromised keys needing rotation | **YES — 5+ keys** (pasted to chat log, per PHOENIX §3.6) | §5 |
| Unsafe deserialization (`torch.load weights_only=False`, `pickle.load`) | **PRESENT — multiple sites** | §6 |
| torch RCE CVE-2025-32434 exposure | **EXPOSED** — torch **2.5.1** installed (fixed in 2.6.0) | §6 |
| Token in shell process args (nb13) | **ALREADY FIXED** (GIT_ASKPASS + env) | §7 |

---

## 2. Git-history secret scan — per-pattern verdicts

**Method.** (1) `git log --all -S/-G` pickaxe per pattern across all 291 commits / all refs;
(2) definitive blob scan of the entire object database with
`git cat-file --batch-all-objects --batch` piped to a regex for real key shapes (this covers
**dangling/unreachable objects too**, not just reachable history); (3) `git fsck` for dangling
commits/tags.

| Pattern | Regex used | Verdict |
|---|---|---|
| HuggingFace token | `hf_[A-Za-z0-9]{34,}` | **CLEAN** — 0 real tokens. `-S 'hf_'` / `-S 'HF_TOKEN'` hits are all the string `HF_TOKEN` (env-var name) and `huggingface_hub`, never a value. |
| OpenRouter key | `sk-or-v1-[A-Fa-f0-9]{48,}` | **CLEAN** — only placeholder `sk-or-v1-...` (commit `aebb5a1`, `FINAL_SUBMIT/REPRODUCE.md`) and `sk-or-your_openrouter_key_here` (`.env.example`). |
| AWS access key | `AKIA[0-9A-Z]{16}` | **CLEAN** — 0 hits. |
| GitHub PAT (classic) | `ghp_[A-Za-z0-9]{34,}` | **CLEAN** — 0 hits. |
| GitHub PAT (fine-grained) | `github_pat_[A-Za-z0-9_]{60,}` | **CLEAN** — 0 hits. |
| W&B key (40-hex) | `WANDB_API_KEY[=: ]+[0-9a-f]{40}` | **CLEAN** — 0 value assignments; only the env-var name. |
| FRED / NewsAPI / NOAA key values | `(FRED|NEWS_API|NOAA)_...=[0-9A-Za-z]{20,}` | **CLEAN** — 0 value assignments; only names + placeholders. |
| GFW / generic `*_API_KEY=<value>` | `_(API_)?KEY=[A-Za-z0-9]{24,}` | **CLEAN** — 0 hits. |
| JWT (GFW token is a long JWT) | `eyJ...\.eyJ...\.` | **CLEAN** — 0 JWTs in any object. GFW's 798-char JWT never entered git. |
| PEM / OpenSSH private key | `BEGIN .*PRIVATE KEY` | **CLEAN** — 0 hits; no tracked `*.pem`/`*.key`/`*.p12`/`*.pfx`. |

**Full-object-store scan result:** `git cat-file --batch-all-objects` piped through a combined
real-secret regex returned **zero matches**. `git fsck` shows dangling commits
(`6e040c98`, `a5636365`) and dangling tags — all of their blobs were included in the all-objects
scan and are also clean.

**`scripts/push_to_hf_space.py` history (the `# SCRUBBED` marker):**
Touched by exactly 2 commits — `a4772d1` (pass 29, file introduced) and `b0d5248` (phase-4 rename).
Both blobs contain `HF_TOKEN = os.environ.get('HF_TOKEN') or ''  # SCRUBBED`. **The token value is
NOT in any committed version of this file.** No BFG/filter-repo purge is required for it.

### Git-history evidence excerpts (SHAs)
- `git log --all -- .env` → **empty** (never committed).
- `git log --all -- scripts/push_to_hf_space.py` → `b0d5248`, `a4772d1` (both scrubbed).
- `sk-or-` placeholder introduced in `aebb5a1` ("pass 10", `FINAL_SUBMIT/REPRODUCE.md`, value `sk-or-v1-...`).
- `git rev-list --all --count` → **291** commits scanned.
- Dangling objects present but clean: commits `6e040c98`, `a5636365`.

---

## 3. Working-tree secret scan

**Method.** `git ls-files -z | xargs -0 grep` over every tracked file for real key shapes,
hex/alnum key assignments (32/40-hex after a key-like name), JWTs, and private-key blocks.

| Check | Result |
|---|---|
| Real HF / OpenRouter / GitHub / AWS token shapes | **NONE** |
| `*_API_KEY = <20+ char value>` assignments (placeholders filtered out) | **NONE** |
| Any `eyJ...` JWT header in any tracked file | **NONE** |
| PEM/OpenSSH private-key blocks | **NONE** |
| Tracked `.env` / `.env.*` with real values | **NONE** — only `.env.example` (placeholders) is tracked |
| `.env` gitignore status | **IGNORED** (`.gitignore:6`) and never committed |
| `.env.example` content | Placeholders only (`your_*_here`, `sk-or-your_openrouter_key_here`) — safe |

**Findings table:** *(empty — no working-tree secret leaks)*

| File:line | Pattern | Severity | Note |
|---|---|---|---|
| — | — | — | No live-looking secrets found in any tracked file. |

**Note (non-security, informational):** `.env.example` is missing `GFW_API_TOKEN`, `EIA_API_KEY`,
and `NASA_FIRMS_KEY` even though those keys are live in `.env` (confirmed by
`FINAL_SUBMIT/receipts/api_keys_live_proof.json`). Not a leak — a documentation-completeness gap for
the `.env.example` owner to close.

---

## 4. Receipts / FINAL_SUBMIT payload leak check

The GFW token is a long JWT; the concern was that a receipt might have captured request headers or a
raw response containing auth material.

- `tests/receipts/*` and `FINAL_SUBMIT/receipts/*` were scanned for `Authorization`, `Bearer <...>`,
  `x-api-key`, `access_token`, `Cookie`, and `eyJ...` JWTs.
- **Only descriptive text** was found — e.g. `FINAL_SUBMIT/DATASET_CARD.md` says "Bearer token" as an
  auth-*type* label, and `api_keys_live_proof.json` says "Key is valid (Bearer auth)". **No token
  values, no captured headers.**
- `FINAL_SUBMIT/receipts/api_keys_live_proof.json` stores only `response_hash_first_1k` (a sha256 of
  the first 1KB of each API response) and `status_code`/`ok`/`endpoint` — **no key material, no
  headers, no body**. This is the correct pattern. The long 64-hex strings across the receipt JSONs
  are all sha256 receipt digests, not secrets.

**Verdict: no receipt leaked a key or JWT.**

---

## 5. Key rotation checklist (OWNER ACTION — do before the Qualcomm event)

Rationale: `versions/v5_phoenix/docs/PHOENIX_PUSH_REPORT.md §3.6` states 5 keys were pasted into a
conversation log and must be **treated as compromised**. That exposure is outside git (git is
clean), so it cannot be scrubbed — it can only be **rotated**. Keep all new values in `.env` only.

| # | Key (`.env` name) | Why | Rotate at | Status to confirm |
|---|---|---|---|---|
| 1 | `FRED_API_KEY` | Pasted to chat log (PHOENIX §3.6) | https://fred.stlouisfed.org/docs/api/api_key.html | ☐ rotated |
| 2 | `NEWS_API_KEY` | Pasted to chat log (PHOENIX §3.6) | https://newsapi.org/account | ☐ rotated |
| 3 | `WANDB_API_KEY` | Pasted to chat log (PHOENIX §3.6) | https://wandb.ai/authorize | ☐ rotated |
| 4 | `HF_TOKEN` | Pasted to chat log + the `# SCRUBBED` marker implies a working-tree token once existed | https://huggingface.co/settings/tokens | ☐ rotated + **re-issued write-scoped** for the new Space; confirm old token **revoked** |
| 5 | `NOAA_TOKEN` | Pasted to chat log (PHOENIX §3.6) | https://www.ncdc.noaa.gov/cdo-web/token | ☐ rotated |
| 6 | `GFW_API_TOKEN` | 798-char JWT; last liveness saw 503/422; JWTs cannot be revoked individually — **re-issue** | https://globalfishingwatch.org/our-apis/tokens | ☐ re-issued (old JWT will remain valid until expiry — treat as burned) |
| 7 | `OPENROUTER_API_KEY` | Primary AI gateway; check key **age** and set a spend cap; rotate if older than the last shared-machine session | https://openrouter.ai/keys | ☐ age-checked + spend-capped |
| 8 | `EIA_API_KEY`, `NASA_FIRMS_KEY` | Live in `.env` (per api-keys receipt); rotate if they were ever pasted into any log | respective portals | ☐ reviewed |

**Also:** delete/rotate any local `~/.huggingface/token`, `wandb` login cache, and shell-history
entries that may contain the pre-rotation values on this machine.

---

## 6. Deserialization / supply-chain risk (ACTION ITEMS for owning agents)

**Context — CVE-2025-32434.** PyTorch `torch.load` has a remote-code-execution vulnerability that
triggers **even with `weights_only=True`**, affecting torch ≤ 2.5.1, **fixed in torch 2.6.0**. This
machine runs **torch 2.5.1+cu121** (verified). Therefore *every* `torch.load` — regardless of
`weights_only` — is an RCE surface if the checkpoint came from an untrusted source. `weights_only=False`
is unconditionally unsafe (arbitrary pickle). This is especially relevant because **nb13 clones the
public HF Space as its code/checkpoint source** — a compromised Space could ship a malicious `.pt`.

### 6.1 First-line remediation (owner / env)
- **Upgrade `torch` to ≥ 2.6.0** in the venv and pin it in `requirements`/`pyproject`. This closes
  CVE-2025-32434 and flips the `torch.load` default to `weights_only=True`.
- Prefer **safetensors** for any weights loaded from a downloaded/remote artifact. (The
  `convert_bge_to_safetensors.py` pattern already does this for BGE-M3 — extend it.)
- Never `torch.load(..., weights_only=False)` on a file that was fetched from the HF Space, a URL,
  or any non-repo source. If provenance is trusted-local-only, add a comment asserting so.

### 6.2 `torch.load(weights_only=False)` call sites — file:line action items
*(All owned by other agents / out of this task's write scope — recorded here, not edited.)*

| File:line | Owner area | Action |
|---|---|---|
| `scripts/generate_hackathon_plots.py:54-55, 87-88` | scripts/ | `torch.load(rapxc.pt, weights_only=False)` — set `weights_only=True` or load via safetensors; assert local provenance. (Flagged in audit_5.) |
| `benchmark/run_full_benchmark.py:37` | benchmark/ | `weights_only=False` **inside the per-step loop** — set `weights_only=True` AND hoist the load out of the step loop (audit_3 §perf). |
| `rl/ensemble.py:86, 98` | rl/ | `weights_only=False` on DT/QR-DQN ckpts — set True (repo-local trusted). |
| `rl/specialist_router.py:67` | rl/ | `weights_only=False` — set True. |
| `rl/export_onnx.py:54` | rl/ | `weights_only=False` — set True. |
| `rl/record_video.py:96` | rl/ | `weights_only=False` — set True. |
| `rl/gnn/attention.py:214` | rl/ | `weights_only=False` — set True. |
| `rl/autoresearch_viz.py:42` | rl/ | `weights_only=False` — set True. |
| `rl/forecasting/tft.py:229` | rl/ | `weights_only=False` — set True. |
| `rl/forecasting/mc_dropout_eval.py:58` | rl/ | no `weights_only` kwarg → defaults **False** on torch 2.5.1 — set explicit `weights_only=True`. |
| `rl/interpretability/shap_real.py:67` | rl/ | no `weights_only` kwarg → defaults False — set explicit True. |
| `rl/forecasting/train_tft_real.py:180` | rl/ | no kwarg → defaults False — set explicit True. |
| `versions/v5_phoenix/arena/runner.py:138` | versions/ | `torch.load` no kwarg + accepts arbitrary `nn.Module` from a path — highest risk if the path is ever remote; require `weights_only=True` or a safetensors policy-format. |
| `versions/v3_arcadia/20_past_self/train_past_self.py:169` | versions/v3 (frozen) | `weights_only=False` — set True if revived. |
| `versions/v3_arcadia/00_emergence/{verify_qwen_vl.py:29-33, verify_embedders_chronos.py:30-34, convert_bge_to_safetensors.py:22-27}` | versions/v3 (frozen) | monkey-patch `torch.load` to force `weights_only=False` globally — scope the patch tightly or remove once on torch ≥ 2.6 + safetensors. |

**Safe (no action) — already `weights_only=True`:** `server/app.py:709`,
`rl/surrogate/world_model.py:253`.

### 6.3 `pickle.load` / `pickle.loads` of repo files — file:line action items
Pickle is arbitrary-code-execution on load. These read repo-local cache/corpus files; the risk is
realized only if an attacker can write those files (e.g. via a poisoned deploy artifact or a shared
cache dir). Mitigations: (a) restrict to trusted-local provenance with a comment, (b) prefer a safe
format (json / npz / safetensors) for anything regenerable, (c) verify a checksum before unpickling.

| File:line | Owner area | Action |
|---|---|---|
| `server/integrated_agent.py:109` | server/ (locked) | `pickle.load` of corpus chunks — validate provenance / switch to a safe format. |
| `rl/data/build_unified_buffer_v2.py:367` | rl/ | `pickle.load(model_obj)` — verify checksum / trusted source. |
| `versions/v4_arcadia_live/realtime/crisis_library.py:109` | versions/ | `pickle.loads(EMBED_CACHE_PATH.read_bytes())` — cache file; add checksum gate or use safetensors/npz for embeddings. |
| `versions/v3_arcadia/90_damocles/app.py:49` | versions/v3 (frozen) | `pickle.load` corpus chunks. |
| `versions/v3_arcadia/40_granite/r5_hard_queries.py:219` | versions/v3 (frozen) | `pickle.load` chunks. |
| `versions/v3_arcadia/10_caramel/shap_fairness_calibration.py:32` | versions/v3 (frozen) | `pickle.load`. |

---

## 7. Notebook / CI token handling

- **nb13 (`notebooks/13_MASTER_HACKATHON_FINAL.ipynb`) HF push — ALREADY FIXED.** The audit
  (audit_4) flagged "token interpolated into a shell command (would appear in process args)". The
  **current cell no longer does this**: it sets `os.environ['HF_TOKEN']`, writes a `GIT_ASKPASS`
  helper script that reads `$HF_TOKEN`/`$HF_USER` from the environment, and runs
  `GIT_ASKPASS=<helper> git push https://huggingface.co/spaces/...` — the token is **not** in
  `argv`, and the literal token is never written to the askpass file (only the env-var reference).
  There is even an inline comment: *"Never interpolate the token into shell args."* **Status: no
  action needed.** (Owner: notebooks/ agent — confirm this cell survives any nb13 cleanup.)
- **nb13 NOAA call** (`headers={'token': os.environ['NOAA_TOKEN']}`) passes the token as an HTTP
  header via `httpx`, not a shell arg — safe.
- **`scripts/push_to_hf_space.py` / `push_to_hf_space_minimal.py`** read `HF_TOKEN` from env and pass
  it to `HfApi(token=...)` — safe (not in argv). (These scripts are slated for deletion per CLAUDE.md
  §6.5 anyway; scripts/ owner.)
- **CI** (`.github/workflows/deploy-hf-space.yml`) — confirm the HF token is consumed only as a
  GitHub Actions **secret** (`${{ secrets.HF_TOKEN }}`) and never `echo`ed into logs. (CI is owned by
  the ci-deploy agent — verify at retarget time; see blockers.)

---

## 8. Git-history remediation plan (CONDITIONAL — not needed today)

**No history rewrite is required** because no secret was found in any committed blob. This section is
a *standby plan only*; do **NOT** rewrite shared history without explicit owner approval and a
coordinated force-push (it breaks every clone and open PR).

If a real secret is ever discovered in history later:
1. **Rotate first** (a purged secret is still burned once it was ever pushed). See §5.
2. Install `git-filter-repo` (preferred) or BFG Repo-Cleaner.
3. Dry-run on a mirror clone: `git clone --mirror <repo> repo.git`.
4. Purge by blob/pattern, e.g.
   `git filter-repo --replace-text patterns.txt` (patterns file maps the secret → `***REMOVED***`),
   or `--path <file> --invert-paths` to drop a file entirely.
5. `git push --force --all && git push --force --tags` **after** every collaborator is warned.
6. Expire reflogs + `git gc --prune=now --aggressive` locally; ask GitHub/HF support to garbage-collect
   server-side and invalidate cached views.
7. Have all collaborators re-clone (rebasing old clones re-introduces the secret).

---

## 9. Hardening rules going forward

1. **Pre-commit secret scan.** Add `gitleaks` or `detect-secrets` as a pre-commit hook (and a CI
   job) so no future key reaches a commit. Baseline it against this clean tree.
2. **Receipts must strip auth.** Any receipt/proof that records an HTTP call must store only
   `status_code` + a sha256 of the (redacted) body — as `api_keys_live_proof.json` already does.
   **Never** log request headers, `Authorization`, cookies, or response bodies that could echo a key.
3. **Tokens via env / askpass only.** Never interpolate a secret into a shell command line (it leaks
   via the process table and shell history). Use env vars + `GIT_ASKPASS` (nb13's pattern) or a
   library `token=` kwarg.
4. **`.env` is the only secret store.** Keep `.env.example` in sync (names + placeholders only). Add
   any new key there without a value.
5. **Deserialization policy.** torch ≥ 2.6.0 pinned; `weights_only=True` everywhere; safetensors for
   remote/downloaded weights; checksum-gate any `pickle.load` of a non-repo file.
6. **CI secrets.** Only GitHub Actions secrets; never `echo`/print them; scrub deploy logs before
   uploading as artifacts.
7. **Key age rotation.** Rotate all live keys on a schedule and immediately after any shared-machine
   or shared-log session.

---

## 10. Scan reproduction commands (evidence)

```sh
# History (all refs, all 291 commits)
git log --all -S 'hf_' --oneline; git log --all -G 'hf_[A-Za-z0-9]{34,}' --oneline
git log --all -G 'sk-or-v1-[A-Fa-f0-9]{48,}' --oneline
git log --all -G 'ghp_[A-Za-z0-9]{34,}|github_pat_[A-Za-z0-9_]{60,}' --oneline
git log --all -G 'AKIA[0-9A-Z]{16}' --oneline
git log --all -G 'eyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.' --oneline
git log --all -- .env               # -> empty (never committed)

# Whole object store incl. dangling (definitive)
git cat-file --batch-all-objects --batch --buffer \
  | grep -aoE 'hf_[A-Za-z0-9]{34,}|sk-or-v1-[A-Fa-f0-9]{48,}|ghp_[A-Za-z0-9]{34,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.'
# -> no output (clean)
git fsck --no-reflogs | grep -iE 'dangling|unreachable'

# Working tree (tracked files)
git ls-files -z | xargs -0 grep -aInE 'hf_[A-Za-z0-9]{34,}|sk-or-v1-[A-Fa-f0-9]{48,}|eyJ[A-Za-z0-9_-]{12,}\.eyJ'
git check-ignore -v .env            # -> .gitignore:6:.env
```
