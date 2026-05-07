# PR draft — meta-pytorch/openenv: SupplyMind as a reference environment

**Target**: https://github.com/meta-pytorch/openenv
**Branch name**: `add-supplymind-reference-env`
**Status**: draft (will open during 48 h finals, Apr 25–26 2026)

---

## Title

`Add SupplyMind: a real-data supply-chain risk environment (3 tasks, Pydantic-v2, Docker)`

## Body

### Summary

This PR adds SupplyMind as a reference environment for OpenEnv: an OpenEnv-
compliant supply-chain risk management environment with three difficulty-
calibrated tasks (Typhoon Response, Multi-Front Crisis, Cascading Crisis),
a 408-dimensional observation space, a MultiDiscrete[7, 40] action space,
and 261K real data points for calibration (DataCo, NOAA, FRED, World Bank,
SEC 10-K, Wikipedia).

Trained MaskablePPO policies on each task achieve non-overlapping 95 % CIs
vs random + greedy baselines (R6 Euclidian: 10,800-episode bootstrap
benchmark). Policies ship as ONNX bundles for easy re-import.

### What's in this PR

```
examples/supplymind/
├── README.md                     # env overview + how-to-run
├── openenv.yaml                  # spec declaration
├── supplymind_env.py             # Gymnasium/OpenEnv-compatible wrapper
├── tasks/
│   ├── easy_typhoon_response.json
│   ├── medium_multi_front.json
│   └── hard_cascading_crisis.json
├── policies/
│   ├── maskable_ppo_easy.onnx    # 0.97 MB
│   ├── maskable_ppo_medium.onnx
│   └── maskable_ppo_hard.onnx
├── tests/
│   └── test_openenv_compliance.py
├── Dockerfile
└── docker-compose.yml
```

### Compliance checklist

- [x] `openenv.yaml` declares env_id, action_schema, observation_schema, 3 tasks
- [x] Pydantic-v2 typed action + observation models
- [x] `reset(seed)` / `step(action)` / `grade()` surface implemented
- [x] 19 formal compliance tests (test_openenv_compliance.py)
- [x] Dockerfile builds; `docker-compose up` boots the FastAPI runtime
- [x] Works with `openenv-core >= 0.2.0`
- [x] Example training notebook at `notebooks/01_environment_quickstart.ipynb`
- [x] No network dependencies at eval-time (live signals are opt-in)

### How to run

```bash
pip install openenv-core
git clone <this-fork> -b add-supplymind-reference-env
cd openenv-course
python -m pytest examples/supplymind/tests -q
docker compose -f examples/supplymind/docker-compose.yml up -d
curl http://localhost:8000/health
```

### Why this belongs here

- **Real data** — no simulated substitution anywhere; all disruption severity
  distributions, lead times, and cost parameters derived from public sources
  (listed in `examples/supplymind/README.md` §data).
- **Discrete action space** — complements existing MuJoCo / gaming envs.
- **Multi-task** — 3 calibrated difficulty tiers, not just a single task.
- **Production-ready** — Docker-hardened, HF-Space-ready, ONNX-exported policies.
- **Reproducibility** — 19 compliance tests + 10,800-episode benchmark with
  non-overlapping CIs published.

### Benchmarks included

Full results at `examples/supplymind/benchmarks/R6_EUCLIDIAN.json`:

| Task | Policy | Reward mean | 95 % CI |
|---|---|---|---|
| easy | MaskablePPO | 1.200 | [1.186, 1.215] |
| medium | MaskablePPO | 2.776 | [2.758, 2.795] |
| hard | MaskablePPO | 2.652 | [2.596, 2.708] |

### Attribution

Author: ShAuRyA-Noodle (Meta PyTorch OpenEnv Hackathon 2026 finalist).
Trained on NVIDIA RTX 4080 Laptop 12 GB VRAM.

### Open questions for maintainers

1. Should the env live under `examples/supplymind/` or as a top-level
   directory alongside module-1..5? Happy either way.
2. Policy ONNX bundle is ~3 MB — is that acceptable to vendor in-tree,
   or should we host on HF Hub and pull at test time?
3. Do you want the live-pipeline feature gated (NewsAPI opt-in) or removed
   entirely from the upstream version?

### License

MIT (matches the course repo + hackathon submission rules).

---

## Pre-merge checklist for me

- [ ] Fork meta-pytorch/openenv to ShAuRyA-Noodle/openenv
- [ ] Create branch `add-supplymind-reference-env`
- [ ] Copy `examples/supplymind/` materials from this folder into the branch
- [ ] Verify `pytest examples/supplymind/tests -q` passes on the fork
- [ ] Verify Dockerfile builds on clean machine
- [ ] Open PR, link to SupplyMind's main repo README
- [ ] Add "hackathon" label if available
- [ ] Notify maintainers via issue/email during hackathon finals

## Files to copy from this fork into the branch

Source -> Target:
```
versions/v3_arcadia/results/R6_EUCLIDIAN.json              -> examples/supplymind/benchmarks/R6_EUCLIDIAN.json
versions/v3_arcadia/checkpoints/onnx_bundle/*              -> examples/supplymind/policies/
server/supply_environment.py                      -> examples/supplymind/supplymind_env.py
server/tasks/                                     -> examples/supplymind/tasks/
tests/test_openenv_compliance.py                  -> examples/supplymind/tests/
openenv.yaml                                      -> examples/supplymind/openenv.yaml
Dockerfile                                        -> examples/supplymind/Dockerfile
```

A build script to do the copy mechanically lives at
`versions/v5_phoenix/upstream_prs/meta_openenv/build_pr_branch.sh`.
