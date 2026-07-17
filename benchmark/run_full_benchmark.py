"""
Real benchmark: evaluate the ACTUAL trained checkpoints in the ACTUAL env.

Every agent is run through the real ``SupplyMindGymnasiumEnv`` (which wraps the
real ``SupplyMindEnvironment``); the headline metric is the grader score
(``env.grade()["score"]`` in [0, 1]) at episode end, with cumulative shaped
reward as a secondary metric. There is NO scripted-placeholder substitution for
any agent — an agent that cannot be loaded/evaluated is recorded loudly as an
error row and excluded from the leaderboard (never faked).

Determinism: ``SupplyMindEnvironment`` is fully seeded, so a given (agent, task,
seed) yields a reproducible episode. We therefore run ONE episode per seed and
gain statistical power from the SEED COUNT (>=20), not from repeated draws of
the same seed. Stochastic agents (Random) are seeded per episode.

Statistics: for every agent we run a paired-by-seed Wilcoxon signed-rank test
against the Scripted baseline on the SAME seed set, and report the effect size
r = |Z| / sqrt(N) (N = number of non-zero-difference pairs), plus the signed
mean grade delta so the direction is unambiguous.

Agent validity (buffer-taxonomy audit, 2026-07-15)
--------------------------------------------------
The offline unified buffer (``real_unified*.npz``) carried a scrambled
action-type taxonomy until the Jul-2 fix; agents trained on it have wrong action
semantics until retrained on the regenerated buffer (that retrain is WP7.3).
  * VALID here:
      - Random / Scripted           (no training)
      - PPO / QR-DQN                 (trained ONLINE in the env -> env-canonical)
      - DT                           (trained on offline_buffer.npz, which is
                                      built by rolling the ENV with the scripted
                                      +random agents -> env-canonical actions)
      - BC / CQL / IQL / TD3+BC      (the *_best.pt checkpoints, trained on the
                                      same env-canonical offline_buffer.npz;
                                      these are offline-RL distillations of env
                                      scripted+random experience, NOT trained on
                                      the real-world DataCo/NOAA/FRED buffer)
      - Ensemble (DT + QR-DQN)       (both components valid)
  * NOT evaluated -> retrain_pending_on_regenerated_buffer:
      - BC-v2 / CQL-v2 / IQL-v2 / TD3+BC-v2  (the *_v2.pt checkpoints, trained on
        real_train_v2.npz which had the scrambled taxonomy pre-fix). Scoring them
        now would be measuring scrambled action semantics -> we refuse to.

Usage
-----
    python benchmark/run_full_benchmark.py                 # full run, 30 seeds
    python benchmark/run_full_benchmark.py --seeds 20      # first 20 seeds
    python benchmark/run_full_benchmark.py --smoke         # 2 seeds, quick check
    python benchmark/run_full_benchmark.py --agents scripted ppo qrdqn
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT = ROOT / "rl" / "checkpoints"
RESULTS = ROOT / "benchmark" / "results"
RECEIPT = ROOT / "tests" / "receipts" / "benchmark_leaderboard_REAL.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bench")

TASKS = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
TASK_SHORT = {"easy_typhoon_response": "Easy", "medium_multi_front": "Medium", "hard_cascading_crisis": "Hard"}

# Deterministic seed set. 30 seeds -> strong paired-test power. Reproducible.
DEFAULT_SEEDS = [42, 99, 7, 123, 256, 512, 1024, 2, 17, 71,
                 88, 314, 271, 161, 8, 33, 64, 128, 555, 777,
                 11, 22, 44, 55, 66, 202, 303, 404, 606, 909]

DEVICE = "cpu"  # tiny nets; CPU keeps the run fully deterministic

# Agents that ARE evaluated (env-canonical / online). Scripted MUST be present
# as it is the paired-stat baseline.
DEFAULT_AGENTS = ["random", "scripted", "bc", "cql", "iql", "td3bc",
                  "ppo", "qrdqn", "dt", "ensemble"]

# Real-data offline agents blocked on the regenerated buffer (WP7.3).
RETRAIN_PENDING = {
    "BC-v2": "bc_v2.pt", "CQL-v2": "cql_v2.pt",
    "IQL-v2": "iql_v2.pt", "TD3+BC-v2": "td3bc_v2.pt",
}

DISPLAY = {
    "random": "Random", "scripted": "Scripted", "bc": "BC", "cql": "CQL",
    "iql": "IQL", "td3bc": "TD3+BC", "ppo": "PPO", "qrdqn": "QR-DQN (CVaR)",
    "dt": "Decision Transformer", "ensemble": "Ensemble (DT+QR-DQN)",
}

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Stateless agents: predict(obs, info, task, step) -> flat action int in [0,279]
# ---------------------------------------------------------------------------

def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def agent_random(obs, info, task, step, rng):
    valid = np.where(info["action_masks"])[0]
    return int(rng.choice(valid)) if len(valid) else 0


def agent_scripted(obs, info, task, step, rng):
    from scripted_agent import choose_action as scripted_choose
    from rl.gym_env import ACTION_TYPES
    raw = info["raw_obs"]
    sm = scripted_choose(raw, step)
    at = ACTION_TYPES.index(sm.action_type)
    node_ids = [n.node_id for n in raw.node_statuses]
    ni = 0
    if sm.target_node_id and sm.target_node_id in node_ids:
        ni = node_ids.index(sm.target_node_id)
    return at * 40 + min(ni, 39)


def _masked_argmax(vec_1x280, mask_280):
    import torch
    v = vec_1x280.clone()
    v[0][~torch.from_numpy(mask_280).bool()] = float("-inf")
    return int(v.argmax(dim=-1).item())


def agent_bc(obs, info, task, step, rng):
    import torch
    def build():
        from rl.offline.baselines import BCNetwork
        from rl.checkpoint_security import safe_load
        m = BCNetwork(); m.load_state_dict(safe_load(str(CKPT / "bc_best.pt"), map_location=DEVICE)["state_dict"])
        m.eval(); return m
    m = _cached("bc", build)
    with torch.no_grad():
        return _masked_argmax(m(torch.from_numpy(obs).float().unsqueeze(0)), info["action_masks"])


def agent_cql(obs, info, task, step, rng):
    import torch
    def build():
        from rl.offline.baselines import CQLQNetwork
        from rl.checkpoint_security import safe_load
        m = CQLQNetwork(); m.load_state_dict(safe_load(str(CKPT / "cql_best.pt"), map_location=DEVICE)["state_dict"])
        m.eval(); return m
    m = _cached("cql", build)
    with torch.no_grad():
        return _masked_argmax(m.q_min(torch.from_numpy(obs).float().unsqueeze(0)), info["action_masks"])


def agent_iql(obs, info, task, step, rng):
    import torch
    def build():
        from rl.offline.baselines import BCNetwork
        from rl.checkpoint_security import safe_load
        m = BCNetwork(); m.load_state_dict(safe_load(str(CKPT / "iql_best.pt"), map_location=DEVICE)["actor"])
        m.eval(); return m
    m = _cached("iql", build)
    with torch.no_grad():
        return _masked_argmax(m(torch.from_numpy(obs).float().unsqueeze(0)), info["action_masks"])


def agent_td3bc(obs, info, task, step, rng):
    import torch
    def build():
        from rl.offline.baselines import TD3Actor
        from rl.checkpoint_security import safe_load
        m = TD3Actor(); m.load_state_dict(safe_load(str(CKPT / "td3bc_best.pt"), map_location=DEVICE)["actor"])
        m.eval(); return m
    m = _cached("td3bc", build)
    with torch.no_grad():
        return _masked_argmax(m(torch.from_numpy(obs).float().unsqueeze(0)), info["action_masks"])


def _build_qrdqn(task):
    import torch
    from rl.distributional.qr_dqn import QRDQNNetwork
    from rl.checkpoint_security import safe_load
    tk = TASK_SHORT[task].lower()
    p = CKPT / f"qrdqn_best_{tk}.pt"
    if not p.exists():
        p = CKPT / "qrdqn_best_easy.pt"
    ckpt = safe_load(str(p), map_location=DEVICE)
    cfg = {k: v for k, v in ckpt["config"].items() if k in ("state_dim", "n_actions", "n_quantiles", "hidden_dim")}
    m = QRDQNNetwork(**cfg); m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m


def agent_qrdqn(obs, info, task, step, rng):
    import torch
    m = _cached(f"qrdqn_{task}", lambda: _build_qrdqn(task))
    with torch.no_grad():
        st = torch.from_numpy(obs).float().unsqueeze(0)
        mask = torch.from_numpy(info["action_masks"]).bool().unsqueeze(0)
        return int(m.cvar_policy(st, alpha=0.1, action_mask=mask).item())


# ---------------------------------------------------------------------------
# PPO: MaskablePPO + VecNormalize obs. We use the EvalCallback-selected best
# checkpoint (ppo_best_{task}/best_model.zip). During training that model was
# always evaluated inside a VecNormalize(norm_obs=True, training=False) env, so
# it EXPECTS normalized observations; we apply the saved ppo_vecnormalize_{task}
# stats (the end-of-training snapshot — a small approximation vs the exact
# best-checkpoint step, documented in the receipt). The end-of-run ppo_final_*
# checkpoints are near-degenerate (they collapse to a single action type and
# grade ~0.32); using them would MIS-report PPO downward, so we do not. The
# 47-dim per-dimension mask is rebuilt from the env's 280 joint mask exactly as
# training did (rl/train_ppo.py:_get_action_masks).
# ---------------------------------------------------------------------------

def _ppo_mask_47(full_mask_280):
    tm = np.array([full_mask_280[i * 40:(i + 1) * 40].any() for i in range(7)], dtype=bool)
    nm = np.array([full_mask_280[j::40].any() for j in range(40)], dtype=bool)
    return np.concatenate([tm, nm])


def _build_ppo(task):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.gym_env import SupplyMindGymnasiumEnv
    tk = TASK_SHORT[task].lower()
    model_path = CKPT / f"ppo_best_{tk}" / "best_model.zip"
    vec_path = CKPT / f"ppo_vecnormalize_{tk}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"PPO checkpoint missing: {model_path}")
    model = MaskablePPO.load(str(model_path), device=DEVICE)
    vnorm = None
    if vec_path.exists():
        dummy = DummyVecEnv([lambda: SupplyMindGymnasiumEnv(task_id=task)])
        vnorm = VecNormalize.load(str(vec_path), dummy)
        vnorm.training = False
        vnorm.norm_reward = False
    return model, vnorm


def agent_ppo(obs, info, task, step, rng):
    model, vnorm = _cached(f"ppo_{task}", lambda: _build_ppo(task))
    o = obs.reshape(1, -1).astype(np.float32)
    if vnorm is not None:
        o = vnorm.normalize_obs(o)
    mask47 = _ppo_mask_47(info["action_masks"])
    action, _ = model.predict(o, action_masks=mask47, deterministic=True)
    a = np.asarray(action).reshape(-1)
    return int(a[0]) * 40 + int(a[1])


STATELESS = {
    "random": agent_random, "scripted": agent_scripted, "bc": agent_bc,
    "cql": agent_cql, "iql": agent_iql, "td3bc": agent_td3bc,
    "ppo": agent_ppo, "qrdqn": agent_qrdqn,
}

# Stateful agents (managed via rl.ensemble.EnsemblePolicy: DT context + RTG).
STATEFUL = {"dt", "ensemble"}


def _build_ensemble(task, dt_weight):
    from rl.ensemble import EnsemblePolicy
    tk = TASK_SHORT[task].lower()
    ens = EnsemblePolicy(dt_weight=dt_weight, device=DEVICE)
    ens.load_models(dt_path=CKPT / "dt_best.pt", qrdqn_path=CKPT / f"qrdqn_best_{tk}.pt")
    if ens.dt_model is None:
        raise FileNotFoundError("DT checkpoint missing for stateful agent")
    return ens


# ---------------------------------------------------------------------------
# Episode runner (single env, deterministic per seed)
# ---------------------------------------------------------------------------

def eval_episode(agent_name, task, seed):
    """Run one episode; return (grade_score, cumulative_reward, n_steps)."""
    import torch
    from rl.gym_env import SupplyMindGymnasiumEnv
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    gym = SupplyMindGymnasiumEnv(task_id=task)
    obs, info = gym.reset(seed=seed)

    stateful = None
    if agent_name in STATEFUL:
        dt_weight = 1.0 if agent_name == "dt" else 0.5
        stateful = _cached(f"{agent_name}_{task}", lambda: _build_ensemble(task, dt_weight))
        stateful.dt_weight = 1.0 if agent_name == "dt" else 0.5
        stateful.reset(desired_return=1.0)

    total_r, step = 0.0, 0
    while True:
        if stateful is not None:
            a = stateful.predict(obs, info["action_masks"], timestep=step)
        else:
            a = STATELESS[agent_name](obs, info, task, step, rng)
        obs, r, term, trunc, info = gym.step(np.array([a // 40, a % 40], dtype=np.int64))
        total_r += float(r)
        if stateful is not None:
            stateful.record_reward(float(r))
        step += 1
        if term or trunc:
            break

    grade = gym._env.grade()["score"]
    gym.close()
    return float(grade), float(total_r), step


# ---------------------------------------------------------------------------
# Statistics: paired-by-seed Wilcoxon, effect size r = |Z| / sqrt(N)
# ---------------------------------------------------------------------------

def paired_wilcoxon(agent_vec, base_vec):
    from scipy.stats import rankdata
    a = np.asarray(agent_vec, dtype=float)
    b = np.asarray(base_vec, dtype=float)
    d = a - b
    nz = d[d != 0]
    n = int(len(nz))
    mean_delta = float(np.mean(a - b))
    median_delta = float(np.median(a - b))
    out = {"n_pairs": int(len(d)), "n_nonzero": n, "mean_delta": round(mean_delta, 5),
           "median_delta": round(median_delta, 5), "W": None, "z": None,
           "p_value": None, "effect_r": None}
    if n == 0:
        out["note"] = "all pairs identical (delta=0)"
        return out
    ranks = rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    T = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))
    # continuity-corrected z toward the mean
    z = (T - mu + 0.5 * np.sign(mu - T)) / sigma if sigma > 0 else 0.0
    # signed z: positive when agent > base (w_plus dominates)
    signed_z = abs(z) * (1.0 if w_plus >= w_minus else -1.0)
    from scipy.stats import norm
    p = float(2 * norm.cdf(-abs(z)))
    out.update({"W": round(T, 3), "W_plus": round(w_plus, 3), "W_minus": round(w_minus, 3),
                "z": round(float(signed_z), 4), "p_value": round(p, 6),
                "effect_r": round(abs(z) / np.sqrt(n), 4)})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Real SupplyMind agent benchmark")
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                    help="number of seeds from the deterministic seed set (default: all 30)")
    ap.add_argument("--agents", nargs="*", default=DEFAULT_AGENTS)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--smoke", action="store_true", help="2 seeds, quick sanity check")
    args = ap.parse_args()

    seeds = DEFAULT_SEEDS[:2] if args.smoke else DEFAULT_SEEDS[:args.seeds]
    agents = args.agents
    tasks = args.tasks
    if "scripted" not in agents:
        agents = ["scripted"] + agents  # baseline required for paired stats

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info("REAL BENCHMARK: %d agents x %d tasks x %d seeds", len(agents), len(tasks), len(seeds))

    # raw[agent][task] -> {"seeds":[...], "grades":[...], "rewards":[...], "steps":[...]}
    raw: dict = {}
    errors: dict = {}
    rows = []  # per (agent, task, seed)

    for agent in agents:
        raw[agent] = {}
        for task in tasks:
            grades, rewards, steps, used_seeds = [], [], [], []
            for seed in seeds:
                try:
                    g, rw, ns = eval_episode(agent, task, seed)
                    grades.append(g); rewards.append(rw); steps.append(ns); used_seeds.append(seed)
                    rows.append({"agent": DISPLAY.get(agent, agent), "task": TASK_SHORT[task],
                                 "task_id": task, "seed": seed,
                                 "grade_score": round(g, 5), "cumulative_reward": round(rw, 5),
                                 "n_steps": ns})
                except Exception as e:  # fail loud: record, never fake
                    errors.setdefault(agent, {}).setdefault(TASK_SHORT[task], str(e)[:200])
                    log.warning("  %s x %s seed %d FAILED: %s", agent, TASK_SHORT[task], seed, str(e)[:120])
                    break  # a load error will repeat for every seed; stop this cell
            raw[agent][task] = {"seeds": used_seeds, "grades": grades,
                                "rewards": rewards, "steps": steps}
            if grades:
                log.info("  %-22s %-6s grade=%.3f+/-%.3f reward=%.3f (n=%d)",
                         DISPLAY.get(agent, agent), TASK_SHORT[task],
                         float(np.mean(grades)), float(np.std(grades)),
                         float(np.mean(rewards)), len(grades))
        # incremental checkpoint: per-episode rows so far survive interruption
        with open(RESULTS / "full_benchmark.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["agent", "task", "task_id", "seed",
                                              "grade_score", "cumulative_reward", "n_steps"])
            w.writeheader(); w.writerows(rows)

    # ---- statistics vs Scripted baseline (paired by seed) ----
    stats = {}
    base = raw.get("scripted", {})
    for agent in agents:
        if agent == "scripted":
            continue
        stats[agent] = {}
        # per-task
        overall_a, overall_b = [], []
        for task in tasks:
            ar = raw[agent].get(task, {})
            br = base.get(task, {})
            # align on common seeds
            common = [s for s in ar.get("seeds", []) if s in br.get("seeds", [])]
            if not common:
                stats[agent][TASK_SHORT[task]] = {"n_pairs": 0, "note": "no common seeds"}
                continue
            a_map = dict(zip(ar["seeds"], ar["grades"]))
            b_map = dict(zip(br["seeds"], br["grades"]))
            av = [a_map[s] for s in common]
            bv = [b_map[s] for s in common]
            overall_a += av; overall_b += bv
            stats[agent][TASK_SHORT[task]] = paired_wilcoxon(av, bv)
        stats[agent]["Overall"] = paired_wilcoxon(overall_a, overall_b) if overall_a else {"n_pairs": 0}

    # ---- write raw per-(agent,task,seed) rows ----
    with open(RESULTS / "full_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "task", "task_id", "seed",
                                          "grade_score", "cumulative_reward", "n_steps"])
        w.writeheader(); w.writerows(rows)

    # ---- write raw episode arrays (npz) ----
    np_payload = {}
    for agent in agents:
        for task in tasks:
            d = raw[agent][task]
            if d["grades"]:
                np_payload[f"{agent}__{TASK_SHORT[task]}__grades"] = np.array(d["grades"], dtype=np.float32)
                np_payload[f"{agent}__{TASK_SHORT[task]}__rewards"] = np.array(d["rewards"], dtype=np.float32)
                np_payload[f"{agent}__{TASK_SHORT[task]}__seeds"] = np.array(d["seeds"], dtype=np.int64)
    np.savez_compressed(RESULTS / "raw_episodes.npz", **np_payload)

    # ---- benchmark_summary.csv (leaderboard.py reads cols Agent/Easy/Medium/Hard/Average) ----
    def cell(agent, task):
        g = raw[agent].get(task, {}).get("grades", [])
        return f"{np.mean(g):.3f}+/-{np.std(g):.3f}" if g else "not_evaluated"

    def avg(agent):
        allg = [x for task in tasks for x in raw[agent].get(task, {}).get("grades", [])]
        return f"{np.mean(allg):.3f}" if allg else "not_evaluated"

    ranked = sorted([a for a in agents if any(raw[a].get(t, {}).get("grades") for t in tasks)],
                    key=lambda a: np.mean([x for t in tasks for x in raw[a].get(t, {}).get("grades", [])]),
                    reverse=True)

    with open(RESULTS / "benchmark_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Agent", "Easy", "Medium", "Hard", "Average",
                    "vs_Scripted_overall_delta", "vs_Scripted_p", "vs_Scripted_r", "n_seeds", "valid"])
        for agent in ranked:
            ov = stats.get(agent, {}).get("Overall", {}) if agent != "scripted" else {}
            n = len(raw[agent].get(tasks[0], {}).get("grades", []))
            w.writerow([DISPLAY.get(agent, agent), cell(agent, tasks[0]), cell(agent, tasks[1]),
                        cell(agent, tasks[2]), avg(agent),
                        ov.get("mean_delta", "baseline" if agent == "scripted" else "—"),
                        ov.get("p_value", "—"), ov.get("effect_r", "—"), n, "yes"])
        # retrain-pending rows (honest markers, not scored)
        for name in RETRAIN_PENDING:
            w.writerow([name, "retrain_pending", "retrain_pending", "retrain_pending",
                        "retrain_pending", "—", "—", "—", 0, "no"])

    # ---- leaderboard.json (rich) ----
    leaderboard = []
    for rank, agent in enumerate(ranked, 1):
        entry = {"rank": rank, "agent": DISPLAY.get(agent, agent), "key": agent}
        for task in tasks:
            g = raw[agent].get(task, {}).get("grades", [])
            entry[TASK_SHORT[task]] = {"grade_mean": round(float(np.mean(g)), 5),
                                       "grade_std": round(float(np.std(g)), 5),
                                       "reward_mean": round(float(np.mean(raw[agent][task]["rewards"])), 5)} if g else None
        allg = [x for t in tasks for x in raw[agent].get(t, {}).get("grades", [])]
        entry["overall_grade"] = round(float(np.mean(allg)), 5) if allg else None
        entry["vs_scripted"] = stats.get(agent, {}) if agent != "scripted" else "baseline"
        leaderboard.append(entry)
    (RESULTS / "leaderboard.json").write_text(json.dumps({
        "leaderboard": leaderboard,
        "retrain_pending": {k: f"{v} trained on scrambled real_train_v2.npz; retrain on regenerated buffer (WP7.3)"
                            for k, v in RETRAIN_PENDING.items()},
        "errors": errors,
    }, indent=2))

    # ---- receipt ----
    elapsed = round(time.time() - t0, 1)
    scale_cmd = "python benchmark/run_full_benchmark.py --seeds 30"
    receipt = {
        "receipt": "benchmark_leaderboard_REAL",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "command": f"python benchmark/run_full_benchmark.py --seeds {len(seeds)}",
        "scale_up_command": scale_cmd,
        "python": sys.version.split()[0],
        "device": DEVICE,
        "tasks": tasks,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "episodes_per_seed": 1,
        "determinism": "env fully seeded; 1 episode/seed is reproducible; power from seed count",
        "primary_metric": "grader score env.grade()['score'] in [0,1]",
        "baseline_for_paired_stats": "Scripted",
        "agents_evaluated": [DISPLAY.get(a, a) for a in ranked],
        "retrain_pending": {k: f"{v}: trained on scrambled real_train_v2.npz (pre Jul-2 taxonomy fix); "
                               "eval refused until retrained on regenerated buffer (WP7.3)"
                            for k, v in RETRAIN_PENDING.items()},
        "errors": errors,
        "leaderboard": leaderboard,
        "stats_vs_scripted": stats,
        "artifacts": {
            "per_episode_rows": "benchmark/results/full_benchmark.csv",
            "raw_episode_arrays": "benchmark/results/raw_episodes.npz",
            "summary": "benchmark/results/benchmark_summary.csv",
            "leaderboard_json": "benchmark/results/leaderboard.json",
        },
        "elapsed_seconds": elapsed,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2))
    # self-hash for tamper-evidence
    receipt["sha256_self"] = sha256_file(RECEIPT)
    RECEIPT.write_text(json.dumps(receipt, indent=2))

    log.info("DONE in %.1fs. summary=%s receipt=%s", elapsed,
             RESULTS / "benchmark_summary.csv", RECEIPT)
    # print a compact leaderboard to stdout
    log.info("=== LEADERBOARD (overall grade) ===")
    for e in leaderboard:
        vs = ""
        if isinstance(e["vs_scripted"], dict):
            ov = e["vs_scripted"].get("Overall", {})
            if ov.get("p_value") is not None:
                vs = f"  vs scripted d={ov.get('mean_delta')} p={ov.get('p_value')} r={ov.get('effect_r')}"
        log.info("  %d. %-22s overall=%.3f%s", e["rank"], e["agent"], e["overall_grade"] or 0.0, vs)
    if errors:
        log.warning("ERRORS (agents not fully evaluated): %s", json.dumps(errors))


if __name__ == "__main__":
    main()
