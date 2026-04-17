"""
SINGLE SCRIPT: Improves everything in one clean run.

1. Generate 1000-episode dataset (proven to work, ~55 min)
2. Retrain offline agents with grade-aligned data
3. Run AutoResearch specialists on all 3 tasks
4. Full benchmark with grade() scores for ALL agents
5. Statistics + backtesting
6. Final pytest verification

Uses Python 3.14 system python with CUDA.
"""

import sys
import os
import time
import gc
import logging
import json
import csv

os.chdir("c:/Users/Dell/Desktop/Sleep-Token")
sys.path.insert(0, ".")

import torch
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("improve_everything.log", mode="w")],
)
logger = logging.getLogger("improve")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("=" * 70)
logger.info("IMPROVE EVERYTHING — ONE CLEAN RUN")
logger.info("GPU: %s | PyTorch: %s | CUDA: %s",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            torch.__version__, torch.version.cuda)
logger.info("=" * 70)

results = {}
overall_start = time.time()


def run(name, fn):
    logger.info("")
    logger.info("[START] %s", name)
    t = time.time()
    try:
        fn()
        elapsed = (time.time() - t) / 60
        results[name] = round(elapsed, 1)
        logger.info("[DONE] %s — %.1f min", name, elapsed)
    except Exception as e:
        results[name] = f"FAILED: {str(e)[:120]}"
        logger.error("[FAIL] %s: %s", name, e, exc_info=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================================================
# STEP 1: Generate fresh 1000-episode dataset
# ================================================================
def step_dataset():
    from rl.offline.dataset import generate_dataset
    generate_dataset(n_scripted=500, n_random=500)

run("1_dataset_1000ep", step_dataset)

# ================================================================
# STEP 2: AutoResearch specialists (grade-aligned, best config)
# ================================================================
def step_specialists():
    from rl.autoresearch import ExperimentConfig, train_experiment

    best_cfg = dict(lr=1e-3, cvar_alpha=0.5, hidden_dim=256, grade_reward=True,
                    alert_penalty=0.0, real_action_bonus=0.10, gamma=0.99, n_quantiles=51)

    for task_id, name in [("easy_typhoon_response", "easy"),
                           ("medium_multi_front", "medium"),
                           ("hard_cascading_crisis", "hard")]:
        logger.info("  Training %s specialist (200K steps)...", name)
        config = ExperimentConfig(name=f"final_{name}", task_id=task_id, total_steps=200_000, **best_cfg)
        result = train_experiment(config, device=device)
        logger.info("  %s: easy=%.3f med=%.3f hard=%.3f avg=%.3f",
                     name, result.grade_easy, result.grade_medium, result.grade_hard, result.grade_avg)
        gc.collect()
        torch.cuda.empty_cache()

run("2_specialists", step_specialists)

# ================================================================
# STEP 3: Retrain offline agents on fresh data
# ================================================================
def step_offline():
    from rl.offline.baselines import train_iql, train_cql, train_td3bc, train_bc

    logger.info("  IQL 100K...")
    train_iql(n_steps=100_000, device=device)
    gc.collect(); torch.cuda.empty_cache()

    logger.info("  CQL 100K...")
    train_cql(n_steps=100_000, device=device)
    gc.collect(); torch.cuda.empty_cache()

    logger.info("  TD3+BC 100K...")
    train_td3bc(n_steps=100_000, device=device)
    gc.collect(); torch.cuda.empty_cache()

    logger.info("  BC 100 epochs...")
    train_bc(epochs=100, device=device)
    gc.collect(); torch.cuda.empty_cache()

run("3_offline_agents", step_offline)

# ================================================================
# STEP 4: DT 30 epochs
# ================================================================
def step_dt():
    from rl.decision_transformer.train import train_dt
    train_dt(epochs=30, device=device, log_wandb=False, log_mlflow=False)

run("4_dt", step_dt)

# ================================================================
# STEP 5: Surrogate 50 epochs
# ================================================================
def step_surrogate():
    from rl.surrogate.world_model import train_world_model
    train_world_model(epochs=50, device=device)

run("5_surrogate", step_surrogate)

# ================================================================
# STEP 6: ONNX export
# ================================================================
def step_onnx():
    from rl.export_onnx import export_to_onnx
    export_to_onnx()

run("6_onnx", step_onnx)

# ================================================================
# STEP 7: Full benchmark — ALL agents with grade() scores
# ================================================================
def step_benchmark():
    from rl.gym_env import SupplyMindGymnasiumEnv, ACTION_TYPES
    from server.supply_environment import SupplyMindEnvironment
    from scripted_agent import choose_action as scripted_choose
    from rl.distributional.qr_dqn import QRDQNNetwork
    from rl.offline.baselines import BCNetwork, CQLQNetwork, TD3Actor
    from pathlib import Path

    TASKS = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    TASK_SHORT = {"easy_typhoon_response": "Easy", "medium_multi_front": "Medium", "hard_cascading_crisis": "Hard"}
    SEEDS = [42, 99, 7, 123, 256]
    N_EPS = 3  # per seed

    def eval_agent(agent_fn, task_id):
        grades, rewards = [], []
        for seed in SEEDS:
            for ep in range(N_EPS):
                env = SupplyMindGymnasiumEnv(task_id=task_id)
                env_core = SupplyMindEnvironment()
                obs, info = env.reset(seed=seed * 1000 + ep)
                env_core.reset(task_id=task_id, seed=seed * 1000 + ep)
                total_r = 0
                while True:
                    flat = agent_fn(obs, info)
                    act = np.array([flat // 40, flat % 40], dtype=np.int64)
                    obs, r, term, trunc, info = env.step(act)
                    sm = env._decode_action(act)
                    env_core.step(sm)
                    total_r += r
                    if term or trunc:
                        break
                grades.append(env_core.grade()["score"])
                rewards.append(total_r)
                env.close()
        return np.mean(grades), np.std(grades), np.mean(rewards)

    # Agent functions
    def random_agent(obs, info):
        valid = np.where(info["action_masks"])[0]
        return np.random.choice(valid) if len(valid) > 0 else 0

    def scripted_agent(obs, info):
        raw = info.get("raw_obs")
        if raw is None:
            return 0
        sm = scripted_choose(raw, 0)
        node_ids = [n.node_id for n in raw.node_statuses]
        at = ACTION_TYPES.index(sm.action_type)
        nd = node_ids.index(sm.target_node_id) if sm.target_node_id and sm.target_node_id in node_ids else 0
        return at * 40 + nd

    def make_qrdqn_agent(ckpt_name):
        p = Path(f"rl/checkpoints/{ckpt_name}")
        if not p.exists():
            return None
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        cfg = {k: v for k, v in ckpt["config"].items() if k in ("state_dim", "n_actions", "n_quantiles", "hidden_dim")}
        model = QRDQNNetwork(**cfg); model.load_state_dict(ckpt["state_dict"]); model.eval()
        def agent(obs, info):
            with torch.no_grad():
                st = torch.from_numpy(obs).float().unsqueeze(0)
                mk = torch.from_numpy(info["action_masks"]).bool().unsqueeze(0)
                return model.cvar_policy(st, alpha=0.5, action_mask=mk).item()
        return agent

    def make_offline_agent(ckpt_path, model_cls, key="state_dict"):
        p = Path(ckpt_path)
        if not p.exists():
            return None
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        model = model_cls(); model.load_state_dict(ckpt[key]); model.eval()
        def agent(obs, info):
            with torch.no_grad():
                logits = model(torch.from_numpy(obs).float().unsqueeze(0))
                logits[0][~torch.from_numpy(info["action_masks"]).bool()] = float("-inf")
                return logits.argmax(dim=-1).item()
        return agent

    # Build agent roster
    agents = {"Random": random_agent, "Scripted": scripted_agent}

    bc = make_offline_agent("rl/checkpoints/bc_best.pt", BCNetwork)
    if bc: agents["BC"] = bc

    cql = make_offline_agent("rl/checkpoints/cql_best.pt", CQLQNetwork, key="state_dict")
    if cql: agents["CQL"] = cql

    td3 = make_offline_agent("rl/checkpoints/td3bc_best.pt", TD3Actor, key="actor")
    if td3: agents["TD3+BC"] = td3

    iql = make_offline_agent("rl/checkpoints/iql_best.pt", BCNetwork, key="actor")
    if iql: agents["IQL"] = iql

    # Task-specific QR-DQN specialists
    for task_id, task_name, ckpt_name in [
        ("easy_typhoon_response", "easy", "autoresearch_final_easy.pt"),
        ("medium_multi_front", "medium", "autoresearch_final_medium.pt"),
        ("hard_cascading_crisis", "hard", "autoresearch_final_hard.pt"),
    ]:
        qr = make_qrdqn_agent(ckpt_name)
        if qr:
            agents[f"QR-DQN_{task_name}"] = qr

    # Also try generic QR-DQN
    qr_gen = make_qrdqn_agent("autoresearch_best_200k.pt") or make_qrdqn_agent("autoresearch_best.pt") or make_qrdqn_agent("qrdqn_best_easy.pt")
    if qr_gen:
        agents["QR-DQN (CVaR)"] = qr_gen

    all_results = []
    for agent_name, agent_fn in agents.items():
        for task_id in TASKS:
            # For specialists, only eval on their task
            if agent_name.startswith("QR-DQN_"):
                specialist_task = agent_name.split("_")[1]
                task_key = task_id.split("_")[0]
                if specialist_task != task_key:
                    continue

            gm, gs, rm = eval_agent(agent_fn, task_id)
            logger.info("  %s x %s: grade=%.3f+/-%.3f reward=%.3f",
                        agent_name, TASK_SHORT[task_id], gm, gs, rm)
            all_results.append({
                "agent": agent_name, "task": TASK_SHORT[task_id],
                "grade_mean": round(gm, 4), "grade_std": round(gs, 4),
                "reward_mean": round(rm, 4),
            })

    # Build specialist composite row
    spec_grades = {}
    for r in all_results:
        if r["agent"].startswith("QR-DQN_"):
            spec_grades[r["task"]] = r["grade_mean"]
    if spec_grades:
        avg = np.mean(list(spec_grades.values()))
        logger.info("  QR-DQN Specialist Composite: %s avg=%.3f",
                     json.dumps(spec_grades), avg)

    # Save
    os.makedirs("benchmark/results", exist_ok=True)
    with open("benchmark/results/final_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "task", "grade_mean", "grade_std", "reward_mean"])
        w.writeheader()
        w.writerows(all_results)

    logger.info("  Saved to benchmark/results/final_benchmark.csv")

run("7_benchmark", step_benchmark)

# ================================================================
# STEP 8: Statistics
# ================================================================
def step_stats():
    from benchmark.statistics import run_all_tests
    run_all_tests()

run("8_statistics", step_stats)

# ================================================================
# STEP 9: Backtesting
# ================================================================
def step_backtest():
    from benchmark.backtesting import run_backtesting
    run_backtesting(n_runs=50)

run("9_backtesting", step_backtest)

# ================================================================
# STEP 10: Explainer cache
# ================================================================
def step_explainer():
    from rl.explainer import pre_populate_cache
    pre_populate_cache(n_scenarios=50)

run("10_explainer", step_explainer)

# ================================================================
# STEP 11: Final pytest
# ================================================================
def step_pytest():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                       capture_output=True, text=True)
    last = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "?"
    logger.info("  Tests: %s", last)

run("11_pytest", step_pytest)

# ================================================================
# SUMMARY
# ================================================================
total = (time.time() - overall_start) / 60
logger.info("")
logger.info("=" * 70)
logger.info("IMPROVE EVERYTHING COMPLETE — %.1f min (%.1f hrs)", total, total / 60)
logger.info("=" * 70)

succeeded = sum(1 for v in results.values() if isinstance(v, (int, float)))
failed = sum(1 for v in results.values() if isinstance(v, str) and "FAILED" in v)
logger.info("Succeeded: %d | Failed: %d", succeeded, failed)

for name, val in results.items():
    if isinstance(val, (int, float)):
        logger.info("  %-30s %.1f min", name, val)
    else:
        logger.info("  %-30s %s", name, val)

logger.info("=" * 70)
