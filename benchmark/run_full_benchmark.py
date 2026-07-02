"""
Full benchmark: ALL trained agents, BOTH cumulative reward AND grade() score.

Evaluates: Random, BC, Scripted, IQL, CQL, TD3+BC, QR-DQN CVaR
On: All 3 tasks x 5 seeds x 5 episodes
Reports: Both cumulative_reward and grade_score (0-1 scale)
"""
import sys, os, time, csv, logging, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CKPT = ROOT / "rl" / "checkpoints"
RESULTS = ROOT / "benchmark" / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench")

from rl.gym_env import SupplyMindGymnasiumEnv, ACTION_TYPES
from server.supply_environment import SupplyMindEnvironment
from scripted_agent import choose_action as scripted_choose

TASKS = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
TASK_SHORT = {"easy_typhoon_response": "Easy", "medium_multi_front": "Medium", "hard_cascading_crisis": "Hard"}
SEEDS = [42, 99, 7, 123, 256]
N_EPS = 5

_MODEL_CACHE: dict = {}


def _cached_model(path, build):
    """Load + build a model once per checkpoint path; reuse across all steps."""
    key = str(path)
    if key not in _MODEL_CACHE:
        from rl.checkpoint_security import safe_load
        ckpt = safe_load(key, map_location="cpu")
        model = build(ckpt)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def eval_with_grade(agent_fn, task_id, seed):
    """Run agent and return BOTH cumulative reward and grade score."""
    env = SupplyMindEnvironment()
    gym_env = SupplyMindGymnasiumEnv(task_id=task_id)
    obs_gym, info = gym_env.reset(seed=seed)
    obs_core = env.reset(task_id=task_id, seed=seed)

    total_reward = 0.0
    step = 0

    while True:
        action_flat = agent_fn(obs_gym, info, obs_core, step)
        action_type_idx = action_flat // 40
        node_idx = action_flat % 40

        # Step gym env
        obs_gym, r, term, trunc, info = gym_env.step(np.array([action_type_idx, node_idx], dtype=np.int64))
        total_reward += r

        # Step core env with same action (for grade)
        node_ids = [n.node_id for n in obs_core.node_statuses]
        action_type = ACTION_TYPES[min(action_type_idx, 6)]
        target_node = node_ids[min(node_idx, len(node_ids) - 1)] if node_ids else None

        from supplymind.contracts import SupplyMindAction
        if action_type == "do_nothing":
            sm_action = SupplyMindAction(action_type="do_nothing")
        elif action_type == "activate_backup_supplier":
            node_status = None
            for n in obs_core.node_statuses:
                if n.node_id == target_node:
                    node_status = n
                    break
            if node_status and node_status.backup_supplier_ids:
                sm_action = SupplyMindAction(action_type=action_type, target_node_id=target_node,
                                              backup_supplier_id=node_status.backup_supplier_ids[0])
            else:
                sm_action = SupplyMindAction(action_type="do_nothing")
        elif action_type == "reroute_shipment":
            alt_ports = [n.node_id for n in obs_core.node_statuses
                        if n.node_type == "port" and n.is_operational and n.node_id != target_node]
            if alt_ports:
                sm_action = SupplyMindAction(action_type=action_type, target_node_id=target_node,
                                              reroute_via=[alt_ports[0]])
            else:
                sm_action = SupplyMindAction(action_type="do_nothing")
        elif action_type == "increase_safety_stock":
            sm_action = SupplyMindAction(action_type=action_type, target_node_id=target_node,
                                          additional_stock_days=10)
        elif action_type == "expedite_order":
            sm_action = SupplyMindAction(action_type=action_type, target_node_id=target_node,
                                          expedite_mode="air")
        elif action_type == "hedge_commodity":
            commodities = obs_core.financials.commodity_price_changes
            if commodities:
                commodity = max(commodities, key=commodities.get)
                hedge_amt = min(obs_core.financials.budget_remaining * 0.05, 500000)
                if hedge_amt > 0:
                    sm_action = SupplyMindAction(action_type=action_type, commodity=commodity,
                                                  hedge_amount_usd=hedge_amt)
                else:
                    sm_action = SupplyMindAction(action_type="do_nothing")
            else:
                sm_action = SupplyMindAction(action_type="do_nothing")
        elif action_type == "issue_supplier_alert":
            sm_action = SupplyMindAction(action_type=action_type, target_node_id=target_node)
        else:
            sm_action = SupplyMindAction(action_type="do_nothing")

        obs_core = env.step(sm_action)
        step += 1

        if term or trunc or obs_core.done:
            break

    grade = env.grade()
    gym_env.close()
    return total_reward, grade["score"]


# Agent functions — each returns a flat action index
def agent_random(obs, info, obs_core, step):
    mask = info["action_masks"]
    valid = np.where(mask)[0]
    return np.random.choice(valid) if len(valid) > 0 else 0

def agent_scripted(obs, info, obs_core, step):
    sm_action = scripted_choose(obs_core, step)
    action_type_idx = ACTION_TYPES.index(sm_action.action_type)
    node_ids = [n.node_id for n in obs_core.node_statuses]
    target_idx = 0
    if sm_action.target_node_id and sm_action.target_node_id in node_ids:
        target_idx = node_ids.index(sm_action.target_node_id)
    return action_type_idx * 40 + target_idx

def _build_qrdqn(ckpt):
    from rl.distributional.qr_dqn import QRDQNNetwork
    cfg = {k: v for k, v in ckpt["config"].items() if k in ("state_dim", "n_actions", "n_quantiles", "hidden_dim")}
    model = QRDQNNetwork(**cfg); model.load_state_dict(ckpt["state_dict"])
    return model

def agent_qrdqn(obs, info, obs_core, step):
    task_key = obs_core.info.get("task_id", "easy").split("_")[0] if hasattr(obs_core, "info") else "easy"
    p = CKPT / f"qrdqn_best_{task_key}.pt"
    if not p.exists():
        p = CKPT / "qrdqn_best_easy.pt"
    model = _cached_model(p, _build_qrdqn)
    with torch.no_grad():
        st = torch.from_numpy(obs).float().unsqueeze(0)
        mask = torch.from_numpy(info["action_masks"]).bool().unsqueeze(0)
        return model.cvar_policy(st, alpha=0.1, action_mask=mask).item()

def _build_bc(ckpt):
    from rl.offline.baselines import BCNetwork
    model = BCNetwork(); model.load_state_dict(ckpt["state_dict"])
    return model

def agent_bc(obs, info, obs_core, step):
    model = _cached_model(CKPT / "bc_best.pt", _build_bc)
    with torch.no_grad():
        logits = model(torch.from_numpy(obs).float().unsqueeze(0))
        logits[0][~torch.from_numpy(info["action_masks"]).bool()] = float("-inf")
        return logits.argmax(dim=-1).item()

def _build_iql(ckpt):
    from rl.offline.baselines import BCNetwork
    model = BCNetwork(); model.load_state_dict(ckpt["actor"])
    return model

def agent_iql(obs, info, obs_core, step):
    model = _cached_model(CKPT / "iql_best.pt", _build_iql)
    with torch.no_grad():
        logits = model(torch.from_numpy(obs).float().unsqueeze(0))
        logits[0][~torch.from_numpy(info["action_masks"]).bool()] = float("-inf")
        return logits.argmax(dim=-1).item()

def _build_cql(ckpt):
    from rl.offline.baselines import CQLQNetwork
    model = CQLQNetwork(); model.load_state_dict(ckpt["state_dict"])
    return model

def agent_cql(obs, info, obs_core, step):
    model = _cached_model(CKPT / "cql_best.pt", _build_cql)
    with torch.no_grad():
        q = model.q_min(torch.from_numpy(obs).float().unsqueeze(0))
        q[0][~torch.from_numpy(info["action_masks"]).bool()] = float("-inf")
        return q.argmax(dim=-1).item()

def _build_td3bc(ckpt):
    from rl.offline.baselines import TD3Actor
    model = TD3Actor(); model.load_state_dict(ckpt["actor"])
    return model

def agent_td3bc(obs, info, obs_core, step):
    model = _cached_model(CKPT / "td3bc_best.pt", _build_td3bc)
    with torch.no_grad():
        logits = model(torch.from_numpy(obs).float().unsqueeze(0))
        logits[0][~torch.from_numpy(info["action_masks"]).bool()] = float("-inf")
        return logits.argmax(dim=-1).item()


AGENTS = {
    "Random": agent_random,
    "BC": agent_bc,
    "TD3+BC": agent_td3bc,
    "CQL": agent_cql,
    "Scripted": agent_scripted,
    "IQL": agent_iql,
    "QR-DQN (CVaR)": agent_qrdqn,
}

results = []
logger.info("FULL BENCHMARK: %d agents x %d tasks x %d seeds x %d eps", len(AGENTS), len(TASKS), len(SEEDS), N_EPS)

for agent_name, agent_fn in AGENTS.items():
    for task_id in TASKS:
        rewards = []
        grades = []
        for seed in SEEDS:
            for ep in range(N_EPS):
                try:
                    rew, grade = eval_with_grade(agent_fn, task_id, seed * 1000 + ep)
                    rewards.append(rew)
                    grades.append(grade)
                    results.append({
                        "agent": agent_name, "task": TASK_SHORT[task_id],
                        "task_id": task_id, "seed": seed,
                        "cumulative_reward": round(rew, 4),
                        "grade_score": round(grade, 4),
                    })
                except Exception as e:
                    logger.warning("  %s x %s ep %d failed: %s", agent_name, TASK_SHORT[task_id], ep, str(e)[:60])

        if rewards:
            logger.info("  %s x %s: reward=%.3f+/-%.3f  grade=%.3f+/-%.3f (n=%d)",
                        agent_name, TASK_SHORT[task_id],
                        np.mean(rewards), np.std(rewards),
                        np.mean(grades), np.std(grades), len(rewards))

# Save detailed results
RESULTS.mkdir(parents=True, exist_ok=True)
with open(RESULTS / "full_benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["agent", "task", "task_id", "seed", "cumulative_reward", "grade_score"])
    w.writeheader()
    w.writerows(results)

# Save summary
with open(RESULTS / "full_benchmark_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Agent", "Easy (grade)", "Medium (grade)", "Hard (grade)", "Avg (grade)", "Easy (reward)", "Medium (reward)", "Hard (reward)", "Avg (reward)"])
    for agent_name in AGENTS:
        row = [agent_name]
        grade_avgs = []
        rew_avgs = []
        for task_id in TASKS:
            task_results = [r for r in results if r["agent"] == agent_name and r["task_id"] == task_id]
            if task_results:
                gm = np.mean([r["grade_score"] for r in task_results])
                gs = np.std([r["grade_score"] for r in task_results])
                rm = np.mean([r["cumulative_reward"] for r in task_results])
                row.append(f"{gm:.3f}+/-{gs:.3f}")
                grade_avgs.append(gm)
                rew_avgs.append(rm)
            else:
                row.append("N/A")
        row.append(f"{np.mean(grade_avgs):.3f}" if grade_avgs else "N/A")
        for task_id in TASKS:
            task_results = [r for r in results if r["agent"] == agent_name and r["task_id"] == task_id]
            if task_results:
                rm = np.mean([r["cumulative_reward"] for r in task_results])
                rs = np.std([r["cumulative_reward"] for r in task_results])
                row.append(f"{rm:.3f}+/-{rs:.3f}")
            else:
                row.append("N/A")
        row.append(f"{np.mean(rew_avgs):.3f}" if rew_avgs else "N/A")
        w.writerow(row)

logger.info("Results saved to benchmark/results/full_benchmark.csv")
logger.info("Summary saved to benchmark/results/full_benchmark_summary.csv")
logger.info("BENCHMARK COMPLETE")
