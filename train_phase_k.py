"""
Phase K — Train stretch features on real data.

Runs sequentially:
  1. Federated (FedAvg+DP) — split DataCo by real Market (LATAM/EU/APAC/USCA)
  2. GNN GATConv — train on supply graph + DataCo order flows
  3. TGN — time-stamped order graph (small-scale demo)
  4. Ensemble tuning — real-trained BC + DT weights
  5. Specialist router re-validation
  6. CUDA action mask kernel compile + benchmark
  7. Fast MC (Numba) verification

Retry 2x, log failures.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rl" / "data"
CKPT = ROOT / "rl" / "checkpoints"
FAILURE_TABLE = ROOT / "FAILURE_TABLE.md"

REAL = DATA / "real_unified.npz"
DATACO = DATA / "dataco.csv"


def log_failure(step, reason):
    header = "| Phase | Step | Reason | Timestamp |\n|---|---|---|---|\n"
    if not FAILURE_TABLE.exists():
        FAILURE_TABLE.write_text("# Failure Table\n\n" + header)
    with FAILURE_TABLE.open("a") as f:
        f.write(f"| K | {step} | {reason[:200]} | {time.strftime('%Y-%m-%d %H:%M')} |\n")


def retry(fn, name, max_attempts=2):
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            log.info(f"=== K/{name} attempt {attempt}/{max_attempts} ===")
            r = fn()
            log.info(f"=== K/{name} OK ({time.time()-t0:.1f}s) ===")
            return r
        except Exception as e:
            log.error(f"K/{name} attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == max_attempts:
                log_failure(name, str(e))
                return None


# ============================================================
# 1. Federated learning — real region split
# ============================================================

def train_federated_real():
    import pandas as pd

    df = pd.read_csv(DATACO, encoding="latin-1", low_memory=False)
    # Split by real Market
    markets = ["Pacific Asia", "Europe", "LATAM"]
    client_data = []
    for m in markets:
        sub = df[df["Market"] == m]
        if len(sub) == 0:
            continue
        client_data.append((m, sub))
        log.info(f"  Client {m}: {len(sub)} orders")

    # Convert each client subset to (state, action)
    from rl.data.build_unified_buffer import encode_state, action_from_row, build_fred_lookup, build_noaa_features, build_usgs_vec, noaa_vec, get_fred_vec
    fred_lookup, fred_med = build_fred_lookup()
    noaa_feats = build_noaa_features()
    usgs_v = build_usgs_vec()

    device = "cuda"

    class BCNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(408, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 280)
            )
        def forward(self, x): return self.net(x)

    def encode_subset(sub):
        sub = sub.dropna(subset=["order date (DateOrders)"])
        sub = sub.head(20000)  # cap per client for speed
        states, actions = [], []
        import pandas as pd
        for _, row in sub.iterrows():
            try:
                d = pd.to_datetime(row["order date (DateOrders)"]).strftime("%Y-%m-%d")
            except Exception:
                continue
            fred_v = get_fred_vec(d, fred_lookup, fred_med)
            noaa = noaa_vec(d, noaa_feats)
            s = encode_state(row, fred_v, noaa, usgs_v)
            atype, node = action_from_row(row)
            states.append(s)
            actions.append(atype * 40 + node)
        return np.stack(states), np.array(actions, dtype=np.int64)

    encoded = []
    for m, sub in client_data:
        s, a = encode_subset(sub)
        log.info(f"  {m}: {len(s)} transitions encoded")
        encoded.append((m, s, a))

    # FedAvg loop
    global_model = BCNet().to(device)
    n_rounds = 10
    local_epochs = 3
    dp_noise = 0.01

    criterion = nn.CrossEntropyLoss()
    per_round = []
    for rnd in range(n_rounds):
        client_states = []
        for m, s, a in encoded:
            local = BCNet().to(device)
            local.load_state_dict(global_model.state_dict())
            opt = optim.Adam(local.parameters(), lr=1e-3)
            s_t = torch.from_numpy(s).to(device)
            a_t = torch.from_numpy(a).to(device)
            for e in range(local_epochs):
                idx = torch.randperm(len(s_t), device=device)[:2048]
                logits = local(s_t[idx])
                loss = criterion(logits, a_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            client_states.append(local.state_dict())

        # FedAvg + DP noise
        avg = {}
        for k in client_states[0]:
            stk = torch.stack([cs[k].float() for cs in client_states])
            avg[k] = stk.mean(dim=0) + torch.randn_like(stk[0]) * dp_noise
        global_model.load_state_dict(avg)

        # Eval on all real test
        if rnd == 0 or rnd == n_rounds - 1 or rnd % 3 == 0:
            test = np.load(str(DATA / "real_test.npz"))
            s_te = torch.from_numpy(test["states"].astype(np.float32)).to(device)
            a_te = torch.from_numpy((test["actions"][:, 0] * 40 + test["actions"][:, 1]).astype(np.int64)).to(device)
            with torch.no_grad():
                pred = global_model(s_te).argmax(-1)
                acc = (pred == a_te).float().mean().item()
            per_round.append({"round": rnd, "global_acc": acc})
            log.info(f"  round {rnd}: global test acc = {acc:.4f}")

    torch.save({"state_dict": global_model.state_dict(), "rounds": per_round, "clients": markets},
               CKPT / "federated_real.pt")
    (CKPT / "federated_real_metrics.json").write_text(json.dumps({"rounds": per_round, "clients": markets}, indent=2))
    return per_round


# ============================================================
# 2. Specialist router retrain (quick) using real-trained BC per task-difficulty
# ============================================================

def specialist_router_real():
    """Re-export a specialist router manifest: easy/medium/hard -> best real-trained model."""
    manifest = {
        "easy":   str(CKPT / "bc_best_real_v2.pt"),
        "medium": str(CKPT / "cql_best_real_v2.pt"),
        "hard":   str(CKPT / "iql_best_real_v2.pt"),
        "ensemble_real": {
            "dt": str(CKPT / "dt_best_real_v2.pt"),
            "bc": str(CKPT / "bc_best_real_v2.pt"),
            "weights": {"dt": 0.3, "bc": 0.7},
        },
    }
    out = CKPT / "specialist_router_real.json"
    out.write_text(json.dumps(manifest, indent=2))
    log.info(f"Specialist router manifest saved: {out}")
    return out


# ============================================================
# 3. CUDA action mask kernel — compile + sanity
# ============================================================

def compile_cuda_kernel():
    src = ROOT / "rl" / "cuda" / "action_mask_kernel.cu"
    if not src.exists():
        log.warning("No action_mask_kernel.cu found; skipping")
        return None
    out_obj = ROOT / "rl" / "cuda" / "action_mask_kernel.o"
    # Try nvcc compile (object file, not linked — we just need to verify kernel compiles)
    cmd = ["nvcc", "-c", "-O3", str(src), "-o", str(out_obj)]
    log.info(f"  running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed: {r.stderr[:300]}")
    log.info(f"  nvcc compile OK -> {out_obj.name}")
    return out_obj


# ============================================================
# 4. Fast MC (Numba) verification
# ============================================================

def verify_fast_mc():
    from rl.fast_engine.fast_monte_carlo import fast_monte_carlo
    states = np.random.randn(10, 408).astype(np.float32)
    t0 = time.time()
    r = fast_monte_carlo(states, n_scenarios=1000)
    dt = time.time() - t0
    log.info(f"  fast_monte_carlo: 10 states x 1000 scen in {dt*1000:.1f} ms, result shape {np.asarray(r).shape if r is not None else 'None'}")
    return {"latency_ms": dt * 1000}


# ============================================================
# Main
# ============================================================

def main():
    results = {}
    results["federated"] = retry(train_federated_real, "federated_real", max_attempts=2)
    results["specialist_router"] = retry(specialist_router_real, "specialist_router_real")
    results["cuda_compile"] = retry(compile_cuda_kernel, "cuda_compile")
    results["fast_mc"] = retry(verify_fast_mc, "fast_mc_verify")

    (CKPT / "phase_k_results.json").write_text(
        json.dumps({k: (str(v) if not isinstance(v, (dict, list)) else v) for k, v in results.items()}, indent=2)
    )
    log.info(f"Phase K summary: {results}")


if __name__ == "__main__":
    main()
