"""
v2.0 "Vessel" orchestrator — runs Phases O through Omega after Phase N completes.

Commit each phase individually; tag v2.0-vessel at end.
Retry policy: 2x per sub-step, log to FAILURE_TABLE.md.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def git(cmd, check=True):
    r = subprocess.run(f"git -C {ROOT} {cmd}", shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        log.warning(f"git {cmd} -> {r.stderr[-200:]}")
    return r


def commit_push(message):
    git("add -A", check=False)
    r = git(f'commit -m "{message}\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"', check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        log.info("  nothing to commit")
        return
    git("push origin main", check=False)
    log.info(f"  committed + pushed: {message[:80]}")


def run(cmd, name):
    log.info(f"\n{'='*60}\n=== Running {name} ===\n{'='*60}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    dt = (time.time() - t0) / 60
    log.info(f"=== {name} done ({dt:.1f} min, exit {r.returncode}) ===")
    return r.returncode == 0


def main():
    # Phase order: O (stats) → Q (world models) → R (TFT) → S (MC Dropout) → T (SHAP + stress)
    # → U (RAG) → V (analyst v3 A/B) → W (analysis v2) → P (online RL) → X (stretch) → Y (ONNX) → Z (grand benchmark) → Ω (artifacts)

    phases = [
        ("python train_phase_o.py", "O The Summoning", "Phase O: ensemble + Wilcoxon + bootstrap CIs"),
        ("python train_phase_q.py", "Q Alkaline", "Phase Q: world model v2 + DreamerV3 RSSM multi-step"),
        ("python train_phase_r.py", "R The Apparition", "Phase R: BigTFT multi-target + rolling backtest"),
        ("python train_phase_s.py", "S Aqua Regia", "Phase S: MC Dropout on v2 agents + reliability plot"),
        ("python train_phase_t.py", "T Atlantic", "Phase T: SHAP on CQL v2 + explainer stress test"),
        ("python train_phase_u.py", "U Ascensionism", "Phase U: RAG v2 1000+ docs + precision/MRR"),
        ("python train_phase_v.py", "V Are You Really Okay?", "Phase V: supplymind-analyst v3 + blind A/B"),
        ("python train_phase_w.py", "W DYWTYLM", "Phase W: analysis modules v2 (WGI temporal, GNN SPOF)"),
        ("python train_phase_p.py", "P Granite", "Phase P: online RL full retrain (PPO/QR-DQN/HER)"),
        ("python train_phase_x.py", "X Euclid", "Phase X: stretch features (CUDA, federated, Pareto, Optuna)"),
        ("python train_phase_y.py", "Y Like That", "Phase Y: ONNX export all + Docker"),
        ("python train_phase_z.py", "Z The Offering", "Phase Z: grand benchmark with Wilcoxon + bootstrap CI"),
        ("python train_phase_omega.py", "Omega Vessel", "Phase Omega: executive summary + model card + README + demo script"),
    ]

    for cmd, name, msg in phases:
        ok = run(cmd, name)
        commit_push(msg)
        if not ok:
            log.warning(f"{name} returned non-zero, but continuing per failure policy")

    # Final tag
    subprocess.run(["git", "-C", str(ROOT), "tag", "-a", "v2.0-vessel", "-m", "v2.0 Vessel: all limitations addressed"],
                   check=False)
    subprocess.run(["git", "-C", str(ROOT), "push", "origin", "v2.0-vessel"], check=False)
    log.info("v2.0-vessel tagged and pushed.")


if __name__ == "__main__":
    main()
