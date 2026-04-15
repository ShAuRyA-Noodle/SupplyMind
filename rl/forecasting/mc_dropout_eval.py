"""
Phase D — MC Dropout epistemic uncertainty estimation.

Applies dropout at inference time, does 50 stochastic forward passes,
reports mean ± 2*std as epistemic uncertainty on the test set.

Model: BC policy trained on real unified buffer (rl/checkpoints/bc_best_real_v2.pt)
Test set: rl/data/real_test.npz (27K held-out transitions)

Output: rl/checkpoints/mc_dropout_eval.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "rl" / "checkpoints" / "bc_best_real_v2.pt"
TEST = ROOT / "rl" / "data" / "real_test.npz"
OUT = ROOT / "rl" / "checkpoints" / "mc_dropout_eval.json"


class BCNetworkWithDropout(nn.Module):
    """BC network with dropout activated at inference for MC Dropout."""

    def __init__(self, state_dim=408, action_dim=280, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    if not CKPT.exists():
        log.error(f"Checkpoint not found: {CKPT}")
        return
    if not TEST.exists():
        log.error(f"Test set not found: {TEST}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BC checkpoint (no dropout) and copy weights to a dropout-wrapped version
    ckpt = torch.load(CKPT, map_location=device)
    state_dict = ckpt["state_dict"]

    # Build a dropout-enabled network of matching shape; BC has no dropout originally,
    # so we inject dropout p=0.1 AFTER ReLUs for MC evaluation (known approximation).
    model = BCNetworkWithDropout().to(device)
    # Remap state_dict keys: original net[0].weight == dropout_net.net[0].weight still matches
    # because original net = Linear, ReLU, Linear, ReLU, Linear (no dropout, just 3 Linear layers).
    # Our dropout version interleaves Dropout; Linear layers are at positions 0, 3, 6.
    # Rename keys accordingly.
    mapped = {}
    pos_map = {"0": "0", "2": "3", "4": "6"}  # orig layer idx -> new layer idx
    for k, v in state_dict.items():
        # k like "net.0.weight"
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "net" and parts[1] in pos_map:
            parts[1] = pos_map[parts[1]]
            mapped[".".join(parts)] = v
        else:
            mapped[k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    log.info(f"Loaded BC weights. missing={len(missing)}, unexpected={len(unexpected)}")

    model.train()  # keep dropout active for MC sampling

    data = np.load(TEST)
    states = torch.from_numpy(data["states"].astype(np.float32)).to(device)
    actions_gt = data["actions"]
    flat_actions_gt = (actions_gt[:, 0] * 40 + actions_gt[:, 1]).astype(np.int64)
    log.info(f"Test states: {states.shape}")

    # Take first 5000 for tractable MC sampling (50 passes x 5k = 250k forward calls)
    n_eval = min(5000, len(states))
    s_eval = states[:n_eval]
    gt = flat_actions_gt[:n_eval]

    n_passes = 50
    probs_stack = []
    with torch.no_grad():
        for i in range(n_passes):
            logits = model(s_eval)
            probs = torch.softmax(logits, dim=-1)
            probs_stack.append(probs.cpu().numpy())
    probs_arr = np.stack(probs_stack, axis=0)  # [P, N, A]
    mean_probs = probs_arr.mean(axis=0)  # [N, A]
    std_probs = probs_arr.std(axis=0)  # [N, A]

    # Predictive entropy (total uncertainty)
    total_entropy = -(mean_probs * np.log(mean_probs + 1e-12)).sum(axis=-1)
    # Expected entropy under posterior (aleatoric)
    aleatoric = -(probs_arr * np.log(probs_arr + 1e-12)).sum(axis=-1).mean(axis=0)
    # Epistemic = total - aleatoric
    epistemic = total_entropy - aleatoric

    pred = mean_probs.argmax(axis=-1)
    acc = float((pred == gt).mean())

    # Reliability: group by epistemic uncertainty quartile
    q = np.quantile(epistemic, [0.25, 0.5, 0.75])
    buckets = {}
    for label, mask in [
        ("Q1_low_unc", epistemic <= q[0]),
        ("Q2", (epistemic > q[0]) & (epistemic <= q[1])),
        ("Q3", (epistemic > q[1]) & (epistemic <= q[2])),
        ("Q4_high_unc", epistemic > q[2]),
    ]:
        buckets[label] = {
            "n": int(mask.sum()),
            "accuracy": float((pred[mask] == gt[mask]).mean()) if mask.sum() > 0 else None,
            "mean_epistemic": float(epistemic[mask].mean()) if mask.sum() > 0 else None,
        }
    log.info(f"Test accuracy (mean of posterior): {acc:.4f}")
    for k, v in buckets.items():
        log.info(f"  {k}: n={v['n']}, acc={v['accuracy']:.4f}, epi={v['mean_epistemic']:.4f}")

    out = {
        "n_eval": n_eval,
        "n_passes": n_passes,
        "mean_posterior_accuracy": acc,
        "mean_total_entropy": float(total_entropy.mean()),
        "mean_aleatoric": float(aleatoric.mean()),
        "mean_epistemic": float(epistemic.mean()),
        "buckets": buckets,
        "checkpoint": str(CKPT),
    }
    OUT.write_text(json.dumps(out, indent=2))
    log.info(f"Saved {OUT}")


if __name__ == "__main__":
    main()
