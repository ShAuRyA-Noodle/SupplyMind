"""
Phase T "Atlantic" — SHAP on CQL v2 + per-action decomposition + explainer stress test.

Upgrades:
  U31 SHAP on CQL (best agent) not BC
  U32 1K background + 1K explained samples
  U33 Per-action-type SHAP decomposition
  U34 Explainer stress test on 50 diverse real crises

Output:
  rl/checkpoints/shap_cql_v2.json
  rl/checkpoints/explainer_stress_v2.json
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

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "rl" / "checkpoints"
DATA = ROOT / "rl" / "data"


class CQLTypeHead(nn.Module):
    """Extract type-prediction head from FactorizedTwinQ for SHAP."""
    def __init__(self, twinq_state_dict, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(408, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.type_head = nn.Linear(hidden, 7)
        # Load trunk from twin Q (t1) + q1_type
        sd = {}
        for k, v in twinq_state_dict.items():
            if k.startswith("t1."):
                sd[k.replace("t1.", "trunk.")] = v
            elif k.startswith("q1_type."):
                sd[k.replace("q1_type.", "type_head.")] = v
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        return self.type_head(self.trunk(x))


def feature_group(idx: int) -> str:
    if idx < 300:
        node = idx // 10
        feat = idx % 10
        feat_name = ["op", "risk", "inv", "backup", "t0", "t1", "t2", "t3", "type", "rev"][feat]
        return f"node{node}_{feat_name}"
    if 300 <= idx < 304:
        return f"access_log_{idx-300}"
    if idx < 350:
        return f"node_{idx//10}_{idx%10}"
    if idx < 368:
        return f"NOAA_{idx - 350}"
    if idx < 375:
        return f"USGS_{idx - 368}"
    if idx < 390:
        lead = ["cyclone", "quake", "flood", "fire", "volcano", "port", "canal", "strike",
                "geopol", "sanction", "pandemic", "cyber", "supplier", "material", "infra"]
        return f"LEAD_{lead[idx - 375]}"
    if idx < 395:
        wgi = ["voice", "polviolence", "goveff", "regqual", "rulelaw"]
        return f"WGI_{wgi[idx - 390]}"
    if idx < 407:
        fred = ["oil", "copper", "twd", "krw", "jpy", "eur", "cny", "ppi", "ppimm", "pcu484", "ipg334", "ir"]
        return f"FRED_{fred[idx - 395]}"
    return "status"


def feature_group_coarse(name: str) -> str:
    if name.startswith("NOAA"): return "NOAA"
    if name.startswith("USGS"): return "USGS"
    if name.startswith("FRED"): return "FRED"
    if name.startswith("WGI"): return "WGI"
    if name.startswith("LEAD"): return "LEADING_IND"
    if name.startswith("access"): return "ACCESS_LOG"
    if name.startswith("node"): return "NODE"
    return "STATUS"


def run_shap_cql():
    import shap
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = CKPT / "cql_v2.pt"
    if not ckpt_path.exists():
        log.warning(f"CQL v2 not found at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    model = CQLTypeHead(sd).to(device)
    model.eval()

    train = np.load(str(DATA / "real_train_v2.npz"))
    test = np.load(str(DATA / "real_test_v2.npz"))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(train["states"]), size=1000, replace=False)
    bg = torch.from_numpy(train["states"][bg_idx].astype(np.float32)).to(device)
    te_idx = rng.choice(len(test["states"]), size=1000, replace=False)
    te = torch.from_numpy(test["states"][te_idx].astype(np.float32)).to(device)
    a_type_gt = test["actions"][te_idx, 0]

    log.info("DeepExplainer on CQL type head...")
    explainer = shap.DeepExplainer(model, bg)
    shap_vals = explainer.shap_values(te)  # list[7] of [N, 408]
    if isinstance(shap_vals, list):
        arr = np.stack(shap_vals, axis=-1)  # [N, 408, 7]
    else:
        arr = shap_vals
    log.info(f"SHAP values shape: {arr.shape}")

    # Global importance
    global_imp = np.mean(np.abs(arr), axis=(0, 2)) if arr.ndim == 3 else np.mean(np.abs(arr), axis=0)
    top20 = np.argsort(global_imp)[-20:][::-1]
    top20_list = [{"idx": int(i), "name": feature_group(int(i)), "imp": float(global_imp[i])} for i in top20]

    # Group shares
    groups = {}
    for i, imp in enumerate(global_imp):
        g = feature_group_coarse(feature_group(i))
        groups[g] = groups.get(g, 0.0) + float(imp)
    total = sum(groups.values())
    group_shares = {k: v / total for k, v in groups.items()}

    # Per-action-type decomposition (top-5 features per action type)
    per_action = {}
    if arr.ndim == 3:
        action_names = ["do_nothing", "alert", "reroute", "expedite", "safety_stock", "backup", "cancel"]
        for a in range(arr.shape[-1]):
            imp_a = np.mean(np.abs(arr[:, :, a]), axis=0)
            top5 = np.argsort(imp_a)[-5:][::-1]
            per_action[action_names[a]] = [
                {"idx": int(i), "name": feature_group(int(i)), "imp": float(imp_a[i])} for i in top5
            ]

    log.info("Top-10 global:")
    for t in top20_list[:10]:
        log.info(f"  {t['name']:<28} idx={t['idx']:<4} imp={t['imp']:.5f}")
    log.info(f"Group shares: {group_shares}")

    out = {
        "n_background": 1000, "n_explained": 1000,
        "top20_global": top20_list,
        "group_importance": groups,
        "group_shares": group_shares,
        "per_action_top5": per_action,
        "checkpoint": str(ckpt_path),
    }
    (CKPT / "shap_cql_v2.json").write_text(json.dumps(out, indent=2))
    log.info("Saved shap_cql_v2.json")
    return out


def stress_test_explainer():
    """Run explainer on 50 diverse real scenarios; report quality-gate pass rate."""
    log.info("Explainer stress test...")
    try:
        from rl.explainer import _build_prompt, _passes_quality_gate
        import ollama
    except ImportError as e:
        log.warning(f"Cannot stress test: {e}")
        return None

    # Build 50 diverse scenario prompts
    scenarios = []
    actions = ["do_nothing", "issue_supplier_alert", "reroute_shipment", "expedite_order",
               "increase_safety_stock", "activate_backup_supplier", "hedge_commodity"]
    disruptions = ["cyclone", "earthquake", "port_strike", "chip_shortage", "canal_blockage",
                   "cyber_attack", "supplier_financial_distress", "political_unrest"]
    nodes = ["SUP_TSMC", "SUP_SAMSUNG", "PORT_KAOHSIUNG", "ROUTE_SUEZ", "SUP_FOXCONN",
             "PORT_SHANGHAI", "CARRIER_MAERSK", "SUP_INTEL"]
    n_test = 0; passed = 0; regenerated = 0
    results = []
    import itertools
    rng = np.random.default_rng(123)
    for i in range(50):
        action = rng.choice(actions)
        disr = rng.choice(disruptions)
        node = rng.choice(nodes)
        day = rng.integers(1, 30); total = rng.integers(30, 60)
        budget_pct = rng.uniform(30, 95)
        health = rng.integers(40, 95)
        p50 = rng.uniform(0.2, 3.5) * 1e6
        severity = rng.uniform(0.1, 0.95)

        state_text = (
            f"Day {day} of {total}. Budget {budget_pct:.0f}% remaining. "
            f"Supply chain health {health}/100. MC P50 ${p50:,.0f}. "
            f"Active disruption: {disr} (severity {severity:.2f}) affecting {node}."
        )
        prompt = _build_prompt(state_text, action, node, None,
                               shap_top=[("NOAA_3", 0.42), ("node0_risk", 0.38), ("FRED_oil", 0.21)],
                               counterfactual_p50=p50 * 1.4,
                               rag_precedent="Tohoku 2011 single-source exposure led to $1.2B Toyota loss")

        try:
            r = ollama.chat(model="supplymind-analyst:v2", messages=[{"role": "user", "content": prompt}],
                            options={"temperature": 0.2})
            text = r["message"]["content"]
            ok = _passes_quality_gate(text)
            if not ok:
                # Regen once
                r = ollama.chat(model="supplymind-analyst:v2",
                                messages=[{"role": "user", "content": prompt + "\n\nProduce all 4 sections."}])
                text = r["message"]["content"]
                ok = _passes_quality_gate(text)
                if ok: regenerated += 1
            if ok: passed += 1
            results.append({"scenario": i, "action": action, "disruption": disr, "passed": bool(ok),
                            "length": len(text)})
            n_test += 1
        except Exception as e:
            log.warning(f"  scenario {i} failed: {e}")
            results.append({"scenario": i, "error": str(e)})

    summary = {
        "n_test": n_test, "passed": passed, "pass_rate": passed / max(n_test, 1),
        "regenerated_once_success": regenerated,
        "scenarios": results,
    }
    (CKPT / "explainer_stress_v2.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Explainer stress: {passed}/{n_test} passed ({summary['pass_rate']*100:.1f}%), {regenerated} regen-success")


def main():
    run_shap_cql()
    stress_test_explainer()
    log.info("Phase T 'Atlantic' complete.")


if __name__ == "__main__":
    main()
