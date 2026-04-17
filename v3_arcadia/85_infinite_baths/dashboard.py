"""R6 Block 9 — Infinite Baths: Streamlit dashboard aggregating all v3 Arcadia results.

Run:
  streamlit run v3_arcadia/85_infinite_baths/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SupplyMind v3 Arcadia", layout="wide", page_icon="🛡")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "v3_arcadia" / "results"

st.title("SupplyMind v3 Arcadia — Executive Dashboard")
st.caption("Meta PyTorch OpenEnv Hackathon submission. Full SOTA stack: 13 foundation models, 6 benchmarks, "
           "real production API.")

# ============================================================
# Sidebar: phase selector
# ============================================================
phases = {
    "R1 Emergence (model verification)": "R1_VERIFIED.json",
    "R2 Caramel (tabular)": "R2_CARAMEL.json",
    "R2 Benefit regression fix": "R2_BENEFIT_FIX.json",
    "R2 SHAP + Fairness + Calibration": "R2_SHAP_FAIRNESS_CALIBRATION.json",
    "R3 Past Self (forecasting)": "R3_PAST_SELF.json",
    "R4 Dangerous V1": "R4_DANGEROUS.json",
    "R4 Dangerous V2 BEAST": "R4_DANGEROUS_V2.json",
    "R5 Granite (RAG)": "R5_GRANITE.json",
    "R6 Gethsemane (RL)": "R6_GETHSEMANE.json",
    "R6 Euclidian (10,800-ep benchmark)": "R6_EUCLIDIAN.json",
    "R6 Provider (GNN)": "R6_PROVIDER.json",
    "R6 Aqua Regia (Conformal)": "R6_AQUA_REGIA.json",
}
phase = st.sidebar.selectbox("Phase", list(phases.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick stats")
if (RESULTS / "R4_DANGEROUS_V2.json").exists():
    d = json.loads((RESULTS / "R4_DANGEROUS_V2.json").read_text())
    st.sidebar.metric("Risk-panel α (Krippendorff)", f"{d['agreement']['krippendorff_alpha_ordinal']:.3f}")
    st.sidebar.metric("Majority-vote GT accuracy", f"{d['accuracy_vs_ground_truth']['majority_vote']['accuracy']:.1%}")
if (RESULTS / "R5_GRANITE.json").exists():
    d = json.loads((RESULTS / "R5_GRANITE.json").read_text())
    best = max(d['pipelines'].items(), key=lambda x: x[1]['mrr'])
    st.sidebar.metric("Best RAG P@1", f"{best[1]['p1']:.3f}", help=best[0])

# ============================================================
# Main panel
# ============================================================
fname = phases[phase]
fpath = RESULTS / fname
if not fpath.exists():
    st.warning(f"Not yet generated: {fname}")
else:
    st.subheader(phase)
    data = json.loads(fpath.read_text())

    if "R4_DANGEROUS_V2" in fname:
        col1, col2, col3 = st.columns(3)
        col1.metric("Scenarios", data["n_scenarios"])
        col2.metric("Krippendorff α", f"{data['agreement']['krippendorff_alpha_ordinal']:.3f}")
        col3.metric("Majority-vote accuracy", f"{data['accuracy_vs_ground_truth']['majority_vote']['accuracy']:.1%}")

        st.markdown("### Per-judge accuracy vs ground truth")
        rows = []
        for j, a in data["accuracy_vs_ground_truth"].items():
            rows.append({"Judge": j, "Correct": a["correct"], "Total": a["total"],
                         "Accuracy": a["accuracy"]})
        st.dataframe(pd.DataFrame(rows).set_index("Judge"))

        st.markdown("### Escalation distribution")
        esc_df = pd.DataFrame([{"Tier": k, "Count": v} for k, v in data["escalation_distribution"].items()])
        st.bar_chart(esc_df.set_index("Tier"))

    elif "R5_GRANITE" in fname:
        st.metric("Corpus", f"{data['n_chunks']} chunks / 48 docs")
        st.metric("Queries", data["n_queries"])
        rows = []
        for p, m in sorted(data["pipelines"].items(), key=lambda x: -x[1]["mrr"]):
            rows.append({"Pipeline": p, "P@1": m["p1"], "P@3": m["p3"], "P@5": m["p5"],
                         "MRR": m["mrr"], "nDCG@10": m["ndcg10"], "Latency (s)": m["latency_s"]})
        st.dataframe(pd.DataFrame(rows).set_index("Pipeline").round(3))

    elif "R3_PAST_SELF" in fname:
        st.markdown("### Forecasting results (20-fold backtest)")
        rows = []
        for tgt, tr in data["per_target"].items():
            for h in ["h7", "h14", "h28"]:
                if h not in tr: continue
                agg = tr[h].get("backtest_agg", {})
                for m, v in agg.items():
                    rows.append({"Target": tgt, "Horizon": h, "Model": m,
                                 "MAE": v.get("mean_mae"), "DirAcc": v.get("mean_dir_acc"),
                                 "PICP80": v.get("mean_picp80")})
        df = pd.DataFrame(rows)
        if len(df):
            st.dataframe(df.set_index(["Target", "Horizon", "Model"]).round(3))

    elif "R6_GETHSEMANE" in fname:
        rows = []
        for task, pols in data["tasks"].items():
            for pol, s in pols.items():
                rows.append({"Task": task, "Policy": pol,
                             "Reward": s.get("reward_mean"),
                             "Reward Std": s.get("reward_std"),
                             "Violations/ep": s.get("violations_mean")})
        st.dataframe(pd.DataFrame(rows).set_index(["Task", "Policy"]).round(2))

    elif "R6_EUCLIDIAN" in fname:
        rows = []
        for task, pols in data["tasks"].items():
            for pol, s in pols.items():
                ci = s.get("reward_ci95", [None, None])
                rows.append({"Task": task, "Policy": pol,
                             "Reward": s.get("reward_mean"),
                             "CI95 lo": ci[0], "CI95 hi": ci[1],
                             "Episodes": s.get("n_episodes")})
        st.dataframe(pd.DataFrame(rows).set_index(["Task", "Policy"]).round(2))
        st.metric("Total episodes", data.get("total_episodes", 0))

    elif "R6_PROVIDER" in fname:
        rows = []
        for g, r in data["graphs"].items():
            gnn = r["gnn_final"]
            base = r["baseline_direct_neighbors"]
            rows.append({"Graph": g, "Nodes": r["n_nodes"], "Edges": r["n_edges"],
                         "GNN F1": gnn["f1"], "GNN Acc": gnn["acc"],
                         "Baseline F1": base["f1"], "Improvement pp": r["improvement_f1_pp"]})
        st.dataframe(pd.DataFrame(rows).set_index("Graph").round(3))

    with st.expander("Raw JSON"):
        st.json(data)
