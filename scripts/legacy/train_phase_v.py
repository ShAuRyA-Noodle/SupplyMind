"""
Phase V "Are You Really Okay?" — supplymind-analyst:v3 + blind A/B eval.

Upgrades:
  U38 10-example Modelfile v3 (diverse real crises, forensic detail)
  U39 Blind A/B vs qwen2.5:7b-instruct on 50 scenarios, judged by gemma4:e4b-it-bf16

Runs `ollama cp` to preserve v2, builds v3, runs A/B, outputs JSON.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "rl" / "checkpoints"
RESULTS = ROOT / "benchmark" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def build_v3():
    # Preserve v2
    log.info("Preserving supplymind-analyst:v2 via ollama cp...")
    subprocess.run(["ollama", "cp", "supplymind-analyst:v2", "supplymind-analyst:v2-backup"], check=False)

    log.info("Building supplymind-analyst:v3...")
    r = subprocess.run(["ollama", "create", "supplymind-analyst:v3", "-f", str(ROOT / "rl" / "lora" / "Modelfile.v3")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"v3 build failed: {r.stderr}")
    log.info("v3 built successfully")


def ab_test():
    import ollama
    from rl.explainer import _build_prompt, _passes_quality_gate
    import numpy as np

    scenarios = []
    rng = np.random.default_rng(42)
    actions = ["do_nothing", "issue_supplier_alert", "reroute_shipment", "expedite_order",
               "increase_safety_stock", "activate_backup_supplier", "hedge_commodity"]
    disruptions = ["cyclone", "earthquake", "port_strike", "chip_shortage", "canal_blockage",
                   "cyber_attack", "supplier_financial_distress", "political_unrest"]
    nodes = ["SUP_TSMC", "SUP_SAMSUNG", "PORT_KAOHSIUNG", "ROUTE_SUEZ", "SUP_FOXCONN",
             "PORT_SHANGHAI", "CARRIER_MAERSK", "SUP_INTEL"]

    for i in range(50):
        action = actions[rng.integers(0, len(actions))]
        disr = disruptions[rng.integers(0, len(disruptions))]
        node = nodes[rng.integers(0, len(nodes))]
        state = (
            f"Day {rng.integers(1, 30)} of {rng.integers(30, 60)}. Budget {rng.uniform(30, 95):.0f}% remaining. "
            f"Supply chain health {rng.integers(40, 95)}/100. MC P50 ${rng.uniform(0.2, 3.5)*1e6:,.0f}. "
            f"Active: {disr} (severity {rng.uniform(0.1, 0.95):.2f}) affecting {node}."
        )
        scenarios.append({"id": i, "action": action, "node": node, "state": state, "disruption": disr})

    results = []
    v3_wins = 0; base_wins = 0; ties = 0
    for s in scenarios:
        prompt = _build_prompt(s["state"], s["action"], s["node"], None,
                               shap_top=[("node0_risk", 0.42), ("NOAA_3", 0.38)],
                               counterfactual_p50=2.1e6,
                               rag_precedent="Tohoku 2011: $1.2B Toyota loss from single-source exposure")

        try:
            r_v3 = ollama.chat(model="supplymind-analyst:v3",
                               messages=[{"role": "user", "content": prompt}],
                               options={"temperature": 0.2})
            r_base = ollama.chat(model="qwen2.5:7b-instruct",
                                 messages=[{"role": "user", "content": prompt}],
                                 options={"temperature": 0.2})
            t_v3 = r_v3["message"]["content"]
            t_base = r_base["message"]["content"]
            pass_v3 = _passes_quality_gate(t_v3)
            pass_base = _passes_quality_gate(t_base)

            # Judge with gemma
            judge_prompt = (
                "You are an expert supply chain analyst judging two AI-generated explanations for the SAME decision. "
                f"Scenario: {s['state']}\nACTION: {s['action']} targeting {s['node']}.\n\n"
                f"=== RESPONSE A ===\n{t_v3}\n\n=== RESPONSE B ===\n{t_base}\n\n"
                "Rate each on: (1) structured 4-section format, (2) factual grounding, (3) precedent-use. "
                "Output EXACTLY one of: 'WINNER: A', 'WINNER: B', or 'TIE'."
            )
            r_judge = ollama.chat(model="gemma4:e4b-it-bf16",
                                  messages=[{"role": "user", "content": judge_prompt}],
                                  options={"temperature": 0.0})
            verdict = r_judge["message"]["content"].strip().upper()
            if "WINNER: A" in verdict:
                winner = "v3"; v3_wins += 1
            elif "WINNER: B" in verdict:
                winner = "base"; base_wins += 1
            else:
                winner = "tie"; ties += 1
            results.append({"scenario": s["id"], "action": s["action"], "disruption": s["disruption"],
                            "pass_v3": bool(pass_v3), "pass_base": bool(pass_base), "winner": winner,
                            "judge_verdict": verdict[:100]})
            log.info(f"  sc{s['id']}: v3_pass={pass_v3} base_pass={pass_base} winner={winner}")
        except Exception as e:
            log.warning(f"  sc{s['id']} error: {e}")
            results.append({"scenario": s["id"], "error": str(e)})

    summary = {
        "n_scenarios": len(scenarios),
        "v3_wins": v3_wins, "base_wins": base_wins, "ties": ties,
        "v3_win_rate": v3_wins / max(len(scenarios), 1),
        "judge_model": "gemma4:e4b-it-bf16",
        "model_a": "supplymind-analyst:v3",
        "model_b": "qwen2.5:7b-instruct",
        "scenarios": results,
    }
    log.info(f"A/B: v3={v3_wins} base={base_wins} ties={ties} (v3 win rate {summary['v3_win_rate']:.2%})")
    (RESULTS / "AB_ANALYST_V3.json").write_text(json.dumps(summary, indent=2))


def main():
    build_v3()
    ab_test()
    log.info("Phase V 'Are You Really Okay?' complete.")


if __name__ == "__main__":
    main()
