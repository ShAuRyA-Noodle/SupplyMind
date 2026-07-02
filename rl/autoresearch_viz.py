"""
AutoResearch Progress Visualization.

Generates publication-quality charts showing:
  1. Experiment leaderboard sorted by grade
  2. Hyperparameter heatmap (which params matter most)
  3. Convergence plot (improvement over experiments)
  4. Per-task specialist comparison

Inspired by karpathy/autoresearch progress.png.

Usage:
    python -m rl.autoresearch_viz
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "autoresearch_results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def load_results() -> list[dict[str, Any]]:
    """Load AutoResearch experiment results."""
    results_path = RESULTS_DIR / "autoresearch_results.json"
    if results_path.exists():
        return json.loads(results_path.read_text())

    # Fallback: scan checkpoints for autoresearch models
    results = []
    for f in CHECKPOINT_DIR.glob("autoresearch_*.pt"):
        import torch
        try:
            ckpt = torch.load(str(f), map_location="cpu", weights_only=True)
            results.append({
                "name": ckpt.get("experiment", f.stem),
                "grade_avg": ckpt.get("grade_avg", 0),
            })
        except Exception:
            pass
    return results


def generate_leaderboard_html() -> str:
    """Generate HTML leaderboard table."""
    results = load_results()
    if not results:
        return "<p>No AutoResearch results yet. Run: python -m rl.autoresearch</p>"

    results.sort(key=lambda r: r.get("grade_avg", 0), reverse=True)

    html = """<table style="width:100%; border-collapse:collapse;">
    <tr style="background:#1976d2; color:white;">
        <th style="padding:8px;">#</th>
        <th style="padding:8px;">Experiment</th>
        <th style="padding:8px;">Easy</th>
        <th style="padding:8px;">Medium</th>
        <th style="padding:8px;">Hard</th>
        <th style="padding:8px;">Avg</th>
        <th style="padding:8px;">LR</th>
        <th style="padding:8px;">CVaR α</th>
        <th style="padding:8px;">Action Bonus</th>
    </tr>"""

    for i, r in enumerate(results[:15]):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        bold = "font-weight:bold;" if i == 0 else ""
        html += f"""
    <tr style="background:{bg};{bold}">
        <td style="padding:6px; text-align:center;">{i+1}</td>
        <td style="padding:6px;">{r.get('name', '?')}</td>
        <td style="padding:6px; text-align:center;">{r.get('grade_easy', 0):.3f}</td>
        <td style="padding:6px; text-align:center;">{r.get('grade_medium', 0):.3f}</td>
        <td style="padding:6px; text-align:center;">{r.get('grade_hard', 0):.3f}</td>
        <td style="padding:6px; text-align:center;">{r.get('grade_avg', 0):.3f}</td>
        <td style="padding:6px; text-align:center;">{r.get('lr', '?')}</td>
        <td style="padding:6px; text-align:center;">{r.get('cvar_alpha', '?')}</td>
        <td style="padding:6px; text-align:center;">{r.get('real_action_bonus', '?')}</td>
    </tr>"""

    html += "</table>"
    return html


def generate_progress_data() -> dict[str, Any]:
    """Generate data for progress chart (improvement over experiments)."""
    results = load_results()
    if not results:
        return {"experiments": [], "best_so_far": []}

    best = 0
    progress = []
    for i, r in enumerate(results):
        avg = r.get("grade_avg", 0)
        best = max(best, avg)
        progress.append({
            "experiment": i + 1,
            "name": r.get("name", f"exp_{i}"),
            "grade_avg": avg,
            "best_so_far": best,
        })

    return {"experiments": progress, "final_best": best}


def generate_hyperparameter_importance() -> dict[str, float]:
    """Analyze which hyperparameters correlate most with high scores."""
    results = load_results()
    if len(results) < 3:
        return {}

    params = ["lr", "cvar_alpha", "hidden_dim", "real_action_bonus", "alert_penalty", "gamma"]
    importance = {}

    for param in params:
        values = [r.get(param, 0) for r in results]
        scores = [r.get("grade_avg", 0) for r in results]

        if len(set(values)) < 2:
            importance[param] = 0.0
            continue

        # Simple correlation
        try:
            corr = float(np.corrcoef(values, scores)[0, 1])
            importance[param] = round(abs(corr), 3) if not np.isnan(corr) else 0.0
        except Exception:
            importance[param] = 0.0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def get_best_config() -> dict[str, Any]:
    """Get the best experiment configuration."""
    results = load_results()
    if not results:
        return {}
    results.sort(key=lambda r: r.get("grade_avg", 0), reverse=True)
    return results[0]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("AutoResearch Progress Report")
    print("=" * 60)

    results = load_results()
    print(f"Total experiments: {len(results)}")

    if results:
        results.sort(key=lambda r: r.get("grade_avg", 0), reverse=True)
        print(f"\nBest: {results[0].get('name', '?')} (avg={results[0].get('grade_avg', 0):.3f})")

        print("\nLeaderboard:")
        for i, r in enumerate(results[:10]):
            print(f"  {i+1}. {r.get('name','?'):25s} easy={r.get('grade_easy',0):.3f} "
                  f"med={r.get('grade_medium',0):.3f} hard={r.get('grade_hard',0):.3f} "
                  f"avg={r.get('grade_avg',0):.3f}")

        print("\nHyperparameter Importance:")
        imp = generate_hyperparameter_importance()
        for param, score in imp.items():
            bar = "#" * int(score * 20)
            print(f"  {param:20s} {score:.3f} {bar}")


if __name__ == "__main__":
    main()
