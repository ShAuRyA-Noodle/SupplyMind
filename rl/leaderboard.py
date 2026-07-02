"""
HuggingFace Spaces Leaderboard for SupplyMind.

Gradio app that displays agent rankings across all 3 tasks. Rankings are
loaded exclusively from benchmark/results/benchmark_summary.csv, which is
produced by benchmark/run_full_benchmark.py. If that file is absent, the
app renders an explicit "no results yet" state — it never fabricates scores.

Usage:
    python -m rl.leaderboard          # Local Gradio server
    python -m rl.leaderboard --share   # Public share link
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = _PROJECT_ROOT / "benchmark" / "results"


def load_leaderboard_data() -> list[dict[str, Any]]:
    """Load benchmark results into leaderboard format.

    Returns rows parsed from benchmark/results/benchmark_summary.csv. Returns
    an empty list if that file does not exist — callers must render an explicit
    "no results" state rather than substitute fabricated numbers.
    """
    summary_path = RESULTS_DIR / "benchmark_summary.csv"

    if not summary_path.exists():
        logger.warning(
            "benchmark_summary.csv not found at %s — no benchmark results to display. "
            "Run benchmark/run_full_benchmark.py to generate it.",
            summary_path,
        )
        return []

    rows = []
    with open(summary_path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            entry = {
                "Agent": row[0],
                "Easy": row[1] if len(row) > 1 else "—",
                "Medium": row[2] if len(row) > 2 else "—",
                "Hard": row[3] if len(row) > 3 else "—",
                "Average": row[4] if len(row) > 4 else "—",
            }
            rows.append(entry)
    return rows


_NO_RESULTS_MESSAGE = (
    "NO BENCHMARK RESULTS YET — run benchmark/run_full_benchmark.py to generate "
    "benchmark/results/benchmark_summary.csv."
)


def create_gradio_app(share: bool = False):
    """Create and launch Gradio leaderboard app."""
    try:
        import gradio as gr
    except ImportError:
        logger.error("gradio not installed. pip install gradio")
        # Create a simple HTML fallback
        _create_html_leaderboard()
        return

    data = load_leaderboard_data()

    with gr.Blocks(title="SupplyMind Leaderboard") as app:
        gr.Markdown("# SupplyMind Agent Leaderboard")

        if not data:
            gr.Markdown(f"## {_NO_RESULTS_MESSAGE}")
        else:
            gr.Markdown(
                "Benchmark results across 3 supply chain risk management tasks. "
                "Higher scores = better. Loaded from benchmark/results/benchmark_summary.csv."
            )

            headers = ["Agent", "Easy", "Medium", "Hard", "Average"]
            table_data = [[d[h] for h in headers] for d in data]

            gr.Dataframe(
                value=table_data,
                headers=headers,
                label="Agent Rankings",
            )

            gr.Markdown("---")
            gr.Markdown(
                "**Environment:** SupplyMind OpenEnv — supply chain risk management\n\n"
                "**Tasks:** Typhoon Response (Easy), Multi-Front Crisis (Medium), Cascading Crisis (Hard)\n\n"
                "**Metrics:** Graded on revenue preservation, timeliness, cost efficiency, stockout prevention"
            )

    app.launch(share=share)


def _create_html_leaderboard() -> Path:
    """Create static HTML leaderboard when Gradio isn't available."""
    data = load_leaderboard_data()
    html = """<!DOCTYPE html>
<html><head><title>SupplyMind Leaderboard</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }
h1 { color: #1976d2; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; }
th { background: #1976d2; color: white; padding: 12px; text-align: center; }
td { padding: 10px; text-align: center; border-bottom: 1px solid #eee; }
tr:hover { background: #f5f5f5; }
.footer { color: #666; font-size: 0.9em; margin-top: 30px; }
.empty { background: #fff3cd; border: 1px solid #ffc107; padding: 16px; border-radius: 6px; color: #664d03; }
</style></head><body>
<h1>SupplyMind Agent Leaderboard</h1>
"""
    if not data:
        html += f'<p class="empty">{_NO_RESULTS_MESSAGE}</p>\n'
    else:
        html += "<p>Benchmark results across 3 supply chain risk management tasks.</p>\n"
        html += "<table>\n<tr><th>Agent</th><th>Easy</th><th>Medium</th><th>Hard</th><th>Average</th></tr>\n"
        for d in data:
            html += f"<tr><td>{d['Agent']}</td><td>{d['Easy']}</td><td>{d['Medium']}</td><td>{d['Hard']}</td><td>{d['Average']}</td></tr>\n"
        html += "</table>\n"

    html += """<p class="footer">SupplyMind OpenEnv — results generated by benchmark/run_full_benchmark.py</p>
</body></html>"""

    path = _PROJECT_ROOT / "leaderboard.html"
    path.write_text(html)
    logger.info("Static leaderboard saved to %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SupplyMind Leaderboard")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    # Always create HTML version
    _create_html_leaderboard()
    # Try Gradio
    create_gradio_app(share=args.share)


if __name__ == "__main__":
    main()
