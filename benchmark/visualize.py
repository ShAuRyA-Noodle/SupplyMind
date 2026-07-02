"""
Chart generation for SupplyMind benchmarks.

Every figure is generated ONLY from a real result file produced by the
benchmark runners. There is no invented / hardcoded fallback data: if the
required result file is missing, the generator raises FileNotFoundError with a
message telling you which command to run first.

Generates:
  - Benchmark comparison table   (from benchmark/results/benchmark_summary.csv)
  - Ablation progressive bar chart (from benchmark/results/ablation_results.csv)
  - Agent comparison radar chart (from benchmark/results/benchmark_summary.csv)

Usage:
    python -m benchmark.visualize
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_CSV = RESULTS_DIR / "benchmark_summary.csv"
ABLATION_CSV = RESULTS_DIR / "ablation_results.csv"


def _require(path: Path, generator: str) -> None:
    """Raise a helpful FileNotFoundError if a required result file is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required benchmark result file not found: {path}\n"
            f"Generate it first by running: {generator}"
        )


def _parse_mean(cell: str) -> float | None:
    """Parse '0.771+/-0.020' or '0.714' -> 0.771 float, or None if not numeric."""
    cell = cell.strip()
    if "+/-" in cell:
        cell = cell.split("+/-")[0]
    try:
        return float(cell)
    except ValueError:
        return None


def generate_benchmark_table() -> Path | None:
    """Generate the benchmark comparison table from benchmark_summary.csv."""
    _require(SUMMARY_CSV, "python -m benchmark.run_benchmark")

    with open(SUMMARY_CSV) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping benchmark table")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#1976d2",
            font=dict(color="white", size=14),
            align="center",
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=[["#f5f5f5", "white"] * (len(rows) // 2 + 1)][:len(rows)],
            font=dict(size=12),
            align="center",
        ),
    )])
    fig.update_layout(
        title=f"SupplyMind Benchmark Results ({len(rows)} agents)",
        height=400, margin=dict(l=10, r=10, t=40, b=10),
    )
    path = FIGURES_DIR / "benchmark_table.html"
    fig.write_html(str(path))
    logger.info("Benchmark table: %s", path)
    return path


def generate_ablation_chart() -> Path | None:
    """Generate the ablation bar chart from ablation_results.csv."""
    _require(ABLATION_CSV, "python -m benchmark.ablation")

    configs: list[str] = []
    scores: list[float] = []
    with open(ABLATION_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(row["configuration"])
            scores.append(float(row["avg_mean"]))

    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping ablation chart")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=configs, y=scores,
        text=[f"{s:.2f}" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        title="Ablation: Component Contribution to Score",
        yaxis=dict(title="Average Score", range=[0, 1]),
        height=450, margin=dict(l=50, r=20, t=50, b=50),
    )
    path = FIGURES_DIR / "ablation_chart.html"
    fig.write_html(str(path))
    logger.info("Ablation chart: %s", path)
    return path


def generate_radar_chart() -> Path | None:
    """Generate an agent comparison radar from real per-task means in
    benchmark_summary.csv. Axes are the three task difficulties (the only
    per-agent breakdown the benchmark actually produces)."""
    _require(SUMMARY_CSV, "python -m benchmark.run_benchmark")

    categories = ["Easy", "Medium", "Hard"]
    agents: dict[str, list[float]] = {}
    with open(SUMMARY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = [_parse_mean(row.get(cat, "")) for cat in categories]
            if any(v is None for v in values):
                continue  # e.g. a 'not_evaluated' agent — skip, never invent
            agents[row["Agent"]] = values  # type: ignore[assignment]

    if not agents:
        raise FileNotFoundError(
            f"No agent in {SUMMARY_CSV} has numeric per-task scores to plot "
            f"(all rows may be 'not_evaluated'). Run: python -m benchmark.run_benchmark"
        )

    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping radar chart")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    for name, values in agents.items():
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", name=name, opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Agent Score by Task Difficulty",
        height=500,
    )
    path = FIGURES_DIR / "radar_chart.html"
    fig.write_html(str(path))
    logger.info("Radar chart: %s", path)
    return path


def generate_all() -> None:
    """Generate every figure that has real data available. Fails loud if
    nothing could be produced."""
    generators = [
        ("benchmark table", generate_benchmark_table),
        ("ablation chart", generate_ablation_chart),
        ("radar chart", generate_radar_chart),
    ]
    produced = 0
    missing: list[str] = []
    for name, fn in generators:
        try:
            if fn() is not None:
                produced += 1
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", name, e)
            missing.append(name)

    if produced == 0:
        raise FileNotFoundError(
            "No figures could be generated: no real benchmark result files were "
            f"found in {RESULTS_DIR}. Run `python -m benchmark.run_benchmark` "
            "(and `python -m benchmark.ablation`) first."
        )
    if missing:
        logger.warning("Figures skipped (missing result files): %s", missing)
    logger.info("Figures generated in %s", FIGURES_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    generate_all()


if __name__ == "__main__":
    main()
