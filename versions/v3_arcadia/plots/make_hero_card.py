"""Generate a single PNG "result card" with the 10 headline numbers — usable
as the hero image on HF Space, pitch deck slide 1, Twitter card, LinkedIn
preview. Single source of truth for cover visual.

Outputs: versions/v3_arcadia/plots/hero_result_card.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS = ROOT / "v3_arcadia" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


TITLE = "SupplyMind v3.0-arcadia — 10 headline numbers"
SUBTITLE = "OpenEnv-compliant supply-chain risk management · 261,175 real data points · 173 tests · zero synthetic"

NUMBERS = [
    ("0.971", "RAG nDCG@10", "Snowflake-Arctic-L, out-of-domain"),
    ("0.962", "RAG P@1", "mxbai bi-encoder, 6,483 chunks"),
    ("0.978", "RAG MRR", "precise queries"),
    ("0.750", "LLM α (ordinal)", "2-judge Krippendorff"),
    ("0.747", "Cohen κ", "Qwen × Mistral weighted"),
    ("0.024", "conformal dev", "per-horizon, WTI @ 95%"),
    ("+26.8%", "masking lift", "isolated, easy task"),
    ("+15.1%", "masking lift", "isolated, hard task"),
    ("−64%", "GNN MAE vs MLP", "hard 40-node graph"),
    ("173", "tests passing", "2m 14s, deterministic"),
]


def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    ax.text(5.0, 9.5, TITLE, ha="center", va="center",
            fontsize=21, fontweight="bold", color="#f0f6fc")
    ax.text(5.0, 9.0, SUBTITLE, ha="center", va="center",
            fontsize=11, color="#8b949e")

    # 2x5 grid of cards
    cols, rows = 5, 2
    x_margin = 0.4; y_margin = 0.4
    card_w = (10 - 2 * x_margin - (cols - 1) * 0.15) / cols
    card_h = 1.85
    y_top = 7.8
    palette = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ff7b72"] * 2

    for i, (big, small, caption) in enumerate(NUMBERS):
        r = i // cols; c = i % cols
        x = x_margin + c * (card_w + 0.15)
        y = y_top - r * (card_h + 0.45) - card_h
        box = FancyBboxPatch((x, y), card_w, card_h,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              linewidth=1, edgecolor="#30363d",
                              facecolor="#161b22")
        ax.add_patch(box)
        ax.text(x + card_w / 2, y + 1.30, big, ha="center", va="center",
                fontsize=24, fontweight="bold", color=palette[i])
        ax.text(x + card_w / 2, y + 0.70, small, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#f0f6fc")
        ax.text(x + card_w / 2, y + 0.28, caption, ha="center", va="center",
                fontsize=8, color="#8b949e")

    ax.text(5.0, 0.4,
            "Every number is reproducible with `python scripts/run_all.py` · "
            "github.com/ShAuRyA-Noodle/Sleep-Token",
            ha="center", va="center", fontsize=9, color="#58a6ff", style="italic")

    out = PLOTS / "hero_result_card.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
