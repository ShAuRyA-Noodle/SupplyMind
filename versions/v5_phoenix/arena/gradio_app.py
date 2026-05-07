"""gradio_app.py — Gradio UI for judges to drop in their policy.

Run standalone:
    python -m versions.v5_phoenix.arena.gradio_app

Or mount inside the Phoenix FastAPI server via `gradio.mount_gradio_app`.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_leaderboard(board):
    rows = []
    for r in board.get("rows", [])[:20]:
        ci = r.get("overall_ci95") or [None, None]
        ci_text = f"[{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "—"
        rows.append([r["rank"], r["policy_name"], round(r["overall_reward_mean"], 3),
                    ci_text, r.get("total_violations", 0), r.get("source", "")])
    return rows


def _run(policy_file, name, episodes):
    from . import leaderboard, runner

    if policy_file is None:
        return None, "Please upload a policy file.", _format_leaderboard(leaderboard.rebuild())
    p = Path(policy_file.name if hasattr(policy_file, "name") else policy_file)
    if not p.exists():
        return None, f"File not found: {p}", _format_leaderboard(leaderboard.rebuild())

    display = name or p.stem
    try:
        t0 = time.time()
        result = runner.evaluate_policy(p, n_episodes_per_task=episodes, policy_name=display)
        elapsed = time.time() - t0

        import json as _json
        out_path = leaderboard.ARENA_DIR / f"{display}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(result.to_dict(), indent=2))

        board = leaderboard.rebuild()
        msg = (f"Evaluated {display} in {elapsed:.1f}s. "
               f"Overall reward {result.overall_reward_mean:.3f} "
               f"[{result.overall_ci95_lower:.3f}, {result.overall_ci95_upper:.3f}]. "
               f"Rank: {result.rank_against_baseline}.")
        return result.to_dict(), msg, _format_leaderboard(board)
    except Exception as e:  # noqa: BLE001
        logger.exception("arena run failed")
        return None, f"Error: {e}", _format_leaderboard(leaderboard.rebuild())


def build_demo():
    import gradio as gr
    from . import leaderboard as _lb

    board = _lb.rebuild()
    with gr.Blocks(title="SupplyMind OpenEnv Arena") as demo:
        gr.Markdown(
            "# SupplyMind OpenEnv Arena\n\n"
            "Drop in your PyTorch policy. Returns bootstrap-CI95 reward on 3 tasks "
            "(easy_typhoon_response, medium_multi_front, hard_cascading_crisis). "
            "Loader dispatch: `sb3_contrib.MaskablePPO` -> `stable_baselines3.PPO` -> "
            "`torch.nn.Module`.\n\n"
            "Run time: ~1-3 min per 50-ep-per-task submission on RTX 4080 Laptop. "
            "Leaderboard is live; your submission ranks against the v3 SOTA baselines "
            "(R6 Euclidian / Algo Comparison).")

        with gr.Row():
            with gr.Column():
                policy_file = gr.File(label="policy.pt / policy.zip / policy.pth", file_types=[".pt", ".zip", ".pth"])
                name_input = gr.Textbox(label="Display name (optional)", placeholder="my_awesome_policy")
                episodes_input = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Episodes per task")
                run_btn = gr.Button("Evaluate on Arena", variant="primary")
                status_out = gr.Textbox(label="Status", lines=3)
            with gr.Column():
                result_json = gr.JSON(label="ArenaResult")

        gr.Markdown("## Leaderboard")
        leaderboard_table = gr.Dataframe(
            headers=["Rank", "Policy", "Reward mean", "CI95", "Violations", "Source"],
            value=_format_leaderboard(board),
            interactive=False,
        )

        run_btn.click(
            fn=_run,
            inputs=[policy_file, name_input, episodes_input],
            outputs=[result_json, status_out, leaderboard_table],
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
