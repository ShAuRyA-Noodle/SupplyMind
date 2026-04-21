"""
leaderboard.py — F5. Live OpenEnv submissions leaderboard.

Anyone can submit an agent as a Python snippet implementing:

    def act(observation: dict, action_mask: list[bool]) -> int:
        # return a flat action index in [0, 280)
        ...

The submission is evaluated on the 3 standard SupplyMind tasks (easy, medium,
hard) across 3 fixed seeds. Scores are stored in a JSONL leaderboard file.

SECURITY: snippets are executed inside a restricted namespace but NOT fully
sandboxed. For production HF Space deployment use a Docker container with
resource limits (CPU/memory/time). This module is intended for local demo +
controlled submissions only.

Dual interface:
    - CLI: `python -m ShAuRyA_Supplymind.features.leaderboard --submit <file> --name foo`
    - Gradio UI: `python -m ShAuRyA_Supplymind.features.leaderboard --ui`
    - HTTP (mount as FastAPI router): see mount_fastapi() helper.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD_PATH = Path(__file__).resolve().parent / "leaderboard.jsonl"
EVAL_SEEDS = (42, 99, 7)
EVAL_TASKS = ("easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis")


@dataclass
class Entry:
    name: str
    author: str = ""
    timestamp: str = ""
    scores_easy: list[float] = field(default_factory=list)
    scores_medium: list[float] = field(default_factory=list)
    scores_hard: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    ci95_lower: float = 0.0
    runtime_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "author": self.author,
            "timestamp": self.timestamp,
            "scores_easy": [round(s, 4) for s in self.scores_easy],
            "scores_medium": [round(s, 4) for s in self.scores_medium],
            "scores_hard": [round(s, 4) for s in self.scores_hard],
            "mean_score": round(self.mean_score, 4),
            "ci95_lower": round(self.ci95_lower, 4),
            "runtime_s": round(self.runtime_s, 1),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Agent execution sandbox
# ---------------------------------------------------------------------------


_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "hasattr", "int", "isinstance", "len", "list", "map",
    "max", "min", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
}


def _load_submission(code: str) -> Callable[[dict, list], int]:
    """Exec the snippet in a restricted namespace; return the `act` callable."""
    import builtins as _b
    safe_builtins = {k: getattr(_b, k) for k in _ALLOWED_BUILTINS if hasattr(_b, k)}
    ns: dict[str, Any] = {"__builtins__": safe_builtins, "numpy": np, "np": np}
    exec(code, ns)   # noqa: S102 — accepted risk per docstring
    act = ns.get("act")
    if not callable(act):
        raise RuntimeError("submission must define `act(observation, action_mask) -> int`")
    return act


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _bootstrap_ci95_lower(scores: list[float], n_boot: int = 500) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=np.float64)
    rng = np.random.default_rng(12345)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5))


def evaluate_agent(act_fn: Callable, per_task_limit_s: float = 60.0) -> Entry:
    """Run act_fn against all 3 tasks x 3 seeds. Returns Entry."""
    from rl.gym_env import SupplyMindGymnasiumEnv
    from server.supply_environment import SupplyMindEnvironment

    start = time.time()
    scores_by_task: dict[str, list[float]] = {t: [] for t in EVAL_TASKS}
    try:
        for task_id in EVAL_TASKS:
            task_start = time.time()
            for seed in EVAL_SEEDS:
                env = SupplyMindGymnasiumEnv(task_id=task_id)
                core = SupplyMindEnvironment()
                obs, info = env.reset(seed=seed)
                core.reset(task_id=task_id, seed=seed)
                done = False
                steps = 0
                while not done and steps < 200:
                    if (time.time() - task_start) > per_task_limit_s:
                        break
                    mask = info.get("action_masks")
                    mask_list = mask.tolist() if hasattr(mask, "tolist") else list(mask or [])
                    try:
                        flat = int(act_fn(obs.tolist() if hasattr(obs, "tolist") else list(obs),
                                          mask_list))
                    except Exception as e:  # noqa: BLE001
                        raise RuntimeError(f"act() raised: {e}") from e
                    # bounds check + mask check
                    if flat < 0 or flat >= 280 or (mask_list and not mask_list[flat]):
                        # pick any valid action as fallback
                        valid_idx = [i for i, ok in enumerate(mask_list) if ok]
                        flat = valid_idx[0] if valid_idx else 0
                    action = np.array([flat // 40, flat % 40], dtype=np.int64)
                    obs, _, term, trunc, info = env.step(action)
                    sm = env._decode_action(action)
                    core.step(sm)
                    done = term or trunc or core.done
                    steps += 1
                scores_by_task[task_id].append(float(core.grade()["score"]))
                env.close()
    except Exception as e:  # noqa: BLE001
        entry = Entry(name="", error=str(e)[:300])
        entry.runtime_s = time.time() - start
        return entry

    entry = Entry(
        name="",
        scores_easy=scores_by_task[EVAL_TASKS[0]],
        scores_medium=scores_by_task[EVAL_TASKS[1]],
        scores_hard=scores_by_task[EVAL_TASKS[2]],
    )
    all_scores = (entry.scores_easy + entry.scores_medium + entry.scores_hard)
    entry.mean_score = float(np.mean(all_scores)) if all_scores else 0.0
    entry.ci95_lower = _bootstrap_ci95_lower(all_scores)
    entry.runtime_s = time.time() - start
    return entry


# ---------------------------------------------------------------------------
# Storage + leaderboard
# ---------------------------------------------------------------------------


def submit(code: str, name: str, author: str = "") -> Entry:
    act_fn = _load_submission(code)
    entry = evaluate_agent(act_fn)
    entry.name = name
    entry.author = author
    entry.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LEADERBOARD_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")
    return entry


def read_leaderboard(top_k: int = 20) -> list[dict]:
    if not LEADERBOARD_PATH.exists():
        return []
    entries = []
    for line in LEADERBOARD_PATH.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    entries.sort(key=lambda e: e.get("ci95_lower", 0), reverse=True)
    return entries[:top_k]


def render_leaderboard_markdown() -> str:
    lines = [
        "| Rank | Name | Author | CI95 lower | Mean | Easy | Medium | Hard | Time (s) |",
        "|------|------|--------|------------|------|------|--------|------|----------|",
    ]
    for i, e in enumerate(read_leaderboard(50), 1):
        def fmt_list(xs):
            if not xs:
                return "—"
            return f"{float(np.mean(xs)):.3f}"
        lines.append(
            f"| {i} | {e.get('name', '?')} | {e.get('author', '?')} | "
            f"{e.get('ci95_lower', 0):.4f} | {e.get('mean_score', 0):.3f} | "
            f"{fmt_list(e.get('scores_easy', []))} | {fmt_list(e.get('scores_medium', []))} | "
            f"{fmt_list(e.get('scores_hard', []))} | {e.get('runtime_s', 0):.0f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reference submissions (used in tests + seeding the leaderboard)
# ---------------------------------------------------------------------------


SUBMISSION_DO_NOTHING = """
def act(observation, action_mask):
    return 0  # action_type=0 = do_nothing; target_node=0
"""


SUBMISSION_RANDOM_VALID = """
import numpy as np

def act(observation, action_mask):
    valid = [i for i, ok in enumerate(action_mask) if ok]
    if not valid:
        return 0
    return int(valid[np.random.randint(len(valid))])
"""


SUBMISSION_ALERT_THEN_DO_NOTHING = """
def act(observation, action_mask):
    # Try issue_supplier_alert (action_type=6) targeting node 0; else do_nothing
    candidate = 6 * 40 + 0
    if 0 <= candidate < len(action_mask) and action_mask[candidate]:
        return candidate
    return 0
"""


# ---------------------------------------------------------------------------
# Gradio UI (optional)
# ---------------------------------------------------------------------------


def launch_gradio(share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("gradio not installed — `pip install gradio` to enable the UI")
        return

    def _submit_ui(code: str, name: str, author: str) -> tuple[str, str]:
        if not code.strip():
            return "[error] empty code", render_leaderboard_markdown()
        try:
            entry = submit(code, name=name or "anon", author=author or "")
            return f"✅ evaluated: mean={entry.mean_score:.3f} ci95_lower={entry.ci95_lower:.3f} time={entry.runtime_s:.0f}s\n{entry.error}", render_leaderboard_markdown()
        except Exception as e:  # noqa: BLE001
            return f"❌ {e}", render_leaderboard_markdown()

    with gr.Blocks(title="SupplyMind Leaderboard") as demo:
        gr.Markdown("# SupplyMind v4 OpenEnv Leaderboard\n"
                    "Submit a Python `act(observation, action_mask) -> int` function. "
                    "We evaluate on 3 tasks x 3 seeds = 9 episodes.")
        with gr.Row():
            with gr.Column():
                code = gr.Code(label="Your agent", language="python", value=SUBMISSION_RANDOM_VALID, lines=14)
                name = gr.Textbox(label="Submission name", placeholder="e.g. greedy_backup_v1")
                author = gr.Textbox(label="Author", placeholder="@you")
                submit_btn = gr.Button("Submit + Evaluate")
                output = gr.Textbox(label="Result", lines=3)
            with gr.Column():
                lb = gr.Markdown(render_leaderboard_markdown())
        submit_btn.click(_submit_ui, [code, name, author], [output, lb])
    demo.launch(share=share)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", type=str, default=None)
    parser.add_argument("--name", type=str, default="submission")
    parser.add_argument("--author", type=str, default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--seed-reference", action="store_true",
                        help="Seed leaderboard with 3 reference submissions")
    args = parser.parse_args()

    if args.ui:
        launch_gradio()
    elif args.list:
        print(render_leaderboard_markdown())
    elif args.seed_reference:
        for code, name in [(SUBMISSION_DO_NOTHING, "ref_do_nothing"),
                           (SUBMISSION_RANDOM_VALID, "ref_random_valid"),
                           (SUBMISSION_ALERT_THEN_DO_NOTHING, "ref_alert_fallback")]:
            entry = submit(code, name=name, author="supplymind-reference")
            print(f"{name}: mean={entry.mean_score:.3f} ci95_lower={entry.ci95_lower:.3f} err={entry.error[:80]}")
    elif args.submit:
        code = Path(args.submit).read_text(encoding="utf-8")
        entry = submit(code, name=args.name, author=args.author)
        print(json.dumps(entry.to_dict(), indent=2))
    else:
        print("usage: --submit <file> | --list | --seed-reference | --ui")
