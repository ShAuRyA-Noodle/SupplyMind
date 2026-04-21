"""
seed_experiments.py — 5 hand-crafted hypothesis diffs to bootstrap the loop.

These are DETERMINISTIC, hand-coded, no LLM involved. They seed state.json
with diverse starting points before the Qwen/Claude agent takes over.

Each seed covers a different search direction:
    S1: bigger network (MlpPolicy [256, 256] instead of [64, 64])
    S2: higher entropy coefficient (ent_coef=0.1 vs 0.01) — more exploration
    S3: curriculum learning (easy -> medium -> hard across training)
    S4: RecurrentPPO with GRU memory
    S5: reward shaping (add action diversity bonus)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = Path(__file__).resolve().parent
CANDIDATE_PATH = AUTORESEARCH_DIR / "candidate_train.py"


@dataclass
class SeedHypothesis:
    name: str
    hypothesis: str
    expected: str
    justification: str
    references: list[str]
    mutator: Callable[[str], str]  # old_code -> new_code


def _replace_block(code: str, start_marker: str, end_marker: str, new_block: str) -> str:
    """Replace content between two marker lines.

    Markers must be the ENTIRE stripped line content (not just a substring) —
    otherwise we'd match occurrences inside docstrings.
    The output is: (code up to and including start marker) + new_block + (end marker and rest).
    """
    lines = code.splitlines(keepends=True)
    start_idx = None
    end_idx = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped == start_marker and start_idx is None:
            start_idx = i
        elif stripped == end_marker and start_idx is not None:
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise ValueError(f"markers not found: {start_marker} / {end_marker}")
    return "".join(lines[: start_idx + 1]) + new_block + "".join(lines[end_idx:])


# -----------------------------------------------------------------------------
# Mutator helpers — each returns a new candidate_train.py text
# -----------------------------------------------------------------------------

def _s1_bigger_network(old: str) -> str:
    return old.replace(
        'policy_kwargs={"net_arch": [64, 64]}',
        'policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU}',
    ).replace(
        'return "MaskablePPO MlpPolicy[64,64], lr=3e-4, n_steps=2048, gamma=0.99"',
        'return "MaskablePPO MlpPolicy[256,256]+ReLU, lr=3e-4, n_steps=2048, gamma=0.99"',
    )


def _s2_higher_entropy(old: str) -> str:
    return old.replace(
        "ent_coef=0.01,",
        "ent_coef=0.1,",
    ).replace(
        'return "MaskablePPO MlpPolicy[64,64], lr=3e-4, n_steps=2048, gamma=0.99"',
        'return "MaskablePPO MlpPolicy[64,64], lr=3e-4, ent_coef=0.1 (exploration), gamma=0.99"',
    )


def _s3_curriculum(old: str) -> str:
    """Inject a CurriculumCallback that switches tasks partway through training."""
    new_block = '''
def _curriculum_env(stage: str):
    from sb3_contrib.common.wrappers import ActionMasker
    task_map = {
        "easy": "easy_typhoon_response",
        "medium": "medium_multi_front",
        "hard": "hard_cascading_crisis",
    }
    def _fn():
        env = SupplyMindGymnasiumEnv(task_id=task_map[stage], training_mode=True, grade_reward=False)
        return ActionMasker(env, lambda env: env.unwrapped._compute_action_mask())
    return _fn


def build_policy_and_env(seed: int):
    """Seed with easy task; training loop will cycle through curriculum."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([_curriculum_env("easy")])
    env.seed(seed)
    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, policy_kwargs={"net_arch": [128, 128]},
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed, verbose=0,
    )
    return model, env


def train_policy(model, env, total_steps: int) -> None:
    """Curriculum: 40% easy, 30% medium, 30% hard."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    budget_easy = int(total_steps * 0.4)
    budget_med = int(total_steps * 0.3)
    budget_hard = total_steps - budget_easy - budget_med

    model.learn(total_timesteps=budget_easy, progress_bar=False, reset_num_timesteps=False)
    model.set_env(DummyVecEnv([_curriculum_env("medium")]))
    model.learn(total_timesteps=budget_med, progress_bar=False, reset_num_timesteps=False)
    model.set_env(DummyVecEnv([_curriculum_env("hard")]))
    model.learn(total_timesteps=budget_hard, progress_bar=False, reset_num_timesteps=False)


def architecture_summary() -> str:
    return "MaskablePPO [128,128] curriculum easy->med->hard (40/30/30 split)"

'''
    return _replace_block(old, "# --- SAFE TO MODIFY BELOW ---", "# --- SAFE TO MODIFY ABOVE ---", new_block)


def _s4_recurrent_ppo(old: str) -> str:
    """Swap MaskablePPO for RecurrentPPO with LSTM."""
    new_block = '''
def build_policy_and_env(seed: int):
    """RecurrentPPO with LSTM memory (128 units)."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _env_fn():
        env = SupplyMindGymnasiumEnv(
            task_id="easy_typhoon_response",
            training_mode=True,
            grade_reward=False,
        )
        return env  # RecurrentPPO handles masking via info

    env = DummyVecEnv([_env_fn])
    env.seed(seed)
    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        learning_rate=3e-4, n_steps=256, batch_size=64, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"lstm_hidden_size": 128, "n_lstm_layers": 1,
                       "net_arch": [64]},
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed, verbose=0,
    )
    return model, env


def train_policy(model, env, total_steps: int) -> None:
    model.learn(total_timesteps=total_steps, progress_bar=False)


def architecture_summary() -> str:
    return "RecurrentPPO MlpLstmPolicy lstm=128, [64], lr=3e-4"

'''
    return _replace_block(old, "# --- SAFE TO MODIFY BELOW ---", "# --- SAFE TO MODIFY ABOVE ---", new_block)


def _s5_reward_shaping(old: str) -> str:
    """Wrap env with an action-diversity reward shaper."""
    new_block = '''
class ActionDiversityWrapper(__import__('gymnasium').Wrapper):
    """Add a small reward bonus when the agent chooses an action not used in
    the last K steps. Encourages exploration of the 280-dim action space."""

    def __init__(self, env, k: int = 5, bonus: float = 0.02):
        super().__init__(env)
        self.k = k
        self.bonus = bonus
        self.history = []

    def reset(self, **kwargs):
        self.history = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        key = tuple(action) if hasattr(action, "__len__") else int(action)
        if key not in self.history:
            reward = float(reward) + self.bonus
        self.history.append(key)
        if len(self.history) > self.k:
            self.history.pop(0)
        return obs, reward, terminated, truncated, info


def build_policy_and_env(seed: int):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _env_fn():
        env = SupplyMindGymnasiumEnv(
            task_id="easy_typhoon_response",
            training_mode=True,
            grade_reward=False,
        )
        env = ActionDiversityWrapper(env, k=5, bonus=0.02)
        return ActionMasker(env, lambda e: e.unwrapped._compute_action_mask())

    env = DummyVecEnv([_env_fn])
    env.seed(seed)
    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, policy_kwargs={"net_arch": [64, 64]},
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed, verbose=0,
    )
    return model, env


def train_policy(model, env, total_steps: int) -> None:
    model.learn(total_timesteps=total_steps, progress_bar=False)


def architecture_summary() -> str:
    return "MaskablePPO [64,64] + ActionDiversityWrapper(k=5, bonus=0.02)"

'''
    return _replace_block(old, "# --- SAFE TO MODIFY BELOW ---", "# --- SAFE TO MODIFY ABOVE ---", new_block)


SEEDS: list[SeedHypothesis] = [
    SeedHypothesis(
        name="s1_bigger_network",
        hypothesis="MlpPolicy [256, 256] + ReLU beats [64, 64] on hard task (more capacity for 408-dim obs).",
        expected="+0.02 to +0.05 on CI95 lower",
        justification="Standard sb3 recommendation for obs_dim > 200. Our 408-dim obs is above the [64,64] capacity regime.",
        references=["https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"],
        mutator=_s1_bigger_network,
    ),
    SeedHypothesis(
        name="s2_higher_entropy",
        hypothesis="ent_coef=0.1 vs 0.01 explores more of the 280-action space early, avoiding greedy local optima.",
        expected="+0.01 to +0.04 on medium/hard (entropy less helpful on easy).",
        justification="Schulman et al. 2017 PPO paper: ent_coef sweep shows 0.01-0.1 optimal for discrete-heavy action spaces.",
        references=["https://arxiv.org/abs/1707.06347"],
        mutator=_s2_higher_entropy,
    ),
    SeedHypothesis(
        name="s3_curriculum_learning",
        hypothesis="Curriculum (easy -> medium -> hard) accelerates learning on cascading crisis via transfer.",
        expected="+0.03 to +0.07 on hard task; neutral on easy.",
        justification="Bengio et al. 2009 curriculum learning. Our hard_cascading_crisis has very sparse reward — warm-starting from easy weights should help.",
        references=["https://dl.acm.org/doi/10.1145/1553374.1553380"],
        mutator=_s3_curriculum,
    ),
    SeedHypothesis(
        name="s4_recurrent_ppo",
        hypothesis="RecurrentPPO with LSTM-128 captures long-horizon dependencies across disruption phases.",
        expected="-0.10 to +0.05 (risky; our R6 data shows RecurrentPPO is -10% on unmasked, but LSTM tuning may flip this).",
        justification="R6_ALGO_COMPARISON.json: RecurrentPPO 1.081 vs MaskablePPO 1.201 out-of-the-box. Tuning LSTM hidden + proper batch may close gap.",
        references=["v3_arcadia/results/R6_ALGO_COMPARISON.json"],
        mutator=_s4_recurrent_ppo,
    ),
    SeedHypothesis(
        name="s5_action_diversity_bonus",
        hypothesis="Bonus reward for actions not used in last 5 steps encourages exploration of the 280-dim space without hand-labeling.",
        expected="+0.01 to +0.03 on medium (most starved for exploration).",
        justification="Pathak et al. 2017 curiosity-driven exploration. We use a cheap lexical proxy (action-history-distinct) instead of full RND since budget is 50k steps.",
        references=["https://arxiv.org/abs/1705.05363"],
        mutator=_s5_reward_shaping,
    ),
]


def get_seed(name: str) -> SeedHypothesis:
    for s in SEEDS:
        if s.name == name:
            return s
    raise ValueError(f"unknown seed: {name}")


def all_seed_names() -> list[str]:
    return [s.name for s in SEEDS]


def apply_seed(seed_name: str) -> str:
    """Read current candidate_train.py, apply seed mutation, write + return the new code."""
    old_code = CANDIDATE_PATH.read_text(encoding="utf-8")
    seed = get_seed(seed_name)
    new_code = seed.mutator(old_code)
    # Validate syntax before writing
    compile(new_code, "<seed>", "exec")
    return new_code


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preview", type=str, default=None)
    args = parser.parse_args()

    if args.list:
        for s in SEEDS:
            print(f"{s.name:30s} — {s.hypothesis}")
    elif args.preview:
        code = apply_seed(args.preview)
        print(code)
