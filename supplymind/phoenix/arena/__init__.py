"""OpenEnv Arena — judges drop in their PyTorch policy, get CI95 reward on 3 tasks.

Key entrypoints:
    runner.evaluate_policy(policy_path, tasks, n_episodes) -> ArenaResult
    leaderboard.rebuild() -> leaderboard.json
    router (FastAPI)     : POST /arena/run, GET /arena/leaderboard

The flagship judge-facing feature of Phoenix v5. Aligns with the hackathon
theme: the env is the product; judges bring their own agents.
"""
