"""Reward bridge: wrap our 3-judge panel as a ROLL LLMJudgeRewardWorker.

Lets the ROLL RLVR / agentic pipeline use the same judge ensemble we've
validated in R4 Dangerous V2 (Krippendorff alpha 0.750 on 2-judge, 100% parse
rate) as the reward signal for further RL training.
"""
