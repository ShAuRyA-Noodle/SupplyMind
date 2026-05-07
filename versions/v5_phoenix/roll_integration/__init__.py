"""versions.v5_phoenix.roll_integration — Alibaba ROLL framework integration.

Subpackages:
  dpo_judge/       DPO fine-tuning of Qwen-2.5-3B as a calibrated supply-chain
                   risk judge, using our 26 crisis scenarios as preference pairs.
                   Has both ROLL-pipeline and standalone-trl fallback paths.
  env/             SupplyMind registered as a ROLL environment (upstream PR
                   candidate to github.com/alibaba/ROLL).
  reward_bridge/   Wrap our existing 3-judge panel (DeepSeek-R1 + Qwen-2.5-14B
                   + Mistral-Nemo) as a ROLL LLMJudgeRewardWorker.
  configs/         Hydra YAML configs for each pipeline above.
  trl_fallback/    Fallback code paths that do NOT require ROLL installation.

Design principle: every ROLL-dependent module has a trl/transformers fallback
twin so the judge-facing demos work even if ROLL install fails.
"""

__version__ = "5.0.0-ascensionism"
