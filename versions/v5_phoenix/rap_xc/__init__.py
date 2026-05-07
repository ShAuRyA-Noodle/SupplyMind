"""rap_xc — Retrieval-Augmented Policy with Crisis-Conditioned Cross-Attention.

Novel 9th leaderboard agent designed to leverage the 1500-event EMDAT
FAISS crisis library + 25-judge panel + supply-chain DAG cascade. See
docs/RAP_XC_DESIGN.md for the full architecture rationale.
"""
from .model import RAPXCPolicy, RAPXCConfig
from .train import harvest_trajectories, train_rapxc

__all__ = ["RAPXCPolicy", "RAPXCConfig", "harvest_trajectories", "train_rapxc"]
