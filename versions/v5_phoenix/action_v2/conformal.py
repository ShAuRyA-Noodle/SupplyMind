"""conformal.py — split-conformal action filter.

Given a calibration set of (state, expert_action, predicted_logits),
computes the empirical α-quantile of the negative-log-likelihood of
the expert action under the policy. At inference, any action whose
NLL > calibrated quantile is rejected (logit set to -inf).

This implements **conformal action filtering**: actions that the
policy is too uncertain about (vs the calibration distribution) are
suppressed even if they're argmax. The result is a *risk-aware*
policy with formal coverage guarantees:

  P[expert_action ∈ accepted_set] >= 1 - α

This pairs cleanly with the hierarchical-intent layer — first the
intent narrows actions to a strategy-compatible subset, then conformal
narrows again to actions the policy is confident about.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ConformalActionFilter:
    """Wraps a policy with a calibrated NLL-quantile threshold."""
    nll_quantile: float                  # calibrated α-quantile of expert NLL
    alpha: float = 0.1                    # 90% nominal coverage
    n_calibration: int = 0
    n_actions: int = 280

    def filter_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask actions whose NLL exceeds the calibrated quantile.

        Action a's NLL = -log_softmax(logits)[a]. We accept actions with
        NLL <= self.nll_quantile.
        """
        log_probs = F.log_softmax(logits, dim=-1)        # (..., n_actions)
        nll = -log_probs                                  # (..., n_actions)
        accept_mask = nll <= self.nll_quantile             # (..., n_actions)
        # Ensure we always accept at least one action (the argmax)
        if accept_mask.dim() == 1:
            if not accept_mask.any():
                accept_mask[logits.argmax()] = True
        else:
            for i in range(accept_mask.size(0)):
                if not accept_mask[i].any():
                    accept_mask[i, logits[i].argmax()] = True
        return logits.masked_fill(~accept_mask, float("-inf"))

    def to_dict(self) -> dict:
        return {
            "nll_quantile": float(self.nll_quantile),
            "alpha": float(self.alpha),
            "n_calibration": int(self.n_calibration),
            "n_actions": int(self.n_actions),
            "expected_coverage": 1.0 - self.alpha,
            "method": "split_conformal_nll",
        }


def calibrate_conformal(
    calibration_logits: torch.Tensor,    # (N, n_actions)
    calibration_actions: torch.Tensor,   # (N,)
    alpha: float = 0.1,
) -> ConformalActionFilter:
    """Split-conformal calibration.

    For each calibration example, compute the NLL of the expert action.
    Pick the (1 - alpha)-quantile, with the +1/(N+1) finite-sample
    correction (Vovk 2005).
    """
    log_probs = F.log_softmax(calibration_logits, dim=-1)         # (N, n_actions)
    expert_nll = -log_probs.gather(1, calibration_actions.unsqueeze(-1)).squeeze(-1)
    expert_nll_np = expert_nll.detach().cpu().numpy()
    n = len(expert_nll_np)
    # Conformal quantile: ceil((1 - alpha)(n+1)) / n
    q_idx = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    q_idx = min(max(q_idx, 0), n - 1)
    nll_quantile = float(np.sort(expert_nll_np)[q_idx])
    logger.info("[conformal] N=%d, alpha=%.2f, NLL quantile=%.3f",
                n, alpha, nll_quantile)
    return ConformalActionFilter(
        nll_quantile=nll_quantile,
        alpha=alpha,
        n_calibration=n,
        n_actions=int(calibration_logits.size(-1)),
    )


def smoke_test() -> dict:
    """Synthetic calibration + filter demo."""
    rng = np.random.default_rng(42)
    N = 500
    n_actions = 280
    # Synthetic policy: peaked logits where expert action is usually the max
    raw = torch.randn(N, n_actions) * 0.5
    expert = torch.tensor(rng.integers(0, n_actions, size=N), dtype=torch.long)
    # Bias the expert action's logit upward so most are "confident-correct"
    bias = torch.zeros_like(raw)
    bias.scatter_(1, expert.unsqueeze(-1), 3.0)
    logits = raw + bias

    cf = calibrate_conformal(logits, expert, alpha=0.1)

    # Test on fresh batch
    raw2 = torch.randn(8, n_actions) * 0.5
    filtered = cf.filter_logits(raw2)
    accepted = (filtered != float("-inf")).sum(dim=-1).tolist()

    return {
        "calibration_summary": cf.to_dict(),
        "test_batch_n_accepted_per_row": accepted,
        "test_batch_at_least_one_per_row": all(a >= 1 for a in accepted),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(smoke_test(), indent=2))
