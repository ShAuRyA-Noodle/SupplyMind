"""Per-horizon split-conformal quantile computation — CI smoke import."""
from __future__ import annotations

import numpy as np


def per_horizon_conformal_band(cal_residuals: np.ndarray, alpha: float) -> np.ndarray:
    """cal_residuals: [n_cal, H]  |y - yhat| at each horizon step per fold.
    Returns q_hat: [H]  finite-sample conformal quantile per horizon step.
    """
    n, H = cal_residuals.shape
    q_hat = np.zeros(H)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    for h in range(H):
        q_hat[h] = float(np.sort(np.abs(cal_residuals[:, h]))[k - 1])
    return q_hat
