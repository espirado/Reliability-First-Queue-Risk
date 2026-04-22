"""Sliding-window recalibration loop.

Triggers on PSI > psi_threshold (feature drift) OR weekly ECE > ece_threshold
(output drift).  Refits the isotonic calibrator on the most recent valid
window and writes a new versioned artifact.  The previous calibrator remains
available for one-click rollback.

This file is deliberately small and free of the training-stack dependencies
so it can be ported into a Kubernetes controller or a Slurm job_submit Lua
shim without additional imports.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class TriggerPolicy:
    psi_threshold: float = 0.10
    weekly_ece_threshold: float = 0.07
    min_window_samples: int = 200


@dataclass
class RecalArtifact:
    fitted_at: float
    window_start: float
    window_end: float
    n_samples: int
    pre_ece: float
    post_ece: float
    psi: dict[str, float] = field(default_factory=dict)


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    live_hist, _ = np.histogram(live, bins=edges)
    p = (ref_hist + 1) / (ref_hist.sum() + bins)
    q = (live_hist + 1) / (live_hist.sum() + bins)
    return float(np.sum((q - p) * np.log(q / p)))


def ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    quantiles = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = 0.0, 1.0
    total = 0.0
    n = len(y_prob)
    for i in range(bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def should_recalibrate(
    reference_features: dict[str, np.ndarray],
    live_features: dict[str, np.ndarray],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: TriggerPolicy = TriggerPolicy(),
) -> tuple[bool, dict[str, float], float]:
    psis = {k: psi(reference_features[k], live_features[k]) for k in reference_features}
    weekly_ece = ece(y_true, y_prob)
    over_psi = any(v > policy.psi_threshold for v in psis.values())
    over_ece = weekly_ece > policy.weekly_ece_threshold
    return (over_psi or over_ece), psis, weekly_ece


def fit_recalibrator(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso


def persist(artifact: RecalArtifact, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(artifact.fitted_at)
    path = out_dir / f"recal_{stamp}.json"
    path.write_text(json.dumps(asdict(artifact), indent=2))
    return path


def recalibration_cycle(
    reference_features: dict[str, np.ndarray],
    live_features: dict[str, np.ndarray],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    policy: TriggerPolicy = TriggerPolicy(),
) -> Optional[Path]:
    """One cycle of the recalibration loop.  Returns the new artifact path
    if a recalibration was triggered; otherwise None."""
    fire, psis, pre = should_recalibrate(
        reference_features, live_features, y_true, y_prob, policy
    )
    if not fire or len(y_true) < policy.min_window_samples:
        return None
    iso = fit_recalibrator(y_true, y_prob)
    post = ece(y_true, iso.transform(y_prob))
    artifact = RecalArtifact(
        fitted_at=time.time(),
        window_start=float("nan"),
        window_end=float("nan"),
        n_samples=int(len(y_true)),
        pre_ece=pre,
        post_ece=post,
        psi=psis,
    )
    return persist(artifact, out_dir)
