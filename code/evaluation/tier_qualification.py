"""Tier-qualification algorithm for the Reliability-First pipeline.

A trained+calibrated model is assigned one of four operational tiers based
on SLO-compliance tests.  Tiers escalate monotonically: warn < advisory <
suggest < gate.  A model qualifies for tier k iff it passes every rule at
tier k AND at every lower tier.

The rules correspond one-for-one to Section "Tier qualification algorithm"
in the paper.  Thresholds are chosen from six weeks of operational
experience on our internal Slurm cluster; operators should treat them as
configurable hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TierMetrics:
    ece: float
    mce: float
    tail_gap_070: float
    tail_gap_max: float
    false_action_rate_070: float


WARN_ECE_MAX = 0.10
ADVISORY_ECE_MAX = 0.05
ADVISORY_FAR_070_MAX = 0.20
SUGGEST_ECE_MAX = 0.05
SUGGEST_MCE_MAX = 0.15
SUGGEST_TAIL_GAP_070_MAX = 0.05
GATE_ECE_MAX = 0.05
GATE_MCE_MAX = 0.10
GATE_TAIL_GAP_MAX = 0.035
GATE_FAR_070_MAX = 0.05


def qualifies_warn(m: TierMetrics) -> bool:
    return m.ece <= WARN_ECE_MAX


def qualifies_advisory(m: TierMetrics) -> bool:
    return (
        qualifies_warn(m)
        and m.ece <= ADVISORY_ECE_MAX
        and m.false_action_rate_070 <= ADVISORY_FAR_070_MAX
    )


def qualifies_suggest(m: TierMetrics) -> bool:
    return (
        qualifies_advisory(m)
        and m.mce <= SUGGEST_MCE_MAX
        and m.tail_gap_070 <= SUGGEST_TAIL_GAP_070_MAX
    )


def qualifies_gate(m: TierMetrics) -> bool:
    return (
        qualifies_suggest(m)
        and m.ece <= GATE_ECE_MAX
        and m.mce <= GATE_MCE_MAX
        and m.tail_gap_max <= GATE_TAIL_GAP_MAX
        and m.false_action_rate_070 <= GATE_FAR_070_MAX
    )


def assign_tier(m: TierMetrics) -> str:
    for name, check in (
        ("gate", qualifies_gate),
        ("suggest", qualifies_suggest),
        ("advisory", qualifies_advisory),
        ("warn", qualifies_warn),
    ):
        if check(m):
            return name
    return "unqualified"


def tail_gaps(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)
) -> dict[float, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: dict[float, float] = {}
    for t in thresholds:
        mask = y_prob >= t
        if mask.sum() == 0:
            out[t] = 0.0
            continue
        predicted = y_prob[mask].mean()
        observed = y_true[mask].mean()
        out[t] = float(abs(predicted - observed))
    return out


def false_action_rate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.7) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = y_prob >= threshold
    if mask.sum() == 0:
        return 0.0
    return float(1.0 - y_true[mask].mean())
