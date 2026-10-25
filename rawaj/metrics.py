"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations
import numpy as np
import math

def _rules(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError('y_true and scores must be list or 1-D arrays')
    if len(y) != len(s):
        raise ValueError('y_true and scores must be of equal lengths')
    if len(y) == 0:
        raise ValueError('input is empty')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('labels must be 0 or 1')
    if np.isnan(s).any():
        raise ValueError('scores must not contain NaN')
    return y.astype(int), s

def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    
    y, s = _rules(y_true, scores)
    flagged = s >= threshold
    positive = y.astype(int) == 1
    return {
        'tp': int(np.sum(flagged & positive)), # Flagged and actually positive
        'fp': int(np.sum(flagged & ~positive)), # Flagged but actually negative
        'tn': int(np.sum(~flagged & ~positive)), # Not flagged and actually negative
        'fn': int(np.sum(~flagged & positive)), # Not flagged but actually positive
    }
    # raise NotImplementedError

def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    cnts = confusion_at(y_true, scores, threshold)

    pred_pos = cnts['tp'] + cnts['fp']
    precision = cnts['tp'] / pred_pos if pred_pos else 0.0

    act_pos = cnts['tp'] + cnts['fn']
    recall = cnts['tp'] / act_pos if act_pos else 0.0

    return float(precision), float(recall)
    # raise NotImplementedError


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.

    ValueError if k <= 0; if k > n, use n. Rank by score descending; ties keep their
    original order.
    """
    y, s = _rules(y_true, scores)
    if k <= 0:
        raise ValueError('K must be positive') 
    n = len(y)
    k = min(k, n) # if k > n, use n
    top_k_idx = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[top_k_idx] == 1))
    # raise NotImplementedError


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    cnts = confusion_at(y_true, scores, threshold)
    return float(cnts['fp'] * cost_fp + cnts['fn'] * cost_fn)
    # raise NotImplementedError


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    y, s = _rules(y_true, scores)
    candidates = list(np.unique(s)) + [math.inf] # sorted unique scores + concat of inf to candidates
    best = min(candidates, 
               key=lambda threshold: (
                   expected_cost(y, s, threshold, cost_fp=cost_fp, cost_fn=cost_fn), 
                   -threshold
                   )
            )
    return float(best)
    # raise NotImplementedError


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y, s = _rules(y_true, scores)
    for threshold in sorted(np.unique(s)):
        precision, _ = precision_recall_at(y, s, float(threshold))
        if precision >= min_precision:
            return float(threshold)
    return None
    # raise NotImplementedError
