"""Membership-inference metrics against brute force (ties included) and the analysis layout."""

from __future__ import annotations

import numpy as np

from horizon.eval.membership import analyze, roc_auc, tpr_at_fpr


def brute_auc(m, n):
    return ((m[:, None] > n[None]).sum() + 0.5 * (m[:, None] == n[None]).sum()) / (len(m) * len(n))


def brute_tpr(m, n, fpr):
    best = 0.0
    for t in np.r_[np.unique(np.r_[m, n]), -np.inf]:
        if (n > t).sum() <= fpr * len(n):
            best = max(best, float((m > t).mean()))
    return best


def test_auc_and_tpr_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(100):
        m = np.round(rng.normal(0.3, 1, rng.integers(5, 300)), 1)  # rounding forces ties
        n = np.round(rng.normal(0.0, 1, rng.integers(5, 300)), 1)
        assert abs(roc_auc(m, n) - brute_auc(m, n)) < 1e-12
        assert abs(tpr_at_fpr(m, n, 0.01) - brute_tpr(m, n, 0.01)) < 1e-12


def test_analyze_separable_and_null():
    labels = ["x", "y"]
    sel = [{"set": "member", "label": "x"}] * 50 + [{"set": "nonmember", "label": "x"}] * 50
    base = np.zeros((100, 2))
    target = base.copy()
    target[:50, 0] = 1.0  # members get a higher true-label score → perfectly separable
    r = analyze(sel, {"base": base, "E3": target}, labels, seed=0)
    assert r["E3"]["loss"]["auc"] == 1.0 and r["E3"]["reference"]["tpr_at_0.01_fpr"] == 1.0
    assert r["base_only"]["loss"]["auc"] == 0.5  # identical scores: all ties
