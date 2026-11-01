"""The no-learning single-feature baseline."""

import numpy as np

from eegtumor.config import AnalysisConfig
from eegtumor.validation import single_feature_baseline


def _table(shift, n_subj=30, rows=12, seed=0):
    rng = np.random.default_rng(seed)
    X, index = [], []
    for s in range(n_subj):
        y = s % 2
        level = rng.normal() + shift * y
        for r in range(rows):
            X.append([level + 0.3 * rng.normal(), rng.normal()])
            index.append({"subject_id": f"S{s}", "recording_id": f"S{s}_r0", "label": y})
    return np.array(X), index


def test_baseline_finds_a_real_shift_and_keeps_its_sign():
    X, index = _table(shift=2.0)
    up = single_feature_baseline(X, index, ["f", "noise"], "f")
    assert up["roc_auc"] > 0.85
    assert up["ci_low"] < up["roc_auc"] < up["ci_high"]
    assert up["n_recordings"] == 30
    X[:, 0] *= -1
    down = single_feature_baseline(X, index, ["f", "noise"], "f")
    assert abs(down["roc_auc"] - (1 - up["roc_auc"])) < 1e-9


def test_baseline_on_noise_is_near_chance():
    X, index = _table(shift=0.0, n_subj=200, seed=3)
    r = single_feature_baseline(X, index, ["f", "noise"], "noise")
    assert 0.35 < r["roc_auc"] < 0.65
    assert r["ci_low"] < 0.5 < r["ci_high"]


def test_baseline_skips_missing_feature_and_unlabelled_rows():
    X, index = _table(shift=1.0)
    assert single_feature_baseline(X, index, ["f", "noise"], "absent") is None
    for r in index[:24]:
        r["label"] = None
    r = single_feature_baseline(X, index, ["f", "noise"], "f")
    assert r["n_recordings"] == 28


def test_baseline_feature_is_in_the_config():
    cfg = AnalysisConfig()
    assert cfg.validation.baseline_feature == "sp_ratio_theta_alpha"
    cfg.validation.baseline_feature = None
    cfg.validate()
