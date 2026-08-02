"""
Negative-control tests for the validation protocol.

Why these exist
---------------
The defect that motivated version 2 was not a crash. It was a pipeline that ran
cleanly and reported ROC-AUC around 0.96 on data containing no signal at all,
because epochs and channels from one subject were split across training and
test folds. Nothing in an ordinary test suite catches that: the code was
correct, the science was not.

These are the guard. They build a cohort in which the label is *known* to be
independent of the signal and assert that the pipeline reports chance
performance. If a future change reintroduces grouping leakage, they fail
immediately.

A second group of tests asserts structurally that every data-dependent step -
imputation, scaling, feature selection - lives inside the Pipeline object, so
that a cross-validator refits it per fold without any caller having to remember
to do so.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from eegtumor.config import AnalysisConfig, ConfigError, SelectionConfig, ValidationConfig
from eegtumor.modeling import build_pipeline
from eegtumor.selection import FeatureFunnel, selection_stability

N_SUBJECTS = 30
ROWS_PER_SUBJECT = 12
N_FEATURES = 25


def make_null_cohort(seed: int = 0):
    """Subject-structured features whose labels carry no information.

    Each subject gets an idiosyncratic offset and scale shared by all of its
    rows - the tabular analogue of one person's montage, impedance and spectral
    fingerprint. Labels are drawn independently of everything else, so the only
    way to score above chance is to recognise the subject.
    """
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for s in range(N_SUBJECTS):
        offset = rng.normal(0, 3, size=N_FEATURES)
        scale = rng.uniform(0.5, 2.0, size=N_FEATURES)
        label = int(rng.integers(0, 2))
        rows = offset + scale * rng.normal(0, 0.3, size=(ROWS_PER_SUBJECT, N_FEATURES))
        X.append(rows)
        y.append(np.full(ROWS_PER_SUBJECT, label))
        groups.append(np.full(ROWS_PER_SUBJECT, s))
    return np.vstack(X), np.concatenate(y), np.concatenate(groups)


# ---------------------------------------------------------------------------
# The headline regression tests
# ---------------------------------------------------------------------------

def test_ungrouped_split_inflates_auc_on_pure_noise():
    """Reproduces the v1 failure mode.

    Kept as a test so the synthetic cohort is verified to actually contain the
    subject structure that makes the guard below meaningful. If this stops
    holding, the guard is testing nothing.
    """
    X, y, groups = make_null_cohort()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
    assert auc > 0.85, (
        f"expected the ungrouped protocol to be strongly optimistic, got {auc:.3f}")


def test_grouped_split_returns_chance_on_pure_noise():
    """The guard: subject-disjoint CV must not beat chance on label-free data."""
    X, y, groups = make_null_cohort()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="roc_auc").mean()
    assert 0.30 < auc < 0.70, (
        f"subject-disjoint CV returned AUC {auc:.3f} on label-free data; "
        f"grouping leakage has been reintroduced")


def test_full_pipeline_returns_chance_on_pure_noise():
    """Same guard through the project's own pipeline builder, so it also covers
    imputation, scaling and the selection funnel."""
    X, y, groups = make_null_cohort(seed=3)
    cfg = AnalysisConfig(selection=SelectionConfig(max_features=10))
    names = [f"f{i}" for i in range(N_FEATURES)]
    pipe = build_pipeline("RandomForest", cfg, feature_names=names)
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=1)
    auc = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="roc_auc").mean()
    assert 0.25 < auc < 0.75, (
        f"pipeline scored AUC {auc:.3f} on label-free data - it is leaking")


def test_pipeline_recovers_a_real_effect():
    """Complement to the null tests: the guard must not be so strict that it
    also suppresses genuine signal."""
    X, y, groups = make_null_cohort(seed=11)
    # The synthetic subjects carry a between-subject offset with SD 3, so the
    # injected effect has to be large relative to that to be detectable across
    # only 30 people - which is itself a useful reminder of how much subject
    # variability a real EEG study has to overcome.
    X[:, 0] += 8.0 * y
    cfg = AnalysisConfig(selection=SelectionConfig(max_features=10))
    pipe = build_pipeline("RandomForest", cfg,
                          feature_names=[f"f{i}" for i in range(N_FEATURES)])
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=1)
    auc = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="roc_auc").mean()
    assert auc > 0.80, f"failed to detect a strong real effect (AUC {auc:.3f})"


# ---------------------------------------------------------------------------
# Fold-internal fitting
# ---------------------------------------------------------------------------

def test_selection_depends_on_the_labels_it_is_fitted_to():
    """Selection must be a fitted step. Two training sets with different
    outcomes must produce different feature subsets; if they do not, selection
    is not really happening inside the fold."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(240, N_FEATURES))
    y_a = (X[:, 0] + 0.2 * rng.normal(size=240) > 0).astype(int)
    y_b = (X[:, 7] + 0.2 * rng.normal(size=240) > 0).astype(int)
    names = [f"f{i}" for i in range(N_FEATURES)]

    a = FeatureFunnel(max_features=5, feature_names=names, random_state=0).fit(X, y_a)
    b = FeatureFunnel(max_features=5, feature_names=names, random_state=0).fit(X, y_b)
    sel_a = set(np.flatnonzero(a.support_))
    sel_b = set(np.flatnonzero(b.support_))

    assert 0 in sel_a, "informative feature f0 was not selected for outcome A"
    assert 7 in sel_b, "informative feature f7 was not selected for outcome B"
    assert sel_a != sel_b, "selection ignored the labels"


def test_every_data_dependent_step_is_inside_the_pipeline():
    """Structural check. Imputation, scaling and selection must all be Pipeline
    steps preceding the classifier, so cross-validation refits them per fold."""
    cfg = AnalysisConfig()
    pipe = build_pipeline("ElasticNetLogistic", cfg,
                          feature_names=[f"f{i}" for i in range(N_FEATURES)])
    assert isinstance(pipe, Pipeline)
    steps = [name for name, _ in pipe.steps]
    for required in ("impute", "scale", "select"):
        assert required in steps, f"'{required}' is not a pipeline step: {steps}"
        assert steps.index(required) < steps.index("clf"), \
            f"'{required}' must come before the classifier"


def test_funnel_never_returns_an_empty_matrix():
    """A fold in which every feature is filtered out must degrade gracefully
    rather than crash the whole run."""
    X = np.zeros((40, N_FEATURES))           # zero variance everywhere
    y = np.array([0, 1] * 20)
    f = FeatureFunnel().fit(X, y)
    assert f.transform(X).shape[1] >= 1


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_stability_resamples_over_subjects():
    """A genuinely informative feature should survive resampling far more often
    than the noise features around it."""
    X, y, groups = make_null_cohort(seed=8)
    X[:, 0] += 2.5 * y
    names = [f"f{i}" for i in range(N_FEATURES)]
    cfg = SelectionConfig(max_features=5, stability_repeats=10)
    report = selection_stability(X, y, groups, names, cfg, random_state=0)

    assert report.n_repeats > 0
    freq = dict(zip(report.feature_names, report.frequency))
    assert freq["f0"] >= 0.7, \
        f"strongly informative feature selected only {freq['f0']:.0%} of the time"
    assert all(0.0 <= v <= 1.0 for v in freq.values())
    assert "f0" in report.stable(threshold=0.6)


# ---------------------------------------------------------------------------
# Configuration cannot be talked out of grouping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["none", "random", "", "epoch"])
def test_grouping_cannot_be_disabled(bad):
    with pytest.raises(ConfigError):
        AnalysisConfig(validation=ValidationConfig(grouping=bad))
