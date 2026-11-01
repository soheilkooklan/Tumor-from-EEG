"""
Validation protocol and inferential statistics.

Purpose
-------
Produce a performance estimate that is not a description of the training data,
and quantify how uncertain that estimate is.

Scientific background
---------------------
Three properties make an estimate defensible in a biomedical journal:

*Subject-disjoint splitting.* Epochs and channels from one person are not
independent observations. `StratifiedGroupKFold` keyed on `subject_id`
guarantees that no subject contributes to both the training and the test side
of any fold. The module refuses to run without a group vector.

*Nested cross-validation.* Selecting a model family and its hyper-parameters
using the same folds that report the score is selection-on-test, and biases the
result upward - by several AUC points in realistic simulations (Varma & Simon,
2006; Cawley & Talbot, 2010). The inner loop tunes; the outer loop, which the
tuner never sees, reports.

*Uncertainty, not a point estimate.* A single AUC from a single split of
sixty subjects is close to meaningless. Repeated outer cross-validation gives
fold-to-fold spread, the bootstrap gives a confidence interval, and a
label-permutation test gives the null distribution against which "better than
chance" actually means something.

Metrics are computed at the level of the *recording*, because that is the level
at which a screening decision would be made, and include the ones an imbalanced
clinical problem requires: balanced accuracy, PR-AUC, MCC, sensitivity at a
fixed specificity, and Brier score with a calibration curve.

Inputs   : row-level feature matrix + index (subject, recording, epoch, channel)
Outputs  : `ValidationResult` with per-fold and pooled metrics
Limits   : does not implement external/temporal validation, which no internal
           resampling scheme can substitute for.

References
----------
- Varma, S., Simon, R. (2006). Bias in error estimation when using
  cross-validation for model selection. BMC Bioinformatics 7, 91.
- Cawley, G.C., Talbot, N.L.C. (2010). On over-fitting in model selection and
  subsequent selection bias in performance evaluation. JMLR 11, 2079-2107.
- DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Comparing the areas
  under two or more correlated ROC curves. Biometrics 44(3), 837-845.
- Ojala, M., Garriga, G.C. (2010). Permutation tests for studying classifier
  performance. JMLR 11, 1833-1863.
- Collins, G.S. et al. (2024). TRIPOD+AI statement. BMJ 385, e078378.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold

from .config import AnalysisConfig
from .modeling import MODEL_SPECS, aggregate_scores, build_pipeline, calibrate

logger = logging.getLogger(__name__)

__all__ = [
    "single_feature_baseline",
    "ValidationResult", "nested_cross_validate", "compute_metrics",
    "bootstrap_ci", "delong_test", "mcnemar_test", "permutation_test",
    "sensitivity_at_specificity", "split_groups", "permutation_null",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sensitivity_at_specificity(y_true, y_score, specificity: float = 0.90) -> float:
    """Sensitivity achievable while holding specificity at least at the target.

    Reported because a screening tool is used at a fixed operating point, not
    at every threshold simultaneously, and AUC hides how the trade-off behaves
    in the region anyone would actually use.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= (1.0 - specificity)
    return float(np.max(tpr[ok])) if ok.any() else float("nan")


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    """Full metric panel at recording level."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    out: Dict[str, float] = {}

    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in
                ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity",
                 "specificity", "ppv", "npv", "f1", "mcc", "brier",
                 "sens_at_spec90", "n", "prevalence")}

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["pr_auc"] = float(average_precision_score(y_true, y_score))
    out["sens_at_spec90"] = sensitivity_at_specificity(y_true, y_score, 0.90)
    out["brier"] = float(brier_score_loss(y_true, y_score))

    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["ppv"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    out["npv"] = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    out["f1"] = float(f1_score(y_true, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, pred))
    out["n"] = float(len(y_true))
    out["prevalence"] = float(y_true.mean())
    return out


# ---------------------------------------------------------------------------
# Uncertainty and comparison
# ---------------------------------------------------------------------------

def _auc(y_true, y_score) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties (same as roc_auc_score)."""
    from scipy.stats import rankdata
    pos = y_true == 1
    n1 = int(pos.sum())
    n0 = y_true.size - n1
    ranks = rankdata(y_score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _balanced_accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    pred = y_score >= threshold
    pos = y_true == 1
    return float(0.5 * (pred[pos].mean() + (~pred[~pos]).mean()))


_FAST_METRICS = {
    "roc_auc": _auc,
    "pr_auc": lambda y, s: float(average_precision_score(y, s)),
    "balanced_accuracy": _balanced_accuracy,
}


def bootstrap_ci(y_true, y_score, metric: str = "roc_auc",
                 n_iterations: int = 2000, alpha: float = 0.05,
                 random_state: int = 42,
                 groups: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Percentile bootstrap interval.

    When `groups` is given the resampling is done over groups (subjects), which
    is the correct unit: bootstrapping rows from correlated recordings produces
    intervals that are far too narrow. Only the requested metric is computed
    per resample.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    fn = _FAST_METRICS.get(metric)
    if fn is None:
        fn = lambda y, s: compute_metrics(y, s)[metric]
    point = fn(y_true, y_score)

    stats: List[float] = []
    if groups is None:
        n = len(y_true)
        for _ in range(n_iterations):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            stats.append(fn(y_true[idx], y_score[idx]))
    else:
        groups = np.asarray(groups)
        unique, inverse = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(unique.size)]
        for _ in range(n_iterations):
            drawn = rng.integers(0, unique.size, unique.size)
            idx = np.concatenate([members[k] for k in drawn])
            if len(np.unique(y_true[idx])) < 2:
                continue
            stats.append(fn(y_true[idx], y_score[idx]))

    if len(stats) < 50:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def _delong_placements(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    # midrank-based structural components
    v01 = np.array([(np.sum(neg < p) + 0.5 * np.sum(neg == p)) / n for p in pos])
    v10 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
    return v01, v10, m, n


def delong_test(y_true, score_a, score_b) -> Dict[str, float]:
    """DeLong test for two correlated ROC curves on the same samples.

    Returns the AUC difference, its standard error, z and a two-sided p-value.
    This is the standard way to answer "is model A really better than model B",
    and it accounts for the correlation induced by evaluating both on the same
    subjects - which a naive two-sample test does not.
    """
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "difference": float("nan"), "se": float("nan"),
                "z": float("nan"), "p_value": float("nan")}

    va01, va10, m, n = _delong_placements(y_true, a)
    vb01, vb10, _, _ = _delong_placements(y_true, b)
    auc_a, auc_b = va01.mean(), vb01.mean()

    s01 = np.cov(np.vstack([va01, vb01]))
    s10 = np.cov(np.vstack([va10, vb10]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    diff = float(auc_a - auc_b)
    if se <= 0:
        return {"auc_a": float(auc_a), "auc_b": float(auc_b),
                "difference": diff, "se": 0.0, "z": float("nan"),
                "p_value": float("nan")}
    from scipy.stats import norm
    z = diff / se
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "difference": diff,
            "se": se, "z": float(z), "p_value": float(2 * (1 - norm.cdf(abs(z))))}


def mcnemar_test(y_true, pred_a, pred_b) -> Dict[str, float]:
    """McNemar's test on the discordant classifications of two models."""
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    a_ok, b_ok = a == y_true, b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "statistic": float("nan"), "p_value": 1.0}
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[0, n01], [n10, 0]], exact=(n01 + n10) < 25)
        return {"n01": n01, "n10": n10,
                "statistic": float(res.statistic), "p_value": float(res.pvalue)}
    except ImportError:                                        # pragma: no cover
        from scipy.stats import binomtest
        p = binomtest(n01, n01 + n10, 0.5).pvalue
        return {"n01": n01, "n10": n10, "statistic": float("nan"), "p_value": float(p)}


def permutation_test(fit_predict_fn, y_true, groups, n_permutations: int = 200,
                     random_state: int = 42) -> Dict[str, float]:
    """Label-permutation null for the whole pipeline.

    Labels are permuted *at the group level*, so the permuted data keeps the
    within-subject correlation structure of the real data. This is what turns
    the test into a check on the pipeline rather than on the labels alone: if a
    leaky pipeline scores highly on permuted labels, the null distribution
    reveals it.
    """
    rng = np.random.default_rng(random_state)
    observed = fit_predict_fn(y_true)

    groups = np.asarray(groups)
    unique = np.unique(groups)
    group_label = {g: y_true[groups == g][0] for g in unique}

    null: List[float] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation([group_label[g] for g in unique])
        mapping = dict(zip(unique, shuffled))
        y_perm = np.array([mapping[g] for g in groups])
        if len(np.unique(y_perm)) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            null.append(fit_predict_fn(y_perm))

    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"observed": float(observed), "null_mean": float("nan"),
                "p_value": float("nan"), "n_permutations": 0}
    p = (np.sum(null >= observed) + 1) / (null.size + 1)
    return {"observed": float(observed), "null_mean": float(np.mean(null)),
            "null_std": float(np.std(null)), "p_value": float(p),
            "n_permutations": int(null.size),
            "null": [round(float(v), 6) for v in null]}


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------

def split_groups(index: Sequence[dict], grouping: str = "subject") -> np.ndarray:
    """Group vector used for every split.

    "recording" is accepted only when no subject has more than one recording,
    in which case it is the same partition as "subject". With repeated
    recordings it would put one person on both sides of a fold, so it is
    refused rather than quietly honoured.
    """
    subjects = np.array([r["subject_id"] for r in index])
    if grouping == "subject":
        return subjects
    if grouping == "recording":
        recordings = np.array([r["recording_id"] for r in index])
        per_subject = {}
        for s_, r_ in zip(subjects, recordings):
            per_subject.setdefault(s_, set()).add(r_)
        repeated = [s_ for s_, recs in per_subject.items() if len(recs) > 1]
        if repeated:
            raise ValueError(
                f"grouping='recording' with {len(repeated)} subject(s) holding more "
                f"than one recording (e.g. {repeated[0]}); this would split a "
                f"subject across folds. Use grouping='subject'.")
        return recordings
    raise ValueError(f"unknown grouping '{grouping}'")


@dataclass
class ValidationResult:
    model_name: str
    fold_metrics: List[Dict[str, float]]
    pooled_metrics: Dict[str, float]
    pooled_scores: np.ndarray
    pooled_labels: np.ndarray
    pooled_recordings: np.ndarray
    pooled_subjects: np.ndarray
    best_params: List[dict] = field(default_factory=list)
    confidence_intervals: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        vals = [f[metric] for f in self.fold_metrics if np.isfinite(f.get(metric, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    def std(self, metric: str) -> float:
        vals = [f[metric] for f in self.fold_metrics if np.isfinite(f.get(metric, np.nan))]
        return float(np.std(vals)) if vals else float("nan")

    def report_line(self, metric: str = "roc_auc") -> str:
        ci = self.confidence_intervals.get(metric)
        base = f"{self.model_name}: {metric} {self.mean(metric):.3f} +/- {self.std(metric):.3f}"
        if ci and np.isfinite(ci[1]):
            base += f"  [95% CI {ci[1]:.3f}-{ci[2]:.3f}]"
        return base


def _tune(model_name: str, cfg: AnalysisConfig, X, y, groups,
          feature_names, seed: int, memory=None) -> dict:
    """Inner-loop hyper-parameter search. Returns the best parameter dict."""
    spec = MODEL_SPECS[model_name]
    if cfg.validation.optimisation == "none":
        return {}

    inner_folds = int(min(cfg.validation.inner_folds,
                          len(np.unique(groups)),
                          np.bincount(y)[np.bincount(y) > 0].min()))
    if inner_folds < 2:
        return {}
    inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    def score_params(params: dict) -> float:
        scores = []
        for tr, va in inner.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            pipe = build_pipeline(model_name, cfg, feature_names, params, seed,
                                  memory=memory)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X[tr], y[tr])
                p = pipe.predict_proba(X[va])[:, 1]
            scores.append(roc_auc_score(y[va], p))
        return float(np.mean(scores)) if scores else 0.0

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:                                        # pragma: no cover
        logger.warning("optuna not installed; falling back to default params")
        return {}
    if cfg.validation.optimisation == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: score_params(spec.search_space(t)),
                   n_trials=cfg.validation.n_trials, show_progress_bar=False)
    return study.best_params


def nested_cross_validate(X: np.ndarray, index: Sequence[dict],
                          cfg: AnalysisConfig,
                          feature_names: Sequence[str],
                          model_names: Optional[Sequence[str]] = None,
                          progress=None,
                          intervals: bool = True) -> Dict[str, ValidationResult]:
    """Repeated, subject-disjoint, nested cross-validation.

    Parameters
    ----------
    X : (n_rows, n_features) row-level feature matrix
    index : one dict per row with subject_id, recording_id, epoch, channel, label
    progress : optional callable(fraction, message) for GUI feedback

    Returns one `ValidationResult` per model. The reported metrics are
    recording-level, obtained by aggregating out-of-fold epoch predictions;
    no model ever sees a recording from a subject in its own test fold.
    """
    X = np.asarray(X, dtype=np.float64)
    y_row = np.array([int(r["label"]) for r in index])
    groups_row = split_groups(index, cfg.validation.grouping)
    model_names = list(model_names or MODEL_SPECS.keys())

    n_subjects = len(np.unique(groups_row))
    subj_label = {g: y_row[groups_row == g][0] for g in np.unique(groups_row)}
    min_class = min(np.bincount(list(subj_label.values())))
    outer_folds = int(min(cfg.validation.outer_folds, n_subjects, max(min_class, 2)))
    if outer_folds < 2:
        raise ValueError(
            f"cannot cross-validate: {n_subjects} subjects, smallest class has "
            f"{min_class}. At least 2 subjects per class are required, and "
            f"anything below ~20 per class will not support a usable estimate.")
    if outer_folds < cfg.validation.outer_folds:
        logger.warning("reduced outer folds to %d (limited by cohort size)", outer_folds)

    results: Dict[str, ValidationResult] = {}
    total_steps = len(model_names) * cfg.validation.n_repeats * outer_folds
    step = 0

    import shutil
    import tempfile
    from joblib import Memory
    cache_dir = tempfile.mkdtemp(prefix="eegtumor_cache_")
    memory = Memory(cache_dir, verbose=0)
    try:
        results = _nested_loop(X, index, cfg, feature_names, model_names,
                               y_row, groups_row, outer_folds, total_steps,
                               progress, intervals, memory)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return results


def _nested_loop(X, index, cfg, feature_names, model_names, y_row, groups_row,
                 outer_folds, total_steps, progress, intervals, memory):
    results: Dict[str, ValidationResult] = {}
    step = 0

    for model_name in model_names:
        fold_metrics: List[Dict[str, float]] = []
        best_params: List[dict] = []
        all_scores, all_labels, all_recs, all_subjs = [], [], [], []

        for repeat in range(cfg.validation.n_repeats):
            seed = cfg.validation.random_state + 1000 * repeat
            outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True,
                                         random_state=seed)
            for fold, (tr, te) in enumerate(outer.split(X, y_row, groups_row)):
                step += 1
                if progress:
                    progress(step / total_steps,
                             f"{model_name}  repeat {repeat + 1}/{cfg.validation.n_repeats}"
                             f"  fold {fold + 1}/{outer_folds}")

                # The split must be subject-disjoint. Not an assert: python -O
                # would strip it.
                overlap = set(groups_row[tr]) & set(groups_row[te])
                if overlap:
                    raise RuntimeError(f"subject leakage in fold: {sorted(overlap)[:5]}")

                if len(np.unique(y_row[tr])) < 2 or len(np.unique(y_row[te])) < 2:
                    continue

                params = _tune(model_name, cfg, X[tr], y_row[tr], groups_row[tr],
                               feature_names, seed + fold, memory)
                best_params.append(params)

                pipe = build_pipeline(model_name, cfg, feature_names, params,
                                      seed + fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = calibrate(pipe, X[tr], y_row[tr],
                                      cfg.validation.calibration,
                                      groups=groups_row[tr],
                                      random_state=seed + fold)
                    row_scores = model.predict_proba(X[te])[:, 1]

                agg = aggregate_scores(row_scores, [index[i] for i in te],
                                       cfg.validation.aggregation)
                if agg.labels is None or len(np.unique(agg.labels)) < 2:
                    continue

                fold_metrics.append(compute_metrics(agg.labels, agg.scores))
                all_scores.append(agg.scores)
                all_labels.append(agg.labels)
                all_recs.append(agg.recording_ids)
                all_subjs.append(agg.subject_ids)

        if not fold_metrics:
            logger.warning("%s: no usable folds", model_name)
            continue

        pooled_scores = np.concatenate(all_scores)
        pooled_labels = np.concatenate(all_labels)
        pooled_subjs = np.concatenate(all_subjs)
        pooled = compute_metrics(pooled_labels, pooled_scores)

        cis = {}
        for metric in (("roc_auc", "pr_auc", "balanced_accuracy") if intervals else ()):
            cis[metric] = bootstrap_ci(
                pooled_labels, pooled_scores, metric,
                cfg.validation.bootstrap_iterations, cfg.validation.alpha,
                cfg.validation.random_state, groups=pooled_subjs)

        results[model_name] = ValidationResult(
            model_name=model_name,
            fold_metrics=fold_metrics,
            pooled_metrics=pooled,
            pooled_scores=pooled_scores,
            pooled_labels=pooled_labels,
            pooled_recordings=np.concatenate(all_recs),
            pooled_subjects=pooled_subjs,
            best_params=best_params,
            confidence_intervals=cis,
        )
        logger.info(results[model_name].report_line())

    return results


def permutation_null(X: np.ndarray, index: Sequence[dict], cfg: AnalysisConfig,
                     feature_names: Sequence[str], model_name: str,
                     n_permutations: int = 200,
                     random_state: int = 42) -> Dict[str, float]:
    """Subject-level permutation null for one model family, end to end.

    Each permutation reruns the outer subject-disjoint cross-validation with
    selection and calibration inside it. Hyper-parameter search is switched off
    and one repeat is used, for the observed score as well as the permuted
    ones, so both sides of the comparison come from the same procedure. The
    observed value here is therefore not the headline score of the full run.
    """
    import copy
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model '{model_name}'")
    fast = copy.deepcopy(cfg)
    fast.validation.optimisation = "none"
    fast.validation.n_repeats = 1
    groups = split_groups(index, cfg.validation.grouping)
    y = np.array([int(r["label"]) for r in index])

    def score(y_new) -> float:
        relabelled = [dict(r, label=int(v)) for r, v in zip(index, y_new)]
        res = nested_cross_validate(X, relabelled, fast, feature_names, [model_name],
                                    intervals=False)
        return res[model_name].mean("roc_auc") if model_name in res else float("nan")

    level = logger.level
    logger.setLevel(logging.WARNING)       # one INFO line per permutation is noise
    try:
        out = permutation_test(score, y, groups, n_permutations, random_state)
    finally:
        logger.setLevel(level)
    out["model"] = model_name
    return out



def single_feature_baseline(X: np.ndarray, index: Sequence[dict],
                            feature_names: Sequence[str], feature: str,
                            n_iterations: int = 2000, alpha: float = 0.05,
                            random_state: int = 42) -> Optional[dict]:
    """ROC-AUC of one named feature used directly as the score, no learning.

    Each recording gets the median of the feature over its rows. Nothing is
    fitted and the sign is not chosen from the data, so the number carries no
    optimism; an AUC below 0.5 means the feature is lower in the positive
    class. A trained model that does not beat this has learned nothing the
    feature did not already say.
    """
    if feature not in feature_names:
        return None
    col = np.asarray(X, dtype=float)[:, list(feature_names).index(feature)]
    rec = np.array([r["recording_id"] for r in index])
    keep = [i for i, r in enumerate(index) if r.get("label") not in (None, "")]
    rec_ids = list(dict.fromkeys(rec[keep]))
    first = {}
    for i in keep:
        first.setdefault(rec[i], i)
    scores = np.array([np.nanmedian(col[rec == r]) for r in rec_ids])
    labels = np.array([int(index[first[r]]["label"]) for r in rec_ids])
    subj = np.array([index[first[r]]["subject_id"] for r in rec_ids])
    ok = np.isfinite(scores)
    if ok.sum() < 4 or len(np.unique(labels[ok])) < 2:
        return None
    point, lo, hi = bootstrap_ci(labels[ok], scores[ok], "roc_auc", n_iterations,
                                 alpha, random_state, groups=subj[ok])
    return {"feature": feature, "roc_auc": float(point), "ci_low": float(lo),
            "ci_high": float(hi), "n_recordings": int(ok.sum())}
