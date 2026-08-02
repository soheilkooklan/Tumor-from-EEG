"""
Model definitions, calibration and prediction aggregation.

Purpose
-------
Provide a small set of model families that are appropriate for a tabular
biomarker matrix of this size, produce probabilities that mean what they say,
and aggregate epoch-level predictions back up to the level at which a clinical
decision would be made.

Scientific background
---------------------
*Model choice.* With a few hundred subjects and fewer than a hundred features,
regularised linear models and tree ensembles are the appropriate families.
Deep architectures are excluded not out of conservatism but because they need
orders of magnitude more labelled recordings than a study of this scale has;
they would fit the training set and learn nothing transferable. A penalised
logistic regression is included as a mandatory baseline: if a gradient-boosted
ensemble does not clearly beat it, the extra complexity is not earning its
place, and reviewers will ask.

*Calibration.* A model that outputs 0.8 should be right about 80% of the time.
Tree ensembles are systematically over-confident near the extremes, and an
uncalibrated probability presented to a user as a percentage is misleading.
Platt scaling (`sigmoid`) is the default rather than isotonic regression:
isotonic is non-parametric and needs on the order of a thousand samples before
it stops over-fitting the calibration set, which is more than this setting
usually has (Niculescu-Mizil & Caruana, 2005).

*Aggregation.* The model sees one (epoch, channel) at a time, but the question
is about a recording. Averaging probabilities is the obvious choice and the
wrong one when a lesion is focal: a real abnormality confined to two of
nineteen channels is diluted by the seventeen normal ones. Several aggregation
rules are therefore provided and the choice is an explicit, reported analysis
decision. Note that aggregating calibrated epoch probabilities does not yield a
calibrated recording probability - the recording-level score is re-calibrated
separately.

Inputs   : feature matrix, labels, group vector
Outputs  : fitted pipelines, recording-level scores
Limits   : no multi-class support; no survival or time-to-event modelling.

References
----------
- Niculescu-Mizil, A., Caruana, R. (2005). Predicting good probabilities with
  supervised learning. ICML.
- Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive
  analytics. BMC Medicine 17, 230.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models
  still outperform deep learning on typical tabular data? NeurIPS.
- Christodoulou, E. et al. (2019). A systematic review shows no performance
  benefit of machine learning over logistic regression for clinical prediction
  models. J Clin Epidemiol 110, 12-22.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .config import AnalysisConfig, SelectionConfig
from .selection import FeatureFunnel

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_SPECS", "available_models", "build_pipeline", "calibrate",
    "aggregate_scores", "AggregationResult",
]

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:                                            # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:                                            # pragma: no cover
    HAS_LGBM = False


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A model family plus the search space used for its hyper-parameters."""

    name: str
    factory: Callable[[int], object]
    search_space: Callable[[object], dict]      # (optuna trial) -> params
    rationale: str
    needs_scaling: bool = True


_SKLEARN_VERSION = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2] if part.isdigit()
)


def _lr(rs: int):
    """Elastic-net regularised logistic regression.

    scikit-learn 1.8 deprecated the explicit `penalty="elasticnet"` argument in
    favour of setting `l1_ratio` directly. Both spellings are constructed here
    so the project works unchanged across 1.3 to 1.9 without emitting a
    FutureWarning on newer versions.
    """
    common = dict(solver="saga", max_iter=20000, tol=1e-3, random_state=rs,
                  class_weight="balanced", l1_ratio=0.5)
    if _SKLEARN_VERSION >= (1, 8):
        return LogisticRegression(**common)
    return LogisticRegression(penalty="elasticnet", **common)


def _lr_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _rf(rs: int):
    return RandomForestClassifier(random_state=rs, n_jobs=-1,
                                  class_weight="balanced_subsample")


def _rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }


def _et(rs: int):
    return ExtraTreesClassifier(random_state=rs, n_jobs=-1,
                                class_weight="balanced")


def _svm(rs: int):
    return SVC(kernel="rbf", probability=True, random_state=rs,
               class_weight="balanced")


def _svm_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 1e0, log=True),
    }


def _xgb(rs: int):
    return XGBClassifier(random_state=rs, eval_metric="logloss",
                         tree_method="hist", verbosity=0, n_jobs=-1)


def _xgb_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lgbm(rs: int):
    return LGBMClassifier(random_state=rs, verbose=-1, n_jobs=-1,
                          class_weight="balanced")


def _lgbm_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


MODEL_SPECS: Dict[str, ModelSpec] = {
    "ElasticNetLogistic": ModelSpec(
        "ElasticNetLogistic", _lr, _lr_space,
        "Mandatory baseline. Regularised linear models match or beat complex "
        "alternatives on most clinical prediction tasks, and their coefficients "
        "are directly interpretable. If nothing beats this, report this."),
    "RandomForest": ModelSpec(
        "RandomForest", _rf, _rf_space,
        "Robust to feature scaling, monotone transforms and outliers; a strong "
        "default for heterogeneous tabular biomarkers.", needs_scaling=False),
    "ExtraTrees": ModelSpec(
        "ExtraTrees", _et, _rf_space,
        "Higher-variance-reduction cousin of Random Forest; often better when "
        "individual features are noisy, as EEG epoch features are.",
        needs_scaling=False),
    "SVM-RBF": ModelSpec(
        "SVM-RBF", _svm, _svm_space,
        "Kernel method that performs well in the small-n, moderate-p regime "
        "typical of biomarker studies. Requires scaling."),
}

if HAS_XGB:
    MODEL_SPECS["XGBoost"] = ModelSpec(
        "XGBoost", _xgb, _xgb_space,
        "Gradient boosting; state of the art on tabular data in controlled "
        "comparisons. Included when the dependency is available.",
        needs_scaling=False)
if HAS_LGBM:
    MODEL_SPECS["LightGBM"] = ModelSpec(
        "LightGBM", _lgbm, _lgbm_space,
        "Histogram-based boosting; similar accuracy to XGBoost at lower cost.",
        needs_scaling=False)


def available_models() -> List[str]:
    return list(MODEL_SPECS)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(model_name: str, cfg: AnalysisConfig,
                   feature_names: Optional[Sequence[str]] = None,
                   params: Optional[dict] = None,
                   random_state: Optional[int] = None) -> Pipeline:
    """Assemble imputation -> scaling -> selection -> classifier.

    Every data-dependent step is inside the pipeline, so when the pipeline is
    handed to a cross-validator each step is fitted on the training fold alone.
    This is the structural fix for the leakage described in docs/METHODS.md;
    scaling or selecting outside this object re-introduces it.
    """
    if model_name not in MODEL_SPECS:
        raise KeyError(f"unknown model '{model_name}'; available: {available_models()}")
    spec = MODEL_SPECS[model_name]
    rs = cfg.validation.random_state if random_state is None else random_state

    steps: List[Tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if spec.needs_scaling:
        # Robust scaling: EEG epoch features have heavy tails even after QC.
        steps.append(("scale", RobustScaler()))
    if cfg.selection.enabled:
        steps.append(("select", FeatureFunnel.from_config(
            cfg.selection, random_state=rs, feature_names=feature_names)))

    clf = spec.factory(rs)
    if params:
        clf.set_params(**params)
    steps.append(("clf", clf))
    return Pipeline(steps)


def calibrate(pipeline: Pipeline, X, y, method: Optional[str] = "sigmoid",
              cv: int = 3) -> object:
    """Wrap a fitted-or-unfitted pipeline in probability calibration.

    Returns the pipeline unchanged when `method` is None. `cv` folds are used
    internally by `CalibratedClassifierCV`; with very small classes it is
    reduced automatically rather than raising.
    """
    if method is None:
        pipeline.fit(X, y)
        return pipeline
    counts = np.bincount(np.asarray(y, dtype=int))
    folds = int(max(2, min(cv, counts[counts > 0].min())))
    if folds < 2:
        logger.warning("too few samples per class to calibrate; returning raw model")
        pipeline.fit(X, y)
        return pipeline
    cal = CalibratedClassifierCV(pipeline, method=method, cv=folds)
    cal.fit(X, y)
    return cal


# ---------------------------------------------------------------------------
# Aggregation: epoch/channel -> recording
# ---------------------------------------------------------------------------

@dataclass
class AggregationResult:
    recording_ids: np.ndarray
    subject_ids: np.ndarray
    scores: np.ndarray
    labels: Optional[np.ndarray]
    n_rows: np.ndarray
    dispersion: np.ndarray = field(default_factory=lambda: np.array([]))


def aggregate_scores(row_scores: np.ndarray, index: Sequence[dict],
                     method: str = "trimmed_mean",
                     trim: float = 0.1) -> AggregationResult:
    """Collapse per-(epoch, channel) probabilities to one score per recording.

    Methods
    -------
    mean          : simple average. Dilutes focal abnormality across channels.
    median        : robust to a few artefactual rows, dilutes focality further.
    trimmed_mean  : default. Drops the extreme `trim` fraction at both ends,
                    keeping robustness without discarding the whole tail.
    max           : most sensitive to focal findings and by far the most
                    sensitive to a single artefactual row. Use only with strict
                    quality control, and say so in the Methods.

    `dispersion` is the interquartile range of the row scores within each
    recording. A high value means the channels disagree, which for a focal
    lesion is expected and for a global artefact is a warning; it is reported
    rather than hidden inside the aggregate.
    """
    rec_ids = np.array([r["recording_id"] for r in index])
    subj_ids = np.array([r["subject_id"] for r in index])
    labels = np.array([r.get("label") for r in index], dtype=object)

    order = []
    out_scores, out_labels, out_subj, out_n, out_disp = [], [], [], [], []
    for rid in dict.fromkeys(rec_ids):                     # preserve first-seen order
        m = rec_ids == rid
        s = np.asarray(row_scores)[m]
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        if method == "mean":
            agg = float(np.mean(s))
        elif method == "median":
            agg = float(np.median(s))
        elif method == "max":
            agg = float(np.max(s))
        elif method == "trimmed_mean":
            k = int(np.floor(trim * s.size))
            srt = np.sort(s)
            core = srt[k:s.size - k] if s.size - 2 * k > 0 else srt
            agg = float(np.mean(core))
        else:
            raise ValueError(f"unknown aggregation '{method}'")
        order.append(rid)
        out_scores.append(agg)
        out_subj.append(subj_ids[m][0])
        lab = labels[m][0]
        out_labels.append(None if lab is None else int(lab))
        out_n.append(int(m.sum()))
        out_disp.append(float(np.subtract(*np.percentile(s, [75, 25]))))

    has_labels = all(v is not None for v in out_labels) and len(out_labels) > 0
    return AggregationResult(
        recording_ids=np.array(order),
        subject_ids=np.array(out_subj),
        scores=np.array(out_scores),
        labels=np.array(out_labels, dtype=int) if has_labels else None,
        n_rows=np.array(out_n),
        dispersion=np.array(out_disp),
    )
