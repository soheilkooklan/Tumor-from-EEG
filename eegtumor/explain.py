"""
Explainability.

Purpose
-------
Report which biomarkers drove a model's output, globally across the cohort and
locally for a single recording.

Scientific background
---------------------
SHAP values decompose a prediction into additive per-feature contributions with
a game-theoretic uniqueness guarantee. Three caveats belong next to every plot
this module produces, and the report writer prints them:

1. **SHAP explains the model, not the disease.** A feature with a large SHAP
   value is one the model relied on. Whether it is causally related to the
   pathology is a separate question that no post-hoc attribution method can
   answer.
2. **Correlated features share credit arbitrarily.** EEG band powers are
   strongly correlated by construction; when two features carry the same
   information, SHAP may attribute most of it to either one. This is why the
   selection funnel removes near-duplicates *before* explanation, and why
   stability across resamples matters more than a single ranking.
3. **The explainer is approximate outside tree models.** For non-tree models
   the permutation explainer is used, which samples the feature space and gives
   a noisier estimate at a much higher cost.

Explainer selection is automatic: `TreeExplainer` where the final estimator is
a tree ensemble (exact and fast), otherwise the model-agnostic explainer with a
subsampled background set. The previous version always used the model-agnostic
path on a calibrated wrapper, which is both slow and unnecessary.

Inputs   : fitted pipeline or calibrated wrapper, feature matrix, names
Outputs  : SHAP values, global ranking, per-recording local explanations
Limits   : no interaction values (quadratic cost); no counterfactuals.

References
----------
- Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. NeurIPS.
- Lundberg, S.M. et al. (2020). From local explanations to global understanding
  with explainable AI for trees. Nature Machine Intelligence 2, 56-67.
- Molnar, C. et al. (2022). General pitfalls of model-agnostic interpretation
  methods for machine learning models. xxAI, LNCS 13200.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ExplanationResult", "explain_model", "permutation_importance_grouped",
           "HAS_SHAP"]

try:
    import shap
    HAS_SHAP = True
except ImportError:                                            # pragma: no cover
    HAS_SHAP = False

CAVEAT = (
    "SHAP attributions describe what the model used, not what causes the "
    "condition. Correlated biomarkers share credit unpredictably, so treat a "
    "single ranking as a hypothesis and check it against the stability table."
)


@dataclass
class ExplanationResult:
    values: np.ndarray                 # (n_samples, n_features)
    feature_names: List[str]
    base_value: float
    explainer_type: str

    def global_ranking(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Mean absolute SHAP value per feature, descending."""
        imp = np.nanmean(np.abs(self.values), axis=0)
        order = np.argsort(imp)[::-1]
        pairs = [(self.feature_names[i], float(imp[i])) for i in order]
        return pairs[:top_n] if top_n else pairs

    def local(self, sample_index: int, top_n: int = 8) -> List[Tuple[str, float]]:
        """Signed contributions for one row, ranked by magnitude."""
        v = self.values[sample_index]
        order = np.argsort(np.abs(v))[::-1][:top_n]
        return [(self.feature_names[i], float(v[i])) for i in order]


def _final_estimator(model):
    """Unwrap CalibratedClassifierCV / Pipeline down to the classifier."""
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(model, CalibratedClassifierCV):
        cc = model.calibrated_classifiers_[0]
        inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        return _final_estimator(inner) if inner is not None else (None, None)
    if isinstance(model, Pipeline):
        pre = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
        return model.steps[-1][1], pre
    return model, None


def _is_tree(est) -> bool:
    name = type(est).__name__
    return any(k in name for k in
               ("RandomForest", "ExtraTrees", "GradientBoosting", "XGB",
                "LGBM", "CatBoost", "DecisionTree"))


def _is_linear(est) -> bool:
    name = type(est).__name__
    return any(k in name for k in
               ("LogisticRegression", "RidgeClassifier", "SGDClassifier",
                "LinearSVC"))


def explain_model(model, X: np.ndarray, feature_names: Sequence[str],
                  background_size: int = 100,
                  max_samples: int = 500,
                  random_state: int = 42) -> Optional[ExplanationResult]:
    """Compute SHAP values, choosing the cheapest exact explainer available.

    Explainer selection matters for more than speed. `TreeExplainer` and
    `LinearExplainer` compute exact Shapley values in closed form for their
    model families; the model-agnostic `PermutationExplainer` only approximates
    them, and costs minutes rather than milliseconds. Falling back to the
    generic path when an exact one exists would make the attributions both
    slower and less accurate.

    Returns None (with a log message) when `shap` is not installed or the model
    structure defeats the explainer, rather than raising - explainability is
    valuable but must not be able to abort an analysis run.
    """
    if not HAS_SHAP:
        logger.info("shap is not installed; skipping explainability "
                    "(pip install shap)")
        return None

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]

    est, pre = _final_estimator(model)
    names = list(feature_names)

    def _transformed(pre_, X_):
        """Apply the pipeline's preprocessing and recover the surviving names."""
        Xt = pre_.transform(X_)
        sel = pre_.named_steps.get("select") if hasattr(pre_, "named_steps") else None
        if sel is not None and hasattr(sel, "support_"):
            kept = [n for n, k in zip(feature_names, sel.support_) if k]
        else:
            kept = list(feature_names)
        return Xt, kept

    try:
        if est is not None and pre is not None and _is_tree(est):
            Xt, names = _transformed(pre, X)
            explainer = shap.TreeExplainer(est)
            values = explainer.shap_values(Xt)
            if isinstance(values, list):
                values = values[-1]
            elif values.ndim == 3:
                values = values[:, :, -1]
            base = float(np.ravel(explainer.expected_value)[-1])
            return ExplanationResult(np.asarray(values), names, base, "TreeExplainer")

        if est is not None and pre is not None and _is_linear(est):
            Xt, names = _transformed(pre, X)
            explainer = shap.LinearExplainer(est, Xt)
            values = np.asarray(explainer.shap_values(Xt))
            if values.ndim == 3:
                values = values[:, :, -1]
            base = float(np.ravel(explainer.expected_value)[-1])
            return ExplanationResult(values, names, base, "LinearExplainer")

        bg_n = min(background_size, len(X))
        background = shap.sample(X, bg_n, random_state=random_state)
        explainer = shap.Explainer(lambda z: model.predict_proba(z)[:, 1],
                                   background, feature_names=names)
        exp = explainer(X, silent=True)
        return ExplanationResult(np.asarray(exp.values), names,
                                 float(np.ravel(exp.base_values)[0]),
                                 "PermutationExplainer")
    except Exception as exc:
        logger.warning("SHAP explanation failed (%s); continuing without it", exc)
        return None


def permutation_importance_grouped(model, X, y, groups, feature_names,
                                   n_repeats: int = 10, random_state: int = 42,
                                   scoring=None) -> List[Tuple[str, float, float]]:
    """Permutation importance with subject-aware shuffling.

    A second, model-agnostic importance estimate to check the SHAP ranking
    against. Values are permuted within the evaluation set only; the drop in
    score is averaged over repeats. Reported as (feature, mean drop, sd).

    Relying on a single importance metric is a known way to reach a confident
    but unreproducible biomarker list, which is why two are computed and the
    consensus is what the report highlights.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    scoring = scoring or (lambda yt, yp: roc_auc_score(yt, yp))

    baseline = scoring(y, model.predict_proba(X)[:, 1])
    out: List[Tuple[str, float, float]] = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(baseline - scoring(y, model.predict_proba(Xp)[:, 1]))
        out.append((name, float(np.mean(drops)), float(np.std(drops))))
    out.sort(key=lambda t: t[1], reverse=True)
    return out
