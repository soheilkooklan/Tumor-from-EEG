"""
Multi-stage feature selection.

Purpose
-------
Reduce an 82-dimensional, partly redundant feature space to a compact,
interpretable subset - without letting any information from the test fold
influence which features are chosen.

Scientific background
---------------------
The critical design constraint is that selection is *part of the model*, not a
preprocessing step applied to the whole dataset. Ranking features by their
association with the label across all data and then cross-validating the
survivors is a classic optimistic-bias generator; Ambroise and McLachlan (2002)
showed it can turn a genuine 50% error rate into an apparent 0%. `FeatureFunnel`
is therefore an sklearn transformer that only ever sees training-fold data
because it lives inside a `Pipeline`.

The stages run in order of increasing cost:

    zero-variance  ->  correlation  ->  mutual information  ->  embedded model

Correlation filtering comes before any supervised stage deliberately: it is
unsupervised, so it cannot leak, and it removes the near-duplicates that make
importance rankings unstable. When two features correlate above the threshold,
the one with the *lower* marginal association is dropped, so the retained
representative is the more informative of the pair.

`selection_stability` re-runs the whole funnel on many resamples and reports how
often each feature survives. A biomarker selected in 95% of resamples is a
finding; one selected in 40% is noise, however high its importance was in the
single run that happened to be published.

Inputs   : (n_samples, n_features) training-fold data
Outputs  : boolean support mask; selection-frequency table
Limits   : univariate stages cannot see interactions; the embedded stage can,
           but only those its model family can represent.

References
----------
- Ambroise, C., McLachlan, G.J. (2002). Selection bias in gene extraction on the
  basis of microarray gene-expression data. PNAS 99(10), 6562-6566.
- Meinshausen, N., Buhlmann, P. (2010). Stability selection. J R Stat Soc B
  72(4), 417-473.
- Nogueira, S., Sechidis, K., Brown, G. (2018). On the stability of feature
  selection algorithms. JMLR 18(174), 1-54.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .config import SelectionConfig

logger = logging.getLogger(__name__)

__all__ = ["FeatureFunnel", "selection_stability", "StabilityReport"]


_SKLEARN_VERSION = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2] if part.isdigit()
)


def _elasticnet_logistic(C: float, random_state: int) -> LogisticRegression:
    """Elastic-net logistic regression that works across scikit-learn versions.

    1.8 deprecated the explicit `penalty="elasticnet"` argument in favour of
    setting `l1_ratio` alone. Constructing it conditionally keeps the project
    warning-free on 1.8+ while still working on 1.3.
    """
    common = dict(solver="saga", l1_ratio=0.5, C=C, max_iter=10000,
                  tol=1e-3, random_state=random_state)
    if _SKLEARN_VERSION >= (1, 8):
        return LogisticRegression(**common)
    return LogisticRegression(penalty="elasticnet", **common)


class FeatureFunnel(BaseEstimator, TransformerMixin):
    """Staged feature selection, safe to place inside a cross-validated pipeline.

    Parameters mirror `SelectionConfig`. `feature_names` is optional and only
    used to make the audit trail readable.
    """

    def __init__(
        self,
        drop_zero_variance: bool = True,
        variance_threshold: float = 1e-10,
        correlation_threshold: Optional[float] = 0.95,
        mutual_information_keep: Optional[int] = None,
        embedded_method: Optional[str] = "elasticnet",
        max_features: Optional[int] = 30,
        random_state: int = 42,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.drop_zero_variance = drop_zero_variance
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mutual_information_keep = mutual_information_keep
        self.embedded_method = embedded_method
        self.max_features = max_features
        self.random_state = random_state
        self.feature_names = feature_names

    # -- sklearn plumbing ---------------------------------------------------
    @classmethod
    def from_config(cls, cfg: SelectionConfig, random_state: int = 42,
                    feature_names: Optional[Sequence[str]] = None) -> "FeatureFunnel":
        return cls(
            drop_zero_variance=cfg.drop_zero_variance,
            variance_threshold=cfg.variance_threshold,
            correlation_threshold=cfg.correlation_threshold,
            mutual_information_keep=cfg.mutual_information_keep,
            embedded_method=cfg.embedded_method,
            max_features=cfg.max_features,
            random_state=random_state,
            feature_names=feature_names,
        )

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        keep = np.ones(n_features, dtype=bool)
        self.stage_log_: List[Dict[str, object]] = []

        def record(stage: str, before: np.ndarray, after: np.ndarray) -> None:
            dropped = np.where(before & ~after)[0]
            self.stage_log_.append({
                "stage": stage,
                "n_before": int(before.sum()),
                "n_after": int(after.sum()),
                "dropped": [self._name(i) for i in dropped],
            })

        # --- 1. constants and near-constants (unsupervised) ----------------
        if self.drop_zero_variance:
            before = keep.copy()
            with np.errstate(invalid="ignore"):
                var = np.nanvar(X, axis=0)
            keep &= np.nan_to_num(var, nan=0.0) > self.variance_threshold
            record("zero_variance", before, keep)

        # --- 2. redundancy (unsupervised, so it cannot leak) ---------------
        if self.correlation_threshold is not None and keep.sum() > 1:
            before = keep.copy()
            keep = self._drop_correlated(X, y, keep)
            record(f"correlation>{self.correlation_threshold}", before, keep)

        # --- 3. univariate relevance (supervised, training fold only) ------
        if (self.mutual_information_keep is not None and y is not None
                and keep.sum() > self.mutual_information_keep):
            before = keep.copy()
            idx = np.where(keep)[0]
            mi = mutual_info_classif(self._clean(X[:, idx]), y,
                                     random_state=self.random_state)
            best = idx[np.argsort(mi)[::-1][:self.mutual_information_keep]]
            keep = np.zeros(n_features, dtype=bool)
            keep[best] = True
            record("mutual_information", before, keep)

        # --- 4. embedded, multivariate -------------------------------------
        if self.embedded_method and y is not None and keep.sum() > 1:
            before = keep.copy()
            keep = self._embedded(X, y, keep)
            record(f"embedded:{self.embedded_method}", before, keep)

        # --- 5. hard cap ----------------------------------------------------
        if self.max_features is not None and keep.sum() > self.max_features:
            before = keep.copy()
            idx = np.where(keep)[0]
            imp = self._importance(X[:, idx], y)
            best = idx[np.argsort(imp)[::-1][:self.max_features]]
            keep = np.zeros(n_features, dtype=bool)
            keep[best] = True
            record(f"cap@{self.max_features}", before, keep)

        if keep.sum() == 0:                    # never hand an empty matrix on
            logger.warning("feature selection removed everything; keeping all")
            keep = np.ones(n_features, dtype=bool)

        self.support_ = keep
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.where(self.support_)[0] if indices else self.support_

    # -- internals ----------------------------------------------------------
    def _name(self, i: int) -> str:
        if self.feature_names is not None and i < len(self.feature_names):
            return str(self.feature_names[i])
        return f"f{i}"

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        """Column-median imputation. Feature extraction should not produce NaN,
        but a single pathological epoch must not abort a fold."""
        X = np.array(X, dtype=np.float64, copy=True)
        for j in range(X.shape[1]):
            col = X[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = np.median(col[~bad]) if (~bad).any() else 0.0
        return X

    def _drop_correlated(self, X, y, keep):
        idx = np.where(keep)[0]
        Xc = self._clean(X[:, idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(Xc, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)

        # Rank by marginal association so the better representative survives
        if y is not None:
            strength = np.array([
                abs(np.corrcoef(Xc[:, j], y)[0, 1]) if np.std(Xc[:, j]) > 0 else 0.0
                for j in range(Xc.shape[1])
            ])
            strength = np.nan_to_num(strength)
        else:
            strength = np.var(Xc, axis=0)

        order = np.argsort(strength)[::-1]
        survivors: List[int] = []
        for j in order:
            if all(corr[j, k] < self.correlation_threshold for k in survivors):
                survivors.append(int(j))
        out = np.zeros_like(keep)
        out[idx[survivors]] = True
        return out

    def _embedded(self, X, y, keep):
        idx = np.where(keep)[0]
        Xc = self._clean(X[:, idx])
        if self.embedded_method == "elasticnet":
            model = _elasticnet_logistic(C=0.1, random_state=self.random_state)
            model.fit(Xc, y)
            imp = np.abs(model.coef_).ravel()
            chosen = imp > 1e-8
        elif self.embedded_method == "rf":
            model = RandomForestClassifier(
                n_estimators=300, random_state=self.random_state, n_jobs=-1)
            model.fit(Xc, y)
            imp = model.feature_importances_
            chosen = imp > np.median(imp)          # keep the upper half
        else:
            raise ValueError(f"unknown embedded_method '{self.embedded_method}'")

        if not chosen.any():
            chosen = imp >= np.max(imp)
        out = np.zeros_like(keep)
        out[idx[chosen]] = True
        return out

    def _importance(self, X, y):
        model = RandomForestClassifier(n_estimators=200,
                                       random_state=self.random_state, n_jobs=-1)
        model.fit(self._clean(X), y)
        return model.feature_importances_


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    """How often each feature survived selection across resamples."""

    feature_names: List[str]
    frequency: np.ndarray
    n_repeats: int
    mean_subset_size: float
    nogueira_index: float

    def stable(self, threshold: float = 0.6) -> List[str]:
        return [n for n, f in zip(self.feature_names, self.frequency) if f >= threshold]

    def table(self) -> List[dict]:
        order = np.argsort(self.frequency)[::-1]
        return [{"rank": r + 1,
                 "feature": self.feature_names[i],
                 "selection_frequency": round(float(self.frequency[i]), 3)}
                for r, i in enumerate(order)]

    def summary(self) -> str:
        return (f"{self.n_repeats} resamples, mean subset {self.mean_subset_size:.1f} "
                f"features, Nogueira stability {self.nogueira_index:.3f} "
                f"(1.0 = identical subsets every time, 0.0 = chance)")


def _nogueira_stability(Z: np.ndarray) -> float:
    """Nogueira et al. (2018) stability index for a binary selection matrix.

    Z is (n_repeats, n_features). Fully general: corrects for subset size and
    is comparable across different numbers of features, unlike raw Jaccard.
    """
    M, d = Z.shape
    if M < 2:
        return float("nan")
    p = Z.mean(axis=0)
    kbar = Z.sum(axis=1).mean()
    numerator = np.sum(p * (1 - p) * M / (M - 1))
    denom = (kbar / d) * (1 - kbar / d) * d
    return float(1.0 - numerator / denom) if denom > 0 else float("nan")


def selection_stability(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        feature_names: Sequence[str], cfg: SelectionConfig,
                        random_state: int = 42) -> StabilityReport:
    """Repeat the funnel on subject-level bootstrap resamples.

    Resampling is done over *subjects*, not rows, for the same reason the
    cross-validation is grouped: rows from one subject are not independent, and
    row-level bootstrapping would make every subset look artificially stable.
    """
    rng = np.random.default_rng(random_state)
    unique = np.unique(groups)
    n_features = X.shape[1]
    Z = np.zeros((cfg.stability_repeats, n_features), dtype=int)

    for rep in range(cfg.stability_repeats):
        drawn = rng.choice(unique, size=max(2, int(0.8 * len(unique))), replace=False)
        rows = np.isin(groups, drawn)
        if len(np.unique(y[rows])) < 2:
            Z[rep] = Z[max(rep - 1, 0)]
            continue
        funnel = FeatureFunnel.from_config(cfg, random_state + rep, feature_names)
        funnel.fit(X[rows], y[rows])
        Z[rep] = funnel.support_.astype(int)

    return StabilityReport(
        feature_names=list(feature_names),
        frequency=Z.mean(axis=0),
        n_repeats=cfg.stability_repeats,
        mean_subset_size=float(Z.sum(axis=1).mean()),
        nogueira_index=_nogueira_stability(Z),
    )
