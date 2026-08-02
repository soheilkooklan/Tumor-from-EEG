"""
Biomarker extraction.

Purpose
-------
Compute a compact set of scientifically interpretable EEG features, each of
which carries its own documentation, so that the feature table exported for
analysis is self-describing and a Methods section can be written from it.

Design
------
Feature families register themselves through `register_group`. A group must be
able to declare the exact names it will produce **before seeing any data**
(`names(cfg, bands)`), which is what guarantees a fixed-width feature matrix.
The previous version derived names by pushing a random 512-sample signal
through the extractor; that made the vector length depend on window length and
perturbed the global NumPy random state.

Adding a new biomarker means adding one module and one `register_group` call.
No existing file changes.

Scientific selection criterion
------------------------------
A feature is included only if it is (a) repeatedly reported in clinical EEG
work, (b) physiologically interpretable, and (c) cheap enough to compute on a
whole cohort. Quantity is not a goal: a compact, non-redundant, interpretable
feature space generalises better and is far easier to defend.

For the specific question of structural brain lesions, the strongest prior
evidence is for **focal slowing** - increased delta and theta power with
reduced alpha, often lateralised. The band-power ratios in `spectral.py` are
therefore the most defensible markers in this set, and the entropy/complexity
families are exploratory by comparison.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import AnalysisConfig, BandConfig, FeatureConfig

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureSpec", "FeatureGroup", "register_group", "REGISTRY",
    "feature_names", "extract_epoch_features", "extract_recording_features",
    "feature_dictionary", "degenerate_features",
]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """Everything a reader needs to evaluate one feature."""

    name: str
    domain: str
    description: str
    interpretation: str
    references: Tuple[str, ...] = ()
    complexity: str = "O(n)"
    amplitude_dependent: bool = False
    unit: str = "a.u."

    def as_row(self) -> dict:
        return {
            "feature": self.name,
            "domain": self.domain,
            "unit": self.unit,
            "description": self.description,
            "physiological_interpretation": self.interpretation,
            "amplitude_dependent": self.amplitude_dependent,
            "computational_complexity": self.complexity,
            "references": "; ".join(self.references),
        }


@dataclass(frozen=True)
class FeatureGroup:
    """A registered family of features."""

    name: str
    domain: str
    names_fn: Callable[[FeatureConfig, BandConfig], List[str]]
    compute_fn: Callable[[np.ndarray, float, FeatureConfig, BandConfig], "OrderedDict[str, float]"]
    docs_fn: Callable[[FeatureConfig, BandConfig], Dict[str, FeatureSpec]]


REGISTRY: "OrderedDict[str, FeatureGroup]" = OrderedDict()


def register_group(group: FeatureGroup) -> None:
    if group.name in REGISTRY:
        raise ValueError(f"feature group '{group.name}' already registered")
    REGISTRY[group.name] = group


# Importing the modules triggers their registration.
from . import time_domain, spectral, time_frequency, entropy, complexity  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Amplitude degeneracy guard
# ---------------------------------------------------------------------------

def degenerate_features(cfg: AnalysisConfig) -> List[str]:
    """Features that are constant by construction under the chosen normalisation.

    Z-scoring each epoch forces mean = 0 and SD = 1, which makes the mean, SD,
    variance, RMS and Hjorth activity identical for every epoch in the dataset.
    Keeping them would put five columns of constants into the feature matrix,
    dilute every importance ranking, and waste selection-stage budget.
    """
    if cfg.preprocessing.amplitude_normalization == "none":
        return []
    dead: List[str] = []
    for group in REGISTRY.values():
        docs = group.docs_fn(cfg.features, cfg.bands)
        for name, spec in docs.items():
            if spec.amplitude_dependent:
                dead.append(name)
    return dead


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _active_groups(cfg: AnalysisConfig) -> List[FeatureGroup]:
    return [g for g in REGISTRY.values() if g.domain in cfg.features.domains]


def feature_names(cfg: AnalysisConfig) -> List[str]:
    """Ordered feature names, computed analytically from the configuration.

    Never touches data and never uses the random number generator.
    """
    dead = set(degenerate_features(cfg))
    out: List[str] = []
    for group in _active_groups(cfg):
        out.extend(n for n in group.names_fn(cfg.features, cfg.bands) if n not in dead)
    dupes = {n for n in out if out.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate feature names across groups: {sorted(dupes)}")
    return out


def feature_dictionary(cfg: AnalysisConfig) -> List[dict]:
    """Row-per-feature documentation table, for export next to the results."""
    dead = set(degenerate_features(cfg))
    rows = []
    for group in _active_groups(cfg):
        docs = group.docs_fn(cfg.features, cfg.bands)
        for name in group.names_fn(cfg.features, cfg.bands):
            if name in dead:
                continue
            spec = docs.get(name)
            rows.append(spec.as_row() if spec else
                        {"feature": name, "domain": group.domain,
                         "description": "(undocumented)"})
    return rows


def extract_epoch_features(x: np.ndarray, fs: float, cfg: AnalysisConfig,
                           names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Feature vector for one epoch of one channel.

    Returns a float array aligned with `feature_names(cfg)`. Failures produce
    NaN for the affected group rather than an exception, so one pathological
    epoch cannot abort a cohort-wide extraction; the NaN rate is reported by
    the caller and is itself a quality signal.
    """
    x = np.asarray(x, dtype=np.float64)
    dead = set(degenerate_features(cfg))
    values: "OrderedDict[str, float]" = OrderedDict()

    for group in _active_groups(cfg):
        expected = group.names_fn(cfg.features, cfg.bands)
        try:
            got = group.compute_fn(x, fs, cfg.features, cfg.bands)
            if list(got.keys()) != expected:
                raise RuntimeError(
                    f"group '{group.name}' declared {len(expected)} names but "
                    f"produced {len(got)}; declaration and computation disagree"
                )
        except Exception as exc:
            logger.debug("feature group '%s' failed: %s", group.name, exc)
            got = OrderedDict((n, float("nan")) for n in expected)
        for n, v in got.items():
            if n not in dead:
                values[n] = float(v)

    if names is None:
        names = feature_names(cfg)
    return np.array([values.get(n, float("nan")) for n in names], dtype=np.float64)


def extract_recording_features(epoched, cfg: AnalysisConfig
                               ) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Feature matrix for one preprocessed recording.

    One row per accepted (epoch, channel) pair. Every row carries the epoch
    index, channel name, recording id and subject id, because those are what
    later stages need in order to keep the split honest and to aggregate
    predictions back up to recording level.

    Returns
    -------
    X    : (n_rows, n_features)
    names: feature names
    index: list of dicts describing each row
    """
    names = feature_names(cfg)
    rows: List[np.ndarray] = []
    index: List[dict] = []

    for i in range(epoched.n_epochs):
        for c, ch in enumerate(epoched.channel_names):
            if not epoched.mask[i, c]:
                continue
            v = extract_epoch_features(epoched.epochs[i, c], epoched.sampling_rate,
                                       cfg, names)
            rows.append(v)
            index.append({
                "subject_id": epoched.subject_id,
                "recording_id": epoched.recording_id,
                "epoch": i,
                "channel": ch,
                "label": epoched.label,
            })

    X = np.vstack(rows) if rows else np.empty((0, len(names)))
    return X, names, index
