"""
Configuration objects for the Tumor-from-EEG analysis pipeline.

Purpose
-------
A single, serialisable description of every analysis choice, so that a run can
be reproduced from the config file alone. Nothing downstream is allowed to hold
its own hardcoded constants.

Scientific background
---------------------
Two design rules follow from reproducibility requirements in biomedical ML
(Poldrack et al., 2020; Kapoor & Narayanan, 2023):

1. Analysis parameters are data, not code. They are versioned, hashed and
   written next to the results.
2. Parameter combinations that are internally inconsistent must fail loudly
   rather than silently producing meaningless numbers. `validate()` enforces
   this - most importantly, it refuses to let you request power in a frequency
   band that your own filter has already removed.

Inputs   : optional YAML file
Outputs  : an `AnalysisConfig` instance; `to_dict()` for provenance logging
Limits   : does not validate that the config is *appropriate* for your data,
           only that it is self-consistent.

References
----------
- Poldrack, R.A., Huckins, G., Varoquaux, G. (2020). Establishment of best
  practices for evidence for prediction. JAMA Psychiatry 77(5), 534-540.
- Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in
  machine-learning-based science. Patterns 4(9), 100804.
- Nolan, H., Whelan, R., Reilly, R.B. (2010). FASTER: fully automated
  statistical thresholding for EEG artifact rejection. J Neurosci Methods.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

__all__ = [
    "PreprocessingConfig",
    "BandConfig",
    "FeatureConfig",
    "SelectionConfig",
    "ValidationConfig",
    "AnalysisConfig",
    "ConfigError",
]


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Signal conditioning applied before any feature is computed.

    `target_sampling_rate` exists because feature values are not comparable
    across recordings sampled at different rates: spectral resolution, entropy
    embedding scale and fractal-dimension k-ranges all depend on fs. Every
    recording is resampled to a common rate first. If two diagnostic classes
    happen to come from two different acquisition systems, skipping this step
    lets the classifier learn the scanner rather than the pathology.
    """

    target_sampling_rate: float = 250.0

    # Band-pass. 0.5-45 Hz keeps the gamma band inside the passband; a 1-30 Hz
    # passband combined with a 30-45 Hz "gamma" feature measures filter roll-off.
    highpass: float = 0.5
    lowpass: float = 45.0
    filter_order: int = 4            # per-direction; sosfiltfilt doubles it
    filter_design: str = "butter"    # butter | cheby2

    # Mains notch. Applied BEFORE the low-pass, otherwise it is a no-op.
    notch_freq: Optional[float] = 50.0     # 50 EU/Asia, 60 US, None to disable
    notch_quality: float = 30.0
    notch_harmonics: int = 1               # also notch 2f0, 3f0 ... if in band

    # Optional wavelet denoising. Off by default: soft thresholding is a
    # non-linear operation that changes entropy and fractal statistics in a way
    # that is hard to characterise, so it should be an explicit choice.
    wavelet_denoise: bool = False
    wavelet: str = "db4"
    wavelet_level: int = 4

    # Amplitude handling. "none" preserves absolute microvolt information,
    # which carries real signal for focal slowing. "zscore" is offered for
    # compatibility with the v1 behaviour but makes several time-domain
    # features analytically constant - the extractor drops them automatically.
    amplitude_normalization: str = "none"   # none | zscore | robust

    # Epoching, expressed in SECONDS so it means the same thing at every fs.
    epoch_seconds: float = 8.0
    epoch_overlap: float = 0.0              # 0.0-0.95
    max_epochs_per_recording: Optional[int] = 60

    # Automatic epoch rejection
    reject_flat_ratio: float = 0.30
    reject_clipping_ratio: float = 0.02
    reject_amplitude_uv: Optional[float] = 500.0   # peak-to-peak, None disables
    reject_muscle_ratio: float = 0.60              # frac of power above 30 Hz
    min_good_epochs: int = 3

    # Referencing
    reference: str = "average"   # none | average

    def validate(self) -> None:
        if self.target_sampling_rate <= 0:
            raise ConfigError("target_sampling_rate must be positive")
        nyq = 0.5 * self.target_sampling_rate
        if not (0 < self.highpass < self.lowpass < nyq):
            raise ConfigError(
                f"need 0 < highpass ({self.highpass}) < lowpass ({self.lowpass}) "
                f"< Nyquist ({nyq}) at fs={self.target_sampling_rate} Hz"
            )
        if self.amplitude_normalization not in {"none", "zscore", "robust"}:
            raise ConfigError("amplitude_normalization must be none|zscore|robust")
        if self.reference not in {"none", "average"}:
            raise ConfigError("reference must be none|average")
        if not (0.0 <= self.epoch_overlap < 1.0):
            raise ConfigError("epoch_overlap must be in [0, 1)")
        if self.epoch_seconds <= 0:
            raise ConfigError("epoch_seconds must be positive")
        if self.filter_design not in {"butter", "cheby2"}:
            raise ConfigError("filter_design must be butter|cheby2")


# ---------------------------------------------------------------------------
# Frequency bands
# ---------------------------------------------------------------------------

@dataclass
class BandConfig:
    """Canonical EEG frequency bands.

    Boundaries are conventional rather than universal; they are exposed here so
    a study can state exactly which definition it used, which is the part that
    matters for reproducibility. The band ratios below are not arbitrary: focal
    polymorphic delta and theta slowing is the classical, long-documented scalp
    EEG correlate of an underlying structural lesion, so a slowing index is the
    most physiologically defensible single marker available for this task.
    """

    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    })

    ratios: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = field(
        default_factory=lambda: {
            # (numerator bands, denominator bands)
            "theta_alpha": (("theta",), ("alpha",)),
            "delta_alpha": (("delta",), ("alpha",)),
            "slowing_index": (("delta", "theta"), ("alpha", "beta")),
            "theta_beta": (("theta",), ("beta",)),
        }
    )

    def validate(self, pre: PreprocessingConfig) -> None:
        for name, (lo, hi) in self.bands.items():
            if lo >= hi:
                raise ConfigError(f"band '{name}': low edge must be below high edge")
            if lo < pre.highpass or hi > pre.lowpass:
                raise ConfigError(
                    f"band '{name}' = ({lo}, {hi}) Hz lies partly outside the "
                    f"filter passband ({pre.highpass}-{pre.lowpass} Hz). Power "
                    f"measured there would be filter roll-off, not brain "
                    f"activity. Widen the passband or drop the band."
                )
        for rname, (num, den) in self.ratios.items():
            for b in tuple(num) + tuple(den):
                if b not in self.bands:
                    raise ConfigError(f"ratio '{rname}' references unknown band '{b}'")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Which feature families to compute, and their parameters."""

    domains: List[str] = field(default_factory=lambda: [
        "time", "spectral", "time_frequency", "entropy", "complexity",
    ])

    # Spectral estimation. "multitaper" (DPSS) has lower variance than a single
    # Welch taper at equal spectral resolution and is the current default in
    # most clinical EEG spectral work; Welch is kept for comparison.
    psd_method: str = "multitaper"      # welch | multitaper
    psd_window_seconds: float = 4.0
    psd_overlap: float = 0.5
    n_tapers: int = 5
    aperiodic_range: Tuple[float, float] = (2.0, 40.0)

    # Time-frequency
    tf_wavelet: str = "db4"
    tf_level: int = 5

    # Entropy
    sampen_m: int = 2
    sampen_r: float = 0.2              # x SD of the epoch
    permutation_order: int = 3
    permutation_delay: int = 1
    multiscale_scales: Tuple[int, ...] = (1, 2)   # scale s needs >= ~750 points after coarse-graining

    # Complexity
    higuchi_kmax: int = 10
    dfa_min_scale: int = 8
    dfa_max_scale_frac: float = 0.1    # of epoch length

    def validate(self) -> None:
        known = {"time", "spectral", "time_frequency", "entropy", "complexity"}
        unknown = set(self.domains) - known
        if unknown:
            raise ConfigError(f"unknown feature domains: {sorted(unknown)}")
        if self.psd_method not in {"welch", "multitaper"}:
            raise ConfigError("psd_method must be welch|multitaper")
        if self.permutation_order < 2:
            raise ConfigError("permutation_order must be >= 2")


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionConfig:
    """Multi-stage feature selection. Every stage runs INSIDE the training fold."""

    enabled: bool = True
    drop_zero_variance: bool = True
    variance_threshold: float = 1e-10
    correlation_threshold: Optional[float] = 0.95
    mutual_information_keep: Optional[int] = None   # None = keep all
    embedded_method: Optional[str] = "elasticnet"   # elasticnet | rf | None
    max_features: Optional[int] = 30
    stability_repeats: int = 20
    stability_threshold: float = 0.6                 # selection frequency


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Resampling protocol.

    `grouping` is the single most consequential setting in this file. EEG
    epochs and channels drawn from one recording are not independent samples;
    splitting them at random across folds lets a model recognise the subject
    instead of the pathology and inflates every reported metric. The default is
    subject-disjoint and there is deliberately no option to disable it silently.
    """

    grouping: str = "subject"          # subject | recording  (never "none")
    outer_folds: int = 5
    inner_folds: int = 3
    n_repeats: int = 5
    random_state: int = 42

    optimisation: str = "optuna"       # optuna | random | none
    n_trials: int = 40
    scoring: str = "roc_auc"

    calibration: Optional[str] = "sigmoid"   # sigmoid | isotonic | None
    aggregation: str = "trimmed_mean"        # mean | median | trimmed_mean | max

    bootstrap_iterations: int = 2000
    permutation_tests: int = 200             # label-permutation null
    alpha: float = 0.05

    def validate(self) -> None:
        if self.grouping not in {"subject", "recording"}:
            raise ConfigError(
                "grouping must be 'subject' or 'recording'. Ungrouped splitting "
                "of within-recording epochs is a known source of optimistic bias "
                "and is not supported."
            )
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ConfigError("outer_folds and inner_folds must be >= 2")
        if self.calibration not in {None, "sigmoid", "isotonic"}:
            raise ConfigError("calibration must be sigmoid|isotonic|None")


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    bands: BandConfig = field(default_factory=BandConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    task_name: str = "abnormality-screening"
    positive_class_label: str = "positive"
    negative_class_label: str = "negative"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.preprocessing.validate()
        self.bands.validate(self.preprocessing)
        self.features.validate()
        self.validation.validate()

        # An epoch must be long enough for the requested wavelet depth and for
        # the PSD window, otherwise the feature vector silently changes length.
        n = int(self.preprocessing.epoch_seconds * self.preprocessing.target_sampling_rate)
        need_psd = int(self.features.psd_window_seconds * self.preprocessing.target_sampling_rate)
        if need_psd > n:
            raise ConfigError(
                f"psd_window_seconds ({self.features.psd_window_seconds}s) exceeds "
                f"epoch_seconds ({self.preprocessing.epoch_seconds}s)"
            )
        try:
            import pywt
            max_level = pywt.dwt_max_level(n, pywt.Wavelet(self.features.tf_wavelet).dec_len)
        except ImportError:                                   # pragma: no cover
            max_level = self.features.tf_level
        if self.features.tf_level > max_level:
            raise ConfigError(
                f"tf_level={self.features.tf_level} is not achievable for an "
                f"{n}-sample epoch with wavelet '{self.features.tf_wavelet}' "
                f"(max {max_level}). Lengthen epoch_seconds or lower tf_level. "
                f"Clamping it silently would make the feature vector length "
                f"depend on the recording."
            )

    # -- provenance ---------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash of the whole configuration, for run directories."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # -- (de)serialisation --------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        def build(klass, key):
            sub = data.get(key, {}) or {}
            return klass(**sub)

        cfg = cls(
            preprocessing=build(PreprocessingConfig, "preprocessing"),
            bands=build(BandConfig, "bands"),
            features=build(FeatureConfig, "features"),
            selection=build(SelectionConfig, "selection"),
            validation=build(ValidationConfig, "validation"),
            task_name=data.get("task_name", "abnormality-screening"),
            positive_class_label=data.get("positive_class_label", "positive"),
            negative_class_label=data.get("negative_class_label", "negative"),
        )
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "AnalysisConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # tuples survive a YAML round-trip as lists; restore where needed
        bands = raw.get("bands", {}).get("bands")
        if bands:
            raw["bands"]["bands"] = {k: tuple(v) for k, v in bands.items()}
        ratios = raw.get("bands", {}).get("ratios")
        if ratios:
            raw["bands"]["ratios"] = {
                k: (tuple(v[0]), tuple(v[1])) for k, v in ratios.items()
            }
        feats = raw.get("features", {})
        if "aperiodic_range" in feats:
            feats["aperiodic_range"] = tuple(feats["aperiodic_range"])
        if "multiscale_scales" in feats:
            feats["multiscale_scales"] = tuple(feats["multiscale_scales"])
        return cls.from_dict(raw)

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
