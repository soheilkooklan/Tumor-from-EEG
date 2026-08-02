"""
Correctness tests for the feature estimators.

These are not smoke tests. Each one checks an estimator against a signal whose
answer is known analytically or by long-standing convention. An estimator that
runs without error but returns the wrong number is worse than one that crashes,
because it produces a plausible-looking result table.
"""

import math

import numpy as np
import pytest

from eegtumor.config import AnalysisConfig, ConfigError, PreprocessingConfig, BandConfig
from eegtumor.features import feature_names, extract_epoch_features, feature_dictionary
from eegtumor.features.complexity import (higuchi_fd, katz_fd, petrosian_fd,
                                          detrended_fluctuation, hurst_rs,
                                          lempel_ziv_complexity)
from eegtumor.features.entropy import sample_entropy, permutation_entropy
from eegtumor.features.spectral import multitaper_psd, compute_psd
from eegtumor.features.time_domain import hjorth

FS = 250.0
N = 4096


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(12345)
    t = np.arange(N) / FS
    return {
        "white": rng.standard_normal(N),
        "sine": np.sin(2 * np.pi * 10 * t),
        "brownian": np.cumsum(rng.standard_normal(N)),
        "constant": np.ones(N),
    }


# --- fractal dimension ------------------------------------------------------

def test_higuchi_known_values(signals):
    assert higuchi_fd(signals["white"], 10) == pytest.approx(2.0, abs=0.10)
    assert higuchi_fd(signals["sine"], 10) == pytest.approx(1.0, abs=0.15)
    assert higuchi_fd(signals["brownian"], 10) == pytest.approx(1.5, abs=0.15)


def test_fractal_dimensions_ordered(signals):
    """Every FD estimator must rank sine < brownian < white."""
    for fn in (higuchi_fd, katz_fd, petrosian_fd):
        s, b, w = fn(signals["sine"]), fn(signals["brownian"]), fn(signals["white"])
        assert s < w, f"{fn.__name__}: sine {s:.3f} not below white noise {w:.3f}"
        assert b < w, f"{fn.__name__}: brownian {b:.3f} not below white noise {w:.3f}"


# --- scaling exponents ------------------------------------------------------

def test_dfa_known_values(signals):
    assert detrended_fluctuation(signals["white"]) == pytest.approx(0.5, abs=0.10)
    assert detrended_fluctuation(signals["brownian"]) == pytest.approx(1.5, abs=0.20)


def test_hurst_known_values(signals):
    assert hurst_rs(signals["white"]) == pytest.approx(0.5, abs=0.15)
    assert hurst_rs(signals["brownian"]) > 0.8


# --- entropy ----------------------------------------------------------------

def test_permutation_entropy_bounds(signals):
    """Normalised PE lies in [0,1]: ~1 for noise, near 0 for a pure sine."""
    pe_noise = permutation_entropy(signals["white"], 3, 1)
    pe_sine = permutation_entropy(signals["sine"], 3, 1)
    assert 0.95 <= pe_noise <= 1.0
    assert pe_sine < 0.7
    assert pe_sine < pe_noise


def test_sample_entropy_ordering(signals):
    assert sample_entropy(signals["sine"]) < sample_entropy(signals["white"])


def test_sample_entropy_matches_reference():
    """Chunked/vectorised implementation must equal the textbook O(n^2) loop."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)

    def reference(x, m=2, r_factor=0.2):
        r = r_factor * np.std(x)
        n = len(x)

        def count(mm):
            tpl = np.array([x[i:i + mm] for i in range(n - m)])
            c = 0
            for i in range(len(tpl)):
                d = np.max(np.abs(tpl - tpl[i]), axis=1)
                c += int(np.sum(d <= r)) - 1
            return c

        return -np.log(count(m + 1) / count(m))

    assert sample_entropy(x) == pytest.approx(reference(x), abs=1e-9)


def test_lempel_ziv_bounds(signals):
    lz_noise = lempel_ziv_complexity(signals["white"])
    lz_sine = lempel_ziv_complexity(signals["sine"])
    assert lz_sine < lz_noise
    assert 0.0 < lz_noise <= 1.3


# --- Hjorth -----------------------------------------------------------------

def test_hjorth_sine_mobility():
    """For a pure sine at f, mobility ~ 2*pi*f/fs in radians per sample."""
    t = np.arange(N) / FS
    f = 10.0
    x = np.sin(2 * np.pi * f * t)
    _, mobility, complexity = hjorth(x)
    assert mobility == pytest.approx(2 * np.pi * f / FS, rel=0.02)
    assert complexity == pytest.approx(1.0, abs=0.05)   # sine is minimally complex


# --- spectral ---------------------------------------------------------------

def test_multitaper_parseval():
    """Integrated one-sided PSD must recover the signal variance."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2048)
    f, p = multitaper_psd(x, FS, 5)
    assert np.trapezoid(p, f) == pytest.approx(np.var(x), rel=0.10)


def test_peak_frequency_recovered():
    cfg = AnalysisConfig()
    t = np.arange(int(cfg.preprocessing.epoch_seconds * FS)) / FS
    x = np.sin(2 * np.pi * 10.0 * t)
    f, p = compute_psd(x, FS, cfg.features)
    assert f[np.argmax(p)] == pytest.approx(10.0, abs=0.5)


def test_band_powers_sum_to_total():
    """Relative band powers must sum to 1 (bands tile the passband)."""
    cfg = AnalysisConfig()
    rng = np.random.default_rng(9)
    x = rng.standard_normal(int(cfg.preprocessing.epoch_seconds * FS))
    names = feature_names(cfg)
    v = extract_epoch_features(x, FS, cfg, names)
    rel = [v[names.index(f"sp_rel_power_{b}")] for b in cfg.bands.bands]
    assert sum(rel) == pytest.approx(1.0, abs=0.02)


# --- registry contract ------------------------------------------------------

def test_feature_vector_length_is_window_independent():
    """The single most important invariant: identical width for every epoch."""
    cfg = AnalysisConfig()
    names = feature_names(cfg)
    rng = np.random.default_rng(4)
    n = int(cfg.preprocessing.epoch_seconds * FS)
    for seed in range(5):
        v = extract_epoch_features(rng.standard_normal(n), FS, cfg, names)
        assert v.shape == (len(names),)


def test_feature_names_do_not_touch_global_rng():
    np.random.seed(0)
    before = np.random.rand()
    np.random.seed(0)
    feature_names(AnalysisConfig())
    after = np.random.rand()
    assert before == after, "feature_names() perturbed the global random state"


def test_every_feature_is_documented():
    cfg = AnalysisConfig()
    rows = feature_dictionary(cfg)
    assert len(rows) == len(feature_names(cfg))
    undocumented = [r["feature"] for r in rows
                    if r.get("description", "").startswith("(undoc")]
    assert not undocumented, f"undocumented features: {undocumented}"


def test_zscore_normalisation_drops_degenerate_features():
    """Under z-scoring, amplitude features are constants and must be excluded."""
    plain = AnalysisConfig()
    zs = AnalysisConfig(preprocessing=PreprocessingConfig(amplitude_normalization="zscore"))
    dropped = set(feature_names(plain)) - set(feature_names(zs))
    assert {"td_mean", "td_std", "td_variance", "td_rms",
            "td_hjorth_activity"} <= dropped


def test_no_nan_on_realistic_signal():
    cfg = AnalysisConfig()
    names = feature_names(cfg)
    rng = np.random.default_rng(11)
    n = int(cfg.preprocessing.epoch_seconds * FS)
    t = np.arange(n) / FS
    x = 20 * np.sin(2 * np.pi * 10 * t) + 5 * rng.standard_normal(n)
    v = extract_epoch_features(x, FS, cfg, names)
    bad = [nm for nm, val in zip(names, v) if not np.isfinite(val)]
    assert not bad, f"non-finite features on clean synthetic EEG: {bad}"


# --- configuration guards ---------------------------------------------------

def test_band_outside_passband_is_rejected():
    """The v1 defect: measuring 30-45 Hz gamma through a 1-30 Hz filter."""
    with pytest.raises(ConfigError, match="outside the filter passband"):
        AnalysisConfig(preprocessing=PreprocessingConfig(highpass=1.0, lowpass=30.0))


def test_unachievable_wavelet_level_is_rejected():
    from eegtumor.config import FeatureConfig
    with pytest.raises(ConfigError, match="not achievable"):
        AnalysisConfig(
            preprocessing=PreprocessingConfig(epoch_seconds=4.0),
            features=FeatureConfig(tf_level=10),
        )


def test_ungrouped_validation_is_rejected():
    from eegtumor.config import ValidationConfig
    with pytest.raises(ConfigError, match="subject"):
        AnalysisConfig(validation=ValidationConfig(grouping="none"))
