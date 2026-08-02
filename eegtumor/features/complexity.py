"""
Nonlinear and complexity biomarkers.

An honest caveat belongs at the top of this module. Fractal dimensions, Hurst
exponents and detrended fluctuation analysis all estimate *scaling behaviour*,
and band-pass filtering deliberately destroys scaling behaviour outside the
passband. Computing a Hurst exponent on data high-passed at 0.5 Hz measures a
mixture of the brain and the filter. These features are therefore reported here
as exploratory, and any result that rests primarily on them should be treated
with more suspicion than one resting on band power.

They are retained because they are extensively reported in the EEG literature,
because they are cheap, and because they do sometimes separate classes that
spectral features do not - but the interpretation is weaker than the spectral
family's and this module says so rather than implying otherwise.

Every estimator below is validated in `tests/test_features.py` against signals
with known answers: white noise (Higuchi FD ~2.0, DFA alpha ~0.5), a pure sine
(FD ~1.0) and Brownian motion (FD ~1.5, DFA alpha ~1.5).

References
----------
- Higuchi, T. (1988). Approach to an irregular time series on the basis of the
  fractal theory. Physica D 31(2), 277-283.
- Katz, M.J. (1988). Fractals and the analysis of waveforms. Computers in
  Biology and Medicine 18(3), 145-156.
- Petrosian, A. (1995). Kolmogorov complexity of finite sequences and
  recognition of different preictal EEG patterns. IEEE CBMS.
- Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides. Physical
  Review E 49(2), 1685-1689.
- Lempel, A., Ziv, J. (1976). On the complexity of finite sequences. IEEE
  Transactions on Information Theory 22(1), 75-81.
- Esteller, R. et al. (2001). A comparison of waveform fractal dimension
  algorithms. IEEE Trans. Circuits and Systems 48(2), 177-183.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np

from . import FeatureGroup, FeatureSpec, register_group

DOMAIN = "complexity"


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension.

    Curve length L(k) is computed for a range of sampling intervals k; the
    fractal dimension is minus the slope of log L(k) against log k.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    kmax = max(2, min(kmax, n // 4))
    lengths = []
    ks = []
    for k in range(1, kmax + 1):
        lk = []
        for m in range(k):
            n_max = (n - m - 1) // k
            if n_max < 1:
                continue
            idx = m + np.arange(n_max + 1) * k
            seg = np.sum(np.abs(np.diff(x[idx])))
            lk.append(seg * (n - 1) / (n_max * k * k))
        if lk:
            lengths.append(np.mean(lk))
            ks.append(k)
    if len(lengths) < 3:
        return float("nan")
    lengths = np.asarray(lengths)
    if np.any(lengths <= 0):
        return float("nan")
    slope = np.polyfit(np.log(ks), np.log(lengths), 1)[0]
    return float(-slope)


def katz_fd(x: np.ndarray) -> float:
    """Katz fractal dimension. Cheap, but sensitive to amplitude scaling."""
    x = np.asarray(x, dtype=np.float64)
    d = np.abs(x - x[0]).max()
    L = np.sum(np.abs(np.diff(x)))
    n = len(x) - 1
    if L <= 0 or d <= 0:
        return float("nan")
    return float(np.log10(n) / (np.log10(d / (L / n)) + np.log10(n)))


def petrosian_fd(x: np.ndarray) -> float:
    """Petrosian fractal dimension from the number of derivative sign changes."""
    d = np.diff(x)
    n_delta = int(np.count_nonzero(np.diff(np.sign(d))))
    n = len(x)
    if n_delta == 0:
        return float("nan")
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))


def detrended_fluctuation(x: np.ndarray, min_scale: int = 8,
                          max_scale: int = None, n_scales: int = 12) -> float:
    """DFA scaling exponent alpha (Peng et al., 1994).

    alpha = 0.5 uncorrelated noise, 0.5 < alpha < 1 long-range correlated,
    alpha ~ 1 pink noise, alpha > 1 non-stationary, alpha ~ 1.5 Brownian.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    max_scale = max_scale or n // 4
    if max_scale <= min_scale or n < 4 * min_scale:
        return float("nan")

    y = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale),
                                   n_scales).astype(int))
    scales = scales[scales >= 4]
    if len(scales) < 4:
        return float("nan")

    fluct = []
    used = []
    for s in scales:
        n_seg = n // s
        if n_seg < 2:
            continue
        segs = y[:n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        # Least-squares detrend of each segment
        coef = np.polyfit(t, segs.T, 1)
        trend = np.outer(coef[0], t) + coef[1][:, None]
        rms = np.sqrt(np.mean((segs - trend) ** 2, axis=1))
        f = np.sqrt(np.mean(rms ** 2))
        if f > 0:
            fluct.append(f)
            used.append(s)
    if len(fluct) < 4:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])


def hurst_rs(x: np.ndarray, min_scale: int = 16) -> float:
    """Hurst exponent by rescaled-range (R/S) analysis.

    Reported alongside DFA because the two disagree in the presence of trends,
    and the disagreement itself is informative about non-stationarity.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4 * min_scale:
        return float("nan")
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(n // 4), 10).astype(int))
    rs_vals, used = [], []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        vals = []
        for i in range(n_seg):
            seg = x[i * s:(i + 1) * s]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std()
            if S > 0:
                vals.append(R / S)
        if vals:
            rs_vals.append(np.mean(vals))
            used.append(s)
    if len(rs_vals) < 4:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(rs_vals), 1)[0])


def lempel_ziv_complexity(x: np.ndarray, normalise: bool = True) -> float:
    """LZ76 complexity of the median-binarised signal.

    Counts distinct substrings encountered when scanning left to right. The
    median threshold makes it amplitude-invariant. Normalised by n/log2(n),
    the asymptotic value for a random binary sequence, so ~1.0 means
    random-like and lower values mean more repetitive.
    """
    b = (np.asarray(x) > np.median(x)).astype(np.uint8)
    s = b.tobytes()
    n = len(s)
    if n < 2:
        return float("nan")
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    if not normalise:
        return float(c)
    return float(c * np.log2(n) / n)


# ---------------------------------------------------------------------------
# Registry interface
# ---------------------------------------------------------------------------

_NAMES = ["cx_higuchi_fd", "cx_katz_fd", "cx_petrosian_fd",
          "cx_dfa_alpha", "cx_hurst_rs", "cx_lempel_ziv"]


def names(fcfg, bands) -> List[str]:
    return list(_NAMES)


def docs(fcfg, bands) -> Dict[str, FeatureSpec]:
    caveat = (" Interpret with caution: band-pass filtering removes the scaling "
              "behaviour this measure is designed to quantify outside the "
              "passband, so part of the value reflects the filter.")
    return {
        "cx_higuchi_fd": FeatureSpec(
            name="cx_higuchi_fd", domain=DOMAIN, unit="dimension",
            description=f"Higuchi fractal dimension, kmax={fcfg.higuchi_kmax}.",
            interpretation=("Waveform irregularity between 1 (smooth curve) and "
                            "2 (space-filling). The best-validated of the "
                            "waveform fractal dimension estimators in "
                            "head-to-head comparisons." + caveat),
            references=("Higuchi 1988", "Esteller et al. 2001"),
            complexity="O(n * kmax)"),
        "cx_katz_fd": FeatureSpec(
            name="cx_katz_fd", domain=DOMAIN, unit="dimension",
            description="Katz fractal dimension.",
            interpretation=("Cheap alternative to Higuchi. Known to be sensitive "
                            "to amplitude scaling and to sample rate, so it is "
                            "reported mainly for comparability with older "
                            "literature." + caveat),
            references=("Katz 1988",), complexity="O(n)"),
        "cx_petrosian_fd": FeatureSpec(
            name="cx_petrosian_fd", domain=DOMAIN, unit="dimension",
            description="Petrosian fractal dimension.",
            interpretation=("Derived from derivative sign changes, so it is close "
                            "to a rescaled zero-crossing rate; expect it to "
                            "correlate strongly with the time-domain ZCR and to "
                            "be removed by correlation filtering." + caveat),
            references=("Petrosian 1995",), complexity="O(n)"),
        "cx_dfa_alpha": FeatureSpec(
            name="cx_dfa_alpha", domain=DOMAIN, unit="exponent",
            description="Detrended fluctuation analysis scaling exponent.",
            interpretation=("Long-range temporal correlation. 0.5 = uncorrelated, "
                            "~1.0 = pink noise, >1 = non-stationary. Altered "
                            "long-range correlations are reported in several "
                            "neurological conditions." + caveat),
            references=("Peng et al. 1994",), complexity="O(n log n)"),
        "cx_hurst_rs": FeatureSpec(
            name="cx_hurst_rs", domain=DOMAIN, unit="exponent",
            description="Hurst exponent by rescaled-range analysis.",
            interpretation=("Persistence of the series. Reported alongside DFA "
                            "because divergence between the two indicates "
                            "non-stationarity rather than genuine long memory."
                            + caveat),
            complexity="O(n log n)"),
        "cx_lempel_ziv": FeatureSpec(
            name="cx_lempel_ziv", domain=DOMAIN, unit="normalised",
            description="Normalised LZ76 complexity of the median-binarised signal.",
            interpretation=("Algorithmic compressibility. ~1 means random-like, "
                            "lower means repetitive. Amplitude-invariant by "
                            "construction and one of the more robust complexity "
                            "measures at short epoch lengths."),
            references=("Lempel & Ziv 1976",), complexity="O(n)"),
    }


def compute(x: np.ndarray, fs: float, fcfg, bands) -> "OrderedDict[str, float]":
    n = len(x)
    max_scale = max(int(fcfg.dfa_max_scale_frac * n), fcfg.dfa_min_scale * 2)
    return OrderedDict([
        ("cx_higuchi_fd", higuchi_fd(x, fcfg.higuchi_kmax)),
        ("cx_katz_fd", katz_fd(x)),
        ("cx_petrosian_fd", petrosian_fd(x)),
        ("cx_dfa_alpha", detrended_fluctuation(x, fcfg.dfa_min_scale, max_scale)),
        ("cx_hurst_rs", hurst_rs(x)),
        ("cx_lempel_ziv", lempel_ziv_complexity(x)),
    ])


register_group(FeatureGroup(name="complexity", domain=DOMAIN,
                            names_fn=names, compute_fn=compute, docs_fn=docs))
