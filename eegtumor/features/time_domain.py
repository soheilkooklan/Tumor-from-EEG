"""
Time-domain biomarkers.

These are the cheapest features in the set and the easiest to interpret, which
is exactly why they are worth keeping: an amplitude asymmetry between
hemispheres is something a clinician can look at and confirm, whereas a
difference in sample entropy is not.

Features flagged `amplitude_dependent=True` become analytically constant if
epochs are z-scored, and are excluded automatically in that case.

References
----------
- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology 29(3), 306-310.
- Esteller, R., Echauz, J., Tcheng, T., Litt, B., Pless, B. (2001). Line length:
  an efficient feature for seizure onset detection. Proc. IEEE EMBS.
- Kaiser, J.F. (1990). On a simple algorithm to calculate the "energy" of a
  signal. ICASSP.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
from scipy import stats

from . import FeatureGroup, FeatureSpec, register_group

DOMAIN = "time"

_NAMES: List[str] = [
    "td_mean", "td_median", "td_std", "td_variance", "td_rms",
    "td_peak_to_peak", "td_iqr", "td_mad",
    "td_skewness", "td_kurtosis",
    "td_zero_crossing_rate", "td_line_length", "td_mean_abs_diff",
    "td_hjorth_activity", "td_hjorth_mobility", "td_hjorth_complexity",
    "td_teager_energy",
]

_AMPLITUDE_DEPENDENT = {
    "td_mean", "td_median", "td_std", "td_variance", "td_rms",
    "td_peak_to_peak", "td_iqr", "td_mad", "td_line_length",
    "td_mean_abs_diff", "td_hjorth_activity", "td_teager_energy",
}

_DOC = {
    "td_mean": ("Arithmetic mean of the epoch.",
                "After DC removal this should sit near zero; a persistent offset "
                "indicates drift or a failing electrode."),
    "td_median": ("Median amplitude.",
                  "Robust centre; differs from the mean when the epoch contains "
                  "asymmetric transients such as spikes."),
    "td_std": ("Standard deviation.",
               "Overall signal magnitude. Suppressed amplitude over a region is a "
               "recognised sign of an underlying structural lesion."),
    "td_variance": ("Second central moment.",
                    "Identical information to the SD but on a power scale; equals "
                    "Hjorth activity."),
    "td_rms": ("Root mean square amplitude.",
               "Energy-equivalent amplitude; stable and widely reported."),
    "td_peak_to_peak": ("Maximum minus minimum.",
                        "Sensitive to isolated transients and to clipping."),
    "td_iqr": ("Interquartile range.",
               "Amplitude spread that ignores the extreme tails, so it is not "
               "driven by a single artefactual sample."),
    "td_mad": ("Median absolute deviation.",
               "Robust dispersion estimate; the basis of the wavelet noise scale."),
    "td_skewness": ("Third standardised moment.",
                    "Asymmetry of the amplitude distribution; sharp positive or "
                    "negative transients such as epileptiform discharges skew it."),
    "td_kurtosis": ("Fourth standardised moment (excess).",
                    "Peakedness. High values indicate rare large excursions - "
                    "spikes, movement or eye artefacts."),
    "td_zero_crossing_rate": ("Sign changes per sample.",
                              "A crude dominant-frequency proxy that needs no "
                              "spectral estimate; falls with focal slowing."),
    "td_line_length": ("Sum of absolute sample-to-sample differences.",
                       "Combines amplitude and frequency in one number; introduced "
                       "for seizure-onset detection and a strong general marker of "
                       "abnormal activity."),
    "td_mean_abs_diff": ("Line length divided by epoch length.",
                         "Duration-independent form of line length."),
    "td_hjorth_activity": ("Variance of the signal.",
                           "Hjorth's amplitude descriptor."),
    "td_hjorth_mobility": ("SD of the first derivative over SD of the signal.",
                           "Proportional to the mean frequency of the power "
                           "spectrum; decreases with delta/theta slowing."),
    "td_hjorth_complexity": ("Mobility of the derivative over mobility of the signal.",
                             "Bandwidth-like measure of how far the waveform "
                             "departs from a pure sine."),
    "td_teager_energy": ("Mean Teager-Kaiser energy operator.",
                         "Instantaneous energy weighted by frequency; sensitive to "
                         "transient high-frequency bursts at low computational cost."),
}

_REFS = {
    "td_hjorth_activity": ("Hjorth 1970",),
    "td_hjorth_mobility": ("Hjorth 1970",),
    "td_hjorth_complexity": ("Hjorth 1970",),
    "td_line_length": ("Esteller et al. 2001",),
    "td_mean_abs_diff": ("Esteller et al. 2001",),
    "td_teager_energy": ("Kaiser 1990",),
}


def names(fcfg, bands) -> List[str]:
    return list(_NAMES)


def docs(fcfg, bands) -> Dict[str, FeatureSpec]:
    out = {}
    for n in _NAMES:
        desc, interp = _DOC[n]
        out[n] = FeatureSpec(
            name=n, domain=DOMAIN, description=desc, interpretation=interp,
            references=_REFS.get(n, ()), complexity="O(n)",
            amplitude_dependent=n in _AMPLITUDE_DEPENDENT,
            unit="uV" if n in {"td_mean", "td_median", "td_std", "td_rms",
                               "td_peak_to_peak", "td_iqr", "td_mad"} else "a.u.",
        )
    return out


def hjorth(x: np.ndarray):
    """Activity, mobility, complexity (Hjorth, 1970)."""
    d1 = np.diff(x)
    d2 = np.diff(d1)
    v0, v1, v2 = np.var(x), np.var(d1), np.var(d2)
    activity = float(v0)
    mobility = float(np.sqrt(v1 / v0)) if v0 > 0 else float("nan")
    mob_d1 = float(np.sqrt(v2 / v1)) if v1 > 0 else float("nan")
    complexity = mob_d1 / mobility if mobility and np.isfinite(mobility) and mobility > 0 else float("nan")
    return activity, mobility, complexity


def teager_energy(x: np.ndarray) -> float:
    """Mean of psi[n] = x[n]^2 - x[n-1] x[n+1]."""
    if len(x) < 3:
        return float("nan")
    psi = x[1:-1] ** 2 - x[:-2] * x[2:]
    return float(np.mean(psi))


def compute(x: np.ndarray, fs: float, fcfg, bands) -> "OrderedDict[str, float]":
    n = len(x)
    d = np.diff(x)
    # Count sign changes ignoring exact zeros, which would otherwise be counted twice
    sign = np.sign(x)
    sign[sign == 0] = 1
    zcr = float(np.count_nonzero(np.diff(sign)) / n)

    activity, mobility, complexity = hjorth(x)
    q75, q25 = np.percentile(x, [75, 25])

    return OrderedDict([
        ("td_mean", float(np.mean(x))),
        ("td_median", float(np.median(x))),
        ("td_std", float(np.std(x))),
        ("td_variance", float(np.var(x))),
        ("td_rms", float(np.sqrt(np.mean(x ** 2)))),
        ("td_peak_to_peak", float(np.ptp(x))),
        ("td_iqr", float(q75 - q25)),
        ("td_mad", float(np.median(np.abs(x - np.median(x))))),
        ("td_skewness", float(stats.skew(x))),
        ("td_kurtosis", float(stats.kurtosis(x))),
        ("td_zero_crossing_rate", zcr),
        ("td_line_length", float(np.sum(np.abs(d)))),
        ("td_mean_abs_diff", float(np.mean(np.abs(d))) if d.size else float("nan")),
        ("td_hjorth_activity", activity),
        ("td_hjorth_mobility", mobility),
        ("td_hjorth_complexity", complexity),
        ("td_teager_energy", teager_energy(x)),
    ])


register_group(FeatureGroup(name="time_domain", domain=DOMAIN,
                            names_fn=names, compute_fn=compute, docs_fn=docs))
