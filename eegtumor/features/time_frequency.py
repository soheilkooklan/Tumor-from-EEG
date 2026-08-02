"""
Time-frequency biomarkers based on the discrete wavelet transform.

EEG is non-stationary: a burst of focal delta lasting two seconds inside an
eight-second epoch is diluted by any whole-epoch spectral estimate but shows up
clearly in the variance of a wavelet sub-band. That is the reason to keep a
time-frequency family alongside the spectral one rather than treating them as
redundant.

The decomposition level is fixed by configuration and validated against epoch
length at config time. Clamping it silently to whatever a particular epoch can
support - the previous behaviour - makes the feature-vector length a function
of the recording, which breaks the train/test contract.

Sub-band correspondence at fs = 250 Hz with `tf_level = 5`:

    D1  62.5-125 Hz    (above the passband, mostly residual)
    D2  31.2-62.5 Hz   gamma / EMG
    D3  15.6-31.2 Hz   beta
    D4   7.8-15.6 Hz   alpha
    D5   3.9-7.8  Hz   theta
    A5   0.0-3.9  Hz   delta

References
----------
- Mallat, S. (1989). A theory for multiresolution signal decomposition. IEEE
  Trans. PAMI 11(7), 674-693.
- Rosso, O.A. et al. (2001). Wavelet entropy: a new tool for analysis of short
  duration brain electrical signals. J Neurosci Methods 105(1), 65-75.
- Subasi, A. (2007). EEG signal classification using wavelet feature extraction
  and a mixture of expert model. Expert Systems with Applications 32, 1084-1093.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
from scipy import stats

from . import FeatureGroup, FeatureSpec, register_group

DOMAIN = "time_frequency"


def _band_labels(level: int) -> List[str]:
    return [f"A{level}"] + [f"D{level - i}" for i in range(level)]


def names(fcfg, bands) -> List[str]:
    labels = _band_labels(fcfg.tf_level)
    out: List[str] = []
    for lab in labels:
        out += [f"tf_energy_{lab}", f"tf_relenergy_{lab}",
                f"tf_std_{lab}", f"tf_kurtosis_{lab}"]
    out += ["tf_wavelet_entropy", "tf_energy_concentration"]
    return out


def docs(fcfg, bands) -> Dict[str, FeatureSpec]:
    fs_hint = "at the configured sampling rate"
    out: Dict[str, FeatureSpec] = {}
    for lab in _band_labels(fcfg.tf_level):
        kind = "approximation (lowest frequencies)" if lab.startswith("A") \
            else f"detail level {lab[1:]}"
        out[f"tf_energy_{lab}"] = FeatureSpec(
            name=f"tf_energy_{lab}", domain=DOMAIN, unit="uV^2",
            description=f"Sum of squared {lab} coefficients ({kind}).",
            interpretation=(
                f"Energy carried by the {lab} sub-band {fs_hint}. The "
                f"approximation band tracks delta activity, which is the "
                f"sub-band of primary interest for structural lesions."),
            references=("Mallat 1989", "Subasi 2007"),
            complexity="O(n)", amplitude_dependent=True)
        out[f"tf_relenergy_{lab}"] = FeatureSpec(
            name=f"tf_relenergy_{lab}", domain=DOMAIN, unit="fraction",
            description=f"{lab} energy as a fraction of total wavelet energy.",
            interpretation=("Scale-resolved analogue of relative band power; "
                            "comparable across recordings with different overall "
                            "amplitude."),
            references=("Rosso et al. 2001",),
            complexity="O(n)", amplitude_dependent=False)
        out[f"tf_std_{lab}"] = FeatureSpec(
            name=f"tf_std_{lab}", domain=DOMAIN, unit="uV",
            description=f"Standard deviation of {lab} coefficients.",
            interpretation="Amplitude dispersion within the sub-band.",
            complexity="O(n)", amplitude_dependent=True)
        out[f"tf_kurtosis_{lab}"] = FeatureSpec(
            name=f"tf_kurtosis_{lab}", domain=DOMAIN, unit="a.u.",
            description=f"Excess kurtosis of {lab} coefficients.",
            interpretation=("High values mean the sub-band energy arrives in "
                            "short bursts rather than continuously - the "
                            "signature of transient events such as sharp waves "
                            "or intermittent focal slowing."),
            complexity="O(n)", amplitude_dependent=False)

    out["tf_wavelet_entropy"] = FeatureSpec(
        name="tf_wavelet_entropy", domain=DOMAIN, unit="bits",
        description="Shannon entropy of the relative sub-band energy distribution.",
        interpretation=("Low when energy concentrates in one scale (a strong, "
                        "well-organised rhythm), high when it spreads evenly "
                        "(disorganised or noise-like background)."),
        references=("Rosso et al. 2001",), complexity="O(n)")
    out["tf_energy_concentration"] = FeatureSpec(
        name="tf_energy_concentration", domain=DOMAIN, unit="fraction",
        description="Largest single sub-band's share of total energy.",
        interpretation=("Simple dominance measure; complements wavelet entropy "
                        "and is easier to interpret."),
        complexity="O(n)")
    return out


def compute(x: np.ndarray, fs: float, fcfg, bands) -> "OrderedDict[str, float]":
    import pywt

    level = fcfg.tf_level
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(fcfg.tf_wavelet).dec_len)
    if level > max_level:
        # Config validation should have caught this; refuse rather than clamp,
        # because clamping changes the feature-vector length.
        raise ValueError(
            f"epoch of {len(x)} samples supports at most level {max_level}, "
            f"config requests {level}")

    coeffs = pywt.wavedec(x, fcfg.tf_wavelet, level=level, mode="periodization")
    labels = _band_labels(level)
    energies = np.array([float(np.sum(c ** 2)) for c in coeffs])
    total = energies.sum()
    rel = energies / total if total > 0 else np.full_like(energies, np.nan)

    out: "OrderedDict[str, float]" = OrderedDict()
    for lab, c, e, r in zip(labels, coeffs, energies, rel):
        out[f"tf_energy_{lab}"] = float(e)
        out[f"tf_relenergy_{lab}"] = float(r)
        out[f"tf_std_{lab}"] = float(np.std(c))
        out[f"tf_kurtosis_{lab}"] = float(stats.kurtosis(c)) if len(c) > 3 else float("nan")

    pos = rel[np.isfinite(rel) & (rel > 0)]
    out["tf_wavelet_entropy"] = float(-np.sum(pos * np.log2(pos))) if pos.size else float("nan")
    out["tf_energy_concentration"] = float(np.nanmax(rel)) if np.isfinite(rel).any() else float("nan")
    return out


register_group(FeatureGroup(name="time_frequency", domain=DOMAIN,
                            names_fn=names, compute_fn=compute, docs_fn=docs))
