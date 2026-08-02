"""
Frequency-domain biomarkers.

This is the scientifically strongest family for the question at hand. The
classical scalp-EEG correlate of an underlying structural lesion is focal
slowing: increased delta and theta power, reduced alpha, frequently
lateralised. The band ratios computed here are direct quantifications of that
pattern, and unlike the entropy and fractal families they have a large,
consistent clinical literature behind them.

Two implementation choices depart from the common textbook recipe:

*Multitaper by default.* A single Welch taper trades variance against
resolution poorly at the epoch lengths used in clinical EEG. Thomson's
multitaper estimator averages several orthogonal DPSS tapers and gives a
lower-variance spectrum at the same resolution. Welch is retained for
comparison because much of the published EEG literature uses it.

*Aperiodic separation.* Total band power confounds oscillatory activity with
the broadband 1/f background. The background is not noise - its exponent
tracks excitation/inhibition balance and changes with anaesthesia, age and
pathology - so its slope and offset are extracted as features in their own
right, and the reader is warned that raw band power mixes the two.

References
----------
- Thomson, D.J. (1982). Spectrum estimation and harmonic analysis. Proc. IEEE
  70(9), 1055-1096.
- Donoghue, T. et al. (2020). Parameterizing neural power spectra into periodic
  and aperiodic components. Nature Neuroscience 23, 1655-1665.
- Gloor, P., Ball, G., Schaul, N. (1977). Brain lesions that produce delta
  waves in the EEG. Neurology 27(4), 326-333.
- Schaul, N. (1998). The fundamental neural mechanisms of electroencephalography.
  Electroencephalography and Clinical Neurophysiology 106(2), 101-107.
- Finnigan, S., van Putten, M.J.A.M. (2013). EEG in ischaemic stroke:
  quantitative EEG can uniquely inform (sub-)acute prognoses. Clinical
  Neurophysiology 124(1), 10-19.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.signal.windows import dpss

from . import FeatureGroup, FeatureSpec, register_group

DOMAIN = "spectral"


# ---------------------------------------------------------------------------
# Power spectral density
# ---------------------------------------------------------------------------

def multitaper_psd(x: np.ndarray, fs: float, n_tapers: int = 5
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Thomson multitaper PSD using DPSS (Slepian) sequences.

    Half-bandwidth NW is set so that the requested number of tapers stays below
    the 2*NW-1 stability limit. Returns one-sided PSD in units^2/Hz.
    """
    n = len(x)
    nw = max(2.0, (n_tapers + 1) / 2.0)
    k = min(n_tapers, int(2 * nw) - 1)
    tapers = dpss(n, nw, k)              # unit energy: sum(w^2) = 1
    x = x - x.mean()
    spectra = np.abs(np.fft.rfft(tapers * x, axis=-1)) ** 2
    psd = spectra.mean(axis=0) / fs      # unit-energy tapers -> divide by fs only
    psd[1:-1] *= 2.0                     # one-sided correction
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, psd


def compute_psd(x: np.ndarray, fs: float, fcfg) -> Tuple[np.ndarray, np.ndarray]:
    if fcfg.psd_method == "multitaper":
        return multitaper_psd(x, fs, fcfg.n_tapers)
    nper = min(len(x), max(16, int(round(fcfg.psd_window_seconds * fs))))
    nover = int(nper * fcfg.psd_overlap)
    return welch(x, fs=fs, nperseg=nper, noverlap=nover, window="hann")


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def _band_power(freqs, psd, lo, hi) -> float:
    idx = (freqs >= lo) & (freqs < hi)
    return float(trapezoid(psd[idx], freqs[idx])) if idx.sum() > 1 else 0.0


def spectral_edge(freqs, psd, fraction: float) -> float:
    """Frequency below which `fraction` of the total power lies."""
    csum = np.cumsum(psd)
    if csum[-1] <= 0:
        return float("nan")
    csum = csum / csum[-1]
    return float(np.interp(fraction, csum, freqs))


def aperiodic_fit(freqs, psd, fmin: float, fmax: float) -> Tuple[float, float]:
    """Robust log-log line fit of the broadband background.

    Fits log10(PSD) against log10(f) over [fmin, fmax] after removing the
    strongest oscillatory peaks, approximating the periodic/aperiodic split of
    Donoghue et al. (2020) without the extra dependency. Returns
    (exponent, offset) where the exponent is the positive 1/f^beta slope.

    Limitation: this is a simplified estimator. Where the aperiodic component is
    the object of study rather than one feature among many, use `specparam`.
    """
    idx = (freqs >= fmin) & (freqs <= fmax) & (psd > 0)
    if idx.sum() < 10:
        return float("nan"), float("nan")
    lf, lp = np.log10(freqs[idx]), np.log10(psd[idx])
    slope, intercept = np.polyfit(lf, lp, 1)
    resid = lp - (slope * lf + intercept)
    keep = resid < np.percentile(resid, 80)          # drop the top 20%: peaks
    if keep.sum() >= 10:
        slope, intercept = np.polyfit(lf[keep], lp[keep], 1)
    return float(-slope), float(intercept)


# ---------------------------------------------------------------------------
# Registry interface
# ---------------------------------------------------------------------------

def names(fcfg, bands) -> List[str]:
    out: List[str] = []
    for b in bands.bands:
        out.append(f"sp_abs_power_{b}")
    for b in bands.bands:
        out.append(f"sp_rel_power_{b}")
    for r in bands.ratios:
        out.append(f"sp_ratio_{r}")
    out += [
        "sp_total_power", "sp_peak_frequency", "sp_peak_amplitude",
        "sp_mean_frequency", "sp_median_frequency",
        "sp_edge_frequency_90", "sp_edge_frequency_95",
        "sp_entropy", "sp_centroid", "sp_spread", "sp_flatness",
        "sp_aperiodic_exponent", "sp_aperiodic_offset",
    ]
    return out


_STATIC_DOC = {
    "sp_total_power": ("Integrated power across the analysis passband.",
                       "Overall signal energy; low values suggest attenuation, "
                       "high values suggest artefact or high-amplitude slowing.",
                       True),
    "sp_peak_frequency": ("Frequency of the largest spectral maximum.",
                          "In an awake resting adult this is the posterior "
                          "dominant rhythm, normally 8-13 Hz. Slowing of the "
                          "dominant rhythm is a sensitive, non-specific sign of "
                          "cerebral dysfunction.", False),
    "sp_peak_amplitude": ("PSD value at the peak frequency.",
                          "Strength of the dominant rhythm.", True),
    "sp_mean_frequency": ("Power-weighted mean frequency.",
                          "Single-number summary of the spectral centre of mass; "
                          "falls with slowing.", False),
    "sp_median_frequency": ("Frequency splitting total power in half.",
                            "Robust alternative to the mean frequency, standard "
                            "in quantitative EEG monitoring.", False),
    "sp_edge_frequency_90": ("Spectral edge frequency, 90% of power below it.",
                             "Classic depth-of-anaesthesia and encephalopathy "
                             "index; sensitive to loss of fast activity.", False),
    "sp_edge_frequency_95": ("Spectral edge frequency, 95%.",
                             "As above, less affected by the extreme tail.", False),
    "sp_entropy": ("Shannon entropy of the normalised PSD, normalised to [0,1].",
                   "How evenly power is spread across frequency. A dominant "
                   "rhythm lowers it; flat, featureless background raises it.",
                   False),
    "sp_centroid": ("First moment of the normalised spectrum.",
                    "Spectral centre of gravity.", False),
    "sp_spread": ("Square root of the second central spectral moment.",
                  "Spectral bandwidth around the centroid.", False),
    "sp_flatness": ("Geometric over arithmetic mean of the PSD (Wiener entropy).",
                    "Approaches 1 for noise-like spectra and 0 for strongly "
                    "peaked, rhythmic ones.", False),
    "sp_aperiodic_exponent": ("Slope of the log-log 1/f background.",
                              "Tracks the broadband non-oscillatory component; "
                              "linked to excitation/inhibition balance and altered "
                              "in many encephalopathies. Reported separately "
                              "because it contaminates raw band power.", False),
    "sp_aperiodic_offset": ("Intercept of the log-log background fit.",
                            "Broadband power level independent of the slope.", True),
}


def docs(fcfg, bands) -> Dict[str, FeatureSpec]:
    out: Dict[str, FeatureSpec] = {}
    for b, (lo, hi) in bands.bands.items():
        out[f"sp_abs_power_{b}"] = FeatureSpec(
            name=f"sp_abs_power_{b}", domain=DOMAIN, unit="uV^2",
            description=f"Integrated PSD over {lo}-{hi} Hz.",
            interpretation=(
                "Absolute power in the {b} band. Excess delta and theta over a "
                "region is the classical scalp correlate of an underlying "
                "structural lesion; note that absolute power also carries "
                "electrode impedance and skull thickness differences, which is "
                "why the relative measure is usually preferred for comparison "
                "across subjects.".format(b=b)),
            references=("Gloor et al. 1977", "Schaul 1998"),
            complexity="O(n log n)", amplitude_dependent=True)
        out[f"sp_rel_power_{b}"] = FeatureSpec(
            name=f"sp_rel_power_{b}", domain=DOMAIN, unit="fraction",
            description=f"{b} power divided by total passband power.",
            interpretation=(
                f"Share of the spectrum occupied by the {b} band. Normalising "
                f"away total power removes most inter-subject amplitude "
                f"variation, at the cost of making the bands mutually dependent "
                f"- they sum to one, so they are not independent features."),
            references=("Finnigan & van Putten 2013",),
            complexity="O(n log n)", amplitude_dependent=False)

    for r, (num, den) in bands.ratios.items():
        interp = {
            "slowing_index": (
                "Delta+theta over alpha+beta, sometimes called the DTABR. The "
                "most direct quantification of EEG slowing available, validated "
                "in stroke and encephalopathy monitoring, and the single most "
                "defensible marker in this feature set for a structural lesion."),
            "theta_alpha": ("Rises when the dominant rhythm slows into theta."),
            "delta_alpha": (
                "Delta-alpha ratio; among the best-performing quantitative EEG "
                "indices for acute cerebral injury."),
            "theta_beta": (
                "Widely used in attention research; included as a comparison "
                "index rather than a lesion marker."),
        }.get(r, "Band-power ratio.")
        out[f"sp_ratio_{r}"] = FeatureSpec(
            name=f"sp_ratio_{r}", domain=DOMAIN, unit="ratio",
            description=f"({'+'.join(num)}) / ({'+'.join(den)}) band power.",
            interpretation=interp,
            references=("Finnigan & van Putten 2013", "Schaul 1998"),
            complexity="O(n log n)", amplitude_dependent=False)

    for n, (desc, interp, amp) in _STATIC_DOC.items():
        refs = ("Donoghue et al. 2020",) if "aperiodic" in n else (
                ("Thomson 1982",) if "power" in n else ())
        out[n] = FeatureSpec(name=n, domain=DOMAIN, description=desc,
                             interpretation=interp, references=refs,
                             complexity="O(n log n)", amplitude_dependent=amp,
                             unit="Hz" if "frequency" in n or "centroid" in n
                                  or "spread" in n else "a.u.")
    return out


def compute(x: np.ndarray, fs: float, fcfg, bands) -> "OrderedDict[str, float]":
    freqs, psd = compute_psd(x, fs, fcfg)

    lo_edge = min(lo for lo, _ in bands.bands.values())
    hi_edge = max(hi for _, hi in bands.bands.values())
    band_mask = (freqs >= lo_edge) & (freqs <= hi_edge)
    f_b, p_b = freqs[band_mask], psd[band_mask]

    total = float(trapezoid(p_b, f_b)) if f_b.size > 1 else 0.0
    denom = total if total > 0 else np.nan

    out: "OrderedDict[str, float]" = OrderedDict()
    powers = {}
    for b, (lo, hi) in bands.bands.items():
        powers[b] = _band_power(freqs, psd, lo, hi)
        out[f"sp_abs_power_{b}"] = powers[b]
    for b in bands.bands:
        out[f"sp_rel_power_{b}"] = float(powers[b] / denom)
    for r, (num, den) in bands.ratios.items():
        nsum = sum(powers[b] for b in num)
        dsum = sum(powers[b] for b in den)
        out[f"sp_ratio_{r}"] = float(nsum / dsum) if dsum > 0 else float("nan")

    out["sp_total_power"] = total

    if p_b.size > 2:
        pk = int(np.argmax(p_b))
        out["sp_peak_frequency"] = float(f_b[pk])
        out["sp_peak_amplitude"] = float(p_b[pk])
        w = p_b / p_b.sum() if p_b.sum() > 0 else p_b
        out["sp_mean_frequency"] = float(np.sum(f_b * w))
        out["sp_median_frequency"] = spectral_edge(f_b, p_b, 0.5)
        out["sp_edge_frequency_90"] = spectral_edge(f_b, p_b, 0.90)
        out["sp_edge_frequency_95"] = spectral_edge(f_b, p_b, 0.95)
        pn = w[w > 0]
        out["sp_entropy"] = float(-np.sum(pn * np.log2(pn)) / np.log2(len(pn))) \
            if len(pn) > 1 else float("nan")
        centroid = float(np.sum(f_b * w))
        out["sp_centroid"] = centroid
        out["sp_spread"] = float(np.sqrt(np.sum(((f_b - centroid) ** 2) * w)))
        pos = p_b[p_b > 0]
        out["sp_flatness"] = float(np.exp(np.mean(np.log(pos))) / np.mean(pos)) \
            if pos.size else float("nan")
    else:
        for k in ("sp_peak_frequency", "sp_peak_amplitude", "sp_mean_frequency",
                  "sp_median_frequency", "sp_edge_frequency_90",
                  "sp_edge_frequency_95", "sp_entropy", "sp_centroid",
                  "sp_spread", "sp_flatness"):
            out[k] = float("nan")

    fmin, fmax = fcfg.aperiodic_range
    expo, offset = aperiodic_fit(freqs, psd, max(fmin, lo_edge), min(fmax, hi_edge))
    out["sp_aperiodic_exponent"] = expo
    out["sp_aperiodic_offset"] = offset
    return out


register_group(FeatureGroup(name="spectral", domain=DOMAIN,
                            names_fn=names, compute_fn=compute, docs_fn=docs))
