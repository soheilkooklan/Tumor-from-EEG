"""
Signal conditioning: resampling, filtering, referencing, epoching and automatic
quality control.

Purpose
-------
Produce fixed-length, artefact-screened epochs on a common sampling grid, so
that a feature computed on recording A means the same thing as the same feature
computed on recording B.

Scientific background
---------------------
Four choices here differ from the previous version and each has a reason:

*Filter order.* The mains notch is applied **before** the low-pass. Applied
after a 30 Hz low-pass, a 50 Hz notch removes nothing that has already been
removed - it is a no-op that looks like due diligence.

*Filter implementation.* Second-order-section (SOS) form is used rather than
transfer-function (b, a) form. High-order IIR filters expressed as polynomial
coefficients accumulate numerical error and can become unstable; SOS is the
standard remedy and is what `scipy.signal` documentation recommends. Zero-phase
`sosfiltfilt` avoids the phase distortion that would corrupt any latency- or
waveform-shape-sensitive feature, at the cost of doubling the effective order.

*Resampling.* Entropy, fractal and spectral features are all functions of the
sampling grid. Mixing 250 Hz and 512 Hz recordings without resampling makes the
feature values incomparable, and if acquisition rate correlates with diagnostic
group the classifier will learn the recording system.

*Epoching in seconds.* A window specified in samples is a different amount of
brain activity at every sampling rate. Epoch length is therefore specified in
time, and epochs are the unit that gets a quality decision.

Inputs   : `Recording`
Outputs  : `EpochedRecording` (epochs x channels x samples) plus a QC report
Limits   : no ICA-based ocular correction, no bad-channel interpolation, no
           montage conversion. Automatic ICA component rejection without human
           review is a known source of silent signal corruption, so it is not
           performed; the decomposition is exposed for manual workflows.

References
----------
- Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., Robbins, K.A. (2015).
  The PREP pipeline: standardized preprocessing for large-scale EEG analysis.
  Frontiers in Neuroinformatics 9, 16.
- Nolan, H., Whelan, R., Reilly, R.B. (2010). FASTER: fully automated
  statistical thresholding for EEG artifact rejection. J Neurosci Methods 192.
- de Cheveigne, A., Nelken, I. (2019). Filters: when, why, and how (not) to use
  them. Neuron 102(2), 280-293.
- Widmann, A., Schroger, E., Maess, B. (2015). Digital filter design for
  electrophysiological data. J Neurosci Methods 250, 34-46.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import (butter, cheby2, iirnotch, resample_poly, sosfiltfilt,
                          sosfreqz, tf2sos, welch)

from .config import PreprocessingConfig
from .io import Recording

logger = logging.getLogger(__name__)

__all__ = [
    "EpochedRecording",
    "EpochQuality",
    "preprocess_recording",
    "design_bandpass",
    "epoch_array",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EpochQuality:
    """Per-epoch, per-channel quality decision with an explicit reason."""

    epoch_index: int
    channel: str
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    peak_to_peak_uv: float = float("nan")
    hf_power_ratio: float = float("nan")


@dataclass
class EpochedRecording:
    """Fixed-length epochs ready for feature extraction.

    epochs : (n_epochs, n_channels, n_samples)
    mask   : (n_epochs, n_channels) boolean, True where the epoch/channel passed QC
    """

    epochs: np.ndarray
    mask: np.ndarray
    channel_names: List[str]
    sampling_rate: float
    subject_id: str
    recording_id: str
    label: Optional[int]
    quality: List[EpochQuality] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return self.epochs.shape[0]

    @property
    def n_good(self) -> int:
        return int(self.mask.sum())

    @property
    def acceptance_rate(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0

    def summary(self) -> str:
        return (f"{self.recording_id}: {self.n_epochs} epochs x "
                f"{len(self.channel_names)} channels, "
                f"{self.n_good} accepted ({100 * self.acceptance_rate:.0f}%)")


# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------

def design_bandpass(cfg: PreprocessingConfig, fs: float) -> np.ndarray:
    """Zero-phase band-pass in SOS form.

    Returns second-order sections. `sosfiltfilt` applies them forwards and
    backwards, so the effective magnitude response is the square of the design
    and the effective order is twice `filter_order`.
    """
    nyq = 0.5 * fs
    lo, hi = cfg.highpass / nyq, cfg.lowpass / nyq
    if not 0 < lo < hi < 1:
        raise ValueError(f"passband {cfg.highpass}-{cfg.lowpass} Hz invalid at fs={fs} Hz")
    if cfg.filter_design == "cheby2":
        return cheby2(cfg.filter_order, 40, [lo, hi], btype="band", output="sos")
    return butter(cfg.filter_order, [lo, hi], btype="band", output="sos")


def design_notches(cfg: PreprocessingConfig, fs: float) -> List[np.ndarray]:
    """Notches at the mains frequency and its in-band harmonics."""
    if not cfg.notch_freq:
        return []
    out = []
    nyq = 0.5 * fs
    for k in range(1, max(1, cfg.notch_harmonics) + 1):
        f0 = cfg.notch_freq * k
        # Only worth notching if the frequency survives the low-pass anyway.
        if f0 >= nyq * 0.95 or f0 > cfg.lowpass:
            continue
        b, a = iirnotch(f0 / nyq, cfg.notch_quality)
        out.append(tf2sos(b, a))
    return out


def filter_attenuation_report(cfg: PreprocessingConfig, fs: float,
                              probe_freqs: Tuple[float, ...] = (0.5, 1, 2, 10, 25, 40, 50, 60)
                              ) -> Dict[float, float]:
    """Magnitude response at a few probe frequencies, in dB.

    Reported in the run log so a reader can see exactly what the filter did
    rather than inferring it from nominal cut-offs.
    """
    sos = design_bandpass(cfg, fs)
    chain = [sos] + design_notches(cfg, fs)
    out = {}
    for f in probe_freqs:
        if f >= 0.5 * fs:
            continue
        mag = 1.0
        for s in chain:
            w, h = sosfreqz(s, worN=[2 * np.pi * f / fs])
            mag *= abs(h[0]) ** 2          # squared: forward + backward pass
        out[f] = float(20 * np.log10(max(mag, 1e-12)))
    return out


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def resample_signal(data: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling (anti-aliased) to the target rate."""
    if abs(fs_in - fs_out) < 1e-9:
        return data
    from fractions import Fraction
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)


def apply_reference(data: np.ndarray, scheme: str) -> np.ndarray:
    """Common average reference, or none.

    CAR removes the reference-electrode contribution shared by all channels. It
    is only valid with reasonable spatial coverage; with fewer than ~8 channels
    it can inject one channel's artefact into all the others, so it is skipped.
    """
    if scheme == "none":
        return data
    if data.shape[0] < 8:
        logger.info("skipping common average reference: only %d channels", data.shape[0])
        return data
    return data - data.mean(axis=0, keepdims=True)


def wavelet_denoise(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Level-dependent soft thresholding (BayesShrink-style noise estimate).

    A single global threshold derived from the finest detail band over-smooths
    coarse scales, where EEG rhythms live. The noise scale is therefore
    re-estimated per level from the median absolute deviation.
    """
    import pywt
    n = len(x)
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    out = [coeffs[0]]
    for c in coeffs[1:]:
        sigma = np.median(np.abs(c)) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(max(len(c), 2))) if sigma > 0 else 0.0
        out.append(pywt.threshold(c, thr, mode="soft"))
    rec = pywt.waverec(out, wavelet, mode="periodization")
    return rec[:n]


def normalise_amplitude(x: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel amplitude normalisation.

    Note that `zscore` forces mean=0 and SD=1 by construction, which makes the
    mean, SD, variance, RMS and Hjorth-activity features analytically constant.
    The feature extractor detects this and excludes them rather than shipping
    columns of ones.
    """
    if mode == "none":
        return x
    if mode == "zscore":
        sd = np.std(x)
        return (x - np.mean(x)) / sd if sd > 1e-12 else x - np.mean(x)
    if mode == "robust":
        med = np.median(x)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        return (x - med) / iqr if iqr > 1e-12 else x - med
    raise ValueError(f"unknown amplitude_normalization '{mode}'")


def epoch_array(x: np.ndarray, fs: float, epoch_seconds: float,
                overlap: float = 0.0, max_epochs: Optional[int] = None) -> np.ndarray:
    """Cut (..., n_samples) into (n_epochs, ..., epoch_samples).

    Trailing samples that do not fill a whole epoch are discarded rather than
    zero-padded; padding would change the spectrum and the entropy of the last
    epoch in a label-dependent way when classes have different durations.
    """
    n_samp = int(round(epoch_seconds * fs))
    if n_samp < 2:
        raise ValueError("epoch shorter than 2 samples")
    step = max(1, int(round(n_samp * (1.0 - overlap))))
    total = x.shape[-1]
    starts = list(range(0, total - n_samp + 1, step))
    if not starts:
        return np.empty((0,) + x.shape[:-1] + (n_samp,))
    if max_epochs is not None and len(starts) > max_epochs:
        # Evenly spaced subsample across the whole recording rather than the
        # first N: the beginning of a clinical EEG is disproportionately noisy
        # and often includes calibration and electrode settling.
        idx = np.linspace(0, len(starts) - 1, max_epochs).round().astype(int)
        starts = [starts[i] for i in idx]
    return np.stack([np.take(x, range(s, s + n_samp), axis=-1) for s in starts], axis=0)


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def _hf_power_ratio(x: np.ndarray, fs: float, split_hz: float = 30.0) -> float:
    """Fraction of total power above `split_hz` - a simple EMG index.

    Sustained scalp EMG is broadband and dominates above ~30 Hz, so a high
    ratio marks muscle contamination (Nolan et al., 2010; Muthukumaraswamy 2013).
    """
    nper = min(len(x), max(64, int(fs)))
    f, p = welch(x, fs=fs, nperseg=nper)
    total = trapezoid(p, f)
    if total <= 0:
        return float("nan")
    hi = trapezoid(p[f >= split_hz], f[f >= split_hz]) if np.any(f >= split_hz) else 0.0
    return float(hi / total)


def assess_epoch(x: np.ndarray, fs: float, cfg: PreprocessingConfig,
                 epoch_index: int, channel: str) -> EpochQuality:
    reasons: List[str] = []
    ptp = float(np.ptp(x))

    flat = float(np.mean(np.abs(np.diff(x)) < 1e-9))
    if flat > cfg.reject_flat_ratio:
        reasons.append(f"flat/disconnected ({100 * flat:.0f}% zero-slope samples)")

    hi, lo = float(np.max(x)), float(np.min(x))
    if hi > lo:
        clipped = float(np.mean((x >= hi - 1e-9) | (x <= lo + 1e-9)))
        if clipped > cfg.reject_clipping_ratio:
            reasons.append(f"clipping/saturation ({100 * clipped:.1f}% at rail)")

    if cfg.reject_amplitude_uv is not None and ptp > cfg.reject_amplitude_uv:
        reasons.append(f"peak-to-peak {ptp:.0f} uV exceeds {cfg.reject_amplitude_uv:.0f} uV")

    if np.std(x) < 1e-9:
        reasons.append("zero variance")

    hf = _hf_power_ratio(x, fs)
    if np.isfinite(hf) and hf > cfg.reject_muscle_ratio:
        reasons.append(f"probable EMG ({100 * hf:.0f}% of power above 30 Hz)")

    return EpochQuality(epoch_index=epoch_index, channel=channel,
                        accepted=not reasons, reasons=reasons,
                        peak_to_peak_uv=ptp, hf_power_ratio=hf)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def preprocess_recording(rec: Recording, cfg: PreprocessingConfig) -> EpochedRecording:
    """Full recording-level pipeline.

    Order of operations, and why:
      1. resample      - put every recording on one sampling grid
      2. DC removal    - stop the high-pass from ringing on a large offset
      3. notch         - BEFORE the low-pass, or it does nothing
      4. band-pass     - zero-phase SOS
      5. reference     - after filtering, so the CAR is computed on clean data
      6. denoise       - optional, off by default
      7. epoch         - fixed duration in seconds
      8. QC            - per epoch and channel, with recorded reasons
      9. normalise     - last, per epoch, so it cannot leak across epochs
    """
    x = rec.data.copy()

    x = resample_signal(x, rec.sampling_rate, cfg.target_sampling_rate)
    fs = cfg.target_sampling_rate

    x = x - x.mean(axis=-1, keepdims=True)

    for sos_notch in design_notches(cfg, fs):
        x = sosfiltfilt(sos_notch, x, axis=-1)
    x = sosfiltfilt(design_bandpass(cfg, fs), x, axis=-1)

    x = apply_reference(x, cfg.reference)

    if cfg.wavelet_denoise:
        x = np.stack([wavelet_denoise(ch, cfg.wavelet, cfg.wavelet_level) for ch in x])

    ep = epoch_array(x, fs, cfg.epoch_seconds, cfg.epoch_overlap,
                     cfg.max_epochs_per_recording)
    if ep.shape[0] == 0:
        logger.warning("%s: too short for a single %.1f s epoch (%.1f s available)",
                       rec.recording_id, cfg.epoch_seconds, x.shape[-1] / fs)
        mask = np.zeros((0, len(rec.channel_names)), dtype=bool)
        return EpochedRecording(ep, mask, list(rec.channel_names), fs,
                                rec.subject_id, rec.recording_id, rec.label,
                                [], {"reason": "recording too short"})

    n_ep, n_ch, _ = ep.shape
    mask = np.zeros((n_ep, n_ch), dtype=bool)
    quality: List[EpochQuality] = []
    for i in range(n_ep):
        for c in range(n_ch):
            q = assess_epoch(ep[i, c], fs, cfg, i, rec.channel_names[c])
            quality.append(q)
            mask[i, c] = q.accepted

    if cfg.amplitude_normalization != "none":
        for i in range(n_ep):
            for c in range(n_ch):
                ep[i, c] = normalise_amplitude(ep[i, c], cfg.amplitude_normalization)

    out = EpochedRecording(
        epochs=ep, mask=mask, channel_names=list(rec.channel_names),
        sampling_rate=fs, subject_id=rec.subject_id,
        recording_id=rec.recording_id, label=rec.label, quality=quality,
        metadata={
            **rec.metadata,
            "original_sampling_rate": rec.sampling_rate,
            "filter_response_db": filter_attenuation_report(cfg, fs),
            "amplitude_normalization": cfg.amplitude_normalization,
        },
    )

    if out.n_good < cfg.min_good_epochs:
        logger.warning("%s: only %d accepted epoch-channels (minimum %d) - "
                       "this recording should probably be excluded",
                       rec.recording_id, out.n_good, cfg.min_good_epochs)
    logger.info(out.summary())
    return out
