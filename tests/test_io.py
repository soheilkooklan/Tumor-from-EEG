"""
Readers, lazy loading and parallel extraction.

Lazy loading and n_jobs exist only to make large cohorts fit in memory and
time. Neither may change a single number, so these tests compare against the
plain eager, single-process path.
"""

import numpy as np
import pytest

from eegtumor.config import AnalysisConfig
from eegtumor.experiment import extract_cohort_features
from eegtumor.features.complexity import lempel_ziv_complexity
from eegtumor.io import (Recording, RecordingStub, ensure_loaded, load_cohort,
                         read_recording)

FS = 128.0


def _write_cohort(tmp_path, n_subjects=4, seconds=24):
    rng = np.random.default_rng(1)
    lines = ["path,subject_id,label,recording_id,sampling_rate"]
    t = np.arange(int(FS * seconds)) / FS
    for s in range(n_subjects):
        alpha = 9 + s * 0.5
        x = np.vstack([20 * np.sin(2 * np.pi * alpha * t + c) + 5 * rng.normal(size=t.size)
                       for c in range(4)])
        np.save(tmp_path / f"s{s}.npy", x)
        lines.append(f"s{s}.npy,S{s},{s % 2},S{s}_r0,{FS:g}")
    (tmp_path / "cohort.csv").write_text("\n".join(lines) + "\n")
    return tmp_path / "cohort.csv"


def _small_cfg():
    cfg = AnalysisConfig()
    cfg.preprocessing.target_sampling_rate = FS
    cfg.preprocessing.lowpass = 40.0
    cfg.preprocessing.min_good_epochs = 1
    cfg.bands.bands["gamma"] = (30.0, 40.0)
    cfg.validate()
    return cfg


def test_lazy_cohort_holds_no_signal_and_loads_identically(tmp_path):
    manifest = _write_cohort(tmp_path)
    eager = load_cohort(str(manifest))
    lazy = load_cohort(str(manifest), lazy=True)
    assert all(isinstance(r, RecordingStub) for r in lazy)
    assert all(not hasattr(r, "data") for r in lazy)
    for e, s in zip(eager, lazy):
        assert e.channel_names == s.channel_names
        assert e.sampling_rate == s.sampling_rate
        assert e.duration_seconds == pytest.approx(s.duration_seconds)
        loaded = ensure_loaded(s)
        assert isinstance(loaded, Recording)
        assert np.array_equal(loaded.data, e.data)
    assert eager.audit() == lazy.audit()


def test_ensure_loaded_passes_recordings_through(tmp_path):
    manifest = _write_cohort(tmp_path, n_subjects=1)
    rec = next(iter(load_cohort(str(manifest))))
    assert ensure_loaded(rec) is rec


def test_parallel_lazy_extraction_matches_eager(tmp_path):
    manifest = _write_cohort(tmp_path)
    cfg = _small_cfg()
    Xa, names_a, ia = extract_cohort_features(load_cohort(str(manifest)), cfg)
    Xb, names_b, ib = extract_cohort_features(load_cohort(str(manifest), lazy=True),
                                              cfg, n_jobs=2)
    assert Xa.shape[0] > 0
    assert names_a == names_b
    assert ia == ib
    assert np.array_equal(Xa, Xb, equal_nan=True)


def test_unsupported_suffix_is_rejected(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"0")
    with pytest.raises(ValueError):
        read_recording(str(p), "S1")


def _lz_reference(x):
    """Kaspar-Schuster LZ76, the version used before 2.2.0."""
    s = (np.asarray(x) > np.median(x)).astype(np.uint8).tobytes()
    n = len(s)
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


@pytest.mark.parametrize("n", [2, 3, 7, 50, 500, 2000])
def test_fast_lempel_ziv_matches_kaspar_schuster(n):
    rng = np.random.default_rng(n)
    for x in (rng.normal(size=n), np.sin(np.arange(n) * 0.3),
              np.cumsum(rng.normal(size=n)), rng.integers(0, 2, n).astype(float)):
        assert lempel_ziv_complexity(x, normalise=False) == _lz_reference(x)


def test_eeglab_set_reader(tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("eeglabio")
    info = mne.create_info(["Fp1", "Fp2", "Cz"], 250.0, "eeg")
    data = np.random.default_rng(0).normal(size=(3, 2500)) * 1e-5
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path / "x.set"
    mne.export.export_raw(str(path), raw, fmt="eeglab", verbose="ERROR")
    rec = read_recording(str(path), "S1")
    assert rec.sampling_rate == 250.0
    assert rec.channel_names == ["FP1", "FP2", "CZ"]
    assert np.allclose(rec.data, data * 1e6, atol=1e-3)


def test_scalp_selection_drops_ecg_and_references():
    from eegtumor.io import harmonise_channel_name
    from eegtumor.preprocessing import select_channels
    raw = ("FP1|FP2|F3|F4|C3|C4|P3|P4|O1|O2|F7|F8|T3|T4|T5|T6|FZ|PZ|CZ|"
           "ECG|A1|A2|Status|EEG EKG1-REF|EOG LOC").split("|")
    names = [harmonise_channel_name(n) for n in raw]
    rec = Recording(np.zeros((len(names), 50)), names, 200.0, "S1", "r1")
    cfg = AnalysisConfig().preprocessing
    out = select_channels(rec, cfg)
    assert len(out.channel_names) == 19
    assert {"ECG", "A1", "A2", "Status"} <= set(out.metadata["dropped_channels"])

    cfg.channels = "all"
    assert select_channels(rec, cfg) is rec

    cfg.channels = ["C3", "C4"]
    assert select_channels(rec, cfg).channel_names == ["C3", "C4"]
    cfg.channels = ["C3", "XX"]
    with pytest.raises(ValueError):
        select_channels(rec, cfg)


def test_ecg_does_not_reach_the_average_reference():
    """A large ECG channel left in the CAR would show up in every scalp channel."""
    from eegtumor.preprocessing import preprocess_recording
    fs, n = 200.0, 200 * 40
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    scalp = [10 * np.sin(2 * np.pi * 10 * t) + rng.normal(size=n) for _ in range(8)]
    ecg = 1000 * (np.sin(2 * np.pi * 1.2 * t) > 0.99)
    names = ["FP1", "FP2", "C3", "C4", "P3", "P4", "O1", "O2", "ECG"]
    rec = Recording(np.vstack(scalp + [ecg]), names, fs, "S1", "r1")
    cfg = AnalysisConfig().preprocessing
    cfg.target_sampling_rate = fs
    cfg.reject_amplitude_uv = None
    ep = preprocess_recording(rec, cfg)
    assert "ECG" not in ep.channel_names
    assert np.abs(ep.epochs).max() < 100
