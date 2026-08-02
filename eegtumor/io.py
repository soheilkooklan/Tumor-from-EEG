"""
Reading EEG recordings, and the cohort bookkeeping that makes honest
validation possible.

Purpose
-------
Turn heterogeneous files (EDF/BDF, CSV, MAT, NumPy) into a uniform `Recording`
object that carries, in addition to the signal, the two pieces of metadata the
rest of the pipeline cannot work correctly without: **who the recording came
from** and **what the label is**.

Scientific background
---------------------
The dominant failure mode in EEG machine learning is not a bad classifier, it
is a bad unit of analysis. Channels and epochs from one recording share the
montage, the impedance profile, the amplifier, the medication state and the
individual's spectral fingerprint. Treated as independent samples and split at
random, they let a model identify the *person* and score well without having
learned anything about the *pathology*. Saeb et al. (2017) named this the
"subject-wise vs record-wise" error and showed multi-fold accuracy inflation in
published health-sensing work; Kapoor and Narayanan (2023) find it to be the
single most common leakage type across ML-based science.

The design decision here is that `subject_id` is a required field, not an
optional one. A file that cannot be attributed to a subject cannot be used for
validation, so it is rejected at load time rather than quietly contributing to
an optimistic number later.

Inputs   : EEG files, plus a cohort manifest (CSV) supplying subject/label
Outputs  : `Recording` objects; `Cohort` with group vectors for CV
Limits   : channel-name harmonisation is best-effort; montage conversion,
           bad-channel interpolation and re-referencing to a template montage
           are not attempted here.

References
----------
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., Kording, K.P. (2017). The
  need to approximate the use-case in clinical machine learning. GigaScience
  6(5), 1-9.
- Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in
  machine-learning-based science. Patterns 4(9), 100804.
- Obeid, I., Picone, J. (2016). The Temple University Hospital EEG data corpus.
  Frontiers in Neuroscience 10, 196.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Recording", "Cohort", "read_recording", "load_cohort", "CohortError"]

SUPPORTED_SUFFIXES = {".edf", ".bdf", ".csv", ".tsv", ".mat", ".npy", ".npz"}

# Standard 10-20 names, used to normalise the many spellings found in the wild
# ("EEG FP1-REF", "Fp1-A1", "fp1"). Recognition is best-effort and never fatal.
_TEN_TWENTY = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "T7", "C3", "CZ", "C4",
    "T4", "T8", "T5", "P7", "P3", "PZ", "P4", "T6", "P8", "O1", "OZ", "O2",
    "A1", "A2", "M1", "M2",
]


class CohortError(ValueError):
    """Raised when the cohort definition would make validation unsound."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    """One EEG recording session.

    Attributes
    ----------
    data : (n_channels, n_samples) float array, microvolts where known
    channel_names : harmonised channel labels
    sampling_rate : Hz, as stored in the file or supplied by the manifest
    subject_id : REQUIRED. Groups all recordings of one person for CV.
    recording_id : unique per file
    label : 0/1, or None for unlabelled data being screened
    """

    data: np.ndarray
    channel_names: List[str]
    sampling_rate: float
    subject_id: str
    recording_id: str
    label: Optional[int] = None
    source_path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"{self.recording_id}: data must be 2-D (channels x time)")
        if self.data.shape[0] != len(self.channel_names):
            raise ValueError(
                f"{self.recording_id}: {self.data.shape[0]} channels of data but "
                f"{len(self.channel_names)} channel names"
            )
        if not str(self.subject_id).strip():
            raise CohortError(
                f"{self.recording_id}: subject_id is empty. Without it, epochs "
                f"from this recording cannot be kept out of the test fold."
            )

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.data.shape[1] / self.sampling_rate

    def describe(self) -> str:
        lab = "unlabelled" if self.label is None else f"label={self.label}"
        return (f"{self.recording_id} [subject {self.subject_id}, {lab}] "
                f"{self.n_channels} ch, {self.duration_seconds:.1f} s @ "
                f"{self.sampling_rate:g} Hz")


@dataclass
class Cohort:
    """A labelled set of recordings plus the group vector CV needs."""

    recordings: List[Recording]

    def __post_init__(self) -> None:
        ids = [r.recording_id for r in self.recordings]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise CohortError(f"duplicate recording_id: {sorted(dupes)}")

    def __len__(self) -> int:
        return len(self.recordings)

    def __iter__(self):
        return iter(self.recordings)

    @property
    def labels(self) -> np.ndarray:
        if any(r.label is None for r in self.recordings):
            raise CohortError("cohort contains unlabelled recordings")
        return np.array([int(r.label) for r in self.recordings])

    @property
    def subjects(self) -> np.ndarray:
        return np.array([r.subject_id for r in self.recordings])

    def summary(self) -> Dict[str, object]:
        y = [r.label for r in self.recordings]
        subj = set(self.subjects)
        pos_subj = {r.subject_id for r in self.recordings if r.label == 1}
        neg_subj = {r.subject_id for r in self.recordings if r.label == 0}
        return {
            "n_recordings": len(self.recordings),
            "n_subjects": len(subj),
            "n_positive_recordings": sum(1 for v in y if v == 1),
            "n_negative_recordings": sum(1 for v in y if v == 0),
            "n_positive_subjects": len(pos_subj),
            "n_negative_subjects": len(neg_subj),
            "subjects_in_both_classes": sorted(pos_subj & neg_subj),
            "sampling_rates": sorted({r.sampling_rate for r in self.recordings}),
            "median_duration_s": float(np.median([r.duration_seconds for r in self.recordings])),
        }

    def audit(self) -> List[str]:
        """Design problems that would undermine any result computed downstream.

        Returned as warnings rather than raised, because some are legitimate in
        specific study designs - but every one of them belongs in a Limitations
        section, so the caller is told about them explicitly.
        """
        warnings: List[str] = []
        s = self.summary()

        if s["n_subjects"] < 20:
            warnings.append(
                f"only {s['n_subjects']} subjects: fold-to-fold variance will "
                f"dominate any performance estimate, and confidence intervals "
                f"will be very wide. Report them, do not hide them."
            )
        if min(s["n_positive_subjects"], s["n_negative_subjects"]) < 5:
            warnings.append(
                f"smallest class has {min(s['n_positive_subjects'], s['n_negative_subjects'])} "
                f"subjects: subject-disjoint cross-validation is barely defined here."
            )
        if s["subjects_in_both_classes"]:
            warnings.append(
                f"subjects appear in both classes: {s['subjects_in_both_classes']}. "
                f"Grouped CV will place both their recordings in the same fold; "
                f"confirm this is intended."
            )
        if len(s["sampling_rates"]) > 1:
            rates_by_class = {}
            for r in self.recordings:
                rates_by_class.setdefault(r.label, set()).add(r.sampling_rate)
            if len(rates_by_class) > 1 and not (
                rates_by_class.get(0, set()) & rates_by_class.get(1, set())
            ):
                warnings.append(
                    "the two classes have completely disjoint sampling rates "
                    f"({rates_by_class}). Any classifier can separate them by "
                    "acquisition system alone. This is a confound, not a result."
                )
        durs = {}
        for r in self.recordings:
            durs.setdefault(r.label, []).append(r.duration_seconds)
        if len(durs) == 2:
            m0, m1 = np.median(durs[0]), np.median(durs[1])
            if max(m0, m1) > 3 * max(min(m0, m1), 1e-9):
                warnings.append(
                    f"median recording duration differs {m0:.0f}s vs {m1:.0f}s "
                    f"between classes: epoch counts will be imbalanced in a way "
                    f"that correlates with the label."
                )
        chan_sets = {}
        for r in self.recordings:
            chan_sets.setdefault(r.label, set()).add(tuple(sorted(r.channel_names)))
        if len(chan_sets) == 2 and not (chan_sets[0] & chan_sets[1]):
            warnings.append(
                "the two classes share no common channel montage; features are "
                "not comparable across classes."
            )
        return warnings


# ---------------------------------------------------------------------------
# Channel-name harmonisation
# ---------------------------------------------------------------------------

def harmonise_channel_name(raw: str) -> str:
    """Best-effort mapping of a vendor channel label onto 10-20 nomenclature.

    'EEG FP1-REF' -> 'FP1',  'Fp1-A1' -> 'FP1',  'X3' -> 'X3' (unchanged).
    Returns the original string when nothing is recognised; this function never
    invents an electrode position.
    """
    s = raw.strip().upper()
    s = re.sub(r"^EEG\s+", "", s)
    s = re.sub(r"[-_](REF|LE|A1|A2|M1|M2|AVG|CAR)$", "", s)
    s = s.replace(" ", "")
    core = re.sub(r"[^A-Z0-9]", "", s)
    if core in _TEN_TWENTY:
        return core
    return raw.strip()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_edf(path: Path) -> Tuple[np.ndarray, List[str], float, dict]:
    try:
        import mne
    except ImportError as exc:                                # pragma: no cover
        raise ImportError("reading EDF/BDF requires mne:  pip install mne") from exc

    reader = mne.io.read_raw_bdf if path.suffix.lower() == ".bdf" else mne.io.read_raw_edf
    raw = reader(str(path), preload=True, verbose="ERROR")
    raw.pick("eeg") if len(raw.copy().pick("eeg").ch_names) else None
    data = raw.get_data() * 1e6            # MNE stores volts; we work in microvolts
    names = [harmonise_channel_name(n) for n in raw.ch_names]
    meta = {"edf_meas_date": str(raw.info.get("meas_date")), "units": "uV"}
    return data, names, float(raw.info["sfreq"]), meta


def _read_delimited(path: Path, sep: str) -> Tuple[np.ndarray, List[str], dict]:
    import pandas as pd
    df = pd.read_csv(path, sep=sep)
    numeric = df.select_dtypes(include=[np.number])
    # A monotonically increasing first column is a time/index axis, not a channel.
    if numeric.shape[1] > 1:
        first = numeric.iloc[:, 0].to_numpy()
        if np.all(np.diff(first) > 0):
            logger.info("%s: dropping monotonic first column '%s' (time axis)",
                        path.name, numeric.columns[0])
            numeric = numeric.iloc[:, 1:]
    if numeric.shape[1] == 0:
        raise ValueError(f"{path.name}: no numeric columns found")
    data = numeric.to_numpy(dtype=float).T          # -> channels x time
    names = [harmonise_channel_name(str(c)) for c in numeric.columns]
    return data, names, {"units": "unknown"}


def _read_mat(path: Path) -> Tuple[np.ndarray, List[str], dict]:
    from scipy.io import loadmat
    mat = loadmat(str(path))
    candidates = {k: np.asarray(v) for k, v in mat.items()
                  if not k.startswith("__") and np.asarray(v).ndim == 2
                  and np.issubdtype(np.asarray(v).dtype, np.number)}
    if not candidates:
        raise ValueError(f"{path.name}: no 2-D numeric array found")
    # Largest array is the signal; ambiguity is resolved explicitly and logged
    key = max(candidates, key=lambda k: candidates[k].size)
    arr = candidates[key]
    if len(candidates) > 1:
        logger.warning("%s: several arrays present %s, using '%s'",
                       path.name, sorted(candidates), key)
    if arr.shape[0] > arr.shape[1]:                # assume more time than channels
        arr = arr.T
    names = [f"{key}_{i}" for i in range(arr.shape[0])]
    return arr.astype(float), names, {"mat_variable": key, "units": "unknown"}


def _read_numpy(path: Path) -> Tuple[np.ndarray, List[str], dict]:
    obj = np.load(str(path), allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = sorted(obj.files)
        arrs = [np.asarray(obj[k]).ravel() for k in keys]
        n = min(len(a) for a in arrs)
        return np.vstack([a[:n] for a in arrs]), keys, {"units": "unknown"}
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr, [f"ch{i}" for i in range(arr.shape[0])], {"units": "unknown"}


def read_recording(
    path: str,
    subject_id: str,
    recording_id: Optional[str] = None,
    label: Optional[int] = None,
    sampling_rate: Optional[float] = None,
) -> Recording:
    """Read one file into a `Recording`.

    `sampling_rate` is mandatory for formats that do not store it (CSV, MAT,
    NumPy). Guessing it would silently rescale every frequency-domain feature,
    so it raises instead.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported format '{suffix}' (supported: "
                         f"{sorted(SUPPORTED_SUFFIXES)})")

    if suffix in {".edf", ".bdf"}:
        data, names, fs, meta = _read_edf(p)
        if sampling_rate and abs(sampling_rate - fs) > 1e-6:
            logger.warning("%s: manifest says %g Hz but the file header says %g Hz; "
                           "trusting the header", p.name, sampling_rate, fs)
    else:
        if suffix in {".csv", ".tsv"}:
            data, names, meta = _read_delimited(p, "," if suffix == ".csv" else "\t")
        elif suffix == ".mat":
            data, names, meta = _read_mat(p)
        else:
            data, names, meta = _read_numpy(p)
        if sampling_rate is None:
            raise ValueError(
                f"{p.name}: this format does not store a sampling rate. Provide "
                f"one in the cohort manifest - it cannot be inferred, and a wrong "
                f"value invalidates every spectral feature."
            )
        fs = float(sampling_rate)

    finite = np.isfinite(data)
    if not finite.all():
        n_bad = int((~finite).sum())
        logger.warning("%s: %d non-finite samples replaced by channel median",
                       p.name, n_bad)
        for i in range(data.shape[0]):
            row = data[i]
            good = np.isfinite(row)
            row[~good] = np.median(row[good]) if good.any() else 0.0

    meta["n_nonfinite_replaced"] = int((~finite).sum())
    return Recording(
        data=data,
        channel_names=names,
        sampling_rate=fs,
        subject_id=str(subject_id),
        recording_id=recording_id or p.stem,
        label=None if label is None else int(label),
        source_path=str(p),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Cohort manifest
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = ("path", "subject_id", "label")


def load_cohort(manifest_path: str, root: Optional[str] = None) -> Cohort:
    """Load every recording listed in a CSV manifest.

    The manifest must have columns: path, subject_id, label
    Optional columns: recording_id, sampling_rate, and any study covariates
    (age, sex, site, ...) which are carried through into `Recording.metadata`
    so they can be checked as confounds later.

    Example
    -------
    path,subject_id,label,sampling_rate,site
    data/s01_a.edf,S01,0,,siteA
    data/s01_b.edf,S01,0,,siteA
    data/s02_a.csv,S02,1,250,siteB
    """
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise CohortError(
            f"manifest is missing required column(s) {missing}. A manifest "
            f"without subject_id cannot support subject-disjoint validation."
        )

    base = Path(root) if root else Path(manifest_path).parent
    covariate_cols = [c for c in manifest.columns
                      if c not in set(MANIFEST_COLUMNS) | {"recording_id", "sampling_rate"}]

    recordings: List[Recording] = []
    failures: List[str] = []
    for _, row in manifest.iterrows():
        rel = str(row["path"])
        full = Path(rel) if Path(rel).is_absolute() else base / rel
        fs = row.get("sampling_rate")
        fs = None if (fs is None or (isinstance(fs, float) and np.isnan(fs))) else float(fs)
        try:
            rec = read_recording(
                str(full),
                subject_id=str(row["subject_id"]),
                recording_id=str(row["recording_id"]) if "recording_id" in manifest.columns
                             and not pd.isna(row.get("recording_id")) else None,
                label=None if pd.isna(row["label"]) else int(row["label"]),
                sampling_rate=fs,
            )
            for c in covariate_cols:
                rec.metadata[c] = row[c]
            recordings.append(rec)
        except Exception as exc:
            failures.append(f"{rel}: {exc}")

    if failures:
        logger.error("%d recording(s) could not be read:\n  %s",
                     len(failures), "\n  ".join(failures))
    if not recordings:
        raise CohortError("no recordings could be loaded from the manifest")

    cohort = Cohort(recordings)
    for w in cohort.audit():
        logger.warning("COHORT DESIGN: %s", w)
    return cohort


def write_manifest_template(path: str) -> None:
    """Write an empty manifest with the expected header, as a starting point."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("path,subject_id,label,recording_id,sampling_rate,site,age,sex\n")
        fh.write("# label: 0 = negative/control, 1 = positive\n")
        fh.write("# subject_id: MUST be identical for all recordings of one person\n")
        fh.write("# sampling_rate: required for csv/mat/npy, leave blank for edf\n")
