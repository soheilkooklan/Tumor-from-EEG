# Changelog

## [2.2.0] - 2026-11

First release tested on real recordings: OpenNeuro ds004504 (Alzheimer's and
frontotemporal dementia vs controls) and a balanced 240-recording subset of
NMT-4K-EEG (normal vs abnormal clinical EEG). Results are in
`docs/REAL_DATA.md`. Every change below came from running the package on that
data.

### Fixed

- Non-EEG channels were treated as EEG. Clinical EDF files carry ECG, ear
  references (A1, A2) and trigger channels, and MNE types them all as EEG.
  They became feature rows, and they entered the common average reference,
  which spreads cardiac signal into every scalp channel. A new
  `preprocessing.channels` setting (default `scalp`) drops them before
  resampling and referencing; `all` keeps the old behaviour and a list keeps
  named channels only.
- The whole cohort was read into memory before extraction. Ten-minute,
  19-channel recordings take about 60 MB each, so 88 of them did not fit
  comfortably in 7 GB. Runs now read headers only, load one recording at a
  time and release it.

- The elastic-net search range for C started at 1e-3. On ds004504 the tuner
  chose a value within 1.7e-3 of that floor in 6 of 10 outer folds, i.e. it
  was clipped. The range now starts at 1e-5.
- The stability analysis ran feature selection on unscaled features while the
  model pipeline ran it on scaled ones; the elastic-net stage is
  scale-dependent, so the report described a different selection from the one
  the model used. The stage now standardises internally, which makes the
  selection invariant to any per-feature affine rescaling of its input.

### Added

- Readers for EEGLAB `.set` (the usual OpenNeuro format), BrainVision `.vhdr`
  and FIF, through MNE.
- `--jobs N` for parallel feature extraction (spawned processes); output is
  identical for any N.
- `examples/openneuro_ds004504.py`: downloads ds004504 and writes manifests for
  AD vs CN, FTD vs CN and AD vs FTD.
- `examples/nmt4k_subset.py`: pulls the 240 recordings listed in
  `examples/nmt4k_subset240.csv` out of the 33 GB split NMT-4K archive with
  HTTP range requests (about 1.9 GB) and writes a manifest.
- A no-learning baseline in every run: one named feature
  (`validation.baseline_feature`, default `sp_ratio_theta_alpha`), median per
  recording, used directly as the score, with a subject-level bootstrap
  interval. On ds004504 it matched the trained models (0.867 for AD vs CN),
  which is why it is now printed next to them. Written to `baseline.json`.
- `docs/REAL_DATA.md` with the real-data results and how to reproduce them.
- Tests for the readers, lazy loading, parallel extraction, channel selection,
  the ECG/reference interaction and the baseline.

### Changed

- Lempel-Ziv complexity uses an exhaustive LZ76 parse with substring search in
  C. Counts are identical to the previous Kaspar-Schuster loop (checked on
  8,560 signals and in the test suite) and it is about 12 times faster.
  Feature extraction on a 10-minute 19-channel recording dropped from about
  137 s to about 68 s on one core.
- The model-based ranking stages of feature selection use a stratified sample
  of at most `selection.ranking_max_rows` training rows (default 20,000).
  One selection fit on 58,000 rows went from 169 s to about 30 s.
- Nested cross-validation caches the fitted imputer, scaler and selection per
  training split (joblib), so tuning trials share one selection fit. Results
  are identical with and without the cache; a nested run was about 3 times
  faster on the test cohort.

## [2.1.0] - 2026-10

A correctness and honesty release. No change to the feature set.

### Fixed

- Calibration folds were row-level. `CalibratedClassifierCV` inside each
  training fold split rows at random, so the sigmoid was fitted on held-out
  scores from subjects the model had partly seen. Calibration folds are now
  subject-disjoint (`modeling.calibration_splits`), in cross-validation, in the
  final explained model and in the GUI screening tab. Test fold separation was
  never affected, so ROC-AUC was not inflated by this; probabilities were.
- `ValidationConfig.grouping = "recording"` was accepted but ignored, and the
  HTML report then claimed recording-level splitting while the code split on
  subject. The setting is now honoured, and refused with a clear error when any
  subject has more than one recording.
- `optimisation = "random"` was documented but fell through to no tuning. It
  now runs Optuna's random sampler. Unknown values raise `ConfigError`.
- The bootstrap recomputed all thirteen metrics on every resample. It now
  computes only the requested one, which makes a cross-validation run about
  15 times faster on the test cohort with identical values.
- `calibrate()` forced at least two folds even when a class had one member,
  then could not reach its own fallback.
- The per-fold subject-overlap check was an `assert`, which `python -O`
  removes. It now raises `RuntimeError`.
- The demo cohort generator redrew each subject's alpha peak, amplitude, noise
  floor, 1/f slope and lesion channels for every session, although its
  docstring said they were shared. Two recordings of one person were therefore
  no more alike than two people, and the demo could not show leakage across
  recordings. The profile is now drawn once per subject, with small session
  jitter. Demo numbers change accordingly.

### Changed

- The subject-level permutation null was implemented but never called. It is
  now wired into a run as an option: `eegtumor run --permutations N` or
  `validation.permutation_tests` in the config (default 0, since each
  permutation is a full cross-validation). Results, including every null
  score, go to `permutation_null.json` and the HTML report.
- Figure titles gave the pooled row count (for example "n = 124 recordings"
  for 62 recordings scored in two repeats). They now say "62 recordings x 2
  repeats, pooled".
- The permutation null no longer prints one log line per permutation.
- `metrics.csv` now carries the bootstrap interval bounds
  (`roc_auc_ci_low`, `roc_auc_ci_high` and the same for PR-AUC and balanced
  accuracy); before, they appeared only in the HTML report.
- The HTML report now states that the family shown in the confusion matrix and
  SHAP figures was chosen by outer-fold ROC-AUC, so its score includes that
  choice.
- Documentation corrected where it claimed more than the code did: the
  recording-level score is not re-calibrated; confidence intervals resample
  subjects, not recordings; Kapoor and Narayanan (2023) list subject-wise
  leakage among the types they found, rather than ranking it first; the
  TUAB accuracy range is widened to what published results actually show.

### Added

- Eight regression tests: subject-disjoint calibration folds, the grouping
  guard, optimisation validation, and the permutation null on a null cohort
  and on a cohort with a real effect.

## [2.0.0] — 2026

Complete rewrite. Version 2 is not backward compatible with version 1; the
old entry point is preserved under `legacy/`.

### Why

A controlled experiment motivated the rewrite. A synthetic cohort was built in
which class labels were statistically independent of the signal (40 subjects,
19 channels, random labels). The correct answer for any honest protocol is
ROC-AUC = 0.500.

| Protocol | ROC-AUC on label-free data |
|---|---|
| v1 (global scaler + channel-level `StratifiedKFold`) | 0.957 ± 0.010 |
| v2 (subject-disjoint `StratifiedGroupKFold` + fold-internal pipeline) | 0.398 ± 0.230 |

Version 1 was measuring subject identity, not pathology.

### Fixed — correctness

- **Grouping leakage.** Channels and epochs from one subject were split across
  train and test folds. `subject_id` is now mandatory and all splitting is
  subject-disjoint. `ValidationConfig.grouping` has no setting that disables it.
- **Preprocessing leakage.** `StandardScaler` was fitted on the whole dataset
  before cross-validation. Imputation, scaling and selection are now Pipeline
  steps, refitted per training fold.
- **Selection on the test set.** The best model was chosen by cross-validated
  score and that same score was reported. Replaced with nested cross-validation.
- **Five features were analytically constant.** Per-epoch z-scoring forced
  mean = 0 and SD = 1, making `mean`, `std`, `variance`, `rms` and
  `hjorth_activity` identical for every row, 13 % of the feature space.
  Amplitude normalisation now defaults to `none`, and degenerate features are
  excluded automatically when it is enabled.
- **Gamma power was measured through a filter that removed it.** The band-pass
  was 1–30 Hz while the gamma band was defined as 30–45 Hz. Bands are now
  validated against the passband at configuration time and the default passband
  is 0.5–45 Hz.
- **The mains notch did nothing.** It ran after the 30 Hz low-pass. It now runs
  before, together with in-band harmonics.
- **Feature-vector width depended on window length.** Wavelet depth was clamped
  silently, so a 55-sample window produced 35 features and a 56-sample window
  produced 37. Depth is now validated, not clamped, and names are derived
  analytically from the configuration.
- **`feature_names()` perturbed the global random state** by pushing a random
  signal through the extractor, breaking `random_state` reproducibility.
- **"Confidence" was |p − 0.5|,** which is not an uncertainty measure. Replaced
  with bootstrap confidence intervals and epoch-level dispersion.
- **Isotonic calibration on small samples.** Default is now Platt scaling.

### Fixed — numerics

- `sosfiltfilt` replaces `filtfilt(b, a)`; high-order IIR filters expressed as
  polynomial coefficients accumulate numerical error.
- Recordings are resampled to a common rate. Feature values are not comparable
  across sampling rates, and if rate correlates with class the classifier learns
  the equipment.
- Epochs are specified in seconds, not samples.
- Wavelet denoising uses level-dependent thresholds rather than one global
  threshold from the finest detail band.

### Verified as already correct in v1

- The Higuchi fractal dimension implementation (white noise 2.005, sine 1.012,
  Brownian 1.509 against expected 2.0 / 1.0 / 1.5).
- The Hurst exponent formulation.
- The decision to exclude CNN/LSTM/transformer architectures at this data scale.

### Added

- 81 documented biomarkers across five domains, each carrying its definition,
  physiological interpretation, references, complexity and amplitude dependence;
  exported as `feature_dictionary.csv`.
- Plugin registry: a new biomarker needs one module and one `register_group`
  call, with no changes to existing files.
- Multitaper (DPSS) power spectral density.
- Aperiodic 1/f exponent and offset, separating the broadband background from
  oscillatory power.
- Band ratios including the (δ+θ)/(α+β) slowing index.
- Katz and Petrosian fractal dimension, DFA, Lempel-Ziv complexity, multiscale
  sample entropy.
- Staged feature-selection funnel with subject-level stability analysis and the
  Nogueira stability index.
- Nested grouped cross-validation with Optuna hyper-parameter search.
- Bootstrap confidence intervals, DeLong test, McNemar test, subject-level
  permutation null.
- Probability calibration and epoch→recording aggregation.
- Exact SHAP explainers by model family, plus grouped permutation importance.
- `Cohort.audit()`: automatic detection of structural confounds.
- Publication-resolution figures (PNG + PDF) and a self-contained HTML report.
- Reproducible CLI: `run`, `audit`, `features`, `dict`.
- Seven-tab GUI with a worker thread and honest progress reporting.
- Synthetic cohort generator including a null-effect negative control.
- Scripted screenshot capture, so documentation images track the software.
- 32 tests: estimator correctness against known analytic answers, plus a
  leakage regression suite.

### Performance

- Sample entropy rewritten with k-d tree neighbour counting: identical values,
  ~40× faster.
- Feature extraction ~16× faster overall (291 s → 18 s for one 19-channel
  recording).
- SHAP for linear models ~100× faster via exact `LinearExplainer`.

### Changed — licence

Version 2 is released under **PolyForm Noncommercial 1.0.0**. Version 1 remains
under MIT; that grant is irrevocable for the code it covered and is preserved in
`LICENSE-v1-MIT.txt`. A noncommercial licence is not OSI-approved open source.

---

## [1.0.0] — 2024

Initial release. Wavelet denoising, Butterworth band-pass, stationary wavelet
transform features, MLP classifier, Tkinter GUI. MIT licence.
