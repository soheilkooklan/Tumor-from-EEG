# Changelog

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
  `hjorth_activity` identical for every row — 13 % of the feature space.
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
