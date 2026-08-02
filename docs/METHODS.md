# Methods

This document is written so that it can be adapted into the Methods section of
a manuscript, and so that a reader can tell exactly what was done without
reading the source.

---

## 1. Unit of analysis and validation design

This is the decision that determines whether every other number in a study is
meaningful, so it comes first.

An EEG recording yields many epochs and many channels. These are **not
independent observations**. Epochs from one recording share the montage, the
electrode impedances, the amplifier, the medication state, the vigilance level
and the individual's own spectral fingerprint. If they are treated as
independent samples and split at random across cross-validation folds, a
classifier can score highly by recognising *which person* a row came from,
without having learned anything about the condition being studied. Saeb et al.
(2017) named this the record-wise versus subject-wise error; Kapoor and
Narayanan (2023) found it to be the most common form of leakage across
machine-learning-based science.

**Protocol.** All cross-validation is subject-disjoint. Splitting uses
`StratifiedGroupKFold` with `subject_id` as the group, so every recording,
epoch and channel belonging to one person falls entirely inside one fold. A
`subject_id` is mandatory in the cohort manifest and a recording without one is
rejected at load time. `ValidationConfig.grouping` accepts only `"subject"` or
`"recording"`; there is no setting that disables grouping.

**Verification.** `tests/test_leakage.py` builds a synthetic cohort with
subject-level structure and labels drawn independently of the signal, and
asserts that the pipeline returns chance performance. The same file asserts that
the ungrouped protocol *does* inflate on that data, so the guard is known to be
testing something real. `examples/make_demo_cohort.py --effect 0.0` provides the
same negative control end-to-end. **Run it before trusting any result.**

**Reporting unit.** Predictions are made per (epoch, channel) and aggregated to
one score per recording before any metric is computed, so that a long recording
does not contribute more weight than a short one.

---

## 2. Cohort audit

Before any analysis, `Cohort.audit()` checks for design problems that would
make a result uninterpretable regardless of how the model is validated:

- fewer than 20 subjects, or fewer than 5 in the smaller class;
- subjects appearing in both classes;
- **sampling rates that differ systematically between classes** — if the two
  groups were recorded on different equipment, any classifier can separate them
  by acquisition system alone;
- recording durations differing by more than 3× between classes;
- disjoint channel montages between classes.

These are reported as warnings, not errors, because some are legitimate in
specific designs. Every one of them belongs in a Limitations section.

---

## 3. Preprocessing

Order of operations, with the reason for each:

1. **Resample** to a common rate (default 250 Hz) with polyphase filtering.
   Entropy, fractal and spectral features are all functions of the sampling
   grid; mixing rates makes values incomparable.
2. **DC removal**, so the high-pass does not ring on a large offset.
3. **Mains notch**, at 50 or 60 Hz and in-band harmonics — applied *before* the
   low-pass. A 50 Hz notch after a 30 Hz low-pass removes nothing.
4. **Band-pass**, default 0.5–45 Hz, zero-phase, in second-order-section form
   (`sosfiltfilt`). SOS rather than transfer-function coefficients because
   high-order IIR filters expressed as polynomials accumulate numerical error.
   Zero-phase because phase distortion would corrupt any waveform-shape feature;
   the cost is that the effective order is doubled.
5. **Common average reference**, skipped below 8 channels where it would inject
   one channel's artefact into all the others.
6. **Wavelet denoising** (optional, off by default). Soft thresholding is a
   non-linear operation that changes entropy and fractal statistics in ways
   that are hard to characterise, so it is an explicit choice rather than a
   default.
7. **Epoching** into fixed-length windows specified in **seconds** (default 8 s).
   A window specified in samples is a different amount of brain activity at
   every sampling rate.
8. **Quality control** per epoch and channel.
9. **Amplitude normalisation** (default: none), applied last and per epoch so it
   cannot leak information across epochs.

The realised filter magnitude response at several probe frequencies is written
into the run log, so a reader can see what the filter did rather than infer it
from nominal cut-offs.

### Band/passband consistency

Frequency bands are validated against the filter passband at configuration
time. Requesting 30–45 Hz gamma power through a 1–30 Hz filter raises a
`ConfigError` rather than silently returning filter roll-off. This was a real
defect in version 1.

### Automatic epoch rejection

An epoch–channel is rejected if it is flat (>30 % zero-slope samples), clipped
(>2 % of samples at the rail), exceeds a peak-to-peak amplitude threshold
(default 500 µV), has zero variance, or carries more than 60 % of its power
above 30 Hz — a simple EMG index following the logic of Nolan et al. (2010).
Every rejection is recorded with its reason and the acceptance rate is reported.

No automatic ICA component rejection is performed. Deciding which independent
components are ocular or muscular requires human review or a trained
classifier; guessing wrong silently corrupts the signal.

---

## 4. Feature extraction

81 features across five domains, each carrying its own definition,
physiological interpretation, references, computational complexity and
amplitude dependence. `eegtumor.features.feature_dictionary(cfg)` exports the
whole table, and every run writes it as `feature_dictionary.csv` alongside the
results.

| Domain | n | Content |
|---|---|---|
| Time | 17 | moments, RMS, IQR, MAD, zero-crossing rate, line length, Hjorth, Teager energy |
| Spectral | 27 | absolute and relative band power, four band ratios, peak/mean/median frequency, spectral edge 90/95, entropy, centroid, spread, flatness, aperiodic exponent and offset |
| Time–frequency | 26 | DWT sub-band energy, relative energy, SD and kurtosis; wavelet entropy; energy concentration |
| Entropy | 5 | sample, permutation, amplitude Shannon, multiscale sample entropy |
| Complexity | 6 | Higuchi, Katz, Petrosian FD; DFA α; Hurst R/S; Lempel-Ziv |

**The features are not equally defensible, and the documentation says so.**
The classical scalp correlate of a structural lesion is focal slowing —
increased delta and theta, reduced alpha, frequently lateralised. The band
ratios, in particular `sp_ratio_slowing_index` (δ+θ)/(α+β), quantify exactly
that and have a substantial clinical literature. The entropy and complexity
families are exploratory by comparison, and they carry an explicit caveat: they
estimate scaling behaviour, and band-pass filtering deliberately destroys
scaling behaviour outside the passband, so part of every such value reflects the
filter rather than the brain.

### Implementation notes

- **PSD** uses Thomson's multitaper estimator with DPSS tapers by default, which
  has lower variance than a single Welch taper at equal resolution. Welch is
  available for comparability with the existing literature.
- **Aperiodic component.** Total band power confounds oscillatory activity with
  the broadband 1/f background. The background exponent and offset are extracted
  as separate features by a robust log-log fit with the strongest peaks removed,
  approximating the periodic/aperiodic split of Donoghue et al. (2020).
- **Feature names are derived analytically** from the configuration, never by
  pushing a signal through the extractor. The vector width therefore cannot
  depend on epoch length, and the global random state is never touched.
- **Sample entropy** uses k-d tree neighbour counting rather than an O(n²)
  double loop. Values are identical to the textbook implementation — verified in
  `tests/test_features.py` against a direct reference — and roughly 40× faster.
- **Wavelet depth is validated, not clamped.** An epoch too short for the
  configured decomposition level raises rather than silently producing a shorter
  feature vector.

### Estimator validation

Every non-trivial estimator is tested against signals with known answers:
Higuchi FD ≈ 2.0 for white noise, ≈ 1.0 for a sine, ≈ 1.5 for Brownian motion;
DFA α ≈ 0.5 for white noise and ≈ 1.5 for Brownian motion; multitaper PSD
integrating to the signal variance (Parseval); Hjorth mobility ≈ 2πf/fs for a
sine; relative band powers summing to 1.

---

## 5. Feature selection

Selection is a **fitted** step. Choosing features on the whole dataset and then
cross-validating the classifier produces near-zero apparent error on pure noise
(Ambroise & McLachlan, 2002). The selector is therefore implemented as a
scikit-learn transformer (`FeatureFunnel`) and placed inside the `Pipeline`, so
cross-validation refits it on each training fold automatically and no caller has
to remember to do the right thing.

Stages, cheapest first:

```
zero/near-zero variance → Spearman correlation filter → mutual information
→ embedded importance (elastic-net or random forest) → cap at max_features
```

The correlation filter keeps, from each redundant pair, the feature with the
stronger univariate association with the outcome.

### Stability

A feature selected in one split but not another is noise that happened to be
useful once. `selection_stability()` repeats the whole funnel on **subject-level**
resamples — not row-level, for the same reason the CV is grouped — and reports
per-feature selection frequency plus the Nogueira et al. (2018) stability index,
which corrects for subset size and chance agreement. Report the frequencies
alongside any importance ranking.

---

## 6. Modelling and validation

**Model set.** Elastic-net logistic regression (mandatory baseline), random
forest, extremely randomised trees, RBF SVM, and XGBoost/LightGBM when
installed. The linear baseline is not decoration: regularised linear models
match or beat complex alternatives on most clinical prediction tasks, and if
nothing beats it, that is the result to report.

**Nested cross-validation.** Hyper-parameters are tuned by Optuna in an inner
subject-disjoint loop; performance is estimated on the outer fold, which the
tuning never saw. Selecting the best model by cross-validated score and then
reporting that same score — the version 1 behaviour — is selection on the test
set and is optimistically biased.

**Repeats.** The whole outer loop is repeated with different seeds. Reported
performance is the mean across repeats with its standard deviation, because a
single fold split on a small cohort is close to a coin toss.

**Calibration.** Probabilities are calibrated with Platt scaling by default.
Isotonic regression is available but overfits below roughly 1000 samples.

**Aggregation.** Epoch/channel probabilities are collapsed to one score per
recording by trimmed mean (default), median, mean or max. The dispersion across
epochs is retained and reported: it is a far more honest uncertainty signal than
distance from 0.5, which version 1 mislabelled as "confidence".

### Metrics

Never accuracy alone. Each run reports ROC-AUC, PR-AUC (which is the informative
one under class imbalance), sensitivity at 90 % specificity, Brier score,
balanced accuracy, sensitivity, specificity, PPV, NPV, F1 and MCC, together with
prevalence.

### Inference

- **Confidence intervals** by bootstrap over recordings.
- **DeLong's test** for paired ROC-AUC differences between models on the same
  recordings.
- **McNemar's test** for paired differences in classification decisions.
- **Permutation test** with labels shuffled *at subject level*, giving an
  empirical null for the whole pipeline including selection.

Report the interval, not just the point estimate. On a 20-subject cohort a
95 % CI on AUC routinely spans 0.4 — that width is the finding.

---

## 7. Explainability

Global and local SHAP attribution, using the cheapest **exact** explainer for
the model family: `TreeExplainer` for forests and boosting, `LinearExplainer`
for linear models, and the model-agnostic permutation explainer only as a
fallback. This matters for accuracy as well as speed — the generic explainer
only approximates Shapley values.

Grouped permutation importance is also available, permuting whole subjects
rather than rows.

**SHAP attributions describe the model, not the biology.** A feature with high
attribution is one the model relied on; whether it reflects pathology, a
confound, or an artefact of your cohort is a separate question that
explainability cannot answer. Do not present it as evidence of causality.

---

## 8. Reproducibility

Every run writes a directory named with a timestamp and a configuration hash
containing: the exact configuration used, the environment (package versions),
the full feature table, the feature dictionary, per-model metrics, pairwise
model comparisons, feature stability, publication-resolution figures in PNG and
PDF, and a self-contained HTML report.

A published result should be reproducible from the manifest plus
`config_used.yaml` alone:

```bash
python -m eegtumor.cli run --manifest cohort.csv \
                           --config results/run_.../config_used.yaml \
                           --out reproduction/
```

---

## References

- Ambroise, C., McLachlan, G.J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 99(10), 6562–6566.
- Bandt, C., Pompe, B. (2002). Permutation entropy. *Phys Rev Lett* 88(17), 174102.
- Bigdely-Shamlo, N. et al. (2015). The PREP pipeline. *Front Neuroinform* 9, 16.
- Costa, M., Goldberger, A.L., Peng, C.-K. (2002). Multiscale entropy analysis. *Phys Rev Lett* 89(6), 068102.
- DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). Comparing areas under two or more correlated ROC curves. *Biometrics* 44(3), 837–845.
- Donoghue, T. et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nat Neurosci* 23, 1655–1665.
- Esteller, R. et al. (2001). A comparison of waveform fractal dimension algorithms. *IEEE Trans Circuits Syst* 48(2), 177–183.
- Finnigan, S., van Putten, M.J.A.M. (2013). EEG in ischaemic stroke. *Clin Neurophysiol* 124(1), 10–19.
- Gloor, P., Ball, G., Schaul, N. (1977). Brain lesions that produce delta waves in the EEG. *Neurology* 27(4), 326–333.
- Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory. *Physica D* 31(2), 277–283.
- Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalogr Clin Neurophysiol* 29(3), 306–310.
- Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9), 100804.
- Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Nogueira, S., Sechidis, K., Brown, G. (2018). On the stability of feature selection algorithms. *JMLR* 18(174), 1–54.
- Nolan, H., Whelan, R., Reilly, R.B. (2010). FASTER. *J Neurosci Methods* 192(1), 152–162.
- Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides. *Phys Rev E* 49(2), 1685–1689.
- Richman, J.S., Moorman, J.R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol* 278(6), H2039–H2049.
- Saeb, S. et al. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience* 6(5), 1–9.
- Schaul, N. (1998). The fundamental neural mechanisms of electroencephalography. *Electroencephalogr Clin Neurophysiol* 106(2), 101–107.
- Thomson, D.J. (1982). Spectrum estimation and harmonic analysis. *Proc IEEE* 70(9), 1055–1096.
- Widmann, A., Schröger, E., Maess, B. (2015). Digital filter design for electrophysiological data. *J Neurosci Methods* 250, 34–46.
