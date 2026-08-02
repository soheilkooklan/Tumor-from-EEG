# Limitations

Written deliberately against this project's own interests. A limitations
section that only lists things that do not matter is worse than none.

---

## Scientific

**The central premise is unvalidated.** No public EEG dataset carries confirmed
brain-tumour labels. The pipeline has never been evaluated on the task the
repository is named after. Everything here is machinery waiting for data.

**EEG is a weak instrument for structural lesions.** A tumour affects the scalp
EEG indirectly, through focal slowing and amplitude attenuation caused by the
surrounding oedema and by cortical deafferentation. Small lesions, deep lesions,
and slow-growing lesions may produce nothing at all. A normal EEG does not
exclude a tumour, and this is not a limitation of the software but of the
modality.

**Focal slowing is non-specific.** The same delta and theta excess appears in
stroke, encephalopathy, migraine, post-ictal states, drowsiness and sedation.
Any classifier trained on slowing features will detect *abnormality*, not
*tumour*, unless the control group is matched on everything else that causes
slowing — which is very hard.

**Entropy and fractal features are compromised by their own preprocessing.**
They estimate scaling behaviour; band-pass filtering removes scaling behaviour
outside the passband. Part of every such value reflects the filter. They are
retained because they are extensively reported and cheap, and flagged as
exploratory in the feature dictionary. A result resting mainly on them deserves
more scepticism than one resting on band power.

**No spatial information is used.** Features are computed per channel and
aggregated. Lateralisation and focality — the properties that make EEG slowing
*suggestive of a lesion* rather than merely abnormal — are not modelled.
Connectivity and graph features are listed as future work, not implemented.

**Relative band powers are not independent.** They sum to one by construction.
Correlation filtering removes the most redundant pairs but the constraint
remains, and importance rankings across them should be read with that in mind.

---

## Statistical

**Small cohorts give wide intervals, and the interval is the finding.** On 20
subjects a 95 % bootstrap CI on ROC-AUC routinely spans 0.4. The point estimate
is close to meaningless on its own. This software reports intervals everywhere
and cannot stop anyone quoting only the mean.

**Correct validation does not fix a confounded cohort.** Subject-disjoint
cross-validation prevents the model from recognising the person. It does nothing
about a scanner difference, a site difference, an age difference or a medication
difference that correlates with the label. `Cohort.audit()` flags a handful of
the most mechanical confounds; the rest is study design, and no code can supply
it.

**Nested cross-validation on small data is itself high-variance.** The outer
estimate is unbiased with respect to hyper-parameter selection, but its variance
across seeds can be large. Repeats are run by default for this reason; report
the spread.

**Calibration on small samples is unreliable.** Platt scaling is the default
because isotonic regression overfits below roughly 1000 samples, but neither is
trustworthy at the cohort sizes typical of this field. Read calibration plots as
diagnostic, not as a guarantee.

**The aggregation step is not itself calibrated.** Averaging calibrated
epoch-level probabilities does not produce a calibrated recording-level
probability. Recording-level calibration would need its own fitting step on
held-out recordings.

**Multiple comparisons are not corrected.** Comparing several models across
several metrics inflates the family-wise error rate. DeLong and McNemar p-values
are reported per pair, uncorrected. Apply your own correction.

---

## Technical

**No ocular or muscular artefact correction.** Epochs are rejected, not
repaired. Automatic ICA component rejection without human review silently
corrupts signal, so it is not performed. On heavily contaminated recordings the
rejection rate can be high enough to leave too little data.

**No bad-channel interpolation and no montage harmonisation.** Channel names are
mapped onto 10-20 nomenclature on a best-effort basis. Recordings with different
montages produce feature tables that are not strictly comparable.

**Common average reference is applied blindly** above 8 channels. With poor
spatial coverage it can spread one channel's artefact across all of them.

**MAT and NumPy loading is heuristic.** The largest 2-D numeric array is taken
as the signal and the orientation is inferred from shape. This is logged, but it
can be wrong. EDF is the reliable path.

**Sampling rate cannot be inferred** for CSV, MAT and NumPy. It must be supplied
in the manifest. A wrong value invalidates every frequency-domain feature and
nothing downstream can detect it.

**Speed.** Roughly 60 ms per epoch-channel, dominated by sample entropy even
after the k-d tree rewrite. A 100-recording, 19-channel cohort takes tens of
minutes. Extraction is not parallelised across recordings.

**Multiscale entropy returns NaN below 500 coarse-grained points**, following
the usual length guidance. At 8 s epochs and 250 Hz this limits usable scales to
1 and 2.

**Deep learning is not implemented,** deliberately. CNN, LSTM and transformer
architectures need thousands of labelled recordings; on research-scale data they
overfit while looking impressive in-sample. If a large labelled corpus becomes
available, a separate module operating on raw or time-frequency input would be
the right addition — not a bolt-on to this feature pipeline.

---

## Software

**The GUI is not a substitute for the CLI.** Reproducible published results
should come from `python -m eegtumor.cli run` with a version-controlled config,
not from a sequence of clicks.

**Tests validate the estimators and the protocol, not the science.** 32 tests
check that Higuchi FD returns 2.0 on white noise and that grouped CV returns
chance on label-free data. No test can check that your cohort means what you
think it means.

**Single-author, no external review.** This code has not been independently
audited. Version 1 ran cleanly for a long time while producing meaningless
numbers; the defect was structural, not a crash. Treat version 2 as better, not
as correct.

---

## What would improve this most

In rough order of value:

1. A cohort with imaging-confirmed labels and matched controls.
2. Validation of the whole pipeline on TUAB, published with the confidence
   intervals, so the machinery has a known operating point.
3. Spatial and connectivity features — lateralisation is the part of the
   hypothesis currently not modelled at all.
4. Independent replication on a second site.
5. Comparison against a clinician reading the same recordings, which is the only
   benchmark that would tell you whether any of this is useful.
