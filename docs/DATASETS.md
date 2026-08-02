# Datasets

## The uncomfortable starting point

**There is no public EEG dataset with confirmed, imaging-verified brain-tumour
labels.**

The literature applying EEG to tumour detection is small and old. The most-cited
work — Murugesan & Sukanesh (2009), Sharanreddy & Kulkarni (2013) — is
single-centre, with sample sizes in the tens, and the data are not available for
reanalysis. Meanwhile, the 2020–2025 brain-tumour detection literature is
overwhelmingly MRI-based; EEG barely appears in it.

This has a direct consequence for how this repository should be read. The
tumour application is an **untested hypothesis**. The software is a validated
*pipeline*, not a validated *method*.

It also has a consequence for how you should read any paper — including any
future paper using this tool — that reports high accuracy for EEG-based tumour
detection on a private dataset of a few dozen recordings. Ask which corpus, ask
whether the split was subject-disjoint, and ask for the confidence interval.

## What to validate the pipeline on instead

If you want to demonstrate that this software works, use a corpus that has real
labels and a subject-disjoint split. The obvious choice:

### TUH Abnormal EEG Corpus (TUAB)

The standard benchmark for binary normal/abnormal EEG classification. Roughly
2 993 recordings from about 2 383 subjects, 1 521 normal and 1 472 abnormal,
19 channels in the 10-20 system, mostly 250 Hz, around 20 minutes each. It ships
with an official train/evaluation split with **no subject overlap**, which is
exactly the property this pipeline is built around.

Published performance on TUAB sits around 85–89 % accuracy. That number is worth
internalising as a sanity anchor: it is the state of the art on a *much easier*
task, with a hundred times more data than a typical tumour study. Any claim of
95 %+ on EEG tumour detection from 40 recordings should be treated as a leakage
report until proven otherwise.

Request access: <https://isip.piconepress.com/projects/nedc/html/tuh_eeg/>

### Other TUH subsets

- **TUEP** — epilepsy vs non-epilepsy, 200 subjects. Heavily imbalanced at the
  recording level despite balanced subject counts, which makes it a good test of
  the aggregation and PR-AUC reporting.
- **TUAR** — artifact annotations, 213 subjects. Useful for evaluating the
  automatic epoch-rejection step against human labels.
- **TUSL** — slowing events. The most directly relevant public corpus to the
  focal-slowing hypothesis, since slowing is the mechanism by which a structural
  lesion would show up on scalp EEG at all.

### Others worth knowing

- **CHB-MIT Scalp EEG** (PhysioNet) — paediatric seizure recordings, freely
  downloadable, good for a first end-to-end run without an access request.
- **Bonn University EEG** — five 100-segment sets, widely used and widely
  misused. Segments are pre-selected, artefact-free and short; near-perfect
  accuracies on it do not transfer. Treat published Bonn results with caution,
  and note that some EEG-tumour papers have effectively relabelled it.
- **TDBRAIN**, **LEMON** — large resting-state cohorts with rich metadata,
  useful as normative controls and for testing whether your features are picking
  up age rather than pathology.

## Toward a real tumour cohort

Two routes exist, and both are more work than they look:

1. **Weak labels from clinical reports.** The full TUH EEG Corpus ships free-text
   neurologist reports. A cohort could in principle be assembled by identifying
   reports mentioning a neoplasm. The labels would be weak, unverified against
   imaging, and confounded by everything that puts a person in a hospital EEG
   suite. It is a legitimate starting point if — and only if — the weakness is
   stated prominently and a manual audit of a sample is reported.

2. **Prospective single-centre collection** with imaging-confirmed labels and
   matched controls. This is what would actually settle the question. It needs
   ethics approval, and it needs controls matched on age, medication and
   recording setup, because those are the confounds that will otherwise produce
   the result for you.

## Building a cohort manifest

The manifest is a CSV. `subject_id` is mandatory — it is what keeps every
recording from one person inside a single cross-validation fold.

```csv
path,subject_id,label,recording_id,sampling_rate,site,age,sex
recordings/aaaaamye_s001.edf,aaaaamye,0,,,siteA,54,F
recordings/aaaaamye_s002.edf,aaaaamye,0,,,siteA,54,F
recordings/aaaaanbw_s001.edf,aaaaanbw,1,,,siteA,61,M
data/control_07.csv,C07,0,,250,siteB,49,F
```

- `label` — 0 negative/control, 1 positive. Blank for unlabelled data to screen.
- `sampling_rate` — required for CSV/MAT/NumPy, which do not store it. Leave
  blank for EDF/BDF, where it is read from the header. It cannot be inferred, and
  a wrong value invalidates every spectral feature.
- Any extra columns (`site`, `age`, `sex`, …) are carried into
  `Recording.metadata` so you can check them as confounds.

Generate a template:

```bash
python -m eegtumor.cli dict > feature_dictionary.csv   # feature docs
python -c "from eegtumor.io import write_manifest_template as w; w('cohort.csv')"
```

## Always audit before you analyse

```bash
python -m eegtumor.cli audit --manifest cohort.csv
```

This prints the cohort summary and flags structural confounds — disjoint
sampling rates between classes, class-correlated duration differences, subjects
appearing in both classes, montage mismatches, insufficient subject counts.

A cohort whose two classes were recorded on different equipment is separable by
acquisition system alone. No amount of correct cross-validation fixes that, and
the audit exists so you find out before you have a result you like too much to
question.
