# Datasets

## The uncomfortable starting point

**There is no public EEG dataset with confirmed, imaging-verified brain-tumour
labels.**

The literature applying EEG to tumour detection is small and old. The most-cited
work (Murugesan & Sukanesh, 2009; Sharanreddy & Kulkarni, 2013) is
single-centre, with sample sizes in the tens, and the data are not available for
reanalysis. Meanwhile, the 2020–2025 brain-tumour detection literature is
overwhelmingly MRI-based; EEG barely appears in it.

This has a direct consequence for how this repository should be read. The
tumour application is an **untested hypothesis**. The software is a validated
*pipeline*, not a validated *method*.

It also has a consequence for how you should read any paper, including any
future paper using this tool, that reports high accuracy for EEG-based tumour
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

Published accuracy on the TUAB evaluation split mostly falls between about
75 and 90 %, with a few recent reports a little above 90 % (Zhu, Canham and
Western, 2023, arXiv:2303.06386, which tabulates earlier results). That range
is worth keeping as a sanity anchor: it is the state of the art on a *much
easier* task, with a hundred times more data than a typical tumour study. Any claim of
95 %+ on EEG tumour detection from 40 recordings should be treated as a leakage
report until proven otherwise.

Request access: <https://isip.piconepress.com/projects/nedc/html/tuh_eeg/>

Access needs an institutional address and an institutional email, and the
form has to be signed by someone at that institution. An independent
researcher without a current affiliation cannot get it alone.

### Other TUH subsets

- TUEP: epilepsy vs non-epilepsy, 200 subjects. Heavily imbalanced at the
  recording level despite balanced subject counts, which makes it a good test of
  the aggregation and PR-AUC reporting.
- TUAR: artifact annotations, 213 subjects. Useful for evaluating the
  automatic epoch-rejection step against human labels.
- TUSL: slowing events. The most directly relevant public corpus to the
  focal-slowing hypothesis, since slowing is the mechanism by which a structural
  lesion would show up on scalp EEG at all.

### Others worth knowing

- CHB-MIT Scalp EEG (PhysioNet): paediatric seizure recordings, freely
  downloadable, good for a first end-to-end run without an access request.
- Bonn University EEG: five 100-segment sets, widely used and widely
  misused. Segments are pre-selected, artefact-free and short; near-perfect
  accuracies on it do not transfer. Treat published Bonn results with caution,
  and note that some EEG-tumour papers have effectively relabelled it.
- TDBRAIN and LEMON: large resting-state cohorts with rich metadata,
  useful as normative controls and for testing whether your features are picking
  up age rather than pathology.

## Corpora used to test version 2.2.0

Both are free to download without an access request, and both have a script in
`examples/` that fetches the data and writes a manifest. Results are in
[REAL_DATA.md](REAL_DATA.md).

OpenNeuro ds004504 (Miltiadous et al., 2023, CC0): eyes-closed resting EEG
from 36 people with Alzheimer's disease, 23 with frontotemporal dementia and 29
cognitively normal controls. 19 channels, 500 Hz, EEGLAB `.set` files, about
2.1 GB. One recording per person, so subject and recording grouping coincide.
Alzheimer's disease slows the scalp EEG, which is the same mechanism the band
ratio features were written for, so it is a fair test of them.
`examples/openneuro_ds004504.py`.

NMT-4K-EEG v1.1 (Masood et al., 2026, Zenodo, doi:10.5281/zenodo.21041582,
CC BY 4.0): 4,500 clinical EDF recordings labelled normal or abnormal, with a
fixed train/evaluation split. It is the closest freely available thing to
TUAB. Two things to know before using it. There is no patient identifier, so
repeat recordings of one person cannot be grouped. And the recording year is
strongly tied to the label (the share of abnormal recordings also differs by
split and hospital code), while in our balanced subset the year could be
predicted from normal EEG alone; year and hospital must be balanced or
stratified. The archive is a 33 GB split ZIP; `examples/nmt4k_subset.py` pulls
only the 240 recordings of the balanced subset (about 1.9 GB).

## Toward a real tumour cohort

Two routes exist, and both are more work than they look:

1. **Weak labels from clinical reports.** The full TUH EEG Corpus ships free-text
   neurologist reports. A cohort could in principle be assembled by identifying
   reports mentioning a neoplasm. The labels would be weak, unverified against
   imaging, and confounded by everything that puts a person in a hospital EEG
   suite. It is a legitimate starting point if, and only if, the weakness is
   stated prominently and a manual audit of a sample is reported.

2. **Prospective single-centre collection** with imaging-confirmed labels and
   matched controls. This is what would actually settle the question. It needs
   ethics approval, and it needs controls matched on age, medication and
   recording setup, because those are the confounds that will otherwise produce
   the result for you.

## Building a cohort manifest

The manifest is a CSV. `subject_id` is mandatory. It is what keeps every
recording from one person inside a single cross-validation fold.

```csv
path,subject_id,label,recording_id,sampling_rate,site,age,sex
recordings/aaaaamye_s001.edf,aaaaamye,0,,,siteA,54,F
recordings/aaaaamye_s002.edf,aaaaamye,0,,,siteA,54,F
recordings/aaaaanbw_s001.edf,aaaaanbw,1,,,siteA,61,M
data/control_07.csv,C07,0,,250,siteB,49,F
```

- `label`: 0 negative/control, 1 positive. Blank for unlabelled data to screen.
- `sampling_rate`: required for CSV/MAT/NumPy, which do not store it. Leave
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

This prints the cohort summary and flags structural confounds: disjoint
sampling rates between classes, class-correlated duration differences, subjects
appearing in both classes, montage mismatches, insufficient subject counts.

A cohort whose two classes were recorded on different equipment is separable by
acquisition system alone. No amount of correct cross-validation fixes that, and
the audit exists so you find out before you have a result you like too much to
question.
