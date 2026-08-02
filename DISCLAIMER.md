# Disclaimer and acceptable use

## This is not a medical device

Tumor-from-EEG is research software. It has not been evaluated or approved by
any regulatory authority, it carries no CE mark and no FDA clearance, and it
has not been validated on any clinical population.

Nothing this software produces is a diagnosis, a screening result, a
recommendation, or a basis for any decision about a person's care. Do not use
it to inform clinical management, to decide whether someone needs imaging, to
reassure anyone, or to alarm anyone.

Brain tumours are diagnosed with structural imaging — MRI or CT — interpreted
by a qualified clinician, together with the clinical history and examination.
An EEG cannot confirm a tumour and cannot exclude one. A normal EEG in a person
with a brain tumour is common; an abnormal EEG has many causes that have
nothing to do with a tumour.

If you are worried about symptoms, contact a doctor. Do not run your own EEG
through this software instead.

## Regulatory note

Software intended to diagnose, prevent, monitor, predict, or treat disease may
be regulated as a medical device — as Software as a Medical Device under
US FDA rules, or under Regulation (EU) 2017/745 in Europe, and equivalently in
other jurisdictions. This project makes no such claim and is not intended for
that purpose.

If you fork this project, add clinical claims, and put it in front of patients
or clinicians, those obligations become yours. That is not a formality: it
means conformity assessment, clinical evaluation, a quality management system,
and post-market surveillance. Please take advice before doing it.

## What the software actually does

It computes quantitative EEG features and evaluates how well a classifier can
separate whatever two groups you label as `0` and `1` in your cohort manifest,
under subject-disjoint cross-validation. That is all.

If you label your data "tumour" and "control", the output tells you how
separable *your particular dataset* is. It does not tell you whether the
separation reflects the pathology rather than the scanner, the ward, the
recording technician, the medication, the age difference between your groups,
or the time of day the recordings were made. Establishing that is the
scientific work, and this software cannot do it for you — it can only stop you
from fooling yourself in a few of the most common ways.

## Known limitation of the central premise

There is no public EEG dataset with confirmed, imaging-verified brain-tumour
labels. The small literature applying EEG to tumour detection is mostly
single-centre work from 2009–2013 with sample sizes in the tens, and the data
are not available for reanalysis. The tumour application of this pipeline is
therefore an **untested hypothesis**, not a validated method.

The pipeline itself can be validated, on corpora that do have real labels and
subject-disjoint splits. See `docs/DATASETS.md`.

## Acceptable use

The licence (PolyForm Noncommercial 1.0.0) restricts commercial use. Beyond
what the licence requires, the author asks that you do not use this software:

- to produce or support any claim about an identifiable person's health;
- to generate results for publication without reporting the cohort audit
  warnings, the confidence intervals, and the negative-control check;
- to advertise a diagnostic capability that has not been demonstrated;
- in any setting where a person might mistake its output for a clinical result.

These requests are not legally binding. They are what the software is for.

## Privacy

EEG recordings and the features derived from them can be identifying.
Anonymised does not mean unidentifiable. Handle any real recordings under the
ethics approval and data-protection rules that apply to them, and do not commit
data or analysis outputs to a public repository. The default `.gitignore`
excludes result directories and raw EEG formats for this reason.

## No warranty

The software is provided as-is, without warranty of any kind, and the author
accepts no liability for any use of it. See the LICENSE file for the full
terms.
