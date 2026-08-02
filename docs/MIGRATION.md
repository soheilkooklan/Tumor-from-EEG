# Migrating from version 1

Version 2 is a rewrite, not a patch. The repository name, URL and public
identity are unchanged; the code inside is new.

---

## File mapping

| Version 1 | Version 2 | Action |
|---|---|---|
| `Main File Tumor from EEG.py` | — | move to `legacy/`, do not delete |
| `config.py` | `eegtumor/config.py` | replaced; adds consistency validation |
| `data_loader.py` | `eegtumor/io.py` | replaced; adds `Recording`, `Cohort`, mandatory `subject_id` |
| `signal_processing.py` | `eegtumor/preprocessing.py` | replaced; SOS filters, resampling, epoching in seconds |
| `features.py` | `eegtumor/features/` (6 files) | replaced; registry + self-documenting specs |
| `pipeline.py` | merged into `eegtumor/features/__init__.py` | replaced |
| `models.py` | `eegtumor/modeling.py` | replaced; leakage-safe `Pipeline` |
| — | `eegtumor/validation.py` | **new**; nested grouped CV, bootstrap, DeLong, McNemar |
| — | `eegtumor/selection.py` | **new**; staged funnel + stability |
| `explain.py` | `eegtumor/explain.py` | replaced; exact explainers by model family |
| — | `eegtumor/reporting.py` | **new**; figures + HTML report |
| — | `eegtumor/experiment.py` | **new**; run orchestration |
| — | `eegtumor/cli.py` | **new**; reproducible headless entry point |
| `gui.py` | `eegtumor/gui.py` | replaced; 7 tabs, worker thread |
| `main.py` | `main.py` | replaced (thin launcher) |
| `test_pipeline.py` | `tests/test_features.py`, `tests/test_leakage.py` | replaced; 4 smoke tests → 32 real tests |
| `README.md` | `README.md` + `docs/` | rewritten |
| `LICENSE` (MIT) | `LICENSE` (PolyForm NC) + `LICENSE-v1-MIT.txt` | see below |

---

## Breaking changes you will actually hit

### `subject_id` is now mandatory

Version 1 took a list of files. Version 2 takes a CSV manifest, and a recording
without a `subject_id` is rejected at load time.

This is the whole point of the rewrite. Without subject identity there is no way
to keep one person's epochs out of the test fold, and the reported performance
measures subject recognition rather than pathology.

If your version 1 data was one file per person, the migration is mechanical:

```python
import csv, pathlib

rows = []
for label, folder in [(0, "data/normal"), (1, "data/tumor")]:
    for p in sorted(pathlib.Path(folder).glob("*.csv")):
        rows.append({"path": str(p), "subject_id": p.stem,
                     "label": label, "sampling_rate": 250})

with open("cohort.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, ["path", "subject_id", "label", "sampling_rate"])
    w.writeheader()
    w.writerows(rows)
```

If several files came from the same person, they **must** share one
`subject_id`. Getting this wrong silently reintroduces the version 1 defect.

### Windows are specified in seconds

Version 1 had "start point" and "end point" in raw sample indices, applied
identically to every file. At 250 Hz that was 12 s; at 500 Hz the same numbers
gave 6 s. Version 2 uses `epoch_seconds` and resamples everything to a common
rate first.

### The default passband changed

1–30 Hz → 0.5–45 Hz, so the gamma band lies inside it. If you keep a 30 Hz
low-pass you must also remove the gamma band from the configuration, or the
config validator will refuse to run — which is the intended behaviour.

### Feature names changed

All features are now prefixed by domain: `mean` → `td_mean`,
`rel_power_alpha` → `sp_rel_power_alpha`, `higuchi_fd` → `cx_higuchi_fd`.
Feature tables produced by version 1 are not compatible.

### Results will be lower, and that is the point

Any performance number carried over from version 1 was inflated by leakage.
Expect version 2 to report substantially worse figures on the same data with
much wider confidence intervals. That is the correction, not a regression.

---

## Step-by-step

```bash
cd Tumor-from-EEG
git checkout -b v2-research-grade

# 1. preserve version 1
mkdir -p legacy
git mv "Main File Tumor from EEG.py" legacy/
git mv LICENSE LICENSE-v1-MIT.txt
cp "Tumor from Sample EEG.jpg" legacy/ 2>/dev/null || true

# 2. unpack the version 2 tree into the repository root
#    (eegtumor/ tests/ docs/ examples/ tools/ configs/ .github/
#     README.md CHANGELOG.md CITATION.cff DISCLAIMER.md
#     pyproject.toml requirements.txt .gitignore main.py)

# 3. licence
curl -o LICENSE https://raw.githubusercontent.com/polyformproject/polyform-licenses/1.0.0/PolyForm-Noncommercial-1.0.0.md
#    then add this line near the top of LICENSE:
#    Required Notice: Copyright Soheil Kooklan (https://github.com/soheilkooklan)

# 4. verify before committing anything
pip install -r requirements.txt
pytest -q                                    # expect 32 passed

python examples/make_demo_cohort.py --out /tmp/null --effect 0.0 --subjects 20
python -m eegtumor.cli run --manifest /tmp/null/cohort.csv --out /tmp/null_run
#    ROC-AUC must be near 0.5. If it is not, stop and open an issue.

# 5. commit and tag
git add -A
git commit -m "v2.0.0: rewrite for subject-disjoint validation and reproducibility"
git push -u origin v2-research-grade
```

Open a pull request against `main` rather than force-pushing, so the diff is
reviewable and version 1 stays in the history.

---

## Release checklist

- [ ] `pytest -q` passes locally and in CI on 3.10, 3.11 and 3.12
- [ ] the `--effect 0.0` negative control returns ROC-AUC near 0.5
- [ ] `LICENSE` contains the PolyForm text with the Required Notice line
- [ ] `LICENSE-v1-MIT.txt` is present and mentioned in `CHANGELOG.md`
- [ ] the GitHub repository description is updated (it still says "detects brain
      tumors from EEG data", which the software does not do)
- [ ] repository topics updated: `eeg`, `qeeg`, `biomarkers`, `machine-learning`,
      `reproducibility`, `explainable-ai`
- [ ] screenshots regenerated: `xvfb-run -a python tools/capture_screenshots.py`
- [ ] no data files, result directories or `.edf` files staged
- [ ] tag `v2.0.0` created and release notes written from `CHANGELOG.md`

---

## Suggested release description

> **v2.0.0 — rewrite for methodological correctness**
>
> Version 1 reported ROC-AUC 0.957 on synthetic data containing no signal at
> all, because EEG channels from one subject were split across training and test
> folds. Version 2 makes subject-disjoint validation structural rather than
> optional, and adds a regression test suite that fails if the defect returns.
>
> Also: 81 documented biomarkers with physiological interpretations and
> references, staged feature selection with stability analysis, nested
> cross-validation with Optuna, calibration, bootstrap confidence intervals,
> DeLong and McNemar testing, exact SHAP explainers, publication-quality figures
> and HTML reports, a reproducible CLI, and a rewritten GUI.
>
> Licence changed to PolyForm Noncommercial 1.0.0. Version 1 remains MIT.
>
> Full detail in CHANGELOG.md. Read DISCLAIMER.md before use.
