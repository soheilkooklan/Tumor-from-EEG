# Version 1 (archived)

`Main File Tumor from EEG.py` is the original single-file implementation,
preserved for provenance. It is **not maintained and should not be used.**

It contained a grouping-leakage defect: EEG channels from one subject were
treated as independent samples and split across cross-validation folds, so
reported performance measured subject recognition rather than pathology. On
synthetic data with no signal at all, that protocol returned ROC-AUC 0.957.

Version 1 is licensed under the MIT licence (`../LICENSE-v1-MIT.txt`). That
grant is irrevocable for this code. Version 2 is licensed separately under
PolyForm Noncommercial 1.0.0.

See `../CHANGELOG.md` for the full list of what changed and why.
