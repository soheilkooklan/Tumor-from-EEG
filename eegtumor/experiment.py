"""
Experiment orchestration.

Purpose
-------
One function that runs a complete analysis from a cohort manifest to a report
directory, so the GUI and the command line execute *identical* code. When the
two diverge, the figure in the paper and the figure on the screen stop being
the same figure.

Every run writes its own directory named after the configuration fingerprint,
containing the config actually used, the feature table, the metrics, the
figures and the HTML report. That directory is the reproducibility unit: it is
what gets attached to a manuscript submission.

Inputs   : manifest path, AnalysisConfig, output directory
Outputs  : RunArtifacts (paths + in-memory results)
Limits   : single-machine, in-memory. A cohort of several thousand long
           recordings needs an out-of-core feature store, which is future work.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import AnalysisConfig
from .explain import explain_model
from .features import extract_recording_features, feature_dictionary, feature_names
from .io import Cohort, ensure_loaded, load_cohort
from .modeling import build_pipeline, calibrate
from .preprocessing import preprocess_recording
from .reporting import (plot_calibration, plot_confusion, plot_feature_importance,
                        plot_model_comparison, plot_pr, plot_roc, plot_stability,
                        save_figure, write_html_report)
from .selection import selection_stability
from .validation import (delong_test, nested_cross_validate, permutation_null,
                         single_feature_baseline,
                         split_groups)

logger = logging.getLogger(__name__)

__all__ = ["RunArtifacts", "extract_cohort_features", "run_analysis"]

Progress = Optional[Callable[[float, str], None]]


@dataclass
class RunArtifacts:
    outdir: Path
    config: AnalysisConfig
    cohort_summary: dict
    cohort_warnings: List[str]
    X: np.ndarray
    index: List[dict]
    feature_names: List[str]
    results: Dict[str, object] = field(default_factory=dict)
    stability: object = None
    explanation: object = None
    comparisons: List[dict] = field(default_factory=list)
    report_path: Optional[Path] = None
    permutation: Optional[dict] = None
    baseline: Optional[dict] = None


def _emit(progress: Progress, frac: float, msg: str) -> None:
    logger.info("[%3.0f%%] %s", 100 * frac, msg)
    if progress:
        progress(frac, msg)


def _features_one(rec, cfg: AnalysisConfig):
    """Load (if needed), preprocess and extract one recording.

    Returns (X, index, message). The signal is released when this returns, so
    memory use stays at one recording per worker.
    """
    try:
        rec = ensure_loaded(rec)
        epoched = preprocess_recording(rec, cfg.preprocessing)
        if epoched.n_good < cfg.preprocessing.min_good_epochs:
            return None, None, (f"excluding {rec.recording_id}: only "
                                f"{epoched.n_good} usable epoch-channels")
        Xr, _, idx = extract_recording_features(epoched, cfg)
        return (Xr, idx, None) if Xr.size else (None, None, None)
    except Exception as exc:
        return None, None, f"{rec.recording_id} failed: {exc}"


def extract_cohort_features(cohort: Cohort, cfg: AnalysisConfig,
                            progress: Progress = None, n_jobs: int = 1):
    """Preprocess and extract features for every recording in a cohort.

    `n_jobs` > 1 extracts recordings in parallel processes. Output order and
    values do not depend on it. Workers are spawned, so a script that calls
    this with n_jobs > 1 needs the usual `if __name__ == "__main__":` guard.
    """
    names = feature_names(cfg)
    recs = list(cohort)
    n = len(recs)
    results = [None] * n

    def note(i, res):
        X_, idx_, msg = res
        if msg:
            (logger.warning if msg.startswith("excluding") else logger.error)(msg)
        results[i] = (X_, idx_)

    if n_jobs and n_jobs > 1 and n > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # spawn, not fork: forking a process that already runs BLAS threads can
        # deadlock, and spawn behaves the same on Linux, macOS and Windows
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_features_one, r, cfg): i for i, r in enumerate(recs)}
            for done, fut in enumerate(as_completed(futures), 1):
                i = futures[fut]
                _emit(progress, 0.05 + 0.45 * done / n,
                      f"features: {recs[i].recording_id} ({done}/{n})")
                note(i, fut.result())
    else:
        for i, rec in enumerate(recs):
            _emit(progress, 0.05 + 0.45 * i / max(n, 1),
                  f"features: {rec.recording_id} ({i + 1}/{n})")
            note(i, _features_one(rec, cfg))

    rows = [r[0] for r in results if r and r[0] is not None]
    index = [row for r in results if r and r[0] is not None for row in r[1]]
    X = np.vstack(rows) if rows else np.empty((0, len(names)))
    return X, names, index


def run_analysis(manifest: str, cfg: AnalysisConfig, outdir: str,
                 models: Optional[Sequence[str]] = None,
                 progress: Progress = None,
                 data_root: Optional[str] = None,
                 n_jobs: int = 1) -> RunArtifacts:
    """End-to-end analysis: manifest -> report directory."""
    out = Path(outdir) / f"run_{datetime.now():%Y%m%d_%H%M%S}_{cfg.fingerprint()}"
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"

    _emit(progress, 0.01, "loading cohort")
    cohort = load_cohort(manifest, data_root, lazy=True)
    summary = cohort.summary()
    warnings_ = cohort.audit()

    cfg.to_yaml(str(out / "config_used.yaml"))
    (out / "environment.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "config_fingerprint": cfg.fingerprint(),
    }, indent=2))

    X, names, index = extract_cohort_features(cohort, cfg, progress, n_jobs=n_jobs)
    if X.shape[0] == 0:
        raise RuntimeError("no usable epochs were extracted from any recording")

    _emit(progress, 0.5, f"feature matrix {X.shape}")
    _save_feature_table(out, X, names, index)
    _write_csv(out / "feature_dictionary.csv", feature_dictionary(cfg))

    y = np.array([int(r["label"]) for r in index])
    groups = split_groups(index, cfg.validation.grouping)

    _emit(progress, 0.55, "nested cross-validation")
    results = nested_cross_validate(
        X, index, cfg, names, models,
        progress=lambda f, m: _emit(progress, 0.55 + 0.30 * f, m))
    if not results:
        raise RuntimeError("cross-validation produced no usable folds")

    _emit(progress, 0.86, "feature stability")
    stability = selection_stability(X, y, groups, names, cfg.selection,
                                    cfg.validation.random_state)

    null = None
    if cfg.validation.permutation_tests > 0:
        model = cfg.validation.permutation_model
        _emit(progress, 0.87, f"permutation null ({cfg.validation.permutation_tests} "
                              f"permutations, {model})")
        null = permutation_null(X, index, cfg, names, model,
                                cfg.validation.permutation_tests,
                                cfg.validation.random_state)
        (out / "permutation_null.json").write_text(json.dumps(null, indent=2))

    baseline = None
    if cfg.validation.baseline_feature:
        baseline = single_feature_baseline(
            X, index, names, cfg.validation.baseline_feature,
            cfg.validation.bootstrap_iterations, cfg.validation.alpha,
            cfg.validation.random_state)
        if baseline is not None:
            (out / "baseline.json").write_text(json.dumps(baseline, indent=2))

    # The family shown in the confusion matrix and explained below is picked on
    # outer-fold ROC-AUC, so its score carries that selection. Fine for
    # exploration; for a confirmatory claim name one model in advance.
    best = max(results, key=lambda k: results[k].mean("roc_auc"))
    _emit(progress, 0.90, f"explaining {best}")
    final = calibrate(build_pipeline(best, cfg, names), X, y,
                      cfg.validation.calibration, groups=groups,
                      random_state=cfg.validation.random_state)
    explanation = explain_model(final, X, names,
                                random_state=cfg.validation.random_state)

    _emit(progress, 0.94, "pairwise model comparison")
    comparisons = _compare_models(results)

    _emit(progress, 0.96, "figures and report")
    figs = {
        "model_comparison": plot_model_comparison(results),
        "roc": plot_roc(results),
        "pr": plot_pr(results),
        "calibration": plot_calibration(results),
        "confusion": plot_confusion(results[best]),
        "stability": plot_stability(stability),
    }
    if explanation is not None:
        figs["shap"] = plot_feature_importance(explanation.global_ranking())
    for name, fig in figs.items():
        save_figure(fig, figdir, name)

    _write_csv(out / "metrics.csv", [
        {"model": n, **{k: round(v, 4) for k, v in r.pooled_metrics.items()},
         "roc_auc_mean": round(r.mean("roc_auc"), 4),
         "roc_auc_sd": round(r.std("roc_auc"), 4),
         **{f"{m}_ci_{side}": round(ci[i], 4)
            for m, ci in r.confidence_intervals.items()
            for i, side in ((1, "low"), (2, "high"))}}
        for n, r in results.items()])
    _write_csv(out / "feature_stability.csv", stability.table())
    if comparisons:
        _write_csv(out / "model_comparisons.csv", comparisons)

    notes = [f"The model shown in the confusion matrix and SHAP figures ({best}) "
             f"was chosen by outer-fold ROC-AUC among {len(results)} families, so "
             f"its score is optimistic by that choice. Pre-specify one model for a "
             f"confirmatory analysis."]
    if comparisons:
        notes.append("Pairwise DeLong comparisons: " +
                     "; ".join(f"{c['model_a']} vs {c['model_b']} "
                               f"p={c['p_value']:.3f}" for c in comparisons))
    if baseline is not None:
        notes.append(f"No-learning baseline: the per-recording median of "
                     f"{baseline['feature']} used directly as the score gives ROC-AUC "
                     f"{baseline['roc_auc']:.3f} [{baseline['ci_low']:.3f}, "
                     f"{baseline['ci_high']:.3f}] over {baseline['n_recordings']} "
                     f"recordings. Below 0.5 means lower in the positive class. "
                     f"Compare the models against this, not against 0.5.")
    if null is not None and null.get("n_permutations", 0):
        notes.append(f"Subject-level permutation null ({null['model']}, "
                     f"{null['n_permutations']} permutations, no tuning, one repeat): "
                     f"observed ROC-AUC {null['observed']:.3f}, null mean "
                     f"{null['null_mean']:.3f}, p = {null['p_value']:.3f}.")

    report = write_html_report(
        out, cfg, summary, warnings_, results, stability, explanation,
        feature_dictionary(cfg), figs, extra_notes=notes)

    _emit(progress, 1.0, f"done -> {out}")
    import matplotlib.pyplot as plt
    plt.close("all")

    art = RunArtifacts(out, cfg, summary, warnings_, X, index, names,
                       results, stability, explanation, comparisons, report)
    art.permutation = null
    art.baseline = baseline
    return art


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _compare_models(results) -> List[dict]:
    """DeLong tests between every pair, on the recordings both scored."""
    out = []
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            common = np.intersect1d(a.pooled_recordings, b.pooled_recordings)
            if len(common) < 10:
                continue
            ia = np.array([np.where(a.pooled_recordings == r)[0][0] for r in common])
            ib = np.array([np.where(b.pooled_recordings == r)[0][0] for r in common])
            test = delong_test(a.pooled_labels[ia], a.pooled_scores[ia],
                               b.pooled_scores[ib])
            out.append({"model_a": names[i], "model_b": names[j],
                        "n_recordings": len(common),
                        **{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in test.items()}})
    return out


def _save_feature_table(out: Path, X, names, index) -> None:
    import pandas as pd
    df = pd.DataFrame(X, columns=names)
    meta = pd.DataFrame(index)
    pd.concat([meta, df], axis=1).to_csv(out / "features.csv", index=False)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
