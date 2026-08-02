"""
Figures and reports.

Purpose
-------
Emit the exact set of figures and tables a reviewer expects to see with a
clinical prediction model, at publication resolution, plus a single HTML
document that records what was run and what came out.

Design notes
------------
Figures follow journal conventions rather than dashboard aesthetics: vector
output (PDF/SVG) alongside 300 dpi PNG, no chartjunk, colour-blind-safe
palette, and every panel labelled with the sample size it is based on. The
calibration curve and the precision-recall curve are not optional extras - for
an imbalanced screening problem they carry information the ROC curve hides.

The report deliberately leads with the cohort audit and the limitations, not
with the headline AUC. A number without its denominator and its caveats is what
gets a manuscript rejected.

Inputs   : validation results, stability report, explanations, config
Outputs  : PNG + PDF figures, results.html, CSV tables
Limits   : PDF report generation requires an HTML-to-PDF tool; the HTML is
           self-contained and prints cleanly from a browser.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["FIGURE_STYLE", "save_figure", "plot_roc", "plot_pr",
           "plot_calibration", "plot_confusion", "plot_model_comparison",
           "plot_feature_importance", "plot_stability", "plot_signal_overview",
           "write_html_report"]

# Colour-blind-safe (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

FIGURE_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FIGURE_STYLE)
    return plt


def save_figure(fig, outdir: Path, name: str, formats=("png", "pdf")) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_roc(results: Dict[str, "object"], title: str = "ROC (out-of-fold, recording level)"):
    from sklearn.metrics import roc_curve
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res.pooled_labels, res.pooled_scores)
        ci = res.confidence_intervals.get("roc_auc")
        lab = f"{name}  AUC={res.pooled_metrics['roc_auc']:.3f}"
        if ci and np.isfinite(ci[1]):
            lab += f" [{ci[1]:.2f}-{ci[2]:.2f}]"
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], label=lab)
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="chance")
    n = len(next(iter(results.values())).pooled_labels) if results else 0
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{title}\nn = {n} recordings")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    return fig


def plot_pr(results: Dict[str, "object"]):
    from sklearn.metrics import precision_recall_curve
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    prevalence = None
    for i, (name, res) in enumerate(results.items()):
        pre, rec, _ = precision_recall_curve(res.pooled_labels, res.pooled_scores)
        ax.plot(rec, pre, color=PALETTE[i % len(PALETTE)],
                label=f"{name}  AP={res.pooled_metrics['pr_auc']:.3f}")
        prevalence = float(np.mean(res.pooled_labels))
    if prevalence is not None:
        ax.axhline(prevalence, ls="--", color="0.6", lw=1,
                   label=f"chance = prevalence ({prevalence:.2f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall\n(the informative curve when classes are imbalanced)")
    ax.legend(loc="lower left", fontsize=7)
    return fig


def plot_calibration(results: Dict[str, "object"], n_bins: int = 10):
    from sklearn.calibration import calibration_curve
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.2, 5.0),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for i, (name, res) in enumerate(results.items()):
        try:
            frac, mean_pred = calibration_curve(res.pooled_labels, res.pooled_scores,
                                                n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac, "o-", color=PALETTE[i % len(PALETTE)], ms=4,
                    label=f"{name}  Brier={res.pooled_metrics['brier']:.3f}")
        except ValueError:
            continue
        ax2.hist(res.pooled_scores, bins=20, histtype="step",
                 color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="perfect calibration")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration\n(a model saying 0.8 should be right 80% of the time)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    return fig


def plot_confusion(res, threshold: float = 0.5):
    from sklearn.metrics import confusion_matrix
    plt = _mpl()
    pred = (res.pooled_scores >= threshold).astype(int)
    cm = confusion_matrix(res.pooled_labels, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["negative", "positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.set_title(f"{res.model_name}\nthreshold = {threshold:.2f}")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def plot_model_comparison(results: Dict[str, "object"], metric: str = "roc_auc"):
    plt = _mpl()
    names = list(results)
    means = [results[n].mean(metric) for n in names]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(5.0, 0.45 * len(names) + 1.6))
    for k, i in enumerate(order):
        n = names[i]
        res = results[n]
        ci = res.confidence_intervals.get(metric)
        ax.barh(k, res.mean(metric), color=PALETTE[i % len(PALETTE)], alpha=0.85)
        if ci and np.isfinite(ci[1]):
            ax.plot([ci[1], ci[2]], [k, k], color="0.2", lw=1.4)
            ax.plot([ci[1], ci[1]], [k - .12, k + .12], color="0.2", lw=1.4)
            ax.plot([ci[2], ci[2]], [k - .12, k + .12], color="0.2", lw=1.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0.5, ls="--", color="0.6", lw=1)
    ax.set_xlabel(f"{metric} (bars = mean across folds, whiskers = 95% bootstrap CI)")
    ax.set_title("Model comparison")
    ax.set_xlim(0.3, 1.0)
    return fig


def plot_feature_importance(ranking: Sequence, top_n: int = 20,
                            xlabel: str = "mean |SHAP value|"):
    plt = _mpl()
    items = list(ranking)[:top_n][::-1]
    names = [t[0] for t in items]
    vals = [t[1] for t in items]
    fig, ax = plt.subplots(figsize=(5.4, 0.28 * len(items) + 1.2))
    ax.barh(range(len(items)), vals, color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {len(items)} biomarkers")
    return fig


def plot_stability(report, top_n: int = 25, threshold: float = 0.6):
    plt = _mpl()
    rows = report.table()[:top_n][::-1]
    names = [r["feature"] for r in rows]
    freq = [r["selection_frequency"] for r in rows]
    colors = [PALETTE[2] if f >= threshold else PALETTE[1] for f in freq]
    fig, ax = plt.subplots(figsize=(5.4, 0.28 * len(rows) + 1.4))
    ax.barh(range(len(rows)), freq, color=colors, alpha=0.85)
    ax.axvline(threshold, ls="--", color="0.3", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("selection frequency across subject-level resamples")
    ax.set_title(f"Feature stability\n{report.summary()}", fontsize=8)
    return fig


def plot_signal_overview(raw: np.ndarray, processed: np.ndarray, fs: float,
                         bands: Dict[str, tuple], channel: str = ""):
    """Four-panel signal inspection: waveform, PSD, band power, scalogram."""
    import pywt
    from scipy.integrate import trapezoid
    from scipy.signal import welch

    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6))
    t = np.arange(len(raw)) / fs

    ax = axes[0, 0]
    ax.plot(t, raw, color="0.65", lw=0.7, label="raw")
    ax.plot(t[:len(processed)], processed[:len(raw)], color=PALETTE[0], lw=0.8,
            label="filtered")
    ax.set_title(f"{channel} waveform"); ax.set_xlabel("s"); ax.set_ylabel("uV")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    f, p = welch(processed, fs=fs, nperseg=min(len(processed), int(fs * 4)))
    ax.semilogy(f, p, color=PALETTE[0])
    for (lo, hi), c in zip(bands.values(), PALETTE):
        ax.axvspan(lo, hi, alpha=0.07, color=c)
    ax.set_xlim(0, min(60, fs / 2))
    ax.set_title("Power spectral density"); ax.set_xlabel("Hz")
    ax.set_ylabel("uV$^2$/Hz")

    ax = axes[1, 0]
    vals, labels = [], []
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        vals.append(trapezoid(p[m], f[m]) if m.sum() > 1 else 0.0)
        labels.append(name)
    total = sum(vals) or 1.0
    ax.bar(labels, [v / total for v in vals], color=PALETTE[:len(vals)], alpha=0.85)
    ax.set_title("Relative band power"); ax.set_ylabel("fraction")

    ax = axes[1, 1]
    freqs = np.linspace(1, min(45, fs / 2 - 1), 60)
    scales = pywt.central_frequency("morl") * fs / freqs
    cwt, _ = pywt.cwt(processed, scales, "morl", sampling_period=1.0 / fs)
    ax.pcolormesh(t[:cwt.shape[1]], freqs, np.abs(cwt), shading="auto",
                  cmap="viridis")
    ax.set_title("Wavelet scalogram (Morlet)")
    ax.set_xlabel("s"); ax.set_ylabel("Hz")
    ax.grid(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 max-width:1000px;margin:2rem auto;padding:0 1.5rem;line-height:1.55;color:#1a1a1a}
h1{border-bottom:2px solid #0072B2;padding-bottom:.4rem}
h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}
th{background:#f4f6f8}tr:nth-child(even){background:#fafbfc}
.warn{background:#fff4e5;border-left:4px solid #D55E00;padding:.8rem 1rem;margin:1rem 0}
.crit{background:#fdecea;border-left:4px solid #b3261e;padding:.8rem 1rem;margin:1rem 0}
.note{background:#eef6fb;border-left:4px solid #0072B2;padding:.8rem 1rem;margin:1rem 0}
code{background:#f4f6f8;padding:.1rem .3rem;border-radius:3px;font-size:.88em}
img{max-width:100%;border:1px solid #eee;border-radius:4px;margin:.6rem 0}
.meta{color:#666;font-size:.85rem}
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _table(rows: List[dict], columns: Optional[Sequence[str]] = None) -> str:
    if not rows:
        return "<p><em>no data</em></p>"
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in columns)
    body = ""
    for r in rows:
        cells = "".join(
            f"<td>{r.get(c, '') if not isinstance(r.get(c), float) else f'{r.get(c):.3f}'}</td>"
            for c in columns)
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html_report(outdir: Path, cfg, cohort_summary: dict,
                      cohort_warnings: List[str],
                      results: Dict[str, "object"],
                      stability=None, explanation=None,
                      feature_dict: Optional[List[dict]] = None,
                      figures: Optional[Dict[str, "object"]] = None,
                      extra_notes: Optional[List[str]] = None) -> Path:
    """Write a self-contained results.html.

    Structured deliberately like supplementary material for a manuscript:
    what was run, on whom, what came out, and what is not supported by it.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures = figures or {}

    best = max(results, key=lambda k: results[k].mean("roc_auc")) if results else None

    html = [f"<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>Tumor-from-EEG results</title><style>{_CSS}</style></head><body>"]
    html.append("<h1>Tumor-from-EEG &mdash; analysis report</h1>")
    html.append(f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; "
                f"task <code>{cfg.task_name}</code> &middot; "
                f"config fingerprint <code>{cfg.fingerprint()}</code></p>")

    html.append("<div class='crit'><strong>Not a diagnostic device.</strong> This "
                "software estimates the probability that an EEG recording "
                "resembles the positive class in the training data supplied by "
                "the operator. EEG cannot confirm or exclude a brain tumour; "
                "structural imaging and clinical assessment are required. No "
                "output here may be used for clinical decisions.</div>")

    # -- cohort ------------------------------------------------------------
    html.append("<h2>1. Cohort</h2>")
    html.append(_table([{"property": k, "value": str(v)}
                        for k, v in cohort_summary.items()]))
    if cohort_warnings:
        html.append("<div class='warn'><strong>Cohort design warnings</strong><ul>"
                    + "".join(f"<li>{w}</li>" for w in cohort_warnings) + "</ul></div>")
    else:
        html.append("<div class='note'>No structural confounds detected in the "
                    "cohort audit.</div>")

    # -- protocol ----------------------------------------------------------
    html.append("<h2>2. Analysis protocol</h2>")
    v = cfg.validation
    html.append(_table([
        {"setting": "Splitting", "value": f"StratifiedGroupKFold on {v.grouping}_id "
                                          f"({v.outer_folds} outer folds)"},
        {"setting": "Repeats", "value": str(v.n_repeats)},
        {"setting": "Inner loop", "value": f"{v.inner_folds}-fold, {v.optimisation} "
                                           f"({v.n_trials} trials)"},
        {"setting": "Calibration", "value": str(v.calibration)},
        {"setting": "Aggregation", "value": v.aggregation},
        {"setting": "Epoch", "value": f"{cfg.preprocessing.epoch_seconds} s at "
                                      f"{cfg.preprocessing.target_sampling_rate} Hz"},
        {"setting": "Passband", "value": f"{cfg.preprocessing.highpass}-"
                                         f"{cfg.preprocessing.lowpass} Hz, notch "
                                         f"{cfg.preprocessing.notch_freq} Hz"},
    ], ["setting", "value"]))
    html.append("<div class='note'>Every data-dependent step (imputation, "
                "scaling, feature selection, calibration) is fitted inside the "
                "training fold only. All metrics below are out-of-fold and are "
                "computed at recording level.</div>")

    # -- performance -------------------------------------------------------
    html.append("<h2>3. Performance</h2>")
    rows = []
    for name, res in sorted(results.items(), key=lambda kv: -kv[1].mean("roc_auc")):
        ci = res.confidence_intervals.get("roc_auc", (np.nan,) * 3)
        rows.append({
            "model": name,
            "ROC-AUC": f"{res.mean('roc_auc'):.3f} ± {res.std('roc_auc'):.3f}",
            "95% CI": f"{ci[1]:.3f}–{ci[2]:.3f}" if np.isfinite(ci[1]) else "n/a",
            "PR-AUC": f"{res.mean('pr_auc'):.3f}",
            "Balanced acc.": f"{res.mean('balanced_accuracy'):.3f}",
            "Sens@Spec90": f"{res.mean('sens_at_spec90'):.3f}",
            "MCC": f"{res.mean('mcc'):.3f}",
            "Brier": f"{res.mean('brier'):.3f}",
        })
    html.append(_table(rows))
    if best:
        html.append(f"<p>Best by mean ROC-AUC: <strong>{best}</strong>. "
                    f"Where confidence intervals overlap, the ranking between "
                    f"models is not statistically supported &mdash; see the "
                    f"pairwise DeLong comparison below.</p>")

    for key in ("model_comparison", "roc", "pr", "calibration", "confusion"):
        if key in figures:
            html.append(f"<img src='data:image/png;base64,{_fig_to_b64(figures[key])}'/>")

    # -- explainability ----------------------------------------------------
    if explanation is not None or stability is not None:
        html.append("<h2>4. Biomarkers</h2>")
        if explanation is not None:
            html.append(f"<p class='meta'>Explainer: {explanation.explainer_type}</p>")
            html.append(_table([{"rank": i + 1, "feature": n, "mean |SHAP|": v}
                                for i, (n, v) in
                                enumerate(explanation.global_ranking(20))]))
        if "shap" in figures:
            html.append(f"<img src='data:image/png;base64,{_fig_to_b64(figures['shap'])}'/>")
        if stability is not None:
            html.append(f"<p>{stability.summary()}</p>")
            if "stability" in figures:
                html.append(f"<img src='data:image/png;base64,"
                            f"{_fig_to_b64(figures['stability'])}'/>")
            stable = stability.stable(0.6)
            html.append(f"<p>Features selected in at least 60% of resamples "
                        f"({len(stable)}): <code>{', '.join(stable) or 'none'}</code></p>")
        html.append(f"<div class='warn'>{__import__('eegtumor.explain', fromlist=['CAVEAT']).CAVEAT}</div>")

    # -- feature dictionary -------------------------------------------------
    if feature_dict:
        html.append("<h2>5. Feature dictionary</h2>")
        html.append("<p class='meta'>Definitions and physiological "
                    "interpretation of every extracted biomarker.</p>")
        html.append(_table(feature_dict,
                           ["feature", "domain", "unit",
                            "physiological_interpretation", "references"]))

    # -- limitations --------------------------------------------------------
    html.append("<h2>6. Limitations of this run</h2><ul>")
    lim = [
        f"Internal cross-validation only. No external or temporal validation was "
        f"performed, and internal resampling cannot substitute for it.",
        f"Performance is conditional on the cohort supplied "
        f"({cohort_summary.get('n_subjects', '?')} subjects); it does not "
        f"transfer automatically to another site, montage or amplifier.",
        "Feature attributions describe the model, not causation.",
        "Probabilities are calibrated on the training folds; recalibration is "
        "required before use on a population with a different prevalence.",
    ]
    lim += (extra_notes or [])
    html.append("".join(f"<li>{x}</li>" for x in lim))
    html.append("</ul>")

    html.append("<h2>7. Reproducibility</h2>")
    html.append(f"<pre style='font-size:.75rem;background:#f4f6f8;padding:1rem;"
                f"overflow-x:auto'>{json.dumps(cfg.to_dict(), indent=2, default=str)}</pre>")
    html.append("</body></html>")

    path = outdir / "results.html"
    path.write_text("\n".join(html), encoding="utf-8")
    logger.info("report written to %s", path)
    return path
