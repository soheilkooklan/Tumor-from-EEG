"""
Graphical interface.

Purpose
-------
Let a researcher run the full protocol without writing Python, while making the
methodological decisions visible rather than hiding them behind a single "run"
button.

Design principles
-----------------
The layout follows the analysis workflow left to right, and each tab states the
scientific constraint it enforces. The interface deliberately does *not* offer
a way to disable subject-disjoint validation, to scale features outside the
pipeline, or to hide the cohort audit - those are the settings that turn a
result into an artefact, so they are not user-configurable.

Long-running work happens on a worker thread; the UI thread only polls a queue,
so the window stays responsive and progress is honest rather than a spinner.

Requires: pip install customtkinter
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

try:
    import customtkinter as ctk
except ImportError:                                            # pragma: no cover
    print("The GUI needs customtkinter:\n    pip install customtkinter")
    sys.exit(1)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import __version__
from .config import AnalysisConfig
from .experiment import extract_cohort_features, run_analysis
from .features import feature_dictionary, feature_names
from .io import load_cohort, read_recording, write_manifest_template
from .preprocessing import preprocess_recording
from .reporting import (FIGURE_STYLE, plot_calibration, plot_feature_importance,
                        plot_model_comparison, plot_roc, plot_signal_overview,
                        plot_stability)

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
plt.rcParams.update(FIGURE_STYLE)

DISCLAIMER = (
    "RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.  This tool estimates how closely "
    "an EEG recording resembles the positive class of the training data you "
    "supply. EEG cannot confirm or exclude a brain tumour; structural imaging "
    "and clinical assessment are required. Do not use any output for clinical "
    "decisions."
)


class TumorFromEEGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"Tumor-from-EEG  ·  research platform v{__version__}")
        self.geometry("1320x880")
        self.minsize(1100, 760)

        self.cfg = AnalysisConfig()
        self.cohort = None
        self.artifacts = None
        self.X = None
        self.index = None
        self.names = None
        self.manifest_path = None
        self.outdir = str(Path.cwd() / "results")

        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._build()
        self.after(120, self._drain_queue)

    # ------------------------------------------------------------------ UI
    def _build(self):
        banner = ctk.CTkLabel(
            self, text=DISCLAIMER, text_color="#FFB45C", justify="left",
            font=ctk.CTkFont(size=12, weight="bold"), wraplength=1260)
        banner.pack(padx=14, pady=(12, 6), fill="x")

        self.tabs = ctk.CTkTabview(self, anchor="w")
        self.tabs.pack(padx=14, pady=(0, 6), fill="both", expand=True)
        for name in ("1 · Cohort", "2 · Signal", "3 · Features",
                     "4 · Validation", "5 · Biomarkers", "6 · Screen",
                     "About"):
            self.tabs.add(name)

        self._build_cohort_tab(self.tabs.tab("1 · Cohort"))
        self._build_signal_tab(self.tabs.tab("2 · Signal"))
        self._build_feature_tab(self.tabs.tab("3 · Features"))
        self._build_validation_tab(self.tabs.tab("4 · Validation"))
        self._build_biomarker_tab(self.tabs.tab("5 · Biomarkers"))
        self._build_screen_tab(self.tabs.tab("6 · Screen"))
        self._build_about_tab(self.tabs.tab("About"))

        bar = ctk.CTkFrame(self, height=32)
        bar.pack(padx=14, pady=(0, 12), fill="x")
        self.progress = ctk.CTkProgressBar(bar, height=10)
        self.progress.set(0)
        self.progress.pack(side="left", padx=10, pady=8, fill="x", expand=True)
        self.status = ctk.CTkLabel(bar, text="ready", anchor="e", width=420)
        self.status.pack(side="right", padx=10)

    # -- tab 1 ---------------------------------------------------------
    def _build_cohort_tab(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(padx=10, pady=10, fill="x")

        ctk.CTkButton(top, text="Open cohort manifest (CSV)",
                      command=self._load_manifest, width=210).grid(row=0, column=0, padx=6, pady=8)
        ctk.CTkButton(top, text="Create manifest template",
                      command=self._make_template, width=190,
                      fg_color="transparent", border_width=1).grid(row=0, column=1, padx=6)
        ctk.CTkButton(top, text="Choose output folder",
                      command=self._choose_outdir, width=180,
                      fg_color="transparent", border_width=1).grid(row=0, column=2, padx=6)

        info = ctk.CTkLabel(
            tab, justify="left", wraplength=1240, text_color="#9BB8CC",
            text=("A manifest is a CSV with columns  path, subject_id, label  "
                  "(optionally recording_id, sampling_rate, site, age, sex).\n"
                  "subject_id is mandatory: it is what keeps every recording of "
                  "one person inside a single cross-validation fold. Without it "
                  "the reported performance would measure subject recognition, "
                  "not pathology."))
        info.pack(padx=14, pady=(0, 6), anchor="w")

        self.cohort_log = ctk.CTkTextbox(tab, font=ctk.CTkFont(family="monospace", size=12))
        self.cohort_log.pack(padx=10, pady=10, fill="both", expand=True)
        self._log(self.cohort_log, "No cohort loaded.\n")

    # -- tab 2 ---------------------------------------------------------
    def _build_signal_tab(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(padx=10, pady=10, fill="x")
        ctk.CTkButton(top, text="Inspect a recording",
                      command=self._inspect_signal, width=190).grid(row=0, column=0, padx=6, pady=8)
        ctk.CTkLabel(top, text="Channel").grid(row=0, column=1, padx=(18, 4))
        self.channel_menu = ctk.CTkOptionMenu(top, values=["—"], width=140,
                                              command=lambda _: self._redraw_signal())
        self.channel_menu.grid(row=0, column=2, padx=4)
        ctk.CTkLabel(top, text="Epoch").grid(row=0, column=3, padx=(18, 4))
        self.epoch_menu = ctk.CTkOptionMenu(top, values=["—"], width=100,
                                            command=lambda _: self._redraw_signal())
        self.epoch_menu.grid(row=0, column=4, padx=4)

        self.signal_fig, self.signal_canvas = self._canvas(tab, (9.6, 5.6))
        self._current_epoched = None
        self._current_raw = None

    # -- tab 3 ---------------------------------------------------------
    def _build_feature_tab(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(padx=10, pady=10, fill="x")
        ctk.CTkButton(top, text="Extract features for the whole cohort",
                      command=self._extract_features, width=280).grid(row=0, column=0, padx=6, pady=8)
        ctk.CTkButton(top, text="Show feature dictionary",
                      command=self._show_dictionary, width=200,
                      fg_color="transparent", border_width=1).grid(row=0, column=1, padx=6)
        self.feature_log = ctk.CTkTextbox(tab, font=ctk.CTkFont(family="monospace", size=12))
        self.feature_log.pack(padx=10, pady=10, fill="both", expand=True)
        self._log(self.feature_log,
                  f"{len(feature_names(self.cfg))} biomarkers configured across "
                  f"{len(self.cfg.features.domains)} scientific domains.\n"
                  "Press Extract to build the feature table.\n")

    # -- tab 4 ---------------------------------------------------------
    def _build_validation_tab(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(padx=10, pady=10, fill="x")
        ctk.CTkButton(top, text="Run nested cross-validation",
                      command=self._run_validation, width=250).grid(row=0, column=0, padx=6, pady=8)
        ctk.CTkLabel(top, text=f"protocol: StratifiedGroupKFold on subject_id · "
                               f"{self.cfg.validation.outer_folds} outer × "
                               f"{self.cfg.validation.inner_folds} inner × "
                               f"{self.cfg.validation.n_repeats} repeats",
                     text_color="#9BB8CC").grid(row=0, column=1, padx=18)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(padx=10, pady=6, fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.validation_log = ctk.CTkTextbox(body, font=ctk.CTkFont(family="monospace", size=12))
        self.validation_log.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.val_fig, self.val_canvas = self._canvas(body, (5.6, 6.0), grid=(0, 1))
        self._log(self.validation_log,
                  "Not run yet.\n\nEvery reported number is out-of-fold and at "
                  "recording level.\nNo subject ever appears in both the "
                  "training and the test side of a fold.\n")

    # -- tab 5 ---------------------------------------------------------
    def _build_biomarker_tab(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(padx=10, pady=10, fill="x")
        self.biomarker_view = ctk.CTkSegmentedButton(
            top, values=["SHAP importance", "Selection stability"],
            command=lambda _: self._redraw_biomarkers())
        self.biomarker_view.set("SHAP importance")
        self.biomarker_view.grid(row=0, column=0, padx=6, pady=8)
        ctk.CTkLabel(top, text="Attributions describe the model, not causation.",
                     text_color="#FFB45C").grid(row=0, column=1, padx=18)
        self.bio_fig, self.bio_canvas = self._canvas(tab, (9.0, 6.0))

    # -- tab 6 ---------------------------------------------------------
    def _build_screen_tab(self, tab):
        top = ctk.CTkFrame(tab)
        top.pack(padx=10, pady=10, fill="x")
        ctk.CTkButton(top, text="Load unlabelled recording(s)",
                      command=self._load_screening, width=230).grid(row=0, column=0, padx=6, pady=8)
        ctk.CTkButton(top, text="Estimate probability",
                      command=self._screen, width=190).grid(row=0, column=1, padx=6)
        self.screen_log = ctk.CTkTextbox(tab, font=ctk.CTkFont(family="monospace", size=12))
        self.screen_log.pack(padx=10, pady=10, fill="both", expand=True)
        self._log(self.screen_log,
                  "Train a model first (tab 4), then load recordings here.\n\n"
                  "The number produced is a similarity score to your training "
                  "positives, with a bootstrap interval. It is not a diagnosis "
                  "and its meaning depends entirely on the cohort it was "
                  "trained on.\n")
        self._screening = []

    # -- about ---------------------------------------------------------
    def _build_about_tab(self, tab):
        box = ctk.CTkTextbox(tab, font=ctk.CTkFont(size=13))
        box.pack(padx=12, pady=12, fill="both", expand=True)
        box.insert("end", ABOUT_TEXT)
        box.configure(state="disabled")

    # ------------------------------------------------------------- helpers
    def _canvas(self, parent, size, grid=None):
        fig = plt.figure(figsize=size)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        w = canvas.get_tk_widget()
        if grid:
            w.grid(row=grid[0], column=grid[1], sticky="nsew")
        else:
            w.pack(padx=10, pady=10, fill="both", expand=True)
        return fig, canvas

    @staticmethod
    def _log(box, text, clear=False):
        box.configure(state="normal")
        if clear:
            box.delete("1.0", "end")
        box.insert("end", text)
        box.see("end")

    def _set_status(self, frac, msg):
        self._queue.put(("progress", frac, msg))

    def _drain_queue(self):
        try:
            while True:
                kind, *payload = self._queue.get_nowait()
                if kind == "progress":
                    frac, msg = payload
                    self.progress.set(frac)
                    self.status.configure(text=msg[:70])
                elif kind == "log":
                    box, text = payload
                    self._log(box, text)
                elif kind == "call":
                    payload[0]()
                elif kind == "error":
                    messagebox.showerror("Error", payload[0])
        except queue.Empty:
            pass
        self.after(120, self._drain_queue)

    def _run_async(self, fn, on_done=None):
        def worker():
            try:
                result = fn()
                if on_done:
                    self._queue.put(("call", lambda: on_done(result)))
            except Exception as exc:
                logger.exception("worker failed")
                self._queue.put(("error", f"{exc}\n\n{traceback.format_exc(limit=2)}"))
                self._queue.put(("progress", 0.0, "failed"))
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------ actions
    def _make_template(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile="cohort.csv")
        if path:
            write_manifest_template(path)
            self._log(self.cohort_log, f"template written to {path}\n")

    def _choose_outdir(self):
        d = filedialog.askdirectory(title="Where should results be written?")
        if d:
            self.outdir = d
            self._log(self.cohort_log, f"output folder: {d}\n")

    def _load_manifest(self):
        path = filedialog.askopenfilename(filetypes=[("CSV manifest", "*.csv")])
        if not path:
            return
        try:
            self.load_manifest(path)
        except Exception as exc:
            messagebox.showerror("Cohort error", str(exc))

    def load_manifest(self, path: str) -> None:
        """Load a cohort manifest and populate the cohort tab.

        Kept separate from the file dialog so the interface can be driven from
        a script - which is how `tools/capture_screenshots.py` regenerates the
        documentation images, and how the GUI can be smoke-tested in CI.
        """
        self.manifest_path = path
        self.cohort = load_cohort(path)
        s = self.cohort.summary()
        self._log(self.cohort_log, "=" * 72 + f"\nLoaded {path}\n" + "=" * 72 + "\n", clear=True)
        for k, v in s.items():
            self._log(self.cohort_log, f"  {k:28s} {v}\n")
        warn = self.cohort.audit()
        if warn:
            self._log(self.cohort_log, "\nCOHORT DESIGN WARNINGS\n")
            for w in warn:
                self._log(self.cohort_log, f"  ! {w}\n")
        else:
            self._log(self.cohort_log, "\nNo structural confounds detected.\n")
        self._log(self.cohort_log, "\nRecordings:\n")
        for r in self.cohort:
            self._log(self.cohort_log, f"  {r.describe()}\n")
        self._set_status(0.0, f"{len(self.cohort)} recordings loaded")

    def _inspect_signal(self):
        if not self.cohort:
            messagebox.showwarning("No cohort", "Load a cohort manifest first (tab 1).")
            return
        rec = self.cohort.recordings[0]
        self._current_raw = rec
        self._current_epoched = preprocess_recording(rec, self.cfg.preprocessing)
        self.channel_menu.configure(values=self._current_epoched.channel_names)
        self.channel_menu.set(self._current_epoched.channel_names[0])
        n = max(self._current_epoched.n_epochs, 1)
        self.epoch_menu.configure(values=[str(i) for i in range(n)])
        self.epoch_menu.set("0")
        self._redraw_signal()

    def _redraw_signal(self):
        if self._current_epoched is None:
            return
        ep = self._current_epoched
        try:
            c = ep.channel_names.index(self.channel_menu.get())
            i = int(self.epoch_menu.get())
        except (ValueError, IndexError):
            return
        fs = ep.sampling_rate
        n = ep.epochs.shape[-1]
        start = int(i * n * (1 - self.cfg.preprocessing.epoch_overlap))
        raw = self._current_raw.data[c]
        step = self._current_raw.sampling_rate / fs
        raw_seg = raw[int(start * step):int((start + n) * step):max(int(step), 1)][:n]
        if len(raw_seg) < n:
            raw_seg = np.pad(raw_seg, (0, n - len(raw_seg)))

        self.signal_fig.clf()
        new = plot_signal_overview(raw_seg, ep.epochs[i, c], fs,
                                   self.cfg.bands.bands,
                                   f"{ep.recording_id} · {ep.channel_names[c]} · epoch {i}")
        self._transfer(new, self.signal_fig, self.signal_canvas)

    def _extract_features(self):
        if not self.cohort:
            messagebox.showwarning("No cohort", "Load a cohort manifest first (tab 1).")
            return
        self._log(self.feature_log, "\nExtracting…\n")

        def job():
            return extract_cohort_features(self.cohort, self.cfg, self._set_status)

        def done(result):
            self.X, self.names, self.index = result
            nan = float(np.isnan(self.X).mean()) if self.X.size else 0.0
            self._log(self.feature_log,
                      f"\nfeature matrix : {self.X.shape[0]} rows × "
                      f"{self.X.shape[1]} biomarkers\n"
                      f"subjects       : {len(set(r['subject_id'] for r in self.index))}\n"
                      f"recordings     : {len(set(r['recording_id'] for r in self.index))}\n"
                      f"missing values : {100 * nan:.2f}%\n")
            self._set_status(1.0, "features ready")

        self._run_async(job, done)

    def _show_dictionary(self):
        self._log(self.feature_log, "\n" + "=" * 72 + "\nFEATURE DICTIONARY\n" + "=" * 72 + "\n")
        for row in feature_dictionary(self.cfg):
            self._log(self.feature_log,
                      f"\n{row['feature']}  [{row['domain']}, {row.get('unit', '')}]\n"
                      f"  {row['description']}\n"
                      f"  -> {row['physiological_interpretation']}\n")

    def _run_validation(self):
        if not self.manifest_path:
            messagebox.showwarning("No cohort", "Load a cohort manifest first (tab 1).")
            return
        self._log(self.validation_log, "\nRunning nested cross-validation…\n", clear=True)

        def job():
            return run_analysis(self.manifest_path, self.cfg, self.outdir,
                                progress=self._set_status)

        def done(art):
            self.artifacts = art
            self.X, self.names, self.index = art.X, art.feature_names, art.index
            self._log(self.validation_log, "=" * 60 + "\nRESULTS (out-of-fold, "
                                                      "recording level)\n" + "=" * 60 + "\n", clear=True)
            for name, res in sorted(art.results.items(),
                                    key=lambda kv: -kv[1].mean("roc_auc")):
                ci = res.confidence_intervals.get("roc_auc", (np.nan,) * 3)
                self._log(self.validation_log,
                          f"\n{name}\n"
                          f"  ROC-AUC        {res.mean('roc_auc'):.3f} ± {res.std('roc_auc'):.3f}"
                          f"   95% CI [{ci[1]:.3f}, {ci[2]:.3f}]\n"
                          f"  PR-AUC         {res.mean('pr_auc'):.3f}\n"
                          f"  Balanced acc.  {res.mean('balanced_accuracy'):.3f}\n"
                          f"  Sens @ Spec90  {res.mean('sens_at_spec90'):.3f}\n"
                          f"  MCC            {res.mean('mcc'):.3f}\n"
                          f"  Brier          {res.mean('brier'):.3f}\n")
            if art.comparisons:
                self._log(self.validation_log, "\n" + "-" * 60 +
                          "\nPAIRWISE DeLONG COMPARISON\n" + "-" * 60 + "\n")
                for c in art.comparisons:
                    verdict = "different" if c["p_value"] < 0.05 else "not distinguishable"
                    self._log(self.validation_log,
                              f"  {c['model_a']} vs {c['model_b']}: "
                              f"ΔAUC={c['difference']:+.3f}, p={c['p_value']:.3f}"
                              f"  → {verdict}\n")
            self._log(self.validation_log, f"\nreport: {art.report_path}\n")
            self.val_fig.clf()
            self._transfer(plot_model_comparison(art.results), self.val_fig, self.val_canvas)
            self._redraw_biomarkers()
            self._set_status(1.0, "validation complete")

        self._run_async(job, done)

    def _redraw_biomarkers(self):
        if not self.artifacts:
            return
        self.bio_fig.clf()
        if self.biomarker_view.get().startswith("SHAP") and self.artifacts.explanation:
            new = plot_feature_importance(self.artifacts.explanation.global_ranking())
        elif self.artifacts.stability is not None:
            new = plot_stability(self.artifacts.stability)
        else:
            return
        self._transfer(new, self.bio_fig, self.bio_canvas)

    def _load_screening(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("EEG", "*.edf *.bdf *.csv *.mat *.npy *.npz")])
        self._screening = []
        for p in paths or []:
            try:
                self._screening.append(read_recording(
                    p, subject_id=Path(p).stem, sampling_rate=
                    self.cfg.preprocessing.target_sampling_rate))
                self._log(self.screen_log, f"loaded {Path(p).name}\n")
            except Exception as exc:
                self._log(self.screen_log, f"[error] {Path(p).name}: {exc}\n")

    def _screen(self):
        if not self.artifacts:
            messagebox.showwarning("No model", "Run validation first (tab 4).")
            return
        if not self._screening:
            messagebox.showwarning("No data", "Load recordings to screen.")
            return
        from .features import extract_recording_features
        from .modeling import aggregate_scores, build_pipeline, calibrate

        art = self.artifacts
        y = np.array([int(r["label"]) for r in art.index])
        best = max(art.results, key=lambda k: art.results[k].mean("roc_auc"))
        model = calibrate(build_pipeline(best, self.cfg, art.feature_names),
                          art.X, y, self.cfg.validation.calibration)

        self._log(self.screen_log, "\n" + "=" * 60 + f"\nSCREENING ({best})\n" + "=" * 60 + "\n")
        for rec in self._screening:
            ep = preprocess_recording(rec, self.cfg.preprocessing)
            Xs, _, idx = extract_recording_features(ep, self.cfg)
            if Xs.size == 0:
                self._log(self.screen_log, f"\n{rec.recording_id}: no usable epochs\n")
                continue
            rows = model.predict_proba(Xs)[:, 1]
            agg = aggregate_scores(rows, idx, self.cfg.validation.aggregation)
            lo, hi = np.percentile(rows, [2.5, 97.5])
            self._log(self.screen_log,
                      f"\n{rec.recording_id}\n"
                      f"  score              {agg.scores[0]:.3f}\n"
                      f"  epoch spread       {lo:.3f} – {hi:.3f} "
                      f"(2.5–97.5 percentile of {len(rows)} epoch-channels)\n"
                      f"  channel dispersion {agg.dispersion[0]:.3f} "
                      f"(high = channels disagree; expected for a focal finding)\n"
                      f"  usable epochs      {ep.n_good}/{ep.mask.size} "
                      f"({100 * ep.acceptance_rate:.0f}%)\n")
        self._log(self.screen_log, "\n" + DISCLAIMER + "\n")

    @staticmethod
    def _transfer(source_fig, target_fig, canvas):
        """Move axes from a freshly built figure into the embedded canvas."""
        target_fig.clf()
        for ax in source_fig.get_axes():
            ax.remove()
            ax.figure = target_fig
            target_fig.add_axes(ax)
        target_fig.set_size_inches(source_fig.get_size_inches())
        try:
            target_fig.tight_layout()
        except Exception:
            pass
        canvas.draw()
        plt.close(source_fig)


ABOUT_TEXT = f"""Tumor-from-EEG · research platform v{__version__}

WHAT THIS IS
  An open, reproducible workbench for quantitative EEG (qEEG) biomarker
  extraction and honest machine-learning evaluation. It computes 80+
  documented biomarkers across five scientific domains, selects among them
  with stability analysis, and evaluates classifiers under repeated,
  subject-disjoint, nested cross-validation.

WHAT THIS IS NOT
  It is not a diagnostic device and it is not validated for brain-tumour
  detection. No public EEG dataset with confirmed tumour labels exists, so
  the tumour application of this pipeline remains an untested hypothesis.
  The pipeline itself can and should be validated on a corpus with real
  labels — see docs/DATASETS.md.

WHAT CHANGED IN VERSION 2
  · subject_id is mandatory; validation is subject-disjoint by construction
  · nested cross-validation replaced select-and-report on the same folds
  · every data-dependent step moved inside the cross-validated pipeline
  · notch filter moved before the low-pass, where it actually does something
  · frequency bands validated against the filter passband
  · epochs defined in seconds, not samples, and resampled to a common rate
  · feature names derived analytically; the global RNG is never touched
  · sample entropy rewritten (~40x faster, identical values)
  · calibration, confidence intervals, DeLong and permutation testing added
  · every biomarker carries its own definition, interpretation and references

WORKFLOW
  1  Cohort       load a manifest; read the design audit before continuing
  2  Signal       inspect filtering, spectrum, band power and scalogram
  3  Features     extract the biomarker table; browse the data dictionary
  4  Validation   nested grouped cross-validation, model comparison, report
  5  Biomarkers   SHAP attribution and selection stability
  6  Screen       score unlabelled recordings with an uncertainty range

LICENCE
  PolyForm Noncommercial 1.0.0 — free for research, teaching and personal
  use; commercial use requires a separate licence from the author.

CITATION
  See CITATION.cff in the repository root.
"""


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S")
    app = TumorFromEEGApp()
    app.mainloop()


if __name__ == "__main__":
    main()
