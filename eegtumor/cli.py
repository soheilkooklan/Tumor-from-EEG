"""
Command-line interface.

Exists so that a published result can be regenerated with one reproducible
command rather than a sequence of mouse clicks:

    python -m eegtumor.cli run --manifest cohort.csv --config configs/default.yaml \
                               --out results/

Subcommands
-----------
run       full analysis: features -> nested CV -> figures -> report
audit     load a cohort and print only the design audit (fast sanity check)
features  extract the biomarker table and stop
dict      print the feature dictionary as CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import AnalysisConfig
from .experiment import extract_cohort_features, run_analysis
from .features import feature_dictionary
from .io import load_cohort
from .modeling import available_models


def _load_cfg(path: str | None) -> AnalysisConfig:
    return AnalysisConfig.from_yaml(path) if path else AnalysisConfig()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eegtumor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="full analysis")
    r.add_argument("--manifest", required=True)
    r.add_argument("--config")
    r.add_argument("--out", default="results")
    r.add_argument("--data-root")
    r.add_argument("--models", nargs="*", default=None,
                   help=f"subset of {available_models()}")

    a = sub.add_parser("audit", help="cohort design audit only")
    a.add_argument("--manifest", required=True)
    a.add_argument("--data-root")

    f = sub.add_parser("features", help="extract the feature table and stop")
    f.add_argument("--manifest", required=True)
    f.add_argument("--config")
    f.add_argument("--out", default="results")

    d = sub.add_parser("dict", help="print the feature dictionary as CSV")
    d.add_argument("--config")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if getattr(args, "models", None):
        unknown = [m for m in args.models if m not in available_models()]
        if unknown:
            p.error(f"unknown model(s) {unknown}. Available: {available_models()}")

    if args.command == "audit":
        cohort = load_cohort(args.manifest, getattr(args, "data_root", None))
        for k, v in cohort.summary().items():
            print(f"{k:28s} {v}")
        warn = cohort.audit()
        print("\n" + ("DESIGN WARNINGS" if warn else "No structural confounds detected."))
        for w in warn:
            print(f"  ! {w}")
        return 0

    if args.command == "dict":
        import pandas as pd
        pd.DataFrame(feature_dictionary(_load_cfg(args.config))).to_csv(sys.stdout, index=False)
        return 0

    if args.command == "features":
        cfg = _load_cfg(args.config)
        cohort = load_cohort(args.manifest)
        X, names, index = extract_cohort_features(cohort, cfg)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.concat([pd.DataFrame(index), pd.DataFrame(X, columns=names)], axis=1) \
          .to_csv(out / "features.csv", index=False)
        print(f"feature matrix {X.shape} -> {out / 'features.csv'}")
        return 0

    art = run_analysis(args.manifest, _load_cfg(args.config), args.out,
                       models=args.models, data_root=args.data_root)
    print(f"\nreport: {art.report_path}")
    for name, res in sorted(art.results.items(), key=lambda kv: -kv[1].mean("roc_auc")):
        print("  " + res.report_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
