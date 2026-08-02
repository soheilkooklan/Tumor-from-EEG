"""
Capture documentation screenshots from the real interface.

Run on any Linux machine with a display, or headlessly with Xvfb:

    xvfb-run -a --server-args="-screen 0 1440x960x24" \
        python tools/capture_screenshots.py --out docs/screenshots

The point of scripting this rather than using a screenshot key is that the
images in the README then regenerate automatically whenever the interface
changes, so the documentation cannot silently drift away from the software.

Requires ImageMagick (`import`) on the PATH, which is what actually grabs the
X11 window.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def grab(path: Path, delay: float = 0.6) -> None:
    time.sleep(delay)
    subprocess.run(["import", "-window", "root", str(path)], check=True)
    print(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--manifest", help="cohort manifest to load, for populated views")
    args = ap.parse_args()

    if not os.environ.get("DISPLAY"):
        print("no DISPLAY. Re-run under xvfb-run (see the module docstring).")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from eegtumor.gui import TumorFromEEGApp

    app = TumorFromEEGApp()
    app.geometry("1420x940")
    app.update()
    time.sleep(1.0)

    if args.manifest:
        try:
            app.load_manifest(args.manifest)
            for _ in range(30):
                app.update()
                time.sleep(0.1)
        except Exception as exc:                                # pragma: no cover
            print(f"  (could not preload cohort: {exc})")

        try:
            app._inspect_signal()          # populates the signal-inspection plots
            for _ in range(20):
                app.update()
                time.sleep(0.1)
        except Exception as exc:                                # pragma: no cover
            print(f"  (could not render the signal tab: {exc})")

    tabs = [
        ("1 · Cohort", "01_cohort.png"),
        ("2 · Signal", "02_signal.png"),
        ("3 · Features", "03_features.png"),
        ("4 · Validation", "04_validation.png"),
        ("5 · Biomarkers", "05_biomarkers.png"),
        ("6 · Screen", "06_screen.png"),
        ("About", "07_about.png"),
    ]
    for tab_name, filename in tabs:
        try:
            app.tabs.set(tab_name)
        except Exception as exc:                                # pragma: no cover
            print(f"  (skipping {tab_name}: {exc})")
            continue
        for _ in range(10):
            app.update()
            time.sleep(0.05)
        grab(out / filename)

    app.destroy()
    print(f"\n{len(tabs)} screenshots in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
