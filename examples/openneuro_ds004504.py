"""
Download OpenNeuro ds004504 and write cohort manifests for this package.

ds004504 (Miltiadous et al., 2023) holds resting-state, eyes-closed EEG from
88 people: 36 with Alzheimer's disease (AD), 23 with frontotemporal dementia
(FTD) and 29 cognitively normal controls (CN). 19 channels, 10-20 montage,
500 Hz, about 10 minutes each, one recording per person. Licence CC0.

Alzheimer's disease is a reasonable real-data test for this pipeline because
its classical scalp-EEG sign is slowing, the same mechanism the band-ratio
features target.

    python examples/openneuro_ds004504.py --out data/ds004504
    python -m eegtumor.cli audit --manifest data/ds004504/cohort_ad_vs_cn.csv
    python -m eegtumor.cli run --manifest data/ds004504/cohort_ad_vs_cn.csv \
           --config configs/quick.yaml --out results/ad_vs_cn

Only the raw recordings are downloaded (about 2.1 GB), not the derivatives.

Cite: Miltiadous A. et al. (2023) A dataset of scalp EEG recordings of
Alzheimer's disease, frontotemporal dementia and healthy subjects from routine
EEG. Data 8(6), 95. doi:10.3390/data8060095
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

BASE = "https://s3.amazonaws.com/openneuro.org/ds004504"
TASKS = {"ad_vs_cn": ("A", "C"), "ftd_vs_cn": ("F", "C"), "ad_vs_ftd": ("A", "F")}


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/ds004504")
    ap.add_argument("--no-download", action="store_true",
                    help="only write manifests for files already present")
    args = ap.parse_args()
    out = Path(args.out)

    fetch(f"{BASE}/participants.tsv", out / "participants.tsv")
    with open(out / "participants.tsv", newline="") as f:
        people = list(csv.DictReader(f, delimiter="\t"))

    for p in people:
        sid = p["participant_id"]
        rel = f"{sid}/eeg/{sid}_task-eyesclosed_eeg.set"
        if not args.no_download:
            print(f"  {rel}")
            fetch(f"{BASE}/{rel}", out / rel)
        p["path"] = rel

    for task, (pos, neg) in TASKS.items():
        rows = [p for p in people if p["Group"] in (pos, neg) and (out / p["path"]).exists()]
        with open(out / f"cohort_{task}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "subject_id", "label", "recording_id", "age", "sex", "mmse"])
            for p in rows:
                w.writerow([p["path"], p["participant_id"], int(p["Group"] == pos),
                            p["participant_id"], p["Age"], p["Gender"], p["MMSE"]])
        n_pos = sum(p["Group"] == pos for p in rows)
        print(f"{task}: {len(rows)} subjects ({n_pos} positive) -> cohort_{task}.csv")


if __name__ == "__main__":
    main()
