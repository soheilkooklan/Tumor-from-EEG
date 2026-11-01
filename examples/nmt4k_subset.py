"""
Fetch a balanced 240-recording subset of NMT-4K-EEG and write a manifest.

NMT-4K-EEG v1.1 (Masood et al., NUST, Zenodo, doi:10.5281/zenodo.21041582,
CC BY 4.0) holds 4,500 clinical EDF recordings labelled normal or abnormal,
with a fixed train/evaluation split. It is shipped as one 33 GB split ZIP64
archive. This script reads the archive's central directory with HTTP range
requests and pulls only the EDF files listed in nmt4k_subset240.csv (about
1.9 GB), so the full archive is never downloaded.

The subset is 60 recordings in each cell of label (normal, abnormal) x year
(2019, 2024), all from one hospital code ("mh"), 200 Hz, 22 channels. The
balanced design is what lets the year effect be measured apart from the label.

    python examples/nmt4k_subset.py --out data/nmt
    python -m eegtumor.cli run --manifest data/nmt/manifest240.csv \
           --config configs/quick.yaml --jobs 2 --out results/nmt

The archive carries no patient identifier (the EDF patient field is a
constant placeholder), so recordings are treated as subjects. Repeat visits
by one person, if any, cannot be grouped. Keep that in mind when reading any
result from this corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://zenodo.org/records/21041582/files/"
PARTS = ["NMT-4K-EEG.z01", "NMT-4K-EEG.z02", "NMT-4K-EEG.z03", "NMT-4K-EEG.zip"]
HERE = Path(__file__).resolve().parent


def _get(part, start, end):
    req = urllib.request.Request(BASE + part + "?download=1",
                                 headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read(), int(r.headers["Content-Range"].split("/")[1])


class SplitZip:
    """Random access to a split ZIP64 archive over HTTP."""

    def __init__(self):
        self.sizes = [_get(p, 0, 0)[1] for p in PARTS]
        last = self.sizes[-1]
        tail, _ = _get(PARTS[-1], last - 1_000_000, last - 1)
        j = tail.rfind(b"PK\x06\x07")
        if j < 0:
            raise RuntimeError("no ZIP64 locator; archive layout changed")
        _, disk, off, _ = struct.unpack("<IIQI", tail[j:j + 20])
        e64, _ = _get(PARTS[disk], off, off + 55)
        v = struct.unpack("<IQHHIIQQQQ", e64[:56])
        cd_disk, cd_size, cd_off = v[5], v[8], v[9]
        cd = self.read(cd_disk, cd_off, cd_size)
        self.entries = self._parse(cd)

    def read(self, disk, off, n):
        out = b""
        while n > 0:
            take = min(n, self.sizes[disk] - off)
            out += _get(PARTS[disk], off, off + take - 1)[0]
            n -= take
            disk, off = disk + 1, 0
        return out

    @staticmethod
    def _parse(cd):
        entries, p = {}, 0
        while cd[p:p + 4] == b"PK\x01\x02":
            f = struct.unpack("<HHHHHHIIIHHHHHII", cd[p + 4:p + 46])
            meth, csize, usize = f[3], f[7], f[8]
            nlen, xlen, clen, disk, off = f[9], f[10], f[11], f[12], f[15]
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                body, k = extra[q + 4:q + 4 + hlen], 0
                if hid == 1:
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                    if off == 0xFFFFFFFF:
                        off = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                    if disk == 0xFFFF:
                        disk = struct.unpack("<I", body[k:k + 4])[0]
                q += 4 + hlen
            entries[name] = dict(method=meth, csize=csize, usize=usize, disk=disk, off=off)
            p += 46 + nlen + xlen + clen
        return entries

    def extract(self, name):
        e = self.entries[name]
        hdr = self.read(e["disk"], e["off"], 30)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        disk, start = e["disk"], e["off"] + 30 + nlen + xlen
        while start >= self.sizes[disk]:
            start -= self.sizes[disk]
            disk += 1
        data = self.read(disk, start, e["csize"])
        if e["method"] == 8:
            data = zlib.decompress(data, -15)
        if len(data) != e["usize"]:
            raise RuntimeError(f"size mismatch for {name}")
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/nmt")
    ap.add_argument("--subset", default=str(HERE / "nmt4k_subset240.csv"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "edf").mkdir(parents=True, exist_ok=True)
    with open(args.subset, newline="") as f:
        rows = list(csv.DictReader(f))

    todo = [r for r in rows if not (out / "edf" / f"{r['file']}.edf").exists()]
    if todo:
        z = SplitZip()

        def fetch(r):
            dest = out / "edf" / f"{r['file']}.edf"
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(z.extract(r["entry"]))
            tmp.rename(dest)
            return r["file"]

        with ThreadPoolExecutor(args.threads) as pool:
            for i, name in enumerate(pool.map(fetch, todo), 1):
                print(f"  {i}/{len(todo)} {name}")

    with open(out / "manifest240.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "subject_id", "label", "recording_id", "year", "split", "age", "sex"])
        for r in rows:
            w.writerow([f"edf/{r['file']}.edf", r["file"], int(r["label"] == "Abnormal"),
                        r["file"], r["year"], r["split"], r["age"], r["gender"]])
    counts = {}
    for r in rows:
        counts[(r["label"], r["year"])] = counts.get((r["label"], r["year"]), 0) + 1
    print(json.dumps({f"{k[0]} {k[1]}": v for k, v in sorted(counts.items())}))
    print(f"wrote {out / 'manifest240.csv'}")


if __name__ == "__main__":
    main()
