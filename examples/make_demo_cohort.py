"""
Generate a synthetic demonstration cohort.

Why this exists
---------------
The repository ships no clinical data. This script builds a small synthetic
cohort with a *known, controlled* effect so that a new user can verify the
installation, see every figure the pipeline produces, and - most usefully -
check that the validation protocol behaves correctly when there is no effect at
all.

The simulated positive class has mild focal slowing: elevated delta and theta
power confined to two of eight channels, which is the pattern a structural
lesion produces on scalp EEG. Each simulated subject also gets an idiosyncratic
alpha peak, amplitude and noise floor shared across all of that subject's
channels, reproducing the between-subject variability that makes ungrouped
cross-validation dishonest.

    python examples/make_demo_cohort.py --out demo_data --effect 0.6
    python examples/make_demo_cohort.py --out demo_null --effect 0.0

With `--effect 0.0` the labels carry no signal whatsoever, so a correct
pipeline must return ROC-AUC near 0.5. If it returns 0.9, the pipeline is
leaking. This is the negative control described in docs/METHODS.md.

This is simulated data. It demonstrates the software; it validates nothing
about EEG or about brain tumours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def synth_recording(rng, fs=250.0, seconds=60.0, n_channels=8,
                    effect=0.6, positive=False):
    n = int(fs * seconds)
    t = np.arange(n) / fs

    # Subject-level idiosyncrasy, shared by every channel of this subject
    alpha_peak = rng.uniform(8.5, 11.5)
    amp = rng.uniform(12.0, 28.0)
    noise = rng.uniform(4.0, 10.0)
    aperiodic = rng.uniform(0.9, 1.6)

    # 1/f background via filtered white noise
    freqs = np.fft.rfftfreq(n, 1 / fs)
    shape = np.zeros_like(freqs)
    shape[1:] = freqs[1:] ** (-aperiodic / 2)

    focal = rng.choice(n_channels, size=2, replace=False) if positive else []

    channels = []
    for c in range(n_channels):
        spec = np.fft.rfft(rng.standard_normal(n)) * shape
        background = np.fft.irfft(spec, n)
        background = noise * background / (np.std(background) + 1e-12)

        alpha = amp * np.sin(2 * np.pi * alpha_peak * t + rng.uniform(0, 6.28))

        slowing = 0.0
        if positive and c in focal:
            # Focal polymorphic delta/theta, amplitude-modulated
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28))
            slowing = effect * amp * env * (
                np.sin(2 * np.pi * rng.uniform(1.5, 3.5) * t) +
                0.6 * np.sin(2 * np.pi * rng.uniform(4.5, 7.0) * t))
            alpha *= (1.0 - 0.4 * effect)       # alpha attenuation over the lesion

        channels.append(background + alpha + slowing)

    return np.vstack(channels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--effect", type=float, default=0.6,
                    help="0.0 = negative control (labels carry no signal)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "recordings").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lines = ["path,subject_id,label,recording_id,sampling_rate"]
    for s in range(args.subjects):
        label = int(s % 2 == 0)
        for sess in range(rng.integers(1, 3)):        # 1-2 sessions per subject
            data = synth_recording(rng, args.fs, args.seconds, args.channels,
                                   args.effect, positive=bool(label))
            rid = f"S{s:03d}_r{sess}"
            np.save(out / "recordings" / f"{rid}.npy", data.astype(np.float32))
            lines.append(f"recordings/{rid}.npy,S{s:03d},{label},{rid},{args.fs:g}")

    (out / "cohort.csv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} recordings from {args.subjects} subjects -> {out}")
    print(f"manifest: {out / 'cohort.csv'}")
    if args.effect == 0.0:
        print("\nNEGATIVE CONTROL: labels are independent of the signal.\n"
              "A correct pipeline must report ROC-AUC ~ 0.50 here.")


if __name__ == "__main__":
    main()
