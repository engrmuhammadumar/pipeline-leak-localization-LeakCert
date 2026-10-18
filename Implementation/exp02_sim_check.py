r"""
exp02_sim_check.py - is the plate simulator realistic ENOUGH? (check before training any network)

Runs the SAME classical baseline (matched-filter picks + TDOA grid search, v = fixed-effects speed from real
data) on
  (a) the 9 real impacts
  (b) many simulated impacts at the same 9 positions (nuisances randomised)
If the simulator is realistic, the simulated error distribution should cover the real errors (~50 mm).
If simulated errors are much smaller, the simulator is too clean and a network trained on it will fail on
real data (sim-to-real gap) -> we must widen the nuisance ranges first.
Also saves a side-by-side plot of real vs simulated waveforms at one position.

Usage:  python .\exp02_sim_check.py
Outputs: results\exp02_sim_check.txt / .png
"""
import os
import sys

import numpy as np

from exp01_plate_baseline import bandpass
from exp01c_picks import NOISE, XTALK_GUARD_US, crosstalk_onset, loc_grid, picks
from loaders import PLATE_DIR, load_plate_real
from plate_sim import SENSORS, simulate

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

V_BASE = 3100.0            # fixed-effects speed of the matched-filter picker on real data (exp01c)
N_PER_POS = 30


def baseline_error(x, fs, src):
    xf = bandpass(x, fs)
    k0 = crosstalk_onset(xf, fs)
    start = (k0 + int(XTALK_GUARD_US * 1e-6 * fs)) if k0 is not None else NOISE[1]
    p = picks(xf, fs, "mf", start) / fs
    if not np.isfinite(p).all():
        return np.nan
    xy, _ = loc_grid(SENSORS, p, V_BASE)
    return np.linalg.norm(xy - src) * 1e3


def signal_features(x, fs):
    """Location-agnostic waveform properties per channel (no labels used):
    dur  - duration [us] of the first strong packet at half of its peak envelope
    late - energy ratio  E[t1+300, t1+800 us] / E[t1, t1+300 us]   (reverberation strength)
    fc   - spectral centroid [kHz] in 100-400 kHz
    """
    from scipy import signal as ss
    xf = bandpass(x, fs)
    k0 = crosstalk_onset(xf, fs)
    start = (k0 + int(XTALK_GUARD_US * 1e-6 * fs)) if k0 is not None else NOISE[1]
    env = np.abs(ss.hilbert(xf, axis=1))
    w = max(1, int(10e-6 * fs))
    env = np.apply_along_axis(lambda e: np.convolve(e, np.ones(w) / w, "same"), 1, env)
    feats = []
    us = lambda a: int(a * 1e-6 * fs)
    for c in range(xf.shape[0]):
        e = env[c, start:]
        k = np.where(e > 0.5 * e.max())[0]
        if not k.size:
            feats.append((np.nan, np.nan, np.nan)); continue
        k = k[0]
        pk = k + int(np.argmax(e[k:k + us(300)]))
        half = 0.5 * e[pk]
        a = pk - np.argmax(e[pk::-1] < half) if np.any(e[pk::-1] < half) else 0
        b = pk + np.argmax(e[pk:] < half) if np.any(e[pk:] < half) else e.size
        dur = (b - a) / fs * 1e6
        t1 = np.where(e > 0.2 * e.max())[0][0]
        early = np.sum(xf[c, start + t1:start + t1 + us(300)] ** 2)
        late = np.sum(xf[c, start + t1 + us(300):start + t1 + us(800)] ** 2)
        F = np.abs(np.fft.rfft(xf[c, start + t1:])) ** 2
        fr = np.fft.rfftfreq(xf.shape[1] - start - t1, 1 / fs)
        m = (fr > 100e3) & (fr < 400e3)
        fc = np.sum(fr[m] * F[m]) / (np.sum(F[m]) + 1e-30) / 1e3
        feats.append((dur, late / (early + 1e-30), fc))
    return np.array(feats)


def main(folder=PLATE_DIR, seed=0):
    os.makedirs("results", exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    rng = np.random.default_rng(seed)
    real = load_plate_real(folder)
    fs = real[0]["fs"]
    say(f"baseline: matched-filter picks + TDOA, v = {V_BASE:.0f} m/s, {N_PER_POS} simulations per position\n")
    say(f"{'position (mm)':<16}{'real err':>10}{'sim median':>12}{'sim p10':>9}{'sim p90':>9}  real within sim 10-90%?")
    all_real, all_sim = [], []
    for r in real:
        er = baseline_error(r["x"], fs, r["source"])
        es = np.array([baseline_error(simulate(r["source"], rng)[0], fs, r["source"]) for _ in range(N_PER_POS)])
        es = es[np.isfinite(es)]
        p10, p50, p90 = np.percentile(es, [10, 50, 90])
        inside = "yes" if p10 <= er <= p90 else ("BELOW (sim harder)" if er < p10 else "ABOVE (sim too easy)")
        say(f"{str(np.round(r['source']*1e3).astype(int).tolist()):<16}{er:10.1f}{p50:12.1f}{p10:9.1f}{p90:9.1f}  {inside}")
        all_real.append(er); all_sim.extend(es)
    all_real, all_sim = np.array(all_real), np.array(all_sim)
    say(f"\nreal: mean {all_real.mean():.1f} mm, median {np.median(all_real):.1f} mm")
    say(f"sim : mean {all_sim.mean():.1f} mm, median {np.median(all_sim):.1f} mm")
    q = np.mean(all_sim <= np.median(all_real))
    say(f"fraction of simulated errors below the real median: {q:.2f}  (about 0.5 = realistic difficulty; "
        f">0.8 = simulator too easy; <0.2 = simulator too hard)")
    say(f"fraction of errors < 15 mm: real {np.mean(all_real < 15):.2f} | sim {np.mean(all_sim < 15):.2f}"
        "   (similar = same error SHAPE, not just the same average)")

    # ---- waveform-property realism (location-agnostic, no labels used)
    say("\nWAVEFORM PROPERTIES (per channel): real vs simulated at the same 9 positions")
    fr = np.concatenate([signal_features(r["x"], fs) for r in real])
    fsim = np.concatenate([signal_features(simulate(r["source"], rng)[0], fs) for r in real for _ in range(10)])
    names = ["first-packet duration [us]", "late/early energy ratio", "spectral centroid [kHz]"]
    say(f"  {'property':<28}{'real median':>12}{'sim p10':>10}{'sim median':>12}{'sim p90':>10}"
        f"{'real median at sim pct':>24}")
    for i, nme in enumerate(names):
        a, b = fr[:, i][np.isfinite(fr[:, i])], fsim[:, i][np.isfinite(fsim[:, i])]
        pct = 100 * np.mean(b <= np.median(a))
        flag = "ok" if 15 <= pct <= 85 else ("sim too HIGH" if pct < 15 else "sim too LOW")
        say(f"  {nme:<28}{np.median(a):12.2f}{np.percentile(b, 10):10.2f}{np.median(b):12.2f}"
            f"{np.percentile(b, 90):10.2f}{pct:20.0f}%  {flag}")
    say("  'ok' = the real median lies inside the bulk of the simulated distribution.")
    with open(os.path.join("results", "exp02_sim_check.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = real[4]
        xs, _ = simulate(r["source"], rng)
        t = np.arange(r["x"].shape[1]) / fs * 1e6
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for a, x, title in ((ax[0], bandpass(r["x"], fs), "REAL"), (ax[1], bandpass(xs, fs), "SIMULATED")):
            for c in range(3):
                a.plot(t, x[c] / (np.abs(x[c]).max() + 1e-12) + 2.2 * c, lw=0.5)
            a.set_yticks([0, 2.2, 4.4]); a.set_yticklabels(["s1", "s2", "s3"])
            a.set_title(f"{title}  source {np.round(r['source']*1e3).astype(int).tolist()} mm")
        ax[1].set_xlim(400, 1600); ax[1].set_xlabel("time [us]")
        plt.tight_layout(); plt.savefig(os.path.join("results", "exp02_sim_check.png"), dpi=120)
        say("saved results\\exp02_sim_check.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PLATE_DIR)