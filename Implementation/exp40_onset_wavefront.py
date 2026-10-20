r"""
exp40_onset_wavefront.py - a SECOND, physically independent localiser: time the opening wavefront at the four sensors.

The lab cut every run from ONE continuous 20-s recording (split_log.csv: 10 normal files + 10 leak files, contiguous).
So each run contains the instant the valve opens. The opening burst reaches each sensor after |x - s_k| / v:
    t_k = t0 + |x - s_k| / v        (a "V" in the (sensor position, arrival time) plane, apex at the source)
This script
  1. picks the arrival of the burst at every sensor (band-limited log-envelope, 1 kHz, AIC change point) and, more
     robustly, the relative delays between sensors by cross-correlating the envelope EDGES;
  2. fits x (and t0) with the label-free speed of the paper (D["v_cal"]) -> x_onset;
  3. compares x_onset with the coherence/echo estimate x_echo of the paper: two independent physics. When they agree
     (|dx| <= --agree m) the system answers, otherwise it abstains -> "consensus" accuracy / coverage;
  4. draws the wavefront ("moveout") figure and acoustic evidence maps over time (position x time, like soil
     temperature cloud maps, but for sound): the leak appears at the valve when the valve opens.

Usage:  python .\exp40_onset_wavefront.py --sessions Z:\20261007 Z:\20261008   [--agree 3] [--start 210]
Outputs: figures_all\wavefront\fig_###_*.png, results\exp40_report.txt, results\exp40_runs.csv
"""
import argparse
import csv
import os
import re

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

import exp28_extra_analyses as X
import make_all_figures as F
import exp13_testbed as T
import exp15_multipath as M

import matplotlib.pyplot as plt

COL, VCOL = F.COL, F.VCOL
FE = 1000.0            # envelope sampling rate [Hz]
REPORT = []


def say(s=""):
    print(s, flush=True); REPORT.append(s)


# ------------------------------------------------------------------------------------------------ envelopes
def envelopes(files_n, files_l, cache):
    if os.path.exists(cache):
        d = np.load(cache); return {k: d[k] for k in d.files}
    xs, fs = [], None
    for fn in list(files_n) + list(files_l):
        x, fs = T.load_mat(fn)
        x = np.asarray(x, float)
        if x.shape[0] != 4 and x.shape[1] == 4:
            x = x.T
        xs.append(x[:4] - x[:4].mean(axis=1, keepdims=True))
    x = np.concatenate(xs, axis=1)
    sos = signal.butter(4, T.MOD["VIB"]["band"], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=1)
    q = int(round(fs / FE))
    e = uniform_filter1d(y ** 2, q, axis=1)[:, ::q]                        # 1-ms energy envelope
    out = dict(env=np.log10(e + 1e-30), fs=fs, n_normal=len(files_n) * xs[0].shape[1] / fs)
    np.savez(cache, **out)
    return out


def aic_pick(z):
    """AIC change point of a 1-D sequence (Maeda 1985 form)."""
    n = len(z); k = np.arange(2, n - 2)
    c1 = np.cumsum(z); c2 = np.cumsum(z ** 2)
    v1 = c2[k - 1] / k - (c1[k - 1] / k) ** 2
    v2 = (c2[-1] - c2[k - 1]) / (n - k) - ((c1[-1] - c1[k - 1]) / (n - k)) ** 2
    a = k * np.log(np.maximum(v1, 1e-12)) + (n - k - 1) * np.log(np.maximum(v2, 1e-12))
    return int(k[np.argmin(a)])


def onset_times(env, t_split, half=1.5, smooth_ms=20):
    """common onset (largest rise of the summed envelope near the split) + per-sensor AIC picks + edge delays."""
    t = np.arange(env.shape[1]) / FE
    es = uniform_filter1d(env, int(smooth_ms * FE / 1000), axis=1)
    tot = es.mean(0)
    w = (t > t_split - 3.0) & (t < t_split + 3.0)
    d = np.gradient(uniform_filter1d(tot, int(0.1 * FE)))
    tc = float(t[w][np.argmax(d[w])])
    win = (t > tc - half) & (t < tc + half); i0 = int(np.argmax(win))
    picks = np.array([(i0 + aic_pick(es[k, win])) / FE for k in range(4)])
    # edge cross-correlation: derivative of the smoothed envelope, positive part (rising edges only)
    e50 = uniform_filter1d(env, int(0.05 * FE), axis=1)[:, win]                 # 50-ms envelope
    edge = uniform_filter1d(np.maximum(np.gradient(e50, axis=1), 0), int(0.03 * FE), axis=1)
    edge = (edge - edge.mean(1, keepdims=True)) / (edge.std(1, keepdims=True) + 1e-12)
    maxlag = int(0.8 * FE); tau = {}
    for i, j in T.PAIRS:
        c = signal.correlate(edge[i], edge[j], mode="full")
        lags = np.arange(-edge.shape[1] + 1, edge.shape[1])
        m = np.abs(lags) <= maxlag
        tau[(i, j)] = float(lags[m][np.argmax(c[m])] / FE)                  # >0: sensor i hears it later
    rise = es[:, (t > tc) & (t < tc + 0.5)].mean(1) - es[:, (t > tc - 2) & (t < tc - 0.5)].mean(1)
    return dict(tc=tc, picks=picks, tau=tau, rise=rise, es=es, t=t)


def fit_x(tau, v, pos, w=None):
    """x minimising the robust misfit of pair delays: (|x-s_i| - |x-s_j|)/v - tau_ij (Huber, 30 ms)."""
    xg = M.XG; cost = np.zeros(len(xg))
    for (i, j), tij in tau.items():
        pred = (np.abs(xg - pos[i]) - np.abs(xg - pos[j])) / v
        r = np.abs(pred - tij); h = 0.03
        cost += (w[(i, j)] if w else 1.0) * np.where(r < h, 0.5 * r ** 2, h * (r - 0.5 * h))
    return float(xg[int(np.argmin(cost))]), cost


def rel_arrivals(tau):
    """arrival times t_k (min = 0) that best explain the pair delays t_i - t_j = tau_ij (least squares)."""
    A, b = [], []
    for (i, j), tij in tau.items():
        a = np.zeros(4); a[i], a[j] = 1, -1; A.append(a); b.append(tij)
    A.append(np.ones(4)); b.append(0.0)
    t = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return t - t.min()


def wavefront_speed(taus, pos):
    """label-free speed from the wavefront alone: v minimising sum over runs of min_x misfit(x, v)."""
    J = np.array([sum(fit_x(t, v, pos)[1].min() for t in taus) for v in M.V_SEARCH])
    return float(M.V_SEARCH[int(np.argmin(J))]), J


def fit_x_picks(picks, v, pos):
    xg = M.XG
    pred = np.abs(xg[:, None] - pos[None, :]) / v                             # [x, sensor]
    t0 = np.median(picks[None, :] - pred, axis=1)
    cost = np.sum(np.abs(picks[None, :] - pred - t0[:, None]), axis=1)
    return float(xg[int(np.argmin(cost))]), cost


# ------------------------------------------------------------------------------------------------ evidence map
def evidence_map(D, r, cache, nwin=2):
    if os.path.exists(cache):
        d = np.load(cache); return d["E"], d["t"]
    nf, lf = D["files"][r["run"]]["VIB"]
    x, fs, _, _ = X._load_seq(list(nf) + list(lf))
    f, G = X._segments(x, fs)
    hop = T.MOD["VIB"]["nper"] // 2 / fs
    ts = np.arange(len(G)) * hop + T.MOD["VIB"]["nper"] / fs / 2
    n_norm = len(nf) * 1.0
    ref = ts + T.MOD["VIB"]["nper"] / fs / 2 < n_norm - 0.5
    SN = G[ref].mean(0)
    E, tt = [], []
    for i in range(0, len(G) - nwin + 1):
        lags, cur = X.curves_ours(f, G[i:i + nwin].mean(0), SN, fs)
        s = M.score(X.as_run(cur, fs), lags, D["ce"], D["v_cal"])[0]
        E.append(s); tt.append(ts[i:i + nwin].mean() - n_norm)
    E, tt = np.array(E), np.array(tt)
    np.savez(cache, E=E, t=tt)
    return E, tt


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--agree", type=float, default=3.0, help="[m] two methods agree if their estimates differ by less")
    ap.add_argument("--start", type=int, default=210)
    args = ap.parse_args()
    X.OUTD = os.path.join("figures_all", "wavefront"); X.CNT[0] = args.start - 1
    F.style(); os.makedirs(X.OUTD, exist_ok=True); os.makedirs("results", exist_ok=True)
    DS = X.prepare(args.sessions)
    pos = T.POS.copy()
    rows, keep = [], {}
    for D in DS:
        cdir = os.path.join("results", "wavefront_cache", D["sname"]); os.makedirs(cdir, exist_ok=True)
        for r in D["runs"]:
            nf, lf = D["files"][r["run"]]["VIB"]
            tag = re.sub(r"[^\w]+", "_", r["run"])
            try:
                ev = envelopes(nf, lf, os.path.join(cdir, tag + "_env.npz"))
                o = onset_times(ev["env"], float(ev["n_normal"]))
                E, tt = evidence_map(D, r, os.path.join(cdir, tag + "_map.npz"))
            except Exception as e:
                print(f"    {r['run']}: {e}", flush=True); continue
            v = D["v_cal"]
            xe, _ = fit_x(o["tau"], v, pos)
            xp, _ = fit_x_picks(o["picks"], v, pos)
            xc = float(r["x_echo"])
            rows.append(dict(day=D["short"], run=r["run"], valve=r["valve"], xt=r["xt"], flag=bool(r["flag"]), x_coh=xc, x_edge=xe,
                             x_aic=xp, tc=o["tc"], t_split=float(ev["n_normal"]), rise_min=float(o["rise"].min()),
                             agree=abs(xe - xc) <= args.agree, v=v))
            keep[(D["short"], r["run"])] = (o, E, tt, D)
        print(f"    {D['short']}: {sum(q['day'] == D['short'] for q in rows)} runs", flush=True)
    if not rows:
        raise SystemExit("no runs")
    # fully independent variant: the speed is estimated from the wavefront itself (no coherence, no labels)
    vw = {}
    for day in sorted({q["day"] for q in rows}):
        R = [q for q in rows if q["day"] == day]
        vw[day], _ = wavefront_speed([keep[(q["day"], q["run"])][0]["tau"] for q in R], pos)
        for q in R:
            q["v_wave"] = vw[day]
            q["x_edge_vw"] = fit_x(keep[(q["day"], q["run"])][0]["tau"], vw[day], pos)[0]

    say("=== Opening-wavefront localisation (independent of the coherence method) ===")
    say(f"  opening found {np.median([q['tc'] - q['t_split'] for q in rows]):+.2f} s from the lab's normal/leak split (median)")
    say("  wavefront-only speed per day: " + ", ".join(f"{d} {v:.0f} m/s" for d, v in vw.items())
        + "  (coherence self-calibration: " + ", ".join(f"{q['day']} {q['v']:.0f}" for q in {q['day']: q for q in rows}.values()) + " m/s)")
    say(f"{'method':<34}{'n':>4}{'median |err|':>14}{'<=1 m':>8}{'<=2 m':>8}{'<=5 m':>8}")
    res = {}
    for key, lab in (("x_coh", "coherence + echoes (paper)"), ("x_edge", "wavefront, edge delays"),
                     ("x_edge_vw", "wavefront, own speed (indep.)"), ("x_aic", "wavefront, AIC picks")):
        e = np.array([abs(q[key] - q["xt"]) for q in rows]); res[lab] = e
        say(f"{lab:<34}{len(e):>4}{np.median(e):14.2f}{np.mean(e <= 1) * 100:7.0f}%{np.mean(e <= 2) * 100:7.0f}%{np.mean(e <= 5) * 100:7.0f}%")
        X.metric("wavefront", f"{lab}: median / <=2 m / <=5 m", f"{np.median(e):.2f} / {np.mean(e <= 2):.3f} / {np.mean(e <= 5):.3f}")
    A = [q for q in rows if q["agree"]]
    if A:
        e = np.array([abs(q["x_coh"] - q["xt"]) for q in A])
        say(f"\n  CONSENSUS (answer only when both physics agree within {args.agree} m): coverage {len(A)}/{len(rows)} "
            f"({len(A) / len(rows) * 100:.0f}%), median |err| {np.median(e):.2f} m, <=2 m {np.mean(e <= 2) * 100:.0f}%, max {e.max():.1f} m")
        X.metric("wavefront", "consensus coverage / <=2 m / max err",
                 f"{len(A) / len(rows):.3f} / {np.mean(e <= 2):.3f} / {e.max():.2f}")
        W = [q for q in rows if abs(q["x_coh"] - q["xt"]) > 2]
        if W:
            say(f"  coherence errors > 2 m: {len(W)}; caught by disagreement: {sum(not q['agree'] for q in W)}")
    for vn in T.LEAK:
        R = [q for q in rows if q["valve"] == vn]
        if R:
            e1 = np.array([abs(q["x_edge"] - q["xt"]) for q in R])
            say(f"  {vn}: wavefront median |err| {np.median(e1):.2f} m (<=2 m {np.mean(e1 <= 2) * 100:.0f}%), "
                f"agreement {np.mean([q['agree'] for q in R]) * 100:.0f}%, weakest-sensor burst rise {np.median([q['rise_min'] for q in R]):.2f} (log10)")

    # ---- figure 1: moveout (wavefront) per valve, arrivals from the edge delays
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, vn in zip(axs, T.LEAK):
        R = [q for q in rows if q["valve"] == vn]
        if not R:
            continue
        rels = np.array([rel_arrivals(keep[(q["day"], q["run"])][0]["tau"]) for q in R])
        for rr in rels:
            ax.plot(pos, rr * 1e3, "o", color=VCOL[vn], alpha=.3, ms=6)
        med = np.median(rels, 0)
        xt = T.LEAK[vn]; v = np.median([q["v_wave"] for q in R])
        xx = np.linspace(-5, T.PIPE_END + 5, 400)
        ax.plot(xx, (np.abs(xx - xt) - np.abs(pos - xt).min()) / v * 1e3, "k--", lw=1.6,
                label=f"V from the true valve ({v:.0f} m/s, wavefront speed)")
        ax.plot(pos, med * 1e3, "s", ms=11, color=VCOL[vn], markeredgecolor="black", label="median arrival (edge delays)")
        ax.axvline(xt, color=VCOL[vn], ls=":")
        ax.set_xlabel("sensor position along the pipe [m]"); ax.set_ylabel("arrival of the opening burst [ms]")
        ax.set_title(f"{vn.upper()} (valve at {xt} m), {len(R)} runs"); ax.legend(fontsize=8)
    fig.suptitle("The opening wavefront: arrival time vs sensor position forms a V with its apex at the leak", fontweight="bold")
    fig.tight_layout(); X.save(fig, "wavefront_moveout")

    # ---- figure 2: example envelopes
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, vn in zip(axs, T.LEAK):
        q = next((q for q in rows if q["valve"] == vn), None)
        if q is None:
            continue
        o = keep[(q["day"], q["run"])][0]
        w = (o["t"] > o["tc"] - 1) & (o["t"] < o["tc"] + 1.2)
        for k in range(4):
            z = o["es"][k, w]; z = (z - np.median(z[:300])) / (np.ptp(z) + 1e-9)
            ax.plot((o["t"][w] - o["tc"]) * 1e3, z + k * 1.1, color=COL["p"][k], lw=1.2)
            ax.axvline((o["picks"][k] - o["tc"]) * 1e3, ymin=(k * 1.1 + .1) / 4.6, ymax=(k * 1.1 + 1.0) / 4.6, color="black", lw=2)
            ax.text(-990, k * 1.1 + .55, f"P{k + 1} ({pos[k]:.0f} m)", fontsize=8)
        ax.set_yticks([]); ax.set_xlabel("time from the opening [ms]")
        ax.set_title(f"{q['day']} {q['run']}: x_wavefront {q['x_edge']:.1f} m, x_coherence {q['x_coh']:.1f} m (true {q['xt']})", fontsize=9)
    fig.suptitle("Band-limited log-envelopes at the four sensors and the picked arrivals", fontweight="bold")
    fig.tight_layout(); X.save(fig, "wavefront_envelopes")

    # ---- figure 3: agreement + accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axs[0]
    for vn in T.LEAK:
        R = [q for q in rows if q["valve"] == vn]
        ax.scatter([q["x_coh"] for q in R], [q["x_edge"] for q in R], s=45, color=VCOL[vn], edgecolor="black", label=vn.upper())
    ax.plot([0, T.PIPE_END], [0, T.PIPE_END], "k-", lw=1)
    ax.fill_between([0, T.PIPE_END], [-args.agree, T.PIPE_END - args.agree], [args.agree, T.PIPE_END + args.agree], color=COL["ok"], alpha=.15,
                    label=f"agree (+-{args.agree} m)")
    ax.set_xlabel("coherence + echoes estimate [m]"); ax.set_ylabel("wavefront estimate [m]"); ax.legend()
    ax.set_title("(a) two independent physics, one answer")
    ax = axs[1]
    for lab, e in res.items():
        e = np.maximum(np.sort(e), 0.02)
        ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", lw=2.2, label=f"{lab}: <=2 m {np.mean(e <= 2) * 100:.0f}%")
    if A:
        e = np.maximum(np.sort([abs(q["x_coh"] - q["xt"]) for q in A]), 0.02)
        ax.step(e, np.arange(1, len(e) + 1) / len(rows), where="post", lw=2.6, color="black",
                label=f"consensus (answers {len(A) / len(rows) * 100:.0f}% of runs)")
    ax.set_xscale("log"); ax.axvline(2, color="black", ls=":"); ax.set_xlabel("|error| [m]"); ax.set_ylabel("share of runs")
    ax.legend(fontsize=8); ax.set_title("(b) accuracy")
    fig.tight_layout(); X.save(fig, "wavefront_agreement")

    # ---- figure 4: evidence maps over time ("acoustic cloud maps")
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
    for ax, vn in zip(axs, T.LEAK):
        R = [keep[(q["day"], q["run"])] for q in rows if q["valve"] == vn]
        if not R:
            continue
        n = min(len(r_[2]) for r_ in R)
        tt = R[0][2][:n]; Ls = []
        for r_ in R:
            S = r_[1][:n]; m = S.max(1)
            pre = m[tt < -1.0]; m0 = np.median(pre); s0 = 1.4826 * np.median(np.abs(pre - m0)) + 1e-9
            wgt = np.clip((m - m0) / (6 * s0), 0, 1)                            # 0 before the event, 1 when it is strong
            tau = 0.25 * (m - np.median(S, 1))[:, None] + 1e-9                 # soft-max: row median -> e^-4, peak -> 1
            Ls.append(wgt[:, None] * np.exp((S - m[:, None]) / tau))                # relative evidence along the pipe
        Em = np.mean(Ls, 0)
        Em = np.maximum.reduce([np.roll(Em, k, axis=1) for k in range(-20, 21)])  # +-2 m so a sharp peak stays visible
        im = ax.pcolormesh(M.XG, tt, Em, shading="auto", cmap="turbo", vmin=0, vmax=1); ax.grid(False)
        ax.axvline(T.LEAK[vn], color="white", ls="--", lw=1.5); ax.axhline(0, color="white", ls=":", lw=1.5)
        for k, p in enumerate(pos):
            ax.plot(p, tt[0], "^", ms=10, color=COL["p"][k], markeredgecolor="black", clip_on=False)
        ax.set_xlabel("position along the pipe [m]"); ax.set_ylabel("time from the valve opening [s]")
        ax.set_title(f"{vn.upper()}: mean of {len(R)} runs (dashed: valve)"); fig.colorbar(im, ax=ax, label="relative likelihood x event strength")
    fig.suptitle("Acoustic evidence maps: where along the pipe the sound comes from, second by second", fontweight="bold")
    fig.tight_layout(); X.save(fig, "evidence_time_maps")

    with open(os.path.join("results", "exp40_runs.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join("results", "exp40_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print(f"\nDONE: results\\exp40_report.txt, figures in {X.OUTD}")


if __name__ == "__main__":
    main()
