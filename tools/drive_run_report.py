"""Report for one training run of the drive package (style_rl/drive): how the run learned, and where its best lap loses time against the speed ceiling.

    python tools/drive_run_report.py RUN_DIR OUT_DIR [--track data/tracks/monza_geometry.csv] [--envelope data/ocd_envelope.json] [--margin 1.5]

RUN_DIR holds evals.jsonl and laps/eval_step_N.npz (what scripts/drive_train.py writes). Writes into OUT_DIR:
  fig_1_progress.png               lap time and clean laps at every evaluation of the run
  fig_2_best_lap_vs_ceiling.png    the best lap against the ceiling: speed, time lost so far, pedals
  evaluations.csv                  one row per evaluation (the evals.jsonl, flattened)
  sectors_best_lap.csv             time of the best lap and of the ceiling in each sector, and the difference
  failures.csv                     where each failed deterministic evaluation left the track

The ceiling is the QSS speed profile of the course (style_rl/drive/course.py): the fastest speed the measured car limits allow along the reference line. Its clock is scaled so that one
lap equals course.qss_lap_time_s. Sectors are the ones of tools/lap_compare.py (corner zones from 120 m before the first brake point to 100 m after the last exit, the rest straights).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from style_rl.drive.course import Course  # noqa: E402

BLUE, ORANGE, GREY, INK, MUTED = "#0072B2", "#D55E00", "#6E6E6E", "#222222", "#6E6E6E"       # Okabe-Ito blue and vermillion: distinguishable with colour-vision differences
CORNER_NAMES = {1: "Rettifilo chicane (T1)", 2: "Rettifilo chicane (T2)", 3: "Roggia chicane (T3)", 4: "Roggia chicane (T4)", 5: "Lesmo 1 (T5)", 6: "Lesmo 2 (T6)",
                7: "Ascari (T7)", 8: "Ascari (T8)", 9: "Ascari (T9)", 10: "Parabolica (T10)"}
ZONE_NAMES = {"C T1-2": "Rettifilo", "C T3-6": "Roggia + Lesmo", "C T7-9": "Ascari", "C T10": "Parabolica"}
REFERENCES = [(111.85, "speed-limit estimate (ceiling) 111.85 s", "--"), (115.72, "earlier best RL lap 115.72 s", ":"), (117.7, "neutral pure pursuit 117.7 s", ":"), (125.0, "scripted teacher at its best, about 125 s", ":")]


def sectors(L, corners_csv):
    c = pd.read_csv(corners_csv)
    zones = []
    for r in c.itertuples():
        a, b = r.s_start - 120.0, r.s_end + 100.0
        if zones and a <= zones[-1]["b"] + 60.0:
            zones[-1]["b"] = max(zones[-1]["b"], b)
            zones[-1]["corners"].append(int(r.corner))
        else:
            zones.append({"a": a, "b": b, "corners": [int(r.corner)]})
    segs, p, ns = [], 0.0, 0
    for z in zones:
        if z["a"] > p + 1.0:
            segs.append((f"S{ns}", "straight", p, z["a"]))
            ns += 1
        cs = z["corners"]
        segs.append((f"C T{cs[0]}" + (f"-{cs[-1]}" if len(cs) > 1 else ""), "corners", z["a"], z["b"]))
        p = z["b"]
    if p < L - 1.0:
        segs.append((f"S{ns}", "straight", p, L))
    return segs


def lap_from_trace(z, L, k):
    """Lap k (0 = first timed lap) of an evaluation trace: stations (monotone, 0..L), time since the line, lap time, sample indices."""
    s, t = z["station_m"], z["time_s"]
    w = np.where(np.diff(s) < -L / 2)[0]                 # the samples just before each crossing of the line
    if len(w) < k + 2:
        return None

    def cross(i):
        f = (L - s[i]) / (s[i + 1] + L - s[i])
        return t[i] + f * (t[i + 1] - t[i])

    lo, hi = w[k], w[k + 1]
    t0, t1 = cross(lo), cross(hi)
    idx = np.arange(lo + 1, hi + 1)
    ss = np.concatenate([[0.0], s[idx], [L]])
    tt = np.concatenate([[0.0], t[idx] - t0, [t1 - t0]])
    return np.maximum.accumulate(ss), tt, t1 - t0, idx


def ceiling_clock(course):
    v = course.v_qss
    dt = course.ds / (0.5 * (v + np.roll(v, -1)))
    t = np.concatenate([[0.0], np.cumsum(dt)])
    t *= course.qss_lap_time_s / t[-1]
    st = np.append(np.arange(course.N) * course.ds, course.L)
    return lambda s: np.interp(s, st, t)


def corner_of(station, corners):
    """The corner a station belongs to: the one whose window (150 m before its start to 60 m after its end) holds it and whose apex is nearest."""
    near = [r for r in corners.itertuples() if r.s_start - 150.0 <= station <= r.s_start + r.length_m + 60.0]
    if not near:
        return f"straight at {station:.0f} m"
    r = min(near, key=lambda r: abs(r.s_apex - station))
    return CORNER_NAMES.get(int(r.corner), f"T{int(r.corner)}") + (" entry" if station < r.s_start else "")


def load_evals(run):
    return [json.loads(line) for line in open(run / "evals.jsonl") if line.strip()]


def evaluation_table(ev):
    rows = [{"step": e["step"], "hours": e["hours"], "deterministic_clean": e["det_clean"], "deterministic_end": e["det_end"], "lap1_s": e["det_lap1_s"], "lap2_s": e["det_lap2_s"],
             "noise_clean_of": f'{e["sto_clean"]}/{e["sto_laps"]}', "noise_clean": e["sto_clean"], "noise_lap2_mean_s": e["sto_lap2_mean_s"], "alpha": e["alpha"]} for e in ev]
    return pd.DataFrame(rows)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)


def fig_progress(ev, best, out):
    ev = [e for e in ev if e["step"] >= 1000]
    steps = np.array([e["step"] for e in ev]) / 1e6
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [3.3, 1.6], "hspace": 0.1})
    for y, label, ls in REFERENCES:
        ax.axhline(y, color=GREY, lw=1.0, ls=ls, zorder=1)
        ax.text(0.006, y + 0.08, label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=9, color=MUTED)
    clean = [(e["step"] / 1e6, e["det_lap2_s"]) for e in ev if e["det_clean"]]
    ax.plot([c[0] for c in clean], [c[1] for c in clean], ls="none", marker="o", ms=9, color=BLUE, mec="white", mew=1.2, zorder=4, label="test without noise: lap 2 of 2 (clean runs only)")
    noisy = [(e["step"] / 1e6, e["sto_lap2_mean_s"]) for e in ev if e["sto_lap2_mean_s"] is not None]
    ax.plot([c[0] for c in noisy], [c[1] for c in noisy], marker="D", ms=6.5, lw=1.4, color=ORANGE, mec="white", mew=1.0, zorder=3, label="average lap with training noise (clean laps only)")
    ax.annotate(f'best: {best["det_lap2_s"]:.2f} s\nat step {best["step"] // 1000}k', xy=(best["step"] / 1e6, best["det_lap2_s"]), xytext=(best["step"] / 1e6 - 0.07, best["det_lap2_s"] - 2.35),
                fontsize=10, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    ax.text(0.006, 0.88, "at the start of the run the copy of the teacher laps in 251 s (off this chart)", transform=ax.transAxes, fontsize=9.5, color=INK, va="top")
    ax.set_ylim(109.5, 126.5)
    ax.set_ylabel("lap time, s  (lower is better)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), frameon=False, fontsize=10, ncol=1)
    ax.set_title("Run a1: lap time and clean laps, tested every 50,000 training steps", loc="left", fontsize=13, color=INK, pad=12)
    style(ax)

    n = np.array([e["sto_clean"] for e in ev])
    bx.bar(steps, n, width=0.036, color=ORANGE, zorder=3)
    for e, x in zip(ev, steps):
        if e["det_clean"]:
            bx.plot(x, 5.75, marker="o", ms=9, color=BLUE, mec="white", mew=1.0, zorder=4)
        else:
            bx.plot(x, 5.75, marker="x", ms=8, mew=2.0, color=GREY, zorder=4)
    bx.set_ylim(0, 7.6)
    bx.set_yticks(range(0, 6))
    bx.set_ylabel("laps that stayed on\ntrack, of 5 (noisy)", fontsize=10)
    bx.set_xlabel("training steps, millions", fontsize=11)
    bx.set_xlim(0, steps.max() + 0.04)
    bx.text(0.006, 0.99, "symbols in the top row, test without noise:   ●  finished 2 clean laps     ✕  left the track.   Bars: noisy laps that stayed on track, out of 5", transform=bx.transAxes, ha="left", va="top", fontsize=9.5, color=MUTED)
    style(bx)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_best_lap(course, segs, z, lap, clock, sector_rows, title, out):
    L = course.L
    ss, tt, lap_time, idx = lap
    s = z["station_m"][idx]
    speed, thr, brk = z["speed"][idx] * 3.6, z["throttle"][idx], z["brake"][idx]
    grid = np.arange(0.0, L, 2.0)
    lost = np.interp(grid, ss, tt) - clock(grid)
    fig, axs = plt.subplots(3, 1, figsize=(12.5, 9.4), sharex=True, gridspec_kw={"height_ratios": [3.0, 2.0, 2.0], "hspace": 0.08})
    for ax in axs:
        for name, kind, a, b in segs:
            if kind == "corners":
                ax.axvspan(a, b, color="#000000", alpha=0.06, lw=0)
        style(ax)
    ax = axs[0]
    ax.plot(grid, course.qss_speed(grid) * 3.6, color=GREY, lw=1.6, ls="--", label="ceiling (speed limit of the car along the reference line)")
    ax.plot(s, speed, color=BLUE, lw=1.8, label=f"RL lap, {lap_time:.2f} s")
    ax.set_ylabel("speed, km/h", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
    for name, kind, a, b in segs:
        if kind == "corners":
            ax.text((a + b) / 2, 0.03, ZONE_NAMES.get(name, name), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5, color=MUTED)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=34)

    ax = axs[1]
    ax.plot(grid, lost, color=ORANGE, lw=2.0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("time lost against\nthe ceiling so far, s", fontsize=11)
    top = max(lost.max(), 1.0)
    ax.set_ylim(min(lost.min() - 0.3, -0.5), top * 1.28)
    for r in sector_rows:
        ax.text((r["from_m"] + r["to_m"]) / 2, 0.97, f'{r["loss_s"]:+.2f} s', transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5, color=INK)
    ax.text(L - 20, lost[-1] + 0.05, f"{lost[-1]:.2f} s at the line", ha="right", va="bottom", fontsize=9.5, color=ORANGE)

    ax = axs[2]
    ax.plot(s, thr, color=BLUE, lw=1.1, label="throttle")
    ax.plot(s, brk, color=ORANGE, lw=1.1, label="brake")
    ax.axhline(1.0, color=GREY, lw=0.9, ls=":")
    ax.text(L - 20, 1.02, "full", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("pedal, 0 to 1", fontsize=11)
    ax.set_xlabel("distance along the lap, m   (shaded = corner zones; the number above each sector in the middle row is the time it loses)", fontsize=10.5)
    ax.set_xlim(0, L)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--track", type=Path, default=ROOT / "data/tracks/monza_geometry.csv")
    ap.add_argument("--envelope", type=Path, default=ROOT / "data/ocd_envelope.json")
    ap.add_argument("--corners", type=Path, default=ROOT / "data/monza_corners.csv")
    ap.add_argument("--margin", type=float, default=1.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    course = Course(a.track, a.envelope, margin_m=a.margin, grid_m=1.0)
    L = course.L
    ev = load_evals(a.run)
    evaluation_table(ev).to_csv(a.out / "evaluations.csv", index=False)
    clean = [e for e in ev if e["det_clean"] and e["det_lap2_s"] is not None and e["step"] >= 1000]
    best = min(clean, key=lambda e: e["det_lap2_s"])
    fig_progress(ev, best, a.out / "fig_1_progress.png")

    segs = sectors(L, a.corners)
    clock = ceiling_clock(course)
    z = np.load(a.run / "laps" / f'eval_step_{best["step"]}.npz')
    lap = lap_from_trace(z, L, 1)
    ss, tt, lap_time, idx = lap
    rows = []
    for name, kind, lo, hi in segs:
        hi = min(hi, L)
        lo = max(lo, 0.0)
        t_rl = np.interp(hi, ss, tt) - np.interp(lo, ss, tt)
        t_q = float(clock(hi) - clock(lo))
        v_rl = z["speed"][idx][(z["station_m"][idx] >= lo) & (z["station_m"][idx] < hi)]
        thr = z["throttle"][idx][(z["station_m"][idx] >= lo) & (z["station_m"][idx] < hi)]
        rows.append({"sector": name, "kind": kind, "from_m": round(lo), "to_m": round(hi), "rl_s": round(t_rl, 2), "ceiling_s": round(t_q, 2), "loss_s": round(t_rl - t_q, 2),
                     "rl_mean_speed_kmh": round(3.6 * float(v_rl.mean()), 1), "ceiling_mean_speed_kmh": round(3.6 * float(course.qss_speed(z["station_m"][idx][(z["station_m"][idx] >= lo) & (z["station_m"][idx] < hi)]).mean()), 1),
                     "mean_throttle": round(float(thr.mean()), 2), "share_full_throttle": round(float((thr > 0.95).mean()), 2)})
    sec = pd.DataFrame(rows)
    sec.to_csv(a.out / "sectors_best_lap.csv", index=False)
    fig_best_lap(course, segs, z, lap, clock, rows, f'Best lap of the run (step {best["step"] // 1000}k, lap 2, {lap_time:.2f} s) against the ceiling of {course.qss_lap_time_s:.2f} s', a.out / "fig_2_best_lap_vs_ceiling.png")

    fails = []
    for e in ev:
        if e["det_clean"] or e["step"] < 1000:
            continue
        zz = np.load(a.run / "laps" / f'eval_step_{e["step"]}.npz')
        s = zz["station_m"]
        done = max(len(np.where(np.diff(s) < -L / 2)[0]) - 1, 0)
        fails.append({"step": e["step"], "laps_finished_before": done, "left_track_at_m": round(float(s[-1])), "where": corner_of(float(s[-1]), course.corners)})
    pd.DataFrame(fails).to_csv(a.out / "failures.csv", index=False)
    print(f"best clean test: step {best['step']}, lap 2 {best['det_lap2_s']:.3f} s; ceiling {course.qss_lap_time_s:.3f} s; scaled-clock loss at the line {np.interp(L, ss, tt) - clock(L):.3f} s")
    print(sec.to_string(index=False))
    print(pd.DataFrame(fails).to_string(index=False))


if __name__ == "__main__":
    main()
