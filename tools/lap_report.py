"""Figures and a replay page that compare the driver's run with the faster RL laps, as the first step towards coaching.

Input: the folder written by tools/lap_compare.py for each driver (lap_driver/keep/fast.csv, laps_chosen.csv, style_numbers.csv, baselines.json), as
    DATA/attacker/..., DATA/smooth/...
plus lap_small.csv + small.json (an RL lap only a little faster than the driver's run: picked by the pod script pick_small.py) and library.csv (all laps of the run, for figure 5).

    python tools/lap_report.py DATA OUT

Writes per driver five figures and, for both drivers together, one self-contained replay page:
    <driver>_1_lap_time.png      lap times of the four laps, time gained per sector, time gap along the lap
    <driver>_2_style.png         the 12 style numbers of the four laps against the target (spread units) and their values
    <driver>_3_lines_per_corner.png  the line on the map in each corner zone (X/Y), with the time of the zone (driver, style kept, fastest)
    <driver>_4_driving.png       speed, pedals and steering along the lap (driver, style kept, fastest)
    <driver>_5_style_vs_lap_time.png  each of the 12 style numbers against lap time: all on-track laps of the run in grey, the four laps highlighted, the target
    replay.html                  the laps as moving dots on the map (both drivers, switch at the top) and the 12 numbers of each lap
    summary.csv                  one row per driver and lap: lap time, style distance D, gain against the driver's run (and against the no-RL track line, which is not shown)
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
sys.path.insert(0, str(ROOT / "tools"))
from lap_compare import COLS, sectors, shade, style_axes  # noqa: E402

COL = {"driver": "#555555", "small": "#CC79A7", "keep": "#0072B2", "fast": "#D55E00"}    # Okabe-Ito: grey, reddish purple, blue, vermillion (green and red are kept free for "toward" / "away from the target")
TOWARD, AWAY = "#1e8449", "#c0392b"
PLAIN = {"B_mean_active": "Brake pressure", "B_event_peak_mean": "Peak brake pressure", "B_apply_rate": "Brake attack (how fast it is applied)", "B_release_rate": "Brake release",
         "T_apply_rate": "Throttle attack", "T_release_rate": "Throttle lift", "S_reversals_5deg_per_km": "Steering corrections", "S_mean_abs_rate": "Hand speed",
         "S_corner_peak_angle": "Peak steering in corners", "L_border_mean_m": "Distance to the track edge", "L_inside_dist_apex_m": "Apex distance", "L_outside_dist_exit_m": "Exit width"}
GROUP_END = {3: "Braking", 5: "Throttle", 8: "Steering", 11: "Line"}      # index of the last number of each group
NAME = {"driver": "driver's run", "small": "RL lap, small gain", "keep": "RL lap, style kept", "fast": "RL fastest lap"}
THREE = ("driver", "keep", "fast")
FOUR = ("driver", "small", "keep", "fast")


def load(d):
    d = Path(d)
    laps = {}
    for k in FOUR:
        df = pd.read_csv(d / f"lap_{k}.csv")[COLS].copy()
        df["progress"] = np.maximum.accumulate(df.progress.to_numpy(float))
        laps[k] = df
    ch = pd.read_csv(d / "laps_chosen.csv")
    base = json.load(open(d / "baselines.json"))
    small = json.load(open(d / "small.json"))
    info = {k: {"lap_time": float(ch.lap_time[i]), "D": float(ch.D[i]), "source": str(ch.source[i])} for i, k in enumerate(THREE)}
    info["small"] = {"lap_time": float(small["lap_time"]), "D": float(small["D"]), "source": small["source"]}
    info["base"] = {"lap_time": float(base["neutral"]["lap_time"]), "D": float(base["neutral"]["D"]), "source": "baseline_neutral.csv"}      # not drawn: only a column in summary.csv
    sn = pd.read_csv(d / "style_numbers.csv")
    sn["small"] = [small["numbers"][k] for k in sn.number]
    lib = pd.read_csv(d / "library.csv")
    return laps, info, sn, lib


def corner_zones(segs):
    return [s for s in segs if s[1] == "corners"]


def fig_time(name, laps, info, g, at, segs, out):
    L = g[-1]
    rows = []
    for (sec, kind, lo, hi) in segs:
        hi = min(hi, L)
        tt = {k: float(np.interp(hi, laps[k].progress, laps[k].t) - np.interp(lo, laps[k].progress, laps[k].t)) for k in laps}
        rows.append({"sector": sec, "from_m": round(lo), "to_m": round(hi), **{f"time_{k}": v for k, v in tt.items()}, **{f"{k}_vs_driver": tt[k] - tt["driver"] for k in ("small", "keep", "fast")}})
    sec = pd.DataFrame(rows)
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.5], hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    order = list(FOUR)
    y = np.arange(len(order))[::-1]
    t0 = info["driver"]["lap_time"]
    for yi, k in zip(y, order):
        t = info[k]["lap_time"]
        ax.plot([t, t0], [yi, yi], color="#cccccc", lw=1.2, zorder=1)
        ax.scatter([t], [yi], color=COL[k], s=90, zorder=3)
        ax.text(t0 + 0.7, yi, f"{t:.2f} s   ({t0 - t:+.2f} s)" if k != "driver" else f"{t:.2f} s", va="center", ha="left", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[k] for k in order], fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    lo_t = min(info[k]["lap_time"] for k in order)
    ax.set_xlim(lo_t - 0.8, t0 + 4.6)
    ax.set_xlabel("lap time, s  (in brackets: time gained against the driver's run)")
    ax.set_title("1. Lap time of the four laps", loc="left", fontsize=11)
    style_axes(ax)
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(sec))
    for j, k in enumerate(("small", "keep", "fast")):
        v = sec[f"{k}_vs_driver"].to_numpy()
        ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=COL[k], label=NAME[k])
        for xi, vi in zip(x + (j - 1) * 0.27, v):
            ax.text(xi, vi + (0.02 if vi >= 0 else -0.02), f"{vi:+.1f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=6)
    ax.axhline(0, color=COL["driver"], lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([r.sector.replace(" ", "\n") for r in sec.itertuples()], fontsize=8)
    ax.set_ylabel("sector time minus the driver's run, s\n(below 0 = faster)", fontsize=9)
    ax.set_title("2. Where the time is gained: per sector (corner zones C, straights S)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    ax = fig.add_subplot(gs[1, :])
    for k in ("small", "keep", "fast"):
        ax.plot(g, at[k]["t"] - at["driver"]["t"], color=COL[k], lw=1.9, label=f"{NAME[k]}  {info[k]['lap_time']:.2f} s")
    ax.axhline(0, color=COL["driver"], lw=1.4, label=f"{NAME['driver']}  {t0:.2f} s")
    shade(ax, segs)
    ax.set_xlim(0, L)
    ax.set_xlabel("distance along the lap, m   (shaded = corner zones)")
    ax.set_ylabel("time behind (+) / ahead (-) of the driver's run, s")
    ax.set_title("3. The gap to the driver's run along the lap", loc="left", fontsize=11, pad=16)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(ax)
    fig.suptitle(f"{name}: lap times", x=0.01, ha="left", fontsize=13)
    fig.savefig(out / f"{name}_1_lap_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    sec.to_csv(out / f"{name}_sectors.csv", index=False)
    return sec


def fig_style(name, sn, info, out):
    keys = sn.number.tolist()
    tgt, spread = sn.target.to_numpy(float), sn.spread.to_numpy(float)
    vals = {"driver": sn.driver_run.to_numpy(float), "small": sn.small.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}
    sd = {k: (v - tgt) / spread for k, v in vals.items()}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    y = np.arange(len(keys))[::-1]
    ax.axvline(0, color="#000000", lw=1.2)
    ax.axvspan(-0.3, 0.3, color="#000000", alpha=0.06, lw=0)
    for k, off in (("driver", 0.27), ("small", 0.09), ("keep", -0.09), ("fast", -0.27)):
        ax.scatter(sd[k], y + off, color=COL[k], s=34, zorder=3, label=f"{NAME[k]}   D = {info[k]['D']:.2f}")
    for yi, a, b in zip(y, sd["driver"], sd["fast"]):
        ax.plot([a, b], [yi + 0.27, yi - 0.27], color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=9)
    ax.set_xlabel("distance from the target style, in spread units   (0 = on the target, grey band = within 0.3)")
    ax.set_title("Where each lap is against the driver's target style", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    style_axes(ax)
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax2.set_ylim(-0.6, len(keys) - 0.4)
    xs = {"target": 0.20, "driver": 0.40, "small": 0.60, "keep": 0.80, "fast": 1.00}
    for lab, k in (("target", "target"), ("driver's\nrun", "driver"), ("small\ngain", "small"), ("style\nkept", "keep"), ("fastest", "fast")):
        ax2.text(xs[k], len(keys) - 0.05, lab, ha="right", va="bottom", fontsize=9, color=COL.get(k, "#000000"), fontweight="bold")
    for yi, i in zip(y, range(len(keys))):
        for k, arr in (("target", tgt), ("driver", vals["driver"]), ("small", vals["small"]), ("keep", vals["keep"]), ("fast", vals["fast"])):
            ax2.text(xs[k], yi, f"{arr[i]:.2f}", ha="right", va="center", fontsize=9, color="#222222")
    ax2.set_title("The numbers themselves (units as in the metric)", loc="left", fontsize=11, pad=34)
    ax.set_title("Where each lap is against the driver's target style", loc="left", fontsize=11, pad=34)
    fig.suptitle(f"{name}: the 12 style numbers", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_2_style.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_metrics(name, sn, info, lib, out):
    """Each of the 12 style numbers against lap time: every on-track lap of the run (grey), the four laps (colour), the target and the band within 0.3 spread."""
    keys = sn.number.tolist()
    tgt, spread = sn.target.to_numpy(float), sn.spread.to_numpy(float)
    vals = {"driver": sn.driver_run.to_numpy(float), "small": sn.small.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}
    ok = lib[lib.lap_time.notna() & (lib.off_share <= 1e-3) & (lib.start_m.fillna(0.0) < 1.0)]
    fig, axs = plt.subplots(4, 3, figsize=(14, 12.5))
    for i, (ax, key) in enumerate(zip(axs.ravel(), keys)):
        ax.axhspan(tgt[i] - 0.3 * spread[i], tgt[i] + 0.3 * spread[i], color="#000000", alpha=0.07, lw=0)
        ax.axhline(tgt[i], color="#000000", lw=1.0, ls="--")
        ax.scatter(ok.lap_time, ok[key], s=7, color="#c8c8c8", zorder=1, linewidths=0)
        order = sorted(FOUR, key=lambda k: -info[k]["lap_time"])
        ax.plot([info[k]["lap_time"] for k in order], [vals[k][i] for k in order], color="#999999", lw=0.9, zorder=2)
        for k in FOUR:
            ax.scatter([info[k]["lap_time"]], [vals[k][i]], s=60, color=COL[k], zorder=3, edgecolors="white", linewidths=0.8)
        d = (vals["keep"][i] - vals["driver"][i]) / spread[i]
        ax.set_title(f"{key}\ndriver's run to style kept: {d:+.1f} spread", loc="left", fontsize=9)
        style_axes(ax)
        ax.tick_params(labelsize=8)
        if i >= 9:
            ax.set_xlabel("lap time, s", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=f"{NAME[k]}  {info[k]['lap_time']:.2f} s") for k in FOUR]
    handles += [plt.Line2D([], [], marker="o", ls="", color="#c8c8c8", label=f"all {len(ok)} on-track laps of the run"), plt.Line2D([], [], color="#000000", ls="--", label="target (band = within 0.3 spread)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{name}: the 12 style numbers against lap time", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{name}_5_style_vs_lap_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _vals(sn):
    return {"driver": sn.driver_run.to_numpy(float), "small": sn.small.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}


def _groups(ax, n):
    """thin separators between the style groups (braking, throttle, steering, line); rows are drawn from the top"""
    for end in GROUP_END:
        if end < n - 1:
            ax.axhline(n - 1 - end - 0.5, color="#cccccc", lw=0.8, zorder=0)


def fig_style_change(name, sn, info, out):
    """How much each style number changes from the driver's run to each RL lap (spread units)."""
    keys = sn.number.tolist()
    spread = sn.spread.to_numpy(float)
    v = _vals(sn)
    n = len(keys)
    y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    ax.axvspan(-0.3, 0.3, color="#000000", alpha=0.07, lw=0)
    ax.axvline(0, color="#000000", lw=1.1)
    for j, k in enumerate(("small", "keep", "fast")):
        d = (v[k] - v["driver"]) / spread
        chg = float(np.mean(np.abs(d)))
        ax.barh(y + (1 - j) * 0.27, d, height=0.25, color=COL[k], zorder=3,
                label=f"{NAME[k]}: {info['driver']['lap_time'] - info[k]['lap_time']:.2f} s faster, the style changes by {chg:.2f} on average")
        for yi, di in zip(y + (1 - j) * 0.27, d):
            ax.text(di + (0.04 if di >= 0 else -0.04), yi, f"{di:+.1f}", va="center", ha="left" if di >= 0 else "right", fontsize=7, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([PLAIN[k] for k in keys], fontsize=10)
    ax.set_ylim(-0.6, n - 0.4)
    _groups(ax, n)
    xmax = max(1.0, float(np.max(np.abs([(v[k] - v["driver"]) / spread for k in ("small", "keep", "fast")]))) * 1.18)
    ax.set_xlim(-xmax, xmax)
    for yi, i in zip(y, range(n)):
        ax.text(1.01, yi, f"{keys[i]}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color="#777777")
        ax.text(1.22, yi, f"{v['driver'][i]:.2f}  \u2192  {v['keep'][i]:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=COL["keep"])
    ax.text(1.01, n - 0.35, "code name", transform=ax.get_yaxis_transform(), fontsize=8, color="#777777", va="bottom")
    ax.text(1.22, n - 0.35, "driver's run \u2192 style kept\n(in the number's own unit)", transform=ax.get_yaxis_transform(), fontsize=8, color=COL["keep"], va="bottom")
    ax.set_xlabel("change from the driver's run, in spread units   (0 = unchanged, shaded = small change under 0.3, 1 = as different as two typical real driver-track cells)", fontsize=9)
    ax.set_title(f"{name}: how much does the style change when the lap gets faster?", loc="left", fontsize=13, pad=52)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
    style_axes(ax)
    fig.subplots_adjust(right=0.68, left=0.2)
    fig.savefig(out / f"{name}_6_style_change.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_toward_target(name, sn, info, out):
    """For each RL lap: does each style number move toward or away from the target compared with the driver's run?"""
    keys = sn.number.tolist()
    spread, tgt = sn.spread.to_numpy(float), sn.target.to_numpy(float)
    v = _vals(sn)
    n = len(keys)
    y = np.arange(n)[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(16, 7.6), sharey=True)
    off0 = (v["driver"] - tgt) / spread
    lim = max(1.5, float(np.max(np.abs([(v[k] - tgt) / spread for k in v]))) * 1.12)
    for ax, k in zip(axs, ("small", "keep", "fast")):
        off1 = (v[k] - tgt) / spread
        ax.axvspan(-0.3, 0.3, color="#000000", alpha=0.07, lw=0)
        ax.axvline(0, color="#000000", lw=1.1)
        closer = 0
        for yi, a, b in zip(y, off0, off1):
            good = abs(b) < abs(a)
            closer += int(good)
            ax.annotate("", xy=(b, yi), xytext=(a, yi), arrowprops=dict(arrowstyle="-|>", color=TOWARD if good else AWAY, lw=2.0, shrinkA=0, shrinkB=0), zorder=3)
            ax.scatter([a], [yi], s=36, color=COL["driver"], zorder=4)
            ax.scatter([b], [yi], s=46, color=COL[k], zorder=5, edgecolors="white", linewidths=0.8)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(-lim, lim)
        _groups(ax, n)
        d0, d1 = info["driver"]["D"], info[k]["D"]
        ax.set_title(f"{NAME[k]}\n{info[k]['lap_time']:.2f} s ({info['driver']['lap_time'] - info[k]['lap_time']:+.2f} s)   D {d0:.2f} \u2192 {d1:.2f}\ncloser to the target in {closer} of 12 numbers", loc="left", fontsize=10)
        ax.set_xlabel("distance from the target style, spread units\n(0 = on the target, shaded = within 0.3)", fontsize=9)
        style_axes(ax)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([PLAIN[k] for k in keys], fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL["driver"], label="driver's run"), plt.Line2D([], [], color=TOWARD, lw=2, label="the RL lap is closer to the target"),
               plt.Line2D([], [], color=AWAY, lw=2, label="the RL lap is further from the target")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{name}: does the faster lap move each style number toward the target or away from it?", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"{name}_7_style_toward_target.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(name, sn, info, lib, out):
    """Lap time against style, for every on-track lap of the run: distance to the target (D) and total change from the driver's run."""
    keys = sn.number.tolist()
    spread = sn.spread.to_numpy(float)
    v = _vals(sn)
    ok = lib[lib.lap_time.notna() & (lib.off_share <= 1e-3) & (lib.start_m.fillna(0.0) < 1.0)].copy()
    ch = (np.abs(ok[keys].to_numpy(float) - v["driver"]) / spread).mean(axis=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4))
    d_drv = info["driver"]["D"]
    a1.scatter(ok.lap_time, ok.D, s=9, color="#c8c8c8", linewidths=0, zorder=1)
    a1.axhline(d_drv, color=COL["driver"], lw=1.2, ls="--")
    a1.axhline(d_drv + 0.05, color="#999999", lw=0.9, ls=":")
    a1.text(ok.lap_time.max(), d_drv + 0.015, "driver's run", ha="right", va="bottom", fontsize=8, color=COL["driver"])
    a1.text(ok.lap_time.max(), d_drv + 0.065, "budget for 'style kept' (driver's D + 0.05)", ha="right", va="bottom", fontsize=8, color="#777777")
    a2.scatter(ok.lap_time, ch, s=9, color="#c8c8c8", linewidths=0, zorder=1)
    for k in ("driver", "small", "keep", "fast"):
        ck = float(np.mean(np.abs(v[k] - v["driver"]) / spread))
        a1.scatter([info[k]["lap_time"]], [info[k]["D"]], s=90, color=COL[k], zorder=4, edgecolors="white", linewidths=0.9, label=f"{NAME[k]}  {info[k]['lap_time']:.2f} s")
        a2.scatter([info[k]["lap_time"]], [ck], s=90, color=COL[k], zorder=4, edgecolors="white", linewidths=0.9)
        a1.annotate(f"D {info[k]['D']:.2f}", (info[k]["lap_time"], info[k]["D"]), textcoords="offset points", xytext=(7, 7), fontsize=8)
        a2.annotate(f"{ck:.2f}", (info[k]["lap_time"], ck), textcoords="offset points", xytext=(7, 7), fontsize=8)
    a1.scatter([], [], s=9, color="#c8c8c8", label=f"all {len(ok)} on-track laps of the run")
    a1.set_xlabel("lap time, s")
    a1.set_ylabel("D = distance of the 12 style numbers to the driver's target style\n(0 = identical, spread units)")
    a1.set_title("How far from the target style is each lap?", loc="left", fontsize=11)
    a2.set_xlabel("lap time, s")
    a2.set_ylabel("average change of the 12 numbers compared with the driver's run\n(spread units, 0 = same style as the driver's run)")
    a2.set_title("How much has the style changed compared with the driver's run?", loc="left", fontsize=11)
    for ax in (a1, a2):
        style_axes(ax)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"{name}: style against lap time", x=0.01, ha="left", fontsize=13, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{name}_8_style_vs_lap_time_overall.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_corners(name, laps, info, segs, sec, geo, corners, out):
    zs = corner_zones(segs)
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.6))
    for ax, (nm, kind, lo, hi) in zip(axs.ravel(), zs):
        gm = (geo.s_geom >= lo) & (geo.s_geom <= hi)
        xl, yl, xr, yr = (geo[c][gm].to_numpy() for c in ("x_left", "y_left", "x_right", "y_right"))
        ax.fill(np.concatenate([xl, xr[::-1]]), np.concatenate([yl, yr[::-1]]), color="#efefef", lw=0, zorder=0)       # the track surface
        ax.plot(xl, yl, color="#b5b5b5", lw=0.9, zorder=1)
        ax.plot(xr, yr, color="#b5b5b5", lw=0.9, zorder=1)
        for k in ("keep", "fast", "driver"):
            d = laps[k]
            m = (d.progress >= lo) & (d.progress <= hi)
            ax.plot(d.x[m], d.y[m], color=COL[k], lw=2.4 if k == "driver" else 2.0, ls="--" if k == "driver" else "-", label=NAME[k], zorder=4 if k == "driver" else 3)
        for r in corners.itertuples():
            if lo <= r.s_apex <= hi:
                j = int(np.argmin(np.abs(geo.s_geom - r.s_apex)))
                ax.plot(geo.x_center[j], geo.y_center[j], "o", ms=4, color="#000000", zorder=4)
                ax.annotate(f"T{int(r.corner)}", (geo.x_center[j], geo.y_center[j]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        t = sec[sec.sector == nm].iloc[0]
        ax.set_title(f"{nm}   {lo:.0f}-{hi:.0f} m\nzone time: driver {t.time_driver:.2f} s, kept {t.time_keep:.2f} s ({t.keep_vs_driver:+.2f}), fastest {t.time_fast:.2f} s ({t.fast_vs_driver:+.2f})", loc="left", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X, m", fontsize=8)
        ax.set_ylabel("Y, m", fontsize=8)
        style_axes(ax)
    axs[0, 0].legend(frameon=False, fontsize=8, loc="best")
    for ax in axs.ravel()[len(zs):]:
        ax.axis("off")
    fig.suptitle(f"{name}: the line in each corner zone (map view; light grey = the track, black dots = corner apexes, dashed dark grey = the driver's run)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{name}_3_lines_per_corner.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_driving(name, at, g, segs, out):
    fig, axs = plt.subplots(4, 1, figsize=(12.5, 10), sharex=True)
    for ax, c, lab in zip(axs, ("speed", "throttle", "brake", "steer"), ("speed, m/s", "throttle (0-1)", "brake (0-1)", "steering command (-1..1)")):
        for k in THREE:
            ax.plot(g, at[k][c], color=COL[k], lw=1.1 if k == "driver" else 1.4, label=NAME[k] if c == "speed" else None)
        ax.set_ylabel(lab, fontsize=9)
        shade(ax, segs, label=(c == "speed"))
        style_axes(ax)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    axs[-1].set_xlabel("distance along the lap, m   (shaded = corner zones)")
    fig.suptitle(f"{name}: what the car does along the lap", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_4_driving.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


REPLAY = (Path(__file__).resolve().parent / "replay_template.html").read_text(encoding="utf-8")


def replay(all_laps, all_info, all_sn, geo, corners, out):
    data = {}
    dt = 0.1
    track = {"s": np.round(geo.s_geom.to_numpy(), 1).tolist(), "xl": np.round(geo.x_left.to_numpy(), 1).tolist(), "yl": np.round(geo.y_left.to_numpy(), 1).tolist(),
             "xr": np.round(geo.x_right.to_numpy(), 1).tolist(), "yr": np.round(geo.y_right.to_numpy(), 1).tolist()}       # full resolution (1.55 m): the chase camera is close to the road
    for name, laps in all_laps.items():
        info, sn = all_info[name], all_sn[name]
        L = float(min(df.progress.iloc[-1] for df in laps.values()))
        segs = sectors(L)
        style = {"keys": sn.number.tolist(), "plain": [PLAIN[k] for k in sn.number], "target": sn.target.round(4).tolist(), "spread": sn.spread.round(4).tolist(),
                 "vals": {"driver": sn.driver_run.round(4).tolist(), "small": sn.small.round(4).tolist(), "keep": sn.style_kept.round(4).tolist(), "fast": sn.fastest.round(4).tolist()}}
        vmax = float(np.ceil(max(df.speed.max() for df in laps.values()) / 10.0) * 10.0)
        out_laps = {}
        for k, df in laps.items():
            tt = np.arange(0.0, float(df.t.iloc[-1]) + dt, dt)
            f = lambda c: np.interp(tt, df.t, df[c])      # noqa: E731
            out_laps[k] = {"x": np.round(f("x"), 1).tolist(), "y": np.round(f("y"), 1).tolist(), "v": np.round(f("speed"), 1).tolist(), "p": np.round(f("progress"), 1).tolist(),
                           "a": np.round(f("throttle"), 2).tolist(), "b": np.round(f("brake"), 2).tolist(), "lap_time": info[k]["lap_time"], "D": info[k]["D"]}
        zones = []
        for nm, kind, lo, hi in corner_zones(segs):
            j = int(np.argmin(np.abs(geo.s_geom - (lo + hi) / 2)))
            zones.append({"name": nm, "a": lo, "b": hi, "x": float(geo.x_center[j]), "y": float(geo.y_center[j]), "t_driver": float(np.interp(lo, laps["driver"].progress, laps["driver"].t))})
        sectors_ = [{"name": nm, "kind": kind, "a": float(lo), "b": float(min(hi, L))} for nm, kind, lo, hi in segs]
        data[name] = {"dt": dt, "L": L, "vmax": vmax, "laps": out_laps, "zones": zones, "sectors": sectors_, "style": style}
    (out / "replay.html").write_text(REPLAY.replace("__DATA__", json.dumps({"track": track, "drivers": data}, separators=(",", ":"))), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(ROOT / "data" / "tracks" / "monza_geometry.csv")
    corners = pd.read_csv(ROOT / "data" / "monza_corners.csv")
    all_laps, all_info, all_sn, summary = {}, {}, {}, []
    for name in ("attacker", "smooth"):
        if not (a.data / name).is_dir():
            continue
        laps, info, sn, lib = load(a.data / name)
        L = min(float(df.progress.iloc[-1]) for df in laps.values())
        g = np.arange(0.0, L, 2.0)
        at = {k: {c: np.interp(g, df.progress, df[c]) for c in ("t", "x", "y", "speed", "lateral", "steer", "throttle", "brake")} for k, df in laps.items()}
        segs = sectors(L)
        sec = fig_time(name, laps, info, g, at, segs, a.out)
        fig_style(name, sn, info, a.out)
        fig_metrics(name, sn, info, lib, a.out)
        fig_style_change(name, sn, info, a.out)
        fig_toward_target(name, sn, info, a.out)
        fig_tradeoff(name, sn, info, lib, a.out)
        fig_corners(name, laps, info, segs, sec, geo, corners, a.out)
        fig_driving(name, at, g, segs, a.out)
        all_laps[name], all_info[name], all_sn[name] = laps, info, sn
        for k in FOUR:
            summary.append({"driver": name, "lap": NAME[k], "lap_time_s": info[k]["lap_time"], "style_distance_D": info[k]["D"], "gain_vs_driver_run_s": info["driver"]["lap_time"] - info[k]["lap_time"],
                            "gain_vs_no_RL_line_s": info["base"]["lap_time"] - info[k]["lap_time"], "source": info[k]["source"]})
        sn.to_csv(a.out / f"{name}_style_numbers.csv", index=False)
    pd.DataFrame(summary).round(3).to_csv(a.out / "summary.csv", index=False)
    replay(all_laps, all_info, all_sn, geo, corners, a.out)
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
