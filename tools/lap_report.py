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
UNIT = {"B_mean_active": "pedal, 0-1", "B_event_peak_mean": "pedal, 0-1", "B_apply_rate": "pedal per s", "B_release_rate": "pedal per s", "T_apply_rate": "pedal per s",
        "T_release_rate": "pedal per s", "S_reversals_5deg_per_km": "per km", "S_mean_abs_rate": "deg per s", "S_corner_peak_angle": "deg", "L_border_mean_m": "m",
        "L_inside_dist_apex_m": "m", "L_outside_dist_exit_m": "m"}
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


def _vals(sn):
    return {"driver": sn.driver_run.to_numpy(float), "small": sn.small.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}


def _title(key):
    return f"{PLAIN[key]}   [{UNIT[key]}]\n{key}"


def fig_style(name, sn, info, out):
    """The 12 style numbers as they are (no normalisation), each in its own unit: target and the four laps."""
    keys = sn.number.tolist()
    tgt = sn.target.to_numpy(float)
    v = _vals(sn)
    rows = [("target", "target"), ("driver", "driver's run"), ("small", "small gain"), ("keep", "style kept"), ("fast", "fastest")]
    fig, axs = plt.subplots(4, 3, figsize=(14.5, 12.5))
    for i, (ax, key) in enumerate(zip(axs.ravel(), keys)):
        ys = np.arange(len(rows))[::-1]
        allv = [tgt[i]] + [v[k][i] for k in ("driver", "small", "keep", "fast")]
        lo, hi = min(allv), max(allv)
        pad = max(hi - lo, 1e-6) * 0.25
        ax.axvline(tgt[i], color="#000000", lw=1.0, ls="--", zorder=1)
        for yi, (k, lab) in zip(ys, rows):
            val = tgt[i] if k == "target" else v[k][i]
            if k != "target":
                ax.plot([tgt[i], val], [yi, yi], color="#d0d0d0", lw=1.2, zorder=1)
            ax.scatter([val], [yi], s=62, marker="D" if k == "target" else "o", color="#000000" if k == "target" else COL[k], zorder=3, edgecolors="white", linewidths=0.7)
            ax.text(val, yi + 0.34, f"{val:.2f}", va="bottom", ha="center", fontsize=8.5)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab in rows], fontsize=8.5)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(_title(key), loc="left", fontsize=9.5)
        ax.set_xlabel(UNIT[key], fontsize=8.5)
        style_axes(ax)
    fig.suptitle(f"{name}: the 12 style numbers of each lap, as they are (dashed line = the driver's target)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / f"{name}_2_style.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_metrics(name, sn, info, lib, out):
    """Each of the 12 style numbers (as they are, in their own unit) against lap time: every on-track lap of the run (grey), the four laps (colour) and the target."""
    keys = sn.number.tolist()
    tgt = sn.target.to_numpy(float)
    vals = {"driver": sn.driver_run.to_numpy(float), "small": sn.small.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}
    ok = lib[lib.lap_time.notna() & (lib.off_share <= 1e-3) & (lib.start_m.fillna(0.0) < 1.0)]
    fig, axs = plt.subplots(4, 3, figsize=(14, 12.5))
    for i, (ax, key) in enumerate(zip(axs.ravel(), keys)):
        ax.axhline(tgt[i], color="#000000", lw=1.0, ls="--")
        ax.scatter(ok.lap_time, ok[key], s=7, color="#c8c8c8", zorder=1, linewidths=0)
        order = sorted(FOUR, key=lambda k: -info[k]["lap_time"])
        ax.plot([info[k]["lap_time"] for k in order], [vals[k][i] for k in order], color="#999999", lw=0.9, zorder=2)
        for k in FOUR:
            ax.scatter([info[k]["lap_time"]], [vals[k][i]], s=60, color=COL[k], zorder=3, edgecolors="white", linewidths=0.8)
        ax.set_title(f"{_title(key)}\ndriver's run {vals['driver'][i]:.2f}  →  style kept {vals['keep'][i]:.2f}", loc="left", fontsize=9)
        ax.set_ylabel(UNIT[key], fontsize=8)
        style_axes(ax)
        ax.tick_params(labelsize=8)
        if i >= 9:
            ax.set_xlabel("lap time, s", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=f"{NAME[k]}  {info[k]['lap_time']:.2f} s") for k in FOUR]
    handles += [plt.Line2D([], [], marker="o", ls="", color="#c8c8c8", label=f"all {len(ok)} on-track laps of the run"), plt.Line2D([], [], color="#000000", ls="--", label="the driver's target")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{name}: the 12 style numbers against lap time", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{name}_5_style_vs_lap_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_style_change(name, sn, info, out):
    """How much each style number changes from the driver's run to each RL lap, in the number's own unit; the diamond marks where the target is."""
    keys = sn.number.tolist()
    tgt = sn.target.to_numpy(float)
    v = _vals(sn)
    fig, axs = plt.subplots(4, 3, figsize=(14.5, 11.5))
    for i, (ax, key) in enumerate(zip(axs.ravel(), keys)):
        ds = {k: v[k][i] - v["driver"][i] for k in ("small", "keep", "fast")}
        tg = tgt[i] - v["driver"][i]
        allv = list(ds.values()) + [tg, 0.0]
        lo, hi = min(allv), max(allv)
        pad = max(hi - lo, 1e-6) * 0.22
        ax.axvline(0, color=COL["driver"], lw=1.4, zorder=2)
        for j, k in enumerate(("small", "keep", "fast")):
            yi = 2 - j
            ax.barh(yi, ds[k], height=0.55, color=COL[k], zorder=3)
            ax.text(ds[k] + (pad * 0.12 if ds[k] >= 0 else -pad * 0.12), yi, f"{ds[k]:+.2f}", va="center", ha="left" if ds[k] >= 0 else "right", fontsize=9)
        ax.scatter([tg], [1], s=95, marker="D", color="#000000", zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate("target", (tg, 1), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["small gain", "style kept", "fastest"], fontsize=8.5)
        ax.set_ylim(-0.6, 2.7)
        ax.set_xlim(lo - pad * 1.6, hi + pad * 1.6)
        ax.set_title(f"{_title(key)}\ndriver's run = {v['driver'][i]:.2f}", loc="left", fontsize=9.5)
        ax.set_xlabel(f"change from the driver's run [{UNIT[key]}]", fontsize=8.5)
        style_axes(ax)
    fig.suptitle(f"{name}: how much each style number changes from the driver's run (0 = unchanged; diamond = where the target is)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / f"{name}_6_style_change.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(name, sn, info, lib, out):
    """Lap time against the style distance D to the target, for every on-track lap of the run."""
    ok = lib[lib.lap_time.notna() & (lib.off_share <= 1e-3) & (lib.start_m.fillna(0.0) < 1.0)].copy()
    fig, a1 = plt.subplots(figsize=(9.5, 6.4))
    d_drv = info["driver"]["D"]
    a1.scatter(ok.lap_time, ok.D, s=10, color="#c8c8c8", linewidths=0, zorder=1)
    a1.axhline(d_drv, color=COL["driver"], lw=1.2, ls="--")
    a1.axhline(d_drv + 0.05, color="#999999", lw=0.9, ls=":")
    a1.text(ok.lap_time.max(), d_drv + 0.012, "driver's run", ha="right", va="bottom", fontsize=8, color=COL["driver"])
    a1.text(ok.lap_time.max(), d_drv + 0.062, "budget for 'style kept' (driver's D + 0.05)", ha="right", va="bottom", fontsize=8, color="#777777")
    for k in ("driver", "small", "keep", "fast"):
        a1.scatter([info[k]["lap_time"]], [info[k]["D"]], s=95, color=COL[k], zorder=4, edgecolors="white", linewidths=0.9, label=f"{NAME[k]}  {info[k]['lap_time']:.2f} s, D {info[k]['D']:.2f}")
    a1.scatter([], [], s=10, color="#c8c8c8", label=f"all {len(ok)} on-track laps of the run")
    a1.set_xlabel("lap time, s")
    a1.set_ylabel("D = style distance to the driver's target\n(the mean over the 12 numbers of |lap - target| / spread between real drivers)")
    a1.set_title(f"{name}: the style distance D against lap time", loc="left", fontsize=12)
    a1.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    style_axes(a1)
    fig.tight_layout()
    fig.savefig(out / f"{name}_7_style_distance_vs_lap_time.png", dpi=130, bbox_inches="tight")
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
        style = {"keys": sn.number.tolist(), "plain": [PLAIN[k] for k in sn.number], "unit": [UNIT[k] for k in sn.number], "target": sn.target.round(4).tolist(), "spread": sn.spread.round(4).tolist(),
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
