"""Three figures per driver and a replay page that compare the driver's run with the RL laps (the first step towards coaching).

Input: the folder written by tools/lap_compare.py for each driver (lap_driver/keep/fast.csv, laps_chosen.csv, style_numbers.csv), as
    DATA/attacker/..., DATA/smooth/...

    python tools/lap_report.py DATA OUT

Writes (per driver):
    <driver>_1_lap_time.png   lap times and the time gained in each sector
    <driver>_2_style.png      the 12 style numbers of the three laps, as they are, each in its own unit
    <driver>_3_driving.png    speed, pedals and steering along the lap
and for both drivers together: replay.html (3D chase camera, sector view, tables), summary.csv, <driver>_sectors.csv, <driver>_style_numbers.csv.
The three laps: the driver's run, the RL lap that keeps the style (the fastest RL lap within the driver's style budget), the fastest RL lap.
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

COL = {"driver": "#555555", "keep": "#0072B2", "fast": "#D55E00"}    # Okabe-Ito: grey, blue, vermillion
NAME = {"driver": "driver's run", "keep": "RL lap, style kept", "fast": "RL fastest lap"}
LAPS = ("driver", "keep", "fast")
PLAIN = {"B_mean_active": "Brake pressure", "B_event_peak_mean": "Peak brake pressure", "B_apply_rate": "Brake attack (how fast it is applied)", "B_release_rate": "Brake release",
         "T_apply_rate": "Throttle attack", "T_release_rate": "Throttle lift", "S_reversals_5deg_per_km": "Steering corrections", "S_mean_abs_rate": "Hand speed",
         "S_corner_peak_angle": "Peak steering in corners", "L_border_mean_m": "Distance to the track edge", "L_inside_dist_apex_m": "Apex distance", "L_outside_dist_exit_m": "Exit width"}
UNIT = {"B_mean_active": "pedal, 0-1", "B_event_peak_mean": "pedal, 0-1", "B_apply_rate": "pedal per s", "B_release_rate": "pedal per s", "T_apply_rate": "pedal per s",
        "T_release_rate": "pedal per s", "S_reversals_5deg_per_km": "per km", "S_mean_abs_rate": "deg per s", "S_corner_peak_angle": "deg", "L_border_mean_m": "m",
        "L_inside_dist_apex_m": "m", "L_outside_dist_exit_m": "m"}


def load(d):
    """The three laps (time series), their lap times and source, and the style numbers table of one driver."""
    d = Path(d)
    laps = {}
    for k in LAPS:
        df = pd.read_csv(d / f"lap_{k}.csv")[COLS].copy()
        df["progress"] = np.maximum.accumulate(df.progress.to_numpy(float))
        laps[k] = df
    ch = pd.read_csv(d / "laps_chosen.csv")
    info = {k: {"lap_time": float(ch.lap_time[i]), "D": float(ch.D[i]), "source": str(ch.source[i])} for i, k in enumerate(LAPS)}
    sn = pd.read_csv(d / "style_numbers.csv")
    return laps, info, sn


def sector_table(laps, segs):
    L = min(float(df.progress.iloc[-1]) for df in laps.values())
    rows = []
    for (sec, kind, lo, hi) in segs:
        hi = min(hi, L)
        tt = {k: float(np.interp(hi, laps[k].progress, laps[k].t) - np.interp(lo, laps[k].progress, laps[k].t)) for k in laps}
        rows.append({"sector": sec, "kind": kind, "from_m": round(lo), "to_m": round(hi), **{f"time_{k}": v for k, v in tt.items()}, **{f"{k}_vs_driver": tt[k] - tt["driver"] for k in ("keep", "fast")}})
    return pd.DataFrame(rows)


def fig_time(name, info, sec, out):
    """Lap times (left) and the time gained in each sector against the driver's run (right)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.9]})
    y = np.arange(len(LAPS))[::-1]
    t0 = info["driver"]["lap_time"]
    for yi, k in zip(y, LAPS):
        t = info[k]["lap_time"]
        ax.plot([t, t0], [yi, yi], color="#cccccc", lw=1.2, zorder=1)
        ax.scatter([t], [yi], color=COL[k], s=90, zorder=3)
        ax.text(t0 + 0.6, yi, f"{t:.2f} s", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[k] for k in LAPS], fontsize=10)
    ax.set_ylim(-0.6, len(LAPS) - 0.4)
    ax.set_xlim(min(info[k]["lap_time"] for k in LAPS) - 0.8, t0 + 3.2)
    ax.set_xlabel("lap time, s")
    ax.set_title("Lap time", loc="left", fontsize=11)
    style_axes(ax)
    x = np.arange(len(sec))
    for j, k in enumerate(("keep", "fast")):
        v = sec[f"{k}_vs_driver"].to_numpy()
        bx.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=COL[k], label=NAME[k])
        if k == "keep":
            for xi, vi in zip(x + (j - 0.5) * 0.38, v):
                bx.text(xi, vi + (0.03 if vi >= 0 else -0.03), f"{vi:+.2f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=8)
    bx.axhline(0, color=COL["driver"], lw=1.2)
    bx.set_xticks(x)
    bx.set_xticklabels([r.sector.replace(" ", "\n") for r in sec.itertuples()], fontsize=9)
    bx.set_ylabel("sector time minus the driver's run, s\n(below 0 = faster)", fontsize=9)
    bx.set_title("Where the time is gained (C = corner zones, S = straights)", loc="left", fontsize=11)
    bx.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(bx)
    fig.tight_layout()
    fig.savefig(out / f"{name}_1_lap_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_style(name, sn, out):
    """The 12 style numbers of the three laps as they are, one small plot per number."""
    keys = sn.number.tolist()
    v = {"driver": sn.driver_run.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}
    fig, axs = plt.subplots(4, 3, figsize=(13, 9.5))
    ys = np.arange(len(LAPS))[::-1]
    for i, (ax, key) in enumerate(zip(axs.ravel(), keys)):
        vals = [v[k][i] for k in LAPS]
        lo, hi = min(vals), max(vals)
        pad = max(hi - lo, 1e-6) * 0.3
        for yi, k in zip(ys, LAPS):
            ax.scatter([v[k][i]], [yi], s=70, color=COL[k], zorder=3, edgecolors="white", linewidths=0.7)
            ax.text(v[k][i], yi + 0.32, f"{v[k][i]:.2f}", va="bottom", ha="center", fontsize=9)
        ax.set_yticks(ys)
        ax.set_yticklabels(["driver's run", "style kept", "fastest"] if i % 3 == 0 else [""] * 3, fontsize=9)
        ax.set_ylim(-0.6, len(LAPS) - 0.3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{PLAIN[key]} [{UNIT[key]}]", loc="left", fontsize=10)
        ax.tick_params(labelsize=8)
        style_axes(ax)
    fig.suptitle(f"{name}: the 12 style numbers of each lap", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"{name}_2_style.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_driving(name, at, g, segs, out):
    fig, axs = plt.subplots(4, 1, figsize=(12.5, 9), sharex=True)
    for ax, c, lab in zip(axs, ("speed", "throttle", "brake", "steer"), ("speed, m/s", "throttle (0-1)", "brake (0-1)", "steering (-1..1)")):
        for k in LAPS:
            ax.plot(g, at[k][c], color=COL[k], lw=1.1 if k == "driver" else 1.4, label=NAME[k] if c == "speed" else None)
        ax.set_ylabel(lab, fontsize=9)
        shade(ax, segs, label=(c == "speed"))
        style_axes(ax)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    axs[-1].set_xlabel("distance along the lap, m   (shaded = corner zones)")
    fig.suptitle(f"{name}: what the car does along the lap", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_3_driving.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


REPLAY = (Path(__file__).resolve().parent / "replay_template.html").read_text(encoding="utf-8")


def replay(all_laps, all_info, all_sn, geo, out):
    data = {}
    dt = 0.1
    track = {"s": np.round(geo.s_geom.to_numpy(), 1).tolist(), "xl": np.round(geo.x_left.to_numpy(), 1).tolist(), "yl": np.round(geo.y_left.to_numpy(), 1).tolist(),
             "xr": np.round(geo.x_right.to_numpy(), 1).tolist(), "yr": np.round(geo.y_right.to_numpy(), 1).tolist()}       # full resolution (1.55 m): the chase camera is close to the road
    for name, laps in all_laps.items():
        info, sn = all_info[name], all_sn[name]
        L = float(min(df.progress.iloc[-1] for df in laps.values()))
        segs = sectors(L)
        style = {"keys": sn.number.tolist(), "plain": [PLAIN[k] for k in sn.number], "unit": [UNIT[k] for k in sn.number],
                 "vals": {"driver": sn.driver_run.round(4).tolist(), "keep": sn.style_kept.round(4).tolist(), "fast": sn.fastest.round(4).tolist()}}
        vmax = float(np.ceil(max(df.speed.max() for df in laps.values()) / 10.0) * 10.0)
        out_laps = {}
        for k, df in laps.items():
            tt = np.arange(0.0, float(df.t.iloc[-1]) + dt, dt)
            f = lambda c: np.interp(tt, df.t, df[c])      # noqa: E731
            out_laps[k] = {"x": np.round(f("x"), 1).tolist(), "y": np.round(f("y"), 1).tolist(), "v": np.round(f("speed"), 1).tolist(), "p": np.round(f("progress"), 1).tolist(),
                           "a": np.round(f("throttle"), 2).tolist(), "b": np.round(f("brake"), 2).tolist(), "lap_time": info[k]["lap_time"]}
        zones = []
        for nm, kind, lo, hi in segs:
            if kind == "corners":
                j = int(np.argmin(np.abs(geo.s_geom - (lo + hi) / 2)))
                zones.append({"name": nm, "a": lo, "b": hi, "x": float(geo.x_center[j]), "y": float(geo.y_center[j])})
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
    all_laps, all_info, all_sn, summary = {}, {}, {}, []
    for name in ("attacker", "smooth"):
        if not (a.data / name).is_dir():
            continue
        laps, info, sn = load(a.data / name)
        L = min(float(df.progress.iloc[-1]) for df in laps.values())
        g = np.arange(0.0, L, 2.0)
        at = {k: {c: np.interp(g, df.progress, df[c]) for c in ("t", "speed", "steer", "throttle", "brake")} for k, df in laps.items()}
        segs = sectors(L)
        sec = sector_table(laps, segs)
        fig_time(name, info, sec, a.out)
        fig_style(name, sn, a.out)
        fig_driving(name, at, g, segs, a.out)
        sec.to_csv(a.out / f"{name}_sectors.csv", index=False)
        sn[["number", "driver_run", "style_kept", "fastest"]].to_csv(a.out / f"{name}_style_numbers.csv", index=False)
        all_laps[name], all_info[name], all_sn[name] = laps, info, sn
        for k in LAPS:
            summary.append({"driver": name, "lap": NAME[k], "lap_time_s": info[k]["lap_time"], "gain_vs_driver_run_s": info["driver"]["lap_time"] - info[k]["lap_time"], "source": info[k]["source"]})
    pd.DataFrame(summary).round(3).to_csv(a.out / "summary.csv", index=False)
    replay(all_laps, all_info, all_sn, geo, a.out)
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
