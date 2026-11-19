"""Compare a lap of a drive run (pure RL, no style constraint) with the pure-pursuit runs of the drivers: the 12 style numbers and the time in every corner and straight.

    python tools/drive_style_compare.py RUN_DIR OUT_DIR --step 950000 --lap 2 \
        --driver attacker=PATH/baseline_pure_pursuit_driver.csv --driver smooth=PATH/baseline_pure_pursuit_driver.csv

RUN_DIR: a run folder with config.json and laps/eval_step_N.npz (scripts/drive_train.py). The driver CSVs are the old library format (t, x, y, speed, lateral, steer, throttle, brake, progress).
The 12 numbers are computed by the same code for all laps (style_rl/dstyle/features.style_vector); for the driver runs this reproduces the numbers stored in baselines.json exactly.
Steering: the project standard is the steering COMMAND (steering wheel degrees = road-wheel angle x 13.4). An RL policy that chatters has a busy command that the car's steering motor
partly filters, so the three steering numbers are also given for an ESTIMATE of the real wheel angle (the command through a first-order lag of 0.06 s with a rate limit of 0.5 rad/s;
this is a model of the motor, not a measurement: re-run the lap in OCD for the measured angle).
D = mean of the 12 |differences|, each in units of its spread (targets/style12.json): 1 = as far apart as two typical real driver-track cells.

Writes into OUT_DIR: style_numbers.csv, style_distance.csv, segment_times.csv, fig_3_style_numbers.png, fig_4_lap_vs_drivers.png
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
sys.path.insert(0, str(ROOT / "tools"))
from drive_run_report import lap_from_trace  # noqa: E402
from style_rl.dstyle.features import KEYS, STYLE, style_vector  # noqa: E402
from style_rl.dstyle.track import Track  # noqa: E402
from style_rl.style12 import StyleTargets  # noqa: E402

WHEEL_DEG_PER_RAD = 57.2958 * 13.4          # road-wheel angle (rad) -> steering wheel degrees (steering ratio 13.4 of the driver model)
OLD_UNIT_RAD = 0.15                         # old driving environment: steering command 1.0 = 0.15 rad at the road wheels
STEER_KEYS = ("S_reversals_5deg_per_km", "S_mean_abs_rate", "S_corner_peak_angle")
INK, MUTED = "#222222", "#6E6E6E"
COL = {"RL": "#0072B2", "attacker": "#D55E00", "smooth": "#009E73"}          # Okabe-Ito blue, vermillion, bluish green
REAL_NAMES = {"T1": "Rettifilo", "T2": "Rettifilo", "T3": "Roggia", "T4": "Roggia", "T5": "Lesmo 1", "T6": "Lesmo 2", "T7": "Ascari", "T8": "Ascari", "T9": "Ascari", "T10": "Parabolica"}
PLAIN = {s[0]: (s[1], s[2], s[4]) for s in STYLE}                         # key -> (plain name, group, unit)


def actuator(cmd_rad, dt=0.04, lag=0.06, rate=0.5, sub=4):
    """Model of the steering motor: first-order lag with a rate limit on the commanded road-wheel angle (estimate)."""
    d, out = float(cmd_rad[0]), np.empty(len(cmd_rad))
    for i, c in enumerate(cmd_rad):
        for _ in range(sub):
            d += float(np.clip((c - d) / lag, -rate, rate)) * dt / sub
        out[i] = d
    return out


def numbers(time_s, throttle, brake, steer_rad, x, y, lap_dist, track, corners):
    df = pd.DataFrame({"time_s": time_s, "throttle": throttle, "brake": brake, "steer_deg": np.asarray(steer_rad) * WHEEL_DEG_PER_RAD, "x": x, "y": y})
    m = style_vector(df, track, lap_dist=lap_dist, corners=corners)
    return np.array([m[k] for k in KEYS], float)


def segments(L, corners_csv):
    """Straights and one segment per corner: corner zones run from 120 m before the corner start to 100 m after its end (tools/lap_compare.py), neighbours split half way between them."""
    cc = pd.read_csv(corners_csv)
    zones = []
    for q in cc.itertuples():
        a, b = q.s_start - 120.0, q.s_end + 100.0
        if zones and a <= zones[-1]["b"] + 60.0:
            zones[-1]["b"] = max(zones[-1]["b"], b)
            zones[-1]["c"].append(q)
        else:
            zones.append({"a": a, "b": b, "c": [q]})
    segs, p, ns = [], 0.0, 0
    for z in zones:
        if z["a"] > p + 1.0:
            segs.append((f"S{ns}", "straight", p, z["a"]))
            ns += 1
        edges = [z["a"]] + [0.5 * (z["c"][i].s_end + z["c"][i + 1].s_start) for i in range(len(z["c"]) - 1)] + [z["b"]]
        for i, q in enumerate(z["c"]):
            segs.append((f"T{int(q.corner)}", "corner", edges[i], edges[i + 1]))
        p = z["b"]
    if p < L - 1.0:
        segs.append((f"S{ns}", "straight", p, L))
    return segs


def rl_lap(run, step, lap, track, corners, cfg):
    L = track.length_m
    z = np.load(run / "laps" / f"eval_step_{step}.npz")
    ss, tt, lap_time, idx = lap_from_trace(z, L, lap - 1)
    cmd = z["steer"][idx] * float(cfg["env"]["steer_lock_rad"])
    args = (z["time_s"][idx], z["throttle"][idx], z["brake"][idx])
    xy = np.column_stack([z["x"][idx], z["y"][idx]])
    nm = {"command": numbers(*args, cmd, xy[:, 0], xy[:, 1], z["station_m"][idx], track, corners),
          "estimate": numbers(*args, actuator(cmd), xy[:, 0], xy[:, 1], z["station_m"][idx], track, corners)}
    q = track.project(xy, s_hint=z["station_m"][idx])
    v = {"progress": np.concatenate([[0.0], np.maximum.accumulate(q["progress_m"]), [L]]), "t": tt, "speed": np.concatenate([[z["speed"][idx][0]], z["speed"][idx], [z["speed"][idx][-1]]]),
         "lap": float(lap_time)}
    return nm, v


def driver_lap(path, track, corners):
    L = track.length_m
    d = pd.read_csv(path)
    cmd = d.steer.to_numpy(float) * OLD_UNIT_RAD
    args = (d.t.to_numpy(float), d.throttle.to_numpy(float), d.brake.to_numpy(float))
    nm = {"command": numbers(*args, cmd, d.x.to_numpy(float), d.y.to_numpy(float), d.progress.to_numpy(float), track, corners),
          "estimate": numbers(*args, actuator(cmd), d.x.to_numpy(float), d.y.to_numpy(float), d.progress.to_numpy(float), track, corners)}
    v = {"progress": np.maximum.accumulate(np.concatenate([[0.0], d.progress.to_numpy(float), [L]])), "t": np.concatenate([[0.0], d.t.to_numpy(float), [d.t.iloc[-1]]]),
         "speed": np.concatenate([[d.speed.iloc[0]], d.speed.to_numpy(float), [d.speed.iloc[-1]]]), "lap": float(d.t.iloc[-1])}
    return nm, v


def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=INK, labelsize=9.5)
    ax.grid(axis=grid, color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)


def fig_numbers(tab, names, out, clip=8.0):
    """One row per style number: the two driver runs and the RL lap, in spread units from the middle of the two driver runs."""
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    rows = list(range(len(KEYS)))
    y, last, ys = 0.0, None, {}
    for i in reversed(rows):
        g = STYLE[i][2]
        if last is not None and g != last:
            y += 1.0
        ys[i], last, y = y, g, y + 1.0
    top = {}
    for i in rows:
        top[STYLE[i][2]] = max(top.get(STYLE[i][2], -1e9), ys[i])
    for g, v in top.items():
        ax.text(-clip - 0.4, v + 0.95, g.upper(), ha="left", va="center", fontsize=10, color=MUTED, fontweight="bold")
    for i, k in enumerate(KEYS):
        r = tab.iloc[i]
        mid = 0.5 * (r["attacker_run"] + r["smooth_run"])
        xs = {"attacker": (r["attacker_run"] - mid) / r["spread"], "smooth": (r["smooth_run"] - mid) / r["spread"], "RL": (r["rl_command"] - mid) / r["spread"]}
        ax.axhline(ys[i], color="#EEEEEE", lw=0.8, zorder=0)
        for n in ("attacker", "smooth"):
            ax.plot(xs[n], ys[i], marker="o", ms=9, color=COL[n], mec="white", mew=1.0, ls="none", zorder=3)
        v = xs["RL"]
        e = (r["rl_estimate"] - mid) / r["spread"] if k in STEER_KEYS else None
        if abs(v) > clip:
            ax.plot(np.sign(v) * clip, ys[i], marker=">" if v > 0 else "<", ms=11, color=COL["RL"], ls="none", zorder=4)
            ax.text(clip, ys[i] + 0.22, f"RL {v:+.1f} off the chart" + (f"; estimate {e:+.1f}" if e is not None else ""), ha="right", va="bottom", fontsize=9, color=COL["RL"])
        else:
            ax.plot(v, ys[i], marker="D", ms=9, color=COL["RL"], mec="white", mew=1.0, ls="none", zorder=4)
        if e is not None and abs(e) <= clip:
            ax.plot(e, ys[i], marker="D", ms=8, mfc="white", mec=COL["RL"], mew=1.6, ls="none", zorder=4)
        ax.text(clip + 0.6, ys[i], f"{r['rl_command']:8.2f} | {r['attacker_run']:7.2f} | {r['smooth_run']:7.2f}   {r['unit']}", ha="left", va="center", fontsize=9, color=INK, family="monospace")
    ax.text(clip + 0.6, max(ys.values()) + 0.95, f"{'RL lap':>8} | {'attacker':>7} | {'smooth':>7}", ha="left", va="center", fontsize=9, color=MUTED, family="monospace")
    ax.axvline(0, color="#999999", lw=0.9)
    ax.set_yticks([ys[i] for i in rows])
    ax.set_yticklabels([PLAIN[k][0] for k in KEYS], fontsize=10.5)
    ax.set_ylim(-0.7, max(ys.values()) + 1.5)
    ax.set_xlim(-clip - 0.5, clip + 0.5)
    ax.set_xlabel("difference from the middle of the two driver runs, in spread units  (1 = as far apart as two typical real driver-track cells)", fontsize=10.5)
    ax.plot([], [], "o", color=COL["attacker"], ms=9, label="attacker's run (pure pursuit)")
    ax.plot([], [], "o", color=COL["smooth"], ms=9, label="smooth driver's run (pure pursuit)")
    ax.plot([], [], "D", color=COL["RL"], ms=9, label=names["RL"])
    ax.plot([], [], "D", mfc="white", mec=COL["RL"], mew=1.6, ms=8, label="RL lap, steering estimated after the car's steering motor")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=9.5)
    style(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_lap(segs, laps, seg_tab, L, names, out):
    fig = plt.figure(figsize=(14, 12.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 2.2, 2.6], hspace=0.34)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None), fig.add_subplot(gs[2])
    grid = np.arange(0.0, L, 2.0)
    for ax in (ax1, ax2):
        for name, kind, a, b in segs:
            if kind == "corner":
                ax.axvspan(a, b, color="#000000", alpha=0.055, lw=0)
        ax.set_xlim(0, L)
        style(ax)
    for n in ("attacker", "smooth", "RL"):
        ax1.plot(grid, np.interp(grid, laps[n]["progress"], laps[n]["speed"]) * 3.6, color=COL[n], lw=2.0 if n == "RL" else 1.5, label=f"{names[n]}  {laps[n]['lap']:.2f} s", zorder=3 if n == "RL" else 2)
    ax1.set_ylabel("speed, km/h", fontsize=11)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.085), ncol=3, frameon=False, fontsize=10.5)
    for name, kind, a, b in segs:
        if kind == "corner":
            ax1.text((a + b) / 2, 1.01, name, transform=ax1.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax1.set_title("Speed along the lap (shaded = corner segments)", loc="left", fontsize=11, color=INK, pad=56)
    for n in ("attacker", "smooth"):
        gap = np.interp(grid, laps["RL"]["progress"], laps["RL"]["t"]) - np.interp(grid, laps[n]["progress"], laps[n]["t"])
        ax2.plot(grid, gap, color=COL[n], lw=2.0, label=f"RL minus the {n}'s run")
        ax2.text(L - 15, gap[-1] + (0.18 if n == "attacker" else -0.18), f"{gap[-1]:+.2f} s", ha="right", va="bottom" if n == "attacker" else "top", fontsize=9.5, color=COL[n])
    ax2.axhline(0, color="#999999", lw=0.9)
    ax2.set_ylim(ax2.get_ylim()[0] - 0.5, ax2.get_ylim()[1] + 0.2)
    ax2.set_ylabel("time gap so far, s\n(below 0 = RL ahead)", fontsize=11)
    ax2.set_xlabel("distance along the lap, m", fontsize=10.5)
    ax2.legend(loc="lower left", frameon=False, fontsize=10)
    ax2.set_title("Where the RL lap gains and loses time", loc="left", fontsize=11, color=INK, pad=8)
    x = np.arange(len(seg_tab))
    for j, n in enumerate(("attacker", "smooth")):
        v = seg_tab[f"RL_vs_{n}"].to_numpy()
        ax3.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=COL[n], label=f"RL minus the {n}'s run", zorder=3)
        for xi, vi in zip(x + (j - 0.5) * 0.38, v):
            ax3.text(xi, vi + (0.04 if vi >= 0 else -0.04), f"{vi:+.2f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=7, color=INK)
    ax3.axhline(0, color="#999999", lw=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{r.segment}\n{REAL_NAMES.get(r.segment, 'straight')}\n{r.from_m}" for r in seg_tab.itertuples()], fontsize=8.5)
    ax3.set_xlabel("segment, and the distance in m along the lap where it starts (full ranges in segment_times.csv)", fontsize=10)
    ax3.set_ylabel("segment time, s\n(below 0 = RL faster)", fontsize=11)
    ax3.legend(loc="lower right", frameon=False, fontsize=10)
    ax3.set_ylim(min(seg_tab[["RL_vs_attacker", "RL_vs_smooth"]].min().min() - 0.35, -0.5), max(seg_tab[["RL_vs_attacker", "RL_vs_smooth"]].max().max() + 0.4, 0.5))
    style(ax3)
    ax3.set_title("Time per segment against each driver's run", loc="left", fontsize=11, color=INK, pad=8)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--lap", type=int, default=2)
    ap.add_argument("--driver", action="append", required=True, help="name=path of a driver run csv (old library format); give it for each driver")
    ap.add_argument("--track", type=Path, default=ROOT / "data/tracks/monza_geometry.csv")
    ap.add_argument("--corners", type=Path, default=ROOT / "data/monza_corners.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(a.run / "config.json"))
    track = Track(a.track)
    corners = track.find_corners()
    L = track.length_m
    tg = StyleTargets()
    drivers = dict(d.split("=", 1) for d in a.driver)
    rl_nm, rl_v = rl_lap(a.run, a.step, a.lap, track, corners, cfg)
    laps, nm = {"RL": rl_v}, {"RL": rl_nm}
    for n, p in drivers.items():
        nm[n], laps[n] = driver_lap(p, track, corners)
    names = {"RL": f"RL lap (step {a.step // 1000}k, lap {a.lap})", **{n: f"{n}'s run (pure pursuit)" for n in drivers}}

    tab = pd.DataFrame({"number": KEYS, "plain_name": [PLAIN[k][0] for k in KEYS], "group": [PLAIN[k][1] for k in KEYS], "unit": [PLAIN[k][2] for k in KEYS], "spread": tg.spread})
    for n in drivers:
        tab[f"{n}_target"] = tg.drivers[n]
        tab[f"{n}_run"] = nm[n]["command"]
    tab["rl_command"], tab["rl_estimate"] = rl_nm["command"], rl_nm["estimate"]
    for n in drivers:
        tab[f"rl_minus_{n}_run_in_spreads"] = (tab.rl_command - tab[f"{n}_run"]) / tab.spread
    tab.to_csv(a.out / "style_numbers.csv", index=False)

    keep = [i for i, k in enumerate(KEYS) if k not in STEER_KEYS]
    rows = []
    for n in drivers:
        for basis in ("command", "estimate"):
            rl = rl_nm[basis]
            rows.append({"driver": n, "steering_basis": basis if basis == "command" else "estimated after the steering motor", "D_driver_run_vs_target": tg.distance(nm[n][basis], tg.drivers[n]),
                         "D_RL_lap_vs_target": tg.distance(rl, tg.drivers[n]), "D_RL_lap_vs_driver_run": tg.distance(rl, nm[n][basis]),
                         "D_RL_vs_run_9_numbers_without_steering": float(np.mean(np.abs(rl[keep] - nm[n][basis][keep]) / tg.spread[keep])), "style_budget": tg.distance(nm[n]["command"], tg.drivers[n]) + 0.05})
    dist = pd.DataFrame(rows)
    dist.to_csv(a.out / "style_distance.csv", index=False)

    segs = segments(L, a.corners)
    out_rows = []
    for name, kind, lo, hi in segs:
        row = {"segment": name, "kind": kind, "from_m": round(lo), "to_m": round(min(hi, L))}
        for k, v in laps.items():
            row[f"time_{k}"] = float(np.interp(min(hi, L), v["progress"], v["t"]) - np.interp(max(lo, 0.0), v["progress"], v["t"]))
            m = (v["progress"] >= lo) & (v["progress"] < hi)
            row[f"mean_kmh_{k}"], row[f"min_kmh_{k}"] = 3.6 * float(v["speed"][m].mean()), 3.6 * float(v["speed"][m].min())
        for n in drivers:
            row[f"RL_vs_{n}"] = row["time_RL"] - row[f"time_{n}"]
        out_rows.append(row)
    seg = pd.DataFrame(out_rows)
    seg.to_csv(a.out / "segment_times.csv", index=False)
    fig_numbers(tab, names, a.out / "fig_3_style_numbers.png")
    fig_lap(segs, laps, seg, L, names, a.out / "fig_4_lap_vs_drivers.png")
    pd.set_option("display.width", 250)
    print(dist.round(2).to_string(index=False))
    print(seg[["segment", "kind", "from_m", "to_m"] + [f"time_{k}" for k in laps] + [f"RL_vs_{n}" for n in drivers]].round(2).to_string(index=False))
    for kind in ("straight", "corner"):
        g = seg[seg.kind == kind]
        print(kind, {n: round(float(g[f"RL_vs_{n}"].sum()), 2) for n in drivers})


if __name__ == "__main__":
    main()
