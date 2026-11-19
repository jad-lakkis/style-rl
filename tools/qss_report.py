"""QSS (quasi-steady-state) lap of the OCD car on Monza: the highest speed the car's measured limits allow at every metre of the track, and the lap time that gives.

    python tools/qss_report.py OUT_DIR [--run RUN_DIR --step 950000 --lap 2] [--driver attacker=CSV --driver smooth=CSV]

The car: data/ocd_envelope.json, measured in the OCD simulator (scripts/measure_ocd_envelope.py): lateral g, braking g and full-throttle acceleration against speed, drag fitted as kd v^2.
The method (style_rl/drive/course.py, speed_profile): point mass on a line, corner speed limit from the lateral g, then a backward pass (brake in time) and a forward pass (accelerate out),
with the friction ellipse between longitudinal and lateral use. Lap time = sum of ds / v.
Writes OUT_DIR/qss_lines.csv (QSS lap time on several lines, and on the lines the given laps drove, with the lap time they actually took), qss_corners.csv, fig_5_qss_limits.png.
"""
import argparse
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
from drive_style_compare import REAL_NAMES, segments  # noqa: E402
from style_rl.drive.course import Course, GridTrack, V_GRID, G, load_envelope, min_curvature_line, path_geometry, speed_profile  # noqa: E402

INK, MUTED, BLUE, VERM = "#222222", "#6E6E6E", "#0072B2", "#D55E00"


def qss(path, env):
    kappa, seg = path_geometry(path)
    v, T = speed_profile(kappa, seg, env)
    return v, float(T), seg


def closed_resample(xy, ds=2.0):
    p = np.vstack([xy, xy[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    n = int(round(s[-1] / ds))
    si = np.arange(n) * (s[-1] / n)
    return np.column_stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=INK, labelsize=9.5)
    ax.grid(color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--step", type=int, default=950000)
    ap.add_argument("--lap", type=int, default=2)
    ap.add_argument("--driver", action="append", default=[])
    ap.add_argument("--track", type=Path, default=ROOT / "data/tracks/monza_geometry.csv")
    ap.add_argument("--envelope", type=Path, default=ROOT / "data/ocd_envelope.json")
    ap.add_argument("--corners", type=Path, default=ROOT / "data/monza_corners.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env = load_envelope(a.envelope)
    gt = GridTrack(a.track)
    L = gt.length_m

    rows = []
    lines = {"centre line": gt.center_grid}
    for m in (3.0, 2.0, 1.5, 1.0, 0.9):
        n, nl = min_curvature_line(gt, m)
        lines[f"minimum-curvature line, {m:g} m from the edges" + ("  (the line the RL was trained on)" if m == 1.5 else "")] = gt.center_grid + nl * n[:, None]
    for name, p in lines.items():
        v, T, seg = qss(p, env)
        rows.append({"line": name, "qss_lap_s": T, "top_kmh": 3.6 * v.max(), "slowest_kmh": 3.6 * v.min(), "mean_kmh": 3.6 * seg.sum() / T, "actual_lap_s": np.nan})
    driven = {}
    if a.run is not None:
        z = np.load(a.run / "laps" / f"eval_step_{a.step}.npz")
        ss, tt, lap_time, idx = lap_from_trace(z, L, a.lap - 1)
        driven[f"RL best lap (step {a.step // 1000}k): the line it drove"] = (np.column_stack([z["x"][idx], z["y"][idx]]), float(lap_time))
    for spec in a.driver:
        name, path = spec.split("=", 1)
        d = pd.read_csv(path)
        driven[f"{name}'s pure-pursuit run: the line it drove"] = (np.column_stack([d.x, d.y]), float(d.t.iloc[-1]))
    for name, (xy, actual) in driven.items():
        v, T, seg = qss(closed_resample(xy), env)
        rows.append({"line": name, "qss_lap_s": T, "top_kmh": 3.6 * v.max(), "slowest_kmh": 3.6 * v.min(), "mean_kmh": 3.6 * seg.sum() / T, "actual_lap_s": actual})
    tab = pd.DataFrame(rows)
    tab["actual_minus_qss_s"] = tab.actual_lap_s - tab.qss_lap_s
    tab["actual_over_qss_percent"] = 100 * (tab.actual_lap_s / tab.qss_lap_s - 1)
    tab.round(2).to_csv(a.out / "qss_lines.csv", index=False)

    # speed limit along the lap on two lines (1 m stations) and the limit speed in every corner
    c15, c09 = Course(a.track, a.envelope, margin_m=1.5), Course(a.track, a.envelope, margin_m=0.9)
    st = np.arange(c15.N) * c15.ds
    segs = segments(L, a.corners)
    crow = []
    for name, kind, lo, hi in segs:
        m = (st >= lo) & (st < min(hi, L))
        crow.append({"segment": name, "kind": kind, "from_m": round(lo), "to_m": round(min(hi, L)), "qss_min_kmh_training_line": 3.6 * c15.v_qss[m].min(), "qss_min_kmh_tight_line": 3.6 * c09.v_qss[m].min(),
                     "qss_max_kmh_training_line": 3.6 * c15.v_qss[m].max()})
    cor = pd.DataFrame(crow)
    cor.round(1).to_csv(a.out / "qss_corners.csv", index=False)

    fig = plt.figure(figsize=(14, 10.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 1.5], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    for name, kind, lo, hi in segs:
        if kind == "corner":
            ax.axvspan(lo, hi, color="#000000", alpha=0.055, lw=0)
            ax.text((lo + hi) / 2, 1.01, f"{name}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.plot(st, 3.6 * c09.v_qss, color="#999999", lw=1.4, ls=":", label=f"limit on a tight line (0.9 m from the edges): {c09.qss_lap_time_s:.2f} s")
    ax.plot(st, 3.6 * c15.v_qss, color=INK, lw=1.8, ls="--", label=f"limit on the training line (1.5 m from the edges): {c15.qss_lap_time_s:.2f} s")
    if a.run is not None:
        ax.plot(z["station_m"][idx], 3.6 * z["speed"][idx], color=BLUE, lw=1.6, label=f"RL best lap, what it actually drove: {lap_time:.2f} s")
    ax.set_xlim(0, L)
    ax.set_ylabel("speed, km/h", fontsize=11)
    ax.set_xlabel("distance along the lap, m   (shaded = corners)", fontsize=10.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=3, frameon=False, fontsize=9.5)
    ax.set_title("QSS: the highest speed the car's limits allow at every metre of Monza", loc="left", fontsize=12.5, color=INK, pad=44)
    style(ax)
    kmh = V_GRID * 3.6
    for j, (arr, lab, col) in enumerate(((env.lat_g, "cornering grip, g", BLUE), (env.brk_g, "braking, g", VERM), (env.acc_ms2, "full-throttle acceleration, m/s²", "#009E73"))):
        b = fig.add_subplot(gs[1, j])
        b.plot(kmh[:71], np.asarray(arr)[:71], color=col, lw=2.2)
        b.set_xlabel("speed, km/h", fontsize=10)
        b.set_title(lab, loc="left", fontsize=11, color=INK)
        b.set_xlim(0, 250)
        style(b)
    fig.text(0.01, 0.005, "The car: Dallara F317 in OCD, 580 kg, measured in the simulator (cornering with constant steering, hardest braking, full throttle, coasting drag). Point-mass QSS: an estimate, a lap can beat it slightly in places.",
             fontsize=8.5, color=MUTED)
    fig.savefig(a.out / "fig_5_qss_limits.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string(index=False))
    print(cor.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
