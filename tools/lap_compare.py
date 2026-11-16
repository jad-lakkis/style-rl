"""Compare three laps of one driver in the simulator, as the first step towards coaching:
  1. the driver's own run     = pure pursuit on the driver's mapped lap (baseline_pure_pursuit_driver.csv): "the actual run"
  2. a faster RL lap that keeps the style = the fastest finished on-track lap that starts on the line and whose style distance D is at most D(driver's run) + tol
  3. the fastest RL lap       = the fastest finished on-track lap, whatever its style
All three are laps of the same car on the same track; the RL laps come from the library of a training run (library.csv + laps/).

    python3 tools/lap_compare.py attacker /workspace/runs12/res1/attacker /workspace/share/attacker [--tol 0.05]

Writes into OUT: figures (01_time, 02_line, 03_line_in_corners, 04_driving, 05_style: PNG), tables (laps_chosen, ladder, sectors, style_numbers: CSV) and the three laps (lap_*.csv).
Sectors are made from the corner table: a corner zone = the corners that follow each other closely, from 120 m before the first brake point to 100 m after the last exit; the rest are straights.
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
from style_rl.dstyle.features import KEYS  # noqa: E402
from style_rl.style12 import StyleTargets  # noqa: E402

COL = {"driver": "#555555", "keep": "#0072B2", "fast": "#D55E00"}          # grey, blue, vermillion (Okabe-Ito: distinguishable with colour-vision differences)
COLS = ["t", "x", "y", "speed", "lateral", "steer", "throttle", "brake", "progress", "inside"]


def load_lap(path):
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        df = pd.DataFrame({k: z[k] for k in COLS})
    else:
        df = pd.read_csv(path)[COLS]
    df["progress"] = np.maximum.accumulate(df.progress.to_numpy(float))
    return df


def sectors(L):
    c = pd.read_csv(ROOT / "data" / "monza_corners.csv")
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
        segs.append((f"C{len([s for s in segs if s[1] == 'corners']) + 1} T{cs[0]}" + (f"-{cs[-1]}" if len(cs) > 1 else ""), "corners", z["a"], z["b"]))
        p = z["b"]
    if p < L - 1.0:
        segs.append((f"S{ns}", "straight", p, L))
    return segs


def shade(ax, segs, label=True):
    for name, kind, a, b in segs:
        if kind == "corners":
            ax.axvspan(a, b, color="#000000", alpha=0.06, lw=0)
            if label:
                ax.text((a + b) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#444444")


def style_axes(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("driver", choices=["attacker", "smooth"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tol", type=float, default=0.05, help="style distance allowed above the driver's run for the 'style kept' lap")
    ap.add_argument("--max-off", type=float, default=1e-3, help="largest share of the lap spent outside the track that still counts as an on-track lap")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    tg = StyleTargets()
    target = tg.drivers[a.driver]
    base = json.load(open(a.run / "baselines.json"))["pure_pursuit_driver"]
    t_drv, d_drv = float(base["lap_time"]), float(base["D"])
    lib = pd.read_csv(a.run / "library.csv")
    ok = lib[lib.lap_time.notna() & (lib.off_share <= a.max_off) & (lib.start_m.fillna(0.0) < 1.0)].copy()
    ok = ok.sort_values("lap_time")

    # the ladder: the fastest lap at each style budget
    rows = []
    for name, d in (("D <= driver's run", d_drv), ("D <= driver's run + 0.05", d_drv + 0.05), ("D <= driver's run + 0.10", d_drv + 0.10), ("D <= driver's run + 0.20", d_drv + 0.20), ("any style", 1e9)):
        g = ok[ok.D <= d]
        if len(g):
            r = g.iloc[0]
            rows.append({"style budget": name, "lap_time": r.lap_time, "gain_vs_driver_s": t_drv - r.lap_time, "D": r.D, "kind": r.kind, "step": int(r.step), "file": r.file})
    ladder = pd.DataFrame(rows)
    ladder.to_csv(out / "ladder.csv", index=False)

    fast = ok.iloc[0]
    pool = ok[(ok.lap_time < t_drv) & (ok.D <= d_drv + a.tol) & (ok.id != fast.id)]
    if len(pool):
        keep, note = pool.iloc[0], f"fastest on-track lap with D <= {d_drv + a.tol:.2f}"
    else:
        cand = ok[(ok.lap_time < t_drv) & (ok.id != fast.id)]
        keep, note = (cand.sort_values("D").iloc[0], "no other faster lap within the style budget: the faster lap with the lowest D") if len(cand) else (fast, "only one faster lap exists")
    laps = {"driver": load_lap(a.run / "baseline_pure_pursuit_driver.csv"), "keep": load_lap(a.run / keep.file), "fast": load_lap(a.run / fast.file)}
    names = {"driver": "driver's run (pure pursuit)", "keep": "RL lap, style kept", "fast": "RL fastest lap"}
    info = {"driver": {"lap_time": t_drv, "D": d_drv, "source": "baseline_pure_pursuit_driver.csv"},
            "keep": {"lap_time": float(keep.lap_time), "D": float(keep.D), "source": f"{keep.kind} step {int(keep.step)} ({keep.file})"},
            "fast": {"lap_time": float(fast.lap_time), "D": float(fast.D), "source": f"{fast.kind} step {int(fast.step)} ({fast.file})"}}
    numbers = {"driver": np.array([base["numbers"][k] for k in KEYS], float), "keep": keep[KEYS].to_numpy(float), "fast": fast[KEYS].to_numpy(float)}
    for k, df in laps.items():
        df.to_csv(out / f"lap_{k}.csv", index=False)
    pd.DataFrame([{"lap": names[k], **info[k], "gain_vs_driver_s": t_drv - info[k]["lap_time"]} for k in laps]).to_csv(out / "laps_chosen.csv", index=False)

    L = min(float(df.progress.iloc[-1]) for df in laps.values())
    g = np.arange(0.0, L, 2.0)
    at = {k: {c: np.interp(g, df.progress, df[c]) for c in ("t", "x", "y", "speed", "lateral", "steer", "throttle", "brake")} for k, df in laps.items()}
    segs = sectors(L)
    bounds = [s[2] for s in segs] + [L]

    # ---- sector times
    rows = []
    for i, (name, kind, lo, hi) in enumerate(segs):
        hi = min(hi, L)
        tt = {k: float(np.interp(hi, laps[k].progress, laps[k].t) - np.interp(lo, laps[k].progress, laps[k].t)) for k in laps}
        m = (g >= lo) & (g < hi)
        rows.append({"sector": name, "kind": kind, "from_m": round(lo), "to_m": round(hi), "time_driver_s": tt["driver"], "time_keep_s": tt["keep"], "time_fast_s": tt["fast"],
                     "keep_vs_driver_s": tt["keep"] - tt["driver"], "fast_vs_driver_s": tt["fast"] - tt["driver"],
                     "line_diff_keep_m": float(np.mean(np.abs(at["keep"]["lateral"][m] - at["driver"]["lateral"][m]))), "line_diff_fast_m": float(np.mean(np.abs(at["fast"]["lateral"][m] - at["driver"]["lateral"][m]))),
                     "mean_speed_driver": float(at["driver"]["speed"][m].mean()), "mean_speed_fast": float(at["fast"]["speed"][m].mean())})
    sec = pd.DataFrame(rows)
    sec.to_csv(out / "sectors.csv", index=False)

    # ---- style numbers
    sd = {k: (numbers[k] - target) / tg.spread for k in numbers}
    pd.DataFrame({"number": KEYS, "target": target, "spread": tg.spread, "driver_run": numbers["driver"], "style_kept": numbers["keep"], "fastest": numbers["fast"],
                  "off_driver_run": sd["driver"], "off_style_kept": sd["keep"], "off_fastest": sd["fast"]}).to_csv(out / "style_numbers.csv", index=False)

    # ---- figure 1: where the time is gained
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [3, 2]})
    for k in ("keep", "fast"):
        ax1.plot(g, at[k]["t"] - at["driver"]["t"], color=COL[k], lw=1.8, label=f"{names[k]}  {info[k]['lap_time']:.2f} s")
    ax1.axhline(0, color=COL["driver"], lw=1.4, label=f"{names['driver']}  {t_drv:.2f} s")
    shade(ax1, segs)
    ax1.set_xlim(0, L)
    ax1.set_ylabel("time behind (+) / ahead (-) of the driver's run, s")
    ax1.set_xlabel("distance along the lap, m")
    ax1.set_title(f"{a.driver}: where the time is gained or lost along the lap", loc="left", fontsize=12, pad=18)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(ax1)
    x = np.arange(len(sec))
    for j, (k, c) in enumerate((("keep", COL["keep"]), ("fast", COL["fast"]))):
        v = sec[f"{k}_vs_driver_s"].to_numpy()
        ax2.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=c)
        for xi, vi in zip(x + (j - 0.5) * 0.38, v):
            ax2.text(xi, vi + (0.01 if vi >= 0 else -0.01), f"{vi:+.2f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=7)
    ax2.axhline(0, color=COL["driver"], lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r.sector}\n{r.from_m}-{r.to_m} m" for r in sec.itertuples()], fontsize=7)
    ax2.set_ylabel("sector time vs driver's run, s")
    style_axes(ax2)
    fig.tight_layout()
    fig.savefig(out / "01_time.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: line difference along the lap
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k in ("keep", "fast"):
        ax.plot(g, at[k]["lateral"] - at["driver"]["lateral"], color=COL[k], lw=1.5, label=names[k])
    ax.axhline(0, color=COL["driver"], lw=1.4, label=names["driver"])
    shade(ax, segs)
    ax.set_ylabel("sideways position minus the driver's run, m")
    ax.set_xlabel("distance along the lap, m")
    ax.set_title(f"{a.driver}: how far the line is from the driver's line (positive = further left)", loc="left", fontsize=12, pad=18)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "02_line.png", dpi=130)
    plt.close(fig)

    # ---- figure 3: the line inside the track in each corner zone
    geo = pd.read_csv(ROOT / "data" / "tracks" / "monza_geometry.csv")
    zs = [s for s in segs if s[1] == "corners"]
    fig, axs = plt.subplots(1, len(zs), figsize=(3.6 * len(zs), 4.2), squeeze=False)
    for ax, (name, kind, lo, hi) in zip(axs[0], zs):
        m = (g >= lo) & (g <= min(hi, L))
        gm = (geo.s_geom >= lo) & (geo.s_geom <= hi)
        ax.plot(geo.s_geom[gm], geo.width_left[gm], color="#999999", lw=1.0)
        ax.plot(geo.s_geom[gm], -geo.width_right[gm], color="#999999", lw=1.0)
        for k in ("driver", "keep", "fast"):
            ax.plot(g[m], at[k]["lateral"][m], color=COL[k], lw=1.6, label=names[k])
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_xlabel("distance, m", fontsize=8)
        style_axes(ax)
    axs[0][0].set_ylabel("sideways position, m (grey = track edges)", fontsize=8)
    axs[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(f"{a.driver}: the line inside the track in each corner zone", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "03_line_in_corners.png", dpi=130)
    plt.close(fig)

    # ---- figure 4: what the car does
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, c, lab in zip(axs, ("speed", "throttle", "brake", "steer"), ("speed, m/s", "throttle (0-1)", "brake (0-1)", "steering command (-1..1)")):
        for k in ("driver", "keep", "fast"):
            ax.plot(g, at[k][c], color=COL[k], lw=1.2 if k == "driver" else 1.4, label=names[k] if c == "speed" else None)
        ax.set_ylabel(lab, fontsize=9)
        shade(ax, segs, label=(c == "speed"))
        style_axes(ax)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    axs[-1].set_xlabel("distance along the lap, m")
    fig.suptitle(f"{a.driver}: speed, pedals and steering of the three laps", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "04_driving.png", dpi=130)
    plt.close(fig)

    # ---- figure 5: the 12 style numbers
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(KEYS))[::-1]
    ax.axvline(0, color="#000000", lw=1.2)
    ax.axvspan(-0.3, 0.3, color="#000000", alpha=0.06, lw=0)
    for k, off in (("driver", 0.22), ("keep", 0.0), ("fast", -0.22)):
        ax.scatter(sd[k], y + off, color=COL[k], s=34, zorder=3, label=f"{names[k]}   D = {info[k]['D']:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels(KEYS, fontsize=8)
    ax.set_xlabel("distance from the target style, in spread units (0 = the target, shaded = within 0.3)")
    ax.set_title(f"{a.driver}: the 12 style numbers of the three laps against the target", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "05_style.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(f"== {a.driver}: driver's run {t_drv:.2f} s, D {d_drv:.2f}")
    print(ladder.round(3).to_string(index=False))
    print("style-kept lap:", note)
    print(pd.DataFrame([{"lap": names[k], **info[k], "gain_s": t_drv - info[k]["lap_time"]} for k in laps]).round(3).to_string(index=False))
    print(sec.round(3).to_string(index=False))
    print(pd.read_csv(out / "style_numbers.csv").round(2)[["number", "target", "driver_run", "style_kept", "fastest", "off_driver_run", "off_style_kept", "off_fastest"]].to_string(index=False))


if __name__ == "__main__":
    main()
