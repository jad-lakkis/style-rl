"""Compare drive runs (scripts/drive_train.py): lap time against training steps, how busy the hands are against training steps, and the 12 style numbers of the chosen laps against the range
of the four real drivers.

    python tools/smooth_sweep_report.py OUT_DIR --run a2=PATH/a2 --run p0=PATH/p0 --run p1_w01=PATH/p1_w01 ...

PATH/<run> is a run folder (config.json, evals.jsonl, laps/eval_step_N.npz). Writes into OUT_DIR: evals_table.csv, chosen_laps.csv (the 12 numbers of the best clean lap and the best gated lap of
every run), fig_1_lap_time.png, fig_2_hands.png, fig_3_style_in_human_range.png, and prints a summary.
The 12 numbers are computed by the same code for every run and lap (style_rl/dstyle/features.style_vector) on the second timed lap (the flying lap); the steering is the steering COMMAND in steering
wheel degrees (road-wheel angle x 13.4, the project standard). "Human range" = the smallest and largest value of every number over the 4 real drivers on 3 tracks (exp_B, clean laps, a rough scale).
Hands: steering change = mean |a(t) - a(t-1)| per 0.04 s decision in units of full lock, on the straights and in the corners of data/monza_corners.csv.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import drive_style_compare as dsc  # noqa: E402
from drive_run_report import lap_from_trace  # noqa: E402
from style_rl.dstyle.features import KEYS  # noqa: E402
from style_rl.dstyle.track import Track  # noqa: E402
from style_rl.style12 import StyleTargets  # noqa: E402

INK, MUTED, GRID = "#222222", "#6E6E6E", "#E6E6E6"
PALETTE = ["#666666", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]          # Okabe-Ito, the first for the reference run
REAL_STYLE_CSV = ROOT.parent / "jad-race" / "exp_B" / "style_numbers_by_driver_and_track.csv"


def load_evals(run):
    rows = []
    for line in (run / "evals.jsonl").read_text().splitlines():
        r = json.loads(line)
        rel = r.get("reliability") or {}
        ok = sum(v["clean"] for v in rel.values()) if rel else None
        of = sum(v["of"] for v in rel.values()) if rel else None
        row = {"step": r["step"], "hours": r["hours"], "det_clean": r["det_clean"], "lap1_s": r.get("det_lap1_s"), "lap2_s": r.get("det_lap2_s"), "gated": r.get("gated"),
               "other_starts_clean": ok, "other_starts_of": of, "sto_clean": r.get("sto_clean"), "sto_laps": r.get("sto_laps")}
        for k, v in (r.get("smooth") or {}).items():
            row[f"logged_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def hands(run, step, track, segs):
    """Hands and feet of lap 2 of the evaluation trace of `step`: steering change per decision (straights / corners), steering jerk, pedal jerk, both pedals."""
    z = np.load(run / "laps" / f"eval_step_{step}.npz")
    out = lap_from_trace(z, track.length_m, 1)
    if out is None:
        return None
    _, _, _, idx = out
    s, st, th, br = z["station_m"][idx], z["steer"][idx], z["throttle"][idx], z["brake"][idx]
    straight = np.zeros(len(s), bool)
    for _, kind, a, b in segs:
        if kind == "straight":
            straight |= (s >= a) & (s < min(b, track.length_m))
    d1 = np.abs(np.diff(st, prepend=st[0]))
    d2 = np.abs(np.diff(st, 2, prepend=[st[0], st[0]]))
    pj = np.abs(np.diff(th, 2, prepend=[th[0], th[0]])) + np.abs(np.diff(br, 2, prepend=[br[0], br[0]]))
    return {"steer_change_straights": float(d1[straight].mean()), "steer_change_corners": float(d1[~straight].mean()), "steer_jerk": float(d2.mean()), "pedal_jerk": float(pj.mean()),
            "both_pedals_share": float((np.minimum(th, br) > 0.2).mean())}


def human_range():
    df = pd.read_csv(REAL_STYLE_CSV)
    df = df[df.source == "real"]
    return {k: (float(df[k].min()), float(df[k].max())) for k in KEYS}


def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=INK, labelsize=9.5)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--run", action="append", required=True, help="name=path of a run folder; the first one is drawn grey as the reference")
    ap.add_argument("--limit-s", type=float, default=84.92, help="the limit lap of the car on the training line (a2: 84.92 s)")
    ap.add_argument("--track", type=Path, default=ROOT / "data/tracks/monza_geometry.csv")
    ap.add_argument("--corners", type=Path, default=ROOT / "data/monza_corners.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for r in a.run:
        n, p = r.split("=", 1)
        runs[n] = Path(p)
    track = Track(a.track)
    corners = track.find_corners()
    segs = dsc.segments(track.length_m, a.corners)
    tg = StyleTargets()
    hr = human_range()
    col = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(runs)}

    evals, chosen = {}, []
    for n, p in runs.items():
        ev = load_evals(p)
        cfg = json.load(open(p / "config.json"))
        rows = []
        for e in ev.itertuples():
            h = hands(p, e.step, track, segs) if (p / "laps" / f"eval_step_{e.step}.npz").exists() and e.det_clean else None
            rows.append(h or {})
        ev = pd.concat([ev, pd.DataFrame(rows, index=ev.index)], axis=1)
        ev.insert(0, "run", n)
        evals[n] = ev
        clean = ev[ev.det_clean & ev.lap2_s.notna()]
        picks = {}
        if len(clean):
            picks["best_clean"] = clean.loc[clean.lap2_s.idxmin()]
            g = clean[clean.gated == True]  # noqa: E712
            if len(g):
                picks["best_gated"] = g.loc[g.lap2_s.idxmin()]
        for what, row in picks.items():
            nm, v = dsc.rl_lap(p, int(row.step), 2, track, corners, cfg)
            x = nm["command"]
            inside = [hr[k][0] <= x[i] <= hr[k][1] for i, k in enumerate(KEYS)]
            rec = {"run": n, "pick": what, "step": int(row.step), "hours": row.hours, "lap2_s": row.lap2_s, "other_starts_clean": row.other_starts_clean, "other_starts_of": row.other_starts_of,
                   "numbers_inside_human_range_of_12": int(sum(inside)), "D_attacker": tg.distance(x, tg.drivers["attacker"]), "D_smooth": tg.distance(x, tg.drivers["smooth"])}
            rec.update({k: row.get(k) for k in ("steer_change_straights", "steer_change_corners", "steer_jerk", "pedal_jerk", "both_pedals_share")})
            rec.update({k: float(x[i]) for i, k in enumerate(KEYS)})
            rec.update({f"inside_{k}": bool(inside[i]) for i, k in enumerate(KEYS)})
            chosen.append(rec)
    table = pd.concat(evals.values(), ignore_index=True)
    table.to_csv(a.out / "evals_table.csv", index=False)
    ch = pd.DataFrame(chosen)
    ch.to_csv(a.out / "chosen_laps.csv", index=False)

    # figure 1: lap time against steps
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for n, ev in evals.items():
        c = ev[ev.det_clean & ev.lap2_s.notna()]
        ax.plot(c.step / 1e3, c.lap2_s, "-", color=col[n], lw=1.4, alpha=0.9)
        ax.plot(c.step / 1e3, c.lap2_s, "o", color=col[n], ms=5, mec="white", mew=0.8, label=f"{n}  (best {c.lap2_s.min():.2f} s)" if len(c) else n)
        g = c[c.gated == True]  # noqa: E712
        ax.plot(g.step / 1e3, g.lap2_s, "D", color=col[n], ms=8, mfc="none", mew=1.6)
    ax.axhline(a.limit_s, color=MUTED, ls="--", lw=1.0)
    ax.text(ax.get_xlim()[1], a.limit_s - 0.12, f"limit lap {a.limit_s:.2f} s", ha="right", va="top", fontsize=9, color=MUTED)
    lo = min((ev[ev.det_clean].lap2_s.min() for ev in evals.values() if ev.det_clean.any()), default=a.limit_s)
    ax.set_ylim(a.limit_s - 0.8, min(lo + 6.0, 100.0))
    ax.set_xlabel("training steps, thousands", fontsize=10.5)
    ax.set_ylabel("flying lap (lap 2) of the noise-free test, s", fontsize=10.5)
    ax.set_title("Lap time of the test laps that stayed on the track (diamond = also finished all other starts)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    style(ax)
    fig.savefig(a.out / "fig_1_lap_time.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # figure 2: hands against steps
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, key, title in ((axs[0], "steer_change_straights", "on the straights"), (axs[1], "steer_change_corners", "in the corners")):
        for n, ev in evals.items():
            c = ev[ev.det_clean & ev[key].notna()]
            ax.plot(c.step / 1e3, c[key], "-o", color=col[n], lw=1.3, ms=4.5, mec="white", mew=0.7, label=n)
        ax.set_title(f"Steering change per 0.04 s step, {title}", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("training steps, thousands", fontsize=10.5)
        style(ax)
    axs[0].set_ylabel("mean |change of the steering|, units of full lock", fontsize=10.5)
    axs[0].legend(frameon=False, fontsize=9.5)
    fig.savefig(a.out / "fig_2_hands.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # figure 3: the 12 numbers of the chosen laps against the human range
    if len(ch):
        names = [f"{r.run} {r.pick.replace('best_', '')}" for r in ch.itertuples()]
        fig, axs = plt.subplots(3, 4, figsize=(15, 9.5))
        for i, k in enumerate(KEYS):
            ax = axs.flat[i]
            lo, hi = hr[k]
            ax.axhspan(lo, hi, color="#D9D9D9", alpha=0.6, lw=0)
            for j, r in enumerate(ch.itertuples()):
                ax.plot(j, getattr(r, k), "D" if r.pick == "best_gated" else "o", color=col[r.run], ms=8, mec="white", mew=0.8)
            ax.set_title(dsc.PLAIN[k][0], fontsize=10, color=INK, loc="left")
            ax.set_xticks([])
            style(ax)
        axs.flat[0].legend(handles=[plt.Line2D([], [], marker="o", color=col[n], ls="", label=n) for n in runs], frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, -0.05))
        fig.suptitle("The 12 style numbers of the best clean lap (circle) and the best reliable lap (diamond); grey band = range of the 4 real drivers", x=0.01, ha="left", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(a.out / "fig_3_style_in_human_range.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    pd.set_option("display.width", 250)
    show = ["run", "pick", "step", "hours", "lap2_s", "other_starts_clean", "other_starts_of", "numbers_inside_human_range_of_12", "D_attacker", "steer_change_straights", "steer_change_corners", "steer_jerk", "pedal_jerk"]
    print(ch[show].round(3).to_string(index=False))
    for n, ev in evals.items():
        print(f"{n}: {int(ev.det_clean.sum())} of {len(ev)} tests clean, {int((ev.gated == True).sum())} gated, last step {int(ev.step.max())}, {ev.hours.max():.2f} h")  # noqa: E712


if __name__ == "__main__":
    main()
