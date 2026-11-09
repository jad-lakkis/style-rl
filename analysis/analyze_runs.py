"""Analyse everything that was saved by a style-rl run:   python analysis/analyze_runs.py --runs <folder with the agent folders> --out <output folder>

Reads, for every agent folder (<driver>_<name>): eval.csv, episodes.csv, progress.csv, baseline.json, records.csv, laps/*.csv, corners/*.csv.
Writes figures (one idea per figure), tables (csv) and REPORT.md (plain language, with the numbers).

Definitions used everywhere
  base driver       the original driver model alone in the OCD simulator (laps/base_driver.csv, baseline.json). It is what the agents are compared with.
  style error       mean over the corners of the sum of the five scaled differences (brake start, brake end, peak brake, mean brake, throttle point) to the reference driver; 0 = identical.
  tolerance         1.25 x the base driver's style error + 0.3 (the same rule that decides `records`).
  matched corner    a corner whose style error is <= 2.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.style import FEATURES, SCALE  # noqa: E402

COL = {"pure_bc2": "#0072B2", "pure_bc3": "#D55E00", "residual": "#7F7F7F", "pure_bc": "#CC79A7", "pure": "#009E73"}
LABEL = {"pure_bc2": "pure, anchored (bc2)", "pure_bc3": "pure, anchored, 5x progress reward (bc3)", "residual": "residual (base driver + correction)", "pure_bc": "pure, first anchor (diverged)", "pure": "pure from scratch"}
DRIVERS = ["attacker", "smooth"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110, "savefig.bbox": "tight"})


def load(runs):
    A = {}
    for d in sorted(p for p in runs.iterdir() if (p / "eval.csv").is_file()):
        m = re.match(r"(attacker|smooth)_(.+)", d.name)
        if not m:
            continue
        drv, spec = m.groups()
        b = json.loads((runs / f"{drv}_residual" / "baseline.json").read_text())      # the original driver of this driver model
        A[d.name] = dict(dir=d, drv=drv, spec=spec, base=b, tol=1.25 * b["style_error"] + 0.3, ev=pd.read_csv(d / "eval.csv"), ep=pd.read_csv(d / "episodes.csv") if (d / "episodes.csv").is_file() else None,
                         pr=pd.read_csv(d / "progress.csv") if (d / "progress.csv").is_file() else None, rec=pd.read_csv(d / "records.csv") if (d / "records.csv").is_file() else None)
        e = A[d.name]["ev"]
        e["ok"] = (e.end == "lap") & (e.off_track_share < 0.01)
        e["delta"] = e.lap_time - b["lap_time"]
    return A


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    return name


def main_specs(A):
    return [s for s in ("pure_bc2", "pure_bc3", "residual") if any(v["spec"] == s for v in A.values())]


def fig_lap_time(A, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, drv in zip(axs, DRIVERS):
        b = next(v["base"] for v in A.values() if v["drv"] == drv)
        for spec in main_specs(A):
            v = A.get(f"{drv}_{spec}")
            if v is None:
                continue
            e = v["ev"][v["ev"].ok]
            ax.plot(e.step / 1000, e.lap_time, "-o", ms=3, lw=1.2, color=COL[spec], label=LABEL[spec])
        ax.axhline(b["lap_time"], color="k", ls="--", lw=1.2, label="original driver in OCD")
        ax.axhline(b["reference_lap_time"], color="k", ls=":", lw=1, label="reference lap (driver model)")
        ax.set_title(f"{drv}: lap time of every finished evaluation lap")
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylabel("lap time (s), lower is faster")
        ax.set_ylim(top=min(ax.get_ylim()[1], b["lap_time"] + 8))
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Did the agents get faster than the original driver? Only briefly, and by a few tenths of a second", y=1.02)
    return save(fig, out, "01_lap_time_over_training.png")


def fig_style(A, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, drv in zip(axs, DRIVERS):
        b = next(v["base"] for v in A.values() if v["drv"] == drv)
        for spec in main_specs(A):
            v = A.get(f"{drv}_{spec}")
            if v is None:
                continue
            e = v["ev"][v["ev"].ok]
            ax.plot(e.step / 1000, e.style_error, "-o", ms=3, lw=1.2, color=COL[spec], label=LABEL[spec])
        ax.axhline(b["style_error"], color="k", ls="--", lw=1.2, label="original driver in OCD")
        ax.axhline(1.25 * b["style_error"] + 0.3, color="k", ls=":", lw=1, label="tolerance for a 'record'")
        ax.set_title(f"{drv}: style error of every finished evaluation lap")
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylabel("style error, lower = closer to the driver")
        ax.set_ylim(0, 10)
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Style: the anchored agents stay near the original driver while the pull to it is strong, then drift away", y=1.02)
    return save(fig, out, "02_style_error_over_training.png")


def fig_tradeoff(A, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, drv in zip(axs, DRIVERS):
        b = next(v["base"] for v in A.values() if v["drv"] == drv)
        tol = 1.25 * b["style_error"] + 0.3
        ax.axvspan(-1.0, 0, ymin=0, ymax=1, color="#2ca02c", alpha=0.05)
        for spec in main_specs(A):
            v = A.get(f"{drv}_{spec}")
            if v is None:
                continue
            e = v["ev"][v["ev"].ok]
            ax.scatter(e.delta, e.style_error, s=18, color=COL[spec], alpha=0.75, label=LABEL[spec])
            if v["rec"] is not None:
                ax.scatter(v["rec"].delta_vs_base, v["rec"].style_error, s=130, facecolors="none", edgecolors="k", lw=1.5)
        ax.axhline(tol, color="k", ls=":", lw=1)
        ax.axhline(b["style_error"], color="k", ls="--", lw=1)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(-1.5, 8)
        ax.set_ylim(0, 10)
        ax.set_xlabel("lap time minus the original driver's (s)    <- faster | slower ->")
        ax.set_ylabel("style error (lower = closer to the driver)")
        ax.set_title(f"{drv}: every evaluation lap (circled = record)")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Pace against style: the goal is the lower-left corner (faster than the original driver, style kept)", y=1.02)
    return save(fig, out, "03_pace_vs_style.png")


def constraint_table(A):
    rows = []
    for name, v in A.items():
        e, b = v["ev"], v["base"]
        for part, m in (("all", e.step > 0), ("step <= 300k", e.step <= 300_000), ("step > 300k", e.step > 300_000)):
            s = e[m]
            if not len(s):
                continue
            okA = s.ok
            okB = okA & (s.style_error <= v["tol"])
            okD = okA & (s.delta.abs() <= 1.0)
            rows.append(dict(agent=name, evaluations=part, n=len(s), on_track_and_lap=okA.mean(), style_within_tolerance=okB.mean(), corners_matched=s.style_ok_share.mean(), lap_within_1s_of_base=okD.mean(),
                             style_and_pace_together=(okB & okD).mean()))
    return pd.DataFrame(rows)


def fig_constraints(A, out, T):
    cols = [("on_track_and_lap", "stays on track\nand finishes"), ("style_within_tolerance", "style within\ntolerance"), ("lap_within_1s_of_base", "lap within 1 s\nof the original"), ("corners_matched", "corners matched\n(error <= 2)")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, drv in zip(axs, DRIVERS):
        names = [f"{drv}_{s}" for s in main_specs(A) if f"{drv}_{s}" in A]
        w = 0.8 / (2 * len(names))
        for i, name in enumerate(names):
            for j, part in enumerate(("step <= 300k", "step > 300k")):
                r = T[(T.agent == name) & (T.evaluations == part)]
                if not len(r):
                    continue
                vals = [float(r[c].iloc[0]) for c, _ in cols]
                ax.bar(np.arange(len(cols)) + (2 * i + j - len(names) + 0.5) * w, vals, w * 0.95, color=COL[A[name]["spec"]], alpha=1.0 if j == 0 else 0.45,
                       label=f"{LABEL[A[name]['spec']]} - {'before 300k' if j == 0 else 'after 300k'}")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([c[1] for c in cols], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{drv}: share of evaluations that meet each constraint")
    axs[0].set_ylabel("share of evaluations")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.suptitle("Constraints: staying on the track holds throughout; style and pace hold only while the pull to the original driver is strong (dark = before 300k steps, light = after)", y=1.03, fontsize=10)
    return save(fig, out, "04_constraint_satisfaction.png")


def fig_reward(A, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, drv in zip(axs, DRIVERS):
        for spec in main_specs(A):
            v = A.get(f"{drv}_{spec}")
            if v is None or v["ep"] is None:
                continue
            ep = v["ep"]
            ret = ep[[c for c in ep.columns if c.startswith("r_")]].sum(axis=1)
            ax.plot(ep.step / 1000, ret.rolling(50, min_periods=10).mean(), color=COL[spec], lw=1.4, label=LABEL[spec])
        ax.set_title(f"{drv}: training return per 700-step episode (mean of 50)")
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylabel("return (bc3 uses 5x progress reward: not comparable with the others)")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Reward: it rises and flattens, but full-lap results do not improve with it", y=1.02)
    return save(fig, out, "05_training_return.png")


def fig_reward_parts(A, out):
    names = [n for n in ("attacker_pure_bc2", "attacker_pure_bc3") if n in A]
    fig, axs = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.2), squeeze=False)
    parts = [("r_progress", "progress", "#0072B2"), ("r_style", "style", "#D55E00"), ("r_jerk", "jerky commands", "#009E73"), ("r_slow", "slower than reference", "#E69F00"), ("r_off_track", "off-track", "#CC79A7"), ("r_line", "off the line", "#7F7F7F")]
    for ax, n in zip(axs[0], names):
        ep = A[n]["ep"]
        for c, lab, col in parts:
            ax.plot(ep.step / 1000, ep[c].rolling(60, min_periods=10).mean(), color=col, lw=1.3, label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{n}: what the training return is made of")
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylabel("reward per episode (mean of 60)")
    axs[0][0].legend(fontsize=8)
    return save(fig, out, "06_reward_parts.png")


def fig_diagnostics(A, out):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    pairs = [("train/ent_coef", "entropy coefficient (log scale)", True), ("train/critic_loss", "critic loss (log scale)", True), ("train/bc_weight", "weight of the pull to the original driver", False),
             ("train/actor_loss", "actor loss", False), ("rollout/ep_len_mean", "mean episode length (of 700)", False), ("train/n_updates", "gradient updates", False)]
    for ax, (c, t, lg) in zip(axs.ravel(), pairs):
        for name, v in A.items():
            pr = v["pr"]
            if pr is None or c not in pr or v["spec"] not in ("pure_bc", "pure_bc2", "pure_bc3", "residual"):
                continue
            ax.plot(pr["time/total_timesteps"] / 1000, pr[c], color=COL[v["spec"]], lw=1.0, alpha=0.85, label=LABEL[v["spec"]] if v["drv"] == "attacker" else None)
        if lg:
            ax.set_yscale("log")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("training steps (thousands)")
    axs[0][0].legend(fontsize=7, loc="center left")
    fig.suptitle("Learning health: the first anchor (pink) let the entropy coefficient and critic loss explode; the fixed anchor (blue, orange) stays stable", y=1.0)
    fig.tight_layout()
    return save(fig, out, "07_learning_diagnostics.png")


def fig_traces(A, out, runs):
    outs = []
    for name in ("attacker_pure_bc2", "smooth_pure_bc3"):
        if name not in A:
            continue
        v = A[name]
        d = v["dir"]
        e = v["ev"][v["ev"].ok]
        best = e.loc[e.lap_time.idxmin()]
        last = e.iloc[-1]
        laps = {"original driver": pd.read_csv(d / "laps" / "base_driver.csv"), f"best lap (step {int(best.step / 1000)}k)": pd.read_csv(d / "laps" / f"step_{int(best.step)}.csv"),
                f"last lap (step {int(last.step / 1000)}k)": pd.read_csv(d / "laps" / f"step_{int(last.step)}.csv")}
        cols = ["k", COL[v["spec"]], "#999999"]
        fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for (lab, df), c in zip(laps.items(), cols):
            axs[0].plot(df.progress, df.speed * 3.6, color=c, lw=1.1, label=lab)
            axs[1].plot(df.progress, df.brake, color=c, lw=1.0)
            axs[2].plot(df.progress, df.throttle, color=c, lw=1.0)
        axs[0].set_ylabel("speed (km/h)")
        axs[1].set_ylabel("brake pedal (0-1)")
        axs[2].set_ylabel("throttle (0-1)")
        axs[2].set_xlabel("distance along the lap (m)")
        axs[0].legend(fontsize=9, loc="lower right")
        axs[0].set_title(f"{name}: speed and pedals of the original driver, the best lap and the last lap")
        outs.append(save(fig, out, f"08_lap_traces_{name}.png"))
    return outs


def fig_corners(A, out):
    outs = []
    for name in ("attacker_pure_bc2", "smooth_pure_bc3"):
        if name not in A:
            continue
        v = A[name]
        e = v["ev"][v["ev"].ok & (v["ev"].corners_matched > 0)]
        best = e.loc[e.lap_time.idxmin()]
        last = e.iloc[-1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        for ax, r, lab in zip(axs, (best, last), ("best lap", "last lap")):
            c = pd.read_csv(v["dir"] / "corners" / f"step_{int(r.step)}.csv")
            M = np.array([[(row[1].agent - row[1].reference) / SCALE[FEATURES.index(f)] for f in FEATURES for row in [(None, c[(c.corner == k) & (c.feature == f)].iloc[0])]] for k in sorted(c.corner.unique())])
            im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
            ax.set_xticks(range(len(FEATURES)))
            ax.set_xticklabels(["brake start", "brake end", "peak brake", "mean brake", "throttle point"], fontsize=8, rotation=20)
            ax.set_yticks(range(len(M)))
            ax.set_yticklabels([f"corner {k}" for k in sorted(c.corner.unique())], fontsize=8)
            ax.set_title(f"{lab} (step {int(r.step / 1000)}k, style error {r.style_error:.2f})", fontsize=9)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7, color="k")
        fig.colorbar(im, ax=axs, fraction=0.025, label="agent minus reference, in units of the style scale (0 = same)")
        fig.suptitle(f"{name}: how the style differs from the reference driver, corner by corner", y=1.04)
        outs.append(save(fig, out, f"09_corner_style_{name}.png"))
    return outs


def best_table(A):
    rows = []
    for name, v in A.items():
        e, b = v["ev"], v["base"]
        ok = e[e.ok]
        if not len(ok):
            rows.append(dict(agent=name, steps=int(e.step.iloc[-1]), evals=len(e), laps_finished=0))
            continue
        best = ok.loc[ok.lap_time.idxmin()]
        rec = v["rec"]
        rows.append(dict(agent=name, steps=int(e.step.iloc[-1]), evals=len(e), laps_finished=int(e.ok.sum()), base_lap=b["lap_time"], base_style=b["style_error"], best_lap=best.lap_time, best_vs_base=best.delta,
                         best_at_step=int(best.step), style_at_best=best.style_error, last_lap=ok.lap_time.iloc[-1], last_style=ok.style_error.iloc[-1], records=0 if rec is None else len(rec),
                         best_record_lap=None if rec is None else rec.lap_time.min()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    A = load(args.runs)
    figs = [fig_lap_time(A, args.out), fig_style(A, args.out), fig_tradeoff(A, args.out)]
    T = constraint_table(A)
    T.to_csv(args.out / "constraints_by_agent.csv", index=False)
    figs += [fig_constraints(A, args.out, T), fig_reward(A, args.out), fig_reward_parts(A, args.out), fig_diagnostics(A, args.out)]
    figs += fig_traces(A, args.out, args.runs) + fig_corners(A, args.out)
    B = best_table(A)
    B.to_csv(args.out / "best_by_agent.csv", index=False)
    print(B.round(2).to_string(index=False))
    print(T.round(2).to_string(index=False))
    print("figures:", figs)


if __name__ == "__main__":
    main()
