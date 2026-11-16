"""History tables, figures and a README for every training run copied from the pods.

    python tools/export_run_history.py RAW OUT

RAW = the folder the pod archive was unpacked into (it holds runs12/, runs/, archive_old_runs/, launch scripts)
OUT = where the organised copy goes. Every run folder is copied in full (nothing is dropped) and gets, next to the raw files:

  history_evaluations.csv   one row per test lap (every 20k steps): lap time, distance reached, style distance D, and the best so far
  history_laps.csv          every finished lap of the run in the order it was driven, with the 12 style numbers, the change of each number against the driver's run
                            (raw units) and flags: record lap time, record lap time inside the style budget
  records_fastest.csv       only the laps that set a new best lap time (any style): the lap-time staircase
  records_in_budget.csv     only the laps that set a new best lap time while keeping D <= D(driver's run) + 0.05
  progress.png              lap time (or distance reached) and style distance D along the training
  style_records.png         the 12 style numbers (raw values with units) of the record laps against the driver's run
  README.md                 what the run is, what it reached, what every file is
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lap_report import PLAIN, UNIT  # noqa: E402

KEYS = ["B_mean_active", "B_event_peak_mean", "B_apply_rate", "B_release_rate", "T_apply_rate", "T_release_rate",
        "S_reversals_5deg_per_km", "S_mean_abs_rate", "S_corner_peak_angle", "L_border_mean_m", "L_inside_dist_apex_m", "L_outside_dist_exit_m"]
TOL = 0.05            # style budget = D of the driver's run + TOL (the same rule as tools/lap_compare.py)
MAX_OFF = 1e-3        # a lap counts as on track if at most this share of it is off the track
LAP_M = 5756.0
GREY, BLUE, RED = "#555555", "#0072B2", "#D55E00"


def read_csv(p):
    try:
        return pd.read_csv(p) if p.is_file() else None
    except Exception:
        return None


def describe(args):
    """One plain sentence on how the agent drives, read from the run's args.json."""
    if not args:
        return "no args.json in this folder (an older run): see the log."
    if "mode" in args and "residual" not in args:      # the first experiments (runs/, runs/v2/), before the whole-lap 12-number environment
        m = args["mode"]
        what = ("the RL agent adds a correction (steering up to %s, pedals up to %s) to a pure-pursuit base driver" % (args.get("res_steer"), args.get("res_pedal"))) if m == "residual" else "the RL agent steers and uses the pedals itself"
        return (f"Early experiment (before the whole-lap 12-number environment), mode '{m}': {what}; "
                f"training episodes of {args.get('episode_steps')} steps; behaviour-cloning weight {args.get('bc_weight')}, warm start {args.get('warm_start')}. Test laps cover the whole lap.")
    if args.get("residual"):
        return (f"RL on top of pure pursuit: the car is driven by the pure-pursuit driver and the RL agent adds a bounded correction "
                f"(steering up to {args.get('res_steer')} of full lock, pedals up to {args.get('res_pedal')}).")
    if args.get("no_warm_start"):
        aids = []
        if args.get("ped_w", 0) > 0:
            aids.append(f"a pedal aid (weight {args['ped_w']}, fading over {args.get('ped_decay')} steps)")
        if args.get("steer_w", 0) > 0:
            aids.append(f"a steering aid (weight {args['steer_w']})")
        if args.get("speed_w", 0) > 0:
            aids.append(f"a speed reward (weight {args['speed_w']})")
        if args.get("hide_base_command"):
            aids.append("the pure-pursuit command hidden from the agent")
        s = "RL from scratch: a random network drives the car with its own steering, throttle and brake (no pure pursuit)"
        s += (", with " + ", ".join(aids)) if aids else ""
        return s + "."
    return ("RL started as a copy of the pure-pursuit driver (the neutral base driver) and was then free to move away from it "
            f"(the pull towards it, bc_weight {args.get('bc_weight')}, fades over {args.get('bc_decay')} steps).")


def group_of(args, rel):
    if rel.startswith("runs/"):
        return "05_older_runs"
    if args and args.get("residual"):
        return "01_RL_on_top_of_pure_pursuit"
    if args and args.get("no_warm_start"):
        return "02_RL_from_scratch"
    if rel.startswith("archive_old_runs/") and re.search(r"stage1|scratch", rel):
        return "02_RL_from_scratch"
    return "03_RL_started_from_pure_pursuit"


def analyse(run, out, name):
    """Returns (summary dict, text for the README). Writes the history files and figures into `out`."""
    args = json.load(open(run / "args.json")) if (run / "args.json").is_file() else None
    ev, lib, ep = read_csv(run / "eval.csv"), read_csv(run / "library.csv"), read_csv(run / "episodes.csv")
    bl = json.load(open(run / "baselines.json")) if (run / "baselines.json").is_file() else None
    S = {"run": name, "how": describe(args)}
    S["old_format"] = bool(args and "mode" in args and "residual" not in args)
    t_drv = d_drv = None
    if bl and "pure_pursuit_driver" in bl:
        t_drv, d_drv = float(bl["pure_pursuit_driver"]["lap_time"]), float(bl["pure_pursuit_driver"]["D"])
        S.update(driver_run_lap_s=t_drv, driver_run_D=d_drv, style_budget_D=d_drv + TOL)
    if bl and "neutral" in bl:
        S.update(neutral_lap_s=float(bl["neutral"]["lap_time"]), neutral_D=float(bl["neutral"]["D"]))
    budget = d_drv + TOL if d_drv is not None else None
    lines = []

    # ---- evaluations
    if ev is not None and len(ev):
        ev = ev.copy()
        if "off_share" not in ev:
            ev["off_share"] = ev["off_track_share"] if "off_track_share" in ev else 0.0
        if "distance_m" not in ev:
            ev["distance_m"] = np.nan
        if "D" not in ev:
            ev["D"] = np.nan
        ev["finished"] = ev.completed.astype(str).str.lower().eq("true")
        ev["on_track_lap"] = ev.finished & (ev.off_share.fillna(1) <= MAX_OFF)
        ev["best_lap_so_far_s"] = ev.lap_time.where(ev.on_track_lap).cummin()
        ev["best_distance_so_far_m"] = ev.distance_m.cummax()
        ev.to_csv(out / "history_evaluations.csv", index=False)
        S.update(steps=int(ev.step.max()), hours=float(ev.hours.max()), evaluations=len(ev), evaluations_finished=int(ev.finished.sum()),
                 best_eval_distance_m=float(ev.distance_m.max()))
        if ev.on_track_lap.any():
            S["best_eval_lap_s"] = float(ev.lap_time.where(ev.on_track_lap).min())
            S["first_finished_eval_step"] = int(ev.step[ev.finished].iloc[0])
    # ---- library (every finished lap)
    L = None
    if lib is not None and len(lib) and "lap_time" in lib:
        L = lib[lib.lap_time.notna()].copy()
        for c, v in (("off_share", 0.0), ("start_m", 0.0), ("D", np.nan), ("id", 0)):
            if c not in L:
                L[c] = v
        L = L.sort_values(["step", "id"]).reset_index(drop=True)
        L["from_line"] = L.start_m.fillna(0.0) < 1.0
        L["on_track"] = L.off_share.fillna(1) <= MAX_OFF
        ok = L.from_line & L.on_track
        if t_drv is not None:
            L["gain_vs_driver_run_s"] = t_drv - L.lap_time
            for k in KEYS:
                if k in L and bl["pure_pursuit_driver"].get("numbers", {}).get(k) is not None:
                    L["change_" + k] = L[k] - float(bl["pure_pursuit_driver"]["numbers"][k])
        L["in_style_budget"] = (L.D <= budget) if budget is not None else False
        L["record_fastest"] = False
        L["record_in_budget"] = False
        best = best_b = np.inf
        for i in L.index:
            if not ok[i]:
                continue
            if L.lap_time[i] < best - 1e-9:
                best = L.lap_time[i]
                L.loc[i, "record_fastest"] = True
            if L.in_style_budget[i] and L.lap_time[i] < best_b - 1e-9:
                best_b = L.lap_time[i]
                L.loc[i, "record_in_budget"] = True
        L["best_lap_so_far_s"] = L.lap_time.where(ok).cummin()
        L["best_in_budget_so_far_s"] = L.lap_time.where(ok & L.in_style_budget).cummin()
        L.to_csv(out / "history_laps.csv", index=False)
        L[L.record_fastest].to_csv(out / "records_fastest.csv", index=False)
        L[L.record_in_budget].to_csv(out / "records_in_budget.csv", index=False)
        S.update(laps_finished=len(L), laps_on_track_from_line=int(ok.sum()))
        if ok.any():
            o = L[ok]
            b = o.loc[o.lap_time.idxmin()]
            S.update(best_lap_s=float(b.lap_time), best_lap_D=float(b.D), best_lap_step=int(b.step), median_lap_s=float(o.lap_time.median()))
            if t_drv is not None:
                S.update(laps_faster_than_driver_run=int((o.lap_time < t_drv).sum()), gain_best_s=t_drv - float(b.lap_time))
            ib = o[o.in_style_budget]
            S["laps_in_style_budget"] = len(ib)
            if len(ib):
                bb = ib.loc[ib.lap_time.idxmin()]
                S.update(best_in_budget_lap_s=float(bb.lap_time), best_in_budget_D=float(bb.D), best_in_budget_step=int(bb.step))
                if t_drv is not None:
                    S["gain_in_budget_s"] = t_drv - float(bb.lap_time)
            late = o[o.step > 0.75 * float(L.step.max())]
            if len(late):
                S.update(late_median_lap_s=float(late.lap_time.median()), late_laps=len(late))
    # ---- figures
    figs = []
    if ev is not None and len(ev):
        figs.append(progress_fig(out, name, ev, L, ep, t_drv, d_drv, budget, S))
        if L is not None and L.record_fastest.any() and t_drv is not None and any(("change_" + k) in L for k in KEYS):
            figs.append(style_fig(out, name, L, bl))
    return S, figs, args, ev, L


def progress_fig(out, name, ev, L, ep, t_drv, d_drv, budget, S):
    finishing = ev.on_track_lap.any() or (L is not None and L.record_fastest.any())
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if finishing:
        a = ax[0]
        if L is not None:
            ok = L.from_line & L.on_track
            a.scatter(L.step[ok], L.lap_time[ok], s=5, color="#bbbbbb", label="every finished lap (from the line)", zorder=1)
            a.step(L.step, L.best_lap_so_far_s, where="post", color=RED, lw=2, label="fastest lap so far, any style", zorder=3)
            a.step(L.step, L.best_in_budget_so_far_s, where="post", color=BLUE, lw=2, label="fastest lap so far inside the style budget", zorder=3)
        a.plot(ev.step, ev.lap_time.where(ev.on_track_lap), color=GREY, lw=0.8, alpha=0.8, label="test lap every 20k steps")
        if t_drv is not None:
            a.axhline(t_drv, color="k", ls="--", lw=1, label=f"driver's run {t_drv:.2f} s")
        a.set_ylabel("lap time (s)")
        lo = np.nanmin([ev.lap_time.where(ev.on_track_lap).min(), (L.lap_time[L.from_line & L.on_track].min() if L is not None and (L.from_line & L.on_track).any() else np.inf)])
        a.set_ylim(lo - 0.5, (t_drv if t_drv else lo + 6) + 3)
        a.legend(fontsize=7, loc="upper right", ncol=2)
        a.set_title(f"{name}: lap time along the training")
    else:
        a = ax[0]
        a.plot(ev.step, ev.distance_m, ".", color="#bbbbbb", ms=4, label="distance reached in a test lap")
        a.step(ev.step, ev.best_distance_so_far_m, where="post", color=RED, lw=2, label="furthest so far")
        if ep is not None and "episode/progress_m" in ep:
            pass
        a.axhline(LAP_M, color="k", ls="--", lw=1, label=f"lap length {LAP_M:.0f} m")
        a.set_ylabel("distance reached before leaving the track (m)")
        a.legend(fontsize=7, loc="center right")
        a.set_title(f"{name}: no lap finished; distance reached along the training")
    b = ax[1]
    if L is not None:
        ok = L.from_line & L.on_track
        b.scatter(L.step[ok], L.D[ok], s=5, color="#bbbbbb", label="every finished lap (from the line)")
    b.plot(ev.step, ev.D, color=GREY, lw=0.8, alpha=0.8, label="test lap every 20k steps")
    if d_drv is not None:
        b.axhline(d_drv, color="k", ls="--", lw=1, label=f"driver's run D = {d_drv:.2f}")
        b.axhline(budget, color=BLUE, ls=":", lw=1.5, label=f"style budget D = {budget:.2f}")
    b.set_ylabel("style distance D (0 = the driver's style)")
    b.set_xlabel("training steps")
    b.set_ylim(0, min(3.0, max(1.2, float(np.nanpercentile(ev.D.dropna(), 95)) * 1.1)) if ev.D.notna().any() else 3.0)
    b.legend(fontsize=7, loc="upper right")
    for x in ax:
        x.grid(True, color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.tight_layout()
    p = out / "progress.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p.name


def style_fig(out, name, L, bl):
    rec = L[L.record_fastest]
    recb = L[L.record_in_budget]
    drv = bl["pure_pursuit_driver"]["numbers"]
    fig, axs = plt.subplots(3, 4, figsize=(14, 8))
    for a, k in zip(axs.ravel(), KEYS):
        a.plot(rec.step, rec[k], "-o", color=RED, ms=3, lw=1, label="record laps, any style")
        if len(recb):
            a.plot(recb.step, recb[k], "-s", color=BLUE, ms=3, lw=1, label="record laps inside the style budget")
        a.axhline(float(drv[k]), color="k", ls="--", lw=1, label="driver's run")
        a.set_title(f"{PLAIN[k].split(' (')[0]} ({UNIT[k]})", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(True, color="#e5e5e5", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axs[0, 0].legend(fontsize=6)
    fig.suptitle(f"{name}: the 12 style numbers of each new record lap (x = training step), raw values", fontsize=10)
    fig.tight_layout()
    p = out / "style_records.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def f(x, d=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def result_text(S, ev=None):
    t = []
    if S.get("old_format"):
        t.append(f"- Trained for {S.get('steps', 0):,} steps. {S.get('evaluations_finished', 0)} of {S.get('evaluations', 0)} test laps (one every 25k steps) finished the whole lap; furthest distance reached {S.get('best_eval_distance_m', 0):.0f} m of {LAP_M:.0f} m.")
        if "best_eval_lap_s" in S:
            t.append(f"- Fastest finished test lap: {S['best_eval_lap_s']:.2f} s. The eval.csv of this older format also lists, per test lap, the lap time of the reference lap and of the base driver (reference_lap_time, base_lap_time) and its own style error (style_error, an earlier definition, not the 12-number D).")
        else:
            t.append("- No test lap was finished.")
        t.append("- This older format has no library of laps and no baselines.json, so there is no per-lap history of the 12 style numbers; history_evaluations.csv holds the test-lap history.")
        return "\n".join(t)
    if "steps" in S:
        t.append(f"- Trained for {S['steps']:,} steps ({S['hours']:.2f} h of the last session; earlier sessions of the same run are in the log).")
    if "driver_run_lap_s" in S:
        t.append(f"- Driver's run (pure pursuit on the driver's line): {S['driver_run_lap_s']:.2f} s, style distance D {S['driver_run_D']:.2f}. Style budget used for 'style kept' laps: D <= {S['style_budget_D']:.2f} (the driver's D plus 0.05).")
    if "neutral_lap_s" in S:
        t.append(f"- Neutral base driver (pure pursuit on the track's own line): {S['neutral_lap_s']:.2f} s, D {S['neutral_D']:.2f}.")
    if "evaluations" in S:
        t.append(f"- Test laps (every 20k steps): {S['evaluations_finished']} of {S['evaluations']} finished the lap; furthest distance reached {S['best_eval_distance_m']:.0f} m of {LAP_M:.0f} m.")
    if "laps_finished" in S and S.get("laps_on_track_from_line", 0):
        t.append(f"- Finished on-track laps from the line: {S['laps_on_track_from_line']} (of {S['laps_finished']} finished laps in the library).")
        t.append(f"- Fastest lap: {S['best_lap_s']:.2f} s (D {S['best_lap_D']:.2f}, step {S['best_lap_step']:,})" + (f", {S['gain_best_s']:.2f} s faster than the driver's run; {S['laps_faster_than_driver_run']} laps were faster than the driver's run." if "gain_best_s" in S else "."))
        if "best_in_budget_lap_s" in S:
            t.append(f"- Fastest lap inside the style budget: {S['best_in_budget_lap_s']:.2f} s (D {S['best_in_budget_D']:.2f}, step {S['best_in_budget_step']:,})" + (f", {S['gain_in_budget_s']:.2f} s faster than the driver's run." if "gain_in_budget_s" in S else "."))
        else:
            t.append("- No lap kept the style budget and was on track.")
        if "late_median_lap_s" in S:
            t.append(f"- Typical (median) lap in the last quarter of the training: {S['late_median_lap_s']:.2f} s over {S['late_laps']} laps.")
    elif "evaluations" in S:
        t.append("- No on-track lap from the line was finished by this run.")
    return "\n".join(t)


FILES = {
    "args.json": "every setting of the run (learning rate, rewards, how the agent drives, style budget settings).",
    "baselines.json": "lap time, D and the 12 style numbers of the driver's run (pure pursuit on the driver's line) and of the neutral base driver.",
    "baseline_pure_pursuit_driver.csv": "the driver's run, sample by sample (time, position, speed, steering, throttle, brake).",
    "baseline_neutral.csv": "the neutral base driver's lap, sample by sample.",
    "eval.csv": "the test lap at every 20k steps: finished or not, lap time, distance reached, D, mean speed, mean brake.",
    "episodes.csv": "every training episode: time reward, style reward, distance reached, D at the end, share off the track.",
    "progress.csv": "the training log of the learner (losses, reward, steps per second), one row every 20 episodes.",
    "library.csv": "every finished lap (training and test): lap time, D, start point, share off the track, the 12 style numbers, and the file of its trajectory.",
    "state.json": "the last state of the style multiplier and of the lap library.",
    "system.txt": "software and machine versions.",
    "latest.zip": "the last saved agent (SAC policy and critics, stable-baselines3 format); the replay buffer is not copied (3.5 GB of training state, still on the pod's volume).",
    "laps": "the trajectory of every finished lap: train_<n>.npz (training laps), eval_step_<step>.csv (test laps).",
    "checkpoints": "the test-lap trajectory at every 20k steps (eval_step_<step>.csv). The saved agents of each step (step_<n>.zip) are not copied.",
    "history_evaluations.csv": "NEW: one row per test lap with the best lap time and the furthest distance so far.",
    "history_laps.csv": "NEW: every finished lap in the order driven, with the 12 style numbers, their change against the driver's run (raw units), and record flags.",
    "records_fastest.csv": "NEW: only the laps that set a new best lap time (the staircase of improvement).",
    "records_in_budget.csv": "NEW: only the laps that set a new best lap time while keeping the style budget.",
}


def write_readme(out, S, figs, rel, group):
    name = rel
    txt = [f"# {name}", "", f"Group: {group}.", ""]
    if rel.startswith("archive_old_runs/"):
        txt += ["Note: this is the archived copy kept when the run was moved aside; the same run (same numbers) is also in this export under its runs12/ name.", ""]
    txt += [f"**How the car is driven:** {S['how']}", "", "## What it reached", "", result_text(S) or "- no evaluation data in this folder.", ""]
    if figs:
        txt += ["## Figures", ""]
        for fg in figs:
            if fg == "progress.png":
                txt.append(f"- `{fg}`: top panel the lap time along the training (grey dots every finished lap, red staircase the fastest lap so far, blue staircase the fastest lap so far inside the style budget, dashed line the driver's run); for a run that never finished a lap the furthest distance reached instead. Bottom panel the style distance D of the laps, with the driver's D and the budget.")
            else:
                txt.append(f"- `{fg}`: the 12 style numbers (raw values, own units) of each new record lap, against the driver's run (dashed). Shows how far the style moves while the lap time improves.")
        txt.append("")
    txt += ["## Files", ""]
    for p in sorted(out.iterdir()):
        if p.name == "README.md" or p.name.endswith(".png"):
            continue
        d = FILES.get(p.name) or ("training log of this run (text)." if p.suffix == ".log" else "TensorBoard log of one training session." if p.name.startswith("events.out") else "")
        txt.append(f"- `{p.name}`: {d}".rstrip(": "))
    txt += ["", "## How to read the style distance D", "",
            "D is the distance between the 12 style numbers of a lap and the driver's style, averaged over the 12 numbers, each difference divided by how much that number normally varies between laps. 0 means the same style. The RL training holds D near the driver's: a lap 'keeps the style' when D is at most the driver's own D plus 0.05.", ""]
    (out / "README.md").write_text("\n".join(txt), encoding="utf-8")
    sub = {"laps": "The trajectory of every finished lap of this run, sampled every 0.04 s: columns t (s), x, y (m), speed (m/s), lateral (m, positive = left of the line), steer, throttle, brake, progress (m along the lap), inside (on the track). train_<n>.npz = a training lap (the same arrays as an .npz file), eval_step_<step>.csv = the test lap at that step. library.csv in the run folder says which file is which lap, with its lap time, D and the 12 style numbers.",
           "checkpoints": "The test lap at every evaluation step as eval_step_<step>.csv (same columns as in the laps folder). The saved agent of each step (step_<n>.zip) is not in this folder (see the top README)."}
    for k, v in sub.items():
        if (out / k).is_dir():
            (out / k / "README.md").write_text(f"# {k}\n\n{v}\n", encoding="utf-8")


GROUPS = {
    "01_RL_on_top_of_pure_pursuit": "The car is driven by the pure-pursuit driver and the RL agent only adds a small bounded correction (residual mode). These are the runs that finish laps and beat the driver's run.",
    "02_RL_from_scratch": "The RL agent drives alone from a random network: no pure pursuit underneath (some runs have training aids: a pedal aid, a steering aid, a speed reward). None of these finished a lap; they leave the track at the first chicane (about 800-1000 m). archive_old_runs__* folders are archived copies of runs that also appear under runs12__*.",
    "03_RL_started_from_pure_pursuit": "The RL agent started as a copy of the pure-pursuit driver and was free to move away (a pull towards it that fades). Short runs (140k-380k steps) from the first whole-lap experiments.",
    "05_older_runs": "The first experiments (before the whole-lap 12-number environment): short training episodes, a different eval.csv format, no lap library. Kept for completeness.",
}


def write_group_readme(gdir, rows):
    g = gdir.name
    t = [f"# {g}", "", GROUPS.get(g, ""), "", "| folder | how the car is driven | steps | test laps finished | furthest (m) | fastest lap (s) |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["folder"]):
        fin = f"{int(r['evaluations_finished'])} of {int(r['evaluations'])}" if "evaluations" in r and r["evaluations"] == r["evaluations"] else "-"
        t.append(f"| [{Path(r['folder']).name}]({Path(r['folder']).name}/README.md) | {r['how'][:110]}... | {int(r['steps']) if r.get('steps') == r.get('steps') and r.get('steps') else '-'} | {fin} | {f(r.get('best_eval_distance_m'), 0)} | {f(r.get('best_lap_s') if r.get('best_lap_s') == r.get('best_lap_s') and r.get('best_lap_s') else r.get('best_eval_lap_s'))} |")
    t += ["", "Every folder has its own README.md (what the run is, what it reached, what each file is)."]
    (gdir / "README.md").write_text("\n".join(t), encoding="utf-8")


def copy_run(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for root in ("runs12", "runs", "archive_old_runs"):
        base = a.raw / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_dir() and ((p / "library.csv").is_file() or (p / "eval.csv").is_file() or (p / "args.json").is_file()):
                runs.append(p)
    rows, index = [], []
    for run in runs:
        rel = run.relative_to(a.raw).as_posix()
        args = json.load(open(run / "args.json")) if (run / "args.json").is_file() else None
        group = group_of(args, rel)
        dst = out / group / rel.replace("/", "__")
        copy_run(run, dst)
        # the log of the run sits next to the run folder
        lg = run.with_name(run.name + ".log")
        if lg.is_file():
            shutil.copy2(lg, dst / lg.name)
        try:
            S, figs, args, ev, L = analyse(run, dst, rel)
        except Exception as e:      # an older run with another file format: keep the files, say so in the README
            S, figs = {"run": rel, "how": describe(args) + f" (the history could not be built automatically: {type(e).__name__}: {e})"}, []
            print("  history failed for", rel, e, flush=True)
        write_readme(dst, S, figs, rel, group)
        S.update(group=group, folder=f"{group}/{dst.name}")
        rows.append(S)
        print(rel, "->", group, "|", S.get("best_lap_s"), S.get("best_eval_distance_m"), flush=True)
    pd.DataFrame(rows).drop(columns=["how"]).to_csv(out / "all_runs_summary.csv", index=False)
    pd.DataFrame(rows)[["group", "folder", "how"]].to_csv(out / "all_runs_how.csv", index=False)
    for g in sorted({r["group"] for r in rows}):
        write_group_readme(out / g, [r for r in rows if r["group"] == g])
    # the scripts the runs were started with
    ls = out / "launch_scripts"
    ls.mkdir(exist_ok=True)
    for p in sorted(a.raw.iterdir()):
        if p.is_file():
            shutil.copy2(p, ls / p.name)
    (ls / "README.md").write_text("# launch_scripts\n\nThe files that sat next to the runs on the pod: launch_*.sh (the exact training commands, with every setting), run_loop.sh (restarts a run that crashes, until the end time), stop_old.sh / stop_all.py (clean stop right after a save), patch*.py (small code changes made on the pod), diag*.py and res_actions.py / rl_vs_base.py (diagnosis scripts), watchdog.py (stops a pod at its end time), *.log (set-up and test logs), end_epoch.txt (the end time of the last session, 2026-11-17 08:00 UTC; the runs were stopped earlier, on 2026-11-16 at about 17:07 UTC).\nThe training code itself is the style-rl repository (scripts/train12.py, style_rl/style12_env.py).\n", encoding="utf-8")
    best = {r["run"]: r for r in rows}
    ra, rs = best.get("runs12/res1/attacker", {}), best.get("runs12/res1/smooth", {})

    def line(r, nm):
        if not r:
            return ""
        return (f"| {nm} | {r['driver_run_lap_s']:.2f} s | {r['best_in_budget_lap_s']:.2f} s ({r['gain_in_budget_s']:.2f} s faster, D {r['best_in_budget_D']:.2f}) | {r['best_lap_s']:.2f} s ({r['gain_best_s']:.2f} s faster, D {r['best_lap_D']:.2f}) | {r['laps_faster_than_driver_run']} of {r['laps_on_track_from_line']} |")
    t = ["# Everything copied from the pods (2026-11-16)", "",
         "All training runs of the 12-number style project, copied from the RunPod network volume to this laptop, with a README in every folder. The two pods were stopped after the copy (status EXITED); the volume was not touched.", "",
         "## How to start", "",
         "1. `all_runs_summary.csv`: one line per run (steps, test laps finished, furthest distance, best lap, best lap inside the style budget).",
         "2. Open a run folder, read its `README.md`, look at `progress.png` (lap time and style distance along the training) and `style_records.png` (the 12 style numbers of every new record lap, raw values).",
         "3. The improvement over the training is in the run folder: `records_fastest.csv` (every time a new best lap time was set), `records_in_budget.csv` (the same, only laps that keep the driver's style), `history_laps.csv` (every finished lap in the order it was driven, with the 12 style numbers and their change against the driver's run) and `history_evaluations.csv` (the test lap every 20k steps, with the best so far).", "",
         "## The two kinds of RL", "",
         "- `01_RL_on_top_of_pure_pursuit/`: pure pursuit drives, RL adds a bounded correction. These finish laps and beat the driver's run.",
         "- `02_RL_from_scratch/`: RL drives alone. None finished a lap; they leave the track at the first chicane.",
         "- `03_RL_started_from_pure_pursuit/` and `05_older_runs/`: earlier, short experiments.", "",
         "## The result in one table (laps from the line, on the track)", "",
         "The style budget is D <= the driver's own D + 0.05 (attacker 0.78, smooth 1.11), the same rule as in the lap comparison.", "",
         "| driver | driver's run | fastest lap inside the style budget | fastest lap, any style | laps faster than the driver's run |", "|---|---|---|---|---|", line(ra, "attacker"), line(rs, "smooth"), "",
         "RL from scratch: the furthest any run got was about 1000 m of the 5756 m lap (stage1b/ped_steer 998 m), apart from `s12fix/attacker` (2019 m, early run).", "",
         "## What the style distance D is", "",
         "D is the distance between the 12 style numbers of a lap and the driver's style, averaged over the 12 numbers, each difference divided by how much that number normally varies between laps; 0 = the same style. The driver's own run has D 0.73 (attacker) and 1.06 (smooth) because the driver's run is a pure-pursuit replay of its line, not equal to the style numbers.", "",
         "## What is NOT copied (it is still on the network volume, which keeps its data while the pods are stopped)", "",
         "- `latest_buffer.pkl` in every run: the replay buffer (170 MB per run, 3.5 GB in total), training state only.",
         "- `checkpoints/step_<n>.zip`: the saved agent at every 20k steps (4 GB in total). The last agent of each run, `latest.zip`, is here. Two packed archives of the step agents of the four runs of the last session (policies_res1.tar 1.3 GB, policies_stage1b.tar 0.7 GB) wait on the volume in /workspace/export_all_1116/ (network volume 5hl0r6pold). They were not transferred because the transfer service kept dropping large files.", "",
         "## Folders", "", *[f"- [{g}]({g}/README.md)" for g in sorted({r['group'] for r in rows})], "- [launch_scripts](launch_scripts/README.md)", "",
         "Made by tools/export_run_history.py (style-rl repository)."]
    (out / "README.md").write_text("\n".join(t), encoding="utf-8")


if __name__ == "__main__":
    main()
