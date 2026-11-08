"""Compare the trained agents with the original driver:   python scripts/compare_runs.py --runs runs/v2 [--plot]

For every agent folder (<driver>_<mode>) under --runs, from eval.csv / records.csv / baseline.json:
  - the original driver's lap in the simulator (the base driver alone) and the reference lap time,
  - the agent's lap time and style error at the first, best and last evaluation,
  - the records: every new fastest lap that is faster than the base driver and keeps the style (checkpoint file named),
  - with --plot: runs/<...>/compare.png (lap time and style error against training steps, original driver as horizontal lines).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    dirs = sorted(p for p in args.runs.iterdir() if (p / "eval.csv").is_file())
    if not dirs:
        print("no evaluations yet in", args.runs)
        return
    fig = None
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(dirs), figsize=(4.2 * len(dirs), 6.5), squeeze=False)
    for k, d in enumerate(dirs):
        ev = pd.read_csv(d / "eval.csv")
        base = json.loads((d / "baseline.json").read_text()) if (d / "baseline.json").is_file() else {}
        if d.name.endswith("_pure"):          # a pure agent has no base driver: its original driver is the base driver of the residual run with the same driver model
            sib = d.parent / (d.name[: -len("_pure")] + "_residual") / "baseline.json"
            if sib.is_file():
                base = json.loads(sib.read_text())
                ev = ev.assign(base_lap_time=base.get("lap_time"), delta_vs_base=ev.lap_time - (base.get("lap_time") or np.nan))
        ref_t = float(ev.reference_lap_time.iloc[0])
        b_t, b_e = base.get("lap_time"), base.get("style_error")
        done = ev[ev.completed]
        print(f"\n== {d.name}: {len(ev)} evaluations, {int(ev.step.iloc[-1]):,} steps, {ev.hours.iloc[-1]:.2f} h")
        print(f"   original driver: reference lap {ref_t:.2f} s; base driver in the simulator " + (f"{b_t:.2f} s, style error {b_e:.2f}" if b_t is not None and b_e is not None else "did not finish a lap"))
        print(f"   laps finished: {len(done)} of {len(ev)} evaluations")
        if len(done):
            f, l, bst = done.iloc[0], done.iloc[-1], done.loc[done.lap_time.idxmin()]
            for name, r in (("first finished", f), ("fastest", bst), ("last", l)):
                print(f"   {name:15s} step {int(r.step):>9,}: lap {r.lap_time:7.2f} s ({r.delta_vs_reference:+.2f} against the reference), style error {r.style_error:.2f}, off-track {100 * r.off_track_share:.1f}%")
        rec = d / "records.csv"
        if rec.is_file():
            r = pd.read_csv(rec)
            print(f"   records (faster than the base driver, style kept): {len(r)}; best {r.lap_time.min():.2f} s ({r.delta_vs_base.min():+.2f} s against the base driver) in {r.loc[r.lap_time.idxmin(), 'checkpoint']}")
        else:
            print("   records: none yet (no finished lap faster than the base driver with the style kept)")
        if fig is not None:
            ax1, ax2 = fig.axes[k], fig.axes[len(dirs) + k]
            ax1.plot(done.step, done.lap_time, "o-", label="agent")
            ax1.axhline(ref_t, color="k", ls="--", label="reference lap")
            if b_t is not None:
                ax1.axhline(b_t, color="g", ls=":", label="base driver")
            ax1.set_title(d.name)
            ax1.set_ylabel("lap time (s)")
            ax1.legend(fontsize=7)
            ax2.plot(done.step, done.style_error, "o-", color="C1")
            if b_e is not None:
                ax2.axhline(b_e, color="g", ls=":")
            ax2.set_ylabel("style error (lower = closer to the driver)")
            ax2.set_xlabel("training steps")
    if fig is not None:
        fig.tight_layout()
        out = args.runs / "compare.png"
        fig.savefig(out, dpi=130)
        print("\nwrote", out)


if __name__ == "__main__":
    main()
