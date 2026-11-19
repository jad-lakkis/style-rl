"""Replay page of one lap of a drive run against the pure-pursuit runs of the drivers: the same page as tools/lap_report.py (3D chase camera, 2D map, sector view, speed traces,
sector and style tables), with the RL lap as the blue car and the chosen driver's run as the grey car.

    python tools/drive_replay.py RUN_DIR OUT_HTML --step 950000 --lap 2 \
        --driver attacker=PATH/baseline_pure_pursuit_driver.csv --driver smooth=PATH/baseline_pure_pursuit_driver.csv --style-csv PATH/style_numbers.csv

RUN_DIR holds laps/eval_step_N.npz (scripts/drive_train.py). --style-csv is the style_numbers.csv written by tools/drive_style_compare.py (the 12 numbers of the laps).
The page is one self-contained HTML file (no network needed). Segments are the straights and one segment per corner (as in drive_style_compare.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from drive_run_report import lap_from_trace  # noqa: E402
from drive_style_compare import REAL_NAMES, segments  # noqa: E402
from style_rl.dstyle.features import KEYS, STYLE  # noqa: E402
from style_rl.dstyle.track import Track  # noqa: E402

TEMPLATE = (ROOT / "tools" / "replay_template.html").read_text(encoding="utf-8")
PLAIN = {s[0]: (s[1], s[4]) for s in STYLE}
DT = 0.1


def rl_lap(run, step, lap, track):
    L = track.length_m
    z = np.load(run / "laps" / f"eval_step_{step}.npz")
    ss, tt, lap_time, idx = lap_from_trace(z, L, lap - 1)
    xy = np.column_stack([z["x"][idx], z["y"][idx]])
    q = track.project(xy, s_hint=z["station_m"][idx])
    df = pd.DataFrame({"t": tt[1:-1], "x": xy[:, 0], "y": xy[:, 1], "speed": z["speed"][idx], "progress": np.maximum.accumulate(q["progress_m"]),
                       "throttle": z["throttle"][idx], "brake": z["brake"][idx]})
    return df, float(lap_time)


def driver_lap(path):
    d = pd.read_csv(path)
    df = d[["t", "x", "y", "speed", "progress", "throttle", "brake"]].copy()
    df["progress"] = np.maximum.accumulate(df.progress.to_numpy(float))
    return df, float(df.t.iloc[-1])


def pack(df, lap_time):
    tt = np.arange(0.0, float(df.t.iloc[-1]) + DT, DT)
    f = lambda c: np.interp(tt, df.t, df[c])      # noqa: E731
    return {"x": np.round(f("x"), 1).tolist(), "y": np.round(f("y"), 1).tolist(), "v": np.round(f("speed"), 1).tolist(), "p": np.round(f("progress"), 1).tolist(),
            "a": np.round(f("throttle"), 2).tolist(), "b": np.round(f("brake"), 2).tolist(), "lap_time": lap_time}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--lap", type=int, default=2)
    ap.add_argument("--driver", action="append", required=True, help="name=path of a driver run csv (old library format)")
    ap.add_argument("--style-csv", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, default=ROOT / "data/tracks/monza_geometry.csv")
    ap.add_argument("--corners", type=Path, default=ROOT / "data/monza_corners.csv")
    a = ap.parse_args()
    geo = pd.read_csv(a.geometry)
    track = Track(a.geometry)
    sty = pd.read_csv(a.style_csv)
    assert sty.number.tolist() == KEYS
    rl_df, rl_time = rl_lap(a.run, a.step, a.lap, track)
    tr = {"s": np.round(geo.s_geom.to_numpy(), 1).tolist(), "xl": np.round(geo.x_left.to_numpy(), 1).tolist(), "yl": np.round(geo.y_left.to_numpy(), 1).tolist(),
          "xr": np.round(geo.x_right.to_numpy(), 1).tolist(), "yr": np.round(geo.y_right.to_numpy(), 1).tolist()}
    drivers, labels = {}, {}
    for spec in a.driver:
        name, path = spec.split("=", 1)
        drv_df, drv_time = driver_lap(path)
        laps = {"driver": drv_df, "keep": rl_df}
        L = float(min(drv_df.progress.iloc[-1], rl_df.progress.iloc[-1]))
        segs = segments(L, a.corners)
        sect = [{"name": (f"{n} {REAL_NAMES[n]}" if kind == "corner" else f"{n} straight"), "kind": "corners" if kind == "corner" else "straight", "a": float(lo), "b": float(min(hi, L))} for n, kind, lo, hi in segs]
        zones = []
        for s in sect:
            if s["kind"] == "corners":
                j = int(np.argmin(np.abs(geo.s_geom - (s["a"] + s["b"]) / 2)))
                zones.append({"name": s["name"].split(" ")[0], "a": s["a"], "b": s["b"], "x": float(geo.x_center[j]), "y": float(geo.y_center[j])})
        style = {"keys": KEYS, "plain": [PLAIN[k][0] for k in KEYS], "unit": [PLAIN[k][1] for k in KEYS], "vals": {"driver": sty[f"{name}_run"].round(4).tolist(), "keep": sty.rl_command.round(4).tolist()}}
        vmax = float(np.ceil(max(drv_df.speed.max(), rl_df.speed.max()) / 10.0) * 10.0)
        drivers[name] = {"dt": DT, "L": L, "vmax": vmax, "laps": {"driver": pack(drv_df, drv_time), "keep": pack(rl_df, rl_time)}, "zones": zones, "sectors": sect, "style": style, "brief": None}
        labels[name] = f"{name}'s run, {drv_time:.2f} s"
    data = {"track": tr, "drivers": drivers, "labels": labels, "nobrief": True, "drvlabel": "Grey car =",
            "title": f"Fastest RL lap on Monza ({rl_time:.2f} s) against the pure-pursuit drivers",
            "names": {"driver": "driver's run (pure pursuit)", "keep": f"RL fastest lap, {rl_time:.2f} s"}, "short": {"driver": "driver's run", "keep": "RL fastest lap"},
            "about": f"Blue: the fastest lap of run a1 (pure RL, step {a.step:,}, lap {a.lap} of the test, {rl_time:.2f} s, stayed on the track), trained for lap time only: no driver style has been applied yet. "
                     "Grey: the pure-pursuit run of the driver you choose at the top. Monza, Dallara F317 in the OCD simulator. Press Play; use the 3D chase camera or the 2D map; click a sector row to zoom in."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("<title>Lap replay: driver's run against the RL laps</title>", f"<title>{data['title']}</title>", 1)
    a.out.write_text(html, encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e3:.0f} kB): RL {rl_time:.2f} s, drivers {', '.join(labels.values())}")


if __name__ == "__main__":
    main()
