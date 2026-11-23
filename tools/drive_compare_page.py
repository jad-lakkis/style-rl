"""One HTML page for a drive run: the lap times against the real laps of the car's season and against an older run, and a replay of the best lap against the limit lap.

    python tools/drive_compare_page.py OUT_HTML --run RAW_A2_DIR --step 650000 --reliable-step 750000 \
        --old-run RAW_A1_DIR --old-step 950000 --envelope data/ac_sf70h/ocd_envelope.json --margin 2.0

The page has (1) headline numbers, (2) a bar chart of the lap times (real 2017 Monza laps from the official timing, the replica's limit lap, the AI driver, the older car) with a table view,
(3) the replay of tools/replay_template.html with n cars: the limit lap of the replica (grey, computed from the car's measured limits, no pedals), the AI driver (blue) and the older car (orange).
One self-contained file, no network needed. Template: tools/compare_template.html. The real numbers below are from formula1.com (checked 2026-11-23) and are listed with their sources on the page.
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
from drive_replay import pack, rl_lap  # noqa: E402
from drive_style_compare import REAL_NAMES, segments  # noqa: E402
from style_rl.dstyle.track import Track  # noqa: E402
from style_rl.drive.course import Course  # noqa: E402

TEMPLATE = (ROOT / "tools" / "compare_template.html").read_text(encoding="utf-8")
DT = 0.1
OFFICIAL_LAP_M = 5793.0                      # Monza, official lap length
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6a66"      # categorical slots 1 and 2 of the reference palette, and a neutral for the computed lap

# the real numbers: formula1.com results of the 2017 Italian Grand Prix (race: fastest laps; practice 2); qualifying was wet (pole 1:35.554) and is not used
FP2_BEST = 81.406                            # Bottas, Mercedes, practice 2, 1:21.406
FP2_VETTEL = FP2_BEST + 0.140                # Vettel, Ferrari, practice 2: the page gives the gap, +0.140 s
RACE_FASTEST = 83.361                        # Ricciardo, Red Bull, lap 49, 1:23.361
RACE_VETTEL = 83.897                         # Vettel, Ferrari, lap 51, 1:23.897


def limit_lap(course):
    """The replica's limit lap as a lap trace: positions on the reference line, the speed of the QSS profile, time from the profile (scaled to its lap time)."""
    v, ds, L = course.v_qss, course.ds, course.L
    dt = ds / (0.5 * (v + np.roll(v, -1)))
    t = np.concatenate([[0.0], np.cumsum(dt)])
    t *= course.qss_lap_time_s / t[-1]
    s = np.append(np.arange(course.N) * ds, L)
    xy = np.vstack([course.ref_xy, course.ref_xy[:1]])
    df = pd.DataFrame({"t": t, "x": xy[:, 0], "y": xy[:, 1], "speed": np.append(v, v[0]), "progress": s, "throttle": 0.0, "brake": 0.0})
    return df, float(course.qss_lap_time_s)


def eval_lap2(run, step):
    for line in open(run / "evals.jsonl"):
        e = json.loads(line)
        if e["step"] == step:
            return float(e["det_lap2_s"])
    raise SystemExit(f"no evaluation at step {step} in {run}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--run", type=Path, required=True, help="the run (folder with evals.jsonl and laps/)")
    ap.add_argument("--step", type=int, required=True, help="step of its best clean lap")
    ap.add_argument("--reliable-step", type=int, default=None, help="step of the best lap that also passed the reliability check")
    ap.add_argument("--old-run", type=Path, required=True)
    ap.add_argument("--old-step", type=int, required=True)
    ap.add_argument("--lap", type=int, default=2)
    ap.add_argument("--envelope", type=Path, required=True, help="the car's limits (json) for the limit lap")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--old-limit", type=float, default=111.85, help="limit lap of the older car on its line, s")
    ap.add_argument("--teacher", type=float, default=94.2, help="lap of the scripted driver at 90 percent of the limit, s")
    ap.add_argument("--geometry", type=Path, default=ROOT / "data/tracks/monza_geometry.csv")
    ap.add_argument("--corners", type=Path, default=ROOT / "data/monza_corners.csv")
    a = ap.parse_args()

    geo = pd.read_csv(a.geometry)
    track = Track(a.geometry)
    course = Course(a.geometry, a.envelope, margin_m=a.margin, grid_m=1.0)
    ghost_df, ghost_t = limit_lap(course)
    ai_df, ai_t = rl_lap(a.run, a.step, a.lap, track)
    old_df, old_t = rl_lap(a.old_run, a.old_step, a.lap, track)
    rel_t = eval_lap2(a.run, a.reliable_step) if a.reliable_step else None
    laps = {"driver": (ghost_df, ghost_t), "keep": (ai_df, ai_t), "old": (old_df, old_t)}
    L = float(min(df.progress.iloc[-1] for df, _ in laps.values()))
    segs = segments(L, a.corners)
    sect = [{"name": (f"{n} {REAL_NAMES[n]}" if kind == "corner" else f"{n} straight"), "kind": "corners" if kind == "corner" else "straight", "a": float(lo), "b": float(min(hi, L))} for n, kind, lo, hi in segs]
    zones = []
    for s in sect:
        if s["kind"] == "corners":
            j = int(np.argmin(np.abs(geo.s_geom - (s["a"] + s["b"]) / 2)))
            zones.append({"name": s["name"].split(" ")[0], "a": s["a"], "b": s["b"], "x": float(geo.x_center[j]), "y": float(geo.y_center[j])})
    vmax = float(np.ceil(max(df.speed.max() for df, _ in laps.values()) / 10.0) * 10.0)
    tr = {"s": np.round(geo.s_geom.to_numpy(), 1).tolist(), "xl": np.round(geo.x_left.to_numpy(), 1).tolist(), "yl": np.round(geo.y_left.to_numpy(), 1).tolist(),
          "xr": np.round(geo.x_right.to_numpy(), 1).tolist(), "yr": np.round(geo.y_right.to_numpy(), 1).tolist()}
    drivers = {"a2": {"dt": DT, "L": L, "vmax": vmax, "laps": {k: pack(df, t) for k, (df, t) in laps.items()}, "zones": zones, "sectors": sect, "style": None, "brief": None}}

    our_m = float(course.L)
    bench = [
        {"id": "fp2", "label": "Fastest dry lap of the weekend: Bottas, Mercedes (practice 2)", "value": FP2_BEST, "group": "real", "len_m": OFFICIAL_LAP_M, "ref": True, "note": "official timing, 1:21.406"},
        {"id": "fp2f", "label": "Best Ferrari lap in practice 2: Vettel (SF70H)", "value": round(FP2_VETTEL, 3), "group": "real", "len_m": OFFICIAL_LAP_M, "note": "official timing: 1:21.406 plus the +0.140 s gap"},
        {"id": "race", "label": "Fastest race lap: Ricciardo, Red Bull", "value": RACE_FASTEST, "group": "real", "len_m": OFFICIAL_LAP_M, "note": "official timing, lap 49, 1:23.361"},
        {"id": "racef", "label": "Best Ferrari race lap: Vettel (SF70H)", "value": RACE_VETTEL, "group": "real", "len_m": OFFICIAL_LAP_M, "note": "official timing, lap 51, 1:23.897"},
        {"id": "limit", "label": "Replica: the fastest lap it can physically do on our line", "value": round(ghost_t, 3), "group": "replica", "len_m": our_m, "note": f"computed from the car's measured limits, line {a.margin:.1f} m from the track edges"},
        {"id": "ai", "label": "Replica: AI driver, best clean lap", "value": round(ai_t, 3), "group": "replica", "len_m": our_m, "hl": True, "note": f"run a2, step {a.step:,}, lap {a.lap} of a flying two-lap test"},
        {"id": "teacher", "label": "Replica: scripted driver at 90 % of the limit", "value": a.teacher, "group": "replica", "len_m": our_m, "note": "the teacher the AI driver started from"},
        {"id": "oldlimit", "label": "Older car F317: the fastest lap it can physically do", "value": a.old_limit, "group": "older", "len_m": our_m, "note": "computed on a line 1.5 m from the edges"},
        {"id": "old", "label": "Older car F317: AI driver, best clean lap", "value": round(old_t, 3), "group": "older", "len_m": our_m, "note": f"run a1, step {a.old_step:,}"},
    ]
    if rel_t:
        bench.insert(6, {"id": "rel", "label": "Replica: AI driver, reliable version", "value": round(rel_t, 3), "group": "replica", "len_m": our_m, "note": f"step {a.reliable_step:,}: also clean from 4 of 4 other starting points"})
    groups = [{"id": "real", "name": "Real 2017 cars, official F1 timing", "css": "--s3"},
              {"id": "replica", "name": "Ferrari SF70H replica in our simulator (run a2)", "css": "--s1"},
              {"id": "older", "name": "Older car in our simulator: Dallara F317 (run a1)", "css": "--s2"}]
    gap_pct = 100.0 * (ai_t / ghost_t - 1.0)
    kpis = [{"label": "AI driver, best clean lap", "value": f"{ai_t:.2f} s", "note": f"{gap_pct:.1f} % above the replica's limit lap", "group": "replica"},
            {"label": "Replica's limit lap on our line", "value": f"{ghost_t:.2f} s", "note": "computed from the measured limits of the car", "group": "replica"},
            {"label": "Real Ferrari, best race lap of 2017", "value": f"{RACE_VETTEL:.2f} s", "note": "Vettel, Italian GP, lap 51", "group": "real"},
            {"label": "Older car (F317), AI driver", "value": f"{old_t:.2f} s", "note": "run a1, same track", "group": "older"}]
    notes = [
        f"Blue and orange bars come from our simulator (OCD): the AI drivers, the scripted driver and the computed limit laps. Green bars are real 2017 cars from the official F1 timing.",
        f"The replica is slower than the real car. Its own limit on our line ({ghost_t:.2f} s) is already {ghost_t - RACE_VETTEL:.1f} s slower than Vettel's best race lap and {ghost_t - FP2_BEST:.1f} s slower than the fastest dry lap of the weekend. So this AI driver cannot match a real F1 lap with this replica; what counts is how close it gets to the replica's own limit ({gap_pct:.1f} %).",
        "Real laps are not like for like: real tyres, fuel and engine modes, and a race lap is not a qualifying lap. Qualifying in 2017 was wet (pole 1:35.554), so the dry practice laps and the race laps are the benchmark.",
        f"Our Monza is {our_m:,.0f} m long, the official lap is {OFFICIAL_LAP_M:,.0f} m ({100 * (OFFICIAL_LAP_M / our_m - 1):.1f} % longer): compare the average speeds in the table view as well as the lap times.",
        "The AI driver is not reliable yet. Its best lap is its best, not its typical lap; only the 'reliable version' also stayed on the track from other starting points. It has driven only in our simulator, not in the real Assetto Corsa.",
        "The older car is the Dallara F317, a Formula 3 car; its numbers are from run a1 on the same track. The grey car in the replay is the computed limit lap, not a driver.",
    ]
    sources = [{"title": "2017 Italian Grand Prix, fastest laps (formula1.com)", "url": "https://www.formula1.com/en/results/2017/races/971/italy/fastest-laps", "what": "race: Ricciardo 1:23.361, Vettel 1:23.897"},
               {"title": "2017 Italian Grand Prix, practice 2 (formula1.com)", "url": "https://www.formula1.com/en/results/2017/races/971/italy/practice/2", "what": "Bottas 1:21.406, Vettel +0.140 s"},
               {"title": "2017 Italian Grand Prix (Wikipedia)", "url": "https://en.wikipedia.org/wiki/2017_Italian_Grand_Prix", "what": "practice and qualifying summary: wet qualifying and third practice"}]
    foot = ("<b>3D chase camera:</b> the camera follows the chosen lap from behind; every lap is a car on the road and its racing line is drawn on the asphalt. \"At the same place\" puts all cars at the same distance along the track, "
            "so only the line and the speed differ; \"at the same moment\" is a live race in which the faster laps pull away. Drag = look around the car, wheel = closer / further, double-click = reset the camera. "
            "Bottom left of the view: speed, brake (red) and throttle (green) of each lap.<br>The grey car is not a driver: it is the speed profile the replica could physically reach on this line (no pedals). "
            "\"gap\" = the seconds this lap needs more (+) or less (-) than the limit lap to reach the place where it is now (in a sector view: counted from the sector entry). "
            "The older car is a different, slower car (a Formula 3 Dallara F317), shown for scale.")
    data = {"track": tr, "drivers": drivers, "labels": {"a2": "run a2"}, "order": ["driver", "keep", "old"], "colors": {"driver": GREY, "keep": BLUE, "old": ORANGE},
            "names": {"driver": f"limit lap of the replica (computed), {ghost_t:.2f} s", "keep": f"AI driver, best clean lap, {ai_t:.2f} s", "old": f"older car, AI driver, {old_t:.2f} s"},
            "short": {"driver": "limit lap", "keep": "AI driver", "old": "older car"}, "nopedals": ["driver"],
            "txt": {"speedDiff": "speed minus the limit lap's speed at the same place, m/s", "gapDiff": "time gap to the limit lap, s (below 0 = ahead)", "lineVs": "line vs limit lap (+ left)", "speedVs": "speed vs limit lap", "ref": "limit lap"},
            "nostyle": True, "nobrief": True, "drvlabel": "Run", "cam": "keep",
            "pagetitle": "Run a2 at Monza: the AI driver against the older car and the real 2017 laps",
            "lede": f"Press Play to watch the AI driver's best clean lap ({ai_t:.2f} s, {gap_pct:.1f} % above the fastest lap the replica can physically do on our line) in 3D: drag to look around, use the wheel to zoom, or switch to the 2D map. Under the replay: how the lap times compare with the older car and the real 2017 laps.",
            "jump_text": "Jump to the lap times ↓",
            "chart_title": "Lap times at Monza, in seconds", "chart_sub": "Green: real 2017 cars from the official timing. Blue and orange: our simulator.",
            "title": "Replay: the AI driver's best lap, the replica's limit lap and the older car",
            "about": f"Blue: the AI driver's best clean lap (run a2, step {a.step:,}, lap {a.lap} of the test, {ai_t:.2f} s). Grey: the limit lap of the replica ({ghost_t:.2f} s), the speed the car could physically reach along the same line. "
                     f"Orange: the older car (Dallara F317), best clean lap of run a1 ({old_t:.2f} s). Press Play; drag to look around, wheel to zoom, click a sector row to zoom in. Untick a car in the table next to the charts to compare two cars more closely.",
            "foot": foot, "bench": bench, "groups": groups, "kpis": kpis, "notes": notes, "sources": sources}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    a.out.write_text(html, encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e3:.0f} kB): AI {ai_t:.2f} s, limit {ghost_t:.2f} s, older car {old_t:.2f} s, reliable {rel_t}, L {L:.0f} m")


if __name__ == "__main__":
    main()
