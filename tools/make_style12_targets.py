"""Targets for the whole-lap 12-number style training.

Run from the research repository `jad-race` (it holds the driver model):

    cd jad-race
    python ../style-rl/tools/make_style12_targets.py

Writes (into style-rl/targets):
  neutral_monza.npz     the driver-independent line and speed profile of the track on the OCD car: the driver model with every style knob at 0 and high skill
                        (minimum-curvature line, car-limit speeds, feed-forward pedals). The agent sees THIS line, never the driver's own.
  style12.json          for each driver: the 12 style numbers of the driver's laps on the OCD-limit car (median over 5 model laps) = the target the agent is given as input,
                        for the neutral driver: the same numbers (the starting guess of the running estimate), and the spread of every number between real driver-track cells (the unit of the distance).
The base driver of the old runs (pure pursuit of the driver's own line and pedals) is still built from <driver>_monza.npz: it is the driver's lap in the simulator, the thing the new laps are compared with.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(HERE))
from environment import synthetic_driver as sd  # noqa: E402
from environment.reference_lap import Envelope  # noqa: E402
from scripts.ocd_targets import twin_lap  # noqa: E402
from style_rl.dstyle.features import KEYS, style_vector  # noqa: E402
from style_rl.dstyle.track import Track  # noqa: E402

SEEDS = range(300, 305)


def main():
    j = json.load(open(HERE / "data/ocd_envelope.json"))
    env = Envelope("ocd_measured", np.array(j["lat_g"]), np.array(j["brk_g"]), np.array(j["acc_ms2"]), v_cap=float(j["v_cap"]), note="measured in OCD")
    kd = float(j["kd"])
    man = json.load(open(HERE / "targets/manifest.json"))
    tm = sd.TrackModel("monza")
    track = Track(HERE / "data/tracks/monza_geometry.csv")
    corners = track.find_corners()

    neutral = sd.Driver("neutral", {k: 0.0 for k in sd.KNOBS}, 0.95)
    tw = twin_lap(tm, env, kd, neutral, 11)
    np.savez_compressed(HERE / "targets/neutral_monza.npz", s=tm.s, n=tw["n"], xy=tw["xy"], v=tw["v"], thr=tw["thr"], brk=tw["brk"], t=tw["t"], length=tm.length)
    man["neutral"] = {"knobs": neutral.z, "skill": neutral.skill, "lap_time_f317_model_s": float(tw["lap_time"]) / 1000.0 if tw["lap_time"] > 1000 else float(tw["lap_time"]),
                      "description": "Driver-independent line and speeds: every style knob at 0, skill 0.95", "car_envelope": "measured in OCD"}
    print(f"neutral: model lap time {man['neutral']['lap_time_f317_model_s']:.2f} s, top speed {tw['v'].max() * 3.6:.0f} km/h", flush=True)
    json.dump(man, open(HERE / "targets/manifest.json", "w"), indent=1)

    spread = pd.read_csv("exp_B/style_numbers_by_driver_and_track.csv").query("source == 'real'")[KEYS].std(ddof=1)
    out = {"keys": KEYS, "spread": {k: float(spread[k]) for k in KEYS}, "drivers": {}}
    for name, drv in (("attacker", sd.Driver("attacker", man["attacker"]["knobs"], man["attacker"]["skill"])), ("smooth", sd.Driver("smooth", man["smooth"]["knobs"], man["smooth"]["skill"])), ("neutral", neutral)):
        rows = []
        for seed in SEEDS:
            lap = sd.drive_lap(tm, env, kd, drv, seed=seed)
            df = pd.DataFrame({"time_s": lap["currentTime"].to_numpy(float) * 1000.0, "throttle": lap["accStatus"], "brake": lap["brakeStatus"], "steer_deg": lap["steerAngle"],
                               "x": lap["world_position_x"], "y": lap["world_position_y"]})
            rows.append(style_vector(df, track, lap_dist=lap["LapDist"].to_numpy(float), corners=corners))
        med = pd.DataFrame(rows).median()
        out["drivers"][name] = {k: float(med[k]) for k in KEYS}
        print(name, {k: round(float(med[k]), 3) for k in KEYS}, flush=True)
    sp = np.array([out["spread"][k] for k in KEYS])
    for a in ("attacker", "smooth"):
        d = np.mean(np.abs(np.array([out["drivers"][a][k] for k in KEYS]) - np.array([out["drivers"]["neutral"][k] for k in KEYS])) / sp)
        print(f"distance {a} to neutral: {d:.2f} spread units", flush=True)
    d = np.mean(np.abs(np.array([out["drivers"]["attacker"][k] for k in KEYS]) - np.array([out["drivers"]["smooth"][k] for k in KEYS])) / sp)
    print(f"distance attacker to smooth: {d:.2f}", flush=True)
    json.dump(out, open(HERE / "targets/style12.json", "w"), indent=1)


if __name__ == "__main__":
    main()
