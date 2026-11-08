"""Rebuild the reference laps of the two artificial drivers for a given car envelope.

The driver model (environment/synthetic_driver.py, with the racing-line optimiser and the track data) lives in the research repository `jad-race`, so run this from there:

    cd jad-race
    python ../style-rl/tools/make_targets.py --envelope ../style-rl/data/ocd_envelope.json --out ../style-rl/targets
    python ../style-rl/tools/make_targets.py --old-car --out /tmp/check      # the original F317 envelope (theoretical grip), to check that the recipe reproduces the old targets

The envelope (lateral g, braking g with drag, acceleration, drag coefficient, top speed against speed) is measured in OCD by scripts/measure_ocd_envelope.py.
The drivers keep their style settings (knobs) and skill from targets/manifest.json; only the car they drive changes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from environment import synthetic_driver as sd  # noqa: E402
from environment.reference_lap import Envelope  # noqa: E402
from scripts.ocd_targets import twin_lap  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope", type=Path, default=HERE / "data/ocd_envelope.json")
    ap.add_argument("--old-car", action="store_true", help="use the original F317 envelope of the research repository instead of the measured one")
    ap.add_argument("--manifest", type=Path, default=HERE / "targets/manifest.json")
    ap.add_argument("--out", type=Path, default=HERE / "targets")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--margin", type=float, default=1.0, help="fraction of the measured lateral and braking limits the reference may use (a controller cannot use the last few percent in transitions)")
    args = ap.parse_args()
    if args.old_car:
        env, kd = sd.load_car()
    else:
        j = json.load(open(args.envelope))
        env = Envelope("ocd_measured", args.margin * np.array(j["lat_g"]), args.margin * np.array(j["brk_g"]), np.array(j["acc_ms2"]), v_cap=float(j["v_cap"]), note="measured in the OCD simulation (scripts/measure_ocd_envelope.py)")
        kd = float(j["kd"])
    manifest = json.load(open(args.manifest))
    tm = sd.TrackModel("monza")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("attacker", "smooth"):
        m = manifest[name]
        tw = twin_lap(tm, env, kd, sd.Driver(name, m["knobs"], m["skill"]), args.seed)
        np.savez_compressed(args.out / f"{name}_monza.npz", s=tm.s, n=tw["n"], xy=tw["xy"], v=tw["v"], thr=tw["thr"], brk=tw["brk"], t=tw["t"], length=tm.length)
        m["lap_time_f317_model_s"] = float(tw["lap_time"])
        m["car_envelope"] = "original F317 envelope (theoretical grip)" if args.old_car else f"measured in OCD, lateral and braking limits x {args.margin}"
        print(f"{name}: lap time {tw['lap_time']:.2f} s, top speed {tw['v'].max() * 3.6:.0f} km/h, peak brake {tw['brk'].max():.2f}, min speed {tw['v'].min() * 3.6:.0f} km/h", flush=True)
    json.dump(manifest, open(args.out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
