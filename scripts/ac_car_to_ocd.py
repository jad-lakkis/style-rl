"""Rebuild an Assetto Corsa car as an OCD car:

    python scripts/ac_car_to_ocd.py PATH/TO/UNPACKED/DATA [--out data/ac_sf70h/replica.json] [--report data/ac_sf70h/REPORT.md] [--fuel-l 55] [--compound 1] [--turbo ac] [--ers]

Reads the unpacked data.acd of the car (the files are recognised by their contents, so a folder whose file names are shifted against the contents still works), converts every number OCD can use
(style_rl/accar/to_ocd.py) and writes
  the replica json     all OCD parameter overrides, the CarSpec fields, the powertrain data; read by the backend `ocdlap:ac_sf70h` (path in AC_REPLICA_JSON)
  the report           a markdown table: AC value -> OCD parameter -> OCD value -> status (exact / converted / fitted / estimated / placeholder / not modelled)
The json and the report hold numbers from the game's data files: keep them out of public repositories (data/ac_sf70h/ is in .gitignore).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.accar.ac_files import AcCar  # noqa: E402
from style_rl.accar.to_ocd import build_replica, report_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ac_sf70h/replica.json")
    ap.add_argument("--report", type=Path, default=None, help="default: REPORT.md next to the json")
    ap.add_argument("--fuel-l", type=float, default=None, help="fuel in the tank (default: the car's own default, 55 L)")
    ap.add_argument("--compound", type=int, default=None, help="tyre compound index (default: the car's default, index 1 = SuperSoft for the SF70H)")
    ap.add_argument("--turbo", choices=["ac", "steady", "linear"], default="ac", help="the default of the backend option turbo (see style_rl/accar/powertrain.py)")
    ap.add_argument("--ers", action="store_true", help="make the electric motor the default (the backend option ers=1 turns it on at run time either way)")
    a = ap.parse_args()
    car = AcCar(a.folder)
    for w in car.warnings:
        print("WARNING:", w)
    replica = build_replica(car, fuel_l=a.fuel_l, compound=a.compound, ers=a.ers, turbo=a.turbo)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(replica, indent=1, default=float))
    report = a.report or a.out.with_name("REPORT.md")
    report.write_text(report_markdown(replica), encoding="utf-8")
    o = replica["ocd"]
    P = "vehicle_dynamics_double_track."
    print(f"{replica['meta']['folder']}: {'file names do NOT match the contents (recognised by content)' if replica['meta']['scrambled'] else 'file names match'}")
    print(f"mass {o[P + 'mass_vehicle_kg']:.1f} kg, wheelbase {o[P + 'wheelbase_m']:.3f} m, CG {o[P + 'cog.distance_from_front_axle_m']:.3f} m behind the front axle, {o[P + 'cog.height_m']:.3f} m high")
    print(f"{len(replica['ocd'])} OCD parameters; {sum(r['status'] == 'placeholder' for r in replica['report'])} placeholders, {sum(r['status'] in ('estimated', 'assumed') for r in replica['report'])} estimates")
    print(f"written {a.out} and {report}")


if __name__ == "__main__":
    main()
