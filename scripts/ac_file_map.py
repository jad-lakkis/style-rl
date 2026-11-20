"""Which file of a (possibly name-scrambled) Assetto Corsa car data folder holds what: one table, written as markdown.

    python scripts/ac_file_map.py PATH/TO/UNPACKED/DATA [--out data/ac_sf70h/FILE_MAP.md]

For every .ini / .lut file: the name it is stored under, what is inside, the name it should have, and how sure that is.
  sure      the contents say it themselves (a section only that file has) or the file is the only one of its kind
  probable  an argument from the numbers (limits of a controller, a shape, the aero balance), not a proof
  family    the file is one of several of the same kind (the seven ERS modes, ten tyre wear curves ...) and the individual one cannot be told
The aero curves, the engine curves and the tyre force curves come from the same code the replica uses (style_rl/accar/aero_curves.py, ac_files.py, to_ocd.py).
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.accar.ac_files import STANDARD_NAMES, AcCar  # noqa: E402
from style_rl.accar.aero_curves import AeroCurveError, find_aero_curves  # noqa: E402
from style_rl.accar.ini import parse_ini  # noqa: E402

INI_BY_SECTION = [                       # (a section or key that only this file has, true name, what it is)
    ("MIRROR_0", "mirrors.ini", "mirror position and size"),
    ("NODES", "proview_nodes.ini", "list of model nodes"),
    ("COCKPIT_HR", "lods.ini", "levels of detail of the 3D model"),
    ("COLLIDER_0", "colliders.ini", "collision boxes"),
    ("SKIDS", "sounds.ini", "sound settings (skids, wind, backfire)"),
    ("KINETIC", "ers.ini", "ERS (electric motor): energy per lap, torque curves"),
    ("SCRATCHES", "damage.ini", "damage model"),
    ("DASH_CAM", "dash_cam.ini", "dashboard camera"),
    ("FUEL_EVAL", "fuel_cons.ini", "fuel consumption: kilometres per litre"),
    ("LIGHT_0", "lights.ini", "light definitions"),
    ("CAMERA_0", "cameras.ini", "camera positions"),
    ("ITEM_0", "digital_instruments.ini", "dashboard display items"),
    ("ANIMATION_0", "wing_animations.ini", "wing animation"),
    ("STEER_ANIMATION", "driver3d.ini", "driver model and animations"),
    ("WING_7", "drs.ini", "DRS: which wing it acts on"),
]


def classify_ini(car, name, text, sections):
    for role, src in car.source.items():
        if src == name:
            return STANDARD_NAMES[role], f"the {role} data of the car", "sure"
    for key, true, what in INI_BY_SECTION:
        if key in sections:
            return true, what, "sure" if key in ("MIRROR_0", "COCKPIT_HR", "COLLIDER_0", "SKIDS", "KINETIC", "SCRATCHES", "FUEL_EVAL", "CAMERA_0", "DASH_CAM") else "probable"
    if "CONTROLLER_0" in sections:
        m = re.search(r"ers(\w+?)_gas\.lut", text)
        if m:
            return f"ctrl_ers_N.ini (the ERS mode that uses ers{m.group(1)}_*.lut)", "ERS deployment controller of one mode", "family"
        if "INPUT=RPMS" in text.replace(" ", ""):
            return "ctrl_turbo0.ini", "turbo controller (rpm to boost)", "probable"
        if "gear_start" in text:
            return "ctrl_single_lock.ini", "differential lock controller", "sure"
    if "SETTINGS" in sections:
        return "ambient_shadows.ini or a camera file", "a small [SETTINGS] file (" + ", ".join(sections["SETTINGS"]) + ")", "unknown"
    if text.lstrip().startswith(";") and "WHEEL INDEXES" in text:
        return "escmode.ini (or another small file)", "a list of wheel indexes", "unknown"
    return "?", "an ini file I could not place", "unknown"


def classify_curve(stored, c, hdr, roles, engine, tyre):
    if stored in roles:
        r = roles[stored]
        return r["role"], "aero curve" if "AOA" in r["role"] or "height" in r["role"] else "wing controller table", r["confidence"]
    for key, true in (("engine", "power.lut"), ("kers", "kers_torque.lut"), ("coast", "kers_torque_coast.lut")):
        if engine.get(key) is not None and stored == engine[key]:
            return true, {"engine": "engine torque curve (rpm to Nm)", "kers": "electric motor torque (rpm to Nm)", "coast": "electric motor coast torque"}[key], "probable"
    if stored in tyre:
        return tyre[stored][0], tyre[stored][1], tyre[stored][2]
    x0, x1, y0, y1, n = c[0, 0], c[-1, 0], c[:, 1].min(), c[:, 1].max(), len(c)
    if 15 <= x1 <= 40 and x0 == 0 and 70 <= y0 and y1 <= 100.5:
        return "slicks*_front/rear.lut", f"tyre wear curve (grip % against distance, to {x1:g} km): one of ten, the compound is not told", "family"
    if x1 == 300 and x0 == 0 and 0.55 <= y0 and y1 <= 1.0 + 1e-9 and n >= 8:
        hot = c[c[:, 1] >= 0.999, 0]
        return "tcurve*.lut", f"tyre temperature curve (best grip {hot.min():g} to {hot.max():g} degrees): one of five, a harder tyre works hotter", "family"
    if n == 11 and x1 == 1 and np.allclose(c[:, 1], 0):
        return "ersN_brake.lut", "ERS brake curve (all zero)", "family"
    if n == 11 and x1 == 1 and abs(c[0, 1] + 0.005) < 1e-9:
        return "ersN_gas.lut (or throttle.lut)", "a throttle-pedal curve (0 to 1)", "family"
    if n == 9 and x1 == 8:
        return "ersN_gear.lut / difflock_gear_mult.lut / gear_start.lut", "a table over the gears 0 to 8", "family"
    if x0 == 0 and x1 == 1 and n <= 4:
        return "difflock_brake_mult.lut (or an unused wing_controller_gas.lut)", "a short table over the throttle pedal (0 to 1)", "family"
    if x0 == 0 and 100 <= x1 <= 340 and n <= 11:
        return "ersN_speed.lut", "an ERS table over speed (km/h)", "family"
    if x0 == 0 and x1 == 100 and y1 == 100:
        return "throttle.lut", "pedal percent to throttle percent", "probable"
    return "?", "a curve I could not place", "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    car = AcCar(a.folder)
    roles = {}
    try:
        _, notes = find_aero_curves(car)
        for n in notes:
            roles[n["stored_in"]] = n
    except AeroCurveError as e:
        print("aero curves not identified:", e)
    engine_curves, _ = car.engine_curves()
    engine = {k: next((n for n, c in car.curves.items() if c is v), None) for k, v in engine_curves.items()}
    tyre = {}
    heads = {n: h for n, h in car.curve_headers.items() if {"C", "E", "CF"} <= set(h)}
    big = {n: c for n, c in car.curves.items() if len(c) == 1000}
    slopes = {n: float((c[1, 1] - c[0, 1]) / (c[1, 0] - c[0, 0])) for n, c in big.items()}
    for n in big:
        if n in heads and len(heads) == 2:
            lat = max(heads, key=lambda k: slopes[k])
            tyre[n] = ("modern_F1.lut or modern_F1_front.lut", "tyre force curve of the car (header [VALUES] C E CF), CF " + f"{slopes[n]:.0f}: " + ("lateral, front and rear (peak where FRICTION_LIMIT_ANGLE minus the flex says)" if n == lat else "longitudinal"), "probable")
        else:
            tyre[n] = ("one of modern_gtslicks_tyres / modern_street_tyres / evotek_tires", f"stock tyre force curve of the game (no header, CF {slopes[n]:.0f}); this car does not use it", "family")
    rows = []
    files = sorted((p for p in car.folder.iterdir() if p.is_file() and p.suffix.lower() in (".ini", ".lut")), key=lambda p: p.name.lower())
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        size = p.stat().st_size
        if size == 0:
            rows.append((p.name, "nothing (0 bytes)", "one of the ERS gear tables (an empty file in the car's data)", "family"))
            continue
        if p.name in car.curves:
            true, what, conf = classify_curve(p.name, car.curves[p.name], car.curve_headers.get(p.name), roles, engine, tyre)
        else:
            true, what, conf = classify_ini(car, p.name, text, parse_ini(text))
        rows.append((p.name, what, true, conf))
    out = ["# What each file of the folder really contains", "", f"Folder: `{car.folder}`. File names {'do NOT match' if car.scrambled else 'match'} the contents. {len(rows)} files.", "",
           "| stored under the name | what is inside | the name it should have | how sure |", "|---|---|---|---|"]
    out += [f"| `{n}` | {w} | `{t}` | {c} |" for n, w, t, c in rows]
    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    out += ["", "How many files per certainty: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
