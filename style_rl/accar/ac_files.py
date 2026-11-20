"""Reads the data folder of an Assetto Corsa car (the unpacked data.acd) into one object.

The files are recognised by what is INSIDE them, not by their names: the data folder we were given (ferrari_sf70h) came out of the unpacking tool with the names and the contents shifted against
each other (the file called aero.ini holds the mirrors, the real aero.ini is called wing_controller_monkey_gas.lut, the suspension is wing_animations.ini, the tyres are sounds.ini, ...).
A correctly named folder works the same way; `AcCar.scrambled` says which case it is, and the curves (.lut) are only trusted by name when it is not scrambled.

    car = AcCar("path/to/data")
    car.section("suspensions", "FRONT")["SPRING_RATE"]
    car.engine_curves()       # (engine torque curve, mgu-k torque curve, mgu-k coast curve) as (rpm, Nm) arrays, found by name or by shape
"""
import re
from pathlib import Path

import numpy as np

from .ini import num, parse_curve, parse_ini, read_text

STANDARD_NAMES = {"car": "car.ini", "engine": "engine.ini", "drivetrain": "drivetrain.ini", "brakes": "brakes.ini", "electronics": "electronics.ini", "suspensions": "suspensions.ini",
                  "tyres": "tyres.ini", "setup": "setup.ini", "aero": "aero.ini", "ers": "ers.ini", "ai": "ai.ini"}


def _has(ini, section, *keys):
    return section in ini and all(k in ini[section] for k in keys)


SIGNATURES = {
    "car": lambda i: _has(i, "BASIC", "TOTALMASS"),
    "engine": lambda i: _has(i, "ENGINE_DATA", "LIMITER") and _has(i, "HEADER", "POWER_CURVE"),
    "drivetrain": lambda i: "GEARS" in i and "DIFFERENTIAL" in i and "TRACTION" in i,
    "brakes": lambda i: _has(i, "DATA", "MAX_TORQUE", "FRONT_SHARE"),
    "electronics": lambda i: "ABS" in i and "TRACTION_CONTROL" in i,
    "suspensions": lambda i: _has(i, "BASIC", "WHEELBASE") and "FRONT" in i,
    "tyres": lambda i: any(k.startswith("FRONT") and "DY_REF" in v for k, v in i.items()),
    "setup": lambda i: sum(1 for v in i.values() if "TAB" in v and "STEP" in v and "MIN" in v) >= 10,
    "aero": lambda i: "WING_0" in i,
    "ers": lambda i: "KINETIC" in i,
    "ai": lambda i: "LOOKAHEAD" in i,
}


class AcCar:
    def __init__(self, folder):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"{folder} is not a folder")
        self.ini, self.source, self.warnings = {}, {}, []
        self.curves, self.curve_headers = {}, {}          # file name -> (n, 2) array / header of the [VALUES] curve files
        for path in sorted(self.folder.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in (".ini", ".lut"):
                continue
            text = read_text(path)
            sections = parse_ini(text)
            if sections:
                for role, test in SIGNATURES.items():
                    if test(sections):
                        if role in self.ini:
                            self.warnings.append(f"two files look like the {role} file: {self.source[role]} and {path.name}; the first is used")
                        else:
                            self.ini[role], self.source[role] = sections, path.name
                        break
            pts, header = parse_curve(text)
            if pts is not None:
                self.curves[path.name] = pts
                if header:
                    self.curve_headers[path.name] = header
        missing = [r for r in ("car", "suspensions", "tyres", "brakes", "drivetrain", "engine") if r not in self.ini]
        if missing:
            self.warnings.append("not found in this folder: " + ", ".join(missing))

    # --- what was found where ---------------------------------------------------------------------
    @property
    def scrambled(self):
        """True when a recognised file does not carry the name it should (see the module docstring)."""
        return any(self.source[r].lower() != STANDARD_NAMES[r] for r in self.source)

    def name_map(self):
        """{role: (file it is stored in, the file name it should have)} for the files that are under another name."""
        return {r: (self.source[r], STANDARD_NAMES[r]) for r in self.source if self.source[r].lower() != STANDARD_NAMES[r]}

    def section(self, role, name):
        try:
            return self.ini[role][name]
        except KeyError:
            raise KeyError(f"{role}: no section [{name}] (found in {self.source.get(role, 'no file')})") from None

    def value(self, role, section, key, default=None):
        return num(self.ini.get(role, {}).get(section, {}).get(key), default)

    def tyre_compounds(self):
        """[{index, name, front: {...}, rear: {...}}] from [FRONT], [REAR], [FRONT_1], [REAR_1] ...; the default compound index is `tyre_default_index`."""
        t = self.ini["tyres"]
        out = []
        for key in sorted(k for k in t if re.fullmatch(r"FRONT(_\d+)?", k)):
            idx = int(key.split("_")[1]) if "_" in key else 0
            rear = t.get(f"REAR_{idx}" if idx else "REAR", {})
            out.append({"index": idx, "name": t[key].get("NAME", key), "short": t[key].get("SHORT_NAME", ""), "front": t[key], "rear": rear})
        return out

    @property
    def tyre_default_index(self):
        return int(num(self.ini["tyres"].get("COMPOUND_DEFAULT", {}).get("INDEX"), 0))

    # --- engine and electric motor curves -------------------------------------------------------------
    def engine_curves(self):
        """(engine torque, mgu-k torque, mgu-k coast torque), each an (n, 2) array of rpm | Nm. Looked up by the names the engine and ers files give (power.lut, kers_torque.lut,
        kers_torque_coast.lut) when the folder is not scrambled, otherwise found by shape among the curves with rpm on the x axis:
          engine torque    rises, peaks between 6000 and 13000 rpm and falls again
          mgu-k torque     never rises (flat at low rpm, then falling with rpm: a power limit), the one with the largest values
          mgu-k coast      the same shape, smaller
        Returns the curves and a list of strings that say how each one was found (name / shape)."""
        notes = []
        names = {"engine": self.ini.get("engine", {}).get("HEADER", {}).get("POWER_CURVE", "power.lut"), "kers": self.ini.get("ers", {}).get("KINETIC", {}).get("TORQUE_CURVE", "kers_torque.lut"),
                 "coast": self.ini.get("ers", {}).get("KINETIC", {}).get("COAST_CURVE", "kers_torque_coast.lut")}
        names = {k: v.split()[0] for k, v in names.items()}
        rpm_curves = {n: c for n, c in self.curves.items() if c[-1, 0] >= 12000 and len(c) >= 15 and c[0, 0] <= 1000}
        out = {}
        if not self.scrambled:
            for k, n in names.items():
                if n in rpm_curves:
                    out[k] = rpm_curves[n]
                    notes.append(f"{k}: file {n} (by name)")
        flat, hump = [], []
        for n, c in rpm_curves.items():
            y = c[:, 1]
            if np.all(np.diff(y) <= 1e-9) and y[0] > 0:
                flat.append((n, c))                                  # never rises: highest at low rpm and falling after, an electric motor
            else:
                i = int(np.argmax(y))
                if 6000 <= c[i, 0] <= 13000 and y[0] < 0.9 * y[i] and y[-1] < 0.7 * y[i]:
                    hump.append((n, c))                              # rises to a peak in the middle of the rev range and falls: the combustion engine
        if "engine" not in out and hump:
            n, c = max(hump, key=lambda nc: nc[1][:, 1].max())
            out["engine"] = c
            notes.append(f"engine: file {n} (by shape: torque rises to a peak and falls)")
        flat.sort(key=lambda nc: -nc[1][:, 1].max())
        if "kers" not in out and flat:
            out["kers"] = flat[0][1]
            notes.append(f"kers: file {flat[0][0]} (by shape: never rises, the larger one)")
        if "coast" not in out and len(flat) > 1:
            out["coast"] = flat[1][1]
            notes.append(f"coast: file {flat[1][0]} (by shape: never rises, the smaller one)")
        return out, notes

    def lateral_curve_header(self):
        """C, E and CF of the one tyre curve file that states them ([VALUES] C= E= CF=), or None."""
        for n, h in self.curve_headers.items():
            if {"C", "E", "CF"} <= set(h):
                return {"C": num(h["C"]), "E": num(h["E"]), "CF": num(h["CF"]), "file": n}
        return None
