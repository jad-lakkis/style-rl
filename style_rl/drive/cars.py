"""Car presets of the driving environment: the simulator behind it, the file with the car's measured limits, and the numbers that belong to the car (the scripted driver's wheelbase and
traction control, the wheel positions of the off-track test, how fast the demonstrations drive).

    f317        Dallara F317 in OCD through our wrapper (style_rl.ocd_env), the car of run a1.
    sf70h_f1    the SF70H-like F1 car of Ibrahim Khaled's ocd_lap package (OcdLapBackend; his files are read from OCD_FIRST_LAP_DIR and are not part of this repository). Its limits file
                (data/f1/ocd_envelope_sf70h_f1.json) is measured with  scripts/measure_ocd_envelope.py --backend ocdlap:sf70h_f1 --vmax 95  and is NOT in git.
    His own F1 runs used demonstrations at 0.5-0.8 of the QSS speed (0.35-0.65 gave a copy that stalled in the first corners) and a traction control that starts at slip 0.06.
"""
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CarPreset:
    name: str
    backend: str
    envelope: str                  # relative to the repository root
    env: dict                      # EnvConfig fields that belong to the car
    demo_fraction: tuple           # (low, high) fraction of the QSS speed the demonstrations aim at
    note: str = ""
    train: dict = field(default_factory=dict)      # TrainConfig fields that belong to the car


PRESETS = {
    "f317": CarPreset("f317", "ocd", "data/ocd_envelope.json", {}, (0.35, 0.65), "Dallara F317 in OCD, wheelbase 2.8 m"),
    # brake cap 0.3: the scripted driver of the F1 runs of Ibrahim Khaled's repository caps the brake pedal at 0.3 (15 kN m of brakes lock the wheels above that); his F1 runs did not require the
    # behaviour-cloned copy to finish a lap either (gate_required False); setup p1_TUM_f1 = heave damping 0.3 (without it the car porpoises) and the p1_TUM knobs, see OcdLapBackend
    "sf70h_f1": CarPreset("sf70h_f1", "ocdlap:sf70h_f1:p1_TUM_f1", "data/f1/ocd_envelope_sf70h_f1.json",
                          dict(wheel_half_base_m=1.8, wheel_half_track_m=0.8, wheelbase_m=3.6, tc_start=0.06, tc_span=0.03, demo_brake_cap=0.3), (0.5, 0.8),
                          "SF70H-like F1 car, wheelbase 3.6 m, ERS off", dict(gate_required=False)),
    # the Assetto Corsa Ferrari SF70H rebuilt in OCD from its data files (style_rl.accar, scripts/ac_car_to_ocd.py): wheelbase 3.58 m, steering lock 18 degrees at the wheels; its numbers are NOT in this
    # repository (data/ac_sf70h/ is ignored by git). margin_m 2.0: with 1.5 m the line runs 1.0 m from the edge in the slowest chicane (station 964) and the scripted driver leaves the track there in 4 of 10
    # starts; with 2.0 m it is clean in 10 of 10 at 50 % and 70 % of the QSS speed (measured on the pod 2026-11-20)
    "ac_sf70h": CarPreset("ac_sf70h", "ocdlap:ac_sf70h", "data/ac_sf70h/ocd_envelope.json",
                          dict(steer_lock_rad=math.radians(18.0), margin_m=2.0, wheel_half_base_m=1.79, wheel_half_track_m=0.80, wheelbase_m=3.58, tc_start=0.07, tc_span=0.04, demo_brake_cap=0.3, demo_throttle_gamma=5.75, demo_throttle_boost=3.4), (0.5, 0.8),
                          "Assetto Corsa Ferrari SF70H in OCD", dict(gate_required=False)),
}


def apply_preset(name, env_cfg):
    """`env_cfg` with the preset car's numbers."""
    if name not in PRESETS:
        raise ValueError(f"unknown car {name!r}; choose from {sorted(PRESETS)}")
    return replace(env_cfg, **PRESETS[name].env)
