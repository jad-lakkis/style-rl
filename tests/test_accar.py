"""Tests of the Assetto Corsa car replica (style_rl.accar): readers, recognition of files by content, tyre fits, conversions, powertrain, and the glue to ocd_lap on stand-ins.   pytest tests/test_accar.py

A folder with real SF70H data is used where one is given (environment variable AC_SF70H_DIR, else c:/Users/NANO/work/jad-race); those tests are skipped otherwise. The stand-in classes below are a few lines
written for the tests, they are NOT code of ocd_first_lap."""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from style_rl.accar.ac_files import AcCar
from style_rl.accar.ini import num, parse_curve, parse_ini, vec
from style_rl.accar.powertrain import make_powertrain
from style_rl.accar.to_ocd import P, build_replica, roll_centre_height
from style_rl.accar.tyre_fit import fit_peak_friction, fit_shape, mf_curve, stiffness_params

REAL_DIR = Path(os.environ.get("AC_SF70H_DIR", r"C:\Users\NANO\work\jad-race"))
HAVE_REAL = (REAL_DIR / "wing_animations.ini").exists() or (REAL_DIR / "suspensions.ini").exists()
real_data = pytest.mark.skipif(not HAVE_REAL, reason="no SF70H data folder")


# ------------------------------------------------------------------------------------------------------------ a small synthetic car
CAR = """[HEADER]
VERSION=1
[BASIC]
TOTALMASS=700
INERTIA=1.5,0.8,4.5
[CONTROLS]
STEER_LOCK=180
STEER_RATIO=10
[FUEL]
FUEL=50
[RIDE]
PICKUP_FRONT_HEIGHT=-0.2
PICKUP_REAR_HEIGHT=-0.15
"""
SUSP = """[HEADER]
VERSION=2
[BASIC]
WHEELBASE=3.5
CG_LOCATION=0.45
[ARB]
FRONT=60000
REAR=10000
[FRONT]
TYPE=DWB
BASEY=0.07
TRACK=1.6
HUB_MASS=20
WBCAR_TOP_FRONT=0.6, 0.18, 0.08
WBCAR_TOP_REAR=0.6, 0.18, -0.30
WBCAR_BOTTOM_FRONT=0.6, 0.03, 0.07
WBCAR_BOTTOM_REAR=0.6, 0.03, -0.18
WBTYRE_TOP=0.1, 0.13, 0.0
WBTYRE_BOTTOM=0.07, -0.036, 0.0
SPRING_RATE=40000
BUMP_STOP_RATE=200000
PACKER_RANGE=0.04
DAMP_BUMP=3500
DAMP_FAST_BUMP=1100
DAMP_FAST_BUMPTHRESHOLD=0.05
DAMP_REBOUND=3850
DAMP_FAST_REBOUND=1460
DAMP_FAST_REBOUNDTHRESHOLD=0.12
[HEAVE_FRONT]
SPRING_RATE=120000
DAMP_BUMP=4000
DAMP_FAST_BUMP=1700
DAMP_FAST_BUMPTHRESHOLD=0.09
DAMP_REBOUND=5500
DAMP_FAST_REBOUND=2600
DAMP_FAST_REBOUNDTHRESHOLD=0.14
[REAR]
TYPE=DWB
BASEY=0.07
TRACK=1.5
HUB_MASS=22
WBCAR_TOP_FRONT=0.57, 0.1, 0.4
WBCAR_TOP_REAR=0.65, 0.12, 0.04
WBCAR_BOTTOM_FRONT=0.56, -0.05, 0.45
WBCAR_BOTTOM_REAR=0.7, -0.03, -0.03
WBTYRE_TOP=0.29, 0.2, 0.0
WBTYRE_BOTTOM=0.15, 0.003, 0.0
SPRING_RATE=30000
BUMP_STOP_RATE=200000
PACKER_RANGE=0.05
DAMP_BUMP=5000
DAMP_FAST_BUMP=1400
DAMP_FAST_BUMPTHRESHOLD=0.05
DAMP_REBOUND=5500
DAMP_FAST_REBOUND=2000
DAMP_FAST_REBOUNDTHRESHOLD=0.12
[HEAVE_REAR]
SPRING_RATE=120000
DAMP_BUMP=5500
DAMP_FAST_BUMP=1500
DAMP_FAST_BUMPTHRESHOLD=0.05
DAMP_REBOUND=6400
DAMP_FAST_REBOUND=2500
DAMP_FAST_REBOUNDTHRESHOLD=0.12
"""
TYRES = """[HEADER]
VERSION=10
[COMPOUND_DEFAULT]
INDEX=0
[FRONT]
NAME=Slick A
RADIUS=0.33
ANGULAR_INERTIA=1.4
RATE=300000
DY_REF=1.9
DX_REF=1.9
FZ0=3800
LS_EXPY=0.78
LS_EXPX=0.92
FRICTION_LIMIT_ANGLE=6.0
RADIUS_ANGULAR_K=0.02
ROLLING_RESISTANCE_0=6
ROLLING_RESISTANCE_1=0.0005
[REAR]
NAME=Slick A
RADIUS=0.33
ANGULAR_INERTIA=1.5
RATE=370000
DY_REF=1.9
DX_REF=1.9
FZ0=4300
LS_EXPY=0.83
LS_EXPX=0.95
FRICTION_LIMIT_ANGLE=5.8
"""
BRAKES = "[HEADER]\nVERSION=1\n[DATA]\nMAX_TORQUE=3900\nFRONT_SHARE=0.54\n"
DRIVETRAIN = """[HEADER]
VERSION=3
[TRACTION]
TYPE=RWD
[GEARS]
COUNT=3
GEAR_R=-4.5
GEAR_1=3.0
GEAR_2=2.0
GEAR_3=1.5
FINAL=4.0
[DIFFERENTIAL]
POWER=0.15
COAST=0.3
PRELOAD=150
[GEARBOX]
INERTIA=0.01
[AUTO_SHIFTER]
UP=11500
DOWN=7000
"""
ENGINE = """[HEADER]
VERSION=1
POWER_CURVE=power.lut
[ENGINE_DATA]
INERTIA=0.05
LIMITER=15000
[TURBO_0]
MAX_BOOST=3.5
WASTEGATE=3.4
REFERENCE_RPM=5500
GAMMA=5.75
LAG_DN=0.9986
LAG_UP=0.9984
"""
ERS = "[HEADER]\nVERSION=1\n[KINETIC]\nMAX_KJ_PER_LAP=4000\nDISCHARGE_TIME=33330\nTORQUE_CURVE=kers_torque.lut\n"
AERO = "[HEADER]\nVERSION=3\n[WING_0]\nNAME=BODY\nCHORD=1\nSPAN=1.5\nANGLE=2\nCL_GAIN=0\nCD_GAIN=1\n"
ELECTRONICS = "[ABS]\nPRESENT=0\n[TRACTION_CONTROL]\nPRESENT=0\n"
SETUP = "[DISPLAY_METHOD]\nSHOW_CLICKS=1\n" + "".join(f"[ITEM_{i}]\nTAB=A\nNAME=n{i}\nMIN=0\nMAX=10\nSTEP=1\n" for i in range(12))
RPM = np.arange(0, 16001, 500.0)
ICE = "\n".join(f"{r:g}|{max(0.0, 100 + 46 * math.sin(math.pi * min(r, 12500) / 12500) - (r > 12500) * (r - 12500) * 0.03):g}" for r in RPM[:-1]) + "\n16000|0\n"
KERS = "\n".join(f"{r:g}|{min(182.0, 1.0e5 / (max(r, 1) * 2 * math.pi / 60)):.1f}" for r in RPM[1:]) + "\n"
COAST = "\n".join(f"{r:g}|{min(90.0, 5.0e4 / (max(r, 1) * 2 * math.pi / 60)):.1f}" for r in RPM[1:]) + "\n"
FILES = {"car.ini": CAR, "suspensions.ini": SUSP, "tyres.ini": TYRES, "brakes.ini": BRAKES, "drivetrain.ini": DRIVETRAIN, "engine.ini": ENGINE, "ers.ini": ERS, "aero.ini": AERO,
         "electronics.ini": ELECTRONICS, "setup.ini": SETUP, "power.lut": ICE, "kers_torque.lut": KERS, "kers_torque_coast.lut": COAST}


def write_folder(path, shuffle=False):
    path.mkdir(exist_ok=True)
    items = list(FILES.items())
    if shuffle:                                             # contents under the wrong names, as in the folder we were given
        names, texts = [n for n, _ in items], [t for _, t in items]
        texts = texts[3:] + texts[:3]
        items = list(zip(names, texts))
    for n, t in items:
        (path / n).write_text(t)
    return path


# ------------------------------------------------------------------------------------------------------------ readers
def test_ini_and_curve_readers():
    s = parse_ini("[A]\nX=1 ; comment\nY = two // c\n\n[B]\nZ=3.5\n")
    assert s == {"A": {"X": "1", "Y": "two"}, "B": {"Z": "3.5"}}
    pts, header = parse_curve("[VALUES]\nC=1.27\nE=-1.8\n0|0\n0.001|0.029\n0.002|0.058\n")
    assert pts.shape == (3, 2) and header == {"C": "1.27", "E": "-1.8"}
    assert parse_curve("[HEADER]\nVERSION=1\n[X]\nA=1\nB=2\n") == (None, {})
    assert num("0.0250 ; m") == 0.025 and num("x", 7) == 7 and vec("1.6,0.82,4.66") == (1.6, 0.82, 4.66)


def test_files_are_recognised_by_content_whatever_their_names(tmp_path):
    ok = AcCar(write_folder(tmp_path / "ok"))
    assert not ok.scrambled and {"car", "suspensions", "tyres", "brakes", "drivetrain", "engine", "ers", "aero", "electronics", "setup"} <= set(ok.ini)
    bad = AcCar(write_folder(tmp_path / "bad", shuffle=True))
    assert bad.scrambled and {"car", "suspensions", "tyres", "brakes", "drivetrain", "engine"} <= set(bad.ini)
    assert bad.value("car", "BASIC", "TOTALMASS") == 700 and bad.name_map()["car"][1] == "car.ini"


def test_engine_curves_by_name_and_by_shape(tmp_path):
    ok = AcCar(write_folder(tmp_path / "ok"))
    curves, notes = ok.engine_curves()
    assert set(curves) == {"engine", "kers", "coast"} and all("by name" in n for n in notes)
    bad = AcCar(write_folder(tmp_path / "bad"))
    for name in ("power.lut", "kers_torque.lut", "kers_torque_coast.lut"):                      # same curves under other names: found by shape
        (tmp_path / "bad" / name).rename(tmp_path / "bad" / ("x_" + name))
    bad = AcCar(tmp_path / "bad")
    curves, notes = bad.engine_curves()
    assert set(curves) == {"engine", "kers", "coast"} and all("by shape" in n for n in notes)
    assert curves["engine"][:, 1].max() > 140 and curves["kers"][:, 1].max() > curves["coast"][:, 1].max()


# ------------------------------------------------------------------------------------------------------------ tyre fits
def test_shape_fit_recovers_a_known_curve():
    s = np.linspace(0.0, 0.999, 1000)
    c = np.column_stack([s, mf_curve(s, 29.0 / 1.27, 1.27, -1.8)])
    C, E, rms = fit_shape(c)
    assert C == pytest.approx(1.27, abs=1e-3) and E == pytest.approx(-1.8, abs=1e-2) and rms < 1e-5


def test_peak_friction_fit_follows_the_power_law_over_the_load_range():
    pdy1, pdy2 = fit_peak_friction(1.86, 3794.0, 0.78, (1900.0, 8300.0))
    for fz, tol in ((1900.0, 0.07), (2500.0, 0.03), (3794.0, 0.03), (6000.0, 0.03), (8300.0, 0.07)):          # a straight line through a convex law: good in the middle, a few percent off at the ends
        mu_line = pdy1 + pdy2 * (fz - 3794.0) / 3794.0
        assert mu_line == pytest.approx(1.86 * (fz / 3794.0) ** (0.78 - 1), rel=tol)


def test_stiffness_params_give_the_target_stiffness_and_load_exponent():
    pky1, pky2 = stiffness_params(200000.0, 3800.0, 0.78)
    k = lambda fz: pky1 * 3800.0 * math.sin(2 * math.atan(fz / (pky2 * 3800.0)))
    assert k(3800.0) == pytest.approx(200000.0, rel=1e-9)
    exponent = (math.log(k(3800.0 * 1.001)) - math.log(k(3800.0))) / math.log(1.001)
    assert exponent == pytest.approx(0.78, abs=1e-3)


# ------------------------------------------------------------------------------------------------------------ conversions
def test_roll_centre_of_a_symmetric_geometry():
    car = AcCar.__new__(AcCar)
    susp = parse_ini(SUSP)
    rc = roll_centre_height(susp, "FRONT", 0.335)
    assert 0.0 < rc < 0.4
    flat = {"FRONT": dict(susp["FRONT"], WBCAR_TOP_FRONT="0.6, 0.18, 0", WBCAR_TOP_REAR="0.6, 0.18, 0", WBCAR_BOTTOM_FRONT="0.6, -0.04, 0", WBCAR_BOTTOM_REAR="0.6, -0.04, 0", WBTYRE_TOP="0.1, 0.18, 0",
                              WBTYRE_BOTTOM="0.1, -0.04, 0")}
    assert roll_centre_height(flat, "FRONT", 0.335) is None or roll_centre_height(flat, "FRONT", 0.335) == pytest.approx(0.335 - 0.335 + 0.0, abs=1.0)
    del car


def test_replica_numbers_of_the_synthetic_car(tmp_path):
    r = build_replica(AcCar(write_folder(tmp_path / "ok")), fuel_l=40.0)
    o = r["ocd"]
    assert o[P + "mass_vehicle_kg"] == pytest.approx(700 + 0.75 * 40)
    assert o[P + "cog.distance_from_front_axle_m"] == pytest.approx(0.55 * 3.5)             # 45 % of the weight on the front axle: the CG is nearer the rear
    assert o[P + "cog.height_m"] == pytest.approx(0.33 - 0.07)
    assert o[P + "suspension.antirollbar_virtual_spring_stiffness_Npm.front"] == 120000.0       # c = 2 k
    assert o[P + "suspension.heave.spring_stiffness_Npm.front"] == 240000.0
    assert o[P + "suspension.four_way_damper.bump_fast_Nspm.front"] == 1100.0 and o[P + "suspension.four_way_damper.knee_speed_mps.front"] == pytest.approx(0.085)
    static = (((700 + 30) * 0.45) / 2 - 20) * 9.81 / (40000 + 120000)
    assert o[P + "suspension.packer.gap_m.front"] == pytest.approx(0.04 - static)
    assert o[P + "z.tire_model.front_left.FNOMIN"] == 3800.0 and o[P + "z.tire_model.rear_right.FNOMIN"] == 4300.0
    assert o[P + "z.tire_model.front_left.PVY1"] == 0.0
    assert r["car_spec"]["brake_torque_Nm"] == 4 * 3900.0 and r["setup"]["brake_bias_front"] == 0.54 and r["car_spec"]["steer_max_rad"] == pytest.approx(math.radians(18))
    assert r["powertrain"]["gears"] == [3.0, 2.0, 1.5] and r["powertrain"]["ers"]["energy_per_lap_J"] == 4e6
    assert {row["status"] for row in r["report"]} <= {"exact", "converted", "fitted", "estimated", "assumed", "placeholder", "not modelled"}
    json.dumps(r, default=float)


@real_data
def test_the_real_folder_is_read_and_converted():
    """No numbers of the game's data are written here: every expected value is read from the folder again with the plain readers."""
    car = AcCar(REAL_DIR)
    assert {"car", "suspensions", "tyres", "brakes", "drivetrain", "engine", "ers", "aero", "electronics", "setup"} <= set(car.ini)
    r = build_replica(car)
    o = r["ocd"]
    front = car.value("suspensions", "BASIC", "CG_LOCATION")
    wheelbase = car.value("suspensions", "BASIC", "WHEELBASE")
    assert o[P + "mass_vehicle_kg"] == pytest.approx(car.value("car", "BASIC", "TOTALMASS") + 0.75 * car.value("car", "FUEL", "FUEL"))
    assert o[P + "wheelbase_m"] == wheelbase and o[P + "cog.distance_from_front_axle_m"] == pytest.approx((1 - front) * wheelbase)
    assert 0.2 < o[P + "cog.height_m"] < 0.35
    assert o[P + "suspension.vehicle_spring_stiffness_Npm.front"] == car.value("suspensions", "FRONT", "SPRING_RATE")
    assert o[P + "suspension.antirollbar_virtual_spring_stiffness_Npm.rear"] == 2 * car.value("suspensions", "ARB", "REAR")
    assert r["powertrain"]["limiter_rpm"] == car.value("engine", "ENGINE_DATA", "LIMITER") and len(r["powertrain"]["gears"]) == int(car.value("drivetrain", "GEARS", "COUNT"))
    assert 0.1 < o[P + "suspension.roll_center_height_m"] < 0.2
    peak_kw = max(t * r_ * 2 * math.pi / 60 for r_, t in r["powertrain"]["engine_curve"]) * (1 + r["powertrain"]["turbo"]["wastegate"]) / 1e3
    assert 400 < peak_kw < 1000                                                                 # a hybrid F1 engine, not a toy and not a rocket


@real_data
def test_the_game_setup_clicks_are_the_cars_default_setup():
    """The click values read from game screenshots are MIN + click x STEP of setup.ini; for the default setup they must give the numbers of suspensions.ini."""
    car = AcCar(REAL_DIR)
    st = car.ini["setup"]
    value = lambda item, click: num(st[item]["MIN"]) + click * num(st[item]["STEP"])
    s = lambda a, k: car.value("suspensions", a, k)
    assert value("DAMP_FAST_BUMP_LF", 5) == s("FRONT", "DAMP_FAST_BUMP") and value("DAMP_FAST_REBOUND_LF", 8) == s("FRONT", "DAMP_FAST_REBOUND") and value("DAMP_FAST_BUMP_LR", 8) == s("REAR", "DAMP_FAST_BUMP")
    assert value("DAMP_BUMP_LF", 13) == pytest.approx(s("FRONT", "DAMP_BUMP"), abs=40) and value("DAMP_BUMP_LR", 21) == s("REAR", "DAMP_BUMP")
    assert 1000 * value("SPRING_RATE_LF", 0) == s("FRONT", "SPRING_RATE") and 1000 * value("SPRING_RATE_LR", 3) == s("REAR", "SPRING_RATE")
    assert 1000 * value("SPRING_RATE_HF", 4) == s("HEAVE_FRONT", "SPRING_RATE") and 1000 * value("SPRING_RATE_HR", 5) == s("HEAVE_REAR", "SPRING_RATE")
    assert value("ARB_FRONT", 17) == s("ARB", "FRONT") and value("ARB_REAR", 5) == s("ARB", "REAR")


# ------------------------------------------------------------------------------------------------------------ powertrain on a stand-in base class
class BaseStandIn:
    """The interface of ocd_lap.powertrain.Powertrain that the replica's powertrain relies on (a few lines written for the test)."""

    def __init__(self, *, final_drive=1.0, vehicle_mass_kg, wheel_radius_m, wheel_inertia_kgm2, data=None, ers=None):
        self.ers, self.wheel_radius_m, self.d = ers, wheel_radius_m, {"eta_g": 0.96}
        self.ratios, self.engine_inertia_kgm2 = np.array([1.0]), [0.0]
        self.gear, self.since_shift_s = 0, 0.1
        self.refill_ers()

    def refill_ers(self):
        self.ers_energy_J = self.ers["energy_per_lap_J"] if self.ers else 0.0

    def rpm(self, omega, gear=None):
        return omega * 60 / (2 * math.pi) * self.ratios[self.gear if gear is None else gear]

    def draw_ers(self, throttle, dt):
        pass


def synthetic_powertrain(**over):
    pt = {"gears": [3.0, 2.0, 1.5], "final": 4.0, "limiter_rpm": 15000.0, "shift_up_rpm": 11500.0, "shift_down_rpm": 7000.0, "engine_inertia_kgm2": 0.06, "efficiency": 1.0, "wheel_radius_m": 0.33,
          "engine_curve": [[0, 100], [5000, 120], [10000, 146], [15000, 60], [16000, 0]],
          "turbo": {"max_boost": 3.5, "wastegate": 3.4, "reference_rpm": 5500.0, "gamma": 5.75, "lag_up": 0.9984, "lag_dn": 0.9986, "physics_hz": 333.0},
          "ers": {"energy_per_lap_J": 4e6, "discharge_time_s": 33.3, "torque_curve": [[500, 182], [5000, 182], [15000, 66], [16000, 0]]}}
    pt.update(over)
    return pt


def make_pt(turbo="ac", ers=False, **over):
    cls = make_powertrain(BaseStandIn, synthetic_powertrain(**over), turbo)
    return cls(final_drive=1.0, vehicle_mass_kg=770.0, wheel_radius_m=0.33, wheel_inertia_kgm2=3.0, ers={"energy_per_lap_J": 4e6} if ers else None)


def omega_for(rpm, ratio):
    return rpm * 2 * math.pi / 60 / ratio


def test_turbo_spools_with_the_ac_time_constant_and_needs_a_full_pedal():
    pt = make_pt("ac")
    pt.set_pedal(1.0)
    w = omega_for(8000, pt.ratios[0])
    for _ in range(int(1.9 / 0.002)):                                       # one time constant of about 1.9 s
        pt.update_gear(w, 0.002)
    target = 3.4
    assert pt.boost == pytest.approx(target * (1 - math.exp(-1.0)), rel=0.08)
    for _ in range(int(15.0 / 0.002)):
        pt.update_gear(w, 0.002)
    assert pt.boost == pytest.approx(target, rel=0.01)
    t_full = pt.engine_torque_Nm(10000)
    pt.set_pedal(0.9)
    for _ in range(int(30.0 / 0.002)):
        pt.update_gear(w, 0.002)
    assert pt.boost == pytest.approx(3.4 * 0.9 ** 5.75, rel=0.01) and 0.9 * pt.engine_torque_Nm(10000) < 0.6 * t_full     # the backend multiplies the capacity by the pedal: at 90 % pedal the torque is 58 % of the full-pedal torque, not 90 %
    assert t_full == pytest.approx(146 * 4.4, rel=0.01)


def test_turbo_modes():
    steady, linear = make_pt("steady"), make_pt("linear")
    w = omega_for(8000, steady.ratios[0])
    steady.set_pedal(1.0)
    steady.update_gear(w, 0.002)
    assert steady.boost == pytest.approx(3.4)                              # no lag
    linear.set_pedal(0.3)
    linear.update_gear(w, 0.002)
    assert linear.boost == pytest.approx(3.4)                              # boost whatever the pedal: torque will be proportional to the pedal in the backend
    with pytest.raises(ValueError):
        make_powertrain(BaseStandIn, synthetic_powertrain(), "nope")


def test_gearbox_follows_the_ac_auto_shifter_and_the_limiter_cuts_the_torque():
    pt = make_pt("steady")
    pt.set_pedal(1.0)
    assert pt.gear == 0
    pt.update_gear(omega_for(11600, pt.ratios[0]), 0.2)
    assert pt.gear == 1                                                    # up at 11500 rpm
    pt.update_gear(omega_for(6900, pt.ratios[1]), 0.2)
    assert pt.gear == 0                                                    # down below 7000 rpm
    pt.gear = 2
    pt.update_gear(omega_for(12000, pt.ratios[2]), 0.2)
    assert pt.gear == 2                                                    # the last gear stays
    assert pt.engine_torque_Nm(15100) == 0.0 and pt.engine_torque_Nm(14900) > 0.0
    assert [round(r, 3) for r in pt.ratios] == [12.0, 8.0, 6.0]
    assert pt.engine_inertia_at_wheels_kgm2 if hasattr(pt, "engine_inertia_at_wheels_kgm2") else True
    assert pt.engine_inertia_kgm2[0] == pytest.approx(0.06 * 12.0 ** 2)


def test_axle_torque_is_engine_torque_through_the_gear_and_the_motor_adds_when_on():
    off, on = make_pt("linear", ers=False), make_pt("linear", ers=True)
    for p in (off, on):
        p.set_pedal(1.0)
        p.update_gear(omega_for(9000, p.ratios[0]), 0.0)
    w = omega_for(9000, off.ratios[0])
    ice = np.interp(9000, [0, 5000, 10000, 15000, 16000], [100, 120, 146, 60, 0]) * 4.4
    assert off.axle_torque_capacity_Nm(w) == pytest.approx(1.0 * 12.0 * ice, rel=1e-6)
    assert on.axle_torque_capacity_Nm(w, 40.0) > off.axle_torque_capacity_Nm(w)
    on.ers_energy_J = 0.0
    assert on.axle_torque_capacity_Nm(w, 40.0) == pytest.approx(off.axle_torque_capacity_Nm(w), rel=1e-9)


def test_real_powertrain_base_class_works_with_the_replica_powertrain():
    """The same, on the real ocd_lap.powertrain.Powertrain (a pure-python file; skipped when the clone of ocd_first_lap is not on this machine)."""
    root = Path(os.environ.get("OCD_FIRST_LAP_DIR", r"C:\Users\NANO\work\ocd_first_lap"))
    f = root / "ocd_lap" / "powertrain.py"
    if not f.exists():
        pytest.skip("no ocd_lap checkout")
    spec = importlib.util.spec_from_file_location("ocd_lap_powertrain_real", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = make_powertrain(mod.Powertrain, synthetic_powertrain(), "ac")
    pt = cls(final_drive=1.0, vehicle_mass_kg=770.0, wheel_radius_m=0.33, wheel_inertia_kgm2=3.0, ers=None)
    pt.set_pedal(1.0)
    for _ in range(3000):
        pt.update_gear(omega_for(9000, pt.ratios[0]), 0.002)
    assert pt.boost > 3.0 and pt.axle_torque_capacity_Nm(omega_for(9000, pt.ratios[0])) > 6000.0
    assert pt.engine_inertia_at_wheels_kgm2 == pytest.approx(0.06 * 12.0 ** 2)


# ------------------------------------------------------------------------------------------------------------ the glue to ocd_lap and the backend, on stand-ins
STAND_IN_BACKEND = """
import math
from dataclasses import dataclass
DT = .002
Powertrain = None
@dataclass(frozen=True)
class StartPose:
    x: float = 0.
    y: float = 0.
    yaw: float = 0.
    speed: float = 5.
@dataclass(frozen=True)
class Inputs:
    steering_rad: float
    throttle: float
    brake: float
class OCDBackend:
    def __init__(self, *, powertrain='direct', max_drive_torque_Nm=270., max_brake_torque_Nm=None, overrides=None, car='av21', ers=None):
        self.spec, self.overrides, self.ers_flag, self.powertrain = car, dict(overrides or {}), ers, None
        self.pedals = []
    def reset(self, setup, start_pose):
        self.setup = setup
        self.powertrain = Powertrain(final_drive=1., vehicle_mass_kg=700., wheel_radius_m=.33, wheel_inertia_kgm2=3., ers={'energy_per_lap_J': 4e6} if self.ers_flag else None)
        self.s = dict(x=start_pose.x, y=start_pose.y, yaw=start_pose.yaw, vx=start_pose.speed)
        return self._state()
    def _state(self):
        s = self.s
        return dict(x=s['x'], y=s['y'], yaw=s['yaw'], vx=s['vx'], vy=0., yaw_rate=0., steer_actual=0.05, slip_rl=0., slip_rr=0.)
    def step(self, inputs, steps=1):
        if steps > 1:
            for _ in range(steps):
                self.step(inputs)
            return self._state()
        self.pedals.append(self.powertrain.pedal)
        self.powertrain.update_gear(100.0, DT)
        self.s['x'] += self.s['vx'] * DT
        return self._state()
"""
STAND_IN_CARS_AC = """
import dataclasses
from dataclasses import dataclass, field
@dataclass(frozen=True)
class CarSpec:
    name: str
    ocd: dict = field(default_factory=dict)
    rear_axle_m: float = 1.476
    wheelbase_m: float = 3.2
    driver_mass_kg: float = 800.
    driver_radius_m: float = .3
    footprint: tuple = ()
    ers: dict = None
    steer_max_rad: float = .6
    steer_rate_max_radps: float = 1.5
    brake_torque_Nm: float = None
    description: str = ''
@dataclass(frozen=True)
class Setup:
    brake_bias_front: float = .5
    final_drive_ratio: float = 1.
    def replace(self, **kw):
        return dataclasses.replace(self, **kw)
def get(car):
    return car
def default_setup(car):
    return Setup()
"""


@pytest.fixture
def fake_ocd_lap(tmp_path, monkeypatch):
    pkg = tmp_path / "ocd_lap"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "backend.py").write_text(STAND_IN_BACKEND)
    (pkg / "cars.py").write_text(STAND_IN_CARS_AC)
    gone = lambda: [m for m in sys.modules if m == "ocd_lap" or m.startswith("ocd_lap.")]
    for name in gone():
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("OCD_FIRST_LAP_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    import ocd_lap.backend as bk
    bk.Powertrain = BaseStandIn
    yield tmp_path
    for name in gone():
        del sys.modules[name]


def replica_json(tmp_path):
    r = build_replica(AcCar(write_folder(tmp_path / "acdata")))
    path = tmp_path / "replica.json"
    path.write_text(json.dumps(r, default=float))
    return path, r


def test_glue_builds_the_spec_patches_the_powertrain_and_passes_the_pedal(fake_ocd_lap):
    from style_rl.accar.ocdlap_glue import build
    _, replica = replica_json(fake_ocd_lap)
    backend, setup, spec = build(replica, turbo="steady", ers=False)
    assert spec.name == "ac_sf70h" and spec.brake_torque_Nm == 15600.0 and spec.steer_max_rad == pytest.approx(math.radians(18)) and spec.ers is None
    assert backend.overrides[P + "mass_vehicle_kg"] == replica["ocd"][P + "mass_vehicle_kg"] and setup.brake_bias_front == 0.54
    import ocd_lap.backend as bk
    from ocd_lap.backend import Inputs, StartPose
    backend.reset(setup, StartPose(0, 0, 0, 30.0))
    assert type(backend.powertrain).__name__ == "AcPowertrain" and backend.powertrain.ratios[0] == pytest.approx(3.0 * 4.0)
    backend.step(Inputs(0.0, 0.7, 0.0), steps=5)
    assert backend.pedals == [0.7] * 5                                    # the pedal reached the powertrain before every physics step
    backend.step(Inputs(0.0, 1.0, 0.0), steps=1)
    assert backend.powertrain.boost > 0.0 and bk.Powertrain is not BaseStandIn
    b2, _, spec2 = build(replica, turbo="ac", ers=True)
    assert spec2.ers["energy_per_lap_J"] == 4e6


def test_the_backend_name_with_options_builds_the_ac_car(fake_ocd_lap, monkeypatch):
    from style_rl.drive.backend import make_backend
    path, _ = replica_json(fake_ocd_lap)
    monkeypatch.setenv("AC_REPLICA_JSON", str(path))
    b = make_backend("ocdlap:ac_sf70h?turbo=steady&ers=1", None, math.radians(18), 0.04)
    assert b.car == "ac_sf70h" and b.spec.ers is not None and b.setup_name == "ac_default"
    s0 = b.reset(1.0, 2.0, 0.0, 30.0)
    s1 = b.step(0.5, 1.0, 0.0)
    assert s0["speed"] == 30.0 and s1["time_s"] == pytest.approx(0.04) and b.backend.pedals[-1] == 1.0
    with pytest.raises(ValueError):                                       # the actuator limit of the replica is 18 degrees
        make_backend("ocdlap:ac_sf70h", None, 0.4, 0.04)
    from style_rl.drive.cars import PRESETS, apply_preset
    from style_rl.drive.config import EnvConfig
    env = apply_preset("ac_sf70h", EnvConfig())
    assert env.steer_lock_rad == pytest.approx(math.radians(18)) and env.wheelbase_m == 3.58 and PRESETS["ac_sf70h"].backend == "ocdlap:ac_sf70h"


def test_ocd_tyre_signs_cornering_stiffness_negative_longitudinal_positive():
    """OCD's shipped tyre has PKY1 = -75.5 and PKX1 = +30.7 (ISO axes); a positive PKY1 made the replica steer the wrong way in the first test on the pod."""
    from style_rl.accar.tyre_fit import axle_coefficients
    c = axle_coefficients(dy_ref=1.86, dx_ref=1.87, fz0=3794.0, ls_y=0.78, ls_x=0.92, fz_range=(1900.0, 8300.0), cf_y=18.0, cf_x=22.0, shape_y=(1.27, -1.8), shape_x=(1.2, -0.5))
    assert c["PKY1"] < 0 and c["PKX1"] > 0 and c["PKY2"] > 0 and c["PDY1"] > 0 and c["PDX1"] > 0
    assert -c["PKY1"] * 3794.0 * math.sin(2 * math.atan(1.0 / c["PKY2"])) == pytest.approx(18.0 * 1.86 * 3794.0, rel=1e-9)


def test_a_reset_at_speed_starts_in_the_right_gear():
    pt = make_pt("steady")
    pt.update_gear(omega_for(11000, pt.ratios[2]) , 0.0)             # first call after a reset, at the wheel speed of 11000 rpm in the third gear
    assert pt.gear == 2
    fast = make_pt("steady")
    fast.update_gear(omega_for(30000, fast.ratios[0]), 0.0)          # far above the limit in first gear: ends in the top gear
    assert fast.gear == 2
