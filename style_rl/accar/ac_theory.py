"""What the replica's own numbers predict for the limits of the car (no simulator needed), to be held against what OCD measures for the same car (scripts/measure_ocd_envelope.py).

The three curves have the format of data/ocd_envelope.json (full-pedal acceleration, braking g, cornering g, each against speed). They use the same aero as the replica (CzA, CxA at the static ride
heights, fitted at 200 km/h: at speed the car sinks, and AC's own aero changes with speed), AC's tyre law for the grip at each tyre's load, and the powertrain of the replica at full boost:

  acceleration   best gear, full pedal, full boost: (wheel force - aero drag - rolling drag) / mass, capped by what the driven (rear) tyres can transmit (weight transfer ignored)
  braking        every tyre at its longitudinal grip limit plus drag, or the brake torque if that is smaller (weight transfer ignored)
  cornering      both axles at their lateral grip limit with the load of each tyre = its share of weight and downforce (load transfer ignored)

These are upper bounds of a point-mass kind: OCD's car, with load transfer, slip-angle mismatch between the axles and the suspension, will come out a few percent to some tens of percent below, never above
by more than the aero difference at speed. A bigger gap means a conversion is wrong.
"""
import numpy as np

from .aero import FRONT_TERMS, REAR_TERMS, ocd_lift
from .powertrain import full_throttle_axle_force

G = 9.81
P = "vehicle_dynamics_double_track."


def _mu(ocd, wheel, axis, fz):
    """peak friction of the fitted MF5.2 tyre of one wheel at load fz (N)"""
    g = lambda k: ocd[P + f"z.tire_model.{wheel}.{k}"]
    fn = g("FNOMIN")
    d = "Y" if axis == "y" else "X"
    return (g(f"PD{d}1") + g(f"PD{d}2") * (fz - fn) / fn) * g("LMU" + d)


def aero_forces(ocd, v, ride_front=None, ride_rear=None):
    """(front downforce, rear downforce, drag) in N at speed v and the given ride heights (default: the static ones)"""
    rho = ocd[P + "aerodynamics.air_density_kgpm3"]
    hf = -ocd[P + "aerodynamics.static_ride_height_front_m"] if ride_front is None else ride_front
    hr = -ocd[P + "aerodynamics.static_ride_height_rear_m"] if ride_rear is None else ride_rear
    czf, czr = ocd_lift({t: ocd.get(P + f"aerodynamics.{t}", 0.0) for t in FRONT_TERMS + REAR_TERMS}, hf, hr)     # OCD's polynomials in the two ride heights, clamped to [0, 5]
    q = 0.5 * rho * np.asarray(v, float) ** 2 * ocd[P + "aerodynamics.A_m2"]
    return q * czf, q * czr, q * ocd[P + "aerodynamics.c_d"]


def theory(replica, speeds):
    """{"acc_ms2", "brk_g", "lat_g"} arrays on `speeds` (m/s)"""
    ocd, pt = replica["ocd"], replica["powertrain"]
    m = ocd[P + "mass_vehicle_kg"]
    lf, lr, wb = ocd[P + "cog.distance_from_front_axle_m"], ocd[P + "wheelbase_m"] - ocd[P + "cog.distance_from_front_axle_m"], ocd[P + "wheelbase_m"]
    w_front = m * G * lr / wb                                   # static axle loads
    w_rear = m * G * lf / wb
    f_roll = ocd[P + "tire.rolling_resistance_coefficient"]
    brake_n = replica["car_spec"]["brake_torque_Nm"] / ocd[P + "tire.rolling_radius_m.rear"]
    acc, brk, lat = [], [], []
    for v in np.asarray(speeds, float):
        dff, dfr, drag = aero_forces(ocd, v)
        ff, fr = w_front + dff, w_rear + dfr                   # axle loads with downforce
        mu = lambda wheels, axis, load: sum(_mu(ocd, w, axis, load / 2) * load / 2 for w in wheels)
        front, rear = ("front_left", "front_right"), ("rear_left", "rear_right")
        # acceleration
        f_eng = full_throttle_axle_force(pt, max(v, 1.0), turbo="steady", wheel_radius_m=ocd[P + "tire.rolling_radius_m.rear"])
        f_trac = mu(rear, "x", fr)
        acc.append((min(f_eng, f_trac) - drag - f_roll * (m * G + dff + dfr)) / m)
        # braking: tyres at their limit (all four), or the brakes
        f_brake = min(mu(front, "x", ff) + mu(rear, "x", fr), brake_n)
        brk.append((f_brake + drag) / (m * G))
        # cornering
        lat.append((mu(front, "y", ff) + mu(rear, "y", fr)) / (m * G))
    return {"acc_ms2": np.array(acc), "brk_g": np.array(brk), "lat_g": np.array(lat)}


def compare(replica, measured, speeds=None):
    """rows (speed, theory, measured, ratio) for the three curves; `measured` is the json written by scripts/measure_ocd_envelope.py"""
    sp = np.asarray(measured["measured_speeds"] if speeds is None else speeds, float)
    th = theory(replica, sp)
    ms = measured["measured"]
    rows = []
    for i, v in enumerate(sp):
        rows.append({"v_mps": float(v), "acc_theory": float(th["acc_ms2"][i]), "acc_ocd": float(ms["acc_ms2"][i]), "brk_theory_g": float(th["brk_g"][i]), "brk_ocd_g": float(ms["brk_g"][i]),
                     "lat_theory_g": float(th["lat_g"][i]), "lat_ocd_g": float(ms["lat_g"][i])})
    return rows
