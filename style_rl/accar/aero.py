"""Assetto Corsa's aerodynamics of a car (the SF70H), evaluated at a speed and two ride heights, and fitted to OCD's per-axle lift polynomial.

AC's law (acs.exe as documented by rustyAC, docs/map/aero.md; the same wing law that ocd_first_lap's docs/stage6/ac_model.md lists):
  every wing has a position (x sideways, y up, z forward, metres from the centre of gravity), an area CHORD x SPAN, a base angle ANGLE (degrees; the setup changes it) and gains CL_GAIN, CD_GAIN.
  angle   = ANGLE, then each dynamic controller of the wing in file order:  ADD: a = a + lut(input)   MULT: a = a * lut(input),  clamped to [DOWN_LIMIT, UP_LIMIT] after every one
  aoa     = the pitch of the body in degrees, positive nose down (rake): the air meets every wing that much steeper (the other parts of aoa, heave and pitch speed, are zero in steady running)
  height  = the height of the wing's point above the road plane, from the ride heights: h_cg + y cos(pitch) - z sin(pitch)
  lift    = 0.5 rho v^2 CHORD SPAN  CL_AOA(angle + aoa) CL_GAIN  CL_GH(height)       (downforce when positive)
  drag    = 0.5 rho v^2 CHORD SPAN  CD_AOA(angle + aoa) CD_GAIN  CD_GH(height)
  each lift acts at the wing's position; the BODY (WING_0) gives the drag of the car body (CL_GAIN 0); the fins only act in yaw (not used here: zero yaw).
OCD's own law (OCD's RIDE_HEIGHT aero model, ride_height.cpp): front and rear lift coefficients (area A = 1 m2) are cubic polynomials in the two ride heights, applied at the axles,
  c_lF = c0 + c1_F rhF + c2_F rhF^2 + c3_F rhF^3 + c1_R rhR + c2_R rhR^2 + c3_R rhR^3 + c4 rhF rhR + c5 rhF^2 rhR + c6 rhF rhR^2   (the rear one the same with a0 ... a6),   drag c_d constant.
`fit_ocd` evaluates AC's law over a box of ride heights at one speed and fits those ten coefficients per axle; the lifts of AC's wings are first moved to the axles by moment balance about the CG.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .ini import num, vec

FRONT_TERMS = ("c0", "c1_F", "c2_F", "c3_F", "c1_R", "c2_R", "c3_R", "c4", "c5", "c6")
REAR_TERMS = ("a0", "a1_F", "a2_F", "a3_F", "a1_R", "a2_R", "a3_R", "a4", "a5", "a6")


def lut_at(curve, x):
    """AC reads a lookup table linearly and holds the end values outside it (np.interp does the same)."""
    return float(np.interp(x, curve[:, 0], curve[:, 1]))


@dataclass
class Controller:
    comb: str                          # ADD or MULT
    inp: str                           # the telemetry channel: GAS, BRAKE, STEER, SPEED_KMH ...
    lut: np.ndarray
    down: float
    up: float


@dataclass
class Wing:
    index: int
    key: str
    name: str
    area: float
    cl_gain: float
    cd_gain: float
    angle: float
    pos: tuple                         # x sideways, y up, z forward, m from the CG
    cl: np.ndarray
    cd: np.ndarray
    gh_cl: object = None
    gh_cd: object = None
    controllers: list = field(default_factory=list)
    fin: bool = False


@dataclass
class Geometry:
    wheelbase: float
    lf: float                          # CG to the front axle (m)
    cg_height: float                   # CG height above the road at the static ride heights (m)
    static_ride: tuple = (0.045, 0.090)
    rho: float = 1.2

    @property
    def lr(self):
        return self.wheelbase - self.lf


def car_geometry(car, *, rho=1.2, static_ride=(0.045, 0.090)):
    """The geometry the aero needs, from the suspension and tyre files (the same numbers build_replica puts into OCD)."""
    susp = car.ini["suspensions"]
    wheelbase = num(susp["BASIC"]["WHEELBASE"])
    front_share = num(susp["BASIC"]["CG_LOCATION"])
    r_front, r_rear = num(car.ini["tyres"]["FRONT"]["RADIUS"]), num(car.ini["tyres"]["REAR"]["RADIUS"])
    h_cg = 0.5 * ((r_front - num(susp["FRONT"]["BASEY"])) + (r_rear - num(susp["REAR"]["BASEY"])))
    return Geometry(wheelbase=wheelbase, lf=(1.0 - front_share) * wheelbase, cg_height=h_cg, static_ride=tuple(static_ride), rho=rho)


def referenced_luts(aero):
    """{lut file name: [(section, key), ...]} for every curve aero.ini names: the angle and height tables of the wings and fins, and the dynamic controllers."""
    refs = {}
    for sec, kv in aero.items():
        for key, val in kv.items():
            if key.startswith("LUT") and val and val.strip():
                refs.setdefault(val.split()[0], []).append((sec, key))
    return refs


class AcAero:
    def __init__(self, wings, geometry):
        self.wings = list(wings)
        self.geo = geometry

    # ------------------------------------------------------------------------------------------------ construction
    @classmethod
    def from_car(cls, car, curves, geometry=None):
        """`curves`: {lut file name as aero.ini writes it: (n, 2) array} (see aero_curves.find_aero_curves)."""
        aero = car.ini["aero"]
        geometry = geometry or car_geometry(car)
        ctrl = {}
        for key, sec in sorted(aero.items()):
            if key.startswith("DYNAMIC_CONTROLLER_"):
                ctrl.setdefault(int(num(sec["WING"])), []).append((int(key.rsplit("_", 1)[1]), sec))
        wings = []
        numbered = [k for k in aero if k.startswith(("WING_", "FIN_")) and k.split("_")[1].isdigit() and "CHORD" in aero[k]]
        for key in sorted(numbered, key=lambda k: (k.split("_")[0] != "WING", int(k.split("_")[1]))):          # wings first, then fins, as AC adds them up
            sec = aero[key]
            idx = int(key.split("_")[1])
            fin = key.startswith("FIN_")
            pick = lambda k: curves[sec[k].split()[0]] if sec.get(k, "").strip() else None
            controllers = []
            if not fin:
                for _, c in sorted(ctrl.get(idx, []), key=lambda t: t[0]):
                    controllers.append(Controller(comb=c.get("COMBINATOR", "").split()[0].upper() if c.get("COMBINATOR") else "", inp=c["INPUT"].split()[0].upper(), lut=curves[c["LUT"].split()[0]],
                                                  down=num(c.get("DOWN_LIMIT"), -1e9), up=num(c.get("UP_LIMIT"), 1e9)))
            wings.append(Wing(index=idx, key=key, name=sec.get("NAME", key), area=num(sec["CHORD"]) * num(sec["SPAN"]), cl_gain=num(sec.get("CL_GAIN"), 1.0), cd_gain=num(sec.get("CD_GAIN"), 1.0),
                              angle=num(sec.get("ANGLE"), 0.0), pos=vec(sec["POSITION"]), cl=pick("LUT_AOA_CL"), cd=pick("LUT_AOA_CD"), gh_cl=pick("LUT_GH_CL"), gh_cd=pick("LUT_GH_CD"),
                              controllers=controllers, fin=fin))
        return cls(wings, geometry)

    # ------------------------------------------------------------------------------------------------ the law
    def attitude(self, rh_f, rh_r):
        """(CG height above the road, body pitch in rad, nose down positive) for ride heights at the two axles (floor above the road, m)"""
        g = self.geo
        h_cg = g.cg_height + ((rh_f - g.static_ride[0]) * g.lr + (rh_r - g.static_ride[1]) * g.lf) / g.wheelbase
        return h_cg, math.atan2(rh_r - rh_f, g.wheelbase)

    def evaluate(self, v_kmh, rh_f, rh_r, *, gas=1.0, brake=0.0, steer=0.0, angle_mult=None, pitch_aoa=True):
        """Straight-line aero at speed v_kmh and the given ride heights. Returns a dict:
        front / rear: the lift coefficient x area (m2, 'CzA') carried by each axle (the lift of every wing moved to the axles by moment balance about the CG),
        cxa: the drag coefficient x area (m2); rows: one entry per wing; downforce_N and drag_N at this speed.
        angle_mult: {wing index: factor} on the angle (DRS 'EFFECT' mode: 0 removes the setup angle of that wing)."""
        g = self.geo
        v = v_kmh / 3.6
        q = 0.5 * g.rho * v * v
        h_cg, theta = self.attitude(rh_f, rh_r)
        aoa = math.degrees(theta) if pitch_aoa else 0.0
        inputs = {"SPEED_KMH": v_kmh, "GAS": gas, "BRAKE": brake, "STEER": steer}
        front = rear = cxa = 0.0
        rows = []
        for w in self.wings:
            if w.fin:
                continue                                                  # fins: side force in yaw only
            a = w.angle
            for c in w.controllers:
                out = lut_at(c.lut, inputs.get(c.inp, 0.0))
                if c.comb == "ADD":
                    a = a + out
                elif c.comb == "MULT":
                    a = a * out
                a = min(max(a, c.down), c.up)
            idx = (angle_mult or {}).get(w.index, 1.0) * a + aoa
            h = h_cg + w.pos[1] * math.cos(theta) - w.pos[2] * math.sin(theta)
            cl = lut_at(w.cl, idx) * w.cl_gain
            if w.gh_cl is not None:
                cl *= lut_at(w.gh_cl, h)
            cd = lut_at(w.cd, idx) * w.cd_gain
            if w.gh_cd is not None:
                cd *= lut_at(w.gh_cd, h)
            lift, drag = cl * w.area, cd * w.area
            share = (w.pos[2] + g.lr) / g.wheelbase                       # moment balance about the rear axle: the share of this lift the front axle carries
            front += lift * share
            rear += lift * (1.0 - share)
            cxa += drag
            rows.append({"wing": w.name, "key": w.key, "angle_deg": a, "index_deg": idx, "height_mm": 1000.0 * h, "cza": lift, "cxa": drag, "front_share": share})
        return {"front": front, "rear": rear, "cxa": cxa, "rows": rows, "aoa_deg": aoa, "downforce_N": q * (front + rear), "drag_N": q * cxa, "q": q}

    # ------------------------------------------------------------------------------------------------ OCD
    def fit_ocd(self, *, v_kmh=200.0, rh_f=(0.025, 0.065), rh_r=(0.055, 0.110), n=13, order=2, ridge=1e-10):
        """Polynomials of OCD's RIDE_HEIGHT aero model fitted to this aero over a box of ride heights at one speed (OCD's coefficients do not depend on speed).
        order 1: c0, c1_F, c1_R (linear); order 2 (default): plus c2_F, c2_R, c4 (quadratic with the cross term, small coefficients, no wild values outside the box); order 3: all ten terms.
        The unused coefficients are 0. Returns {coefficients (c0..c6, a0..a6), c_d (mean drag over the box), errors: {axle: max, mean |fit - AC| in m2 inside the box}, box, v_kmh, order}."""
        fs = np.linspace(rh_f[0], rh_f[1], n)
        rs = np.linspace(rh_r[0], rh_r[1], n)
        xf, xr = np.meshgrid(fs, rs, indexing="ij")
        xf, xr = xf.ravel(), xr.ravel()
        res = [self.evaluate(v_kmh, a, b) for a, b in zip(xf, xr)]
        target_f = np.array([r["front"] for r in res])
        target_r = np.array([r["rear"] for r in res])
        cxa = float(np.mean([r["cxa"] for r in res]))
        design = np.stack([np.ones_like(xf), xf, xf ** 2, xf ** 3, xr, xr ** 2, xr ** 3, xf * xr, xf ** 2 * xr, xf * xr ** 2], axis=1)
        used = {1: [0, 1, 4], 2: [0, 1, 2, 4, 5, 7], 3: list(range(10))}[order]
        out, errors = {}, {}
        for terms, target, label in ((FRONT_TERMS, target_f, "front"), (REAR_TERMS, target_r, "rear")):
            sub = design[:, used]
            scale = np.abs(sub).max(axis=0)
            scaled = sub / scale
            coef_used = np.linalg.solve(scaled.T @ scaled + ridge * np.eye(len(used)), scaled.T @ target) / scale
            coef = np.zeros(10)
            coef[used] = coef_used
            out.update({t: float(c) for t, c in zip(terms, coef)})
            err = np.abs(design @ coef - target)
            errors[label] = {"max": float(err.max()), "mean": float(err.mean())}
        return {"coefficients": out, "c_d": cxa, "errors": errors, "box": {"rh_f": list(rh_f), "rh_r": list(rh_r)}, "v_kmh": v_kmh, "order": order}


def ocd_lift(coefficients, rh_f, rh_r):
    """(front, rear) lift coefficient x area of OCD's RIDE_HEIGHT model for the fitted coefficients, clamped to [0, 5] as OCD does."""
    f, r = rh_f, rh_r
    basis = np.array([1.0, f, f ** 2, f ** 3, r, r ** 2, r ** 3, f * r, f ** 2 * r, f * r ** 2])
    cf = float(basis @ np.array([coefficients[t] for t in FRONT_TERMS]))
    cr = float(basis @ np.array([coefficients[t] for t in REAR_TERMS]))
    return min(max(cf, 0.0), 5.0), min(max(cr, 0.0), 5.0)
