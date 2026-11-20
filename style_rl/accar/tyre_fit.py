"""Fits of the Assetto Corsa tyre model to the MF5.2 tyre of OCD (Pacejka 2002 MF-Tyre 5.2, as in OCD's mf_52.cpp).

AC (tyres.ini):  peak friction  mu(Fz) = DY_REF (Fz / FZ0) ^ (LS_EXPY - 1)    (x: DX_REF, LS_EXPX)   -- the law of the reverse-engineered AC physics (rustyAC, projectd-core), see ocd_first_lap docs/stage6/ac_model.md
                 force curve     F / F_peak = sin(C atan(B s - E (B s - atan(B s)))), slope B C = CF at zero slip (the [VALUES] C E CF header of a curve file, or a fit to its 1000 points)
MF5.2:           peak friction   mu = PDY1 + PDY2 dfz,  dfz = (Fz - FNOMIN) / FNOMIN          (linear in load: fitted over the load range instead of matched at one point)
                 stiffness       K_y = PKY1 FNOMIN sin(2 atan(Fz / (PKY2 FNOMIN)))             (PKY2 chosen so that d ln K / d ln Fz = LS_EXPY at FNOMIN, PKY1 for K_y(FNOMIN))
                 shape           C = PCY1, E = PEY1 (the other E terms 0)
"""
import math

import numpy as np
from scipy.optimize import least_squares

WHEELS = ("front_left", "front_right", "rear_left", "rear_right")


def ac_mu(fz, ref, fz0, ls_exp):
    return ref * (np.asarray(fz, float) / fz0) ** (ls_exp - 1.0)


def fit_peak_friction(ref, fz0, ls_exp, fz_range):
    """(PDY1, PDY2): least-squares line through AC's power law over the load range (N), dfz measured from FNOMIN = fz0."""
    fz = np.linspace(fz_range[0], fz_range[1], 200)
    dfz = (fz - fz0) / fz0
    a = np.column_stack([np.ones_like(dfz), dfz])
    coef, *_ = np.linalg.lstsq(a, ac_mu(fz, ref, fz0, ls_exp), rcond=None)
    return float(coef[0]), float(coef[1])


def stiffness_params(k_target_at_fnomin, fnomin, ls_exp):
    """(PKY1, PKY2) of K = PKY1 FNOMIN sin(2 atan(Fz / (PKY2 FNOMIN))) with K(FNOMIN) = k_target and the log-slope ls_exp there. The slope of this K at x = Fz / (PKY2 FNOMIN) is (1 - x^2) / (1 + x^2)."""
    n = float(np.clip(ls_exp, 0.05, 0.98))
    x = math.sqrt((1.0 - n) / (1.0 + n))
    return k_target_at_fnomin / (fnomin * math.sin(2.0 * math.atan(x))), 1.0 / x


def mf_curve(s, B, C, E):
    bs = B * s
    return np.sin(C * np.arctan(bs - E * (bs - np.arctan(bs))))


def fit_shape(curve, slope0=None, s_max=0.6):
    """(C, E) of the MF shape that best follows a normalised force curve (an (n, 2) array of slip | F / F_peak); B = slope0 / C (the zero-slip slope of the curve if not given)."""
    s, y = curve[:, 0], curve[:, 1]
    m = (s > 0) & (s <= s_max)
    if slope0 is None:
        slope0 = float((y[1] - y[0]) / (s[1] - s[0]))

    def resid(p):
        C, E = p
        return mf_curve(s[m], slope0 / C, C, E) - y[m]

    best = None
    for c0, e0 in ((1.3, -1.0), (1.6, 0.0), (1.1, -3.0), (1.9, 0.5)):
        r = least_squares(resid, [c0, e0], bounds=([1.0, -10.0], [2.5, 1.0]))
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(best.x[1]), float(np.sqrt(2 * best.cost / m.sum()))


def axle_coefficients(*, dy_ref, dx_ref, fz0, ls_y, ls_x, fz_range, cf_y, cf_x, shape_y, shape_x):
    """The MF5.2 coefficients (name -> value) of the tyres of one axle. shape_* = (C, E). Zero-slip side-force offsets and the slip-ratio side force are set to 0 as in AC (the mirrored MF5.2
    offsets make load transfer raise an axle's side force at small slip, which AC's tyre does not do)."""
    pdy1, pdy2 = fit_peak_friction(dy_ref, fz0, ls_y, fz_range)
    pdx1, pdx2 = fit_peak_friction(dx_ref, fz0, ls_x, fz_range)
    ky = cf_y * float(ac_mu(fz0, dy_ref, fz0, ls_y)) * fz0
    pky1, pky2 = stiffness_params(ky, fz0, ls_y)
    kx_over_fz = cf_x * float(ac_mu(fz0, dx_ref, fz0, ls_x))
    params = {"FNOMIN": fz0, "LFZO": 1.0, "LMUX": 1.0, "LMUY": 1.0,
              "PDY1": pdy1, "PDY2": pdy2, "PDX1": pdx1, "PDX2": pdx2,
              "PKY1": pky1, "PKY2": pky2, "PKY4": 2.0, "PCY1": shape_y[0], "PEY1": shape_y[1],
              "PKX1": kx_over_fz, "PKX2": 0.0, "PKX3": 0.0, "PCX1": shape_x[0], "PEX1": shape_x[1]}
    for k in ("PEY2", "PEY3", "PEY4", "PEY5", "PEX2", "PEX3", "PEX4", "PHY1", "PHY2", "PVY1", "PVY2", "RVY1", "RVY2"):
        params[k] = 0.0
    return params
