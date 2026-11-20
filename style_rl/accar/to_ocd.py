"""The Assetto Corsa Ferrari SF70H as an OCD car: every number of the AC data folder that OCD can use, converted, with where it came from and how sure we are.

    replica = build_replica(AcCar(folder), fuel_l=None, compound=None, ers=False)
    replica["ocd"]          OCD parameter overrides (names as in OCD's vehicle JSON and the patched build of ocd_first_lap)
    replica["car_spec"]     fields of ocd_lap.cars.CarSpec
    replica["setup"]        knobs of ocd_lap's Setup that the backend still reads (brake bias, final drive)
    replica["powertrain"]   engine, gearbox, turbo, MGU-K, for style_rl.accar.powertrain
    replica["env"]          EnvConfig fields (steering lock, wheel positions, wheelbase)
    replica["report"]       rows (group, item, AC value, OCD parameter, OCD value, status, note) -> markdown with `report_markdown`

Status words:  exact (same number, same unit)   converted (a unit or convention change, no loss)   fitted (a curve fitted to OCD's functional form)   estimated (derived with an assumption)
               placeholder (not available, a stand-in number)   not modelled
AC's force laws are not published by Kunos; the conversions follow the commented Kunos files and two independent reimplementations of AC (see ocd_first_lap docs/stage6/ac_model.md).
"""
import math

import numpy as np

from .ini import num, vec
from .tyre_fit import axle_coefficients, fit_shape

P = "vehicle_dynamics_double_track."
SUSP = P + "suspension."
G = 9.81
FUEL_DENSITY_KG_PER_L = 0.75          # assumed, the density AC is generally taken to use
AIR_DENSITY = 1.20                    # kg/m3: AC takes it from the weather (about 1.18 at 26 C); OCD's own default is 1.225
# TUM F1_Shanghai aero numbers, used where the AC aero curves cannot be read (see report): CzA front, CzA rear, CxA at the reference ride heights 35 mm / 85 mm
AERO_PLACEHOLDER = {"cza_front": 2.20, "cza_rear": 2.68, "cxa": 1.56, "h_ref_front": 0.035, "h_ref_rear": 0.085, "per_m": 2.0}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, group, item, ac_value, ocd_key, ocd_value, status, note=""):
        self.rows.append({"group": group, "item": item, "ac": ac_value, "ocd_key": ocd_key, "ocd": ocd_value, "status": status, "note": note})


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def report_markdown(replica):
    meta = replica["meta"]
    out = ["# The Assetto Corsa Ferrari SF70H in OCD: what was copied, how, and how sure we are", ""]
    out.append(f"Source folder: `{meta['folder']}`. Compound: {meta['compound']}. Fuel: {meta['fuel_l']:g} L. Mass in OCD: {replica['ocd'][P + 'mass_vehicle_kg']:.2f} kg.")
    if meta["scrambled"]:
        out += ["", "**The file names in this folder do not match their contents** (an unpacking problem). The files were recognised by their contents:", ""]
        out += [f"- the {role} data is stored in `{disk}` (it should be `{std}`)" for role, (disk, std) in sorted(meta["name_map"].items())]
        out += ["", "The curve files (.lut) can therefore not be trusted by name. What was found by shape is listed under *Engine* and *Tyres*; the aero curves were not used (see *Aero*)."]
    out += [""]
    group = None
    for r in replica["report"]:
        if r["group"] != group:
            group = r["group"]
            out += ["", f"## {group}", "", "| Item | AC value | OCD parameter | OCD value | Status | Note |", "|---|---|---|---|---|---|"]
        out.append(f"| {r['item']} | {_fmt(r['ac'])} | `{r['ocd_key']}` | {_fmt(r['ocd'])} | {r['status']} | {r['note']} |")
    if replica["not_modelled"]:
        out += ["", "## What OCD cannot do (or we did not do)", ""] + [f"- {x}" for x in replica["not_modelled"]]
    return "\n".join(out) + "\n"


def roll_centre_height(susp, axle, wheel_radius):
    """Front-view roll centre height (m above the ground) from the double-wishbone hard points of suspensions.ini, with the convention (to be checked): hard points are given relative to the wheel
    centre as (inboard distance, height above the wheel centre, longitudinal), the chassis points of an axle are averaged over their front and rear mounts, the car is at its design height."""
    s = susp[axle]
    track = num(s["TRACK"])
    pts = {k: vec(s[k]) for k in ("WBCAR_TOP_FRONT", "WBCAR_TOP_REAR", "WBCAR_BOTTOM_FRONT", "WBCAR_BOTTOM_REAR", "WBTYRE_TOP", "WBTYRE_BOTTOM")}
    ct = (np.array(pts["WBCAR_TOP_FRONT"]) + np.array(pts["WBCAR_TOP_REAR"])) / 2
    cb = (np.array(pts["WBCAR_BOTTOM_FRONT"]) + np.array(pts["WBCAR_BOTTOM_REAR"])) / 2
    ut, ub = np.array(pts["WBTYRE_TOP"]), np.array(pts["WBTYRE_BOTTOM"])

    def line(p_car, p_up):                       # in the (inboard, height) plane: height = a + b * inboard
        b = (p_car[1] - p_up[1]) / (p_car[0] - p_up[0])
        return p_up[1] - b * p_up[0], b

    a1, b1 = line(ct, ut)
    a2, b2 = line(cb, ub)
    if abs(b1 - b2) < 1e-9:
        return None
    x_ic = (a2 - a1) / (b1 - b2)
    y_ic = a1 + b1 * x_ic
    slope = (y_ic - (-wheel_radius)) / x_ic                     # from the contact patch (0, -radius) through the instant centre
    y_rc = -wheel_radius + slope * (track / 2)                  # at the car centre line (inboard distance track / 2)
    return float(wheel_radius + y_rc)


def build_replica(car, *, fuel_l=None, compound=None, ers=False, turbo="ac"):
    rep = Report()
    ocd, notes = {}, []

    def put(key, value):
        ocd[key] = value
        return value

    # ---------------------------------------------------------------- mass and geometry
    basic = car.section("car", "BASIC")
    dry = num(basic["TOTALMASS"])
    if fuel_l is None:
        fuel_l = car.value("car", "FUEL", "FUEL", 0.0)
    mass = dry + FUEL_DENSITY_KG_PER_L * fuel_l
    put(P + "mass_vehicle_kg", mass)
    rep.add("Mass and geometry", "total mass (car with driver, no fuel, plus fuel)", f"TOTALMASS {dry:g} kg + {fuel_l:g} L x {FUEL_DENSITY_KG_PER_L} kg/L", P + "mass_vehicle_kg", mass, "converted",
            "AC's default fuel is 55 L; the fuel density 0.75 kg/L is assumed; the mass is constant over the lap (no fuel burn)")
    susp = car.ini["suspensions"]
    wheelbase = num(susp["BASIC"]["WHEELBASE"])
    front_share = num(susp["BASIC"]["CG_LOCATION"])
    lf, lr = (1.0 - front_share) * wheelbase, front_share * wheelbase
    put(P + "wheelbase_m", wheelbase)
    put(P + "cog.distance_from_front_axle_m", lf)
    rep.add("Mass and geometry", "wheelbase", f"WHEELBASE {wheelbase:g} m", P + "wheelbase_m", wheelbase, "exact")
    rep.add("Mass and geometry", "centre of gravity position", f"CG_LOCATION {front_share:g} (share of the weight on the front axle)", P + "cog.distance_from_front_axle_m", lf, "converted",
            f"{100 * front_share:.2f} % of the weight on the front axle; the CG is {lf:.3f} m behind the front axle, {lr:.3f} m in front of the rear")
    r_front, r_rear = (num(car.ini["tyres"]["FRONT"]["RADIUS"]), num(car.ini["tyres"]["REAR"]["RADIUS"]))
    h_f, h_r = r_front - num(susp["FRONT"]["BASEY"]), r_rear - num(susp["REAR"]["BASEY"])
    h_cg = 0.5 * (h_f + h_r)
    pick_f, pick_r = car.value("car", "RIDE", "PICKUP_FRONT_HEIGHT"), car.value("car", "RIDE", "PICKUP_REAR_HEIGHT")
    put(P + "cog.height_m", h_cg)
    rep.add("Mass and geometry", "centre of gravity height", f"wheel radius - BASEY = {h_f:.3f} (front), {h_r:.3f} (rear) m", P + "cog.height_m", h_cg, "estimated",
            f"BASEY is read as the CG below the wheel centre; the ride-height pickups agree (45 mm + {-pick_f:.3f} = {0.045 - pick_f:.3f} m front, 90 mm + {-pick_r:.3f} = {0.090 - pick_r:.3f} m rear)")
    tf, tr_ = num(susp["FRONT"]["TRACK"]), num(susp["REAR"]["TRACK"])
    put(P + "track_width_m.front", tf)
    put(P + "track_width_m.rear", tr_)
    rep.add("Mass and geometry", "track widths", f"TRACK {tf:g} / {tr_:g} m", P + "track_width_m.front / .rear", [tf, tr_], "exact")
    hub_f, hub_r = num(susp["FRONT"]["HUB_MASS"]), num(susp["REAR"]["HUB_MASS"])
    put(P + "mass_wheel_kg.front", hub_f)
    put(P + "mass_wheel_kg.rear", hub_r)
    rep.add("Mass and geometry", "unsprung mass per corner", f"HUB_MASS {hub_f:g} / {hub_r:g} kg", P + "mass_wheel_kg.front / .rear", [hub_f, hub_r], "exact")
    box = vec(basic["INERTIA"])                  # width, height, length of the inertia box (m)
    w_, h_, l_ = box
    ix, iy, iz = mass * (w_ ** 2 + h_ ** 2) / 12, mass * (h_ ** 2 + l_ ** 2) / 12, mass * (w_ ** 2 + l_ ** 2) / 12
    put(P + "moment_of_inertia_kgpm2.x", ix)
    put(P + "moment_of_inertia_kgpm2.y", iy)
    put(P + "moment_of_inertia_kgpm2.z", iz)
    rep.add("Mass and geometry", "body inertia (roll, pitch, yaw)", f"INERTIA box {w_:g} x {h_:g} x {l_:g} m", P + "moment_of_inertia_kgpm2.x / .y / .z", [ix, iy, iz], "estimated",
            "AC takes the inertia of a homogeneous box of the car's mass, I = m (a^2 + b^2) / 12; that is how it is read here")
    steer_lock_deg = car.value("car", "CONTROLS", "STEER_LOCK", 180.0) / car.value("car", "CONTROLS", "STEER_RATIO", 10.0)
    steer_max = math.radians(steer_lock_deg)
    rep.add("Mass and geometry", "steering lock at the road wheels", f"STEER_LOCK {car.value('car', 'CONTROLS', 'STEER_LOCK'):g} deg / STEER_RATIO {car.value('car', 'CONTROLS', 'STEER_RATIO'):g}",
            "steering_actuator.angle_max_rad", steer_max, "converted", f"{steer_lock_deg:g} degrees at the wheels; OCD's steering actuator (60 ms lag, 1.5 rad/s) is kept, AC's steering is quicker")

    # ---------------------------------------------------------------- tyres
    comps = car.tyre_compounds()
    idx = car.tyre_default_index if compound is None else compound
    comp = next(c for c in comps if c["index"] == idx)
    fr, rr = comp["front"], comp["rear"]
    rad_f, rad_r = num(fr["RADIUS"]), num(rr["RADIUS"])
    put(P + "tire.rolling_radius_m.front", rad_f)
    put(P + "tire.rolling_radius_m.rear", rad_r)
    growth = num(fr.get("RADIUS_ANGULAR_K"), 0.0) * 1e-3 * (100.0 / rad_f)       # AC: the radius grows by K millimetres per rad/s of wheel speed
    put(P + "tire.rolling_radius_m.velocity_scaling.velocity_points_mps", [0.0, 100.0])
    put(P + "tire.rolling_radius_m.velocity_scaling.factors", [1.0, 1.0 + growth / rad_f])
    rep.add("Tyres", "radius", f"RADIUS {rad_f:g} / {rad_r:g} m", P + "tire.rolling_radius_m.front / .rear", [rad_f, rad_r], "exact",
            f"plus AC's growth of the radius with wheel speed (RADIUS_ANGULAR_K): factor {1 + growth / rad_f:.4f} at 100 m/s")
    put(P + "tire.spring_stiffness_Npm.front", num(fr["RATE"]))
    put(P + "tire.spring_stiffness_Npm.rear", num(rr["RATE"]))
    rep.add("Tyres", "vertical stiffness", f"RATE {fr['RATE']} / {rr['RATE']} N/m", P + "tire.spring_stiffness_Npm.front / .rear", [num(fr["RATE"]), num(rr["RATE"])], "exact")
    put("drivetrain.I_wheel_front_kgm2", num(fr["ANGULAR_INERTIA"]))
    put("drivetrain.I_wheel_rear_kgm2", num(rr["ANGULAR_INERTIA"]))
    rep.add("Tyres", "wheel + tyre + disc spin inertia", f"ANGULAR_INERTIA {fr['ANGULAR_INERTIA']} / {rr['ANGULAR_INERTIA']} kg m2", "drivetrain.I_wheel_front_kgm2 / rear", [num(fr["ANGULAR_INERTIA"]), num(rr["ANGULAR_INERTIA"])], "exact")
    rr0, rr1 = num(fr["ROLLING_RESISTANCE_0"], 0.0), num(fr["ROLLING_RESISTANCE_1"], 0.0)
    f_roll = 4 * (rr0 + rr1 * 60.0 ** 2) / (mass * G)
    put(P + "tire.rolling_resistance_coefficient", f_roll)
    rep.add("Tyres", "rolling resistance", f"{rr0:g} N + {rr1:g} v^2 per tyre", P + "tire.rolling_resistance_coefficient", f_roll, "estimated", "OCD has a constant coefficient: AC's formula evaluated at 60 m/s")
    lat_h = car.lateral_curve_header()
    curve_notes = []
    shapes = {}
    for name, c in car.curves.items():
        if len(c) == 1000 and abs(c[0, 0]) < 1e-12 and abs(c[-1, 0] - 0.999) < 1e-4:
            C, E, rms = fit_shape(c)
            peak = float(c[np.argmax(c[:, 1]), 0])
            shapes[name] = {"C": C, "E": E, "CF": float((c[1, 1] - c[0, 1]) / (c[1, 0] - c[0, 0])), "peak": peak, "rms": rms}
    # the force curves of the tyre model: normalised force against slip (rad for the lateral one). Which curve belongs to which tyre is not stated anywhere readable: the lateral curves are taken
    # as the ones whose peak slip angle is closest to FRICTION_LIMIT_ANGLE of the front / rear tyre, the longitudinal one as the remaining curve that occurs twice (front and rear)
    ordered = sorted(shapes.items(), key=lambda kv: kv[1]["peak"])
    lim_f, lim_r = math.radians(num(fr["FRICTION_LIMIT_ANGLE"])), math.radians(num(rr["FRICTION_LIMIT_ANGLE"]))
    dup = [k for k, v in shapes.items() if sum(abs(v["CF"] - w["CF"]) < 1e-6 and abs(v["peak"] - w["peak"]) < 1e-6 for w in shapes.values()) >= 2]
    lat_pool = {k: v for k, v in shapes.items() if k not in dup}
    if len(lat_pool) >= 2:
        k_lat_f = min(lat_pool, key=lambda k: abs(lat_pool[k]["peak"] - lim_f))
        k_lat_r = min((k for k in lat_pool if k != k_lat_f), key=lambda k: abs(lat_pool[k]["peak"] - lim_r))
        lat_f, lat_r = lat_pool[k_lat_f], lat_pool[k_lat_r]
        long_curve = shapes[dup[0]] if dup else lat_pool[ordered[-1][0]]
        curve_notes.append(f"lateral front: curve {k_lat_f} (peak {math.degrees(lat_f['peak']):.1f} deg, FRICTION_LIMIT_ANGLE {math.degrees(lim_f):.1f}); lateral rear: curve {k_lat_r} "
                           f"(peak {math.degrees(lat_r['peak']):.1f} deg, limit {math.degrees(lim_r):.1f}); longitudinal: {'the curve that occurs twice' if dup else 'the flattest curve'}")
    else:
        lat_f = lat_r = {"C": 1.27, "E": -1.8, "CF": 29.0}
        long_curve = {"C": 1.5, "E": -0.5, "CF": 18.0}
        curve_notes.append("no force curves found: the stand-in values C 1.27, E -1.8, CF 29 (lateral) and C 1.5, E -0.5, CF 18 (longitudinal) are used")
    loads = {"front": (0.5 * num(fr["FZ0"]), 2.2 * num(fr["FZ0"])), "rear": (0.5 * num(rr["FZ0"]), 2.2 * num(rr["FZ0"]))}
    for axle, t, lat in (("front", fr, lat_f), ("rear", rr, lat_r)):
        coef = axle_coefficients(dy_ref=num(t["DY_REF"]), dx_ref=num(t["DX_REF"]), fz0=num(t["FZ0"]), ls_y=num(t["LS_EXPY"]), ls_x=num(t["LS_EXPX"]), fz_range=loads[axle],
                                 cf_y=lat["CF"], cf_x=long_curve["CF"], shape_y=(lat["C"], lat["E"]), shape_x=(long_curve["C"], long_curve["E"]))
        wheels = ("front_left", "front_right") if axle == "front" else ("rear_left", "rear_right")
        for w in wheels:
            for k, v in coef.items():
                put(P + f"z.tire_model.{w}.{k}", v)
        rep.add("Tyres", f"{axle} peak grip against load", f"mu = {t['DY_REF']} (Fz / {t['FZ0']}) ^ ({t['LS_EXPY']} - 1) lateral, {t['DX_REF']} (Fz / FZ0) ^ ({t['LS_EXPX']} - 1) longitudinal",
                f"z.tire_model.{axle}_*.PDY1, PDY2, PDX1, PDX2", [coef["PDY1"], coef["PDY2"], coef["PDX1"], coef["PDX2"]], "fitted",
                f"line fitted to AC's power law over {loads[axle][0]:.0f} to {loads[axle][1]:.0f} N (OCD's peak grip is linear in load); FNOMIN = FZ0 = {t['FZ0']}")
        rep.add("Tyres", f"{axle} stiffness and curve shape", f"lateral CF {lat['CF']:g}, C {lat['C']:.3g}, E {lat['E']:.3g}; longitudinal CF {long_curve['CF']:g}, C {long_curve['C']:.3g}, E {long_curve['E']:.3g}",
                f"z.tire_model.{axle}_*.PKY1, PKY2, PKX1, PCY1, PEY1, PCX1, PEX1", [coef["PKY1"], coef["PKY2"], coef["PKX1"], coef["PCY1"], coef["PEY1"], coef["PCX1"], coef["PEX1"]], "fitted",
                "stiffness = zero-slip slope of the normalised force curve x peak friction; its load dependence follows LS_EXPY. Which force curve belongs to which tyre is a guess (see below)")
    rep.add("Tyres", "which force curves", "; ".join(curve_notes), "-", "-", "estimated", "the 1000-point force curves have lost their names; ASSIGNMENT UNVERIFIED")
    put(SUSP + "camber_out_rad.front_left", 0.0)
    put(SUSP + "camber_out_rad.front_right", 0.0)
    put(SUSP + "camber_out_rad.rear_left", 0.0)
    put(SUSP + "camber_out_rad.rear_right", 0.0)
    rep.add("Tyres", "camber", "static camber -3.0 / -2.0 deg in the file; about 0 / -0.5 deg read in the game with the car standing", SUSP + "camber_out_rad.*", 0.0, "placeholder",
            "AC's double wishbone changes camber with travel and roll; OCD cannot, so 0")
    rep.add("Tyres", "compound", f"{comp['name']} (default index {car.tyre_default_index}: {next(c['name'] for c in comps if c['index'] == car.tyre_default_index)})", "-", comp["name"], "exact",
            "temperature, wear, pressure and speed sensitivity of the grip are not modelled: the tyre is at its best temperature all the time")

    # ---------------------------------------------------------------- suspension
    kf, kr = num(susp["FRONT"]["SPRING_RATE"]), num(susp["REAR"]["SPRING_RATE"])
    put(SUSP + "vehicle_spring_stiffness_Npm.front", kf)
    put(SUSP + "vehicle_spring_stiffness_Npm.rear", kr)
    rep.add("Suspension", "corner springs (wheel rate)", f"SPRING_RATE {kf:g} / {kr:g} N/m", SUSP + "vehicle_spring_stiffness_Npm.front / .rear", [kf, kr], "exact", "AC's progressive part (PROGRESSIVE_SPRING_RATE) is not modelled")
    arb_f, arb_r = num(susp["ARB"]["FRONT"]), num(susp["ARB"]["REAR"])
    put(SUSP + "antirollbar_virtual_spring_stiffness_Npm.front", 2 * arb_f)
    put(SUSP + "antirollbar_virtual_spring_stiffness_Npm.rear", 2 * arb_r)
    rep.add("Suspension", "anti-roll bars", f"[ARB] {arb_f:g} / {arb_r:g} N/m", SUSP + "antirollbar_virtual_spring_stiffness_Npm.front / .rear", [2 * arb_f, 2 * arb_r], "converted", "AC's bar acts on each hub with k, OCD's with c/2 on a pair: c = 2 k")
    hv = {a: susp["HEAVE_" + a.upper()] for a in ("front", "rear")}
    put(SUSP + "heave.enabled", 1.0)
    put(SUSP + "four_way_damper.enabled", 1.0)
    put(SUSP + "heave.four_way_damper.enabled", 1.0)
    put(SUSP + "packer.enabled", 1.0)
    for axle in ("front", "rear"):
        s = susp[axle.upper()]
        put(SUSP + f"heave.spring_stiffness_Npm.{axle}", 2 * num(hv[axle]["SPRING_RATE"]))
        bump, reb, fb, fr_ = (num(s["DAMP_BUMP"]), num(s["DAMP_REBOUND"]), num(s["DAMP_FAST_BUMP"]), num(s["DAMP_FAST_REBOUND"]))
        knee = 0.5 * (num(s["DAMP_FAST_BUMPTHRESHOLD"]) + num(s["DAMP_FAST_REBOUNDTHRESHOLD"]))
        put(SUSP + f"vehicle_damper_coefficient_Nspm.{axle}", bump)
        for name, v in (("bump_slow", bump), ("rebound_slow", reb), ("bump_fast", fb), ("rebound_fast", fr_)):
            put(SUSP + f"four_way_damper.{name}_Nspm.{axle}", v)
        put(SUSP + f"four_way_damper.knee_speed_mps.{axle}", knee)
        h = hv[axle]
        hb, hr, hfb, hfr = num(h["DAMP_BUMP"]), num(h["DAMP_REBOUND"]), num(h["DAMP_FAST_BUMP"]), num(h["DAMP_FAST_REBOUND"])
        hknee = 0.5 * (num(h["DAMP_FAST_BUMPTHRESHOLD"]) + num(h["DAMP_FAST_REBOUNDTHRESHOLD"]))
        put(SUSP + f"heave.damper_coefficient_Nspm.{axle}", 2 * hb)
        for name, v in (("bump_slow", hb), ("rebound_slow", hr), ("bump_fast", hfb), ("rebound_fast", hfr)):
            put(SUSP + f"heave.four_way_damper.{name}_Nspm.{axle}", 2 * v)
        put(SUSP + f"heave.four_way_damper.knee_speed_mps.{axle}", hknee)
        k_corner, k_heave = num(s["SPRING_RATE"]), num(h["SPRING_RATE"])
        sprung = (mass * (front_share if axle == "front" else 1 - front_share)) / 2 - num(s["HUB_MASS"])
        static = sprung * G / (k_corner + k_heave)                      # corner spring and heave spring share the load, equal travel
        gap = max(0.0, num(s["PACKER_RANGE"]) - static)
        put(SUSP + f"packer.gap_m.{axle}", gap)
        put(SUSP + f"packer.stiffness_Npm.{axle}", num(s["BUMP_STOP_RATE"]))
        rep.add("Suspension", f"{axle} corner damper", f"bump {bump:g}, rebound {reb:g}, fast bump {fb:g}, fast rebound {fr_:g} N s/m, thresholds {s['DAMP_FAST_BUMPTHRESHOLD']} / {s['DAMP_FAST_REBOUNDTHRESHOLD']} m/s",
                SUSP + f"four_way_damper.*.{axle}", [bump, reb, fb, fr_, knee], "exact", "same law in OCD (patched build); one knee speed for bump and rebound: the mean of the two thresholds")
        rep.add("Suspension", f"{axle} heave spring and damper", f"SPRING_RATE {num(h['SPRING_RATE']):g} N/m, damper {hb:g}/{hr:g}/{hfb:g}/{hfr:g}", SUSP + f"heave.spring_stiffness_Npm.{axle}",
                [2 * num(h["SPRING_RATE"]), 2 * hb, 2 * hr, 2 * hfb, 2 * hfr], "converted", "AC applies the heave force to each hub, OCD half the axle force on each: all heave numbers x 2; progressive rate and heave packer not modelled")
        rep.add("Suspension", f"{axle} packers (bump stops)", f"PACKER_RANGE {s['PACKER_RANGE']} m, BUMP_STOP_RATE {s['BUMP_STOP_RATE']} N/m", SUSP + f"packer.gap_m.{axle} / stiffness",
                [gap, num(s["BUMP_STOP_RATE"])], "converted", f"gap = range - static compression ({1000 * static:.1f} mm, corner and heave spring in parallel); the upper bump stop (BUMPSTOP_UP/DN) is not modelled")
    rc_f, rc_r = roll_centre_height(susp, "FRONT", r_front), roll_centre_height(susp, "REAR", r_rear)
    rc = 0.5 * (rc_f + rc_r) if rc_f is not None and rc_r is not None else 0.1
    put(SUSP + "roll_center_height_m", rc)
    rep.add("Suspension", "roll centre height", f"from the wishbone hard points: front {rc_f:.3f} m, rear {rc_r:.3f} m" if rc_f is not None and rc_r is not None else "not computable", SUSP + "roll_center_height_m", rc, "estimated",
            "OCD has one roll centre height for both axles: the mean; camber gain, bump steer and the rest of the double-wishbone geometry are not modelled")
    ride_f, ride_r = 0.045, 0.090
    put(P + "aerodynamics.static_ride_height_front_m", -ride_f)
    put(P + "aerodynamics.static_ride_height_rear_m", -ride_r)
    rep.add("Suspension", "ride heights", "about 45 mm front, 90 mm rear (read in the game with the default setup)", P + "aerodynamics.static_ride_height_front_m / rear", [-ride_f, -ride_r], "converted",
            "OCD's convention: static ride height h is passed as -h")

    # ---------------------------------------------------------------- aero
    a = AERO_PLACEHOLDER
    put(P + "aerodynamics.air_density_kgpm3", AIR_DENSITY)
    put(P + "aerodynamics.A_m2", 1.0)
    put(P + "aerodynamics.c_d", a["cxa"])
    put(P + "aerodynamics.l", wheelbase)
    put(P + "aerodynamics.l_f", lf)
    put(P + "aerodynamics.c0", a["cza_front"] * (1 + a["per_m"] * a["h_ref_front"]))
    put(P + "aerodynamics.c1_F", -a["cza_front"] * a["per_m"])
    put(P + "aerodynamics.a0", a["cza_rear"] * (1 + a["per_m"] * a["h_ref_rear"]))
    put(P + "aerodynamics.a1_R", -a["cza_rear"] * a["per_m"])
    for k in ("c2_F", "c3_F", "c1_R", "c2_R", "c3_R", "c4", "c5", "c6", "a1_F", "a2_F", "a3_F", "a2_R", "a3_R", "a4", "a5", "a6"):
        put(P + f"aerodynamics.{k}", 0.0)
    wings = [(k, v["NAME"], num(v["CHORD"]) * num(v["SPAN"]), num(v["ANGLE"]), num(v["CL_GAIN"]), num(v["CD_GAIN"])) for k, v in sorted(car.ini["aero"].items()) if k.startswith(("WING_", "FIN_"))]
    rep.add("Aero", "wings in AC", "; ".join(f"{n} {a_:.2f} m2 angle {ang:g} CLgain {cl:g} CDgain {cd:g}" for _, n, a_, ang, cl, cd in wings), "-", "-", "not modelled",
            "downforce comes from the wing LUT curves (angle -> CL, CD; height -> multiplier) whose files lost their names: they could not be matched, so the aero is NOT AC's yet")
    rep.add("Aero", "downforce and drag (stand-in)", f"TUM F1_Shanghai: CzA {a['cza_front']} front + {a['cza_rear']} rear, CxA {a['cxa']} m2, +{a['per_m']:g} % per cm below the reference height",
            P + "aerodynamics.c0, c1_F, a0, a1_R, c_d", [ocd[P + "aerodynamics.c0"], ocd[P + "aerodynamics.c1_F"], ocd[P + "aerodynamics.a0"], ocd[P + "aerodynamics.a1_R"], a["cxa"]], "placeholder",
            "the same stand-in as ocd_first_lap's car; replace it by AC's own wing model when the curve files are named correctly")

    # ---------------------------------------------------------------- brakes
    brk = car.ini["brakes"]["DATA"]
    max_t, share = num(brk["MAX_TORQUE"]), num(brk["FRONT_SHARE"])
    total = 4 * max_t
    rep.add("Brakes", "brake torque", f"MAX_TORQUE {max_t:g} N m", "CarSpec.brake_torque_Nm (total, all four wheels)", total, "assumed",
            "read as the torque of ONE wheel at full pedal (4 x 3900 = 15.6 kN m, which is what an F1 car needs: ocd_first_lap uses 15 kN m); if AC means the whole car this is 4 times too strong. Check in AC with a braking test")
    rep.add("Brakes", "front brake share (bias)", f"FRONT_SHARE {share:g} (setup range 52 to 68 %)", "Setup.brake_bias_front", share, "exact", "default setup value")

    # ---------------------------------------------------------------- drivetrain and engine
    dr = car.ini["drivetrain"]
    gears = [num(dr["GEARS"][f"GEAR_{i}"]) for i in range(1, int(num(dr["GEARS"]["COUNT"])) + 1)]
    final = num(dr["GEARS"]["FINAL"])
    put("drivetrain.LSD_lock_drive", num(dr["DIFFERENTIAL"]["POWER"]))
    put("drivetrain.LSD_lock_coast", num(dr["DIFFERENTIAL"]["COAST"]))
    put("drivetrain.LSD_preload", num(dr["DIFFERENTIAL"]["PRELOAD"]))
    rep.add("Drivetrain", "differential", f"POWER {dr['DIFFERENTIAL']['POWER']}, COAST {dr['DIFFERENTIAL']['COAST']}, PRELOAD {dr['DIFFERENTIAL']['PRELOAD']}", "drivetrain.LSD_lock_drive / lock_coast / preload",
            [ocd["drivetrain.LSD_lock_drive"], ocd["drivetrain.LSD_lock_coast"], ocd["drivetrain.LSD_preload"]], "exact", "OCD's LSD uses the same 0..1 lock scale (to be confirmed by a test)")
    rep.add("Drivetrain", "gears", f"{len(gears)} gears {', '.join(f'{g:g}' for g in gears)}, final drive {final:g}", "powertrain.gears, final", gears + [final], "exact",
            f"overall ratios {gears[0] * final:.2f} (1st) to {gears[-1] * final:.2f} ({len(gears)}th)")
    shifter = dr.get("AUTO_SHIFTER", {})
    shift_up, shift_down = num(shifter.get("UP"), 11500.0), num(shifter.get("DOWN"), 7000.0)
    en = car.ini["engine"]
    limiter = num(en["ENGINE_DATA"]["LIMITER"])
    curves, curve_found = car.engine_curves()
    if "engine" not in curves:
        raise ValueError("no engine torque curve found in this folder")
    tb = en.get("TURBO_0", {})
    turbo_p = {"max_boost": num(tb.get("MAX_BOOST"), 0.0), "wastegate": num(tb.get("WASTEGATE"), 0.0), "reference_rpm": num(tb.get("REFERENCE_RPM"), 5500.0), "gamma": num(tb.get("GAMMA"), 1.0),
               "lag_up": num(tb.get("LAG_UP"), 0.0), "lag_dn": num(tb.get("LAG_DN"), 0.0), "physics_hz": 333.0}
    t0 = curves["engine"]
    boost_full = min(turbo_p["wastegate"] or turbo_p["max_boost"], turbo_p["max_boost"])
    p_ice = t0[:, 1] * (1 + boost_full) * t0[:, 0] * 2 * math.pi / 60
    i_pk = int(np.argmax(p_ice))
    rep.add("Engine", "torque curve", f"POWER_CURVE ({'; '.join(curve_found)})", "powertrain.engine_curve", f"{len(t0)} points, {t0[:, 1].max():g} Nm base torque", "estimated",
            "found by its shape (rises to a peak at 10000 rpm and falls); the file name is not trustworthy in this folder")
    rep.add("Engine", "turbo", f"MAX_BOOST {turbo_p['max_boost']:g}, WASTEGATE {turbo_p['wastegate']:g}, REFERENCE_RPM {turbo_p['reference_rpm']:g}, GAMMA {turbo_p['gamma']:g}, LAG {turbo_p['lag_up']:g} / {turbo_p['lag_dn']:g}",
            "powertrain.turbo", turbo_p, "estimated",
            f"torque = base curve x pedal x (1 + boost), boost -> {boost_full:g} x pedal^{turbo_p['gamma']:g} (x rpm / reference below it) with AC's lag; the formulas are read from AC's comments and reimplementations. "
            f"Gives {p_ice[i_pk] / 1e3:.0f} kW at {t0[i_pk, 0]:g} rpm for the combustion engine at full pedal")
    rep.add("Engine", "rev limiter, auto shift", f"LIMITER {limiter:g} rpm, [AUTO_SHIFTER] up {shift_up:g}, down {shift_down:g} rpm", "powertrain.limiter / shift_up / shift_down", [limiter, shift_up, shift_down], "exact",
            "the gearbox shifts by AC's automatic shifter rule; shifts take no time (AC: 15 to 20 ms)")
    ers_p = None
    if "kers" in curves and "ers" in car.ini:
        ers_dat = car.ini["ers"]["KINETIC"]
        ers_p = {"energy_per_lap_J": num(ers_dat["MAX_KJ_PER_LAP"]) * 1e3, "discharge_time_s": num(ers_dat["DISCHARGE_TIME"]) / 1e3, "torque_curve": curves["kers"].tolist()}
    k_note = f"{ers_p['energy_per_lap_J'] / 1e6:g} MJ per lap, empty after {ers_p['discharge_time_s']:.1f} s of full deployment, torque up to {max(c[1] for c in ers_p['torque_curve']):g} Nm" if ers_p else ""
    rep.add("Engine", "electric motor (MGU-K)", f"MAX_KJ_PER_LAP {car.value('ers', 'KINETIC', 'MAX_KJ_PER_LAP', 0):g}, torque curve {'found' if 'kers' in curves else 'not found'}", "powertrain.ers",
            ("on at run time with the option ers=1: " if ers_p else "") + (k_note or "none"), "estimated" if ers_p else "not modelled",
            "OFF by default: the policy cannot see the battery, and AC's deployment strategies (the ERS controllers) are not copied: full pedal deploys; MGU-H not modelled")
    eng_inertia = num(en["ENGINE_DATA"]["INERTIA"], 0.05) + num(dr["GEARBOX"].get("INERTIA"), 0.02)
    rep.add("Engine", "engine + gearbox inertia", f"{en['ENGINE_DATA']['INERTIA']} + {dr['GEARBOX'].get('INERTIA')} kg m2", "powertrain.engine_inertia", eng_inertia, "exact", "reflected to the wheels by the gear ratio squared")
    el = car.ini["electronics"]
    rep.add("Electronics", "traction control and ABS", f"TC PRESENT {el['TRACTION_CONTROL'].get('PRESENT')}, ABS PRESENT {el['ABS'].get('PRESENT')}", "-", "none", "exact", "the car has neither as a driver aid in this data; OCD has none either")

    powertrain = {"gears": gears, "final": final, "limiter_rpm": limiter, "shift_up_rpm": shift_up, "shift_down_rpm": shift_down, "engine_curve": t0.tolist(), "turbo": turbo_p, "turbo_model": turbo,
                  "engine_inertia_kgm2": eng_inertia, "efficiency": 1.0, "ers": ers_p, "wheel_radius_m": rad_r}
    car_spec = {"name": "ac_sf70h", "rear_axle_m": lr, "wheelbase_m": wheelbase, "steer_max_rad": steer_max, "steer_rate_max_radps": 1.5, "brake_torque_Nm": total,
                "footprint": [[lf + 0.5, 1.0], [lf + 0.5, -1.0], [-lr - 0.5, -1.0], [-lr - 0.5, 1.0]], "description": "Assetto Corsa Ferrari SF70H (data.acd) as an OCD car"}
    env = {"steer_lock_rad": steer_max, "wheel_half_base_m": 0.5 * wheelbase, "wheel_half_track_m": 0.5 * 0.5 * (tf + tr_), "wheelbase_m": wheelbase}
    not_modelled = ["AC's double-wishbone geometry: camber gain, bump steer, roll and pitch centres (OCD has one fixed roll centre per car, taken from the hard points, and fixed pitch centres)",
                    "tyre temperature, wear, pressure and speed sensitivity; grain and blister",
                    "progressive spring rates; the upper bump stops and the heave packers",
                    "DRS and the dynamic wing angles (steering, speed, throttle controllers)",
                    "the MGU-H, the ERS deployment strategies, engine braking, shift times",
                    "AC's aero curves (see the Aero section)"]
    return {"meta": {"folder": str(car.folder), "scrambled": car.scrambled, "name_map": car.name_map(), "compound": comp["name"], "fuel_l": fuel_l, "ers_default": bool(ers), "turbo_default": turbo,
                     "warnings": list(car.warnings), "found_by_shape": curve_found},
            "ocd": ocd, "car_spec": car_spec, "setup": {"brake_bias_front": share, "final_drive_ratio": 1.0}, "powertrain": powertrain, "env": env,
            "report": rep.rows, "not_modelled": not_modelled}
