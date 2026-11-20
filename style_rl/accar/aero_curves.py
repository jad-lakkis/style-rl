"""The curve (.lut) files the aero of an Assetto Corsa car needs: found by name when the folder is named right, and by what the curves look like when the names are shifted against the contents
(the SF70H folder we were given: the file called wing_front_AOA_CL.lut holds something else).

How the shape identification works, for the SF70H-type aero.ini (body, front wings, rear wing, diffusers, DRS flap, monkey seat, fins and dynamic controllers):
  - aero.ini says which curve every wing and controller needs (by its true name) and what the curve is used for (angle -> CL or CD of a wing; ground height -> multiplier; a controller input -> angle),
    including the limits of each controller. Every role has a recognisable domain (an angle table starts at a negative angle, a height table starts at 0 m and ends below 0.2 m, a speed table runs to
    hundreds of km/h) and the roles with the same domain differ in shape (a stalling diffuser table, a large body drag, a speed multiplier that only falls a little ...).
  - the one real ambiguity, which of the two long wing table pairs belongs to the front wings and which to the rear wing, is decided by the aero balance: the designers tuned the
    gains so that the downforce carried by the front axle equals the weight share of the front axle, and only one of the two assignments does that.
  - every curve found says how (`rule`) and how sure that is (`confidence`): "sure" (the only curve of its kind), "by balance", "little effect" (the choice changes the aero by less than 2 %).
"""
import numpy as np

from .aero import AcAero, car_geometry, referenced_luts


class AeroCurveError(ValueError):
    pass


def _row(role, stored, rule, confidence):
    return {"role": role, "stored_in": stored, "rule": rule, "confidence": confidence}


def _at(a, x):
    return float(np.interp(x, a[:, 0], a[:, 1]))


def _same_grid(a, b):
    return len(a) == len(b) and bool(np.allclose(a[:, 0], b[:, 0]))


def find_aero_curves(car):
    """({lut name as aero.ini writes it: (n, 2) array}, notes). Raises AeroCurveError when a curve cannot be found or identified."""
    if "aero" not in car.ini:
        raise AeroCurveError("no aero.ini found in this folder")
    refs = referenced_luts(car.ini["aero"])
    if not refs:
        raise AeroCurveError("aero.ini names no .lut files (no curves to read)")
    if not car.scrambled:
        lower = {n.lower(): n for n in car.curves}
        missing = [r for r in refs if r.lower() not in lower]
        if missing:
            raise AeroCurveError("the aero curve files " + ", ".join(sorted(missing)) + " are not in this folder")
        return ({r: car.curves[lower[r.lower()]] for r in refs}, [_row(r, lower[r.lower()], "by name", "sure") for r in sorted(refs)])
    return _identify_by_shape(car, refs)


def _controllers(aero):
    """{lut name: {input, comb, up, down, wings, base_angle}} of the dynamic controllers of aero.ini"""
    out = {}
    for key, sec in aero.items():
        if not key.startswith("DYNAMIC_CONTROLLER_"):
            continue
        lut = sec["LUT"].split()[0]
        wing = aero.get(f"WING_{int(float(sec['WING']))}", {})
        info = out.setdefault(lut, {"input": sec["INPUT"].split()[0].upper(), "comb": sec.get("COMBINATOR", "").split()[0].upper(), "up": float(sec.get("UP_LIMIT", 1e9)),
                                    "down": float(sec.get("DOWN_LIMIT", -1e9)), "base_angles": []})
        info["base_angles"].append(float(wing.get("ANGLE", 0.0)))
    return out


def _identify_by_shape(car, refs):
    aero = car.ini["aero"]
    pool = {n: a for n, a in car.curves.items() if 2 <= len(a) <= 100}
    found, notes = {}, []

    def take(pred, what, role, rule, confidence="sure", many=1, among=None):
        names = [n for n, a in pool.items() if (among is None or n in among) and pred(a)]
        if len(names) != many:
            raise AeroCurveError(f"{role}: {len(names)} candidate curves ({', '.join(names) or 'none'}) where {many} expected ({what})")
        for n in names:
            pool.pop(n)
        return names

    def give(role, stored, rule, confidence):
        if role in refs:
            found[role] = car.curves[stored]
            notes.append(_row(role, stored, rule, confidence))

    # ---------------------------------------------------------------- dynamic controllers (inputs and limits from aero.ini)
    ctrl = _controllers(aero)
    mult_speed = []
    for lut, c in sorted(ctrl.items()):
        inp, comb, up = c["input"], c["comb"], c["up"]
        if inp == "STEER":
            (n,) = take(lambda a: abs(a[0, 0] + 1) < 1e-6 and abs(a[-1, 0] - 1) < 1e-6, "steering table, input -1 .. 1", lut, "")
            give(lut, n, "steering input table (-1 .. +1) used by the front diffuser", "sure")
        elif inp == "SPEED_KMH" and comb == "ADD":
            (n,) = take(lambda a: a[0, 0] == 0 and a[-1, 0] >= 150 and abs(a[:, 1].max() - up) < 1e-6, "speed table whose largest angle is the controller's UP_LIMIT", lut, "")
            give(lut, n, f"speed table ADDing angle, its largest value {up:g} equals the controller's UP_LIMIT", "sure")
        elif inp == "SPEED_KMH" and comb == "MULT":
            mult_speed.append(lut)
        elif inp == "GAS" and comb == "ADD":
            (n,) = take(lambda a: a[0, 0] == 0 and 0.5 <= a[-1, 0] <= 1.05 and abs(a[:, 1].max() - up) < 1e-6, "gas table whose largest angle is the controller's UP_LIMIT", lut, "")
            give(lut, n, f"gas table ADDing angle, its largest value {up:g} equals the controller's UP_LIMIT", "sure")
        elif inp == "GAS" and comb == "MULT":
            (n,) = take(lambda a: a[0, 0] == 0 and 0.5 <= a[-1, 0] <= 1.05 and a[:, 1].min() >= 0.85 and a[:, 1].max() <= 1.05 and np.all(np.diff(a[:, 1]) <= 1e-9), "gas multiplier near 1", lut, "")
            give(lut, n, "gas table MULTiplying the rear wing angle: close to 1, never rising", "sure")
        else:
            raise AeroCurveError(f"controller {lut}: input {inp} / {comb} not known to the shape identification")
    if mult_speed:
        # speed multipliers: tables starting at 1 and never rising; the gentle one belongs to the front and rear wings, the deep one to the monkey seat
        names = take(lambda a: a[0, 1] == 1.0 and a[-1, 0] >= 150 and np.all(np.diff(a[:, 1]) <= 1e-9) and a[0, 0] <= 250, "speed multipliers starting at 1", "speed multipliers", "", many=len(mult_speed) + 1)
        names.sort(key=lambda n: -car.curves[n][:, 1].min())
        by_angle = sorted(mult_speed, key=lambda lut: -float(np.mean(ctrl[lut]["base_angles"])))      # the controller of the wings with the large angles first
        give(by_angle[0], names[0], "speed multiplier that falls only a little: the front and rear wings (large angles)", "sure")
        give(by_angle[1], names[-1], "speed multiplier that falls a lot: the monkey seat (small angle)", "sure")
        notes.append(_row("(second copy of the gentle speed multiplier)", names[1], "a second file with exactly the same table (an unused controller table of the car)", "little effect"))

    # ---------------------------------------------------------------- angle tables (angle in degrees -> CL or CD)
    fin_cd, = take(lambda a: a[0, 0] <= -80 and a[-1, 0] >= 80, "fin drag table, +-90 degrees", "fin_AOA_CD", "")
    fin_cl, = take(lambda a: -60 <= a[0, 0] <= -30 and 30 <= a[-1, 0] <= 60, "fin lift table, +-40 degrees", "fin_AOA_CL", "")
    for role, n, text in (("fin_AOA_CD.lut", fin_cd, "table over +-90 degrees: flat-plate drag"), ("fin_AOA_CL.lut", fin_cl, "table over +-40 degrees, odd about zero: side force")):
        give(role, n, text, "sure")
    angle_tables = lambda a: -12 <= a[0, 0] <= -8 and a[-1, 0] >= 8 and a[0, 0] < 0 < a[-1, 0]
    in_range = lambda a: a[(a[:, 0] >= -2) & (a[:, 0] <= 10), 1]
    body_cd, = take(lambda a: angle_tables(a) and len(in_range(a)) >= 3 and bool(np.all(in_range(a) >= 0.4)), "body drag: large at small angles", "wing_body_AOA_CD", "")
    give("wing_body_AOA_CD.lut", body_cd, "the only angle table with a large drag at small angles: the car body", "sure")
    diff_cl, = take(lambda a: angle_tables(a) and a[-1, 0] <= 12 and a[a[:, 0] >= 0, 1].max() >= 0.8, "diffuser lift, stalls early", "wing_diffuser_AOA_CL", "")
    give("wing_diffuser_AOA_CL.lut", diff_cl, "the only short angle table with a high lift peak that stalls early: the diffusers", "sure")
    long_tables = {n: a for n, a in pool.items() if angle_tables(a) and len(a) >= 20 and a[-1, 0] >= 24}
    if len(long_tables) != 4:
        raise AeroCurveError(f"{len(long_tables)} long angle tables found where 4 (two wings, lift and drag each) expected")
    groups = []
    for n, a in long_tables.items():
        for g in groups:
            if _same_grid(long_tables[g[0]], a):
                g.append(n)
                break
        else:
            groups.append([n])
    if sorted(len(g) for g in groups) != [2, 2]:
        raise AeroCurveError("the four long angle tables do not form two lift/drag pairs on the same angle grids")
    pairs = []
    for g in groups:
        g.sort(key=lambda n: -_at(long_tables[n], 14))                                      # lift (large) first, drag second
        pairs.append(tuple(g))
        for n in g:
            pool.pop(n)
    rear2_cl, = take(lambda a: angle_tables(a) and len(a) <= 5 and a[-1, 0] >= 14 and _at(a, 15) >= 0.3, "DRS flap lift at 15 degrees", "wing_rear2_AOA_CL", "")
    rear2_cd, = take(lambda a: angle_tables(a) and len(a) <= 8 and a[-1, 0] >= 16, "DRS flap drag, tabulated just beyond the flap angle", "wing_rear2_AOA_CD", "")
    monkey_cl, = take(lambda a: angle_tables(a) and bool(np.all(np.abs(a[(a[:, 0] >= 0) & (a[:, 0] <= 8), 1]) < 0.01)) and _at(a, 10) >= 0.1, "monkey seat lift: zero at small angles, rising at larger ones", "wing_monkey_AOA_CL", "")
    monkey_cd, = take(lambda a: angle_tables(a) and _same_grid(a, car.curves[monkey_cl]), "monkey seat drag, on the same grid as its lift table", "wing_monkey_AOA_CD", "")
    rest = sorted(n for n, a in pool.items() if angle_tables(a))
    body_cl = [n for n in rest if len(car.curves[n]) == 3]
    diff_cd = [n for n in rest if n not in body_cl]
    if len(body_cl) != 1 or len(diff_cd) != 1:
        raise AeroCurveError(f"after the other angle tables {len(rest)} remain where 2 expected (body lift, diffuser drag)")
    for n in rest:
        pool.pop(n)

    # ---------------------------------------------------------------- height tables (ground height in m -> multiplier)
    height = lambda a: a[0, 0] == 0 and a[-1, 0] <= 0.2
    gh_diff_cl, = take(lambda a: height(a) and _at(a, 0.02) < 0.05 and _at(a, 0.05) >= 0.9 and _at(a, 0.1) <= 0.6, "front diffuser height table: nothing near the road, then a plateau", "height_diffuser_CL", "")
    near_one = lambda a: height(a) and bool(np.all((a[(a[:, 0] >= 0.01) & (a[:, 0] <= 0.06), 1] > 0.9) & (a[(a[:, 0] >= 0.01) & (a[:, 0] <= 0.06), 1] < 1.1)))
    gh_cd_a, gh_cd_b = take(near_one, "two drag height tables close to 1", "height_*_CD", "", many=2)
    gh_front_cl, = take(lambda a: height(a) and 0.4 <= _at(a, 0.01) <= 0.6 and _at(a, 0.02) >= 0.95, "front wing height table: about half at 1 cm, full at 2 cm", "height_frontwing_CL", "")
    gh_rdiff_cl, = take(lambda a: height(a) and _at(a, 0.01) >= 1.0, "rear diffuser height table: above 1 from 1 cm", "height_diffuser_rear_CL", "")
    gh_diff_cd, = take(height, "the remaining height table", "height_diffuser_CD", "")

    # ---------------------------------------------------------------- the two long wing pairs: which is the front wing, which the rear wing (by the aero balance)
    chosen = {}
    base = {"wing_body_AOA_CD.lut": body_cd, "wing_body_AOA_CL.lut": body_cl[0], "wing_diffuser_AOA_CL.lut": diff_cl, "wing_diffuser_AOA_CD.lut": diff_cd[0], "wing_rear2_AOA_CL.lut": rear2_cl,
            "wing_rear2_AOA_CD.lut": rear2_cd, "wing_monkey_AOA_CL.lut": monkey_cl, "wing_monkey_AOA_CD.lut": monkey_cd, "fin_AOA_CL.lut": fin_cl, "fin_AOA_CD.lut": fin_cd,
            "height_diffuser_CL.lut": gh_diff_cl, "height_diffuser_CD.lut": gh_diff_cd, "height_frontwing_CL.lut": gh_front_cl, "height_frontwing_CD.lut": gh_cd_a,
            "height_diffuser_rear_CL.lut": gh_rdiff_cl, "height_diffuser_rear_CD.lut": gh_cd_b}
    geo = car_geometry(car)
    target = 1.0 - geo.lf / geo.wheelbase                                                       # the front axle's share of the weight
    shares = []
    for front_pair, rear_pair in (pairs, pairs[::-1]):
        trial = dict(found)
        for role, stored in base.items():
            if role in refs:
                trial[role] = car.curves[stored]
        trial.update({"wing_front_AOA_CL.lut": car.curves[front_pair[0]], "wing_front_AOA_CD.lut": car.curves[front_pair[1]],
                      "wing_rear_AOA_CL.lut": car.curves[rear_pair[0]], "wing_rear_AOA_CD.lut": car.curves[rear_pair[1]]})
        missing = [r for r in refs if r not in trial]
        if missing:
            raise AeroCurveError("curves not identified: " + ", ".join(sorted(missing)))
        r = AcAero.from_car(car, trial, geo).evaluate(150.0, *geo.static_ride)
        shares.append((abs(r["front"] / (r["front"] + r["rear"]) - target), r["front"] / (r["front"] + r["rear"]), front_pair, rear_pair))
    shares.sort(key=lambda s: s[0])
    best, other = shares[0], shares[1]
    if other[0] - best[0] < 0.05:
        raise AeroCurveError(f"the aero balance does not decide between the two wing assignments ({best[1]:.3f} and {other[1]:.3f} front share, target {target:.3f})")
    front_pair, rear_pair = best[2], best[3]
    why = f"front share of the downforce {best[1]:.3f} with this assignment ({other[1]:.3f} with the other); the front axle carries {target:.3f} of the weight"
    chosen.update({"wing_front_AOA_CL.lut": front_pair[0], "wing_front_AOA_CD.lut": front_pair[1], "wing_rear_AOA_CL.lut": rear_pair[0], "wing_rear_AOA_CD.lut": rear_pair[1]})
    for role, stored in chosen.items():
        give(role, stored, why, "by balance")
    little = {"wing_body_AOA_CL.lut", "wing_diffuser_AOA_CD.lut", "wing_monkey_AOA_CD.lut", "height_frontwing_CD.lut", "height_diffuser_rear_CD.lut", "height_diffuser_CD.lut"}
    rules = {"wing_body_AOA_CL.lut": "the 3-point lift table left over (the body has CL_GAIN 0)", "wing_diffuser_AOA_CD.lut": "the angle table left over (a small drag)",
             "wing_rear2_AOA_CL.lut": "3-point table with the same lift at 15 degrees as the rear wing table (one family)",
             "wing_rear2_AOA_CD.lut": "only angle table tabulated beyond the flap angle; with it opening the DRS cuts the drag by a realistic share",
             "wing_monkey_AOA_CL.lut": "zero lift at small angles, rising where the gas controller puts the monkey seat",
             "wing_monkey_AOA_CD.lut": "the table on the same angle grid as the monkey seat lift table",
             "height_diffuser_CL.lut": "no lift near the road, a plateau, then falling: a diffuser that stalls when too low",
             "height_frontwing_CL.lut": "about half at 1 cm and full at 2 cm (the shape of the front wing ground-effect tables of the other Kunos cars)", "height_diffuser_rear_CL.lut": "the table that is above 1 from 1 cm",
             "height_frontwing_CD.lut": "one of two drag height tables that stay near 1 (interchangeable)", "height_diffuser_rear_CD.lut": "one of two drag height tables that stay near 1 (interchangeable)",
             "height_diffuser_CD.lut": "the height table left over (the front diffuser has CD_GAIN 0)"}
    for role, stored in base.items():
        if role in rules:
            give(role, stored, rules[role], "little effect" if role in little else "sure")
    missing = [r for r in refs if r not in found]
    if missing:
        raise AeroCurveError("curves not identified: " + ", ".join(sorted(missing)))
    return found, sorted(notes, key=lambda n: n["role"])
