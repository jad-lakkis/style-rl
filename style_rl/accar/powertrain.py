"""The engine, gearbox and electric motor of the Assetto Corsa SF70H as a drop-in for the Powertrain class of ocd_first_lap's `ocd_lap` package.

`make_powertrain(Powertrain, replica["powertrain"], turbo="ac")` returns a subclass of his Powertrain (the class is passed in, so this module imports none of his code). The backend of ocd_lap uses
it as:  update_gear(axle speed, dt) every physics step, axle_torque_capacity_Nm(axle speed, vehicle speed) = the rear-axle torque at FULL pedal, which the backend multiplies by the pedal, and
draw_ers(pedal, dt) after the step when there is an electric motor. For the turbo the pedal has to be known too: `set_pedal(pedal)` is called by `AcBackend.step` before each step.

Engine torque = base curve (rpm | Nm) x (1 + boost) x pedal.  Boost (Assetto Corsa's turbo model, as read from the comments of engine.ini and from two reimplementations of AC):
    target = min(WASTEGATE, MAX_BOOST) x pedal ^ GAMMA x min(1, rpm / REFERENCE_RPM)
    boost moves to the target with the lag LAG_UP (target above) / LAG_DN (below) per physics step of 1 / 333 s:  boost += (target - boost) (1 - lag ^ (dt x 333))
  turbo = "ac"      this model (a time constant of about 2 s, and a pedal that must be nearly full for full torque)
  turbo = "steady"  the same without the lag (boost = target at once)
  turbo = "linear"  full boost whatever the pedal: torque proportional to the pedal, as in a plain OCD car
Gearbox: AC's automatic shifter: up at `shift_up_rpm`, down at `shift_down_rpm`, at most one shift per 0.1 s, no torque interruption. Rev limiter: no torque above `limiter_rpm`.
"""
import numpy as np

MIN_SHIFT_INTERVAL_S = 0.1
SLIP_RPM_FLOOR = 4500.0          # below this engine speed the clutch slips (AC's autoclutch engages at 4300 to 4900 rpm): the torque is read at 4500 rpm


def make_powertrain(base, pt, turbo="ac"):
    if turbo not in ("ac", "steady", "linear"):
        raise ValueError("turbo must be 'ac', 'steady' or 'linear'")
    gears = np.asarray(pt["gears"], float) * float(pt["final"])
    curve = np.asarray(pt["engine_curve"], float)
    kers = np.asarray(pt["ers"]["torque_curve"], float) if pt.get("ers") else None
    tb = pt["turbo"]
    boost_cap = min(tb["wastegate"] or tb["max_boost"], tb["max_boost"])

    class AcPowertrain(base):
        def __init__(self, *, final_drive=1.0, vehicle_mass_kg, wheel_radius_m, wheel_inertia_kgm2, data=None, ers=None):
            ers_dict = None
            if ers:
                ers_dict = {"pow_e_motor_W": 120e3, "eta_e_motor": 1.0, "torque_e_motor_max_Nm": float(kers[:, 1].max()) if kers is not None else 0.0, "vel_min_e_motor_mps": 0.0,
                            "energy_per_lap_J": float(pt["ers"]["energy_per_lap_J"]) if pt.get("ers") else 0.0}
            super().__init__(final_drive=final_drive, vehicle_mass_kg=vehicle_mass_kg, wheel_radius_m=wheel_radius_m, wheel_inertia_kgm2=wheel_inertia_kgm2, ers=ers_dict)
            self.ratios = float(final_drive) * gears
            self.d = dict(self.d, eta_g=float(pt["efficiency"]))
            self.engine_inertia_kgm2 = [float(pt["engine_inertia_kgm2"]) * float(r) ** 2 for r in self.ratios]
            self.limiter, self.shift_up, self.shift_down = float(pt["limiter_rpm"]), float(pt["shift_up_rpm"]), float(pt["shift_down_rpm"])
            self.boost, self.pedal = 0.0, 0.0
            self.gear, self.since_shift_s = 0, MIN_SHIFT_INTERVAL_S
            self.refill_ers()

        # --- the pedal and the turbo -----------------------------------------------------------------
        def set_pedal(self, pedal):
            self.pedal = float(min(max(pedal, 0.0), 1.0))

        def _boost_target(self, rpm):
            if turbo == "linear":
                return boost_cap
            return boost_cap * self.pedal ** tb["gamma"] * min(1.0, rpm / tb["reference_rpm"])

        def _advance_boost(self, rpm, dt_s):
            target = self._boost_target(rpm)
            if turbo != "ac" or dt_s <= 0.0:
                self.boost = target if turbo != "ac" else self.boost
                return
            lag = tb["lag_up"] if target > self.boost else tb["lag_dn"]
            k = lag ** (dt_s * tb["physics_hz"])
            self.boost = k * self.boost + (1.0 - k) * target

        # --- engine ------------------------------------------------------------------------------------
        def engine_torque_Nm(self, rpm):
            if rpm > self.limiter:
                return 0.0
            n = max(float(rpm), SLIP_RPM_FLOOR)
            return float(np.interp(n, curve[:, 0], curve[:, 1])) * (1.0 + self.boost)

        def e_motor_torque_Nm(self, rpm, speed_mps, *, energy_J=None):
            energy = self.ers_energy_J if energy_J is None else energy_J
            if not self.ers or kers is None or energy <= 0.0 or rpm <= 0.0:
                return 0.0
            return float(np.interp(rpm, kers[:, 0], kers[:, 1]))

        def update_gear(self, axle_omega_radps, dt_s=0.0):
            self.since_shift_s += dt_s
            rpm = self.rpm(axle_omega_radps)
            self._advance_boost(rpm, dt_s)
            if self.since_shift_s < MIN_SHIFT_INTERVAL_S:
                return self.gear
            gear = self.gear
            if gear < len(self.ratios) - 1 and rpm > self.shift_up:
                self.gear += 1
            elif gear > 0 and rpm < self.shift_down and self.rpm(axle_omega_radps, gear - 1) < self.limiter:
                self.gear -= 1
            if self.gear != gear:
                self.since_shift_s = 0.0
            return self.gear

        def axle_torque_capacity_Nm(self, axle_omega_radps, speed_mps=None):
            rpm = self.rpm(axle_omega_radps)
            ice = self.engine_torque_Nm(rpm)
            motor = 0.0
            if self.ers:
                speed = axle_omega_radps * self.wheel_radius_m if speed_mps is None else speed_mps
                self._em_rpm = rpm
                motor = self._em_crank_Nm = self.e_motor_torque_Nm(rpm, speed)
            return self.d["eta_g"] * self.ratios[self.gear] * (ice + motor)

    return AcPowertrain


def full_throttle_axle_force(pt, speed_mps, turbo="steady", wheel_radius_m=0.335):
    """Best-gear full-pedal driving force at the wheels (N) at a steady speed, for a check of the replica without OCD: the largest force over the gears whose rpm is under the limiter."""
    gears = np.asarray(pt["gears"], float) * float(pt["final"])
    curve = np.asarray(pt["engine_curve"], float)
    tb = pt["turbo"]
    boost = min(tb["wastegate"] or tb["max_boost"], tb["max_boost"])
    best = 0.0
    for r in gears:
        rpm = speed_mps / wheel_radius_m * r * 60 / (2 * np.pi)
        if rpm > pt["limiter_rpm"]:
            continue
        b = boost * min(1.0, rpm / tb["reference_rpm"]) if turbo != "linear" else boost
        t = float(np.interp(max(rpm, SLIP_RPM_FLOOR), curve[:, 0], curve[:, 1])) * (1.0 + b)
        best = max(best, t * r * float(pt["efficiency"]) / wheel_radius_m)
    return best
