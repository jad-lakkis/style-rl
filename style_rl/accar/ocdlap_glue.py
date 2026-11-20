"""Puts the replica of the Assetto Corsa SF70H (style_rl.accar.to_ocd.build_replica, as json) on the backend of ocd_first_lap's `ocd_lap` package, without changing any file of his.

What it does, at run time and in this process only:
  - builds an `ocd_lap.cars.CarSpec` from the replica (mass, geometry, tyres, springs, ... as OCD parameter overrides; steering lock; brake torque),
  - replaces `ocd_lap.backend.Powertrain` by the AC engine, gearbox and turbo (style_rl.accar.powertrain),
  - subclasses `OCDBackend` so that the pedal reaches the turbo model before every physics step, and so that the AC powertrain is the one a reset creates.
"""


def build(replica, *, turbo="ac", ers=False):
    """(backend, setup, spec) for the AC replica; needs ocd_lap on the path (OCD_FIRST_LAP_DIR) and its native module."""
    from ocd_lap import backend as bk
    from ocd_lap import cars

    from .powertrain import make_powertrain
    pt = replica["powertrain"]
    cs = dict(replica["car_spec"])
    footprint = tuple(tuple(float(v) for v in p) for p in cs.pop("footprint"))
    ers_dict = None
    if ers:
        if not pt.get("ers"):
            raise ValueError("this replica has no electric motor data")
        ers_dict = {"pow_e_motor_W": 120e3, "eta_e_motor": 1.0, "torque_e_motor_max_Nm": max(c[1] for c in pt["ers"]["torque_curve"]), "vel_min_e_motor_mps": 0.0,
                    "energy_per_lap_J": pt["ers"]["energy_per_lap_J"]}
    mass = replica["ocd"]["vehicle_dynamics_double_track.mass_vehicle_kg"]
    spec = cars.CarSpec(name=cs["name"], ocd=dict(replica["ocd"]), rear_axle_m=cs["rear_axle_m"], wheelbase_m=cs["wheelbase_m"], driver_mass_kg=mass, driver_radius_m=pt["wheel_radius_m"],
                        footprint=footprint, ers=ers_dict, steer_max_rad=cs["steer_max_rad"], steer_rate_max_radps=cs["steer_rate_max_radps"], brake_torque_Nm=cs["brake_torque_Nm"],
                        description=cs["description"])
    if not hasattr(bk, "_original_Powertrain"):
        bk._original_Powertrain = bk.Powertrain
    pt_class = make_powertrain(bk._original_Powertrain, pt, turbo)

    class AcBackend(bk.OCDBackend):
        def reset(self, setup, start_pose):
            bk.Powertrain = pt_class                                       # the class a reset builds its powertrain from
            return super().reset(setup, start_pose)

        def step(self, inputs, steps=1):
            if self.powertrain is not None:
                self.powertrain.set_pedal(inputs.throttle)                 # the turbo needs the pedal; the base backend only multiplies the torque by it
            return super().step(inputs, steps)

    backend = AcBackend(powertrain="engine", car=spec, ers=bool(ers), overrides=dict(replica["ocd"]))
    setup = cars.default_setup(spec).replace(**replica["setup"])
    return backend, setup, spec
