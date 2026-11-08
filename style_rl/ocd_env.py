import json
from pathlib import Path

import numpy as np
import pandas as pd

from style_rl.track import MonzaTrack
from style_rl.dallara_controls import DallaraControls

from ocd_vehicle_models_py import (
    VehicleFactory,
    DrivetrainType,
    SteeringActuatorType,
    VehicleDynamicsType,
    TireType,
    AerodynamicsType,
)


ROOT = Path(__file__).resolve().parents[1]


class OCDMonzaEnv:
    """
    Standalone Monza environment using Open-Car-Dynamics.

    Vehicle:
        Dallara F317 candidate configuration.

    Controls:
        steering: [-1, 1]
        throttle: [0, 1]
        brake:    [0, 1]

    Physics:
        OCD integration timestep = 0.0008 seconds
        Control timestep = 0.04 seconds by default
    """

    def __init__(self, track_path, dt=0.04):

        # ==========================================
        # TRACK
        # ==========================================

        self.track = MonzaTrack(track_path)

        # ==========================================
        # SIMULATION TIMESTEPS
        # ==========================================

        self.physics_dt = 0.0008
        self.control_dt = float(dt)

        ratio = self.control_dt / self.physics_dt
        self.substeps = round(ratio)

        if self.substeps < 1 or not np.isclose(
            ratio, self.substeps
        ):
            raise ValueError(
                "Control timestep must be a positive "
                "integer multiple of the OCD physics timestep."
            )

        # ==========================================
        # CREATE OCD VEHICLE
        # ==========================================

        self.vehicle = VehicleFactory.create_vehicle(
            drivetrain=DrivetrainType.RWD_LSD,
            steering=SteeringActuatorType.PT1,
            vdm=VehicleDynamicsType.DOUBLE_TRACK,
            tire=TireType.MF52,
            aero=AerodynamicsType.DEFAULT,
        )

        # ==========================================
        # LOAD CALIBRATED VEHICLE PARAMETERS
        # ==========================================

        self.parameter_file = (
            ROOT
            / "data/vehicle_calibration"
            / "dallara_f317_tyre_125.json"
        )

        with open(self.parameter_file) as f:
            self.params = json.load(f)

        self.params["integration_step_size_s"] = (
            self.physics_dt
        )

        # ==========================================
        # LOAD ACCELERATION CONTROL
        # ==========================================

        self.drive_file = (
            ROOT
            / "data/calibration"
            / "dallara_f317_effective_torque.csv"
        )

        # Keep the existing control class.
        self.controls = DallaraControls(
            self.drive_file
        )

        # ==========================================
        # LOAD BRAKING CALIBRATION
        # ==========================================

        self.brake_file = (
            ROOT
            / "data/calibration"
            / "dallara_f317_brake_calibration_working.csv"
        )

        brake_df = pd.read_csv(self.brake_file)

        required = {
            "speed_kmh",
            "brake_torque_per_wheel_nm",
        }

        missing = required - set(brake_df.columns)

        if missing:
            raise ValueError(
                f"Missing braking calibration columns: {missing}"
            )

        brake_df = (
            brake_df[
                [
                    "speed_kmh",
                    "brake_torque_per_wheel_nm",
                ]
            ]
            .dropna()
            .sort_values("speed_kmh")
        )

        self.brake_speeds = brake_df[
            "speed_kmh"
        ].to_numpy(dtype=float)

        self.brake_torques = brake_df[
            "brake_torque_per_wheel_nm"
        ].to_numpy(dtype=float)

        if len(self.brake_speeds) < 2:
            raise ValueError(
                "Braking calibration requires at least two points."
            )

        if (
            not np.all(np.isfinite(self.brake_speeds))
            or not np.all(np.isfinite(self.brake_torques))
        ):
            raise ValueError(
                "Braking calibration contains non-finite values."
            )

        if not np.all(np.diff(self.brake_speeds) > 0):
            raise ValueError(
                "Braking calibration speeds must be strictly increasing."
            )

        if np.any(self.brake_torques < 0):
            raise ValueError(
                "Braking torques must be nonnegative."
            )

        # ==========================================
        # CONTROL LIMITS
        # ==========================================

        # Provisional steering-command safety limit.
        # Not a verified Dallara mechanical steering limit.
        self.max_steering_rad = 0.15

        self.time = 0.0

        self.drivetrain = None
        self.steering = None
        self.external = None

        # ==========================================
        # INITIALIZATION SUMMARY
        # ==========================================

        print("\n=== OCD DALLARA F317 ENVIRONMENT ===")

        print(
            "Parameter file:",
            self.parameter_file.name,
        )

        print(
            "Mass parameter:",
            self.params[
                "vehicle_dynamics_double_track.mass_vehicle_kg"
            ],
        )

        print(
            "Wheelbase:",
            self.params[
                "vehicle_dynamics_double_track.wheelbase_m"
            ],
        )

        print(
            "Lift coefficient:",
            self.params[
                "vehicle_dynamics_double_track.aerodynamics.c_l"
            ],
        )

        print(
            "Drive calibration:",
            self.drive_file.name,
        )

        print(
            "Brake calibration:",
            self.brake_file.name,
        )

        print(
            "Maximum steering command:",
            self.max_steering_rad,
        )

    # ==============================================
    # RESET
    # ==============================================

    def reset(self, start_idx=100, speed=0.0):

        if not np.isfinite(speed) or speed < 0:
            raise ValueError(
                "Initial speed must be finite and nonnegative."
            )

        x, y = self.track.center[start_idx]

        heading = self.track.query(
            x, y
        )["heading_rad"]

        p = self.params.copy()

        # Initial global position
        p["initial_state.vehicle_dynamics.x_m"] = float(x)
        p["initial_state.vehicle_dynamics.y_m"] = float(y)
        p["initial_state.vehicle_dynamics.psi_rad"] = float(
            heading
        )

        # Initial longitudinal and lateral velocities
        p["initial_state.vehicle_dynamics.v_x_mps"] = float(
            speed
        )

        p["initial_state.vehicle_dynamics.v_y_mps"] = 0.0
        p["initial_state.vehicle_dynamics.psi_dot_radps"] = 0.0

        # ==========================================
        # INITIAL WHEEL SPEEDS
        # ==========================================

        front_radius = p[
            "vehicle_dynamics_double_track.tire.rolling_radius_m.front"
        ]

        rear_radius = p[
            "vehicle_dynamics_double_track.tire.rolling_radius_m.rear"
        ]

        p["initial_state.drivetrain.omega_FL_radps"] = (
            speed / front_radius
        )

        p["initial_state.drivetrain.omega_FR_radps"] = (
            speed / front_radius
        )

        p[
            "initial_state.drivetrain.omega_rear_axle_radps"
        ] = speed / rear_radius

        p[
            "initial_state.drivetrain.omega_diff_rear_radps"
        ] = 0.0

        # ==========================================
        # RESET OCD VEHICLE
        # ==========================================

        self.vehicle.get_param_manager().set_parameters_from_dict(
            p
        )

        self.vehicle.reset()

        self.drivetrain = (
            self.vehicle.construct_drivetrain_input()
        )

        self.steering = (
            self.vehicle.construct_steering_input()
        )

        self.external = (
            self.vehicle.construct_external_influences()
        )

        self.drivetrain.transmission_output_torque_Nm = 0.0

        self.steering.steering_angle_rad = 0.0

        brakes = (
            self.drivetrain.brake_torque_per_wheel_Nm
        )

        brakes.front_left = 0.0
        brakes.front_right = 0.0
        brakes.rear_left = 0.0
        brakes.rear_right = 0.0

        self.vehicle.set_drivetrain_input(
            self.drivetrain
        )

        self.vehicle.set_steering_input(
            self.steering
        )

        self.vehicle.set_external_influences(
            self.external
        )

        # Populate OCD outputs.
        self.vehicle.step()

        self.time = self.physics_dt

        return self.observe()

    # ==============================================
    # OBSERVE
    # ==============================================

    def observe(self):

        output = self.vehicle.get_vehicle_model_output()

        state = output.vehicle_dynamics_output

        x = float(state.position_m.x)
        y = float(state.position_m.y)

        vx = float(state.velocity_mps.x)
        vy = float(state.velocity_mps.y)

        yaw = float(state.orientation_rad.z)

        yaw_rate = float(
            state.angular_velocity_radps.z
        )

        speed = float(
            np.hypot(vx, vy)
        )

        track_state = self.track.query(x, y)

        return {
            "time_s": self.time,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "yaw": yaw,
            "yaw_rate": yaw_rate,
            **track_state,
        }

    # ==============================================
    # PHYSICAL CONTROL STEP
    # ==============================================

    def step(
        self,
        steering_rad,
        drive_torque_nm,
        brake_torque_nm=0.0,
    ):

        if self.drivetrain is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        values = np.array(
            [
                steering_rad,
                drive_torque_nm,
                brake_torque_nm,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Control inputs must be finite."
            )

        if brake_torque_nm < 0:
            raise ValueError(
                "Brake torque must be nonnegative."
            )

        if drive_torque_nm < 0:
            raise ValueError(
                "Drive torque must be nonnegative."
            )

        # Steering command
        self.steering.steering_angle_rad = float(
            steering_rad
        )

        # Transmission torque
        self.drivetrain.transmission_output_torque_Nm = (
            float(drive_torque_nm)
        )

        # Equal braking torque per wheel
        brakes = (
            self.drivetrain.brake_torque_per_wheel_Nm
        )

        brakes.front_left = float(brake_torque_nm)
        brakes.front_right = float(brake_torque_nm)
        brakes.rear_left = float(brake_torque_nm)
        brakes.rear_right = float(brake_torque_nm)

        # Apply control inputs
        self.vehicle.set_steering_input(
            self.steering
        )

        self.vehicle.set_drivetrain_input(
            self.drivetrain
        )

        # Integrate physics
        for _ in range(self.substeps):
            self.vehicle.step()

        self.time += self.control_dt

        return self.observe()

    # ==============================================
    # NORMALIZED CONTROL STEP
    # ==============================================

    def step_normalized(
        self,
        steering,
        throttle,
        brake,
    ):

        actions = np.asarray(
            [steering, throttle, brake],
            dtype=float,
        )

        if not np.all(np.isfinite(actions)):
            raise ValueError(
                "Normalized actions must be finite."
            )

        steering = float(
            np.clip(steering, -1.0, 1.0)
        )

        throttle = float(
            np.clip(throttle, 0.0, 1.0)
        )

        brake = float(
            np.clip(brake, 0.0, 1.0)
        )

        # Brake takes priority.
        if brake > 0.0:
            throttle = 0.0

        obs = self.observe()

        speed_mps = obs["speed"]
        speed_kmh = speed_mps * 3.6

        # ==========================================
        # DRIVE TORQUE
        # ==========================================

        # Reuse the existing DallaraControls class.
        drive_torque = float(
            self.controls.throttle_to_torque(
                speed_mps,
                throttle,
            )
        )

        # ==========================================
        # BRAKE TORQUE
        # ==========================================

        reference_brake_torque = float(
            np.interp(
                speed_kmh,
                self.brake_speeds,
                self.brake_torques,
            )
        )

        brake_torque = (
            brake * reference_brake_torque
        )

        # ==========================================
        # STEERING
        # ==========================================

        steering_rad = (
            steering * self.max_steering_rad
        )

        return self.step(
            steering_rad=steering_rad,
            drive_torque_nm=drive_torque,
            brake_torque_nm=brake_torque,
        )