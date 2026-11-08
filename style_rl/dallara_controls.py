import numpy as np
import pandas as pd


class DallaraControls:

    def __init__(self, torque_csv):

        data = pd.read_csv(torque_csv)

        data = data.dropna(
            subset=["effective_torque_nm"]
        )

        data = data.sort_values("speed_kmh")

        self.speeds = data["speed_kmh"].to_numpy(float)
        self.torques = data["effective_torque_nm"].to_numpy(float)

        if len(self.speeds) < 2:
            raise ValueError("Insufficient calibration points.")

    def throttle_to_torque(self, speed_mps, throttle):

        throttle = float(np.clip(throttle, 0.0, 1.0))

        speed_kmh = abs(float(speed_mps)) * 3.6

        torque = np.interp(
            speed_kmh,
            self.speeds,
            self.torques
        )

        return float(throttle * torque)