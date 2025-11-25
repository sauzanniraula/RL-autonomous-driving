import os
import sys
import glob

# Path setup
carla_root = 'C:/Carla/CARLA_0.9.16/PythonAPI/carla/dist/'
try:
    egg_path = glob.glob(carla_root + 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(egg_path)
except IndexError:
    pass

import carla
import numpy as np
from collections import deque
from utils import get_speed

class PIDLongitudinalController():
    def __init__(self, vehicle, max_throttle=0.75, max_brake=0.8, K_P=1.2, K_I=0.05, K_D=0.01, dt=0.03):
        """
        Tesla Model 3 Tuned PID.
        Higher K_P for sharper steering response.
        Higher max_brake (0.8) because Tesla is heavy.
        """
        self._vehicle = vehicle
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed):
        current_speed = get_speed(self._vehicle)
        acceleration = self._pid_control(target_speed, current_speed)
        control = carla.VehicleControl()
        
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            # Apply stronger braking for Tesla
            control.brake = min(abs(acceleration), self.max_brake)
            
        return control

    def _pid_control(self, target_speed, current_speed):
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)