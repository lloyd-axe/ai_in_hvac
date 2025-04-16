import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

'''
Room Thermal Capacity = Net Heat
C_room * dT/dt = P_server + Q_hvac - k_env * (T_room - T_env)
'''


class ServerRoomDigitalTwin:
    def __init__(self, C_room: float, Q_hvac_max: float, P_server: float, k_env: float, T_env: float,
                 temp_threshold: tuple[float, float]):
        self.C_room = C_room
        self.Q_hvac_max = Q_hvac_max
        self.P_server = P_server
        self.k_env = k_env
        self.T_env = T_env
        self.temp_threshold = temp_threshold

    def calculate_thermal_rate(self, t: float, T: list[float]) -> list[float]:
        if T[0] > self.temp_threshold[1]:  # Turn cooling Max
            Q_hvac = self.Q_hvac_max
        elif T[0] < self.temp_threshold[0]:  # Turn cooling OFF
            Q_hvac = 0
        else: # Turn cooling ON
            Q_hvac = self.Q_hvac_max / 2
        dTdt = (self.P_server + Q_hvac - self.k_env * (T[0] - self.T_env)) / self.C_room
        return [dTdt]

    def simulate(self, initial_temp: list[float], time_span: tuple[float, float]):
        solution = solve_ivp(self.calculate_thermal_rate, time_span, initial_temp,
                             t_eval=np.linspace(time_span[0], time_span[1], 100))
        return solution

    def plot(self, solution) -> None:
        plt.plot(solution.t, solution.y[0], label='Room Temperature')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (K)')
        plt.title('Room Temperature in 24 hours')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    room_model = ServerRoomDigitalTwin(
        C_room=5000,
        Q_hvac_max=-3000,
        P_server=2000,
        k_env=0.5,
        T_env=298,
        temp_threshold=(300, 303)
    )

    solution = room_model.simulate(
        initial_temp=[300],
        time_span=(0, 3600 * 24)  # 24 hours
    )

    room_model.plot(solution)
