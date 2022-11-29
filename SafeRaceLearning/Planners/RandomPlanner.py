import numpy as np
import os 
from SafeRaceLearning.Utils.utils import init_file_struct


class RandomPlanner:
    def __init__(self, run, conf):
        self.d_max = conf.max_steer # radians  
        self.name = run.run_name

        self.conf = conf
        self.run = run

        if run.racing_mode == "Fast":
            self.max_speed = 8
        elif run.racing_mode == "Slow":
            self.max_speed = 2
        else:
            raise Exception("Unknown racing mode")

        self.path = conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)

    def plan(self, obs):
        steering = np.random.uniform(-self.d_max, self.d_max)
        speed = np.random.uniform(2, self.max_speed)
        return np.array([steering, speed])

    # def plan(self, obs):
    #     return np.array([0, self.max_speed]) # every action not allowed

    def lap_complete(self):
        pass

if __name__ == '__main__':
    pass

