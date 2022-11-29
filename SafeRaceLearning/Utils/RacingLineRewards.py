from SafeRaceLearning.Planners.PurePursuit import PurePursuit
import numpy as np
from SafeRaceLearning.Utils.RewardUtils import *
from SafeRaceLearning.Utils.RacingTrack import RacingTrack

class PppsReward:
    def __init__(self, conf, run):
        run.pp_speed_mode = "link"
        run.raceline = False
        self.pp = PurePursuit(conf, run, False) 

        self.beta_c = 0.4
        self.beta_steer_weight = 0.4
        if run.architecture =="fast":
            self.beta_velocity_weight = 0.4
        else:
            self.beta_velocity_weight = 0.0

        self.max_steer_diff = 0.8
        self.max_velocity_diff = 2.0
        # self.max_velocity_diff = 4.0

    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        pp_act = self.pp.plan(prev_obs)

        steer_reward =  (abs(pp_act[0] - action[0]) / self.max_steer_diff)  * self.beta_steer_weight

        throttle_reward =   (abs(pp_act[1] - action[1]) / self.max_velocity_diff) * self.beta_velocity_weight

        # reward = self.beta_c - steer_reward
        reward = self.beta_c - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= 0.5

        return reward


class RewardV1:
    def __init__(self, race_track: RacingTrack, conf):
        self.race_track = race_track
        self.r_veloctiy = conf.r_velocity
        self.r_distance = conf.r_distance
        self.max_v = conf.max_v

    def __call__(self, observation, prev_obs, prev_action):
        if prev_obs is None: return 0
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.race_track.get_cross_track_heading(position)
        vx = self.race_track.get_velocity(position)

        d_heading = abs(robust_angle_difference_rad(heading, theta))

        dV  = abs(prev_action[1] - vx)
        # dV  = abs(observation['state'][3] - vx)
        scaled_dV = (self.max_v-dV) /self.max_v
        reward =  (1-scaled_dV) * np.cos(d_heading) * self.r_veloctiy - distance * self.r_distance

        reward = max(reward, 0) # cap at 0

        return reward


class RewardV2:
    def __init__(self, race_track: RacingTrack, conf):
        self.race_track = race_track
        self.r_veloctiy = conf.r_velocity
        self.r_distance = conf.r_distance
        self.max_v = conf.max_v

    def __call__(self, observation, prev_obs, prev_action):
        if prev_obs is None: return 0
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.race_track.get_cross_track_heading(position)
        vx = self.race_track.get_velocity(position)

        d_heading = abs(robust_angle_difference_rad(heading, theta))

        scaled_v = observation['state'][3] / self.max_v
        dV  = abs(prev_action[1] - vx)
        # dV  = abs(observation['state'][3] - vx)
        scaled_dV = (self.max_v-dV) /self.max_v
        
        reward =  scaled_v * np.cos(d_heading) * self.r_veloctiy - distance * self.r_distance - scaled_dV * 0.5

        reward = max(reward, 0) # cap at 0

        return reward


class RewardV3:
    def __init__(self, race_track: RacingTrack, conf):
        self.race_track = race_track
        self.r_veloctiy = conf.r_velocity
        self.r_distance = conf.r_distance
        self.max_v = conf.max_v

    def __call__(self, observation, prev_obs, prev_action):
        if prev_obs is None: return 0
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.race_track.get_cross_track_heading(position)
        vx = self.race_track.get_velocity(position)

        if prev_action[1] > vx:
        # if observation['state'][3] > vx:
            reward = -distance * self.r_distance
            reward = max(reward, -0.2) # cap at
            return reward 
        
        d_heading = abs(robust_angle_difference_rad(heading, theta))

        scaled_v = observation['state'][3] / self.max_v

        reward =  scaled_v * np.cos(d_heading) * self.r_veloctiy - distance * self.r_distance

        reward = max(reward, 0) # cap at 0

        return reward
