from SafeRaceLearning.Utils.RewardUtils import *

from RacingRewards.Utils.utils import *
from SafeRaceLearning.Utils.StdTrack import StdTrack

# rewards functions
class ProgressReward:
    def __init__(self, track: StdTrack) -> None:
        self.track = track

    def __call__(self, observation, prev_obs, pre_action):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        
        position = observation['state'][0:2]
        prev_position = prev_obs['state'][0:2]
        theta = observation['state'][2]

        s = self.track.calculate_progress(prev_position)
        ss = self.track.calculate_progress(position)
        reward = (ss - s) / self.track.total_s
        if abs(reward) > 0.5: # happens at end of eps
            return 0.001 # assume positive progress near end

        # self.race_track.plot_vehicle(position, theta)


        reward *= 10 # remove all reward
        return reward 


class CrossTrackHeadReward:
    def __init__(self, track: StdTrack, conf):
        self.track = track
        self.r_veloctiy = 1
        self.r_distance = 1
        self.max_v = conf.max_v # used for scaling.

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.track.get_cross_track_heading(position)
        # self.race_track.plot_vehicle(position, theta)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (observation['state'][3] / self.max_v)

        r_distance = distance * self.r_distance 

        reward = r_heading - r_distance
        reward = max(reward, 0)
        # reward *= 0.1
        return reward
        # return 0 # test super #!1!!!!!!!!!!!!


class ZeroReward:
    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        return 0




class VelocityReward:
    def __init__(self, conf, run):
        self.max_speed = run.max_speed

    def __call__(self, observation, prev_obs, pre_action):
        if pre_action is None: return 0
        if observation['lap_done']:
            return 1
        if observation['colision_done']:
            return -1

        v = pre_action[1]
        reward = v / self.max_speed

        reward ** 2

        return reward
