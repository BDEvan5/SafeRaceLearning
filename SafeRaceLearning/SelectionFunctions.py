from SafeRaceLearning.Planners.Architectures import *
from SafeRaceLearning.Utils.StdRewards import *
from SafeRaceLearning.Utils.RacingLineRewards import *


def select_architecture(architecture: str): 
    if architecture == "slow": 
        architecture_type = SlowArchitecture
    elif architecture == "fast":
        architecture_type = FastArchitecture
    else: raise Exception("Unknown architecture")

    return architecture_type


def select_reward_function(run, conf, std_track, race_track):
    reward = run.reward
    if reward == "Zero":
        reward_function = ZeroReward()
    elif reward == "Progress":
        reward_function = ProgressReward(std_track)
    elif reward == "Cth": 
        reward_function = CrossTrackHeadReward(std_track, conf)
    elif reward == "Velocity":
        reward_function = VelocityReward(conf, run)
    else: raise Exception("Unknown reward function: " + reward)
        
    return reward_function