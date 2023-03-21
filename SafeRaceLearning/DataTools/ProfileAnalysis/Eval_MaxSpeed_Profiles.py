import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from RacingRewards.DataTools.MapData import MapData
from RacingRewards.RewardSignals.StdTrack import StdTrack 

from SafeRaceLearning.Utils.utils import *
from SafeRaceLearning.DataTools.plotting_utils import *


class TestLapData:
    def __init__(self, path, lap_n=0):
        self.path = path
        self.vehicle_name = self.path.split("/")[-2]
        self.map_name = self.vehicle_name.split("_")[4]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[5]
        self.map_data = MapData(self.map_name)
        self.race_track = StdTrack(self.map_name)

        self.states = None
        self.actions = None
        self.lap_n = lap_n

        self.load_lap_data()

    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def generate_state_progress_list(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.race_track.calculate_progress_percent(pt)
            # if p < progresses[-1]: continue
            progresses.append(p)
            
        return np.array(progresses[:-2])



def make_slip_compare_graph():
    # map_name = "f1_gbr"
    map_name = "f1_esp"
    # pp_path = f"Data/Vehicles/RacingResultsWeekend/PP_Std_{map_name}_1_0/"
    # agent_path = f"Data/Vehicles/RacingResultsWeekend/Agent_Cth_{map_name}_2_1/"

    pp_path = f"Data/Vehicles/PerformanceSpeed/PP_Std5_{map_name}_1_0/"
    agent_path = f"Data/Vehicles/PerformanceSpeed/Agent_Cth_{map_name}_3_0/"


    pp_data = TestLapData(pp_path)
    agent_data = TestLapData(agent_path)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    ax1.plot(agent_data.states[:, 6], color=pp[1], label="Agent")
    ax1.plot(pp_data.states[:, 6], color=pp[0], label="PP")


    ax1.set_ylabel("Slip angle")
    ax1.set_xlabel("Time steps")
    ax1.legend(ncol=2)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Data/HighSpeedEval/SlipCompare_{map_name}.pdf", bbox_inches='tight')

    plt.show()
    
    
def compare_fast_speed():
        # map_name = "f1_gbr"
    map_name = "f1_esp"
    path1 = f"Data/Vehicles/Std_TrainMaps_old/"
    path2 = f"Data/Vehicles/Safe_TrainMaps/"
    path3 = f"Data/Vehicles/PP_TestMaps/"
    path1 = path1 + f"fast_Std_Std_Cth_f1_esp_5_1_1/"
    path2 = path2 + f"fast_Online_Std_Velocity_f1_esp_5_1_0/"
    path3 = path3 + f"PP_PP_Std_PP_f1_esp_5_1_0/"

    pp_data = TestLapData(path1)
    agent_data = TestLapData(path2, 1)
    agent_data2 = TestLapData(path3, 1)
    progresses1 = pp_data.generate_state_progress_list() * 100
    progresses2 = agent_data.generate_state_progress_list() * 100
    progresses3 = agent_data2.generate_state_progress_list() * 100

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 1.6), sharex=True)
    # fig, (ax1) = plt.subplots(1, 1, figsize=(4, 1.7), sharex=True)
    # fig, (ax1) = plt.subplots(1, 1, figsize=(6.5, 2), sharex=True)
    ax1.plot(progresses1, pp_data.states[:-1, 3], color=pp[0], label="Conventional", alpha=0.8, linewidth=2)
    ax1.plot(progresses3, agent_data2.states[:-1, 3], color=pp[2], label="Classic", alpha=0.9, linewidth=2)
    ax1.plot(progresses2, agent_data.states[:-1, 3], color=pp[4], label="Safe", alpha=0.9, linewidth=2)

    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Track Progress (%)")
    fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.0), loc='center')
    
    plt.xlim([0, 60])
    plt.ylim([0.5, 5.5])
    plt.yticks([1, 3, 5])

    plt.grid(True)
    plt.tight_layout()

    name = "Data/Images/speed_profile_compare"
    std_img_saving(name)

    # plt.savefig(f"{pat}/fast_speed_compare{map_name}.pdf", bbox_inches='tight')
    # plt.savefig(f"{path}/fast_speed_compare{map_name}.svg", bbox_inches='tight')


def make_velocity_compare_graph():
    map_name = "f1_esp"
    pp_path = f"Data/Vehicles/SSS_ppValidation/PP_PP_Std_PP_{map_name}_6_1_0/"
    super_path = f"Data/Vehicles/SSS_ppValidation/PP_PP_Super_PP_{map_name}_6_1_0/"

    pp_data = TestLapData(pp_path)
    x_pp = pp_data.generate_state_progress_list()*100
    agent_data = TestLapData(super_path, 2)
    x_super = agent_data.generate_state_progress_list()*100

    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 1.7), sharex=True)
    # fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    ax1.plot(x_super, agent_data.states[:-1, 3], color=pp[1], label="Supervisor", linewidth=1.5)
    ax1.plot(x_pp, pp_data.states[:-1, 3], color=pp[0], label="PP", linewidth=1.5)

    ax1.axes.yaxis.set_major_locator(MultipleLocator(2))
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Track Progress (%)")
    fig.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, 0))
    # ax2.set_ylabel("Slip Angle")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/SSS_ppValidation/SpeedCompare_{map_name}.pdf", bbox_inches='tight')
    plt.savefig(f"Data/Vehicles/SSS_ppValidation/SpeedCompare_{map_name}.svg", bbox_inches='tight')

    # plt.show()


# make_velocity_compare_graph()    
compare_fast_speed()