from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from SafeRaceLearning.DataTools.MapData import MapData
from SafeRaceLearning.Utils.StdTrack import StdTrack 
from SafeRaceLearning.Utils.RacingTrack import RacingTrack
from SafeRaceLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator

SAVE_PDF = True
# SAVE_PDF = False

def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.lap_n = 0

        self.all_safety_data = None
        self.lap_inds = []
        self.all_states = None
        self.all_actions = None
        self.state_inds = []

    def process_folder(self, folder):
        self.path = folder 
        self.initialise_folder()

        for self.lap_n in range(20):
            if not self.load_lap_data(): break # no more laps
            
            self.load_safety_data()

        # self.plot_interventions()   
        self.plot_interventions_hist()         
        # self.plot_steering_graph()
        # self.plot_deviation_training()
        # self.plot_combined_steering_deviation()
        self.save_complete_training_data()

        # self.save_action_profile()

    def initialise_folder(self):
        self.vehicle_name = self.path.split("/")[-2]
        self.map_name = self.vehicle_name.split("_")[4]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[5]
        if self.vehicle_name.split("_")[6] == "wide":
            self.map_name += "_" + self.vehicle_name.split("_")[6]

        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)


        self.all_safety_data = None
        self.lap_inds = []
        self.all_states = None
        self.all_actions = None
        self.state_inds = []

    def load_lap_data(self):
        open_path = self.path + "Training/" + f"Lap_train_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy"
        try:
            data = np.load(open_path)
        except Exception as e:
            # print(e)
            print(f"No data for: " + open_path)
            return 0

        self.states = data[:, :7]
        self.actions = data[:, 7:]
        self.all_states = np.concatenate((self.all_states, self.states), axis=0) if self.all_states is not None else self.states
        self.all_actions = np.concatenate((self.all_actions, self.actions), axis=0) if self.all_actions is not None else self.actions
        self.state_inds.append(len(self.states))

        return 1 # to say success

    def load_safety_data(self):
        open_path = self.path + "SafeHistory/" + f"Lap_{self.lap_n}_safeHistory_{self.vehicle_name}.npy"
        try:
            data = np.load(open_path)
            self.safety_data = data

            self.lap_inds.append(len(data))

            self.all_safety_data = np.concatenate((self.all_safety_data, data), axis=0) if self.all_safety_data is not None else data
        except:
            print(f"No safety data for: " + open_path)
            self.safety_data = None
        
    def save_complete_training_data(self):
        np.save(self.path + f"Training/AllTrainingStates_{self.vehicle_name}_{self.map_name}.npy", self.all_states)
        np.save(self.path + f"Training/AllTrainingActions_{self.vehicle_name}_{self.map_name}.npy", self.all_actions)


        np.save(self.path + f"SafeHistory/AllSafeHistory_{self.vehicle_name}_{self.map_name}.npy", self.all_safety_data)

    def plot_action_profiles(self):

        if self.lap_n == 0:
            plt.figure(2, figsize=(6, 2.1))
            plt.clf()
        else: plt.figure(2)

        
        # actual = self.safety_data[:, 2]
        # interventions = self.safety_data[:, 4]
        # planned = self.safety_data[:, 0]
        # planned_plot = planned[interventions==1]
        # plot_xs = np.arange(0, len( planned))
        # plot_xs = plot_xs[interventions==1]

        # plt.plot(actual, linewidth=1, color=pp[self.lap_n])
        # plt.plot(plot_xs, planned_plot, 'x', markersize=4, color=pp[self.lap_n])

        speed = self.states[:, 3]
        plt.plot(speed, linewidth=1, color=pp[self.lap_n])


        plt.ylabel(f"Deviation ")
        plt.xlabel("Training Steps")
        plt.tight_layout()
        # plt.xlim(100, 600)
        # plt.ylim(-2, 50)
        plt.grid(True)

        # plt.show()
        save = self.path + "ActionProfiles/"
        ensure_path_exists(save)      
        plt.savefig(save + f"{self.vehicle_name}_Deviation_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)

    def save_action_profile(self):
        save = self.path + "ActionProfiles/"
        ensure_path_exists(save)

        plt.savefig(save + f"{self.vehicle_name}_Deviation.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save + f"{self.vehicle_name}_Deviation.pdf", bbox_inches='tight', pad_inches=0)


    def plot_interventions(self):
        save = self.path 
        # save = self.path + "TrainInterventions/"
        ensure_path_exists(save)

        plt.figure(1, figsize=(6, 2.1))
        plt.clf()
        interventions = self.all_safety_data[:, 4]
        n= 100
        i = 0 
        inter_ns = []
        while i < len(interventions)-1:
            inter_ns.append(0)
            for _ in range(n):
                if interventions[i] > 0.001:
                    inter_ns[-1] += 1
                i += 1
                if i >= len(interventions)-1:
                    break
        
        plt.plot(inter_ns, linewidth=2, color=pp[0])
        plt.plot(true_moving_average(inter_ns, 5), linewidth=2, color=pp[1])
        plt.xlabel(f"Training steps (x{n})")
        plt.ylabel("Interventions")
        plt.tight_layout()
        plt.xlim(0, 50)
        plt.grid()
        plt.savefig(save + f"{self.vehicle_name}_AllInterventions.svg", bbox_inches='tight', pad_inches=0)

        # plt.show()
        # plt.pause(0.0001)


    def plot_interventions_hist(self):
        save = self.path 
        # save = self.path + "TrainInterventions/"
        ensure_path_exists(save)

        plt.figure(2, figsize=(6, 1.7))
        # plt.figure(2, figsize=(6, 2.1))
        plt.clf()
        interventions = self.all_safety_data[:, 4]
        n= 100
        i = 0 
        inter_ns = []
        while i < len(interventions)-1:
            inter_ns.append(0)
            for _ in range(n):
                if interventions[i] > 0.001:
                    inter_ns[-1] += 1
                i += 1
                if i >= len(interventions)-1:
                    break
        
        inters = np.arange(0, len(interventions)) / 100
        inters = inters[interventions==1]
        # print(f"Interventions: {inters}")
        plt.hist(inters, bins=len(inter_ns), color=pp[1])
        plt.plot(true_moving_average(inter_ns, 5), linewidth=2, color=pp[0])
        plt.xlabel(f"Training steps (x{n})")
        plt.ylabel("Interventions")
        plt.tight_layout()
        # plt.xlim(-1, 50)
        # plt.ylim(-2, 50)
        plt.grid()
        plt.savefig(save + f"{self.vehicle_name}_AllInterventionsHist.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save + f"{self.vehicle_name}_AllInterventionsHist.pdf", bbox_inches='tight', pad_inches=0)

        # plt.show()
        
    def plot_combined_steering_deviation(self):
        save = self.path 
        # save = self.path + "TrainInterventions/"
        ensure_path_exists(save)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 2.5), sharex=True)
        # plt.clf()
        interventions = self.all_safety_data[:, 4]
        
        planned = self.all_safety_data[:, 0]
        actual = self.all_safety_data[:, 2]
        planned_plot = planned[interventions==1]
        plot_xs = np.arange(0, len( planned))
        plot_xs = plot_xs[interventions==1]

        ax1.plot(actual, linewidth=1, color=pp[1])
        ax1.plot(plot_xs, planned_plot, 'x', markersize=4, color=pp[0])
        
        
        pts = self.all_states[:, 0:2]
        hs = []
        for point in pts:
            idx, dists = self.std_track.get_trackline_segment(point)
            x, h = self.std_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        
        ax2.plot(hs[1:], linewidth=2, color=pp[3])
        ax2.plot(true_moving_average(hs, 200), linewidth=2, color=pp[2])

        ax2.set_xlabel(f"Training steps ")
        ax1.set_ylabel("Steering Angle")
        ax2.set_ylabel("Deviation")
        ax2.get_yaxis().set_major_locator(MultipleLocator(0.25))
        plt.tight_layout()
        plt.xlim(0, 1500)
        # plt.ylim(-2, 50)
        ax1.grid(True)
        ax2.grid(True)
        plt.savefig(save + f"{self.vehicle_name}_CombinedOnlineTrainingGraph.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save + f"{self.vehicle_name}_CombinedOnlineTrainingGraph.pdf", bbox_inches='tight', pad_inches=0)

        # plt.show()

    def plot_steering_graph(self):
        save = self.path 
        # save = self.path + "TrainInterventions/"
        ensure_path_exists(save)

        plt.figure(2, figsize=(6, 2.1))
        plt.clf()
        interventions = self.all_safety_data[:, 4]
        
        planned = self.all_safety_data[:, 0]
        actual = self.all_safety_data[:, 2]
        planned_plot = planned[interventions==1]
        plot_xs = np.arange(0, len( planned))
        plot_xs = plot_xs[interventions==1]

        plt.plot(actual, linewidth=1, color=pp[0])
        plt.plot(plot_xs, planned_plot, 'x', markersize=4, color=pp[2])


        plt.xlabel(f"Training steps ")
        plt.ylabel("Steering Angle")
        plt.tight_layout()
        plt.xlim(0, 1500)
        # plt.ylim(-2, 50)
        plt.grid()
        plt.savefig(save + f"{self.vehicle_name}_SteeringGraph.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save + f"{self.vehicle_name}_SteeringGraph.pdf", bbox_inches='tight', pad_inches=0)

        # plt.show()

    def plot_speed_graph(self):
        save = self.path 
        # save = self.path + "TrainInterventions/"
        ensure_path_exists(save)

        plt.figure(2, figsize=(6, 2.1))
        plt.clf()
        interventions = self.safety_data[:, 4]
        
        planned = self.safety_data[:, 1]
        actual = self.safety_data[:, 3]
        planned_plot = planned[interventions==1]
        plot_xs = np.arange(0, len( planned))
        plot_xs = plot_xs[interventions==1]

        plt.plot(actual, linewidth=1, color=pp[0])
        plt.plot(plot_xs, planned_plot, 'x', markersize=4, color=pp[2])


        plt.xlabel(f"Training steps ")
        plt.ylabel("Speed (m/s)")
        plt.tight_layout()
        # plt.xlim(0, 1500)
        # plt.ylim(-2, 50)
        plt.grid()
        plt.savefig(save + f"{self.vehicle_name}_SpeedGraph_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save + f"{self.vehicle_name}_SpeedGraph_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)

        # plt.show()

    def plot_deviation_training(self):
        save = self.path

        plt.figure(2, figsize=(6, 2.1))
        plt.clf()
        interventions = self.all_safety_data[:, 4]
        
        pts = self.all_states[:, 0:2]
        hs = []
        for point in pts:
            idx, dists = self.std_track.get_trackline_segment(point)
            x, h = self.std_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        
        plt.plot(hs[1:], linewidth=2, color=pp[0])
        plt.plot(true_moving_average(hs, 200), linewidth=2, color=pp[2])
        # plt.yscale("log")
        
        plt.set_ylabel(f"Deviation ")
        plt.set_xlabel("Training Steps")
        plt.tight_layout()
        plt.xlim(0, 1500)
        # plt.ylim(-2, 50)
        plt.grid()
        # plt.savefig(save + f"{self.vehicle_name}_Deviation.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save + f"{self.vehicle_name}_Deviation.pdf", bbox_inches='tight', pad_inches=0)

        # plt.show()

    


def explore_folder(path):
    TestData = AnalyseTestLapData()

    vehicle_folders = glob.glob(f"{path}*/")
    print(f"{len(vehicle_folders)} folders found")

    for j, folder in enumerate(vehicle_folders):
        print(f"Vehicle folder being opened: {folder}")
        
        # if os.path.exists(folder + "TrainSummaryStatistics.txt"):
        #     continue

        TestData.process_folder(folder)


def analyse_folder():

    path = "Data/Vehicles/Safe_TrainMaps/"


    explore_folder(path)

    # path += "Slow_Online_Std_None_f1_mco_4_0/"
    # path += "Slow_Online_Std_None_f1_aut_4_0/"
    # path += "Fast_Online_Std_Velocity_f1_esp_1_0/"
    # TestData = AnalyseTestLapData()
    # TestData.process_folder(path)


analyse_folder()

