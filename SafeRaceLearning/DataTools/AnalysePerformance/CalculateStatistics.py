from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.collections import LineCollection

from FoneTenth.DataTools.MapData import MapData
from FoneTenth.Utils.StdTrack import StdTrack 
from FoneTenth.Utils.RacingTrack import RacingTrack
from FoneTenth.Utils.utils import *

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
        self.safety_data = None

    def process_folder(self, folder):
        self.path = folder 

        self.initialise_folder()

        # for self.lap_n in range(2):
        for self.lap_n in range(50):
            if not self.load_lap_data(): break # no more laps
            
            self.load_safety_data()
            self.calculate_lap_statistics()

        self.generate_summary_stats()

    def initialise_folder(self):

        with open(self.path + "TestingStatistics.txt", "w") as file:
            file.write(f"Name: {self.path}\n")
            file.write("Lap" + "Steering".rjust(16) + "Total Distance".rjust(16) + "Mean Curvature".rjust(16) + "Total Curvature".rjust(16) + "Mean Deviation".rjust(16) + "Total Deviation".rjust(16) + "Progress".rjust(16) + "Time".rjust(16) + "Avg Velocity".rjust(16) + "Mean R Deviation".rjust(16) + "Total R Deviation".rjust(16) + "Interventions".rjust(16) + "\% Inter".rjust(16) + "\n")

        self.vehicle_name = self.path.split("/")[-2]
        vehicle_type = self.vehicle_name.split("_")[1] 
        # if vehicle_type != "PP" and vehicle_type != "Rando":
        if True:
            self.map_name = self.vehicle_name.split("_")[4]
            if self.map_name == "f1":
                self.map_name += "_" + self.vehicle_name.split("_")[5]
            if self.vehicle_name.split("_")[6] == "wide":
                self.map_name += "_" + self.vehicle_name.split("_")[6]
        else:
            self.map_name = self.vehicle_name.split("_")[3]
            if self.map_name == "f1":
                self.map_name += "_" + self.vehicle_name.split("_")[4]
            if self.vehicle_name.split("_")[5] == "wide":
                self.map_name += "_" + self.vehicle_name.split("_")[5]

        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)


    def load_lap_data(self):
        try:
            data = np.load(self.path + "Testing/" + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            # print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def load_safety_data(self):
        try:
            data = np.load(self.path + f"/SafeHistory/Lap_{self.lap_n}_safeHistory_{self.vehicle_name}.npy")
            self.safety_data = data
        except:
            self.safety_data = None

    def calculate_lap_statistics(self):
        if not self.load_lap_data(): return

        steering = np.abs(self.actions[:, 0])
        rms_steering = np.mean(np.abs(steering))

        pts = self.states[:, 0:2]
        ss = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_distance = np.sum(ss)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(pts, ss, False)
        mean_curvature = np.mean(np.abs(ks))
        total_curvature = np.sum(np.abs(ks))

        hs = []
        for point in pts:
            idx, dists = self.std_track.get_trackline_segment(point)
            x, h = self.std_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        mean_deviation = np.mean(hs)
        total_deviation = np.sum(hs)
        hs = []
        for point in pts:
            idx, dists = self.racing_track.get_trackline_segment(point)
            x, h = self.racing_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        mean_race_deviation = np.mean(hs)
        total_race_deviation = np.sum(hs)

        time = len(pts) /10
        vs = self.states[:, 3]
        avg_velocity = np.mean(vs)

        progress = self.std_track.calculate_progress(pts[-1])/self.std_track.total_s
        if progress < 0.01 or progress > 0.99:
            progress = 1 # it is finished

        if self.safety_data is not None: 
            interventions = np.sum(self.safety_data[:, 4])
            percent_inters = interventions *100 / len(self.safety_data)
        else: interventions, percent_inters = 0, 0

        with open(self.path + "TestingStatistics.txt", "a") as file:
            file.write(f"{self.lap_n}, {rms_steering:14.4f}, {total_distance:14.4f}, {mean_curvature:14.4f}, {total_curvature:14.4f}, {mean_deviation:14.4f}, {total_deviation:14.4f}, {progress:14.4f}, {time:14.2f}, {avg_velocity:14.4f}, {mean_race_deviation:14.2f}, {total_race_deviation:14.2f}, {interventions:10.2f}, {percent_inters:10.2f}\n")

    def generate_summary_stats(self):
        progress_ind = 7
        n_values = 14
        data = []
        for i in range(n_values): 
            data.append([])

        n_success, n_total = 0, 0
        progresses = []
        with open(self.path + "TestingStatistics.txt", 'r') as file:
            lines = file.readlines()
            if len(lines) < 3: return
            
            for lap_n in range(len(lines)-2):
                line = lines[lap_n+2] # first lap is heading
                line = line.split(',')
                progress = float(line[progress_ind])
                n_total += 1
                progresses.append(progress)
                if progress < 0.01 or progress > 0.99:
                    n_success += 1
                    for i in range(n_values):
                        data[i].append(float(line[i]))
                else:
                    continue
        
        progresses = np.array(progresses)
        data = np.array(data)
        with open(self.path + "TestingSummaryStatistics.txt", "w") as file:
            file.write(lines[0])
            file.write(lines[1])
            file.write("Mean")
            for i in range(1, n_values):
                if i == progress_ind:
                    file.write(f", {np.mean(progresses*100):14.4f}")
                else:
                    file.write(f", {np.mean(data[i]):14.4f}")
            file.write(f", {n_success/n_total * 100}")
            file.write("\n")

            file.write("Std")
            for i in range(1, n_values):
                if i == progress_ind:
                    file.write(f", {np.std(progresses*100):14.4f}")
                else:
                    file.write(f", {np.std(data[i]):14.4f}")
            file.write(f", {n_success/n_total * 100}")
            file.write("\n")



def explore_folder(path):
    TestData = AnalyseTestLapData()

    vehicle_folders = glob.glob(f"{path}*/")
    print(f"{len(vehicle_folders)} folders found")

    for j, folder in enumerate(vehicle_folders):
        print(f"Vehicle folder being opened: {folder}")
        
        # if os.path.exists(folder + "TestingStatistics.txt"):
        #     continue

                    

        TestData.process_folder(folder)


def analyse_folder():
    # path = "Data/Vehicles/SSS_RandomValidation/"
    # path = "Data/Vehicles/SSS_ppValidation/"
    # path = "Data/Vehicles/SSS_RandomSpeeds/"
    # path = "Data/Vehicles/SSS_NoisyAblation/"

    # path = "Data/Vehicles/Safe_TrainSteps/"
    path = "Data/Vehicles/Safe_TrainSpeeds/"
    # path = "Data/Vehicles/Std_TrainSteps/"
    
    
    
    explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
