from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter, MultipleLocator, MaxNLocator
from matplotlib.collections import LineCollection

from SafeRaceLearning.DataTools.MapData import MapData
from SafeRaceLearning.Utils.StdTrack import StdTrack 
from SafeRaceLearning.Utils.RacingTrack import RacingTrack
from SafeRaceLearning.Utils.utils import *

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

        for self.lap_n in range(2):
        # for self.lap_n in range(50):
        # for self.lap_n in range(5):
            if not self.load_lap_data(): break # no more laps
            self.load_safety_data()

            self.plot_velocity_heat_map()
            self.generate_safety_path()


    def initialise_folder(self):

        with open(self.path + "TestingStatistics.txt", "w") as file:
            file.write(f"Name: {self.path}\n")
            file.write("Lap" + "Steering".rjust(16) + "Total Distance".rjust(16) + "Mean Curvature".rjust(16) + "Total Curvature".rjust(16) + "Mean Deviation".rjust(16) + "Total Deviation".rjust(16) + "Progress".rjust(16) + "Time".rjust(16) + "Avg Velocity".rjust(16) + "Mean R Deviation".rjust(16) + "Total R Deviation".rjust(16) + "Interventions".rjust(16) + "\% Inter".rjust(16) + "\n")

        self.vehicle_name = self.path.split("/")[-2]
        vehicle_type = self.vehicle_name.split("_")[1] 
        if vehicle_type != "PP" and vehicle_type != "Rando":
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

    def generate_steering_graphs(self):
        save_path = self.path + "TestingSteeringGraphs/"
        ensure_path_exists(save_path)
        # if self.lap_n != 0: return # only do it for one lap.
        steering = self.actions[:, 0]

        name = self.vehicle_name.split("_")[1]
        if name == "C2": name = "PP"
        
        plt.figure(1)
        plt.clf()
        color = "orange"
        plt.gca().hist(steering, bins=9, density=False, weights=np.ones(len(steering)) / len(steering), color=color)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.xlim(-0.45, 0.45)
        plt.ylim(0, 0.35)

        plt.xticks([])
        plt.yticks([])
        plt.title(f"{name}", fontsize=24)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{save_path}{self.vehicle_name}_steering_hist_{self.lap_n}.png", bbox_inches='tight')

    def plot_curvature_heat_map(self): 
        save_path = self.path + "TestingCurvatures/"
        ensure_path_exists(save_path)
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        diffs = np.diff(self.states[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(self.states[:, :2], seg_lengths, False)

        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, ks.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(abs(ks))
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line,fraction=0.046, pad=0.04)
        plt.gca().set_aspect('equal', adjustable='box')

        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.savefig(save_path + f"{self.vehicle_name}_curve_map_{self.lap_n}.svg", bbox_inches='tight')
        if SAVE_PDF:
            plt.savefig(save_path + f"Curvature_{self.lap_n}_{self.vehicle_name}.pdf", bbox_inches='tight', pad_inches=0)

    def plot_velocity_heat_map(self): 
        save_path  = self.path + "SafeVelocities/"
        ensure_path_exists(save_path)
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 6)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        # cbar.ax.major_locator(MaxNLocator(integer=True))\
        cbar.ax.get_yaxis().set_major_locator(MultipleLocator(1))
        cbar.ax.tick_params(labelsize=20)
        # cbar.ax.tick_params(labelsize=14)
        # plt.xlim(points[:, 0, 0].min()-10, points[:, 0, 0].max()+10)
        # plt.ylim(points[:, 0, 1].min()-10, points[:, 0, 1].max()+10)
        plt.gca().set_aspect('equal', adjustable='box')


        # txt = self.vehicle_name.split('_')[3]
        # if len(txt)==8: txt = "Online"
        # elif len(txt)==3: txt = "E2e"
        # txt = self.vehicle_name.split("_")[2]
        # if len(txt)==5: txt = "SSS"
        # elif len(txt)==3: txt = "PP"
        # plt.text(20, 50, txt, fontsize=25, ha='left', backgroundcolor='white', color="#1B4F72")

        # plt.xlim(0, 400)
        # plt.ylim(20, 520)
        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # plt.show()

        plt.savefig(save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}.svg", bbox_inches='tight')
        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_traj_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)

        plt.close()

    def generate_path_plot(self):
        save_path = self.path + "TestingPaths/"
        ensure_path_exists(save_path)
        resolution = 300
        plt.figure(1, dpi=resolution)
        # plt.figure(1, figsize=(3, 3), dpi=resolution)
        plt.clf()

        points = self.states[:, 0:2]        
        self.map_data.plot_map_img()
        self.map_data.plot_centre_line()
        xs, ys = self.map_data.pts2rc(points)
        plt.plot(xs, ys, color=path_orange, linewidth=4)
        # plt.plot(xs, ys, color='darkorange')

        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # txt = self.vehicle_name.split('_')[3]
        # if len(txt)==4: txt = "Online"
        # elif len(txt)==3: txt = "E2e"
        # elif len(txt)==2: txt = "PP"
        # plt.text(20, 50, txt, fontsize=25, ha='left', backgroundcolor='white', color="#1B4F72")

        # plt.xlim(0, 400)
        # plt.ylim(20, 520)
        # plt.xlim(700, 1100)
        # plt.ylim(60, 440)
        plt.tight_layout()
        # plt.show()

        plt.savefig(save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}.svg", bbox_inches='tight', dpi=resolution)
        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_clipPath_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0, dpi=resolution)

    def generate_safety_path(self):
        save_path = self.path + "TestingSafety/"
        ensure_path_exists(save_path)
        plt.figure(1)
        plt.clf()

        points = self.states[:, 0:2]        
        self.map_data.plot_map_img()
        # self.map_data.plot_centre_line()
        xs, ys = self.map_data.pts2rc(points)

        if self.safety_data is not None:
            for i in range(len(xs)-2):
                if self.safety_data[i, 4] > 0.001:
                    plt.plot(xs[i], ys[i], 'ro', markersize=4)
                else:
                    plt.plot(xs[i], ys[i], 'go', markersize=6)
        else: 
            for i in range(len(xs)-2):
                plt.plot(xs[i], ys[i], 'go', markersize=2)

        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # plt.xlim(700, 1100)
        # plt.ylim(60, 440)
        plt.tight_layout()

        plt.savefig(save_path + f"{self.vehicle_name}_safety_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_clipPath_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)
        
        # plt.show()
        plt.close()
        



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
    path = "Data/Vehicles/SSS_NoisyAblation/"
    # path = "Data/Vehicles/KernelValidationPP/"

    explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
