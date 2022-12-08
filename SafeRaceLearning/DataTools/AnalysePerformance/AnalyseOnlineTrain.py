# from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from SuperSafety.DataTools.MapData import MapData
from SuperSafety.Utils.StdTrack import StdTrack 
from SuperSafety.Utils.RacingTrack import RacingTrack
from SuperSafety.Utils.utils import *
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

    def process_folder(self, folder):
        self.path = folder 
        self.initialise_folder()

        # for self.lap_n in range(2):
        for self.lap_n in range(20):
            if not self.load_lap_data(): break # no more laps
            
            self.load_safety_data()

            # self.plot_velocity_heat_map()
            # self.plot_friction_diagram()
            # self.generate_path_plot()
            # self.plot_friction_graphs()
            # if self.lap_n < 1:
            # self.plot_safety_training()
            self.plot_safe_velocity_heat_map()
            # self.plot_interventions()
            

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

        return 1 # to say success

    def load_safety_data(self):
        open_path = self.path + "SafeHistory/" + f"Lap_{self.lap_n}_safeHistory_{self.vehicle_name}.npy"
        try:
            data = np.load(open_path)
            self.safety_data = data
        except:
            print(f"No safety data for: " + open_path)
            self.safety_data = None

    def plot_velocity_heat_map(self): 
        save_path  = self.path + "TrainVelocities/"
        ensure_path_exists(save_path)
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        # N = len(points)
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.75)
        # plt.xlim(points[:, 0, 0].min()-10, points[:, 0, 0].max()+10)
        # plt.ylim(points[:, 0, 1].min()-10, points[:, 0, 1].max()+10)
        plt.gca().set_aspect('equal', adjustable='box')

        
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

    def plot_safe_velocity_heat_map(self): 
        save_path  = self.path + "TrainSafeVelocity/"
        ensure_path_exists(save_path)
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        # N = len(points)
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(8)
        line = plt.gca().add_collection(lc)
        plt.gca().set_aspect('equal', adjustable='box')

        pts = points[:-1]
        i_pts = pts[self.safety_data[:, 4]==1]
        for pt in i_pts:
            plt.plot(pt[0, 0], pt[0, 1], 'ro', markersize=8)

        
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        txt = "Lap " + str(self.lap_n)
        plt.text(190, 30, txt, fontsize=35, ha='left', backgroundcolor='white', color="#1B4F72")

        plt.xlim(160, 590)
        plt.ylim(0, 350)
        plt.tight_layout()

        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_traj_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.95)
        cbar.ax.tick_params(labelsize=25)
        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_traj_cb_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig(save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}.svg", bbox_inches='tight')

        plt.close()

    def generate_path_plot(self):
        save_path = self.path + "TrainPaths/"
        ensure_path_exists(save_path)
        plt.figure(1, figsize=(3, 3))
        plt.clf()

        points = self.states[:, 0:2]        
        self.map_data.plot_map_img()
        self.map_data.plot_centre_line()
        xs, ys = self.map_data.pts2rc(points)
        plt.plot(xs, ys, color='darkorange')

        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        txt = self.vehicle_name.split('_')[1]
        if len(txt)==3: txt = "Centre line"
        elif len(txt)==7: txt = "Race line"
        plt.text(730, 90, txt, fontsize=11, ha='left', backgroundcolor='white', color='darkblue')

        plt.xlim(700, 1100)
        plt.ylim(60, 440)
        plt.tight_layout()
        # plt.show()

        plt.savefig(save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}.svg", bbox_inches='tight')
        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_clipPath_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)
    
    def plot_safety_training(self):
        save_path = self.path + "TrainSafety/"
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
                    plt.plot(xs[i], ys[i], 'go', markersize=4)
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
        
        txt = "Lap " + str(self.lap_n)
        plt.text(190, 30, txt, fontsize=25, ha='left', backgroundcolor='white', color="#1B4F72")

        plt.xlim(160, 590)
        plt.ylim(0, 350)
        plt.tight_layout()

        plt.savefig(save_path + f"{self.vehicle_name}_safety_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            plt.savefig(save_path + f"{self.vehicle_name}_SafetyPath_{self.lap_n}.pdf", bbox_inches='tight', pad_inches=0)
        
        # plt.show()
        plt.close()
        
    def plot_interventions(self):
        save = self.path + "TrainInterventions/"
        ensure_path_exists(save)

        if self.safety_data.all() == None: return

        plt.figure(1)
        plt.clf()
        interventions = self.safety_data[:, 4]
        n= 20
        i = 0 
        inter_ns = []
        while i < len(interventions)-1:
            inter_ns.append(0)
            for _ in range(20):
                if interventions[i] > 0.001:
                    inter_ns[-1] += 1
                i += 1
                if i >= len(interventions)-1:
                    break
        
        plt.plot(inter_ns)
        plt.xlabel(f"Training steps ({n})")
        plt.ylabel("Interventions")
        plt.tight_layout()
        plt.savefig(save + f"{self.vehicle_name}_interventions_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)

        

def get_velcoity(delta, v_max, f_s):
    b = 0.523
    g = 9.81
    L = 0.33

    delta = np.abs(delta)
    delta = np.clip(delta, 0.01, 1) # prevents zeros
    # delta[delta<0.01] = 0.01
    vs = f_s * np.sqrt(b*g*L/np.tan(abs(delta)))
    vs = np.clip(vs, 0, v_max)
    # vs[0] = 0
    # vs[-1] = 0

    return vs

def explore_folder(path):
    TestData = AnalyseTestLapData()

    vehicle_folders = glob.glob(f"{path}*/")
    print(f"{len(vehicle_folders)} folders found")

    for j, folder in enumerate(vehicle_folders):
        print(f"Vehicle folder being opened: {folder}")
        
        # if os.path.exists(folder + "TrainStatistics.txt"):
        #     continue
        # if j > 0:
        #     continue
        
        # vehicle = folder.split('/')[-2]
        # set_n = int(vehicle.split('_')[6])
        # if set_n != 3: 
        #     print(f"Not set 3: {set_n}")
        #     continue

        TestData.process_folder(folder)


def analyse_folder():

    # path = "Data/Vehicles/FastOnline/"
    # path = "Data/Vehicles/SlowOnline/"
    # path = "Data/Vehicles/OnlineDevel/"
    # path = "Data/Vehicles/OnlineFastT/"
    # path = "Data/Vehicles/RewardVelocity/"
    path = "Data/Vehicles/Safe_TrainMaps/"
    # path = "Data/Vehicles/FastCompareESP/"
    
    # path = "Data/Vehicles/SlowAutStudy/"
    


    # explore_folder(path)
# 
    TestData = AnalyseTestLapData()
    # TestData.process_folder(path + "Fast_Rando_Super_f1_mco_1_0/")
    TestData.process_folder(path + "fast_Online_Std_Velocity_f1_aut_6_1_3/")
    # TestData.process_folder(path + "Fast_Online_Std_Ppps_f1_mco_6_0/")
    # TestData.process_folder(path + "Slow_Rando_Std_f1_aut_wide_1_0/")

if __name__ == '__main__':
    analyse_folder()
