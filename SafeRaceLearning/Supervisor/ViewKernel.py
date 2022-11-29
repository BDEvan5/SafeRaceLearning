import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from SafeRaceLearning.Supervisor.KernelGenerator import *
from SafeRaceLearning.Utils.utils import *

kernel_cm = ListedColormap(["#2ECC71", "#E74C3C", "#CACFD2"])

class VeiwKernel:
    def __init__(self, conf, track_img):
        self.name = f"Kernel_{conf.kernel_mode}_{conf.max_v}_{conf.map_name}"
        self.kernel_mode = conf.kernel_mode
        kernel_name = f"{conf.kernel_path}Kernel_{conf.kernel_mode}_{conf.max_v}_{conf.map_name}.npy"
        self.kernel = np.load(kernel_name)
        print(f"Kernel Shape: {self.kernel.shape}")

        self.o_map = np.copy(track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        
        self.phis = np.linspace(-conf.phi_range/2, conf.phi_range/2, conf.n_phi)

        if conf.kernel_mode == "fast": m = FastModes(conf)
        elif conf.kernel_mode == "slow": m = SlowModes(conf)
        else: raise ValueError(f"Kernel mode not found: {conf.kernel_mode}")
        self.qs = m.qs

        # self.heat_map_view()

        self.view_kernel_angles(True)
        # self.view_speed_build(True)
     
    def view_speed_build(self, show=True):
        plt.close()
        self.fig, self.axs = plt.subplots(2, 2)

        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        phi_ind = int(len(self.phis)/2)

        inds = np.array([1, 10, 25, 26], dtype=int)
        # inds = np.array([1, 1, 8, 10], dtype=int)

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, inds[0]].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel Mode: {self.qs[inds[0]]}")
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, inds[1]].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel Mode: {self.qs[inds[1]]}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, inds[2]].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel Mode: {self.qs[inds[2]]}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, inds[3]].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel Mode: {self.qs[inds[3]]}")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()

    def heat_map_view(self):

        assert self.kernel_mode == "fast", "Heat map only works for fast kernel"

        modes = np.arange(1, 25, 3)
        print(f"Modes to use: {modes}")
        print(f"M vals: {self.qs[modes]}")

        grey = "#CACFD2" # 9
        green = "#2ECC71"  # 8
        y_light = "#F7DC6F"
        yellow = "#F1C40F" # 7
        y2 = "#F4D03F" # 6
        o0 = "#F5B041" # 5
        o2 = "#F39C12" # 4
        o1 = "#EB984E" # 2
        o3 = "#E67E22" # 3
        r2 = "#EC7063" #1 
        red = "#E74C3C" # 0
        heat_cm = ListedColormap([red, r2, o3, o2, o1, o0, yellow, y2, y_light, green, grey])

        for phi_ind in range(len(self.phis)):
            plt.close()
            plt.figure(1)
            plt.clf()
            img = np.zeros_like(self.kernel[:, :, 0, 0], dtype=int)
            for m in modes:
                img += 1-self.kernel[:, :, phi_ind, m]

            plt.imshow(img.T + self.o_map.T * 9, origin='lower', cmap=heat_cm)
            cbar = plt.colorbar(shrink=0.9, aspect=8)
            cbar.ax.get_yaxis().set_ticks([])
            tiks = [f"{self.qs[m, 1]}" for m in modes]
            tiks.insert(0, "None")
            tiks.append("All")
            tiks.append("Track")
            for j, lab in enumerate(tiks):
                cbar.ax.text(.5, 0.5 + j*0.8, lab, ha='center', va='center')
            cbar.ax.set_ylabel('Mode Speed', rotation=270)
            cbar.ax.get_yaxis().labelpad = 15

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


            plt.xticks([])
            plt.yticks([])

            plt.tight_layout()
            plt.savefig(f"Data/KernelImgs/HeatMaps/KernelHeatMap8_{phi_ind}.pdf", pad_inches=0, bbox_inches='tight')
            plt.savefig(f"Data/KernelImgs/HeatMaps/KernelHeatMap8_{phi_ind}.svg", pad_inches=0, bbox_inches='tight')

            # plt.show()
        
    def make_picture(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)


        self.axs[0, 0].set(xticks=[])
        self.axs[0, 0].set(yticks=[])
        self.axs[1, 0].set(xticks=[])
        self.axs[1, 0].set(yticks=[])
        self.axs[0, 1].set(xticks=[])
        self.axs[0, 1].set(yticks=[])
        self.axs[1, 1].set(xticks=[])
        self.axs[1, 1].set(yticks=[])

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        
        plt.pause(0.0001)
        plt.pause(1)
        plt.savefig(f"{self.conf.kernel_path}Kernel_build_{self.conf.kernel_mode}.svg")

        if show:
            plt.show()

    def save_kernel(self, name):

        self.view_speed_build(False)
        plt.savefig(f"{self.conf.kernel_path}KernelSpeed_{name}_{self.conf.kernel_mode}.png")

        self.view_angle_build(False)
        plt.savefig(f"{self.conf.kernel_path}KernelAngle_{name}_{self.conf.kernel_mode}.png")

    def view_kernel_angles(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        mode_ind = 0

        self.axs[0, 0].imshow(self.kernel[:, :, 0, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        plt.pause(0.0001)
        # plt.pause(1)

        if show:
            plt.show()

    def save_angle_imgs(self):
        mode_ind = 2 # straigt

        for p, _  in enumerate(self.phis):
            plt.figure(1, dpi=300)
            plt.clf()

            img = self.kernel[:, :, p, mode_ind].T + self.o_map.T
            plt.imshow(img, origin='lower', cmap=kernel_cm)

            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            deg = int(np.rad2deg(self.phis[p]))
            plt.savefig(f"Data/KernelImgs/Angles/{self.name}_{deg}.png", dpi=300, pad_inches=0, bbox_inches='tight')


    def save_mode_imgs(self):
        phi_ind = 0

        for m, _  in enumerate(self.qs):
            plt.figure(1, dpi=300)
            plt.clf()

            img = self.kernel[:, :, phi_ind, m].T + self.o_map.T
            plt.imshow(img, origin='lower', cmap=kernel_cm)

            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            # deg = int(np.rad2deg(self.phis[m]))
            plt.savefig(f"Data/KernelImgs/KernelModes/{self.name}_{m}.png", dpi=300, pad_inches=0, bbox_inches='tight')


def view_kernel():
    conf = load_conf("kernel_config")
    conf.map_name = "f1_esp"
    # conf.map_name = "f1_aut_wide"
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, 5)
    k = VeiwKernel(conf, img2)
    # k.heat_map_view()
    # k.save_mode_imgs()
    # k.save_angle_imgs()


view_kernel()
