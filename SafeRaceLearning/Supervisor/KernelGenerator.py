import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SafeRaceLearning.Utils.utils import load_conf, save_kernel_config
from SafeRaceLearning.Supervisor.DynamicsBuilder import build_dynamics_table
from SafeRaceLearning.Supervisor.Modes import FastModes, SlowModes

from SafeRaceLearning.Supervisor.GenerationUtils import *

class KernelGenerator:
    def __init__(self, track_img, conf):
        self.track_img = np.array(track_img, dtype=bool)
        self.conf = conf
        self.n_dx = int(conf.n_dx)
        self.t_step = conf.kernel_time_step
        self.n_phi = conf.n_phi
        self.max_steer = conf.max_steer 
        self.L = conf.l_f + conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-np.pi, np.pi, self.n_phi)
        
        if conf.kernel_mode =="fast":
            m = FastModes(conf)
        elif conf.kernel_mode == "slow":
            m = SlowModes(conf)
        self.n_modes = m.n_modes
        self.qs = m.qs

        self.o_map = np.copy(track_img)    
        self.fig, self.axs = plt.subplots(2, 2, dpi=1200)

        self.ref_table = - np.ones_like(self.track_img, dtype=int)
        self.kl = None
        self.x_inds, self.y_inds = None, None
        self.init_kernel_table()
        print(f"KernelList Shape: {self.kl.shape}")
        self.previous_kl = np.zeros_like(self.kl)

        self.get_filled_kernel()

        filename = f"{conf.dynamics_path}{conf.kernel_mode}_{conf.max_v}_{int(conf.kernel_time_step*10)}_dyns.npy"
        self.dynamics = np.load(filename)
        print(f"Dynamics Loaded: {self.dynamics.shape}")
        print(f"Dynamics table Loaded from {filename}")


        self.name = f"Kernel_{conf.kernel_mode}_{conf.max_v}_{conf.map_name}"
        self.loop = 0

    def init_kernel_table(self):
        self.x_inds, self.y_inds = np.asarray(self.track_img == 0).nonzero()
        self.kl = np.zeros((len(self.x_inds), self.n_phi, self.n_modes), dtype=bool)

        print(f"Shape: {self.x_inds.shape}")

        self.ref_table[self.x_inds, self.y_inds] = np.arange(len(self.x_inds))

    def get_filled_kernel(self):
        prev_filled = np.count_nonzero(self.previous_kl)
        filled = np.count_nonzero(self.kl)
        total = self.kl.size
        print(f"Filled: {filled} / {total} -> {(100*filled/total):.4f}% --> diff: {max(filled-prev_filled, 0)}")
        return filled/total


    def view_kernel_angles(self, show=True, save=False):
        resolution = 100
        plt.close()
        self.fig, self.axs = plt.subplots(2, 2)
        # self.fig, self.axs = plt.subplots(2, 2,  dpi=resolution)

        # plt.gcf().clf()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        if self.kl.shape[2] > 1:
            mode_ind = int((self.n_modes-1)/2)
        else: mode_ind = 0
        mode_ind = 2
        # mode_ind = 10

        import matplotlib
        cmap = matplotlib.colors.ListedColormap(["#CACFD2", "#2ECC71", "#E74C3C"])

        img = make_imgs(self.ref_table, self.kl, self.x_inds, self.y_inds, 0, mode_ind)
        self.axs[0, 0].imshow(img.T, origin='lower', cmap=cmap)

        diff = np.logical_xor(self.kl, self.previous_kl)
        img = make_imgs(self.ref_table, diff, self.x_inds, self.y_inds, quarter_phi, mode_ind)
        self.axs[1, 0].imshow(img.T, origin='lower', cmap=cmap)

        img = make_imgs(self.ref_table, self.kl, self.x_inds, self.y_inds, half_phi, mode_ind)
        self.axs[0, 1].imshow(img.T, origin='lower', cmap=cmap)

        img = make_imgs(self.ref_table, self.kl, self.x_inds, self.y_inds, 3*quarter_phi, mode_ind)
        self.axs[1, 1].imshow(img.T, origin='lower', cmap=cmap)

        self.axs[0, 0].set_xticks([])
        self.axs[0, 0].set_yticks([])
        self.axs[1, 0].set_xticks([])
        self.axs[1, 0].set_yticks([])
        self.axs[0, 1].set_xticks([])
        self.axs[0, 1].set_yticks([])
        self.axs[1, 1].set_xticks([])
        self.axs[1, 1].set_yticks([])

        plt.tight_layout()

        if save:
            plt.savefig(f"{self.conf.kernel_path}{self.name}.svg", pad_inches=0.0, bbox_inches='tight')

        # plt.pause(0.0001)
        # plt.pause(1)

        if show:
            plt.show()

    def calculate_kernel(self, n_loops=20):
        for self.loop in range(1, n_loops):
            print(f"Running loop: {self.loop}")
            if np.all(self.previous_kl == self.kl) and self.loop > 1:
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kl = np.copy(self.kl)
            self.kl = viability_loop(self.kl, self.ref_table, self.x_inds, self.y_inds, self.dynamics)

            self.view_kernel_angles(False)
            self.get_filled_kernel()

        return self.get_filled_kernel()

@njit(cache=True)
def make_imgs(ref_table, kl, x_inds, y_inds, p, m):
    img = np.copy(ref_table)
    for ind in range(len(x_inds)): #TODO: can probs just slice this array
        x, y = x_inds[ind], y_inds[ind]
        k_ref = ref_table[x, y]
        img[x, y] = kl[k_ref, p, m]

    return img

@njit(cache=True)
def viability_loop(kl, r_tab, xi, yi, dynamics):
    previous_kl = np.copy(kl)
    l_is, l_ps, l_qs = kl.shape

    for ind in range(l_is):
        i, j = xi[ind], yi[ind]
        for p in range(l_ps):
            for q in range(l_qs):
                if kl[ind, p, q] == 1:
                    continue
                a = check_viable_state(i, j, p, q, dynamics, previous_kl, r_tab)
                kl[ind, p, q] = a 

    return kl


@njit(cache=True)
def check_viable_state(i, j, p, q, dynamics, previous_kl, r_tab):
    # l_inds, l_phis, n_modes = previous_kl.shape
    l_xs, l_ys = r_tab.shape
    _p, n_state_ms, n_act_ms, _i, _dim = dynamics.shape
    for l in range(n_act_ms):
        safe = True
        di, dj, new_k, new_q = dynamics[p, q, l, 0, :]
        if new_q == -9223372036854775808 or new_q == -1:
            continue # the transition isn't valid, 

        for n in range(dynamics.shape[3]): 
            di, dj, new_k, new_q = dynamics[p, q, l, n, :]

            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

            ind = r_tab[new_i, new_j]
            if ind == -1:
                safe = False # breached a limit.
                break # try again and look for a new action
            elif previous_kl[ind, new_k, new_q]:
                safe = False # not safe.
                break # try again and look for a new action

        if safe: # there exists a valid action
            return False # it is safe

    return True # it isn't safe because I haven't found a valid action yet...




def build_track_kernel(conf):
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, conf.track_shrink_pixels)
    kernel = KernelGenerator(img2, conf)
    kernel.view_kernel_angles(False)
    kernel.calculate_kernel(100)

    np.save(f"{conf.kernel_path}{kernel.name}_kl.npy", kernel.kl)
    print(f"Saved kernel to file: {kernel.name}")

    process_result(conf, kernel)

    save_kernel_config(conf, kernel.name)
    kernel.view_kernel_angles(False, True)



def generate_slow_kernels():
    print("Generating SLOW kernels")
    print("--------------------------------------------------")
    conf = load_conf("slow_kernel_config")
    build_dynamics_table(conf)
    map_names = ["f1_esp", "f1_mco", "f1_aut", "f1_aut_wide"]
    for map_name in map_names:
        conf.map_name = map_name

        print_kernel_conf_params(conf)

        build_track_kernel(conf)
        
        filter_slow_kernel(conf.map_name)



def generate_fast_kernels():
    print("Generating FAST kernels")
    print("--------------------------------------------------")
    conf = load_conf("fast_kernel_config")
    
    build_dynamics_table(conf)
    map_names = ["f1_esp", "f1_mco", "f1_aut", "f1_aut_wide"]
    for map_name in map_names:
        conf.map_name = map_name

        print_kernel_conf_params(conf)

        build_track_kernel(conf)



def generate_single_kernel():
    print("Generating single kernel for testing")
    print("--------------------------------------------------")
    
    conf = load_conf("fast_kernel_config")
    # conf = load_conf("slow_kernel_config")
    
    build_dynamics_table(conf)
    
    conf.map_name = "f1_esp"
    # conf.map_name = "f1_mco"
    # conf.map_name = "f1_aut"
    # conf.map_name = "f1_aut_wide"

    print_kernel_conf_params(conf)

    build_track_kernel(conf)


if __name__ == "__main__":

    # generate_single_kernel()
    generate_fast_kernels()
    # generate_slow_kernels()


