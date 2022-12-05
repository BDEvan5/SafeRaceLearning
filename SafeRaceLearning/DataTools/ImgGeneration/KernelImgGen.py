import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from SafeRaceLearning.Supervisor.KernelGenerator import *
from SafeRaceLearning.Utils.utils import *


class KernelImg:
    def __init__(self, conf, map_name):
        kernel_name = conf.kernel_path + f"Kernel_fast_{conf.max_v}_{map_name}"
        kernel_conf = load_kernel_config(kernel_name)
        self.m = FastModes(kernel_conf)
        self.qs = self.m.qs
        
        
        self.d_max = kernel_conf.max_steer
        self.resolution = kernel_conf.n_dx
        self.phi_range = kernel_conf.phi_range
        self.max_steer = kernel_conf.max_steer
        self.phis = np.linspace(-conf.phi_range/2, conf.phi_range/2, conf.n_phi)

        file_name =  f'maps/' + map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])

        self.kl = np.load(kernel_name + "_kl.npy")
        track_img = np.load(conf.kernel_path + "track_img_" + map_name + ".npy")
        self.ref_table = - np.ones_like(track_img, dtype=int)
        self.xs, self.ys = np.asarray(track_img == 0).nonzero()
        self.ref_table[self.xs, self.ys] = np.arange(self.kl.shape[0]) # this is a check that the length is correct
        print(f"Kernel Shape: {self.kl.shape}")
        
    def heat_map_view(self):
        # import matplotlib 
        # matplotlib.rc('ytick', labelsize=15) 
        plt.rcParams.update({'font.size': 15})
        modes = np.arange(2, 27, 5)
        print(f"Modes to use: {modes}")
        print(f"M vals: {self.qs[modes]}")

        kernel_colors = KernelColours()
        
        o_map = self.ref_table.copy()
        o_map[self.ref_table != -1] = 0
        o_map[self.ref_table == -1] = 1
        
        phis = np.arange(0, len(self.phis), 5, dtype=int)
        for phi_ind in phis:
            plt.close()
            plt.figure(1)
            plt.clf()
            img = np.zeros_like(self.ref_table, dtype=int)
            for m in modes:
                img += compose_kernel_img(self.kl, self.ref_table, self.xs, self.ys, phi_ind, m)
                # img += 1-self.kernel[:, :, phi_ind, m]

            plt.imshow(img.T + o_map.T, origin='lower', cmap=kernel_colors.cm)
            # plt.imshow(img.T + o_map.T * 7, origin='lower', cmap=kernel_colors.cm)
            cbar = plt.colorbar(shrink=0.85, aspect=7)
            cbar.ax.get_yaxis().set_ticks([])
            tiks = [f"{self.qs[m, 1]}" for m in modes]
            tiks.insert(0, "None")
            tiks.append("All")
            tiks.append("Track")
            for j, lab in enumerate(tiks):
                cbar.ax.text(.5, 0.3 + j*0.75, lab, ha='center', va='center', fontsize=15)
            cbar.ax.set_ylabel('Mode Speed (m/s)', rotation=270)
            cbar.ax.get_yaxis().labelpad = 15
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


            plt.xticks([])
            plt.yticks([])
            plt.xlim(80, 1140)
            plt.ylim(40, 920)

            plt.tight_layout()
            # plt.savefig(f"Data/KernelImgs/HeatMaps/KernelHeatMap6_{phi_ind}.pdf", pad_inches=0, bbox_inches='tight')
            plt.savefig(f"Data/KernelImgs/HeatMaps/KernelHeatMap6_{phi_ind}.svg", pad_inches=0, bbox_inches='tight')

            # plt.show()
        
    def compose_kernel_img(self):
        kernel = compose_kernel_img(self.kl, self.ref_table, self.xs, self.ys, 0, 2)
        plt.imshow(kernel.T)
        
        plt.show()
        
# @njit(cache=True)     
def compose_kernel_img(kl, ref_table, xs, ys, phi, q):
    kernel = ref_table.copy()

    kernel[kernel != -1] = 0
    kernel[kernel==-1] = 1
    
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        kernel[x, y] = kl[ref_table[x, y], phi, q]
        
    return kernel
     
        
class KernelColours:
    def __init__(self):
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
        self.cm = ListedColormap([green, y_light, yellow, o2, o3, r2, red, grey])
        # self.cm = ListedColormap([red, r2, o2, o0, yellow, y2, y_light, green, grey])
        # self.cm = ListedColormap([red, r2, o3, o2, o1, o0, yellow, y2, y_light, green, grey])


    
    
if __name__ == "__main__":
    conf = load_conf("kernel_generation_config")
    kernel = KernelImg(conf, "f1_aut")
    
    kernel.heat_map_view()
    # kernel.compose_kernel_img()
