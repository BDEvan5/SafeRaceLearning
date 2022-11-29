import yaml
from PIL import Image
import numpy as np
from numba import njit

from SafeRaceLearning.Utils.utils import *
from argparse import Namespace


def prepare_track_img(conf):
    file_name = 'maps/' + conf.map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())
    img_resolution = yaml_file['resolution']
    map_img_path = 'maps/' + yaml_file['image']

    resize = int(conf.n_dx * img_resolution)

    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float64)
    if len(map_img.shape) == 3:
        map_img = map_img[:, :, 0]
    map_img[map_img <= 128.] = 1.
    map_img[map_img > 128.] = 0.

    img = Image.fromarray(map_img.T)
    img = img.resize((map_img.shape[0]*resize, map_img.shape[1]*resize))
    img = np.array(img)
    map_img2 = img.astype(np.float64)
    map_img2[map_img2 != 0.] = 1.

    return map_img2


@njit(cache=True)
def shrink_img(img, n_shrinkpx):
    o_img = np.copy(img)

    search = np.array([[0, 1], [1, 0], [0, -1], 
                [-1, 0], [1, 1], [1, -1], 
                [-1, 1], [-1, -1]])
    for i in range(n_shrinkpx):
        t_img = np.copy(img)
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j, k] == 1:
                    continue
                for l in range(len(search)):
                    di, dj = search[l, :]
                    new_i = min(max(0, j + di), img.shape[0]-1)
                    new_j = min(max(0, k + dj), img.shape[1]-1)
                    if t_img[new_i, new_j] == 1:
                        img[j, k] = 1.
                        break

    print(f"Finished Shrinking")
    return o_img, img #



def process_result(conf, kernel):
    unsafe = kernel.get_filled_kernel()
    print(f"\% Safe states: {1-unsafe}")
    print(f"Kernel shape: {kernel.kl.shape}")
    print(f"Reference table shape: {kernel.ref_table.shape}")
    print(f"Total positions: {kernel.ref_table.shape[0] * kernel.ref_table.shape[1]}")
    print(f"Iterations: {kernel.loop}")

    conf.safe_rate = (1-unsafe)*100
    conf.kernel_shape = kernel.kl.shape
    conf.ref_table_shape = kernel.ref_table.shape
    conf.total_positions = kernel.ref_table.shape[0] * kernel.ref_table.shape[1]
    conf.iterations = kernel.loop


def print_kernel_conf_params(conf):
    print(f"Starting to build kernel for: {conf.map_name}")
    print(f"Kernel Mode: {conf.kernel_mode}")
    print(f"Kernel Resolution: {conf.n_dx}")
    print(f"Kernel Shrink Pixels: {conf.track_shrink_pixels}")
    print(f"KernelTime: {conf.kernel_time_step}")
    print(f"Vehicle speed: {conf.max_v}")
    print(f"Steer modes: {conf.nq_steer}")

    print("---------------------------------------------")
    print()





def filter_slow_kernel(map_name):
    p = "Data/Kernels/"
    kl = np.load(p + f"Kernel_slow_2_{map_name}_kl.npy")
    new_kl = np.zeros((kl.shape[0], kl.shape[1], 1), dtype=bool)

    new_kl = run_slow_filter(kl, new_kl)

    np.save(p + f"Kernel_slow_2_{map_name}_kl_filter.npy", new_kl)

    conf = {}
    conf['map_name'] = map_name
    conf["n_dx"] = 40
    conf = Namespace(**conf)
    unsafe = np.count_nonzero(new_kl)/new_kl.size
    conf.safe_rate = (1-unsafe)*100
    conf.kernel_shape = new_kl.shape

    save_kernel_config(conf, f"Kernel_slow_2_{map_name}_kl_filter")

    

@njit(cache=True)
def run_slow_filter(kl, new_kl):
    for i in range(kl.shape[0]):
        for j in range(kl.shape[1]):
            new_kl[i, j] = kl[i, j].any()

    return new_kl


def save_track_imgs():
    p = "Data/Kernels/"
    # map_names = ["f1_aut", "f1_gbr", "Levine"]
    map_names = ["f1_esp", "f1_aut_wide", "f1_mco", "f1_aut"]
    for map_name in map_names:
        conf = {}
        conf['map_name'] = map_name
        conf["n_dx"] = 40
        conf = Namespace(**conf)
        track_img = prepare_track_img(conf) 
        track, track_img = shrink_img(track_img, 8)
        track_img = np.array(track_img, dtype=bool)

        np.save(p + f"track_img_{conf.map_name}.npy", track_img)



if __name__ == "__main__":
    save_track_imgs()
