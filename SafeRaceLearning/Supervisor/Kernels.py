import numpy as np
from matplotlib import pyplot as plt
import yaml
from SafeRaceLearning.Supervisor.Modes import *
from numba import njit
from SafeRaceLearning.Utils.utils import *


class KernelList:
    def __init__(self, conf, run):
        assert run.filter == False, "Incorrect filter mode in Kernel List"

        if run.architecture == "fast":
            kernel_name = conf.kernel_path + f"Kernel_{run.architecture.lower()}_{conf.max_v}_{run.map_name}"
            kernel_conf = load_kernel_config(kernel_name)
            self.m = FastModes(kernel_conf)
        elif run.architecture == "slow":
            kernel_name = conf.kernel_path + f"Kernel_{run.architecture.lower()}_2_{run.map_name}"
            kernel_conf = load_kernel_config(kernel_name)
            self.m = SlowModes(kernel_conf)
        else:
            raise ValueError(f"Unknown racing mode: {run.racing_mode}")

        self.d_max = kernel_conf.max_steer
        self.resolution = kernel_conf.n_dx
        self.phi_range = kernel_conf.phi_range
        self.max_steer = kernel_conf.max_steer

        file_name =  f'maps/' + run.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])

        self.kl = np.load(kernel_name + "_kl.npy")
        try: 
            track_img = np.load(conf.kernel_path + "track_img_" + run.map_name + ".npy")
        except FileNotFoundError:
            print(f"Track images not yet generated: run GenerationUtils.py to generate images for track")
        self.ref_table = - np.ones_like(track_img, dtype=int)
        self.xs, self.ys = np.asarray(track_img == 0).nonzero()
        self.ref_table[self.xs, self.ys] = np.arange(self.kl.shape[0]) # this is a check that the length is correct
        print(f"Kernel Shape: {self.kl.shape}")

    def check_state_safe(self, state):
        mode = self.m.get_mode_id(state)
        if mode == -1:
            return False # the action is bad, so don't bother the kernel
            # should I have this check?????

        on_track, kernel_ind = check_on_track(state, self.ref_table, self.origin, self.resolution)
        if not on_track:
            return False
        
        phi = calculate_angle_ind(state[2], self.phi_range, self.kl.shape[1])

        if self.kl[kernel_ind, phi, mode] != 0:
            return False # unsfae state
        return True # safe state

    
class KernelListFilter:
    def __init__(self, conf, run):
        assert run.filter == True, "Incorrect filter mode"

        if run.racing_mode == "Fast":
            kernel_name = conf.kernel_path + f"Kernel_{run.racing_mode.lower()}_{conf.max_v}_{run.map_name}"
            kernel_conf = load_kernel_config(kernel_name)
            self.m = FastModesFilter(kernel_conf)
            self.check_state_safe = self.check_state_safe_fast
        elif run.racing_mode == "Slow":
            kernel_name = conf.kernel_path + f"Kernel_{run.racing_mode.lower()}_2_{run.map_name}"
            kernel_conf = load_kernel_config(kernel_name)
            self.check_state_safe = self.check_state_safe_slow
        else:
            raise ValueError(f"Unknown racing mode: {run.racing_mode}")

        self.phi_range = kernel_conf.phi_range
        self.resolution = kernel_conf.n_dx

        file_name =  f'maps/' + run.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])

        self.kl = np.load(kernel_name + "_kl_filter.npy")
        track_img = np.load(conf.kernel_path + "track_img_" + run.map_name + ".npy")
        self.ref_table = - np.ones_like(track_img, dtype=int)
        self.xs, self.ys = np.asarray(track_img == 0).nonzero()
        self.ref_table[self.xs, self.ys] = np.arange(self.kl.shape[0]) # this is a check that the length is correct
        print(f"Kernel Shape: {self.kl.shape}")

    def check_state_safe_slow(self, state):
        on_track, kernel_ind = check_on_track(state, self.ref_table, self.origin, self.resolution)
        if not on_track:
            return False
        
        phi = calculate_angle_ind(state[2], self.phi_range, self.kl.shape[1])

        if self.kl[kernel_ind, phi, 0] != 0:
            return False # unsfae state
        return True # safe state

    def check_state_safe_fast(self, state):
        speed_m  = self.m.get_mode_id(state)

        on_track, kernel_ind = check_on_track(state, self.ref_table, self.origin, self.resolution)
        if not on_track:
            return False
        
        phi = calculate_angle_ind(state[2], self.phi_range, self.kl.shape[1])

        if self.kl[kernel_ind, phi, speed_m] != 0:
            return False # unsfae state
        return True # safe state

    


@njit(cache=True) 
def check_on_track(state, rt, origin, resolution):
        x_ind = min(max(0, int(round((state[0]-origin[0])*resolution))), rt.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-origin[1])*resolution))), rt.shape[1]-1)

        kernel_ind = rt[x_ind, y_ind]
        if kernel_ind == -1:
            return False, -1
        return True, kernel_ind


@njit(cache=True) 
def calculate_angle_ind(phi, phi_range, phi_len):
        if phi >= phi_range/2:
            phi = phi - phi_range
        elif phi < -phi_range/2:
            phi = phi + phi_range
        phi_ind = int(round((phi + phi_range/2) / phi_range * (phi_len-1)))

        return phi_ind


