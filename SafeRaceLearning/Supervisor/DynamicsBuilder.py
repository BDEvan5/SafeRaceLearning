
from SafeRaceLearning.Utils.utils import limit_phi, load_conf

from SafeRaceLearning.Supervisor.Dynamics import run_dynamics_update

import numpy as np
from matplotlib import pyplot as plt

import numpy as np
from numba import njit

from SafeRaceLearning.Supervisor.Modes import *


def generate_dynamics_entry(state, action, m, time, resolution, phis):
    dyns = np.zeros(4)
    new_state = run_dynamics_update(state, action, time)
    dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
    new_q = m.get_mode_id(new_state)

    phi = limit_phi(phi)
    new_k = int(round((phi + np.pi) / (2*np.pi) * (len(phis)-1)))
    dyns[2] = min(max(0, new_k), len(phis)-1)
    dyns[0] = int(round(dx * resolution))                  
    dyns[1] = int(round(dy * resolution))                  
    dyns[3] = int(new_q)       

    return dyns           

def generate_fast_ns(state, action, time, resolution, phis):
    dyns = np.zeros(3)
    new_state = run_dynamics_update(state, action, time)
    dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]

    phi = limit_phi(phi)
    new_k = int(round((phi + np.pi) / (2*np.pi) * (len(phis)-1)))
    dyns[2] = min(max(0, new_k), len(phis)-1)
    dyns[0] = int(round(dx * resolution))                  
    dyns[1] = int(round(dy * resolution))                  

    return dyns           


# @njit(cache=True)
def build_slow_dynamics(state_m, act_m, conf):
    phis = np.linspace(-np.pi, np.pi, conf.n_phi)

    ns = conf.n_intermediate_pts
    dt = conf.kernel_time_step / ns

    dynamics = np.zeros((len(phis), len(state_m), len(act_m), ns, 4), dtype=int)
    invalid_counter = 0
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(state_m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(act_m.qs): # searches through actions

                for l in range(ns):
                    dynamics[i, j, k, l] = generate_dynamics_entry(state.copy(), action, state_m, dt*(l+1), conf.n_dx, phis)                                
                
    print(f"Slow Dynamics Table has been built: {dynamics.shape}")

    return dynamics


def build_fast_dynamics(state_m, act_m, conf):
    phis = np.linspace(-np.pi, np.pi, conf.n_phi)

    ns = conf.n_intermediate_pts
    dt = conf.kernel_time_step / ns

    dynamics = np.zeros((len(phis), len(state_m), len(act_m), ns, 4), dtype=int)
    invalid_counter = 0
    valid_counter = 0
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(state_m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(act_m.qs): # searches through actions
                if state_m.mode_table[j, k] == -1: 
                    dynamics[i, j, k] = - np.ones_like(dynamics[i, j, k])
                    invalid_counter += 1
                    continue
                valid_counter += 1
                for l in range(ns):
                    e = generate_fast_ns(state.copy(), action, dt*(l+1), conf.n_dx, phis)
                    dynamics[i, j, k, l, 0:3] = generate_fast_ns(state.copy(), action, dt*(l+1), conf.n_dx, phis)           
                    dynamics[i, j, k, l, 3] = state_m.mode_table[j, k]
                    # print(f": {dynamics[i, j, k, l, :]}")                 

    print(f"Invalid transitions: {invalid_counter}")       
    print(f"Valid transitions: {valid_counter}")
    print(f"Dynamics Table has been built: {dynamics.shape}")

    return dynamics


def build_dynamics_table(conf):
    if conf.kernel_mode == "slow":
        state_m = SlowModes(conf)
        act_m = SlowModes(conf)
        dynamics = build_slow_dynamics(state_m, act_m, conf)
    elif conf.kernel_mode == "fast":
        state_m = FastModes(conf)
        act_m = FastModes(conf)
        dynamics = build_fast_dynamics(state_m, act_m, conf)
    else:
        raise ValueError(f"Unknown kernel mode: {conf.kernel_mode}")

    filename = f"{conf.dynamics_path}{conf.kernel_mode}_{conf.max_v}_{int(conf.kernel_time_step*10)}_dyns.npy"
    np.save(filename, dynamics)
    print(f"Dynamics table saved to {filename}")



if __name__ == "__main__":
    conf = load_conf("kernel_config")

    build_dynamics_table(conf)

