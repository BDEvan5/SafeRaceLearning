from struct import Struct
import numpy as np
from matplotlib import pyplot as plt
from SafeRaceLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from SafeRaceLearning.Supervisor.Dynamics import run_dynamics_update


class SlowModes:
    def __init__(self, conf) -> None:
        assert conf.kernel_mode == "slow", "Incorrect kernel mode"
        ds = np.linspace(-conf.max_steer, conf.max_steer, conf.nq_steer)
        vs = np.ones_like(ds) * conf.max_v

        self.qs = np.stack((ds, vs), axis=1)
        self.n_modes = conf.nq_steer

    def get_mode_id(self, state):
        id = my_argmin(np.abs(self.qs[:, 0] - state[4]))
        return id

    def action2mode(self, action):
        id = self.get_mode_id(action[0])
        return self.qs[id]

    def __len__(self): return self.n_modes

    def check_action_valid(self, action):
        return True # all actions valid


class FastModes:
    def __init__(self, conf) -> None:
        assert conf.kernel_mode == "fast", "Incorrect kernel mode"

        self.vs = np.linspace(conf.min_v, conf.max_v, conf.nq_speed)
        self.ds = np.zeros((conf.nq_speed, conf.nq_steer))
        self.qs = np.zeros((conf.nq_steer* conf.nq_speed, 2))
        for i in range(conf.nq_speed):
            d_lim = calculate_max_steering(self.vs[i]) - 0.01
            self.ds[i] = np.linspace(-d_lim, d_lim, conf.nq_steer)
            for j in range(conf.nq_steer):
                self.qs[i*conf.nq_steer + j] = np.array([self.ds[i, j], self.vs[i]])

        self.n_modes = len(self.qs)
        self.ts = [[] for _ in range(self.n_modes)]
        self.mode_table = - np.ones((self.n_modes, self.n_modes), dtype=int) # init all values to -1
        self.dyns = np.zeros((self.n_modes, self.n_modes, 4), dtype=float) # init all values to -1
        self.time_step = conf.kernel_time_step

        self.build_mode_transition_table()
        # self.print_modes()

    def build_mode_transition_table(self): 
        for i in range(self.n_modes):
            m = self.qs[i]
            for j in range(self.n_modes):
                state = np.array([0, 0, 0, m[1], m[0]])
                n_state = run_dynamics_update(state, self.qs[j], self.time_step)
                
                if check_state_friction_valid(n_state):
                    self.dyns[i, j, 0:3] = n_state[0:3]
                    nm = calculate_mode(n_state, self.vs, self.ds)
                    self.mode_table[i, j], self.dyns[i, j, 3] = nm, nm
                    if nm not in self.ts[i]: self.ts[i].append(nm)

    def get_mode_id(self, state):
        if not check_state_friction_valid(state):
            return -1 # checks for valid mode
        m = calculate_mode(state, self.vs, self.ds)
        return m

    def check_action_valid(self, action):
        # checks if an action is inside the friction limit
        return check_action_friction_valid(action)

    def __len__(self): return self.n_modes

    def print_modes(self): 
        for i in range(self.qs.shape[0]):
            print(f"Mode {i}: {self.qs[i]}")


class FastModesFilter:
    def __init__(self, conf) -> None:
        assert conf.kernel_mode == "fast", "Incorrect kernel mode"

        self.vs = np.linspace(conf.min_v, conf.max_v, conf.nq_speed)
        self.n_modes = len(self.vs)

    def get_mode_id(self, state):
        m = calculate_speed_mode(state, self.vs)
        return m

    def __len__(self): return self.n_modes



@njit(cache=True)
def calculate_speed_mode(state, vs):
    v_diff = np.abs(vs - state[3])
    vi = my_argmin(v_diff)

    return vi

@njit(cache=True)
def calculate_mode(state, vs, ds):
    # assume that action is valid already
    v_diff = np.abs(vs - state[3])
    vi = my_argmin(v_diff)
    d_diff = np.abs(ds[vi] - state[4])
    di = my_argmin(d_diff)

    mode = vi * len(ds[0]) + di
    return int(mode)

@njit(cache=True)
def my_argmin(data):
    d_min, di = 1000000000, None
    for i in range(len(data)):
        if data[i] < d_min:
            d_min = data[i]
            di = i
    assert d_min != 1000000000, "Error in my_argmin function"
    return int(di)


def get_velcoity(delta, v_max, f_s):
    b = 0.523
    g = 9.81
    L = 0.329

    delta = np.abs(delta)
    delta[delta<0.01] = 0.01
    vs = f_s * np.sqrt(b*g*L/np.tan(abs(delta)))
    vs = np.clip(vs, 0, v_max)
    vs[0] = 0
    vs[-1] = 0

    return vs

def calculate_max_steering(v):
    b = 0.523
    g = 9.81
    L = 0.329

    d = np.arctan(L*b*g/(v**2))
    # note always positive d return

    return d

def check_state_friction_valid(state):
    v_max = calculate_speed(state[4], 1, 8)
    if state[3] > v_max:
        return False
    return True

def check_action_friction_valid(action):
    v_max = calculate_speed(action[0], 1, 8)
    if action[1] > v_max:
        return False
    return True