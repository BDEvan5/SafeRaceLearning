import numpy as np
from matplotlib import pyplot as plt
from SafeRaceLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from SafeRaceLearning.Supervisor.Dynamics import run_dynamics_update



class FastModesPlot:
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
        self.mode_table = - np.ones((self.n_modes, self.n_modes), dtype=int) # init all values to -1

        self.build_mode_transition_list()

    def build_mode_transition_list(self): 
        for i in range(self.n_modes):
            m = self.qs[i]
            for j in range(self.n_modes):
                state = np.array([0, 0, 0, m[1], m[0]])
                n_state = run_dynamics_update(state, self.qs[j], 0.2)
                if check_state_friction_valid(n_state):
                    self.mode_table[i, j] = calculate_mode(n_state, self.vs, self.ds)
            
            # q_print = np.array2string(self.qs[i], sign=' ', precision=2, separator=', ', floatmode='fixed')
            # print(f"{i} {q_print}: {mode_table[i]}")

    def plot_mode_transitions(self):
        for i in range(self.n_modes):
            self.plot_friction_line()
            plt.title(f"Mode {i}: {self.qs[i]}")
            for j in range(self.n_modes):
                if j == i:
                    plt.plot(self.qs[j][0], self.qs[j][1], color='orange', marker='*', markersize=25)

                n_mode = self.mode_table[i, j]
                if n_mode != -1:
                    plt.plot(self.qs[n_mode][0], self.qs[n_mode][1], color='green', marker='o')
            
            plt.tight_layout()
            plt.show()

    def print_modes(self): 
        for i in range(self.qs.shape[0]):
            print(f"Mode {i}: {self.qs[i]}")

    def plot_friction_line(self):
        delta = np.linspace(-0.41, 0.41, 1000)
        vels = get_velcoity(delta, 6, 1)

        plt.figure(1, figsize=(4, 2))
        # plt.figure(1, figsize=(8, 5))
        plt.clf()

        # color = 'mediumorchid'
        color = '#8E44AD'
        plt.plot(delta, vels, linewidth=2, color=color)
        plt.fill_between(delta, vels, 0, color=color, alpha=0.35)
        plt.plot([0,0], [0, 9], color='black', linewidth=2)
        plt.plot([-0.5,0.5], [0, 0], color='black', linewidth=2)

        plt.xlabel("Steering Angle (rad)")
        plt.ylabel("Speed (m/s)")
        plt.gca().get_yaxis().set_major_locator(MultipleLocator(2))
        plt.gca().get_xaxis().set_major_locator(MultipleLocator(0.1))

        plt.ylim([0, 6.4])
        plt.xlim([-0.44, 0.44])
        plt.grid(True)

    def plot_modes(self):
        self.plot_friction_line()

        for m in self.qs:
            plt.plot(m[0], m[1], 'o', color='red')

        plt.show()

    def plot_mode_picture(self):


        for i in range(self.n_modes):
        # for i in [12]:
            self.plot_friction_line()
            plotted = []
            for j in range(self.n_modes):
                n_mode = self.mode_table[i, j]
                if j == i:
                    plt.plot(self.qs[j][0], self.qs[j][1], color='#3498DB', marker='X', markersize=10)
                    plotted.append(n_mode)
                elif n_mode != -1:
                    plt.plot(self.qs[n_mode][0], self.qs[n_mode][1], color='#58D68D', marker='o', markersize=6)
                    plotted.append(n_mode)

            for j, m in enumerate(self.qs):
                if not j in plotted:
                    plt.plot(m[0], m[1], 'o', color="#E74C3C", markersize=6)

            plt.tight_layout()

            # plt.savefig(f"Data/KernelImgs/ModeDiagramFone_{i}.svg", bbox_inches='tight', pad_inches=0)
            # if i == 12:
            # plt.savefig(f"Data/KernelImgs/ModeDiagramFone_{i}.pdf", bbox_inches='tight', pad_inches=0)

            plt.show()


    def get_mode_id(self, state):
        m = calculate_mode(state, self.vs, self.ds)
        return m

    def __len__(self): return self.n_modes


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



if __name__ == "__main__":
    kernel_conf = load_conf("kernel_config_fast")
    fm = FastModesPlot(kernel_conf)
    fm.print_modes()
    fm.plot_mode_picture()
    # fm.plot_transitions()
    # fm.plot_mode_transitions()

    # fm = FastModes(kernel_conf)

