import numpy as np
import yaml
from matplotlib import pyplot as plt
from SafeRaceLearning.Utils.utils import *

def KernelAblationGraph():
    p = "Data/Vehicles/KernelAblation/"
    sets = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14]
    # sets = [2, 4, 6, 8, 10, 12, 14]
    # map_name = "f1_aut"
    map_name = "f1_aut_wide"
    mode = "Fast"
    agents_pp = [f"{mode}_PP_Super_{map_name}_2_{i}_0" for i in sets]
    agents_rand = [f"{mode}_Rando_Super_{map_name}_2_{i}_0" for i in sets]
    # mode = "Slow"
    # agents_pp_s = [f"{mode}_PP_Super_{map_name}_2_{i}_0" for i in sets]
    # agents_rand_s = [f"{mode}_Rando_Super_{map_name}_2_{i}_0" for i in sets]
    # agent_lists = [agents_pp, agents_rand, agents_pp_s, agents_rand_s]
    agent_lists = [agents_pp, agents_rand]
    # labels = ["PP8", "Random8", "PP2", "Random2"]
    labels = ["PP", "Random"]

    data = [[] for i in agent_lists]
    for j, agents in enumerate(agent_lists):
        for i in range(len(agents)):
            with open(p + agents[i] + "/" + agents[i] + "_record.yaml") as file:
                file_data = yaml.load(file, Loader=yaml.FullLoader)
                data[j].append(file_data["success_rate"])


    plt.figure(1, figsize=(5, 2.5))
    # xs = np.arange(2, 16, 2) / 10
    xs = np.array(sets) /10
    for i in range(len(agent_lists)):
        plt.plot(xs, data[i], 'o-', color=pp[i], label=labels[i])

    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel("Standard Deviation (m)")
    plt.ylabel("Success Rate (%)")

    plt.savefig(p + f"KernelAblationGraph_{map_name}.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(p + f"KernelAblationGraph_{map_name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


KernelAblationGraph()