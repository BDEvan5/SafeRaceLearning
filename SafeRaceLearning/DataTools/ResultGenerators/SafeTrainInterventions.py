import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, MaxNLocator
from SafeRaceLearning.DataTools.plotting_utils import *






def compare_map_interventions():
    p = "Data/Vehicles/Safe_TrainMaps/"

    map_name = "f1_esp"
    reward = "Velocity"
    map_names = ["f1_aut", "f1_mco", "f1_esp"]
    map_names_print = ["AUT", "MCO", "ESP"]
    steps_list = [[] for _ in range(len(map_names))]
    intervention_list = [[] for _ in range(len(map_names))]

    set_n = 1
    repeats = 5

    for r in range(len(map_names)):
        for i in range(repeats):
            path = p + f"fast_Online_Std_{reward}_{map_names[r]}_6_{set_n}_{i}/"

            states, actions, safety_data = load_all_training_data(path)

            interventions = safety_data[:, 4]
            n= 50
            plot_len = 10000 / n
            # n= 100
            i = 0 
            inter_ns = []
            while i < len(interventions)-1:
                inter_ns.append(0)
                for _ in range(n):
                    if interventions[i] > 0.001:
                        inter_ns[-1] += 1
                    i += 1
                    if i >= len(interventions)-1:
                        break

            # steps_list[r].append(steps)
            intervention_list[r].append(inter_ns)
            while len(intervention_list[r][-1]) < plot_len:
                intervention_list[r][-1].append(0)

    # plt.figure(1, figsize=(6, 1.8))
    plt.figure(1, figsize=(4, 1.8))
    plt.clf()

    # fig, axs = plt.subplots(3, 1, figsize=(6, 3.87), sharex=True)

    xs = np.arange(0, plot_len, 1)
    dark_pp = ["#186A3B", "#7D6608", "#4A235A"]
    # for r in range(len(map_names)):
    #     for j in range(repeats):
    #         plt.plot(xs, intervention_list[r][j], '-', color=pp[r+2], alpha=0.6)
    for r in range(len(map_names)):
        min_line, max_line, avg_line = convert_to_min_max_avg(xs, intervention_list[r], xs)
        plt.fill_between(xs, min_line, max_line, alpha=0.4, color=pp[r+2])
    for r in range(len(map_names)-1, -1, -1):
        min_line, max_line, avg_line = convert_to_min_max_avg(xs, intervention_list[r], xs)
        plt.plot(xs, avg_line, '-', label=map_names_print[r], color=dark_pp[r], linewidth=2)
        # plt.plot(xs, avg_line, '-', label=map_names_print[r], color=pp[r+2], linewidth=2)
    # plt.legend(loc='upper left')
    # plt.legend(loc='upper left', ncol=3)
    plt.gcf().legend(loc='center', bbox_to_anchor=(0.5,1), ncol=3)
    plt.grid(True)
    plt.ylim(-2, 65)


    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Interventions")
    plt.xlim(0, 50)
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5, integer=True))
    plt.tight_layout()

    name = p + f"Safe_TrainMaps_Interventions"
    std_img_saving(name)


compare_map_interventions()

