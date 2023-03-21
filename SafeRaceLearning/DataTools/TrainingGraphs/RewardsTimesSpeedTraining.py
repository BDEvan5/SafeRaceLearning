import matplotlib.pyplot as plt
import numpy as np 
from SafeRaceLearning.DataTools.plotting_utils import *




def fast_onlineT_speed_training():
    p = "Data/Vehicles/Safe_TrainSpeeds/"

    map_name = "f1_esp"
    speeds = range(3, 7)
    reward_name = "Velocity"
    steps_list = [[] for _ in range(len(speeds))]
    length_list = [[] for _ in range(len(speeds))]
    reward_list = [[] for _ in range(len(speeds))]

    set_n = 1
    repeats = 3


    for r in range(len(speeds)):
        for i in range(repeats):
            path = p + f"fast_Online_Std_{reward_name}_{map_name}_{speeds[r]}_{set_n}_{i}/"

            rewards, lengths, progresses, laptimes = load_csv_training_data(path)
            steps = np.cumsum(lengths) / 100

            steps_list[r].append(steps)
            length_list[r].append(lengths/10)
            reward_list[r].append(rewards)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.3, 2.90))

    xs = np.arange(0, 100, 1)
    for r in range(len(speeds)):
        min_line, max_line, avg_line = convert_to_min_max_avg_multi_step(steps_list[r], length_list[r], xs)
        ax1.plot(xs, avg_line, '-', label=f"{speeds[r]} m/s", color=pp[r], linewidth=2)

        min_line, max_line, avg_line = convert_to_min_max_avg_multi_step(steps_list[r], reward_list[r], xs)
        ax2.plot(xs, avg_line, '-', label=f"{speeds[r]} m/s", color=pp[r], linewidth=2)

    ax1.get_yaxis().set_major_locator(plt.MultipleLocator(10))
    ax2.set_xlabel("Training Steps (x100)")
    ax2.set_ylabel("Reward per Lap")
    ax1.set_ylabel("Lap time (s)")
    ax2.get_yaxis().set_major_locator(plt.MultipleLocator(200))
    # ax1.legend(loc='center right', ncol=1, bbox_to_anchor=(1.22, 0.5))
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=4, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    ax1.grid(True)
    ax2.grid(True)

    name = p + f"Safe_TrainSpeeds_RewardsTimes" 
    std_img_saving(name)



def fast_onlineT_maps_training():
    p = "Data/Vehicles/Safe_TrainMaps/"

    map_names = ["f1_aut", "f1_mco", "f1_esp", "f1_gbr"]
    print_map_names = ["AUT", "MCO", "ESP", "GBR"]
    speed = 5
    # map_name = "f1_esp"
    # speeds = range(3, 7)
    reward_name = "Velocity"
    steps_list = [[] for _ in range(len(map_names))]
    length_list = [[] for _ in range(len(map_names))]
    reward_list = [[] for _ in range(len(map_names))]

    set_n = 1
    repeats = 4


    for r in range(len(map_names)):
        for i in range(repeats):
            path = p + f"fast_Online_Std_{reward_name}_{map_names[r]}_{speed}_{set_n}_{i}/"

            rewards, lengths, progresses, laptimes = load_csv_training_data(path)
            steps = np.cumsum(lengths) / 100

            steps_list[r].append(steps)
            length_list[r].append(lengths/10)
            reward_list[r].append(rewards)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(8.2, 2.2))
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.3, 2.70))

    xs = np.arange(0, 100, 1)
    for r in range(len(map_names)):
        min_line, max_line, avg_line = convert_to_min_max_avg_multi_step(steps_list[r], length_list[r], xs)
        ax1.plot(xs, avg_line, '-', label=f"{print_map_names[r]}", color=pp[r], linewidth=2)
        ax1.fill_between(xs, min_line, max_line, color=pp[r], alpha=0.3)

        min_line, max_line, avg_line = convert_to_min_max_avg_multi_step(steps_list[r], reward_list[r], xs)
        ax2.plot(xs, avg_line, '-', label=f"{print_map_names[r]}", color=pp[r], linewidth=2)
        ax2.fill_between(xs, min_line, max_line, color=pp[r], alpha=0.3)

    ax1.get_yaxis().set_major_locator(plt.MultipleLocator(15))
    ax1.set_xlabel("Training Steps (x100)")
    ax2.set_xlabel("Training Steps (x100)")
    ax2.set_title("Reward per Lap")
    ax1.set_title("Lap time (s)")
    ax2.get_yaxis().set_major_locator(plt.MultipleLocator(150))
    # ax1.legend(loc='center right', ncol=1, bbox_to_anchor=(1.22, 0.5))
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=4, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    ax1.grid(True)
    ax2.grid(True)

    name = p + f"Safe_TrainMaps_RewardsTimes" 
    std_img_saving(name)

# fast_onlineT_speed_training()
fast_onlineT_maps_training()


