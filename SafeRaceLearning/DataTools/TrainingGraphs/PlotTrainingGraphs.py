import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import csv
import glob, os 
from matplotlib.ticker import MultipleLocator, PercentFormatter

from SafeRaceLearning.Utils.utils import *


def load_csv_data(path):
    """loads data from a csv training file

    Args:   
        path (file_path): path to the agent

    Returns:
        rewards: ndarray of rewards
        lengths: ndarray of episode lengths
        progresses: ndarray of track progresses
        laptimes: ndarray of laptimes
    """
    rewards, lengths, progresses, laptimes = [], [], [], []
    with open(f"{path}training_data_episodes.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[2]) > 0:
                rewards.append(float(row[1]))
                lengths.append(float(row[2]))
                progresses.append(float(row[3]))
                laptimes.append(float(row[4]))

    rewards = np.array(rewards)[:-1]
    lengths = np.array(lengths)[:-1]
    progresses = np.array(progresses)[:-1]
    laptimes = np.array(laptimes)[:-1]
    
    return rewards, lengths, progresses, laptimes

def plot_lap_times(path):
    name = path.split("/")[-2]
    rewards, lengths, progresses, laptimes = load_csv_data(path)
    steps = np.cumsum(lengths) / 100

    laptimes_success = laptimes[progresses>0.98]
    avg_lap_times = true_moving_average(laptimes_success, 20)
    steps_success = steps[progresses>0.98]

    laptimes_crash = laptimes[progresses<0.98]
    steps_crash = steps[progresses<0.98]

    plt.figure(1, figsize=(3.2, 2))
    plt.clf()

    plt.plot(steps_success, laptimes_success, '.', color='darkblue', markersize=4)
    plt.plot(steps_success, avg_lap_times, '-', color='red')
    plt.plot(steps_crash, laptimes_crash, '.', color='green', markersize=4)
    # plt.plot(steps_success, laptimes_success, '-')

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Laptimes (s)")
    plt.tight_layout()
    plt.grid()
    plt.ylim(0, 60)

    plt.savefig("Data/HighSpeedEval/" + f"Laptimes_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_reward_steps(path):
    rewards, lengths, progresses, _ = load_csv_data(path)
    steps = np.cumsum(lengths) / 100

    rewards_success = rewards[progresses>0.98]
    steps_success = steps[progresses>0.98]

    rewards_crash = rewards[progresses<0.98]
    steps_crash = steps[progresses<0.98]

    plt.figure(1, figsize=(3.2, 2))
    # plt.figure(1, figsize=(6, 2.5))
    plt.plot(steps_success, rewards_success, '.', color='darkblue', markersize=3, alpha=0.8)
    plt.plot(steps_crash, rewards_crash, '.', color='green', markersize=3, alpha=0.8)
    ys = true_moving_average(rewards, 50)
    xs = np.linspace(steps[0], steps[-1], 200)
    ys = np.interp(xs, steps, ys)
    plt.plot(xs, ys, linewidth=3, color='r')
    # plt.gca().get_xaxis().set_major_locator(MultipleLocator(250))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Reward per Ep.")

    plt.tight_layout()
    plt.grid()

    name = path.split("/")[-2]
    # plt.savefig("Data/HighSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig("Data/RacingEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig("Data/LowSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_progress_steps(path):
    rewards, lengths, progresses, _ = load_csv_data(path)
    steps = np.cumsum(lengths) / 100

    progress_success = progresses[progresses>0.98]
    steps_success = steps[progresses>0.98]

    progress_crash = progresses[progresses<0.98]
    steps_crash = steps[progresses<0.98]

    plt.figure(1, figsize=(3.2, 2))
    # plt.figure(1, figsize=(6, 2.5))
    plt.plot(steps_success, progress_success*100, '.', color='darkblue', markersize=3, alpha=0.8)
    plt.plot(steps_crash, progress_crash*100, '.', color='green', markersize=3, alpha=0.8)
    ys = true_moving_average(progresses*100, 50)
    xs = np.linspace(steps[0], steps[-1], 200)
    ys = np.interp(xs, steps, ys)
    plt.plot(xs, ys, linewidth=3, color='r')
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Progress %")
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.grid()

    name = path.split("/")[-2]
    # plt.savefig("Data/HighSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig("Data/RacingEval/" + f"progress_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig("Data/LowSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_progress_steps_list(path_list):
    plt.figure(1, figsize=(3.5, 2))
    # plt.figure(1, figsize=(6, 2.5))
    colors = ['red', 'blue', 'green']
    for i, path in enumerate(path_list):
        rewards, lengths, progresses, _ = load_csv_data(path)
        steps = np.cumsum(lengths) / 100

        ys = true_moving_average(progresses*100, 50)
        xs = np.linspace(steps[0], steps[-1], 200)
        ys = np.interp(xs, steps, ys)
        plt.plot(xs, ys, linewidth=2, color=colors[i])
        plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Progress %")
        plt.ylim(0, 105)

        plt.tight_layout()
        plt.grid(True)

    # name = path.split("/")[-2]
    map_name = "MCO"
    # plt.savefig("Data/HighSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig("Data/RacingEval/" + f"progress_steps_comparision_{map_name}.pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig("Data/LowSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()



# Slow tests
def slow_progress_training_comparision():
    # p = "Data/Vehicles/SlowTests/"
    p = "Data/Vehicles/Eval_RewardsSlow/"

    map_names = ["f1_esp", "f1_gbr", "f1_aut", "f1_mco"]
    repeats = 5
    moving_avg = 2
    reward_list = ["Cth", "Progress", "Std"]
    xs = np.arange(300)
        
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 4))
    axs = axs.reshape(-1)
        
    for ax, map_name in enumerate(map_names):
    
        step_list = [[] for _ in range(len(reward_list))]
        progresses_list = [[] for _ in range(len(reward_list))]
            
        for i in range(repeats):
            for j in range(len(reward_list)):
                path = p + f"Slow_Std_Std_{reward_list[j]}_{map_name}_1_{i}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                
                
                steps = np.cumsum(lengths)/100
                progresses = true_moving_average(progresses, moving_avg)*100
                pr = np.interp(xs, steps, progresses)
                if np.mean(pr) > 35:
                    # print(np.mean(pr))
                    step_list[j].append(steps)
                    progresses_list[j].append(progresses)
                else:
                    print("Problem")
                    print(progresses)
                    print(path)


        for j in [2, 1, 0]:
            mins, maxes, means = convert_to_min_max_avg(step_list[j], progresses_list[j], xs)
            axs[ax].plot(xs, means, '-', color=pp[j], linewidth=2, label=reward_list[j])
            # for z in range(len(progresses_list[j])):
                # if np.mean(progresses_list[j][z]) > 0.1:
                # plt.plot(step_list[j][z], progresses_list[j][z], '-', color=pp[j], linewidth=1, alpha=0.4)
                # plt.plot(step_list[j][z], progresses_list[j][z], '-', color=pp[j], linewidth=1)
                
            if np.mean(mins) > 0.1:
                axs[ax].fill_between(xs, mins, maxes, color=pp[j], alpha=0.3)
            
            axs[ax].grid(True)
            axs[ax].text(220, 10, map_name.split('_')[1].upper(),  fontsize=12)
            # plt.show()
            # axs[ax].get_yaxis().set_major_locator(MultipleLocator(10))

    axs[2].set_xlabel("Training Steps (x100)")
    axs[3].set_xlabel("Training Steps (x100)")
    axs[0].set_ylabel("Avg. Progress %")
    axs[2].set_ylabel("Avg. Progress %")
    axs[0].legend(loc='center', ncol=3, bbox_to_anchor=(1.1, 1.1))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)
    # plt.tight_layout()

    plt.savefig(p + f"slow_training_reward_all.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(p + f"slow_training_reward_all.svg", bbox_inches='tight', pad_inches=0)



def compare_ep_vs_step_rewards():
    n = 1
    name = "Agent_Progress_f1_esp_1_0"
    path = f"/home/benjy/Documents/AutonomousRacing/RacingRewards/Data/Vehicles/SlowTests/{name}/"
    save_path = "Data/LowSpeedEval/"
    rewards, lengths, progresses, _ = load_csv_data(path)

    plt.figure(1, figsize=(3.5,1.8))
    plt.clf()
    steps = np.cumsum(lengths) / 100
    plt.plot(steps, true_moving_average(rewards, 25), linewidth=1.5, color='darkgreen')
    plt.plot(steps, rewards, '.', color='darkblue', markersize=6)
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Ep. Rewards")

    plt.tight_layout()
    plt.grid()

    plt.savefig(save_path + f"step_rewards_{name}.pdf", bbox_inches='tight')

    plt.pause(0.01)

    plt.figure(2, figsize=(3.5,1.8))
    plt.clf()
    steps = np.cumsum(lengths) / 100

    plt.plot(true_moving_average(rewards, 25), linewidth=1.5, color='darkgreen')
    plt.plot(rewards, '.', color='darkblue', markersize=6)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Episode Number")
    plt.ylabel("Ep. Rewards")

    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path + f"ep_rewards_{name}.pdf", bbox_inches='tight')

    plt.show()

# Fast tests
def fast_reward_comparision():
    p = "Data/Vehicles/FastRewardTests/"

    progress_steps = []
    progress_progresses = []
    cth_steps = []
    cth_progresses = []

    set_n =1
    repeats = 5
    # speed = 5
    speed = 8
    # map_name = "f1_mco"
    map_name = "f1_esp"
# 
    for i in range(repeats):
        # path_progress = p + f"Fast_Std_Std_Progress_{map_name}_{set_n}_{i}/"
        # path_cth = p + f"Fast_Std_Std_Cth_{map_name}_{set_n}_{i}/"
        path_progress = p + f"Fast_Std_Std_Progress_{map_name}_{speed}_{set_n}_{i}/"
        path_cth = p + f"Fast_Std_Std_Cth_{map_name}_{speed}_{set_n}_{i}/"

        rewards_progress, lengths_progress, progresses_progress, _ = load_csv_data(path_progress)
        rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

        steps_progress = np.cumsum(lengths_progress) / 100
        avg_progress_progress = true_moving_average(progresses_progress, 20)* 100
        steps_cth = np.cumsum(lengths_cth) / 100
        avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

        progress_steps.append(steps_progress)
        progress_progresses.append(avg_progress_progress)
        cth_steps.append(steps_cth)
        cth_progresses.append(avg_progress_cth)


    plt.figure(1, figsize=(3.5, 2.0))
    # plt.figure(1, figsize=(6, 2.5))
    plt.clf()

    xs = np.linspace(0, 1000, 300)
    min_progress, max_progress, mean_progress = convert_to_min_max_avg(progress_steps, progress_progresses, xs)
    min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, xs)

    plt.plot(xs, mean_progress, '-', color=pp[0], linewidth=2, label='Progress')
    plt.gca().fill_between(xs, min_progress, max_progress, color='red', alpha=0.2)
    plt.plot(xs, mean_cth, '-', color=pp[2], linewidth=2, label='Cth')
    plt.gca().fill_between(xs, min_cth, max_cth, color='green', alpha=0.2)
    
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.legend(loc='lower right', ncol=2)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)
    plt.tight_layout()
    plt.grid()

    plt.savefig(p + f"fast_rewards_{map_name}_{speed}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(p + f"fast_rewards_{map_name}_{speed}.svg", bbox_inches='tight', pad_inches=0)

    plt.show()# Fast tests
    
# def fast_reward_comparision():
#     p = "Data/Vehicles/FastRewardTests/"

#     progress_steps = []
#     progress_progresses = []
#     cth_steps = []
#     cth_progresses = []

#     set_n =1
#     repeats = 5
#     speed = 5
#     # speed = 8
#     map_name = "f1_mco"
#     # map_name = "f1_esp"

#     for i in range(repeats):
#         path_progress = p + f"Fast_Std_Std_Progress_f1_esp_{speed}_{set_n}_{i}/"
#         path_cth = p + f"Fast_Std_Std_Cth_f1_esp_{speed}_{set_n}_{i}/"

#         rewards_progress, lengths_progress, progresses_progress, _ = load_csv_data(path_progress)
#         rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

#         steps_progress = np.cumsum(lengths_progress) / 100
#         avg_progress_progress = true_moving_average(progresses_progress, 20)* 100
#         steps_cth = np.cumsum(lengths_cth) / 100
#         avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

#         progress_steps.append(steps_progress)
#         progress_progresses.append(avg_progress_progress)
#         cth_steps.append(steps_cth)
#         cth_progresses.append(avg_progress_cth)


#     plt.figure(1, figsize=(3.5, 2.0))
#     # plt.figure(1, figsize=(6, 2.5))
#     plt.clf()

#     xs = np.linspace(0, 1000, 300)
#     min_progress, max_progress, mean_progress = convert_to_min_max_avg(progress_steps, progress_progresses, xs)
#     min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, xs)

#     plt.plot(xs, mean_progress, '-', color=pp[0], linewidth=2, label='Progress')
#     plt.gca().fill_between(xs, min_progress, max_progress, color='red', alpha=0.2)
#     plt.plot(xs, mean_cth, '-', color=pp[2], linewidth=2, label='Cth')
#     plt.gca().fill_between(xs, min_cth, max_cth, color='green', alpha=0.2)

#     plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))
#     plt.xlabel("Training Steps (x100)")
#     plt.ylabel("Track Progress %")
#     plt.legend(loc='lower right', ncol=2)
#     # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)
#     plt.tight_layout()
#     plt.grid()

#     plt.savefig(p + f"fast_rewards_{map_name}_{speed}.pdf", bbox_inches='tight', pad_inches=0)
#     plt.savefig(p + f"fast_rewards_{map_name}_{speed}.svg", bbox_inches='tight', pad_inches=0)

#     plt.show()

def fast_progress_training_comparision_maxspeed():
    p = "Data/Vehicles/MaxV/"
    # p = "Data/Vehicles/MaxSpeedTests/"

    steps_list = []
    progresses_list = []

    n_repeats = 10
    for i, v in enumerate(range(4, 9)): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"Fast_Cth_f1_esp_{v}_1_{j}/"
            # path = p + f"Agent_Cth_f1_gbr_{v}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 100
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    # plt.figure(2, figsize=(5, 2.5))
    plt.figure(2, figsize=(6.5, 2.5))

    colors = ['red', 'darkblue', 'green', 'orange', 'purple']
    labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    xs = np.linspace(0, 1000, 300)
    for i in range(len(steps_list)):
    # for i in range(len(steps_list)-1, -1, -1):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], len(xs))
        plt.plot(xs, mean, '-', color=colors[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=colors[i], alpha=0.2)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='lower right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1)
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/HighSpeedEval/" + "training_comparision_maxspeed.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig("Data/HighSpeedEval/" + "training_comparision_maxspeed.svg", bbox_inches='tight', pad_inches=0)

    # plt.show()
    # plt.pause(0.00001)

def plot_maps_graph():
    p = "Data/Vehicles/"
    # name = "CthMaps"
    # name = "PaperPPPS"
    name = "LinkEval"
    # p = "Data/Vehicles/Fast5T/"
    # p = "Data/Vehicles/CthVEval/"
    # p = "Data/Vehicles/PppsEval/"
    # p = "Data/Vehicles/MaxSpeedTests/"

    p += name + "/"

    # maps = ["aut", "esp", "mco"]
    maps = ["aut", "esp", "gbr", "mco"]
    # labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']
    # labels = ['Aut', 'Esp', 'Mco']
    labels = ['AUT', 'ESP', 'GBR', 'MCO']

    steps_list = []
    progresses_list = []

    n_repeats = 10
    for i in range(len(maps)): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            # path = p + f"Fast_Progress_f1_{maps[i]}_5_1_{j}/"
            # path = p + f"Fast_Cth_f1_{maps[i]}_8_1_{j}/"
            path = p + f"Link_Cth_f1_{maps[i]}_2_1_{j}/"
            # path = p + f"Fast_CthV_f1_{maps[i]}_8_5_{j}/"
            # path = p + f"Fast_Gps_f1_{maps[i]}_8_1_{j}/"
            # path = p + f"Agent_Cth_f1_gbr_{v}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 100
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(6, 2.2))
    # plt.figure(2, figsize=(6, 3))

    # colors = ['red', 'darkblue', 'green', 'orange', 'purple']


    xs = np.linspace(0, 1000, 300)
    for i in range(len(steps_list)-1, -1, -1):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], len(xs))
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.25)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='lower right')
    plt.legend(loc='lower right', ncol=len(steps_list))
    # plt.legend(loc='upper center', ncol=len(steps_list))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.tight_layout()
    plt.grid()

    # plt.savefig("Data/HighSpeedEval/" + "training_comparision_maxspeed.pdf", bbox_inches='tight', pad_inches=0)

    plt.savefig(p + f"training_{name}.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(p + f"training_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()
    plt.pause(0.00001)

def plot_v_reward_graph():
    p = "Data/Vehicles/"
    name = "RewardVelocity"
    

    p += name + "/"

    map_name = "esp"
    reward_names = ["v1", "v2", "v3", "Gps"]

    steps_list = []
    progresses_list = []

    n_repeats = 1
    set_n = 1
    for i in range(len(reward_names)): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"Fast_{reward_names[i]}_f1_{map_name}_8_{set_n}_{j}/"
            
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 100
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(6, 2.2))
    # plt.figure(2, figsize=(6, 3))

    reward_names[-1] = "PPPS"
    xs = np.linspace(0, 1000, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], len(xs))
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=reward_names[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.25)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='lower right')
    plt.legend(loc='center', ncol=1, bbox_to_anchor=(1.1, 0.5))
    # plt.legend(loc='lower right', ncol=len(steps_list))
    # plt.legend(loc='upper center', ncol=len(steps_list))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.tight_layout()
    plt.grid()

    # plt.savefig("Data/HighSpeedEval/" + "training_comparision_maxspeed.pdf", bbox_inches='tight', pad_inches=0)

    plt.savefig(p + f"training_{name}.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(p + f"training_{name}.pdf", bbox_inches='tight', pad_inches=0)

    # plt.show()
    # plt.pause(0.00001)

def fast_progress_training_ppps_maps():
    p = "Data/Vehicles/PaperPPPS/"
    # p = "Data/Vehicles/PolicySearchPP/"
    # p = "Data/Vehicles/MaxSpeedTests/"

    steps_list = []
    progresses_list = []

    n_repeats = 10
    maps = ["AUT", "ESP", "GBR", "MCO"]
    # map_names = 
    for i in range(4): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            map_name = "f1_" + maps[i].lower()
            path = p + f"Fast_Gps_{map_name}_8_1_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 100
            avg_progress = true_moving_average(progresses[:-1], 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    # plt.figure(2, figsize=(5, 2.5))
    plt.figure(2, figsize=(6, 2.5))

    colors = ['red', 'darkblue', 'green', 'orange', 'purple']
    # labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    # end_x = steps_list[0][-1]
    end_x = 1500
    xs = np.linspace(0, end_x, 300)
    for i in range(len(steps_list)-1, -1, -1):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], len(xs))
        plt.plot(xs, mean, '-', color=colors[i], linewidth=2, label=maps[i])
        plt.gca().fill_between(xs, min, max, color=colors[i], alpha=0.2)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(-5, 105)
    # plt.legend(loc='lower right')
    plt.legend(loc='lower right', ncol=len(steps_list))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/HighSpeedEval/" + "training_ppps_maps.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()
    plt.pause(0.00001)

def fast_repeatability():
    p = "Data/Vehicles/PaperPPPS/"
    # p = "Data/Vehicles/FastTests/"

    steps_list = []
    progresses_list = []
    
    map_name = "f1_mco"

    n_repeats = 10
    for i in range(n_repeats):
        path = p + f"Fast_Gps_{map_name}_8_1_{i}/"
        # path = p + f"Agent_Cth_f1_mco_5_{i}/"
        rewards, lengths, progresses, _ = load_csv_data(path)
        steps = np.cumsum(lengths) / 100
        avg_progress = true_moving_average(progresses, 20)* 100
        steps_list.append(steps)
        progresses_list.append(avg_progress)

    plt.figure(1, figsize=(6, 2.2))

    color = "#B7950B"
    # color = 'blue'
    xs = np.linspace(0, 1000, 300)
    for i in range(len(steps_list)):
        xs = steps_list[i]
        ys = true_moving_average(progresses_list[i], 50)
        plt.plot(xs, ys, '-', color=color, linewidth=2)
        # plt.gca().fill_between(xs, min, max, color=color, alpha=0.2)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='lower right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/HighSpeedEval/" + "ppps_mco_repeatability.pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig("Data/HighSpeedEval/" + "fast_repeatability.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def convert_to_min_max_avg(step_list, progress_list, xs):
    """Returns the 3 lines 
        - Minimum line
        - maximum line 
        - average line 
    """ 
    n = len(step_list)

    # xs = np.arange(length_xs)
    ys = np.zeros((n, len(xs)))
    # xs = np.linspace(0, x_lim, length_xs)
    # xs = np.linspace(step_list[0][0], step_list[0][-1], length_xs)
    for i in range(n):
        ys[i] = np.interp(xs, step_list[i], progress_list[i])

    min_line = np.min(ys, axis=0)
    max_line = np.max(ys, axis=0)
    avg_line = np.mean(ys, axis=0)

    return min_line, max_line, avg_line

def smooth_line(steps, progresses, length_xs=300):
    xs = np.linspace(steps[0], steps[-1], length_xs)
    smooth_line = np.interp(xs, steps, progresses)

    return xs, smooth_line

### -------------------------------------

def plot_ep_lengths(path):
    rewards, lengths, progresses, _ = load_csv_data(path)

    plt.figure(1, (4,2.5))
    plt.clf()
    steps = np.cumsum(lengths) / 100
    plt.plot(steps, progresses, '.', color='darkblue', markersize=4)
    xs, ys =  normalised_true_moving_average(steps, progresses, 20)
    plt.plot(xs, ys, linewidth='4', color='r')
    # plt.gca().get_yaxis().set_major_locator(MultipleLocator(2))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Max Progess")

    plt.tight_layout()
    plt.grid()

    # if save_tex:
    #     tikzplotlib.save(path + "baseline_reward_plot.tex", strict=True, extra_axis_parameters=['height=4cm', 'width=0.5\\textwidth', 'clip mode=individual'])

    name = path.split("/")[-2]
    plt.savefig(path + f"training_progress_steps_{name}.pdf")
    plt.show()
    # plt.pause(0.001)



def generate_racing_training_graphs():
    p = "Data/Vehicles/FastTests/"

    rewards = ["Cth", "Progress"]
    for r in rewards:
        name = f"Agent_{r}_f1_esp_3_0/"

        path = p + name
        plot_reward_steps(path)
        plot_lap_times(path)


def link_ee_compare_training():
    ee5 = "Data/Vehicles/MaxSpeedTests/Agent_Cth_f1_gbr_5_0/"
    ee8 = "Data/Vehicles/MaxSpeedTests/Agent_Cth_f1_gbr_7_0/"
    # link8 = "Data/Vehicles/LinkTests/Link_Cth_f1_esp_1_0/"
    link7 = "Data/Vehicles/LinkTests/Link_Cth_f1_gbr_6_0/"

    agents = [ee5, ee8, link7]

    steps_list = []
    progresses_list = []

    n_repeats = 1
    for i in range(3):
        rewards, lengths, progresses, _ = load_csv_data(agents[i])
        steps = np.cumsum(lengths) / 100
        # xs = np.linspace(0, steps[-1], 100)
        # avg_progress = normalised_true_moving_average(steps, progresses, 20)* 100
        avg_progress = true_moving_average(progresses, 20)* 100
        # avg_progress = smooth_line(steps, avg_progress, 100)
        steps_list.append(steps)
        progresses_list.append(avg_progress)

    plt.figure(1, figsize=(6, 2))

    colors = ['darkblue', 'green', 'red']
    labels = ["E2e5", "E2e7", "Link7"]

    for i in range(3):
        s_xs, s_ys = smooth_line(steps_list[i], progresses_list[i], 300)
        plt.plot(s_xs, s_ys, '-', color=colors[i], linewidth=2, label=labels[i])

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 105)
    # plt.legend(loc='lower right')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/RacingEval/" + "link_ee_compare_training.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()
    print("--------------------------------")


# slow_progress_training_comparision()
# compare_ep_vs_step_rewards()

# fast_progress_training_comparision_maxspeed()
# fast_progress_training_ppps_maps()
fast_reward_comparision()
# generate_racing_training_graphs()
# fast_repeatability()

# plot_reward_steps("Data/Vehicles/GpsTests/Fast_Gps_f1_gbr_3_0/")
# plot_reward_steps("Data/Vehicles/GpsTests/Fast_Gps_f1_gbr_3_0/")
# plot_progress_steps("Data/Vehicles/GpsTests/Fast_Gps_f1_gbr_3_0/")

# link_ee_compare_training()

# path_list = ["Data/Vehicles/PaperTests/Fast_Gps_f1_mco_1_0/", "Data/Vehicles/PaperTests/Fast_CthV_f1_mco_4_0/"]
# plot_progress_steps_list(path_list)
# plot_maps_graph()


# plot_v_reward_graph()