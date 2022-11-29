import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import csv
import glob, os 
from matplotlib.ticker import MultipleLocator, PercentFormatter

from SafeRaceLearning.Utils.utils import *
from SafeRaceLearning.DataTools.TrainingGraphs.TrainingUtils import *



def Eval_RewardsSlow_TrainingGraphAllMaps():
    p = "Data/Vehicles/Eval_RewardsSlow/"

    map_names = ["f1_esp", "f1_gbr", "f1_aut", "f1_mco"]
    repeats = 5
    moving_avg = 2
    reward_list = ["Cth", "Progress", "Zero"]
    xs = np.arange(300)
        
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 4))
    axs = axs.reshape(-1)
        
    for ax, map_name in enumerate(map_names):
    
        step_list = [[] for _ in range(len(reward_list))]
        progresses_list = [[] for _ in range(len(reward_list))]
            
        for i in range(repeats):
            for j in range(len(reward_list)):
                path = p + f"slow_Std_Std_{reward_list[j]}_{map_name}_2_1_{i}/"
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
                
            if np.mean(mins) > 0.1:
                axs[ax].fill_between(xs, mins, maxes, color=pp[j], alpha=0.3)
            
            axs[ax].grid(True)
            axs[ax].text(220, 10, map_name.split('_')[1].upper(),  fontsize=12)

    axs[2].set_xlabel("Training Steps (x100)")
    axs[3].set_xlabel("Training Steps (x100)")
    axs[0].set_ylabel("Avg. Progress %")
    axs[2].set_ylabel("Avg. Progress %")
    axs[0].legend(loc='center', ncol=3, bbox_to_anchor=(1.1, 1.1))

    name = p + f"Eval_RewardsSlow_TrainingGraphAllMaps"
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)


Eval_RewardsSlow_TrainingGraphAllMaps()

