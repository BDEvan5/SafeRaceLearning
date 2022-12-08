import matplotlib.pyplot as plt
import numpy as np 
from SafeRaceLearning.DataTools.plotting_utils import *



def safe_sample_efficiency():
    folder = "Data/Vehicles/Safe_TrainSteps/"
    
    xs = [2, 3, 4, 5, 6, 8, 10]
    y_means = []
    y_mins = []
    y_maxes = []
    
    for i in [2, 3, 4, 5, 6, 8, 10]:
        mins, maxes, means = load_time_data(folder, f"5_{i}.")
        
        y_mins.append(mins['progress'][0])
        y_maxes.append(maxes['progress'][0])
        y_means.append(means['progress'][0])
        
    plt.figure(figsize=(4, 2))

    print(y_mins)
    print(y_means)
    print(y_maxes)

    plt.plot(xs, y_means, color='#7B241C')
    plt.fill_between(xs, y_mins, y_maxes, color="#EC7063", alpha=0.7)
    plt.grid(True)
    
    std_img_saving(folder + f"Safe_TrainSteps_sampleEfficiency")
    
    
def safe_sample_efficiency_bars():
    folder = "Data/Vehicles/Safe_TrainSteps/"
    
    xs = [2, 3, 4, 5, 6, 8, 10]
    y_means = []
    y_mins = []
    y_maxes = []
    
    for i in [2, 3, 4, 5, 6, 8, 10]:
        mins, maxes, means = load_time_data(folder, f"5_{i}.")
        
        y_mins.append(mins['progress'][0])
        y_maxes.append(maxes['progress'][0])
        y_means.append(means['progress'][0])
        
    plt.figure(figsize=(4, 2))

    # print(y_mins)
    # print(y_means)
    # print(y_maxes)

    barWidth = 0.4
    w = 0.05
    br1 = xs 
    plt.bar(br1, y_means)

    # plt.plot(xs, y_means, color='#7B241C')
    # plt.fill_between(xs, y_mins, y_maxes, color="#EC7063", alpha=0.7)
    plt.grid(True)
    
    std_img_saving(folder + f"Safe_TrainSteps_sampleEfficiency_bars")
    

def std_sample_efficiency():
    folder = "Data/Vehicles/Std_TrainSteps/"
    
    xs = [1, 2, 3, 4, 5, 6, 8, 10]
    y_means = []
    y_mins = []
    y_maxes = []
    
    for i in [1, 2, 3, 4, 5, 6, 8, 10]:
        mins, maxes, means = load_time_data(folder, f"6_{i}.")
        
        y_mins.append(mins['progress'][0])
        y_maxes.append(maxes['progress'][0])
        y_means.append(means['progress'][0])
        
    plt.figure(figsize=(4, 2))

    print(y_mins)
    print(y_means)
    print(y_maxes)

    plt.plot(xs, y_means, color='#7B241C')
    plt.fill_between(xs, y_mins, y_maxes, color="#EC7063", alpha=0.7)
    plt.grid(True)
    
    std_img_saving(folder + f"Std_TrainSteps_sampleEfficiency")
    
# std_sample_efficiency()
# safe_sample_efficiency()
safe_sample_efficiency_bars()