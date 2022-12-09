

from SafeRaceLearning.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator




   
def comparison_performance_Barplot():
    safe_folder = "Data/Vehicles/Safe_TrainMaps/"
    std_folder = "Data/Vehicles/Std_TrainMaps_old/"
    pp_folder = "Data/Vehicles/PP_TestMaps/"
    
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 1.8))
    xs = np.arange(4)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    keys = ["time", "success"]
    ylabels = "Time (s), Success (%)".split(", ")

    for z in range(2):
        key = keys[z]
        plt.sca(axs[z])
        mins, maxes, means = load_time_data(safe_folder, "")
        
        plt.bar(br1, means[key], color=pp_light[0], width=barWidth, label="Safe")
        plot_error_bars(br1, mins[key], maxes[key], pp_darkest[4], w)
        
        mins, maxes, means = load_time_data(std_folder, "")
        plt.bar(br2, means[key], color=pp_light[2], width=barWidth, label="Conventional")
        plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
            
        plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
        plt.xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
        plt.ylabel(ylabels[z])
        plt.grid(True)
    
    axs[0].yaxis.set_major_locator(MultipleLocator(15))
    axs[1].yaxis.set_major_locator(MultipleLocator(25))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="center", bbox_to_anchor=(0.55, 0.01))
        
    name = "Data/Images/" + f"PerformanceComparison_Barplot"
    
    std_img_saving(name)
   
   
comparison_performance_Barplot()
   