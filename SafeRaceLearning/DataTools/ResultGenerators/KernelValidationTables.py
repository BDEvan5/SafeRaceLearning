import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from SafeRaceLearning.Utils.utils import *

def KernelValidationRandom():
    folder = "Data/Vehicles/SSS_RandomValidation/"

    map_names = ["f1_aut", "f1_mco", "f1_esp", "f1_gbr"]
    fast_agents = [f"Rando_Std_Super_None_{map_name}_6_1_0" for map_name in map_names]
    
    print_maps = ["AUT", "MCO", "ESP", "GBR"]
    metrics = ["No. of Interventions per Lap", "Intervention Rate (\%)", "Total Crashes"]
    metric_inds = [-3, -2, -1]

    data_fast, data_std_fast = [[] for _ in metrics], [[] for _ in metrics]

    for agent in fast_agents:
        with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
            lines = file.readlines()
            line = lines[2] # first lap is heading
            line = line.split(',')
            for i in range(len(metrics)):
                metric_data = float(line[metric_inds[i]])
                data_fast[i].append(metric_data)
            line = lines[3] # first lap is heading
            line = line.split(',')
            for i in range(len(metrics)):
                metric_data = float(line[metric_inds[i]])
                data_std_fast[i].append(metric_data)

    with open(folder + f"RandoValidation.txt", 'w') as file:
        file.write(f"\\toprule \n")
        file.write("\\textbf{Metric} & \\textbf{ " + " } &  \\textbf{ ".join(metrics) + "}   \\\\ \n")
        file.write(f"\\midrule \n")
        for i in range(len(print_maps)):
            file.write(f"{print_maps[i]} ".ljust(20))
            for j in range(len(metrics)-1):
                file.write(f"& {data_fast[j][i]:.1f} $\pm$  {data_std_fast[j][i]:.1f}  ".ljust(25))
            file.write(f"& {int(100-data_fast[-1][i])}  ".ljust(15))
            file.write("\\\\ \n")
        file.write(f"\\bottomrule \n")


def KernelValidationPP():
    folder = "Data/Vehicles/SSS_ppValidation/"

    map_name = "f1_mco"
    for map_name in ["f1_aut", "f1_mco", "f1_esp", "f1_gbr"]:
        speeds = [3, 4, 5, 6]
        std_agents = [f"PP_PP_Std_PP_{map_name}_{i}_1_0" for i in speeds]
        super_agents = [f"PP_PP_Super_PP_{map_name}_{i}_1_0" for i in speeds]
        
        print_names = speeds
        metrics_std = ["Lap Time (s)", "Avg. Velocity (m/s)"]
        metrics_super = ["No. Interventions", "Lap Time (s)", "Avg. Velocity (m/s)"]
        std_metric_inds = [8, 9]
        super_metric_inds = [-3, 8, 9]

        data_std = [[] for _ in metrics_std]
        data_super = [[] for _ in metrics_super]

        for agent in std_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_std)):
                    metric_data = float(line[std_metric_inds[i]])
                    data_std[i].append(metric_data)
        
        for agent in super_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_super)):
                    metric_data = float(line[super_metric_inds[i]])
                    data_super[i].append(metric_data)

        with open(folder + f"KernelValidationPP_{map_name}.txt", 'w') as file:
            file.write(f"\\toprule \n")
            file.write(" &  \multicolumn{2}{c}{\\textit{Pure Pursuit}}  & &  \multicolumn{3}{c}{\\textit{Pure Pursuit with SSS}}  \\\\ \n")
            file.write("\\cmidrule(lr){2-3} \n")
            file.write("\\cmidrule(lr){5-7} \n")
            file.write("\\textbf{Max Speed} & \\textbf{ " + " } &  \\textbf{ ".join(metrics_std) + "} & {" + "} & \\textbf{ " + " } &  \\textbf{ ".join(metrics_super) + "}   \\\\ \n")
            file.write(f"\\midrule \n")
            for i in range(len(print_names)):
                file.write(f"{print_names[i]} ".ljust(20))
                for j in range(len(metrics_std)):
                    file.write(f"& {data_std[j][i]:.1f}   ".ljust(15))
                file.write(" & ")
                for j in range(len(metrics_super)):
                    file.write(f"& {data_super[j][i]:.1f}   ".ljust(15))
                file.write("\\\\ \n")
            file.write(f"\\bottomrule \n")


def KernelValidationPP_Graphs():
    folder = "Data/Vehicles/SSS_ppValidation/"

    map_name = "f1_mco"
    for map_name in ["f1_aut", "f1_mco", "f1_esp", "f1_gbr"]:
        speeds = [3, 4, 5, 6]
        std_agents = [f"PP_PP_Std_PP_{map_name}_{i}_1_0" for i in speeds]
        super_agents = [f"PP_PP_Super_PP_{map_name}_{i}_1_0" for i in speeds]
        
        print_names = speeds
        metrics_std = ["Lap Time (s)", "Avg. Velocity (m/s)"]
        metrics_super = ["No. Interventions", "Lap Time (s)", "Avg. Velocity (m/s)"]
        std_metric_inds = [8, 9]
        super_metric_inds = [-3, 8, 9]

        data_std = [[] for _ in metrics_std]
        data_super = [[] for _ in metrics_super]

        for agent in std_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_std)):
                    metric_data = float(line[std_metric_inds[i]])
                    data_std[i].append(metric_data)
        
        for agent in super_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_super)):
                    metric_data = float(line[super_metric_inds[i]])
                    data_super[i].append(metric_data)

        fig, ax = plt.subplots(1, 2, figsize=(4, 2))
        barWidth = 0.4
        
        
        br1 = np.arange(4) - barWidth/2
        br2 = [x + barWidth for x in br1]
        
        ax[0].bar(br1, data_std[0], color='#E67E22', width=barWidth, edgecolor='white', label='PP')
        ax[0].bar(br2, data_super[1], color='#9B59B6', width=barWidth, edgecolor='white', label='Supervised PP')

        ax[0].set_ylabel('Lap Time (s)')
        ax[0].set_xticks([0, 1, 2, 3], [3, 4, 5, 6])
        ax[0].set_xlabel('Max Speed (m/s)')
        ax[0].grid(True)
        fig.legend(loc='center', bbox_to_anchor=(0.5, 1.0), ncol=2, fancybox=True)
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        # ax[0].yaxis.set_major_locator(MultipleLocator(10))
        
        ax[1].plot(data_super[0], linewidth=2, color="#17A589")

        ax[1].grid(True)
        ax[1].set_xticks([0, 1, 2, 3], [3, 4, 5, 6])
        ax[1].set_ylabel('Interventions')
        ax[1].set_xlabel('Max Speed (m/s)')
        # ax[1].yaxis.set_major_locator(MultipleLocator(20))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        
        plt.tight_layout()
        plt.savefig(f"Data/Vehicles/SSS_ppValidation/KernelValidationPP_{map_name}.svg", pad_inches=0, bbox_inches='tight')
        plt.savefig(f"Data/Vehicles/SSS_ppValidation/KernelValidationPP_{map_name}.pdf", pad_inches=0, bbox_inches='tight')
        
def KernelValidationPP_GraphsIntervention():
    folder = "Data/Vehicles/SSS_ppValidation/"

    plt.figure(figsize=(4, 2))
    colors = ["#16A085", "#E74C3C", "#3498DB", "#D35400"]

    print_map_names = ["AUT", "MCO", "ESP", "GBR"]
    for m, map_name in enumerate(["f1_aut", "f1_mco", "f1_esp", "f1_gbr"]):
        speeds = [3, 4, 5, 6]
        std_agents = [f"PP_PP_Std_PP_{map_name}_{i}_1_0" for i in speeds]
        super_agents = [f"PP_PP_Super_PP_{map_name}_{i}_1_0" for i in speeds]
        
        print_names = speeds
        metrics_std = ["Lap Time (s)", "Avg. Velocity (m/s)"]
        metrics_super = ["No. Interventions", "Lap Time (s)", "Avg. Velocity (m/s)"]
        std_metric_inds = [8, 9]
        super_metric_inds = [-3, 8, 9]

        data_std = [[] for _ in metrics_std]
        data_super = [[] for _ in metrics_super]

        for agent in std_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_std)):
                    metric_data = float(line[std_metric_inds[i]])
                    data_std[i].append(metric_data)
        
        for agent in super_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_super)):
                    metric_data = float(line[super_metric_inds[i]])
                    data_super[i].append(metric_data)

        plt.plot(data_super[0], linewidth=2, color=pp[m], label=print_map_names[m])

    plt.grid(True)
    plt.xticks([0, 1, 2, 3], [3, 4, 5, 6])
    plt.ylabel('Interventions')
    plt.xlabel('Max Speed (m/s)')
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    plt.legend(ncol=4)
    
    plt.tight_layout()
    plt.savefig(f"Data/Vehicles/SSS_ppValidation/KernelValidationPP_interventions.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/Vehicles/SSS_ppValidation/KernelValidationPP_interventions.pdf", pad_inches=0, bbox_inches='tight')
                
def KernelRandomSpeeds_GraphsIntervention():
    folder = "Data/Vehicles/SSS_RandomSpeeds/"

    plt.figure(figsize=(4., 2))
    colors = ["#16A085", "#E74C3C", "#3498DB", "#D35400"]

    print_map_names = ["AUT", "MCO", "ESP", "GBR"]
    for m, map_name in enumerate(["f1_aut", "f1_mco", "f1_esp", "f1_gbr"]):
        speeds = [2, 3, 4, 5, 6]
        super_agents = [f"Rando_Std_Super_None_{map_name}_{i}_1_0" for i in speeds]
        
        print_names = speeds
        metrics_super = ["No. Interventions", "Lap Time (s)", "Avg. Velocity (m/s)"]
        super_metric_inds = [-3, 8, 9]

        data_super = [[] for _ in metrics_super]
        for agent in super_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_super)):
                    metric_data = float(line[super_metric_inds[i]])
                    data_super[i].append(metric_data)

        plt.plot(data_super[0], linewidth=2, color=pp[m], label=print_map_names[m])

    plt.grid(True)
    plt.xticks([0, 1, 2, 3, 4], [2, 3, 4, 5, 6])
    plt.ylabel('Interventions')
    plt.xlabel('Max Speed (m/s)')
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    plt.legend(ncol=4, loc='center', bbox_to_anchor=(0.42, 1.24), framealpha=0)
    
    plt.tight_layout()
    plt.savefig(f"Data/Vehicles/SSS_RandomSpeeds/KernelRandomSpeeds_interventions.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/Vehicles/SSS_RandomSpeeds/KernelRandomSpeeds_interventions.pdf", pad_inches=0, bbox_inches='tight')
        
        
def KernelValidationPP_BarGraphs():
    folder = "Data/Vehicles/SSS_ppValidation/"

    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    barWidth = 0.4
    
    br1 = np.arange(4) - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    map_names = ["f1_mco", "f1_esp"]
    print_map_names = ["MCO", "ESP"]
    for m, map_name in enumerate(map_names):
        speeds = [3, 4, 5, 6]
        std_agents = [f"PP_PP_Std_PP_{map_name}_{i}_1_0" for i in speeds]
        super_agents = [f"PP_PP_Super_PP_{map_name}_{i}_1_0" for i in speeds]
        
        print_names = speeds
        metrics_std = ["Lap Time (s)", "Avg. Velocity (m/s)"]
        metrics_super = ["No. Interventions", "Lap Time (s)", "Avg. Velocity (m/s)"]
        std_metric_inds = [8, 9]
        super_metric_inds = [-3, 8, 9]

        data_std = [[] for _ in metrics_std]
        data_super = [[] for _ in metrics_super]

        for agent in std_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_std)):
                    metric_data = float(line[std_metric_inds[i]])
                    data_std[i].append(metric_data)
        
        for agent in super_agents:
            with open(folder + f"{agent}/TestingSummaryStatistics.txt", 'r') as file:
                lines = file.readlines()
                line = lines[2] # first lap is heading
                line = line.split(',')
                for i in range(len(metrics_super)):
                    metric_data = float(line[super_metric_inds[i]])
                    data_super[i].append(metric_data)


        ax[m].bar(br1, data_std[0], color='#E67E22', width=barWidth, edgecolor='white', label='PP')
        ax[m].bar(br2, data_super[1], color='#9B59B6', width=barWidth, edgecolor='white', label='Supervised PP')

        ax[m].set_ylabel('Lap Time (s)')
        ax[m].set_xticks([0, 1, 2, 3], [3, 4, 5, 6])
        ax[m].set_xlabel('Max Speed (m/s)')
        ax[m].grid(True)
        ax[m].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax[m].set_title(print_map_names[m])
    
    handles, labels = ax[0].get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=2, loc="center", bbox_to_anchor=(0.55, 0.01))
    
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0), ncol=2, fancybox=True)
    plt.tight_layout()
    plt.savefig(f"Data/Vehicles/SSS_ppValidation/KernelValidationPP_BAR_{map_name}.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/Vehicles/SSS_ppValidation/KernelValidationPP_BAR_{map_name}.pdf", pad_inches=0, bbox_inches='tight')

# KernelValidationRandom()
# KernelValidationPP_BarGraphs()
# KernelValidationPP_GraphsIntervention()
KernelRandomSpeeds_GraphsIntervention()
# KernelValidationPP_Graphs()
# KernelValidationPP()


