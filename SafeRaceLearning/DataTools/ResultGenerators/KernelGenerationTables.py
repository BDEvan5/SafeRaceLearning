import numpy as np
import yaml


def make_kernel_gen_table():
    kp = "Data/Kernels3/"

    metric_keys = ["total_positions", "kernel_shape", "iterations", "Size", "safe_rate"]
    metrics = ["No. of Positions", "No. Track States", "Iterations Required", "Kernel Size (MB)", "\% Track Safe"]
    assert len(metric_keys) == len(metrics), "Must have same number of metric keys and metrics"

    np.set_printoptions(formatter={'int_kind': '{:,}'.format})
    with open(kp + "KernelGenTable.txt", "w") as file:
        file.write(f"\\toprule \n")
        file.write("\\textbf{Map} & \\textbf{ " + " } &  \\textbf{ ".join(metrics) + "}  \\\\ \n")
        file.write(f"\\midrule \n")

    k_modes = ["slow_2", "fast_8"]
    k_mode_names = ["slow", "fast"]
    for k, k_mode in enumerate(k_modes):
        k_names = [f"Kernel_{k_mode}_f1_aut_wide", f"Kernel_{k_mode}_f1_esp", f"Kernel_{k_mode}_f1_mco"]
        print_names = ["WAUT", "ESP", "MCO"]
        assert len(k_names) == len(print_names), "Must have same number of kernel names and print names"

        data = []
        for i, k_name in enumerate(k_names):
            data.append({})
            path = kp + k_name + ".yaml"
            with open(path) as file:
                conf_dict = yaml.load(file, Loader=yaml.FullLoader)

            for key in metric_keys:
                if key == "kernel_shape": 
                    data[i]["kernel_shape"] = np.prod(conf_dict[key]) 
                    data[i]["Size"] = np.prod(conf_dict[key]) / 1e6
                    continue
                elif key == "Size": continue
                data[i][key] = conf_dict[key]

        with open(kp + "KernelGenTable.txt", "a") as file:
            for i in range(len(print_names)):
                file.write(f"{print_names[i]} - {k_mode_names[k]} ".ljust(30))
                for j in range(len(metrics)):
                    value = np.array(data[i][metric_keys[j]])
                    string = np.array2string(value, precision=2)
                    file.write(f"& {string} ".ljust(20))
                file.write("\\\\ \n")
            if k == 0: file.write(f"\\midrule \n")
            else: file.write(f"\\bottomrule \n")


    print("Done")

def make_filter_gen_table():
    kp = "Data/Kernels/"

    metric_keys = ["kernel_shape", "safe_rate"]
    metrics = ["No. Track States", "\% Track Safe"]
    assert len(metric_keys) == len(metrics), "Must have same number of metric keys and metrics"

    np.set_printoptions(formatter={'int_kind': '{:,}'.format})
    with open(kp + "KernelFilterTable.txt", "w") as file:
        file.write(f"\\toprule \n")
        file.write(" &  \\multicolumn{2}{c}{\\textit{Original}}  & &  \\multicolumn{2}{c}{\\textit{Filtered}}  \\\\ \n")
        file.write(" \\cmidrule{2-3} \\cmidrule{5-6} \n")
        file.write("\\textbf{Map} & \\textbf{ " + " } &  \\textbf{ ".join(metrics) + "} & & \\textbf{ " + " } &  \\textbf{ ".join(metrics) + "}  \\\\ \n")
        file.write(f"\midrule \n")

    filter_modes = ["", "filter"]

    k_modes = ["slow_2"]
    k_mode_names = ["slow", "fast"]
    for k, k_mode in enumerate(k_modes):
        k_names = [f"Kernel_{k_mode}_f1_aut_wide", f"Kernel_{k_mode}_f1_esp"]
        print_names = ["WAUT", "ESP"]
        assert len(k_names) == len(print_names), "Must have same number of kernel names and print names"

        data = []
        data_filter = []
        for i, k_name in enumerate(k_names):
            data.append({})
            data_filter.append({})
            path = kp + k_name 
            with open(path+ ".yaml") as file:
                conf_dict = yaml.load(file, Loader=yaml.FullLoader)

            with open(path + "_kl_filter.yaml") as file:
                conf_dict_filter = yaml.load(file, Loader=yaml.FullLoader)

            for key in metric_keys:
                if key == "kernel_shape": 
                    data[i]["kernel_shape"] = np.prod(conf_dict[key]) 
                    data_filter[i]["kernel_shape"] = np.prod(conf_dict_filter[key]) 
                    continue
                data_filter[i][key] = conf_dict_filter[key]
                data[i][key] = conf_dict[key]

        with open(kp + "KernelFilterTable.txt", "a") as file:
            for i in range(len(print_names)):
                file.write(f"{print_names[i]} - {k_mode_names[k]} ".ljust(30))
                for j in range(len(metrics)):
                    value = np.array(data[i][metric_keys[j]])
                    string = np.array2string(value, precision=2)
                    file.write(f"& {string} ".ljust(20))
                file.write(f" & ")
                for j in range(len(metrics)):
                    value = np.array(data_filter[i][metric_keys[j]])
                    string = np.array2string(value, precision=2)
                    file.write(f"& {string} ".ljust(20))
                file.write("\\\\ \n")
            if k == 0: file.write(f"\\midrule \n")
            else: file.write(f"\\bottomrule \n")


    print("Done")

# make_kernel_gen_table()
make_filter_gen_table()







