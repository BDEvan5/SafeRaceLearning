import numpy as np
import yaml



def make_kernel_gen_table_paper():
    kp = "Data/Kernels/"

    metric_keys = ["kernel_shape", "iterations", "safe_rate"]
    metrics = ["No. Track States", "Iterations Required", "\% Track Safe"]
    assert len(metric_keys) == len(metrics), "Must have same number of metric keys and metrics"

    np.set_printoptions(formatter={'int_kind': '{:,}'.format})
    with open(kp + "KernelGenTable.txt", "w") as file:
        file.write(f"\\toprule \n")
        file.write("\\textbf{Map} & \\textbf{ " + " } &  \\textbf{ ".join(metrics) + "}  \\\\ \n")
        file.write(f"\\midrule \n")

    k_modes = ["fast_6"]
    for k, k_mode in enumerate(k_modes):
        k_names = [f"Kernel_{k_mode}_f1_aut", f"Kernel_{k_mode}_f1_esp", f"Kernel_{k_mode}_f1_mco", f"Kernel_{k_mode}_f1_gbr"]
        print_names = ["AUT", "ESP", "MCO", "GBR"]
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
                file.write(f"{print_names[i]}  ".ljust(30))
                for j in range(len(metrics)):
                    value = np.array(data[i][metric_keys[j]])
                    string = np.array2string(value, precision=2)
                    file.write(f"& {string} ".ljust(20))
                file.write("\\\\ \n")
            if k == 0: file.write(f"\\midrule \n")
            else: file.write(f"\\bottomrule \n")


    print("Done")

make_kernel_gen_table_paper()







