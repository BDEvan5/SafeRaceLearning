
import os
import numpy as np 
import glob


def load_state_and_safety_data(folder, vehicle_name, map_name, lap_number):
    
    # files = glob.glob(folder + f"{vehicle_name}/SafeHistory/*")
    # print(files)
    # files = glob.glob(folder + f"{vehicle_name}/RawData/*")
    # print(files)
    # files = glob.glob(folder + f"{vehicle_name}/*")
    # print(files)
    # files = glob.glob(folder + "*")
    # print(files)
    # files = glob.glob("Data/Vehicles/*")
    # print(files)
    
    lidar_data = np.load(folder + f"{vehicle_name}/" + f"RawData/RandoData_Std_Super_None_{map_name}_2_1_0_Lap_{lap_number}_scans.npy")
    
    safety_history = np.load(folder + f"{vehicle_name}/SafeHistory/Lap_{lap_number}_safeHistory_{vehicle_name}.npy")
    
    interventions = safety_history[:, 4]

    return lidar_data, interventions



def format_data_set():
    folder = "Data/Vehicles/SSS_DataGen/"
    vehicle_model = "RandoData_Std_Super_None"
    map_name = "f1_gbr"
    vehicle_code = "2_1_0"
    
    vehicle_name = vehicle_model + "_" + map_name + "_" + vehicle_code

    xs, ys = [], []
    for i in range(10):
        x, y = load_state_and_safety_data(folder, vehicle_name, map_name, i)
        for i in range(len(x)):
            xs.append(x[i] / 10)
            ys.append(y[i])
        
    xs = np.array(xs)
    ys = np.array(ys)
    
    save_path = folder + vehicle_name + "/DataSets/"
    if not os.path.exists(save_path): os.mkdir(save_path)
    
    np.save(save_path + f"input.npy", xs)
    np.save(save_path + f"output.npy", ys)




format_data_set()
