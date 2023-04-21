


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

import numpy as np
import os
from matplotlib import pyplot as plt


NN_LAYER_1 = 100
NN_LAYER_2 = 100



class StdNetworkTwo(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(StdNetworkTwo, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)
        
    def forward_loss(self, x, targets):
        mu = self.forward(x)
        loss = F.cross_entropy(mu, targets)
        
        return mu, loss
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        # mu = F.softmax(x, dim=1)
        
        return mu
    
    
def load_data(folder):
    
    states = np.load(folder + f"DataSets/input.npy")
    actions = np.load(folder + f"DataSets/output.npy")
    
    # test_size = 200
    test_size = int(0.1*states.shape[0])
    test_inds = np.random.choice(states.shape[0], size=test_size, replace=False)
    
    test_x = states[test_inds]
    test_y = actions[test_inds]
    
    train_x = states[~np.isin(np.arange(states.shape[0]), test_inds)]
    train_y = actions[~np.isin(np.arange(states.shape[0]), test_inds)]
    
    # train_y = train_y.reshape(train_y.shape[0], 1)
    # test_y = test_y.reshape(test_y.shape[0], 1)
    
    # y2 = np.zeros((train_y.shape[0], 2))
    # y2[train_y == 0, 0] = 1
    # y2[train_y == 1, 1] = 1
    # train_y = y2
    
    # y2 = np.zeros((test_y.shape[0], 2))
    # y2[test_y == 0, 0] = 1
    # y2[test_y == 1, 1] = 1
    # test_y = y2
    
    test_x = torch.FloatTensor(test_x)
    test_y = torch.LongTensor(test_y)
    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    
    print(f"X --> Train: {train_x.shape} --> Test: {test_x.shape}")
    print(f"Y --> Train: {train_y.shape} --> Test: {test_y.shape}")
    
    return train_x, train_y, test_x, test_y
    
def train_networks(folder, seed):
    train_x, train_y, test_x, test_y = load_data(folder)
    
    network = StdNetworkTwo(train_x.shape[1], 2)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    train_iterations = 300
    train_losses, test_losses = [], []
    
    for i in range(train_iterations):
        network.eval()
        test_pred_y, test_loss = network.forward_loss(test_x, test_y)
        _, predictions = torch.max(test_pred_y, dim=1)
        n_correct = torch.sum(predictions == test_y)
        print(f"Test Accuracy: {n_correct.item() / test_y.shape[0] * 100}%")
        network.train()
        
        pred_y, loss = network.forward_loss(train_x, train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_loss = test_loss.item() ** 0.5
        train_loss = loss.item() ** 0.5
        test_losses.append(test_loss)
        train_losses.append(train_loss)
        
        if i % 50 == 0:
            print(f"{i}: TrainLoss: {train_loss} --> TestLoss: {test_loss}")
         
    plt.figure()
    plt.plot(test_losses, label="Test")
    plt.plot(train_losses, label="Train")
    
    plt.legend()
    plt.show()
    
    return train_losses, test_losses
    
    
def run_seeded_test(folder, seeds):
    train_losses, test_losses = [], []    
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        train_loss, test_loss = train_networks(folder, seeds[i])
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    
    return train_losses, test_losses
        
    
def run_experiment(folder, experiment_name):
    save_path = folder + "LossResults/"
    if not os.path.exists(save_path): os.mkdir(save_path)
    spacing = 30
    with open(folder + f"LossResults/{experiment_name}_LossResults.txt", "w") as f:
        f.write(f"Name,".ljust(spacing))
        f.write(f"TrainLoss mean, TrainLoss std ,   TestLoss mean, TestLoss std \n")
        
    seeds = np.arange(1)
    train_losses, test_losses = run_seeded_test(folder, seeds)
    
    # Add some individual plotting here....
    np.save(folder + f"LossResults/{experiment_name}_train_losses.npy", train_losses)
    np.save(folder + f"LossResults/{experiment_name}_test_losses.npy", test_losses)
    
    with open(folder + f"LossResults/{experiment_name}_LossResults.txt", "a") as f:
        f.write(f",".ljust(spacing))
        f.write(f"{np.mean(train_losses[:, -1]):.5f},     {np.std(train_losses[:, -1]):.5f},       {np.mean(test_losses[:, -1]):.5f},       {np.std(test_losses[:, -1]):.5f} \n")
    
        
        
def run_training():
    vehicle_model = "RandoData_Std_Super_None"
    map_name = "f1_gbr"
    vehicle_code = "2_1_0"
    
    vehicle_name = vehicle_model + "_" + map_name + "_" + vehicle_code
    folder = "Data/Vehicles/SSS_DataGen/" + vehicle_name + "/"
    
    run_experiment(folder, vehicle_name)
    
    

    
    
     
if __name__ == "__main__":
    run_training()
    