import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    
    test_x = torch.FloatTensor(test_x)
    test_y = torch.LongTensor(test_y)
    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    
    print(f"X --> Train: {train_x.shape} --> Test: {test_x.shape}")
    print(f"Y --> Train: {train_y.shape} --> Test: {test_y.shape}")
    
    # return train_x, train_y
    return train_x, train_y, test_x, test_y
    

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 100) # input layer to hidden layer
        self.fc2 = nn.Linear(100, 2) # hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training function
def train_model(net, input_data, labels, num_epochs=5000, learning_rate=0.01):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = net(input_data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 10 == 0:
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return net


def test_model(model, input_data, labels):
    # Evaluate the model on the test data
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        print('Accuracy: {:.2f}%'.format(accuracy*100))


input_data, labels, test_data, test_labels = load_data("Data/Vehicles/SSS_DataGen/RandoData_Std_Super_None_f1_gbr_2_1_0/")
net = Net()

for i in range(1):
    train_model(net, input_data, labels, 2000)

    test_model(net, test_data, test_labels)
