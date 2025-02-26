import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(FeedForwardNetwork, self).__init__()
        # Initialize the layers of the network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_final = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Use tanh and scale the output to the range [-2, 2] for Pendulum
        x = torch.tanh(self.fc_final(x))
        return x * 2.0
