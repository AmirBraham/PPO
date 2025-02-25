import torch
import torch.nn as nn
import torch.optim as optim

# This network module to define our actor and critic, and they both will take in an observation and return either an action or a value,

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(FeedForwardNetwork, self).__init__()
        # Step 1 of the PPO algorithm: initialize the policy parameters
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x is the observation
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
