import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,device):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.GRU(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.device = device
        self.to(device) 

    def forward(self, state):
        state = state.to(self.device)  # Ensure the input is on the correct device
        x = torch.relu(self.layer1(state))
        x,_ = self.layer2(x)
        x = torch.relu(x)
        x = torch.tanh(self.layer3(x))
        return x * self.max_action