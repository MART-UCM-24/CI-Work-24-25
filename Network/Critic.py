import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,device):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.GRU(400, 300)
        self.layer3 = nn.Linear(300, 1)
        self.device = device
        self.to(device) 

    def forward(self, state, action):
        state = state.to(self.device)  # Ensure the input is on the correct device
        action = action.to(self.device)  # Ensure the input is on the correct device

        x = torch.relu(self.layer1(torch.cat([state, action], 1)))
        x,_= self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x