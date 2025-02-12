import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,device,dtype=torch.float):
        super(Critic, self).__init__()
        self.device = device
        self.dtype = dtype
        self.lstm = nn.LSTM(input_size= state_dim + action_dim,hidden_size= 128, batch_first=True,num_layers=2,dropout=0.1)
        self.fc1 = nn.Linear(128, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self = self.to(device=device,dtype = dtype)

    def forward(self, x, u):
        x = x.to(self.device)
        u = u.to(self.device)
        xu = torch.cat([x, u], dim=-1).unsqueeze(1)  # Add batch dimension
        xu, _ = self.lstm(xu)
        x = F.relu(self.fc1(xu[:, -1, :]))  # Use the output of the last LSTM cell
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x