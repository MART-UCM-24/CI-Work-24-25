import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,device,dtype=torch.float):
        super(Actor, self).__init__()
        self.device = device
        self.dtype = dtype
        self.lstm = nn.LSTM(state_dim, 128, batch_first=True,num_layers=2)
        self.fc1 = nn.Linear(128, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self = self.to(device=device,dtype=dtype)

    def forward(self, x):

        x = x.to(self.device)

        # x, _ = self.lstm(x)
        # Ensure x is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing

        x, _ = self.lstm(x)
        #print(x.shape)  # Check the shape of x after LSTM

        # Ensure x is 3D before indexing
        if x.dim() == 3:
            x = F.relu(self.fc1(x[:, -1, :]))  # Use the output of the last LSTM cell
        else:
            raise ValueError("Unexpected tensor shape after LSTM: {}".format(x.shape))
        #x = F.relu(self.fc1(x[:, -1]))  # Adjust indexing if x has only two  # Use the output of the last LSTM cell
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x