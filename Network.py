import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy
import gc
import logging
import os
import torch.optim.lr_scheduler as lr_scheduler

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,device,dtype=torch.float):
        super(Actor, self).__init__()
        self.device = device
        self.dtype = dtype

        self.max_action = max_action
        self.action_dim = action_dim

        # Input layer 
        self.input = nn.LSTM(state_dim, 128, batch_first=True,
                            num_layers=2,dropout=0.1,device = self.device)
        #fan_in_uniform_init(self.input.bias)
        self.input.flatten_parameters()

        # Hidden Layers
        self.fc1 = nn.Linear(128, 400,device = self.device)
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)

        self.fc2 = nn.Linear(400, 300,device = self.device)
        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)

        # self.rnn3 = nn.GRU(300,50,3,dropout=0.1,device = self.device)
        # #fan_in_uniform_init(self.rnn3.bias)
        # self.rnn3.flatten_parameters()

        # Output Layer
        self.out = nn.Linear(300, action_dim,device = self.device)
        fan_in_uniform_init(self.out.weight)
        fan_in_uniform_init(self.out.bias)
        
        self = self.to(device=device,dtype=dtype)

    def forward(self, x):
        x = x.to(self.device)
        if x.dim() == 1:   x = x.view((1,-1))
        if x.dim() == 2:  x = x.unsqueeze(1)
        
        # Flatten parameters for RNN modules
        self.input.flatten_parameters()
        # self.rnn3.flatten_parameters()

        # Input Layer
        x,_=self.input(x)
        x = x[:, -1, :]  # Use the last output from GRU

        # Hidden Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = x.unsqueeze(1)  # Add batch dimension
        # x,_ = self.rnn3(x)
        # x = x[:, -1, :]  # Use the last output from GRU

        # Output Layer
        x = torch.tanh(self.out(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,device,dtype=torch.float):
        super(Critic, self).__init__()
        self.device = device
        self.dtype = dtype
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input Layer
        self.input = nn.LSTM(input_size= state_dim + action_dim,hidden_size= 128, batch_first=True,
                            num_layers=2,dropout=0.1,device =self.device)
        #fan_in_uniform_init(self.rnn3.bias)
        self.input.flatten_parameters()

        # Hidden Layers
        self.fc1 = nn.Linear(128, 400,device = self.device)
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)
        self.fc2 = nn.Linear(400, 300,device = self.device)
        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)
        # self.rnn3 = nn.GRU(300,50,3,dropout=0.1,device = self.device)
        # self.rnn3.flatten_parameters()
        #fan_in_uniform_init(self.rnn3.bias)

        # Output Layer
        self.out = nn.Linear(300, 1,device = self.device)
        fan_in_uniform_init(self.out.weight)
        fan_in_uniform_init(self.out.bias)

        self = self.to(device=device,dtype = dtype)

    def forward(self, x, u):
        x = x.to(self.device)
        u = u.to(self.device)

        if x.dim() == 1: x = x.view((1,-1))
        if u.dim() == 1:  u = u.view((1,-1))

        if x.dim() == 2:   x = x.unsqueeze(1)
        if u.dim() == 2:  u = u.unsqueeze(1)

        # Flatten parameters for RNN modules
        self.input.flatten_parameters()
        #self.rnn3.flatten_parameters()

        xu = torch.cat([x, u], dim=-1)#.unsqueeze(1)  # Add batch dimension
        xu, _ = self.input(xu)
        x = xu[:, -1, :]  # Use the last output from GRU

        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        # x = x.unsqueeze(1)  # Add batch dimension
        # x,_ = self.rnn3(x)
        # x = x[:, -1, :]  # Use the last output from GRU

        x = self.out(x)
        return x
    
class ReplayBuffer():
    def __init__(self,state_shape=3,action_shape=2, max_size=1e6, device='cpu'):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device

        self.storage = {
            'states': torch.empty((self.max_size, state_shape), dtype=torch.float32, device=device),
            'actions': torch.empty((self.max_size, action_shape), dtype=torch.float32, device=device),
            'next_states': torch.empty((self.max_size, state_shape), dtype=torch.float32, device=device),
            'rewards': torch.empty((self.max_size,), dtype=torch.float32, device=device),
            'dones': torch.empty((self.max_size,), dtype=torch.float32, device=device)
        }

    def add(self, state, action, next_state, reward, done):
        self.storage['states'][self.ptr] = state.to(dtype=torch.float32, device=self.device)
        self.storage['actions'][self.ptr] = action.to(dtype=torch.float32, device=self.device)
        self.storage['next_states'][self.ptr] = next_state.to(dtype=torch.float32, device=self.device)
        self.storage['rewards'][self.ptr] = reward.to(dtype=torch.float32, device=self.device)
        self.storage['dones'][self.ptr] = done.to(dtype=torch.float32, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return (
            self.storage['states'][ind],
            self.storage['actions'][ind],
            self.storage['next_states'][ind],
            self.storage['rewards'][ind],
            self.storage['dones'][ind]
        )
    def getSize(self):
        return self.size
    
    def getLength(self):
        return len(self.storage['states'])

class DDPG():
    def __init__(self, state_dim, action_dim, max_action, device,dtype=torch.float,batch_size = 64,discount = 0.99,buffer_capacity=1e6,tau=0.005,actor_lr=1e-4,critic_lr=1e-3):
        self.device = device
        self.tau = tau
        self.actor_lr  = actor_lr
        self.critic_lr = critic_lr
        self.buffer_capacity = int(buffer_capacity)
        self.device = device
        self.dtype = dtype
        self.max_action = max_action
        self.batch_size= batch_size
        self.discount= discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_dir = './Checkpoint'

        # self.logger = logging.getLogger('ddpg')
        # self.logger.setLevel(logging.INFO)
        # self.logger.addHandler(logging.StreamHandler())

        self.lossFcn = F.mse_loss
        
        self.actor = Actor(state_dim, action_dim, max_action, device,dtype)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_scheduler = lr_scheduler.StepLR(self.actor_optimizer,step_size=100, gamma=0.5)

        self.critic = Critic(state_dim, action_dim, device,dtype)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_scheduler = lr_scheduler.StepLR(self.critic_optimizer,step_size=100, gamma=0.5)

        self.replay_buffer = ReplayBuffer(self.state_dim,self.action_dim,self.buffer_capacity,self.device)
        
    def select_action(self, state:torch.Tensor)->torch.Tensor:
        #with torch.no_grad():
        state = state.to(device=self.device,dtype=self.dtype).reshape(1,-1)
        return self.actor(state).detach()
        #.cpu().data.numpy().flatten()

    def save(self,episode,rank=0):
        torch.save(self.actor.state_dict(), f"./model/Actor/Episode_{episode}_Rank_{rank}.pth")
        torch.save(self.critic.state_dict(), f"./model/Critic/Episode_{episode}_Rank_{rank}.pth")
    
    def load(self,episode,rank=0):
        self.actor.load_state_dict(torch.load(f"./model/Actor/Episode_{episode}_Rank_{rank}.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"./model/Critic/Episode_{episode}_Rank_{rank}.pth", map_location=self.device))
    
    def loadFromModel(self,name):
        self.actor.load_state_dict(torch.load(f"./model/Actor/{name}", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"./model/Critic/{name}", map_location=self.device))

    def train(self):
        # with torch.no_grad():
        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)

        # Move tensors to the correct device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device).view(-1, 1)
        done = done.to(self.device).view(-1, 1)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.discount * target_Q).detach()
        # no grad until here

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = self.lossFcn(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        #with torch.no_grad():
            # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # no_grad until here
        
        return critic_loss.item(),actor_loss.item()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def save_checkpoint(self, last_timestep,rank=0):
        checkpoint_name =self.checkpoint_dir+'/ep_{}_rank{}.pth.tar'.format(last_timestep,rank)
        #self.logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
        }
        #self.logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        #self.logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def get_path_of_latest_file(self):
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_path = self.get_path_of_latest_file()
        else:
            checkpoint_path =self.checkpoint_dir + '/' + checkpoint_name
        
        if os.path.isfile(checkpoint_path):
            #logger.info("Loading checkpoint...({})".format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            #logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep
        else:
            raise OSError('Checkpoint not found')
