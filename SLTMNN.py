import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Environment.GridMap import GridMap
from Network.Actor import Actor
from Network.Critic import Critic
from Environment.Wrapper import DifferentialDriveEnv
from Environment.AGV import DifferentialDriveAGV

# Initialize the models
state_dim = 3  # Example state dimension
action_dim = 2  # Example action dimension
max_action = 1.0  # Example max action
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


dtype = torch.float
agv = DifferentialDriveAGV(torch.tensor([[10],[10],[torch.pi/2]],device=device),device=device,dtype=dtype)
agv.fowardDynamics(10,10)

actor = Actor(state_dim, action_dim, max_action, device)
critic = Critic(state_dim, action_dim, device)

# Load the saved state dictionaries
actor.load_state_dict(torch.load("Actor/actor_model_episode_200.pth"))
critic.load_state_dict(torch.load("Critic/critic_model_episode_200.pth"))

# Move the models to the appropriate device
actor.to(device)
critic.to(device)

# MAP
map_1 = GridMap(40, 40, 0.01)

# Walls
map_1.add_obstacle(0, 0, 1, map_1.height)
map_1.add_obstacle(map_1.width-1, 0, 1, map_1.height)
map_1.add_obstacle(0, 0, map_1.width, 1)
map_1.add_obstacle(0, map_1.height-1, map_1.width, 1)

# Obstacles
map_1.add_obstacle(5, 5, 2, 10)
map_1.add_obstacle(10, 15, 3, 5)
map_1.add_obstacle(15, 25, 2, 8)
map_1.add_obstacle(20, 10, 4, 3)
map_1.add_obstacle(25, 30, 3, 6)
map_1.add_obstacle(30, 5, 2, 10)
map_1.add_obstacle(35, 20, 3, 5)
map_1.add_obstacle(5, 30, 2, 8)
map_1.add_obstacle(10, 35, 4, 3)
map_1.add_obstacle(20, 20, 3, 6)
map_1.add_obstacle(25, 5, 2, 10)
map_1.add_obstacle(30, 25, 3, 5)
map_1.add_obstacle(35, 10, 2, 8)

env = DifferentialDriveEnv(grid_map=map_1)

# Run a test episode
state = env.reset()
episode_reward = 0
dt = 0.5

fig, ax = plt.subplots()

def update(frame, ax):
    try:
        global state, episode_reward
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().detach().numpy().flatten()
        print("Action")
        print(action)
        print("State")
        print(state)
        ax.clear()
        ax = env.render(ax)
        next_state, reward, done, _ = env.step(action, dt)
        state = next_state
        episode_reward += reward
        if done:
            ani.event_source.stop()
    except Exception as e:
        print(e)
        ani.event_source.stop()
    return ax

ani = animation.FuncAnimation(fig, update, frames=1200, fargs=(ax,), repeat=False,interval = 10)
plt.show(block=True)

print(f"Test Episode Reward: {episode_reward}")

# class STLMNN(nn.Module):
#     def __init__(self, input_shape):
#         super(STLMNN, self).__init__()
#         self.fc1 = nn.Linear(input_shape, 128)
#         self.gru1= nn.GRU(128,2,2,dropout=0.1)
#         self.fc2 = nn.Conv1d(128, 128)
#         self.fc3 = nn.Linear(128, 2)  # Output layer: [steering angle, velocity]
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.parameters(), lr=0.001)
        

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#     # def create_stlm_nn(self,input_shape):
#     #     model = self.__init__(input_shape)
#     #     criterion = nn.MSELoss()
#     #     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     #     return model, criterion, optimizer

# def reinforcement_learning(env, model:nn.Module, episodes=1000):
#     gamma = 0.95
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         total_reward = 0
        
#         while not done:
#             action = model.predict(state.reshape(1, -1))
#             next_state, reward, done, _ = env.step(action)
            
#             # Update the model using the reward
#             target = reward + gamma * np.amax(model.predict(next_state.reshape(1, -1)))
#             target_f = model.predict(state.reshape(1, -1))
#             target_f[0][np.argmax(action)] = target
#             model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            
#             state = next_state
#             total_reward += reward
        
#         print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

