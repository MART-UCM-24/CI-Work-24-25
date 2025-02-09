import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class STLMNN(nn.Module):
    def __init__(self, input_shape):
        super(STLMNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.gru1= nn.GRU(128,2,2,dropout=0.1)
        self.fc2 = nn.Conv1d(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output layer: [steering angle, velocity]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # def create_stlm_nn(self,input_shape):
    #     model = self.__init__(input_shape)
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     return model, criterion, optimizer


# Reinforcement Learning Function
def reinforcement_learning(env, model:nn.Module, episodes=1000):
    gamma = 0.95
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = model.predict(state.reshape(1, -1))
            next_state, reward, done, _ = env.step(action)
            
            # Update the model using the reward
            target = reward + gamma * np.amax(model.predict(next_state.reshape(1, -1)))
            target_f = model.predict(state.reshape(1, -1))
            target_f[0][np.argmax(action)] = target
            model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")