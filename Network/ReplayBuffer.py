import torch
import torch.nn as nn
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.array(state))
            batch_actions.append(np.array(action))
            batch_next_states.append(np.array(next_state))
            batch_rewards.append(np.array(reward))
            batch_dones.append(np.array(done))

        return (
            torch.FloatTensor(np.array(batch_states)),
            torch.FloatTensor(np.array(batch_actions)),
            torch.FloatTensor(np.array(batch_next_states)),
            torch.FloatTensor(np.array(batch_rewards)),
            torch.FloatTensor(np.array(batch_dones))
        )
