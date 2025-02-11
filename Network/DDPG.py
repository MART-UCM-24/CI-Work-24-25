import torch
from Network.Actor import Actor
from Network.Critic import Critic
from Network.ReplayBuffer import ReplayBuffer
from torch import optim
import torch.nn as nn

class DDPG:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action, device)
        self.actor_target = Actor(state_dim, action_dim, max_action, device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim, device)
        self.critic_target = Critic(state_dim, action_dim, device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Move tensors to the correct device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

