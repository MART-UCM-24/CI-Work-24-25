from Environment.Map_1 import Map_1
from Environment.Wrapper import DifferentialDriveEnv
from Network.ReplayBuffer import ReplayBuffer
from Network.DDPG import DDPG
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
torch.set_default_device(device)





# Initialize a list to store rewards
mean_rewards = []
episode_rewards = []
current_rewards = []

# Enable interactive mode
plt.ion()
# Create a figure and axis for the plot
fig, ax = plt.subplots()
line, = ax.plot(episode_rewards, label='Episode Reward')
mean_line, = ax.plot(mean_rewards, label='Mean Reward (every 50 episodes)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Episode Rewards Over Time')
ax.legend()


map_1 = Map_1()

env = DifferentialDriveEnv(grid_map=map_1)
state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action
episodes = 500
dt = 0.1 # seconds
simulation_time = 60 # seconds
max_steps = int(simulation_time / dt)


replay_buffer = ReplayBuffer()
ddpg_agent = DDPG(state_dim, action_dim, max_action,device )


counter = 0
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    current_rewards.append([])

    for t in range(max_steps):
        action = ddpg_agent.select_action(state)
        next_state, reward, done, _ = env.step(action,dt)
        replay_buffer.add((state, action, next_state, reward, done))
        
        current_rewards[len(current_rewards)-1].append(reward)
        
        state = next_state
        episode_reward += reward

        if done:
            break
    
    episode_rewards.append(episode_reward)  # Save the reward for this episode
        # Calculate mean reward every 50 episodes

    if (episode + 1) % 50 == 0:
        mean_reward = np.mean(episode_rewards[-50:])
        mean_rewards.append(mean_reward)
    else:
        mean_rewards.append(mean_rewards[-1] if mean_rewards else episode_reward)

    if len(replay_buffer.storage) > 1000:
        ddpg_agent.train(replay_buffer, batch_size=64)

    print(f"Episode: {episode}, Reward: {episode_reward}")
    counter += 1
    # Save the model after each episode
    if counter > 100:
        counter = 0
        torch.save(ddpg_agent.actor.state_dict(), f"Actor/actor_model_episode_{episode}.pth")
        torch.save(ddpg_agent.critic.state_dict(), f"Critic/critic_model_episode_{episode}.pth")
    # Update the plot
    line.set_ydata(episode_rewards)
    line.set_xdata(range(len(episode_rewards)))
    mean_line.set_ydata(mean_rewards)
    mean_line.set_xdata(range(len(mean_rewards)))
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)  # Pause to update the plot
    
np.savetxt("episode_rewards.txt", episode_rewards)
np.savetxt("episode_mean_50_rewards.txt", mean_rewards)
np.savetxt("all_rewards.txt", current_rewards)
torch.save(ddpg_agent.actor.state_dict(), f"Actor/actor_model_episode_{'FINAL'}.pth")
torch.save(ddpg_agent.critic.state_dict(), f"Critic/critic_model_episode_{'FINAL'}.pth")
# # Keep the plot open after training
# plt.ioff()
# plt.show()


# # Load the rewards from the file
# episode_rewards = np.loadtxt("episode_rewards.txt")

# Plot the rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.show()

print(f"End of program")