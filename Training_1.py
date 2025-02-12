from Environment.GridMap import GridMap
from Environment.Wrapper import DifferentialDriveEnv
from Network.ReplayBuffer import ReplayBuffer
from Network.DDPG import DDPG
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    vicedevice = torch.device("cuda")  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple MPS GPU
else:
    device = torch.device("cpu")  # Fallback to CPU

torch.set_default_device(device)

# Enable interactive mode
plt.ion()

# Initialize a list to store rewards
episode_rewards = []
mean_rewards = []

# Create a figure and axis for the plot
fig, ax = plt.subplots()
line, = ax.plot(episode_rewards, label='Episode Reward')
mean_line, = ax.plot(mean_rewards, label='Mean Reward (every 50 episodes)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Episode Rewards Over Time')
ax.legend()


# MAP
map_1 = GridMap(40, 40, 0.01)


# Walls
map_1.add_obstacle(0, 0, 1, map_1.height)
map_1.add_obstacle(map_1.width-1, 0, 1, map_1.height)
map_1.add_obstacle(0, 0, map_1.width,1)
map_1.add_obstacle(0, map_1.height-1, map_1.width,1)

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

#map_1.display()


env = DifferentialDriveEnv(grid_map=map_1)
state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

replay_buffer = ReplayBuffer()
ddpg_agent = DDPG(state_dim, action_dim, max_action,device )
episode_rewards = []
dt = 0.1 # seconds
for episode in range(1000):
    state = env.reset()
    episode_reward = 0

    for t in range(1200):
        action = ddpg_agent.select_action(state)
        next_state, reward, done, _ = env.step(action,dt)
        replay_buffer.add((state, action, next_state, reward, done))

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
    # Save the model after each episode
    if episode%100 == 0:
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
# Keep the plot open after training
plt.ioff()
plt.show()


# # Load the rewards from the file
# episode_rewards = np.loadtxt("episode_rewards.txt")

# # Plot the rewards
# plt.plot(episode_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Episode Rewards Over Time')
# plt.show()

# print(f"End of episode")

print(f"End of program")