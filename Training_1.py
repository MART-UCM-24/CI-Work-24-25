import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from Environment.GridMap import GridMap
from Environment.Wrapper import DifferentialDriveEnv
from Network.ReplayBuffer import ReplayBuffer
from Network.DDPG import DDPG
import matplotlib.pyplot as plt
import numpy as np

def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, backend):
    setup(rank, world_size, backend)
    
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    elif backend == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Enable interactive mode
    plt.ion()

    # Initialize a list to store rewards
    episode_rewards = []

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    line, = ax.plot(episode_rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards Over Time')

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
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action

    replay_buffer = ReplayBuffer()
    ddpg_agent = DDPG(state_dim, action_dim, max_action, device)
    ddpg_agent.actor = DDP(ddpg_agent.actor, device_ids=[rank] if backend == "nccl" else None)
    ddpg_agent.critic = DDP(ddpg_agent.critic, device_ids=[rank] if backend == "nccl" else None)
    
    dt = 0.1  # seconds
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0

        for t in range(1200):
            action = ddpg_agent.select_action(state)
            next_state, reward, done, _ = env.step(action, dt)
            replay_buffer.add((state, action, next_state, reward, done))

            state = next_state
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)  # Save the reward for this episode
        if len(replay_buffer.storage) > 1000:
            ddpg_agent.train(replay_buffer, batch_size=64)

        print(f"Rank {rank}, Episode: {episode}, Reward: {episode_reward}")
        # Save the model after each episode
        torch.save(ddpg_agent.actor.module.state_dict(), f"Actor/actor_model_episode_{episode}_rank_{rank}.pth")
        torch.save(ddpg_agent.critic.module.state_dict(), f"Critic/critic_model_episode_{episode}_rank_{rank}.pth")
        # Update the plot
        line.set_ydata(episode_rewards)
        line.set_xdata(range(len(episode_rewards)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

    np.savetxt(f"episode_rewards_rank_{rank}.txt", episode_rewards)
    # Keep the plot open after training
    plt.ioff()
    plt.show()

    cleanup()

if __name__ == "__main__":
    if torch.cuda.is_available():
        backend = "nccl"
        world_size = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        backend = "gloo"  # MPS uses the Gloo backend
        world_size = 1  # MPS currently supports only single-device training
    else:
        backend = "gloo"
        world_size = 1

    mp.spawn(train, args=(world_size, backend), nprocs=world_size, join=True)