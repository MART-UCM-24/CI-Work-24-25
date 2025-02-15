import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from Maps import *
from Environment import *
from Network import *
import time
import numpy as np
import torch.profiler

def evaluate(agent, env, max_steps, dt):
    state = env.reset()
    episode_reward = 0
    step = 0
    while step < max_steps:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action, dt)
        state = next_state
        episode_reward += reward.cpu().numpy()

        if done.cpu().numpy().item():
            break
        step += 1

    return episode_reward


def train(rank,eval_interval,save_interval,map_name='Map_1',episodes=1000,dt=0.5,simulation_time = 60
          ,training_size=1000,checkpoint_interval=40,batch_size=128,load_checkpoint='None',load_model='None'):
    
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    torch.set_default_device(device)
    
    # Initialize TensorBoard
    log_dir = f'runs/DDPG_{rank}'
    writer = SummaryWriter(log_dir=log_dir)

    # Set the random seed for reproducibility
    seed =int (time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.random.seed(seed)

    map = MapConstructor.construct(map_name)
    env = DifferentialDriveEnv(grid_map=map, device=device)
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action
    max_steps = int(simulation_time / dt)

    ddpg_agent = DDPG(state_dim, action_dim, max_action, device,batch_size=batch_size)
    if load_model is not None and load_model.lower() != 'none':
        ddpg_agent.loadFromModel(load_model)
    elif load_checkpoint is not None and load_checkpoint.lower() != 'none':
        ddpg_agent.load_checkpoint(load_checkpoint)

    episode_rewards = []
    eval_rewards = []
    sTime = time.time()
    print(f"Time of start {sTime} - Rank: {rank}")
    episode = 0
    try:
        while episode < episodes:
            state = env.reset()
            episode_reward = 0
            step = 0
            while step < max_steps:
                action = ddpg_agent.select_action(state)
                next_state, reward, done, _ = env.step(action, dt)
                ddpg_agent.replay_buffer.add(state, action, next_state, reward, done)

                state = next_state
                episode_reward += reward.cpu().numpy()

                if done.cpu().numpy().item():
                    break
                step += 1
            
            episode_rewards.append(episode_reward)  # Save the reward for this episode
            writer.add_scalar('Episode_Rewards', episode_reward, episode)  # Log reward to TensorBoard

            if ddpg_agent.replay_buffer.getSize() > training_size:
                critic_loss,actor_loss=ddpg_agent.train() 
                writer.add_scalar('Critic_Loss', critic_loss, episode)  # Log critic loss to TensorBoard
                writer.add_scalar('Actor_Loss', actor_loss, episode)  # Log actor loss to TensorBoard
            # Evaluate the agent every eval_interval episodes
           
            if eval_interval > 0 and (episode + 1) % eval_interval == 0:
                try:
                    eval_reward = evaluate(ddpg_agent, env, max_steps, dt)
                    eval_rewards.append(eval_reward)
                    writer.add_scalar('Evaluation_Rewards', eval_reward, episode)
                    print(f"Episode: {episode}, Evaluation Reward: {eval_reward}, Critic Loss: {critic_loss}, Actor Loss: {actor_loss}")
                except Exception as e:
                    print(f"While evaluating episode {episode}: {e}")   
            
            if save_interval > 0 and (episode + 1) % save_interval == 0:
                ddpg_agent.save(episode=episode,rank=rank)
            
            if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
                ddpg_agent.save_checkpoint(episode,rank)
            
            episode +=1
                
        eTime = time.time()
        print(f"Time of finish {eTime} - Rank: {rank}")
        print(f"Duration of training {eTime-sTime} - Rank: {rank}")
        ddpg_agent.save('FINAL',rank)
        np.savetxt(f"episode_rewards_final_rank_{rank}_seed_{seed}.txt", episode_rewards)
        np.savetxt(f"eval_rewards_final_rank_{rank}_seed_{seed}.txt", eval_rewards)

    except KeyboardInterrupt:
        print(f"Training interrupted. Performing cleanup... - Rank: {rank}")
        ddpg_agent.save('INTERRUPTED',rank)
        np.savetxt(f"episode_rewards_interrupted_rank_{rank}_seed_{seed}.txt", episode_rewards)
        np.savetxt(f"eval_rewards_interrupted_rank_{rank}_seed_{seed}.txt", eval_rewards)

    finally:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Map_1', help='Select the map to be trained, default Map_1')
    parser.add_argument('--n_process', type=int, default=1, help='Number of processes to run, default 1')
    parser.add_argument('--eval_interval', type=int, default=-1, help='Evaluation interval, default -1 -> Do not save')
    parser.add_argument('--save_interval', type=int, default=50, help='Save interval, default 50')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes, default 1000')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size, default 0.1')
    parser.add_argument('--sim_time', type=float, default=60*2, help='Simulation time in seconds, default 120')
    parser.add_argument('--training_size', type=int, default=1500, help='Training size, default 1500')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='checkpoint_interval, default -1 -> Do not save')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size, default 128')
    parser.add_argument('--load_checkpoint',type=str,default='None',help='Name of the checkpoint to start from')
    parser.add_argument('--load_model',type=str,default='None',help='Name of the model to start from')


    args = parser.parse_args()
    if args.dt > 0.15:
        raise Exception("The sampling time must be under 0.12 seconds")
    if args.n_process <= 0:
        raise Exception("The amount of parallel processes must be positive")


    if args.n_process == 1:
        train(0,args.eval_interval, args.save_interval, args.map, args.episodes, args.dt, args.sim_time,
               args.training_size,args.checkpoint_interval,args.batch_size,args.load_checkpoint,args.load_model)
    elif args.n_process > 1:
        mp.spawn(train, args=(args.eval_interval, args.save_interval, args.map, args.episodes, args.dt, args.sim_time, 
                args.training_size,args.checkpoint_interval,args.batch_size,args.load_checkpoint,args.load_model),
                nprocs=args.n_process, join=True)
    
