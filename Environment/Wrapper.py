import matplotlib.pyplot as plt
from Environment.NavigationTask import NavigationTask
from Environment.AGV import DifferentialDriveAGV
from Environment.LiDAR import LIDARSensor
import numpy as np
import torch

class DifferentialDriveEnv:
    def __init__(self, grid_map,device = 'cpu',dtype = torch.float):
        self.grid_map = grid_map
        self.task = NavigationTask(grid_map)
        self.robot = DifferentialDriveAGV(pos_ini=torch.tensor(self.task.start_position.transpose()),device=device,dtype=dtype)
        self.lidar = LIDARSensor(range=8,map=grid_map,angles=np.linspace(0,2*np.pi,12))
        self.state_dim = 3  # x, y, theta
        self.action_dim = 2  # torques for left and right wheels
        self.max_action = torch.tensor([0.7,1])  # Define the maximum action value
        self.device = device
        self.dtype = dtype

    def reset(self):
        self.task.update()
        self.robot.resetState(torch.tensor(self.task.start_position.transpose(),device=self.device,dtype=self.dtype))
        return self.robot.getState()

    def step(self, action, dt):
        acc_L, acc_W = action[0]
        tl,tr = self.robot.inverseDynamics(acc_L, acc_W)
        # Here another controler would be neccessary
        state = self.robot.move(tl, tr, dt=dt)  # Assuming a time step of 0.1 seconds
        
        # Calculate distance to objective
        distance_to_objective = torch.norm(state[:2] - torch.tensor(self.task.objective_position[:2], device=self.device, dtype=self.dtype))
        done = self.robot.check_collision(self.grid_map) or distance_to_objective < 0.1
        
        # Calculate reward
        reward:torch.FloatTensor = 0.0
        
        # Reward for getting closer to the objective
        reward += 1000 / (distance_to_objective**2 + 1e-5)  # Add a small value to avoid division by zero
        
        # Penalty for collisions
        if self.robot.check_collision(self.grid_map):
            reward -= 500
         
        # Penalty for being too close to obstacles
        x,y,angle = self.robot.pos
        _, dists = self.lidar.get_readings(x.item(), y.item(), angle.item())
        dists_tensor = torch.tensor(dists, device=self.device, dtype=self.dtype)
        distance_to_nearest_object = dists_tensor.min()
        #if distance_to_nearest_object < 0.5:  # Threshold distance to obstacles
        reward += 0.5*distance_to_nearest_object**2
        
        # Encourage smooth movements
        v_sq = self.robot.dpos[0] ** 2
        w_sq = self.robot.dpos[1] ** 2
        reward += 0.04 * (acc_L ** 2) - 0.3 * (acc_W ** 2) + 1.3 * v_sq - 0.5 * w_sq
        reward = torch.tensor(reward,device=self.device)
        done = torch.tensor(done,device=self.device)
        return state, reward, done, {}

    def render(self,ax = None):
        if(ax is None):
            fig,ax = plt.subplots()
        ax = self.grid_map.display(ax)
        self.task.display(ax)
        self.robot.display(ax)
        x,y,angle = self.robot.getState()
        self.lidar.display(ax,x.item(),y.item(),angle.item())
        return ax