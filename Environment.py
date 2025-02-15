import random
import torch
import torch.nn as nn
import numpy as np
from matplotlib import animation, patches  # type: ignore
import matplotlib.pyplot as plt # type: ignore

class GridMap:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.map = np.zeros((int(height / resolution), int(width / resolution)))

    def add_obstacle(self, x, y, width, height):
        x_start = int(x / self.resolution)
        y_start = int((self.height - y - height) / self.resolution)  # Adjust for bottom-left origin
        x_end = int((x + width) / self.resolution)
        y_end = int((self.height - y) / self.resolution)  # Adjust for bottom-left origin
        
        # Set the obstacle area to 1 in the map array.
        self.map[y_start:y_end, x_start:x_end] = 1
    
    def is_occupied(self, x, y):
        x_idx = int(x / self.resolution)
        y_idx = int((self.height - y) / self.resolution)
        
        if x_idx < 0 or x_idx >=  int(self.width / self.resolution) or y_idx < 0 or y_idx >= int(self.height / self.resolution):
            print(f"Error: Index out of range. x_idx: {x_idx}, y_idx: {y_idx}")
            result = True  # Assuming out-of-bounds areas are considered occupied
        else:
            result = self.map[y_idx, x_idx] == 1
        return result

    def display(self,ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.map, cmap='gray_r', extent=[0, self.width, 0, self.height])
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        return ax
    
    def find_random_position(self):
        random.seed()
        start = 1_000_000
        while start> 0:
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            if (not self.is_occupied(x, y) and not self.is_occupied(x + 1, y) 
                and not self.is_occupied(x - 1, y) and not self.is_occupied(x, y + 1) 
                and not self.is_occupied(x, y - 1) and not self.is_occupied(x+1,y+1)
                and not self.is_occupied(x +1,y-1) and not self.is_occupied(x-1,y+1)
                and not self.is_occupied(x-1, y-1)):
                return np.array([x, y])
            start -= 1
        raise Exception('No position found')  


class LIDARSensor:
    
    def __init__(self, range, map,angles):
        self.range = range
        self.map = map
        self.angles = angles
    
    def get_readings(self, x, y, theta):
        rads = self.angles + theta  # Convert angles to radians
        distances = np.full(len(rads), self.range)  # Initialize distances with max range
        
        for i, angle in enumerate(rads):
            dx = x + np.arange(1, int(self.range / self.map.resolution)) * self.map.resolution * np.cos(angle)
            dy = y + np.arange(1, int(self.range / self.map.resolution)) * self.map.resolution * np.sin(angle)
            
            for d, (dx_i, dy_i) in enumerate(zip(dx, dy)):
                if self.map.is_occupied(dx_i, dy_i):
                    distances[i] = d * self.map.resolution
                    break
        
        return self.angles, distances  # Convert angles back to degrees

    def display(self, ax, x, y, theta):
        angles, distances = self.get_readings(x, y, theta)
        for angle, distance in zip(angles, distances):
            rad = angle + theta
            end_x = x + distance * np.cos(rad)
            end_y = y + distance * np.sin(rad)
            ax.plot([x, end_x], [y, end_y], 'b-',linewidth=1)
        return ax


class DifferentialDriveAGV:

    mass:float  = 115 # kg
    width:float = 0.81 # m
    length:float = 1.09 # m
    height:float = 0.26 # m
    wheel_radius:float = 0.26/2.0 # m

    # x, y, theta
    pos:torch.Tensor
    dpos:torch.Tensor
    # ddpos: torch.Tensor

    # Unused
    max_speed:list = 2 # m/s
    max_acceleration:list = 0.7 # m/s

    def __init__(self, pos_ini:torch.Tensor,radious:float = 0.26/2.0,width:float=0.81,mass:float  = 115,length:float = 1.09,device='cpu',dtype = torch.float32 ):
        
        self.wheel_radius = radious
        self.device = device
        self.pos = pos_ini
        self.width = width
        self.mass = mass
        self.length = length

        self.dtype = dtype
        
        self.dpos = torch.zeros(3,device=device,dtype = dtype)
        # self.ddpos = torch.zeros(3,1,device=device,dtype = dtype)

        self.forwardK = torch.tensor(data=[[radious/2.0,radious/2.0],[radious/width,-radious/width]],device=device,dtype=dtype)
        self.inverseK = self.forwardK.inverse()

        self.J = 1/12 * mass * ( width*width + length*length )
        self.forwardDyn= torch.tensor(data=[[1/mass,1/mass],[width/self.J,-width/self.J]],device=device,dtype=dtype).mul(1.0/radious)
        self.inverseDyn = self.forwardDyn.inverse()

    def forwardKinematics(self, W_L, W_R)->torch.Tensor:
        angular_speeds = torch.tensor([W_R,W_L],device=self.device,dtype=self.dtype)
        speeds = self.forwardK @ angular_speeds
        return speeds#np.array([speed, angular_speed])

    def inverseKinematics(self, speed, angular_speed)->torch.Tensor:
        speeds = torch.tensor([[speed],[angular_speed]],device=self.device,dtype=self.dtype)
        angular_speeds = self.inverseK @ speeds
        return angular_speeds#np.array([r_angular_speed, l_angular_speed])

    def forwardDynamics(self,t_l,t_r) -> torch.Tensor:
        M_R=torch.tensor(data=[[t_r],[t_l]],dtype=self.dtype,device=self.device)
        acc = self.forwardDyn @ M_R
        return acc
    
    def inverseDynamics(self,acc_L,acc_A) -> torch.Tensor:
        M_R = torch.tensor(data=[[acc_L],[acc_A]],dtype=self.dtype,device=self.device)
        torques = self.inverseDyn @ M_R
        return torques
    
    def move(self, t_l, t_r , dt) -> torch.Tensor:
        # Compute accelerations
        acc = self.forwardDynamics(t_l, t_r)

        # Rotation matrix for the current orientation
        rotation_matrix = torch.tensor([
            [torch.cos(self.pos[2]), -torch.sin(self.pos[2]), 0],
            [torch.sin(self.pos[2]),  torch.cos(self.pos[2]), 0],
            [0, 0, 1]
        ], device=self.device, dtype=self.dtype)

        # Acceleration vector in the local frame
        acc_local = torch.tensor([acc[0], 0, acc[1]], device=self.device, dtype=self.dtype)
        #v_local = torch.tensor([self.dpos[0],0,self.dpos[1]], device=self.device, dtype=self.dtype)

        # Transform acceleration to the global frame
        acc_global = rotation_matrix @ acc_local
        v_global = rotation_matrix @ self.dpos

        # Update position and velocity using in-place operations
        self.pos.add_(v_global * dt + 0.5 * acc_global * dt * dt)
        self.dpos.add_(acc_local * dt)
        
        return self.pos
    
    def getState(self) -> torch.Tensor:
        """
        Returns the current position tensor.
        """
        return self.pos

    def getDerivativeState(self) -> torch.Tensor:
        """
        Returns the current derivative of the position tensor.
        """
        # Rotation matrix for the current orientation
        rotation_matrix = torch.tensor([
            [torch.cos(self.pos[2]), -torch.sin(self.pos[2]), 0],
            [torch.sin(self.pos[2]),  torch.cos(self.pos[2]), 0],
            [0, 0, 1]
        ], device=self.device, dtype=self.dtype)
        v = torch.tensor([self.dpos[0], 0 , self.dpos[1]], device=self.device, dtype=self.dtype)
        return rotation_matrix @ v
    
    def getSpeeds(self)->torch.Tensor:
        return self.dpos

    def setState(self,state:torch.Tensor)->torch.Tensor:
        self.pos = state.to(device=self.device,dtype=self.dtype)
        return self
    
    def resetState(self,state:torch.Tensor):
        self.pos = state.to(device=self.device,dtype=self.dtype)
        self.dpos = torch.zeros_like(self.dpos,device=self.device,dtype=self.dtype)
        return self
    
    def __get_corners__(self)->torch.Tensor:
        # Center position and orientation
        center = self.pos[:2]
        angle = self.pos[2]

        # Define the relative positions of the corners
        half_length = self.length / 2
        half_width = self.width / 2
        corners = torch.tensor([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ], device=self.device, dtype=self.dtype)

        # Rotation matrix
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ], device=self.device, dtype=self.dtype)

        # Rotate and translate corners
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners.add_(center)
        return rotated_corners

    def __get_polygon__(self)->patches.Polygon:
        """
        Returns a matplotlib polygon representing the object.
        """
        corners = self.__get_corners__().cpu().numpy()
        polygon = patches.Polygon(corners, closed=True, edgecolor='r', facecolor='r')
        return polygon

    def __get_wheel_positions__(self)->torch.Tensor:
        center = self.pos[:2]
        width = self.width
        angle_rad = self.pos[2]
        
        # Define the relative positions of the wheels
        half_width = width / 2
        wheel_offsets = torch.tensor([
            [-half_width, half_width],
            [half_width, -half_width]
        ], device=self.device, dtype=self.dtype)
        
        # Rotation matrix
        rotation_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad), torch.cos(angle_rad)]
        ], device=self.device, dtype=self.dtype)
        
        # Rotate and translate wheel positions
        rotated_wheel_offsets = rotation_matrix @ wheel_offsets
        wheel_positions = rotated_wheel_offsets.T + center
        
        #left_wheel, right_wheel = wheel_positions[0], wheel_positions[1]
        
        return wheel_positions

    def display(self, ax=None)->plt.Axes:
        """
        Displays the object and its wheels on a matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        polygon = self.__get_polygon__()
        ax.add_patch(polygon)
        
        left_wheel, right_wheel= self.__get_wheel_positions__()
        ax.add_patch(patches.Circle(left_wheel.item(), self.wheel_radius / 5, color='black'))
        ax.add_patch(patches.Circle(right_wheel.item(), self.wheel_radius / 5, color='black'))
        
        return ax

    def check_collision(self, grid_map)->bool:
        """
        Checks if any of the object's corners are in collision with the grid map.
        """
        for corner in self.__get_corners__():
            if grid_map.is_occupied(corner[0].item(), corner[1].item()):
                return True
        return False
 

class NavigationTask:
    grid_map: GridMap
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.update()

    def update(self):
        self.start_position = self.grid_map.find_random_position()
        self.start_position = np.append(self.start_position,random.uniform(0,2*np.pi))
        start = 100
        self.objective_position = self.grid_map.find_random_position()
        self.objective_position = np.append(self.objective_position,random.uniform(0,2*np.pi))
        while (start > 0 and self.objective_position[0] == self.start_position[0] and 
               self.objective_position[1] == self.start_position[1]):
            self.objective_position = self.grid_map.find_random_position()
            start -= 1

    def display(self,ax=None):
        ax = self.grid_map.display(ax)
        ax.plot(self.start_position[0], self.start_position[1], 'go', label='Start')
        ax.plot(self.objective_position[0], self.objective_position[1], 'ro', label='Objective')
        ax.legend()
        return ax


class DifferentialDriveEnv:
    def __init__(self, grid_map,device = 'cpu',dtype = torch.float):
        self.grid_map = grid_map
        self.task = NavigationTask(grid_map)
        self.robot = DifferentialDriveAGV(pos_ini=torch.tensor(self.task.start_position.transpose()),device=device,dtype=dtype)
        self.lidar = LIDARSensor(range=8,map=grid_map,angles=np.linspace(0,2*np.pi,12))
        self.state_dim = 3  # x, y, theta
        self.action_dim = 2  # torques for left and right wheels
        maxTorque = self.robot.inverseDynamics(self.robot.max_acceleration,0)
        maxT = maxTorque[0].item()
        self.max_action = torch.tensor([maxT,maxT],device=device,dtype=dtype)  # Define the maximum action value
        self.device = device
        self.dtype = dtype

    def reset(self):
        self.task.update()
        self.robot.resetState(torch.tensor(self.task.start_position.transpose(),device=self.device,dtype=self.dtype))
        return self.robot.getState()

    def step(self, action, dt):
        tr,tl= action[0]
        # tl,tr= self.robot.inverseDynamics(acc_L, acc_W)
        # Here another controler would be neccessary
        state = self.robot.move(tl, tr, dt=dt)  # Assuming a time step of 0.1 seconds
        
        # Calculate distance to objective
        distance_to_objective = torch.norm(state[:2] - torch.tensor(self.task.objective_position[:2], device=self.device, dtype=self.dtype))
        done = self.robot.check_collision(self.grid_map) or (distance_to_objective < 0.1).cpu().item()
        
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
        reward += 5 * ((tl+tr) ** 2) - 0.5 * ((tl-tr) ** 2)
        #reward = torch.tensor(reward,device=self.device)
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

