import torch
import torch.nn as nn
import numpy as np
from matplotlib import animation, patches  # type: ignore
import matplotlib.pyplot as plt # type: ignore


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
    max_speed:list = [-2,2] # m/s
    max_acceleration:list = [-0.7,0.7] # m/s

    def __init__(self, pos_ini:torch.Tensor,radious:float = 0.26/2.0,width:float=0.81,mass:float  = 115,length:float = 1.09,device='cpu',dtype = torch.float32 ):
        
        self.wheel_radius = radious
        self.device = device
        self.pos = pos_ini
        self.width = width
        self.mass = mass
        self.length = length

        self.dtype = dtype
        
        self.dpos = torch.zeros(3,1,device=device,dtype = dtype)
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
        self.dpos = torch.zeros_like(self.pos,device=self.device,dtype=self.dtype)
        self.ddpos = torch.zeros_like(self.pos,device=self.device,dtype=self.dtype)
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
    # def __get_corners__(self):
    #     pos = self.pos.cpu().numpy()
    #     center = [pos[0], pos[1]]
    #     length = self.length
    #     width = self.width
    #     angle = pos[2]
    #     dx_length = length / 2 * np.cos(angle)
    #     dy_length = length / 2 * np.sin(angle)
    #     dx_width = width / 2 * np.sin(angle)
    #     dy_width = width / 2 * np.cos(angle)
    #     corners = [
    #         (center[0] - dx_length - dx_width, center[1] - dy_length + dy_width),
    #         (center[0] + dx_length - dx_width, center[1] + dy_length + dy_width),
    #         (center[0] + dx_length + dx_width, center[1] + dy_length - dy_width),
    #         (center[0] - dx_length + dx_width, center[1] - dy_length - dy_width)
    #     ]
    #     return corners

    # def __get_polygon__(self):
    #     # 'none' is a color too
    #     polygon = patches.Polygon(self.__get_corners__(), closed=True, edgecolor='r', facecolor='r')
    #     return polygon
    
    # def __get_wheel_positions__(self):
    #     pos = self.pos.cpu().numpy()
    #     center = [pos[0], pos[1]]
    #     width = self.width
    #     angle_rad = pos[2]
    #     dx_width = width / 2 * np.sin(angle_rad)
    #     dy_width = width / 2 * np.cos(angle_rad)
        
    #     left_wheel = (center[0] - dx_width, center[1] + dy_width)
    #     right_wheel = (center[0] + dx_width, center[1] - dy_width)
        
    #     return left_wheel, right_wheel
    
    # def display(self, ax = None):
    #     if ax is None:
    #         fig,ax = plt.subplots()
    #     polygon = self.__get_polygon__()
    #     ax.add_patch(polygon)
    #     left_wheel, right_wheel = self.__get_wheel_positions__()
    #     ax.add_patch(patches.Circle(left_wheel, self.wheel_radius/5, color='black'))
    #     ax.add_patch(patches.Circle(right_wheel, self.wheel_radius/5, color='black'))
    #     return ax
    
    # def check_collision(self, grid_map):
    #     for corner in self.__get_corners__():
    #         if grid_map.is_occupied(corner[0], corner[1]):
    #             return True
    #     return False
 
