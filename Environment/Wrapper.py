import matplotlib.pyplot as plt
from Environment.NavigationTask import NavigationTask
from Environment.AGV import Differential_Drive_AGV
from Environment.LiDAR import LIDARSensor
import numpy as np

class DifferentialDriveEnv:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.task = NavigationTask(grid_map)
        self.robot = Differential_Drive_AGV(angle=self.task.start_position[2], x=self.task.start_position[0], y=self.task.start_position[1])
        self.lidar = LIDARSensor(range=8,map=grid_map,angles=np.linspace(-np.pi/2,np.pi/2,4))
        self.state_dim = 3  # x, y, theta
        self.action_dim = 2  # torques for left and right wheels
        self.max_action = 1.0  # Define the maximum action value

    def reset(self):
        self.task.update()
        self.robot = Differential_Drive_AGV(angle=self.task.start_position[2], x=self.task.start_position[0], y=self.task.start_position[1])
        return self.robot.getState()

    def step(self, action,dt):
        tr, tl = action
        self.robot.setTorque(tr, tl)
        state = self.robot.move(dt=dt)  # Assuming a time step of 0.1 seconds
        
        distance_to_objective = np.linalg.norm(state[:2] - self.task.objective_position[:2])
        done = self.robot.check_collision(self.grid_map) or distance_to_objective < 0.1
        # calculate reward
        distance_to_objective = distance_to_objective*distance_to_objective
        _,dists = self.lidar.get_readings(self.robot.x,self.robot.y,self.robot.angle)
        distance_to_nearest_object = dists.min()
        distance_to_nearest_object = distance_to_nearest_object*distance_to_nearest_object
        t_sum = tr+tr
        t_sum = t_sum*t_sum
        t_diff = tr-tl
        t_diff = t_diff*t_diff
        reward = 5/(distance_to_objective + 1) + 0.005*distance_to_nearest_object + 1.3*t_sum - 0.5*t_diff
        return state, reward, done, {}

    def render(self):
        ax = self.grid_map.display()
        self.task.display(ax)
        self.robot.display(ax)
        plt.show()