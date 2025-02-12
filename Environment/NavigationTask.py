from Environment import GridMap
import numpy as np
import random
import matplotlib.pyplot as plt

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
        #ax.legend()
        return ax