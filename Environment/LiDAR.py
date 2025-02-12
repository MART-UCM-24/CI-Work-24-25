import numpy as np
import random
import matplotlib.pyplot as plt
from Environment import GridMap

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
