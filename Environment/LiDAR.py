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
        #angles = np.arange(0, 360, self.resolution)
        angles = self.angles
        distances = np.zeros(len(angles))
        for i, angle in enumerate(angles):
            rad = angle + theta
            for d in range(1, int(self.range / self.map.resolution)):
                dx = x + d * self.map.resolution * np.cos(rad)
                dy = y + d * self.map.resolution * np.sin(rad)
                
                if self.map.is_occupied(dx, dy):
                    distances[i] = d * self.map.resolution
                    break
                else:
                    distances[i] = d * self.map.resolution
                    if distances[i] > self.range:
                        distances[i] = self.range
                        break
        
        return angles, distances

    def display(self, ax, x, y, theta):
        angles, distances = self.get_readings(x, y, theta)
        for angle, distance in zip(angles, distances):
            rad = angle + theta
            end_x = x + distance * np.cos(rad)
            end_y = y + distance * np.sin(rad)
            ax.plot([x, end_x], [y, end_y], 'b-',linewidth=1)
        return ax
