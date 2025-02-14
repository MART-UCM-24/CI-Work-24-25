import numpy as np
import random
import matplotlib.pyplot as plt

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
        y_idx = int( (self.height - y) / self.resolution)
        return self.map[y_idx, x_idx] == 1

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
