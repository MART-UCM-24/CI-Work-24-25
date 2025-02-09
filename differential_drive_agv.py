import time
from matplotlib import animation, patches
import numpy as np
import matplotlib.pyplot as plt
import SLTMNN

class Wheel:
    def __init__(self, radius, angular_speed=0):
        self.radius = radius
        self.angular_speed = angular_speed

    def setRadius(self, rad):
        self.radius = rad

    def setAngularSpeed(self, w):
        self.angular_speed = w
    
    def setSpeed(self,v):
        self.angular_speed = v/self.radius

    def getSpeed(self):
        return self.radius * self.angular_speed

    def getAngularSpeed(self):
        return self.angular_speed

    def getRadius(self):
        return self.radius

class Differential_Drive_AGV:
    def __init__(self, radius, width, length, angle=0, x=0, y=0):
        self.lWheel = Wheel(radius)
        self.rWheel = Wheel(radius)
        self.width = width
        self.length = length
        self.speed = 0
        self.angle = angle
        self.angular_speed = 0
        self.x = x
        self.y = y

    def forwardKinematics(self, left_angular_speed, right_angular_speed):
        lSpeed = left_angular_speed * self.lWheel.getRadius()
        rSpeed = right_angular_speed * self.rWheel.getRadius()
        speed = (lSpeed + rSpeed) / 2
        angular_speed = (-lSpeed + rSpeed) / self.width
        return np.array([speed, angular_speed])

    def inverseKinematics(self, speed, angular_speed):
        r_angular_speed = (speed + angular_speed * self.width * 0.5) / self.lWheel.getRadius()
        l_angular_speed = (speed - angular_speed * self.width * 0.5) / self.lWheel.getRadius()
        return np.array([r_angular_speed, l_angular_speed])

    def move(self, dt):
        self.x = self.x + self.speed * np.cos(self.angle) * dt
        self.y = self.y + self.speed * np.sin(self.angle) * dt
        self.angle = self.angle + self.angular_speed * dt
        # angle =  self.angle #self.angular_speed * dt
        # radio = self.speed/self.angular_speed
        # self.x = self.x + radio * np.cos(angle) 
        # self.y = self.y + radio * np.sin(angle)
        return np.array([self.x, self.y, self.angle])
    
    def setAngularSpeed(self,wl,wr):
        self.lWheel.setAngularSpeed(wl)
        self.rWheel.setAngularSpeed(wr)
        self.speed, self.angular_speed = self.forwardKinematics(wl,wr)
    
    def setSpeed(self,vl,vr):
        self.lWheel.setSpeed(vl)
        self.rWheel.setSpeed(vr)
        self.speed, self.angular_speed = self.forwardKinematics(self.lWheel.getAngularSpeed(),self.rWheel.getAngularSpeed())

    def setPosition(self, x, y):
        self.x = x
        self.y = y

    def updateSpeeds(self, left_angular_speed, right_angular_speed):
        self.speed, self.angular_speed = self.forwardKinematics(left_angular_speed, right_angular_speed)

    def get_state(self):
        return self.x, self.y, self.angle

    def __get_corners__(self):
        center = [self.x, self.y]
        length = self.length
        width = self.width
        angle = self.angle
        angle_rad = angle
        dx_length = length / 2 * np.cos(angle_rad)
        dy_length = length / 2 * np.sin(angle_rad)
        dx_width = width / 2 * np.sin(angle_rad)
        dy_width = width / 2 * np.cos(angle_rad)
        corners = [
            (center[0] - dx_length - dx_width, center[1] - dy_length + dy_width),
            (center[0] + dx_length - dx_width, center[1] + dy_length + dy_width),
            (center[0] + dx_length + dx_width, center[1] + dy_length - dy_width),
            (center[0] - dx_length + dx_width, center[1] - dy_length - dy_width)
        ]
        return corners

    def __get_polygon__(self):
        # 'none' is a color too
        polygon = patches.Polygon(self.__get_corners__(), closed=True, edgecolor='r', facecolor='r')
        return polygon
    
    def __get_wheel_positions__(self):
        center = [self.x, self.y]
        width = self.width
        angle_rad = self.angle
        dx_width = width / 2 * np.sin(angle_rad)
        dy_width = width / 2 * np.cos(angle_rad)
        
        left_wheel = (center[0] - dx_width, center[1] + dy_width)
        right_wheel = (center[0] + dx_width, center[1] - dy_width)
        
        return left_wheel, right_wheel
    
    def display(self, ax = None):
        if ax is None:
            fig,ax = plt.subplots()
        polygon = self.__get_polygon__()
        ax.add_patch(polygon)
        left_wheel, right_wheel = self.__get_wheel_positions__()
        ax.add_patch(patches.Circle(left_wheel, self.lWheel.getRadius()/5, color='black'))
        ax.add_patch(patches.Circle(right_wheel, self.rWheel.getRadius()/5, color='black'))
        return ax
    
    def check_collision(self, grid_map):
        for corner in self.__get_corners__():
            if grid_map.is_occupied(corner[0], corner[1]):
                return True
        return False

# LIDAR Sensor Class
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
        # for i, angle in enumerate(angles):
        #     rad = angle + theta
        #     for d in range(1, int(self.range / self.map.resolution)):
        #         dx = x + d * self.map.resolution * np.cos(rad)
        #         dy = y + d * self.map.resolution * np.sin(rad)
                
        #         if self.map.is_occupied(dx, dy):
        #             distances[i] = d * self.map.resolution
        #             break
        #         else:
        #             distances[i] = d * self.map.resolution
        #             if distances[i] > self.range:
        #                 distances[i] = self.range
        #                 break
        
        return angles, distances

    def display(self, ax, x, y, theta):
        angles, distances = self.get_readings(x, y, theta)
        for angle, distance in zip(angles, distances):
            rad = angle + theta
            end_x = x + distance * np.cos(rad)
            end_y = y + distance * np.sin(rad)
            ax.plot([x, end_x], [y, end_y], 'b-')
        return ax

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
# Main Function
# if __name__ == "__main__":
#     agv = Differential_Drive_AGV()
#     grid_map = GridMap(width=10, height=10, resolution=0.1)
#     grid_map.add_obstacle(2, 2, 1, 1)
#     grid_map.display()

#     lidar = LIDARSensor(range=5, resolution=1, map=grid_map)
    
#     input_shape = (6,)  # [x, y, theta, v, phi, L]
#     stlm_nn = SLTMNN(input_shape)

#     dt = 0.1  # time step
#     for _ in range(100):
#         v = 1  # constant speed
#         phi = 0.1  # constant steering angle
#         agv.update(v, phi, dt)
        
#         x, y, theta = agv.get_state()
#         readings = lidar.get_readings(x, y, theta)
        
#         print(f"AGV State: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
#         print(f"LIDAR Readings: {readings}")

bitmap = GridMap(20, 20, 0.01)
agv = Differential_Drive_AGV(0.5, 1, 2, np.deg2rad(90), 4, 5)
#ax = agv.display()
agv.setSpeed(0.5,0.4) # 

bitmap.add_obstacle(1, 1, 0.5, 0.5)
bitmap.add_obstacle(2, 2, 0.5, 0.5)
bitmap.add_obstacle(5, 5, 1.5, 1.5)
bitmap.add_obstacle(0,0,1,1)
lidar = LIDARSensor(range=5,map=bitmap,angles=np.linspace(0,2*np.pi,30))


def updateFrame(frame):
    pos= agv.move(dt=0.1)
    ax.clear()
    bitmap.display(ax)
    agv.display(ax)
    lidar.display(ax, pos[0], pos[1], pos[2])
    ax.grid(True)
    print("NEXT FRAME")
    if agv.check_collision(bitmap):
       ani.event_source.stop()
    return ax.patches


fig,ax = plt.subplots()
fps = 100 # time between frame in ms
amountFrames = 300 # amount of frames to call
# If frames is not specified, it is infinite


ani = animation.FuncAnimation(fig, updateFrame, frames=range(amountFrames), repeat=False, interval= fps)
plt.show(block=True)