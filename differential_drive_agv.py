import random
import time
from matplotlib import animation, patches
import numpy as np
import matplotlib.pyplot as plt
import SLTMNN

class Wheel:
    radius:float = 0.26*2/3
    angular_speed:float = 0 
    angular_acceleration:float = 0
    torque:float = 0

    def __init__(self, radius, angular_speed=0,angular_acceleration=0,torque=0):
        self.radius = radius
        self.angular_speed = angular_speed
        self.angular_acceleration = angular_acceleration
        self.torque = torque

    def setRadius(self, rad):
        self.radius = rad
        return self

    def setAngularSpeed(self, w):
        self.angular_speed = w
        return self
    
    def setSpeed(self,v):
        self.angular_speed = v/self.radius
        return self

    def getSpeed(self):
        return self.radius * self.angular_speed

    def getAngularSpeed(self):
        return self.angular_speed

    def getRadius(self):
        return self.radius
    
    def setTorque(self,torque):
        self.torque = torque
        return self
    def setAngularAcceleration(self,acc):
        self.angular_acceleration = acc
        return self
    def getAngularAcceleration(self):
        return self.angular_acceleration
    def getTorque(self):
        return self.torque
    def getForce(self):
        return self.torque*self.radius
    def setForce(self,F):
        self.torque = F/self.radius
        return self

class Differential_Drive_AGV:

    mass:float  = 115 # kg
    width:float = 0.81 # m
    length:float = 1.09 # m

    max_speed:list = [-2,2] # m/s
    max_acceleration:list = [-0.7,0.7] # m/s

    x:float = 0
    y:float = 0
    theta:float = 0

    linear_speed:float = 0
    linear_acceleration:float = 0

    angular_speed:float = 0
    angular_acceleration = 0

    lWheel:Wheel
    rWheel:Wheel

    def __init__(self, angle, x, y,radious = 0.26*2/3):
        self.lWheel = Wheel(radious)
        self.rWheel = Wheel(radious)
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

    def fowardDynamics(self,t_r=None,t_l=None):
        if(t_r is None or t_l is None):
            t_r = self.rWheel.getTorque()
            t_l = self.lWheel.getTorque()
        J = 1/12 * self.mass * ( self.width*self.width + self.length*self.length  )
        dtype = torch.float32 
        M_L=torch.tensor(data=[[1/self.mass,1/self.mass],[self.width/J,-self.width/J]],dtype=dtype)
        M_R=torch.tensor(data=[[t_r],[t_l]],dtype=dtype)
        acc = 1/self.rWheel.getRadius()*torch.matmul(M_L,M_R)
        return acc
    
    def setTorque(self,tr,tl):
        self.rWheel.setTorque(tr)
        self.lWheel.setTorque(tl)
        return self
    
    def move(self, dt):
        acc = self.fowardDynamics().cpu().numpy()
        linear_acc = acc[0][0]
        angular_acc = acc[1][0]
        self.x = self.x + self.speed*np.cos(self.angle)*dt+0.5*linear_acc*np.cos(self.angle)*dt*dt
        self.y = self.y + self.speed*np.sin(self.angle)*dt+0.5*linear_acc*np.sin(self.angle)*dt*dt
        self.angle = self.angle + self.angular_speed*dt + 0.5*angular_acc*dt*dt
        self.speed = self.speed + linear_acc*dt
        self.angular_speed = self.angular_acceleration + angular_acc*dt
        return np.array([self.x, self.y, self.angle])
    
    def getState(self):
        return np.array([self.x, self.y, self.angle])
    
    def getDerivateState(self):
        return np.array([self.speed*np.cos(self.angle),self.speed*np.sin(self.angle), self.angular_speed])
    
    # def setAngularSpeed(self,wl,wr):
    #     self.lWheel.setAngularSpeed(wl)
    #     self.rWheel.setAngularSpeed(wr)
    #     self.speed, self.angular_speed = self.forwardKinematics(wl,wr)
    #     return self
    
    # def setSpeed(self,vl,vr):
    #     self.lWheel.setSpeed(vl)
    #     self.rWheel.setSpeed(vr)
    #     self.speed, self.angular_speed = self.forwardKinematics(self.lWheel.getAngularSpeed(),self.rWheel.getAngularSpeed())
    #     return self

    # def setPosition(self, x, y):
    #     self.x = x
    #     self.y = y
    #     return self

    # def getPosition(self):
    #     return self.x,self.y 
    
    # def getSpeed(self):
    #     return self.speed,self.angular_speed
    
    # def updateSpeeds(self, left_angular_speed, right_angular_speed):
    #     self.speed, self.angular_speed = self.forwardKinematics(left_angular_speed, right_angular_speed)
    #     return self
    
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
                return x, y
            start -= 1
        raise Exception('No position found')  

class NavigationTask:
    grid_map: GridMap
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.update()

    def update(self):
        self.start_position = self.grid_map.find_random_position()
        self.start_position.append(random.uniform(0,2*np.pi))
        start = 100
        self.objective_position = self.grid_map.find_random_position()
        self.objective_position.append(random.uniform(0,2*np.pi))
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

bitmap = GridMap(40, 40, 0.01)


# Walls
bitmap.add_obstacle(0, 0, 1, bitmap.height)
bitmap.add_obstacle(bitmap.width-1, 0, 1, bitmap.height)
bitmap.add_obstacle(0, 0, bitmap.width,1)
bitmap.add_obstacle(0, bitmap.height-1, bitmap.width,1)

# Obstacles
bitmap.add_obstacle(5, 5, 2, 10)
bitmap.add_obstacle(10, 15, 3, 5)
bitmap.add_obstacle(15, 25, 2, 8)
bitmap.add_obstacle(20, 10, 4, 3)
bitmap.add_obstacle(25, 30, 3, 6)
bitmap.add_obstacle(30, 5, 2, 10)
bitmap.add_obstacle(35, 20, 3, 5)
bitmap.add_obstacle(5, 30, 2, 8)
bitmap.add_obstacle(10, 35, 4, 3)
bitmap.add_obstacle(20, 20, 3, 6)
bitmap.add_obstacle(25, 5, 2, 10)
bitmap.add_obstacle(30, 25, 3, 5)
bitmap.add_obstacle(35, 10, 2, 8)

navigation_task = NavigationTask(bitmap)

lidar = LIDARSensor(range=5,map=bitmap,angles=np.linspace(0,2*np.pi,12))
agv = Differential_Drive_AGV(0.5, 0.81, 1.09, np.deg2rad(90), navigation_task.start_position[0], navigation_task.start_position[1])

#ax = agv.display()
agv.setTorque(0.5,0.5) # 


# Display the map with start and objective positions

ax = navigation_task.display()
ax = agv.display(ax)
lidar.display(ax,agv.x,agv.y,agv.angle)
#bitmap.display()


def updateFrame(frame):
    pos= agv.move(dt=0.1)
    ax.clear()
    navigation_task.display(ax)
    agv.display(ax)
    lidar.display(ax, pos[0], pos[1], pos[2])
    ax.grid(False)
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