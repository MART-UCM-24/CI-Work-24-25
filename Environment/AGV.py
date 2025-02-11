import torch
import torch.nn as nn
import numpy as np
from matplotlib import animation, patches  # type: ignore
import matplotlib.pyplot as plt # type: ignore


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
 
