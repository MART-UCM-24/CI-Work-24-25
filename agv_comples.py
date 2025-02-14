import numpy as np 

class AGVModel:
    def __init__(self, x=0, y=0, theta=0, v=0, L=1):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L
    
    def update(self, v, phi, dt):
        self.v = v
        self.theta += (v / self.L) * np.tan(phi) * dt
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
    
    def get_state(self):
        return self.x, self.y, self.theta
    
class Wheel:
    radio: float
    angular_speed: float
    def __init__(self, radious, angular_speed=0):
        self.radio = radious
        self.angular_speed = angular_speed
    def setRadious(self,rad):
        self.radio = rad
    def setAngularSpeed(self,w):
        self.angular_speed = w
    def getSpeed(self):
        return self.radio * self.angular_speed
    def getAngularSpeed(self):
        return self.angular_speed
    def getRadious(self):
        return self.radio
    
class Driver:
    rightWheel: Wheel
    leftWheel: Wheel
    width:float
    angle:float
    def __init__(self,rad,width,angle=0):
        self.rightWheel = Wheel(rad)
        self.leftWheel = Wheel(rad)
        self.width=width
        self.angle= angle
    def getSpeed(self):
        return 0.5*(self.rightWheel.getSpeed()+self.leftWheel.getSpeed())
    def getAngularSpeed(self):
        return (1/self.width)*(self.leftWheel.getSpeed()-self.rightWheel.getSpeed())
    def getRadioOfTurn(self):
        return self.getAngularSpeed()/self.getSpeed()

class Robot:
    driver:Driver
    length:float
    width: float
    angle: float
    def __init__(self, length, width,angle,rad,driver_angle):
        self.driver=Driver(rad,width,driver_angle)
        self.length=length
        self.width=width
        self.angle=angle
    def calculateSpeed(self,time):
        return self.driver.getSpeed()*time
    def calculatePos(self,time):
        x=self.driver.length*np.cos(self.angle)