import torch as tr
import numpy as np
import matplotlib as plot
import math as mt

v = 10 # AGV Speed
theta = mt.radians(45) # angle between X and the AGV.
beta = mt.radians(3) # angle of the motorize wheel respective to the vehicle
vs = 2 # Speed of the motorized wheel 
L = 1 # length of the AGV

# Speed of the AGV depending on the speed of the motorized wheel
v = vs*mt.cos(beta)
w = vs/L * mt.sin(beta)

# AGV triciclo cinematica
dx = v*mt.cos(theta) 
dy = v*mt.sin(theta)
Rcurva = L*mt.tan(beta)