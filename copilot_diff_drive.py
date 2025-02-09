import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DifferentialDrivenAGV:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.x = 0
        self.y = 0
        self.theta = 0
        self.left_wheel_speed = 0
        self.right_wheel_speed = 0

    def update_wheel_speeds(self, left_wheel_speed, right_wheel_speed):
        self.left_wheel_speed = left_wheel_speed
        self.right_wheel_speed = right_wheel_speed

    def step(self, dt):
        # Calculate the linear and angular velocities
        v = (self.left_wheel_speed + self.right_wheel_speed) / 2.0
        omega = (self.right_wheel_speed - self.left_wheel_speed) / self.width

        # Update the position and orientation of the AGV
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt

    def get_position(self):
        return (self.x, self.y, self.theta)

# Create a DifferentialDrivenAGV instance
agv = DifferentialDrivenAGV(length=1.0, width=0.5)
agv.update_wheel_speeds(left_wheel_speed=0.5, right_wheel_speed=0.5)

# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)

# Set the limits of the grid map
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Plot the initial position of the AGV
x_positions = [agv.x]
y_positions = [agv.y]

# Function to update the plot for each frame of the animation
def update(frame):
    agv.step(dt=1.0)
    x, y, _ = agv.get_position()
    x_positions.append(x)
    y_positions.append(y)
    ax.plot(x_positions, y_positions, marker='o', color='b')
    return ax

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=range(100), repeat=False)

plt.show()