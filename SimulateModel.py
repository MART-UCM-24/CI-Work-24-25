import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Network import *
from Environment import *
from Maps import MapConstructor

# Create environment and agent
dev = 'cpu'
map = MapConstructor.construct('Map_1')
env = DifferentialDriveEnv(map, dev, torch.float32)
ddpg = DDPG(env.state_dim, env.action_dim, env.max_action, dev, torch.float32)
ddpg.loadFromModel('Episode_4999_Rank_0.pth')

# Initialize the environment and get the initial state
state = env.reset()

# Enable interactive mode
plt.ion()

# Create the figure and axis for the animation
fig, ax = plt.subplots()
env.task.display(ax)

# Initialize plot elements
robot_patch = env.robot.__get_polygon__()
ax.add_patch(robot_patch)
lidar_lines = []
for _ in range(360):  # Assuming 360 LiDAR readings
    line, = ax.plot([], [], 'b-', linewidth=1)
    lidar_lines.append(line)

# Animation update function
def update(frame):
    global state, ddpg, env
    action = ddpg.select_action(state)
    print(state)
    print(action)
    state, reward, done, _ = env.step(action, 0.1)

    robot_patch.set_xy(env.robot.__get_corners__().cpu().numpy())

    x, y, angle = env.robot.getState().cpu().numpy()
    angles, distances = env.lidar.get_readings(x, y, angle)
    for line, dist, ang in zip(lidar_lines, distances, angles):
        rad = ang + angle
        end_x = x + dist * np.cos(rad)
        end_y = y + dist * np.sin(rad)
        line.set_data([x, end_x], [y, end_y])

    if done:
        ani.event_source.stop()

    return [robot_patch] + lidar_lines

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=600*2, interval=500, blit=True,repeat = False)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('AGV Movement Animation')

# Keep updating the plot window
plt.pause(0.1)
plt.ioff()  # Turn off interactive mode
plt.show()