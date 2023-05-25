import numpy as np


def calculate_circle_waypoints():

    # circular trajectory parameters
    radius = 1.0
    circumference = 2*3.14*1
    time_step = 0.01
    total_time = 100
    num_waypoints = int(total_time / time_step)
    angle_increment = 2 * 3.14 / num_waypoints


    start_point = [[0.0], [0.0], [0.0]]
    waypoints = np.array([start_point])

    for step in range(num_waypoints):
        angle = step * angle_increment
        x = np.array([start_point[0][0] + radius * np.cos(angle)])
        y = np.array([start_point[1][0] + radius * np.sin(angle)])
        z = np.array([start_point[2][0]])
        # print(np.array([x, y, z]))
        waypoints = np.append(waypoints, np.array([[x, y, z]]), axis=0)

    return waypoints


