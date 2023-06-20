import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from global_vars_mpc import tvp
from global_vars_mpc import global_simulator
from global_vars_mpc import mpc_global_controller


with open("control_strategies/mpc/12_states_linear_controller.py") as f:
    exec(f.read())

with open("control_strategies/mpc/12_states_nonlin_sim.py") as f:
    exec(f.read())


mpc_controller = mpc_global_controller.controller
simulator = global_simulator.sim
estimator = global_simulator.est

mpc_controller.set_initial_guess()
simulator.set_initial_guess()

dt = .04
curr_roll = 0.0
curr_pitch =0.0
last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

desired_velocities = np.array([[0.0, 0.0, 0.2],[0.0, 0.0, 0.0],[0.2, 0.0, 0.0],[0.5, 0.0, 0.0],[0.1, 0.0, 0.0],[0.0, -0.1, 0.0], [0.0,-0.5,0.0], [0.0,-0.1,0.0],[0.1,0.0,0.0], [0.5,0.0,0.0]  ])


# print("u")
# print(tvp.u)
# print("\n")
# print("x")
# print(tvp.x)
# print("\n")
# print("a")
# print(tvp.drone_accel)
# print("\n")

for target_vel in desired_velocities:
    tvp.target_velocity = target_vel
    for i in range(20):
        start = time.time()
        
        u0 = mpc_controller.make_step(x0)

        end = time.time()
        print("Computation time: ", end-start)
        
        ynext= simulator.make_step(u0)
        x0 = estimator.make_step(ynext)
        print("sim")
        # sim is pos, theta, dpos, dtheta
        # controller is dpos, dtheta, theta, pos 
        drone_acceleration = (np.array(x0[6:12]) - last_x0_dot )/dt
        tvp.x = x0
        tvp.u = u0
        tvp.drone_accel = drone_acceleration
        #tvp.target_velocity = [0.3, 0.0, 0.0]
        print("target velocity is ", tvp.target_velocity)
        
        print("u")
        print(u0)
        # print("\n")
        # print("x")
        # print(x0)
        # print("\n")
        # print("a")
        # print(drone_acceleration)
        print(i)
        last_x0_dot = np.array(x0[6:12])

fig, ax = plt.subplots()

t = mpc_controller.data['_time']
x_vel = mpc_controller.data['_x'][:, 6]
y_vel = mpc_controller.data['_x'][:, 7]
z_vel = mpc_controller.data['_x'][:, 8]

roll_graph = mpc_controller.data['_x'][:, 3]
pitch_graph = mpc_controller.data['_x'][:,4]
yaw_graph = mpc_controller.data['_x'][:, 5]

til1 =mpc_controller.data['_u'][:, 4]
til2 = mpc_controller.data['_u'][:, 5] 
til3 = mpc_controller.data['_u'][:, 6]
til4 = mpc_controller.data['_u'][:, 7]

# Plot the data
ax.plot(t, x_vel, label='xv')
ax.plot(t, y_vel, label='yv')
ax.plot(t, z_vel, label='zv')
ax.plot(t, roll_graph, label='r')
ax.plot(t, pitch_graph, label='p')
ax.plot(t, yaw_graph, label='yaw')

# ax.plot(t, til1, label='til1')
# ax.plot(t, til2, label='til2')
# ax.plot(t, til3, label='til3')
# ax.plot(t, til4, label='til4')


# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()