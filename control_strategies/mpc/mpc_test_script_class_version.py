import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from twelve_states_linear_controller_class_version import MPC_Controller, TVPData, MPC_Model, MPC_Simulator, MPC_Estimator
from utils import m, g

#-----------------------Initialise System------------------------
target_velocity = np.array([0.0, 0.0, 0.0])
control_model = MPC_Model(is_linear=True)
# control_model.mpc_model.setup()

mpc_controller = MPC_Controller(control_model)
mpc_controller.set_bounds(pi/(2.2), 30)
mpc_controller.set_current_input(np.array([m*g/4,m*g/4,m*g/4,m*g/4,0.0,0.0,0.0,0.0]))

tvp = TVPData(mpc_controller.x0, mpc_controller.u0, mpc_controller.z0, target_velocity)

simulation_model = MPC_Model(is_linear=False)
mpc_simulator = MPC_Simulator(simulation_model)

mpc_estimator = MPC_Estimator(simulation_model)

mpc_controller.initial_guess()
mpc_simulator.initial_guess()

dt = .04
curr_roll = 0.0
curr_pitch = 0.0
last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

desired_velocities = np.array([[0.01, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0]])


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
    for i in range(40):
        start = time.time()
        
        u0 = mpc_controller.controller.make_step(x0)

        end = time.time()
        print("Computation time: ", end-start)
        
        ynext= mpc_simulator.simulator.make_step(u0)
        x0 = mpc_estimator.estimator.make_step(ynext)
        print("sim")
        # sim is pos, theta, dpos, dtheta
        # controller is dpos, dtheta, theta, pos 
        drone_acceleration = (np.array(x0[6:12]) - last_x0_dot )/dt
        tvp.x = x0
        tvp.u = u0
        tvp.drone_accel = drone_acceleration
        tvp.target_velocity = [0.3, 0.0, 0.0]
        print("target velocity is ", tvp.target_velocity)
        
        # print("u")
        # print(u0)
        # print("\n")
        # print("x")
        # print(x0)
        # print("\n")
        # print("a")
        # print(drone_acceleration)
        print(i)
        last_x0_dot = np.array(x0[6:12])

fig, ax = plt.subplots()

# TODO: CHANGE THIS BY ADDING MORE FUNCTIONS IN CONTROLLER
t = mpc_controller.controller.data['_time']
x_vel = mpc_controller.controller.data['_x'][:, 6]
y_vel = mpc_controller.controller.data['_x'][:, 7]
z_vel = mpc_controller.controller.data['_x'][:, 8]

roll_graph = mpc_controller.controller.data['_x'][:, 3]
pitch_graph = mpc_controller.controller.data['_x'][:,4]
yaw_graph = mpc_controller.controller.data['_x'][:, 5]


# Plot the data
ax.plot(t, x_vel, label='xv')
ax.plot(t, y_vel, label='yv')
ax.plot(t, z_vel, label='zv')
ax.plot(t, roll_graph, label='r')
ax.plot(t, pitch_graph, label='p')
ax.plot(t, yaw_graph, label='yaw')

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()