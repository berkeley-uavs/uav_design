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



with open("control_strategies/mpc/12_states_lin_sim.py") as f:
    exec(f.read())
linear_simulator = global_simulator.sim
linear_estimator = global_simulator.est

with open("control_strategies/mpc/12_states_nonlin_sim.py") as f:
    exec(f.read())
nonlinear_simulator = global_simulator.sim
nonlinear_estimator = global_simulator.est

linear_simulator.set_initial_guess()
nonlinear_simulator.set_initial_guess()

dt = .04
curr_roll = 0.0
curr_pitch =0.0
last_x0_dot_lin = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

difference_list_pos = []
difference_list_euler_angs = []
difference_list_lin_vel = []
difference_list_body_rates = []

n = 40
# This represents one step of the MPC. Aka, if the MPC horizon was n steps
# This is technically open loop because there is no state feedback over one MPC step

for i in range(n):
    start = time.time()
    
    u0 = np.array([0.3, 0.3, 0.3, 0.3, 0.1, 0.0, 0.1, 0.0]).reshape(8, 1)

    end = time.time()
    print("Computation time: ", end-start)
     
    ynext_lin = linear_simulator.make_step(u0)
    x0_lin = linear_estimator.make_step(ynext_lin)

    ynext_nonlin = nonlinear_simulator.make_step(u0)
    x0_nonlin = nonlinear_estimator.make_step(ynext_nonlin)
    print("sim")
    # sim is pos, theta, dpos, dtheta
    # controller is dpos, dtheta, theta, pos 
    # drone_acceleration = (np.array(x0_lin[6:12]) - last_x0_dot_lin )/dt
    # tvp.x = x0_lin
    # tvp.u = u0
    # tvp.drone_accel = drone_acceleration

    # print("\n")
    # print("a")
    # print(drone_acceleration)
    diff_pos = norm_2(x0_lin[0:3] -x0_nonlin[0:3])
    diff_euler_angs = norm_2(x0_lin[3:6] -x0_nonlin[3:6])
    diff_lin_vel = norm_2(x0_lin[6:9] -x0_nonlin[6:9])
    diff_body_rates = norm_2(x0_lin[9:12] -x0_nonlin[9:12])
    # print(diff)
    difference_list_pos.append(diff_pos)
    difference_list_euler_angs.append(diff_euler_angs)
    difference_list_lin_vel.append(diff_lin_vel)
    difference_list_body_rates.append(diff_body_rates)
    print(i)
    last_x0_dot_lin = np.array(x0_lin[6:12])

# print("u")
# print(u0)
# print("\n")
print("x linear")
print(x0_lin)
print("x nonlinear")
print(x0_nonlin)

fig, ax = plt.subplots()

t = np.arange(n)


# Plot the data
ax.plot(t, difference_list_pos, label='pos')
ax.plot(t, difference_list_euler_angs, label='euler angles')
ax.plot(t, difference_list_lin_vel, label='linear velocity')
ax.plot(t, difference_list_body_rates, label='body rates')
ax.set_xlabel("iterations")
ax.set_ylabel("L2 norm of the difference between the states of the linear and non linear controller")

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()