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

difference_list = []

n = 40
# This represents one step of the MPC. Aka, if the MPC horizon was n steps
# This is technically open loop because there is no state feedback over one MPC step

for i in range(n):
    start = time.time()
    
    u0 = np.array([3, 1, 5, 2, 0.2, 0.3, 0.2, 0.3]).reshape(8, 1)

    end = time.time()
    print("Computation time: ", end-start)
     
    ynext_lin = linear_simulator.make_step(u0)
    x0_lin = linear_estimator.make_step(ynext_lin)

    ynext_nonlin = nonlinear_simulator.make_step(u0)
    x0_nonlin = nonlinear_estimator.make_step(ynext_nonlin)
    print("sim")
    # sim is pos, theta, dpos, dtheta
    # controller is dpos, dtheta, theta, pos 
    drone_acceleration = (np.array(x0_lin[6:12]) - last_x0_dot_lin )/dt
    tvp.x = x0_lin
    tvp.u = u0
    tvp.drone_accel = drone_acceleration

    # print("\n")
    # print("a")
    # print(drone_acceleration)
    diff = norm_2(x0_lin -x0_nonlin)
    # print(diff)
    difference_list.append(diff)
    print(i)
    last_x0_dot_lin = np.array(x0_lin[6:12])

# print("u")
# print(u0)
# print("\n")
print("x linear")
print(x0_lin)
print("x nonlinear")
print(x0_nonlin)

print(difference_list)

fig, ax = plt.subplots()

t = np.arange(n)


# Plot the data
ax.plot(t, difference_list, label='norm_2(linear, nonlinear) controller')
ax.set_xlabel("iterations")
ax.set_ylabel("L2 norm of the difference between the states of the linear and non linear controller")

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()