import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Basic rocket moving in 1D

m = 1.8  # rocket_mass
g = 9.81

model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None
u = None
x = None

x_vel = mpc_model.set_variable('_x',  'x_vel', (1, 1))
u_thrust = mpc_model.set_variable('_u',  'u_th', (1, 1))


dvel = mpc_model.set_variable('_z',  'dvel', (1, 1))
# last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(1, 1))
# last_input = mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(1, 1))
# drone_accel = mpc_model.set_variable(var_type='_tvp', var_name='drone_accel',shape=(1, 1))
#hardcode in targetpoint later for now

mpc_model.set_rhs('x_vel', dvel)


#would have to change to add roll and pitch for g term (still from last state input ig)
f = vertcat((u_thrust[0] / m))

# A = jacobian(f, last_state)
# print((A.shape))
# B = jacobian(f, last_input)
# print((B.shape))

# euler_lagrange = (dvel-drone_accel) - (A@(x_vel-last_state)) - (B@(u_thrust-last_input))
euler_lagrange = dvel - f

# TARGET VEL
target_velocity = np.array([[0.5]])
mpc_model.set_alg('euler_lagrange', euler_lagrange)

sqrt_vel_diff = sqrt((x_vel[0]-target_velocity[0])**2)
minimizing_term = u_thrust[0]**2 
mpc_model.set_expression(expr_name='sqrt_vel_diff', expr=sqrt_vel_diff)
mpc_model.set_expression(expr_name='minimizing_term', expr=minimizing_term)

mpc_model.setup()

mpc_controller = do_mpc.controller.MPC(mpc_model)
n_horizon = 20

setup_mpc = {
    'n_horizon': n_horizon,
    'n_robust': 0,
    'open_loop': 0,
    't_step': 0.04,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 1,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps', 'ipopt.print_level':0}
}

mpc_controller.set_param(**setup_mpc)


lterm = 0.9 * mpc_model.aux['sqrt_vel_diff'] + 0.0000001 * mpc_model.aux['minimizing_term']
mterm = 0.9 * mpc_model.aux['sqrt_vel_diff']

mpc_controller.set_objective(mterm=mterm, lterm=lterm)
# Input force is implicitly restricted through the objective.
mpc_controller.set_rterm(u_th=0.1)

thrust_limit = 50
u_upper_limits = np.array([thrust_limit])
u_lower_limits =  np.array([0.00])

mpc_controller.bounds['lower','_u','u_th'] = -u_lower_limits
mpc_controller.bounds['upper','_u','u_th'] = u_upper_limits

# class TVPData:
#     def __init__(self, x, u, drone_accel):
#         self.x = x
#         self.u = u
#         self.drone_accel = drone_accel

# TODO: SETTING INITIAL X0 and U0

# u0 = [0.0]

# init_acceleration = np.array([[0]])

# tvp = TVPData(x0, u0, init_acceleration)


# controller_tvp_template = mpc_controller.get_tvp_template()
# def controller_tvp_fun(t_now):
#     for k in range(n_horizon+1):
#         controller_tvp_template['_tvp',k,'last_state'] = tvp.x
#         controller_tvp_template['_tvp',k,'last_input'] = tvp.u
#         controller_tvp_template['_tvp',k,'drone_accel'] = tvp.drone_accel


#         return controller_tvp_template
# mpc_controller.set_tvp_fun(controller_tvp_fun)
mpc_controller.setup()


##setting up nonlinear simulator 
# mpc_model_sim = do_mpc.model.Model("continuous")

# x_pos_sim = mpc_model_sim.set_variable('_x',  'x_pos_sim', (1, 1))
# x_vel_sim = mpc_model_sim.set_variable('_x',  'x_vel_sim', (1, 1))
# u_thrust_sim = mpc_model_sim.set_variable('_u',  'u_th_sim', (1, 1))

# dvel_sim = mpc_model_sim.set_variable('_z',  'dvel_sim', (1, 1))

#would have to change to add roll and pitch for g term (still from last state input ig)
# f_sim = (u_thrust_sim[0] / m)

# euler_lagrange_sim = dvel_sim - f_sim

# mpc_model_sim.set_rhs('x_pos_sim', x_vel_sim)
# mpc_model_sim.set_rhs('x_vel_sim', dvel_sim)

# mpc_model_sim.set_alg('euler_lagrange_sim', euler_lagrange_sim)
# mpc_model_sim.setup()

estimator = do_mpc.estimator.StateFeedback(mpc_model)
simulator = do_mpc.simulator.Simulator(mpc_model)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 0.04
}

simulator.set_param(**params_simulator)
simulator.setup()

x0 = np.array([0.0])
mpc_controller.x0 = x0
# x0_sim = np.array([[0.0], [0.0]])
simulator.x0 = x0
estimator.x0 = x0

mpc_controller.set_initial_guess()


# Plotting

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

u0 = mpc_controller.make_step(x0)
mpc_controller.reset_history()


dt = .04
# last_x0_dot = np.array([0.0])
for i in range(20):
    start = time.time()
    u0 = mpc_controller.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    end = time.time()
    print("Computation time: ", end-start)

    # x0 = x0_sim[1]
    # drone_acceleration = (x0 - last_x0_dot )/dt
    # tvp.x = x0
    # tvp.u = u0
    # tvp.drone_accel = drone_acceleration

    # print("u")
    # print(u0)
    #print("\n")
    #print("x")
    #print(x0sim)
    #print("\n")
    #print("a")
    #print(drone_acceleration)
    #print("\n")

    print(i)
    # last_x0_dot = x0

fig, ax = plt.subplots()
t = mpc_controller.data['_time']
vel = mpc_controller.data['_x']

# Plot the data
ax.plot(t, vel, label='xv')

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()