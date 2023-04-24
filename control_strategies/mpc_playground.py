import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


m = .5  # drone_mass
arm_length = 1
Ixx = 1.2
Iyy = 1.1
Izz = 1.0


model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None
u = None
x = None

pos = mpc_model.set_variable('states',  'pos', (3, 1))
theta = mpc_model.set_variable('states',  'theta', (3, 1))

dpos = mpc_model.set_variable('states',  'dpos', (3, 1))
dtheta = mpc_model.set_variable('states',  'dtheta', (3, 1))

u_th = mpc_model.set_variable('inputs',  'u_th', (4, 1))
u_ti = mpc_model.set_variable('inputs',  'u_ti', (4, 1))


ddpos = mpc_model.set_variable('algebraic',  'ddpos', (3, 1))
ddtheta = mpc_model.set_variable('algebraic',  'ddtheta', (3, 1))
last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(13, 1))
last_input = mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(8, 1))
drone_acc = mpc_model.set_variable(var_type='_tvp', var_name='drone_acc',shape=(6, 1))






mpc_model.set_rhs('pos', dpos)
mpc_model.set_rhs('theta', dtheta)
mpc_model.set_rhs('dpos', ddpos)
mpc_model.set_rhs('dtheta', ddtheta)

T1 = last_input[0]
T2 = last_input[1]
T3 = last_input[2]
T4 = last_input[3]
theta1 = last_input[4]
theta2 = last_input[5]
theta3 = last_input[6]
theta4 = last_input[7]

x = last_state[0]
y = last_state[1]
z = last_state[2]
roll = last_state[3]
pitch = last_state[4]
yaw = last_state[5]

dx = last_state[6]
dy = last_state[7]
dz = last_state[8]
droll = last_state[9]
dpitch = last_state[10]
dyaw = last_state[11]

ddx = ddpos[0]
ddy = ddpos[1]
ddz = ddpos[2]
ddroll = ddtheta[0]
ddpitch = ddtheta[1]
ddyaw = ddtheta[2]

g = last_state[12]

#second order taylor series approx of sin
def sinTE(x):
    return x - ((x)**3)/6
    #return sin(x)
def cosTE(x):
    return 1 -(x**2)/2
    #return cos(x)



f = vertcat(
    last_state[6],
    last_state[7],
    last_state[8],
    last_state[9],
    last_state[10],
    last_state[11],

    (last_input[1]*sinTE(last_input[5]) - last_input[3]*sinTE(last_input[7]) - m*last_state[12]*sinTE(last_state[4]))/m,
    # 2
    (last_input[0]*sinTE(last_input[4]) - last_input[2]*sinTE(last_input[6]) - m*last_state[12]*sinTE(last_state[3]))/m,
    # 3
    (last_input[0]*cosTE(last_input[4]) + last_input[1]*cosTE(last_input[5]) + last_input[2]*cosTE(last_input[6]) + last_input[3]*cosTE(last_input[7]) - m*last_state[12]*cosTE(roll)*cosTE(pitch))/m,
    # 4
    ((last_input[1]*cosTE(last_input[5])*arm_length) - (last_input[3]*cosTE(last_input[7])*arm_length) + (Iyy*last_state[10]*last_state[11] + Izz*last_state[10]*last_state[11]))/Ixx,
    # 5
    (last_input[0]*cosTE(last_input[4])*arm_length - last_input[2]*cosTE(last_input[6])*arm_length + (-Ixx*last_state[9]*last_state[11] + Izz*last_state[9]*last_state[11]))/Iyy,
    # 6
    (last_input[0]*sinTE(last_input[4])*arm_length + last_input[1]*sinTE(last_input[5])*arm_length + last_input[2]*sinTE(last_input[6])*arm_length + last_input[3]*sinTE(last_input[7])*arm_length + (Ixx*last_state[9]*last_state[10] - Iyy*last_state[9]*last_state[10]))/Izz
)






u_vec = vertcat(
    u_th,
    u_ti
)
state_vec = vertcat(
    pos,
    theta,
    dpos,
    dtheta,
    9.81
)




A = jacobian(f, last_state)
print((A.shape))
B = jacobian(f, last_input)
print((B.shape))

result_vec = vertcat(
    ddpos,
    ddtheta
)
euler_lagrange = (result_vec-drone_acc) - (A@(state_vec-last_state))[6:] - (B@(u_vec-last_input))[6:]


#print(euler_lagrange)

target_point = np.array([[0.0],[0.0],[0.15]])
mpc_model.set_alg('euler_lagrange', euler_lagrange)
mpc_model.set_expression(expr_name='cost', expr=sum1(.9*sqrt((pos[0]-target_point[0])**2 + (pos[1]-target_point[1])**2 + (pos[2]-target_point[2])**2) +.00000000001*sqrt((u_th[0])**2 + (u_th[1])**2 + (u_th[2])**2 + (u_th[3])**2 )))
mpc_model.set_expression(expr_name='mterm', expr=sum1(.9*sqrt((pos[0]-target_point[0])**2 + (pos[1]-target_point[1])**2 + (pos[2]-target_point[2])**2)))

mpc_model.setup()


mpc_controller = do_mpc.controller.MPC(mpc_model)


setup_mpc = {
    'n_horizon': 30,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 0.01,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 1,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps', 'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
}

mpc_controller.set_param(**setup_mpc)
tvp_template = mpc_controller.get_tvp_template()
n_horizon = 32
def tvp_fun(t_now):
    for k in range(n_horizon+1):
        tvp_template['_tvp',k, 'last_state'] = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[9.81]]
        tvp_template['_tvp',k, 'last_input'] = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        tvp_template['_tvp',k, 'drone_acc'] = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        return tvp_template
mpc_controller.set_tvp_fun(tvp_fun)


mterm = mpc_model.aux['mterm']
lterm = mpc_model.aux['cost']

mpc_controller.set_objective(mterm=mterm, lterm=lterm)
# Input force is implicitly restricted through the objective.
mpc_controller.set_rterm(u_th=1e-5)
mpc_controller.set_rterm(u_ti=1e-4)

tilt_limit = pi/2
thrust_limit = 50
u_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
u_lower_limits =  np.array([0, 0, 0, 0])
u_ti_upper_limits = np.array([tilt_limit, tilt_limit, tilt_limit, tilt_limit])
u_ti_lower_limits =  np.array([-tilt_limit, -tilt_limit, -tilt_limit, -tilt_limit])

x_limits = np.array([inf, inf, inf, pi/6, pi/6, pi/6, .1, .1, .1, 1, 1, 1])

mpc_controller.bounds['lower','_u','u_th'] = u_lower_limits
mpc_controller.bounds['upper','_u','u_th'] = u_upper_limits
mpc_controller.bounds['lower','_u','u_ti'] = u_ti_lower_limits
mpc_controller.bounds['upper','_u','u_ti'] = u_ti_upper_limits

#mpc_controller.bounds['lower','_x','pos'] = -x_limits[0:3]
#mpc_controller.bounds['upper','_x','pos'] = x_limits[0:3]

#mpc_controller.bounds['lower','_x','theta'] = -x_limits[3:6]
#mpc_controller.bounds['upper','_x','theta'] = x_limits[3:6]

#mpc_controller.bounds['lower','_x','dpos'] = -x_limits[6:9]
#mpc_controller.bounds['upper','_x','dpos'] = x_limits[6:9]

#mpc_controller.bounds['lower','_x','dtheta'] = -x_limits[9:12]
#mpc_controller.bounds['upper','_x','dtheta'] = x_limits[9:12]


mpc_controller.setup()
x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,.0,0.0,0.0,0.0,0.0,0.0])
mpc_controller.x0 = x0



u_val = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

estimator = do_mpc.estimator.StateFeedback(mpc_model)
simulator = do_mpc.simulator.Simulator(mpc_model)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.01
}

simulator.set_param(**params_simulator)
tvp_template2 = simulator.get_tvp_template()
n_horizon = 16
def tvp_fun2(t_now):
    tvp_template2['last_state'] = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[9.81]]
    tvp_template2['last_input'] = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
    tvp_template2['drone_acc'] = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
    return tvp_template2
simulator.set_tvp_fun(tvp_fun2)
simulator.setup()
estimator.x0 = x0

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc_controller.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the positions
    g.add_line(var_type='_x', var_name='pos', axis=ax[0])
    #g.add_line(var_type='_x', var_name='theta', axis=ax[0])
    #g.add_line(var_type='_x', var_name='z', axis=ax[2])

    # Plot the thrusts
    g.add_line(var_type='_u', var_name='u_th', axis=ax[1])
    #g.add_line(var_type='_u', var_name='T2', axis=ax[4])
    #g.add_line(var_type='_u', var_name='T3', axis=ax[5])
    #g.add_line(var_type='_u', var_name='T4', axis=ax[6])



ax[0].set_ylabel('pos')
#ax[1].set_ylabel('theta')
ax[1].set_ylabel('thrusts')
ax[1].set_xlabel('time (s)')





#u0 = mpc_controller.make_step(x0)



simulator.reset_history()
simulator.x0 = x0

mpc_controller.set_initial_guess()
dt = .01

last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
for i in range(800):
    u0 = mpc_controller.make_step(x0)
    x0 = simulator.make_step(u0)
    tvp_template_c = mpc_controller.get_tvp_template()
    n_horizon = 30
    drone_acceleration = (np.array(x0[6:]) - last_x0_dot )/dt
    #print(drone_acceleration)
    def tvp_fun_c(t_now):
        for k in range(n_horizon+1):
            tvp_template_c['_tvp',k,'last_state'] = vertcat(x0,9.81)
            tvp_template_c['_tvp',k,'last_input'] = u0
            tvp_template_c['_tvp',k,'drone_acc'] = drone_acceleration
            return tvp_template_c
    mpc_controller.set_tvp_fun(tvp_fun_c)

    tvp_template_c2 = simulator.get_tvp_template()
    def tvp_fun2_c(t_now):
        tvp_template_c2['last_state'] = vertcat(x0,9.81)
        tvp_template_c2['last_input'] = u0
        tvp_template_c2['drone_acc'] = drone_acceleration
        return tvp_template_c2
  
    simulator.set_tvp_fun(tvp_fun2_c)
    print("u")
    print(u0)
    print("\n")
    print("x")
    print(x0)
    print("\n")
    print("a")
    print(drone_acceleration)
    print("\n")

    #print("sep")
    last_x0_dot = np.array(x0[6:])

mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()#
fig
    




