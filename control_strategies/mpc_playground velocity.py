import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


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



dpos = mpc_model.set_variable('states',  'dpos', (3, 1))
dtheta = mpc_model.set_variable('states',  'dtheta', (3, 1))

u_th = mpc_model.set_variable('inputs',  'u_th', (4, 1))
u_ti = mpc_model.set_variable('inputs',  'u_ti', (4, 1))


ddpos = mpc_model.set_variable('algebraic',  'ddpos', (3, 1))
ddtheta = mpc_model.set_variable('algebraic',  'ddtheta', (3, 1))
last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(6, 1))
last_input = mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(8, 1))
drone_acc = mpc_model.set_variable(var_type='_tvp', var_name='drone_acc',shape=(6, 1))
roll_and_pitch = mpc_model.set_variable(var_type='_tvp', var_name='roll_and_pitch',shape=(2, 1))
#hardcode in targetpoint later for now




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


dx = last_state[0]
dy = last_state[1]
dz = last_state[2]
droll = last_state[3]
dpitch = last_state[4]
dyaw = last_state[5]

ddx = ddpos[0]
ddy = ddpos[1]
ddz = ddpos[2]
ddroll = ddtheta[0]
ddpitch = ddtheta[1]
ddyaw = ddtheta[2]

g = 9.81

#second order taylor series approx of sin
def sinTE(x):
    return x - ((x)**3)/6
    #return sin(x)
def cosTE(x):
    return 1 -(x**2)/2
    #return cos(x)


#would have to change to add roll and pitch for g term (still from last state input ig)
f = vertcat(

    (last_input[1]*sinTE(last_input[5]) - last_input[3]*sinTE(last_input[7]) - m*g*sinTE(roll_and_pitch[1]))/m,
    # 2
    (last_input[0]*sinTE(last_input[4]) - last_input[2]*sinTE(last_input[6]) - m*g*sinTE(roll_and_pitch[0]))/m,
    # 3
    (last_input[0]*cosTE(last_input[4]) + last_input[1]*cosTE(last_input[5]) + last_input[2]*cosTE(last_input[6]) + last_input[3]*cosTE(last_input[7]) - m*g*cosTE(0)*cosTE(0))/m,
    # 4
    ((last_input[1]*cosTE(last_input[5])*arm_length) - (last_input[3]*cosTE(last_input[7])*arm_length) + (Iyy*last_state[4]*last_state[5] + Izz*last_state[4]*last_state[5]))/Ixx,
    # 5
    (last_input[0]*cosTE(last_input[4])*arm_length - last_input[2]*cosTE(last_input[6])*arm_length + (-Ixx*last_state[3]*last_state[5] + Izz*last_state[3]*last_state[5]))/Iyy,
    # 6
    (last_input[0]*sinTE(last_input[4])*arm_length + last_input[1]*sinTE(last_input[5])*arm_length + last_input[2]*sinTE(last_input[6])*arm_length + last_input[3]*sinTE(last_input[7])*arm_length + (Ixx*last_state[3]*last_state[4] - Iyy*last_state[3]*last_state[4]))/Izz
)




u_vec = vertcat(
    u_th,
    u_ti
)
state_vec = vertcat(
    dpos,
    dtheta,
)



A = jacobian(f, last_state)
print((A.shape))
B = jacobian(f, last_input)
print((B.shape))

result_vec = vertcat(
    ddpos,
    ddtheta
)
euler_lagrange = (result_vec-drone_acc) - (A@(state_vec-last_state)) - (B@(u_vec-last_input))


#print(euler_lagrange)

target_point = np.array([[.5],[0.0],[1.1]])
mpc_model.set_alg('euler_lagrange', euler_lagrange)
mpc_model.set_expression(expr_name='cost', expr=sum1(.9*sqrt((dpos[0]-target_point[0])**2 + (dpos[1]-target_point[1])**2 + (dpos[2]-target_point[2])**2) +.00000000001*sqrt((u_th[0])**2 + (u_th[1])**2 + (u_th[2])**2 + (u_th[3])**2 )))
mpc_model.set_expression(expr_name='mterm', expr=sum1(.9*sqrt((dpos[0]-target_point[0])**2 + (dpos[1]-target_point[1])**2 + (dpos[2]-target_point[2])**2)))

mpc_model.setup()

mpc_controller = do_mpc.controller.MPC(mpc_model)

setup_mpc = {
    'n_horizon': 20,
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

n_horizon = 20

mterm = mpc_model.aux['mterm']
lterm = mpc_model.aux['cost']

mpc_controller.set_objective(mterm=mterm, lterm=lterm)
# Input force is implicitly restricted through the objective.
mpc_controller.set_rterm(u_th=1e-5)
mpc_controller.set_rterm(u_ti=1e-4)

tilt_limit = pi/2
thrust_limit = 50
u_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
u_lower_limits =  np.array([0.0, 0.0, 0.0, 0.0])
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



class TVPData:
    def __init__(self, x, u, drone_accel, roll_and_pitch):
        self.x = x0
        self.u = u0
        self.drone_accel = drone_accel
        self.roll_and_pitch = roll_and_pitch
       
    

x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
mpc_controller.x0 = x0
u0 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
init_acceleration = np.array([[0.0],[0.0],[-9.81],[0.0],[0.0],[0.0]])
roll_and_pitch0 = np.array([[0.0],[0.0]])
#target_point0 = np.array([[2.0],[0.0], [3.0]])
tvp = TVPData(x0, u0, init_acceleration, roll_and_pitch0)


controller_tvp_template = mpc_controller.get_tvp_template()
def controller_tvp_fun(t_now):
    for k in range(n_horizon+1):
        controller_tvp_template['_tvp',k,'last_state'] = tvp.x
        controller_tvp_template['_tvp',k,'last_input'] = tvp.u
        controller_tvp_template['_tvp',k,'drone_acc'] = tvp.drone_accel
        controller_tvp_template['_tvp',k,'roll_and_pitch'] = tvp.roll_and_pitch


        return controller_tvp_template
mpc_controller.set_tvp_fun(controller_tvp_fun)
mpc_controller.setup()


estimator = do_mpc.estimator.StateFeedback(mpc_model)
estimator.x0 = x0


##setting up nonlinear simulator 

mpc_modelsim = do_mpc.model.Model(model_type)


pos_s = mpc_modelsim.set_variable('states',  'pos_s', (3, 1))
theta_s = mpc_modelsim.set_variable('states',  'theta_s', (3, 1))
dpos_s = mpc_modelsim.set_variable('states',  'dpos_s', (3, 1))
dtheta_s = mpc_modelsim.set_variable('states',  'dtheta_s', (3, 1))
ddpos_s = mpc_modelsim.set_variable('algebraic',  'ddpos_s', (3, 1))
ddtheta_s = mpc_modelsim.set_variable('algebraic',  'ddtheta_s', (3, 1))

#inputs
u_th_s = mpc_modelsim.set_variable('inputs',  'u_th_s', (4, 1))
u_ti_s = mpc_modelsim.set_variable('inputs',  'u_ti_s', (4, 1))

#representing dynamics

T1 = u_th_s[0]
T2 = u_th_s[1]
T3 = u_th_s[2]
T4 = u_th_s[3]
theta1 = u_ti_s[0]
theta2 = u_ti_s[1]
theta3 = u_ti_s[2]
theta4 = u_ti_s[3]

x = pos_s[0]
y = pos_s[1]
z = pos_s[2]
roll = theta_s[0]
pitch = theta_s[1]
yaw = theta_s[2]

dx = dpos_s[0]
dy = dpos_s[1]
dz = dpos_s[2]
droll = dtheta_s[0]
dpitch = dtheta_s[1]
dyaw = dtheta_s[2]

ddx = ddpos_s[0]
ddy = ddpos_s[1]
ddz = ddpos_s[2]
ddroll = ddtheta_s[0]
ddpitch = ddtheta_s[1]
ddyaw = ddtheta_s[2]


euler_lagrange_sim= vertcat(
    
    # 1
ddx - (T2*sin(theta2) - T4*sin(theta4) - m*g*sin(pitch))/m,
    # 2
ddy - (T1*sin(theta1) - T3*sin(theta3) - m*g*sin(roll))/m,
    # 3
ddz -  (T1*cos(theta1) + T2*cos(theta2) + T3*cos(theta3) + T4*cos(theta4) - m*g*cos(roll)*cos(pitch))/m,
    # 4
ddroll -  ((T2*cos(theta2)*arm_length) - (T4*cos(theta4)*arm_length) + (Iyy*dpitch*dy - Izz*dpitch*dy))/Ixx,
    # 5
ddpitch  -  (T1*cos(theta1)*arm_length - T3*cos(theta3)*arm_length + (-Ixx*droll*dy + Izz*droll*dy))/Iyy,
    # 6
ddyaw - (T1*sin(theta1)*arm_length + T2*sin(theta2)*arm_length + T3*sin(theta3)*arm_length + T4*sin(theta4)*arm_length + (Ixx*droll*dpitch - Iyy*droll*dpitch))/Izz,
)






mpc_modelsim.set_rhs('pos_s', dpos_s)
mpc_modelsim.set_rhs('theta_s', dtheta_s)
mpc_modelsim.set_rhs('dpos_s', ddpos_s)
mpc_modelsim.set_rhs('dtheta_s', ddtheta_s)

mpc_modelsim.set_alg('euler_lagrange_sim', euler_lagrange_sim)
mpc_modelsim.setup()

simulator = do_mpc.simulator.Simulator(mpc_modelsim)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.01
}

simulator.set_param(**params_simulator)


simulator.setup()



# Plotting

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

# mpc_graphics = do_mpc.graphics.Graphics(mpc_controller.data)
# sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))




#u0 = mpc_controller.make_step(x0)

simulator.reset_history()
x0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
simulator.x0 = x0sim




mpc_controller.set_initial_guess()
dt = .01
curr_roll = 0.0
curr_pitch =0.0
last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

reached_target = False
step_counter = 0
convergence_counter = 0

while convergence_counter < 10 or (not reached_target):
    start = time.time()
    u0 = mpc_controller.make_step(x0)
    x0sim = simulator.make_step(u0)
    x0 = x0sim[6:]
    drone_acceleration = (np.array(x0) - last_x0_dot )/dt
    tvp.x = x0
    tvp.u = u0
    tvp.drone_accel = drone_acceleration
    #tvp.target_point = np.array([[cos(i*pi/180)*1.],[sin(i*pi/180)*1.],[3.]])
    #tvp.target_point = np.array([[2.0],[0.0], [3.0]])
    #curr_roll = curr_roll + float(x0[3]*dt)
    #curr_pitch = curr_pitch + float(x0[4]*dt)
    curr_roll = x0sim[3]
    curr_pitch = x0sim[4]
    rparray = np.array([curr_roll, curr_pitch])
    tvp.roll_and_pitch = rparray

    end = time.time()
    computation_time = end-start
    print("Computation time: ", computation_time)

    #print("u")
    #print(u0)
    #print("\n")
    #print("x")
    #print(x0sim)
    #print("\n")
    #print("a")
    #print(drone_acceleration)
    #print("\n")

    #print("sep")
    step_counter += 1
    if convergence_counter > 0:
        convergence_counter += 1
    last_x0_dot = np.array(x0)

    if mpc_controller.data['_x'][-1, 0] >= target_point[0, 0] - 0.1 and mpc_controller.data['_x'][-1, 0] <= target_point[0, 0] + 0.1:
        convergence_counter += 1
        reached_target = True

    print(step_counter)

fig, ax = plt.subplots()

t = mpc_controller.data['_time']
x_pos = mpc_controller.data['_x'][:, 0]

# Plot the data
ax.plot(t, x_pos, label='x')

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()
