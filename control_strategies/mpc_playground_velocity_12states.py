import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


m = 1.8  # drone_mass
arm_length = .2286
Ixx = 1.0
Iyy = 1.0
Izz = 1.0


model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None
u = None
x = None

g = 9.81

#second order taylor series approx of sin
def sinTE(x):
    return x - ((x)**3)/6
    #return sin(x)
def cosTE(x):
    return 1 -(x**2)/2
    #return cos(x)

def rotBE(r,p,y):

    
    rotBErow1 = horzcat((cos(y)*cos(p)), (sin(y)*cos(p)), (-sin(p)),0,0,0)
    rotBErow2 = horzcat((cos(y)*sin(p) *sin(r) - sin(y)*cos(r)), (sin(y)*sin(p) * sin(r) + cos(y)*cos(r)),(cos(p)*sin(r)),0,0,0)
    rotBErow3 = horzcat((cos(y)*sin(p) * cos(r) + sin(y)*sin(r)), (sin(y)*sin(p) * cos(r) - cos(y)*sin(r)), (cos(p)*cos(r)),0,0,0)
    rotBErow4 = horzcat(0,0,0,1,0,0)
    rotBErow5 = horzcat(0,0,0,0,1,0)
    rotBErow6 = horzcat(0,0,0,0,0,1)

    rotBEm = vertcat(rotBErow1, rotBErow2,rotBErow3,rotBErow4,rotBErow5,rotBErow6)
    return rotBEm
    

def rotEB(r,p,y):

    rotEBm = transpose(rotBE(r,p,y))
    return rotEBm

#print(rotEB(rotBE(0,0,math.pi)))
dpos = mpc_model.set_variable('_x',  'dpos', (3, 1))
dtheta = mpc_model.set_variable('_x',  'dtheta', (3, 1))
eulerang = mpc_model.set_variable('_x',  'eulerang', (3, 1))
#dtheta is in terms of BODY ANGULAR VELOCITIES, while theta is in terms of SPATIAL EULER ANGLES
pos = mpc_model.set_variable('_x',  'pos', (3, 1))

u_th = mpc_model.set_variable('_u',  'u_th', (4, 1))
u_ti = mpc_model.set_variable('_u',  'u_ti', (4, 1))


ddpos = mpc_model.set_variable('_z',  'ddpos', (3, 1))
ddtheta = mpc_model.set_variable('_z',  'ddtheta', (3, 1))
last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(12, 1))
last_input = mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(8, 1))
drone_acc = mpc_model.set_variable(var_type='_tvp', var_name='drone_acc',shape=(6, 1))
#hardcode in targetpoint later for now

eulroll = eulerang[0]
eulpitch = eulerang[1]
eulyaw = eulerang[2]
droll_c = dtheta[0]
dpitch_c = dtheta[1]
dyaw_c = dtheta[2]

euler_ang_vel = vertcat((droll_c + dyaw_c*cos(eulroll)*tan(eulpitch) + dpitch_c*sin(eulroll)*tan(eulpitch)),
                        (dpitch_c*cos(eulroll) - dyaw_c*sin(eulroll)),
                        ((dyaw_c*cos(eulroll)/(cos(eulpitch))) + dpitch_c*(sin(eulroll)/cos(eulpitch)))
)
mpc_model.set_rhs('pos', dpos)
mpc_model.set_rhs('dpos', ddpos)
mpc_model.set_rhs('eulerang', euler_ang_vel)
mpc_model.set_rhs('dtheta', ddtheta)

roll = last_state[6]
pitch = last_state[7]
yaw = last_state[8]

ddx = ddpos[0]
ddy = ddpos[1]
ddz = ddpos[2]
ddroll = ddtheta[0]
ddpitch = ddtheta[1]
ddyaw = ddtheta[2]




#would have to change to add roll and pitch for g term (still from last state input ig)
f = vertcat(

    (last_input[1]*sinTE(last_input[5]) - last_input[3]*sinTE(last_input[7]))/m,
    # 2
    (last_input[0]*sinTE(last_input[4]) - last_input[2]*sinTE(last_input[6]))/m,
    # 3
    (last_input[0]*cosTE(last_input[4]) + last_input[1]*cosTE(last_input[5]) + last_input[2]*cosTE(last_input[6]) + last_input[3]*cosTE(last_input[7]))/m,
    # 4
    ((last_input[1]*cosTE(last_input[5])*arm_length) - (last_input[3]*cosTE(last_input[7])*arm_length) + (Iyy*last_state[4]*last_state[5] + Izz*last_state[4]*last_state[5]))/Ixx,
    # 5
    (last_input[0]*cosTE(last_input[4])*arm_length - last_input[2]*cosTE(last_input[6])*arm_length + (-Ixx*last_state[3]*last_state[5] + Izz*last_state[3]*last_state[5]))/Iyy,
    # 6
    (last_input[0]*sinTE(last_input[4])*arm_length + last_input[1]*sinTE(last_input[5])*arm_length + last_input[2]*sinTE(last_input[6])*arm_length + last_input[3]*sinTE(last_input[7])*arm_length + (Ixx*last_state[3]*last_state[4] - Iyy*last_state[3]*last_state[4]))/Izz
)


rotEBMatrix = rotEB(roll, pitch, yaw)
zero_row = horzcat(0,0,0,0,0,0)

w_b = vertcat(last_state[3:6])
v_b = vertcat(last_state[0:3])
alpha_b = vertcat(drone_acc[3:6])
r_b = vertcat(last_state[9:12])

fspatial_linear_acc = vertcat((rotEBMatrix@(f))[0:3] + 2 * skew(w_b)@v_b + skew(alpha_b)@r_b + skew(w_b)@(skew(w_b)@r_b))
fspatial_rotation_acc = vertcat(f[3:6]) #+ skew(w_b)@alpha_b)
#fspatial_rotation_acc = vertcat(f[3:6])
fspatial = vertcat(fspatial_linear_acc, fspatial_rotation_acc)
# print(fspatial)
u_vec = vertcat(
    u_th,
    u_ti
)
state_vec = vertcat(
    dpos,
    dtheta,
    eulerang,
    pos
)
result_vec = vertcat(ddx, ddy, ddz, ddroll, ddpitch, ddyaw)

A = jacobian(fspatial, last_state)
print((A.shape))
B = jacobian(fspatial, last_input)
print((B.shape))
C = jacobian(drone_acc - fspatial, drone_acc)
print(C)

euler_lagrange = C@(result_vec-drone_acc) -(A@(state_vec-last_state))- (B@(u_vec-last_input)) + (drone_acc - fspatial + vertcat(0,0,g,0,0,0))

mpc_model.set_alg('euler_lagrange', euler_lagrange)
targetvel = np.array([[0.0],[0.0],[0.5]])

diff = ((dpos[0]-targetvel[0])**2 + (dpos[1]-targetvel[1])**2 + (dpos[2]-targetvel[2])**2)
mpc_model.set_expression('diff', diff)

mpc_model.setup()

mpc_controller = do_mpc.controller.MPC(mpc_model)
n_horizon = 15

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

mterm = mpc_model.aux['diff'] # terminal cost
lterm = mpc_model.aux['diff'] # stage cost

mpc_controller.set_objective(mterm=mterm, lterm=lterm)
# Input force is implicitly restricted through the objective.
mpc_controller.set_rterm(u_th=0.1)
mpc_controller.set_rterm(u_ti=0.01)

tilt_limit = pi/(2.2)
thrust_limit = 30
u_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
u_lower_limits =  np.array([0.00, 0.00, 0.00, 0.00])
u_ti_upper_limits = np.array([tilt_limit, tilt_limit, tilt_limit, tilt_limit])
u_ti_lower_limits =  np.array([-tilt_limit, -tilt_limit, -tilt_limit, -tilt_limit])


mpc_controller.bounds['lower','_u','u_th'] = u_lower_limits
mpc_controller.bounds['upper','_u','u_th'] = u_upper_limits
mpc_controller.bounds['lower','_u','u_ti'] = u_ti_lower_limits
mpc_controller.bounds['upper','_u','u_ti'] = u_ti_upper_limits


x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
u0 = np.array([m*g/4,m*g/4,m*g/4,m*g/4,0.0,0.0,0.0,0.0]).T
#u0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

drone_acceleration = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
mpc_controller.x0 = x0
mpc_controller.u0 = u0
mpc_controller.z0 = drone_acceleration
class TVPData:
    def __init__(self, x, u, drone_accel):
        self.x = x
        self.u = u
        self.drone_accel = drone_accel
     
       
tvp = TVPData(x0, u0, drone_acceleration)

controller_tvp_template = mpc_controller.get_tvp_template()
def controller_tvp_fun(t_now):
    for k in range(n_horizon+1):
        controller_tvp_template['_tvp',k,'last_state'] = tvp.x
        controller_tvp_template['_tvp',k,'last_input'] = tvp.u
        controller_tvp_template['_tvp',k,'drone_acc'] = tvp.drone_accel


        return controller_tvp_template
mpc_controller.set_tvp_fun(controller_tvp_fun)
mpc_controller.setup()

##setting up nonlinear simulator 

mpc_modelsim = do_mpc.model.Model("continuous")

pos_s = mpc_modelsim.set_variable('states',  'pos_s', (3, 1))
eulerang_s = mpc_modelsim.set_variable('states',  'eulerang_s', (3, 1))
dpos_s = mpc_modelsim.set_variable('states',  'dpos_s', (3, 1))
dtheta_s = mpc_modelsim.set_variable('states',  'dtheta_s', (3, 1))
ddpos_s = mpc_modelsim.set_variable('algebraic',  'ddpos_s', (3, 1))
ddtheta_s = mpc_modelsim.set_variable('algebraic',  'ddtheta_s', (3, 1))

#inputs
u_th_s = mpc_modelsim.set_variable('inputs',  'u_th_s', (4, 1))
u_ti_s = mpc_modelsim.set_variable('inputs',  'u_ti_s', (4, 1))

eulroll = eulerang_s[0]
eulpitch = eulerang_s[1]
eulyaw = eulerang_s[2]
droll_c = dtheta_s[0]
dpitch_c = dtheta_s[1]
dyaw_c = dtheta_s[2]

euler_ang_vel_s = vertcat((droll_c + dyaw_c*cos(eulroll)*tan(eulpitch) + dpitch_c*sin(eulroll)*tan(eulpitch)),
                        (dpitch_c*cos(eulroll) - dyaw_c*sin(eulroll)),
                        ((dyaw_c*cos(eulroll)/(cos(eulpitch))) + dpitch_c*(sin(eulroll)/cos(eulpitch)))
)


mpc_modelsim.set_rhs('pos_s', dpos_s)
mpc_modelsim.set_rhs('dpos_s', ddpos_s)
mpc_modelsim.set_rhs('eulerang_s', euler_ang_vel_s)
mpc_modelsim.set_rhs('dtheta_s', ddtheta_s)

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
roll = eulerang_s[0]
pitch = eulerang_s[1]
yaw = eulerang_s[2]

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


f_sim= vertcat(
    # 1
(T2*sin(theta2) - T4*sin(theta4))/m,
    # 2
(T1*sin(theta1) - T3*sin(theta3))/m,
    # 3
(T1*cos(theta1) + T2*cos(theta2) + T3*cos(theta3) + T4*cos(theta4))/m,
    # 4
((T2*cos(theta2)*arm_length) - (T4*cos(theta4)*arm_length) + (Iyy*dpitch*dy - Izz*dpitch*dy))/Ixx,
    # 5
(T1*cos(theta1)*arm_length - T3*cos(theta3)*arm_length + (-Ixx*droll*dy + Izz*droll*dy))/Iyy,
    # 6
(T1*sin(theta1)*arm_length + T2*sin(theta2)*arm_length + T3*sin(theta3)*arm_length + T4*sin(theta4)*arm_length + (Ixx*droll*dpitch - Iyy*droll*dpitch))/Izz,
)

w_b = vertcat(droll, dpitch, dyaw)
v_b = vertcat(dx, dy, dz)
alpha_b = vertcat(ddroll, ddpitch, ddyaw)
r_b = vertcat(x, y, z)

f_linear_acc_sim = vertcat(((rotEB(roll, pitch, yaw)@(f_sim))[0:3] + 2 * skew(w_b)@v_b + skew(alpha_b)@r_b + skew(w_b)@(skew(w_b)@r_b)))
f_rotation_acc_sim = vertcat(f_sim[3:6]) #+ skew(w_b)@alpha_b)
f_spatial_sim = vertcat(f_linear_acc_sim, f_rotation_acc_sim)

euler_lagrange_simspatial = vertcat(ddx, ddy, ddz, ddroll, ddpitch, ddyaw)- f_spatial_sim + vertcat(0,0,g,0,0,0)



mpc_modelsim.set_alg('euler_lagrange_simspatial', euler_lagrange_simspatial)
mpc_modelsim.setup()

simulator = do_mpc.simulator.Simulator(mpc_modelsim)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 0.04
}

simulator.set_param(**params_simulator)


simulator.setup()

estimator = do_mpc.estimator.StateFeedback(mpc_modelsim)

# Plotting

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

x0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
u0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
sim_acc = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).T
simulator.x0 = x0sim
estimator.x0 = x0sim

simulator.u0 = u0sim
simulator.z0 = sim_acc

estimator.u0 = u0sim
estimator.z0 = sim_acc

mpc_controller.set_initial_guess()
simulator.set_initial_guess()

dt = .04
curr_roll = 0.0
curr_pitch =0.0
last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])


# print("u")
# print(tvp.u)
# print("\n")
# print("x")
# print(tvp.x)
# print("\n")
# print("a")
# print(tvp.drone_accel)
# print("\n")


for i in range(40):
    start = time.time()
    
    u0 = mpc_controller.make_step(x0)

    end = time.time()
    print("Computation time: ", end-start)
     
    ynext= simulator.make_step(u0)
    x0sim = estimator.make_step(ynext)
    print("sim")
    # sim is pos, theta, dpos, dtheta
    # controller is dpos, dtheta, theta, pos 
    x0 = vertcat(x0sim[6:12],x0sim[3:6], x0sim[0:3])
    drone_acceleration = (np.array(x0[0:6]) - last_x0_dot )/dt
    tvp.x = x0
    tvp.u = u0
    tvp.drone_accel = drone_acceleration
    
    # print("u")
    # print(u0)
    # print("\n")
    # print("x")
    # print(x0sim)
    # print("\n")
    # print("a")
    # print(drone_acceleration)
    print(i)
    last_x0_dot = np.array(x0[0:6])

fig, ax = plt.subplots()

t = mpc_controller.data['_time']
x_vel = mpc_controller.data['_x'][:, 0]
y_vel = mpc_controller.data['_x'][:, 1]
z_vel = mpc_controller.data['_x'][:, 2]

roll_graph = mpc_controller.data['_x'][:, 6]
pitch_graph = mpc_controller.data['_x'][:,7]
yaw_graph = mpc_controller.data['_x'][:, 8]


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