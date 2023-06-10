import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


m = 1.8  # drone_mass
arm_length = .2286
Ixx = 1.2
Iyy = 1.1
Izz = 1.0


model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None
u = None
x = None


def rotBE(r,p,y):

    
    rotBErow1 = horzcat((cos(y)*cos(p)), (sin(y)*cos(p)), (-sin(p)),0,0,0)
    rotBErow2 = horzcat((cos(y)*sin(p) *sin(r) - sin(y)*cos(r)), (sin(y)*sin(p) * sin(r) + cos(y)*cos(r)),(cos(p)*sin(r)),0,0,0)
    rotBErow3 = horzcat((cos(y)*sin(p) * cos(r) + sin(y)*sin(r)), (sin(y)*sin(p) * cos(r) - cos(y)*sin(r)), (cos(p)*cos(r)),0,0,0)
    rotBErow4 = horzcat(0,0,0,1,0,0)
    rotBErow5 = horzcat(0,0,0,0,1,0)
    rotBErow6 = horzcat(0,0,0,0,0,1)

    rotBEm = vertcat(rotBErow1, rotBErow2,rotBErow3,rotBErow4,rotBErow5,rotBErow6)


    #rotBEm = [[(cos(y)*cos(p)), (sin(y)*cos(p)), (-sin(p)),0,0,0], 
                    #[(cos(y)*sin(p) *sin(r) - sin(y)*cos(r)), (sin(y)*sin(p) * sin(r) + cos(y)*cos(r)),(cos(p)*sin(r)),0,0,0],
                    #[(cos(y)*sin(p) * cos(r) + sin(y)*sin(r)), (sin(y)*sin(p) * cos(r) - cos(y)*sin(r)), (cos(p)*cos(r)),0,0,0]]
    return rotBEm
    

def rotEB(r,p,y):

    rotEBm = transpose(rotBE(r,p,y))
    #rotEBm= [[(cos(y)*cos(p)), (sin(y)*cos(p)),(cos(y)*sin(p) *sin(r) - sin(y)*cos(r)),(cos(y)*sin(p) * cos(r) + sin(y)*sin(r)),1,0,0],
    #[(sin(y)*cos(p)), (sin(y)*sin(p) * sin(r) + cos(y)*cos(r)),(sin(y)*sin(p) * cos(r) - cos(y)*sin(r)),0,1,0],
    #[(-sin(p)),(cos(p)*sin(r)),(cos(p)*cos(r)),0,0,1]]
    return rotEBm

#print(rotEB(rotBE(0,0,math.pi)))
dpos = mpc_model.set_variable('_x',  'dpos', (3, 1))
dtheta = mpc_model.set_variable('_x',  'dtheta', (3, 1))
theta = mpc_model.set_variable('_x',  'theta', (3, 1))

u_th = mpc_model.set_variable('_u',  'u_th', (4, 1))
u_ti = mpc_model.set_variable('_u',  'u_ti', (4, 1))


ddpos = mpc_model.set_variable('_z',  'ddpos', (3, 1))
ddtheta = mpc_model.set_variable('_z',  'ddtheta', (3, 1))
last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(9, 1))
last_input = mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(8, 1))
drone_acc = mpc_model.set_variable(var_type='_tvp', var_name='drone_acc',shape=(6, 1))
#hardcode in targetpoint later for now




mpc_model.set_rhs('dpos', ddpos)
mpc_model.set_rhs('theta', dtheta)
mpc_model.set_rhs('dtheta', ddtheta)

#T1 = last_input[0]
#T2 = last_input[1]
#T3 = last_input[2]
#T4 = last_input[3]
#theta1 = last_input[4]
#theta2 = last_input[5]
#theta3 = last_input[6]
#theta4 = last_input[7]


#dx = last_state[0]
#dy = last_state[1]
#dz = last_state[2]
#droll = last_state[3]
#dpitch = last_state[4]
#dyaw = last_state[5]

roll = last_state[6]
pitch = last_state[7]
yaw = last_state[8]



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
#print(skew(w_t)[0,:].shape)
#print(horzcat(skew(w_t)[0,:] ,0,0,0).shape)
#fspatial  = f
fspatial = rotEBMatrix@(f) 
print(fspatial.shape)
u_vec = vertcat(
    u_th,
    u_ti
)
state_vec = vertcat(
    dpos,
    dtheta,
    theta
)



A = jacobian(fspatial, last_state)
print((A.shape))
B = jacobian(fspatial, last_input)
print((B.shape))

result_vec =vertcat(ddx, ddy, ddz, ddroll, ddpitch, ddyaw)
euler_lagrange = (result_vec-drone_acc) - (A@(state_vec-last_state)) - (B@(u_vec-last_input)) #+ vertcat(0.0,0.0,g,0.0,0.0,0.0)


#print(euler_lagrange)

mpc_model.set_alg('euler_lagrange', euler_lagrange)
targetvel = np.array([[0.6],[0.0],[0.9]])

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
mpc_controller.set_rterm(u_ti=0.001)

tilt_limit = pi/(2.2)
thrust_limit = 50
u_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
u_lower_limits =  np.array([0.00, 0.00, 0.00, 0.00])
#u_tilt_bounds = np.array([0.0, 0.0, 0.0, 0.0])
u_ti_upper_limits = np.array([tilt_limit, tilt_limit, tilt_limit, tilt_limit])
u_ti_lower_limits =  np.array([-tilt_limit, -tilt_limit, -tilt_limit, -tilt_limit])


mpc_controller.bounds['lower','_u','u_th'] = -u_upper_limits
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
    def __init__(self, x, u, drone_accel):
        self.x = x
        self.u = u
        self.drone_accel = drone_accel
     
       
    

x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
mpc_controller.x0 = x0
#u0 = [m*g/4,m*g/4,m*g/4,m*g/4,0.0,0.0,0.0,0.0]
u0 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

init_acceleration = np.array([[0.0],[0.0],[-g],[0.0],[0.0],[0.0]])
#init_acceleration = np.array([[0.0],[0.0],[-9.81],[0.0],[0.0],[0.0]])

#target_point0 = np.array([[2.0],[0.0], [3.0]])
tvp = TVPData(x0, u0, init_acceleration)


controller_tvp_template = mpc_controller.get_tvp_template()
def controller_tvp_fun(t_now):
    for k in range(n_horizon+1):
        controller_tvp_template['_tvp',k,'last_state'] = tvp.x
        controller_tvp_template['_tvp',k,'last_input'] = tvp.u
        controller_tvp_template['_tvp',k,'drone_acc'] = tvp.drone_accel


        return controller_tvp_template
mpc_controller.set_tvp_fun(controller_tvp_fun)
mpc_controller.setup()





#euler_lagrange_simspatial= vertcat(((rotEBMatrixsim[0:3, 0:3] + skew(w_tsim) + skew(w_tsim)@skew(w_tsim))@euler_lagrange_sim[0:3]), euler_lagrange_sim[3],euler_lagrange_sim[4], euler_lagrange_sim[5])
#euler_lagrange_simspatial = euler_lagrange_sim


simulator = do_mpc.simulator.Simulator(mpc_model)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 0.04
}

simulator.set_param(**params_simulator)


sim_tvp_template = simulator.get_tvp_template()
def sim_tvp_fun(t_now):
    sim_tvp_template['last_state'] = tvp.x
    sim_tvp_template['last_input'] = tvp.u
    sim_tvp_template['drone_acc'] = tvp.drone_accel


    return sim_tvp_template
simulator.set_tvp_fun(sim_tvp_fun)

simulator.setup()

estimator = do_mpc.estimator.StateFeedback(mpc_model)



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
#simulator.x0 = x0sim
#estimator.x0 = x0sim

simulator.x0 = x0
estimator.x0 = x0



mpc_controller.set_initial_guess()
dt = .04
curr_roll = 0.0
curr_pitch =0.0
last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
for i in range(100):
    start = time.time()
    
    u0 = mpc_controller.make_step(x0)

    end = time.time()
    print("Computation time: ", end-start)
     
    ynext= simulator.make_step(u0)
    x0 = estimator.make_step(ynext)
    #x0 = vertcat(x0sim[6:12],x0sim[3:6])
    drone_acceleration = (np.array(x0[0:6]) - last_x0_dot )/dt
    tvp.x = x0
    tvp.u = u0
    tvp.drone_accel = drone_acceleration
    #tvp.target_point = np.array([[cos(i*pi/180)*1.],[sin(i*pi/180)*1.],[3.]])
    #tvp.target_point = np.array([[2.0],[0.0], [3.0]])
    #curr_roll = curr_roll + float(x0[3]*dt)
    #curr_pitch = curr_pitch + float(x0[4]*dt)
    

    print("u")
    print(u0)
    #print("\n")
    #print("x")
    #print(x0sim)
    #print("\n")
    #print("a")
    #print(drone_acceleration)
    #print("\n")

   # print(x0[2])

    #print("sep")
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
#ax.plot(t, roll_graph, label='r')
#ax.plot(t, pitch_graph, label='p')
#ax.plot(t, yaw_graph, label='yaw')




# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()
    




