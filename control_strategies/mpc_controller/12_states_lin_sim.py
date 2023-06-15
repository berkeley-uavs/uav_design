import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from global_vars_mpc import tvp
from global_vars_mpc import global_simulator




m = 1.8  # drone_mass
g = 9.81
arm_length = .2286
Ixx = 1.0
Iyy = 1.0
Izz = 1.0


model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
u = None
x = None

#second order taylor series approx of sin
def sinTE(x):
    return x - ((x)**3)/6
    #return sin(x)
def cosTE(x):
    return 1 -(x**2)/2
    #return cos(x)

def rotBE(r,p,y):    
    rotBErow1 = horzcat(
                            (cos(y)*cos(p)), 
                            (sin(y)*cos(p)), 
                            (-sin(p)))
    rotBErow2 = horzcat(
                            (cos(y)*sin(p) *sin(r) - sin(y)*cos(r)), 
                            (sin(y)*sin(p) * sin(r) + cos(y)*cos(r)),
                            (cos(p)*sin(r)))
    rotBErow3 = horzcat(
                            (cos(y)*sin(p) * cos(r) + sin(y)*sin(r)), 
                            (sin(y)*sin(p) * cos(r) - cos(y)*sin(r)), 
                            (cos(p)*cos(r)))
    
    rotBEm = vertcat(rotBErow1, rotBErow2,rotBErow3)
    return rotBEm
    

def rotEB(r,p,y):
    rotEBm = transpose(rotBE(r,p,y))
    return rotEBm


def f_acc(T1, T2, T3, T4, tilt1, tilt2, tilt3, tilt4,droll, dpitch,dyaw, euler_roll, euler_pitch, euler_yaw ):
    f_acc = vertcat(

    (T2*sinTE(tilt2) - T4*sinTE(tilt4) - m*g*sin(euler_pitch))/m, # 1
    
    (T1*sinTE(tilt1) - T3*sinTE(tilt3) - m*g*sin(euler_roll))/m, # 2
    
    (T1*cosTE(tilt1) + T2*cosTE(tilt2) + T3*cosTE(tilt3) + T4*cosTE(tilt4) - m*g*cos(euler_roll)*cos(euler_pitch))/m, # 3
    
    ((T2*cosTE(tilt2)*arm_length) - (T4*cosTE(tilt4)*arm_length) + (Iyy*dpitch*dyaw + Izz*dpitch*dyaw))/Ixx, # 4    
    
    (T1*cosTE(tilt1)*arm_length - T3*cosTE(tilt3)*arm_length + (-Ixx*droll*dyaw + Izz*droll*dyaw))/Iyy, # 5
    
    (T1*sinTE(tilt1)*arm_length + T2*sinTE(tilt2)*arm_length + T3*sinTE(tilt3)*arm_length + T4*sinTE(tilt4)*arm_length + (Ixx*droll*dpitch - Iyy*droll*dpitch))/Izz) # 6)
    return f_acc
#T_dot and T are for finding euler angle angular accelerations 

def T_dot(euler_roll, euler_pitch, euler_yaw, droll_euler, dpitch_euler, dyaw_euler):
    T_dot = vertcat(
        horzcat(0,      
            (cos(euler_roll)*droll_euler*tan(euler_pitch) + dpitch_euler*sin(euler_roll)*1/cos(euler_pitch)**2),
            (-sin(euler_roll)*droll_euler*tan(euler_pitch) + dpitch_euler*cos(euler_roll)*1/cos(euler_pitch)**2)),

        horzcat(0,      
            (droll_euler*-sin(euler_roll)), 
            (droll_euler*-cos(euler_roll))),
            
        horzcat(0,      
            (cos(euler_roll)*droll_euler*1/cos(euler_pitch) + tan(euler_pitch)*dpitch_euler*sin(euler_roll)*1/cos(euler_pitch)),      
            (sin(euler_roll)*droll_euler*1/cos(euler_pitch) + tan(euler_pitch)*dpitch_euler*cos(euler_roll)*1/cos(euler_pitch))))
    return T_dot


def T(euler_roll, euler_pitch, euler_yaw):
    T = vertcat(
    horzcat(1, sin(euler_roll)*tan(euler_pitch), cos(euler_roll)*tan(euler_pitch)),
    horzcat(0,cos(euler_roll), - sin(euler_roll)),
    horzcat(0, sin(euler_roll)/cos(euler_pitch), cos(euler_roll)/cos(euler_pitch)))
    return T
# STATES
#dtheta is in terms of BODY ANGULAR VELOCITIES, while euler_ang is in terms of SPATIAL EULER ANGLES
pos = mpc_model.set_variable('_x',  'pos', (3, 1))
euler_ang = mpc_model.set_variable('_x',  'euler_ang', (3, 1))
dpos = mpc_model.set_variable('_x',  'dpos', (3, 1))
dtheta = mpc_model.set_variable('_x',  'dtheta', (3, 1))
# INPUTS
u_th = mpc_model.set_variable('_u',  'u_th', (4, 1))
u_ti = mpc_model.set_variable('_u',  'u_ti', (4, 1))

# ALGEBRAIC TERMS
ddpos = mpc_model.set_variable('_z',  'ddpos', (3, 1))
ddtheta = mpc_model.set_variable('_z',  'ddtheta', (3, 1))

#TIME VARYING PARAMETERS
last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(12, 1))
last_input = mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(8, 1))
last_acc = mpc_model.set_variable(var_type='_tvp', var_name='last_acc',shape=(6, 1))

# Continuous variables -xyz pos, dx dy dz, and euler roll pitch yaw are spatial, while droll, dpitch, dyaw are body rates



#states
xpos_cont = pos[0]
ypos_cont = pos[1]
zpos_cont = pos[2]

euler_roll_cont = euler_ang[0]
euler_pitch_cont = euler_ang[1]
euler_yaw_cont = euler_ang[2]

dx_cont = dpos[0]
dy_cont = dpos[1]
dz_cont = dpos[2] 


droll_cont = dtheta[0]
dpitch_cont = dtheta[1]
dyaw_cont = dtheta[2]


#accelerations
ddx_cont = ddpos[0]
ddy_cont = ddpos[1]
ddz_cont = ddpos[2]

ddroll_cont = ddtheta[0]
ddpitch_cont = ddtheta[1]
ddyaw_cont = ddtheta[2]


# Inputs
T1_cont = u_th[0]
T2_cont = u_th[1]
T3_cont = u_th[2]
T4_cont = u_th[3]
tilt1_cont = u_ti[0]
tilt2_cont = u_ti[1]
tilt3_cont = u_ti[2]
tilt4_cont = u_ti[3]

euler_ang_vel_cont = vertcat(
                            (droll_cont + 
                            dyaw_cont*cos(euler_roll_cont)*tan(euler_pitch_cont) + 
                            dpitch_cont*sin(euler_roll_cont)*tan(euler_pitch_cont)),

                            (dpitch_cont*cos(euler_roll_cont) - 
                            dyaw_cont*sin(euler_roll_cont)),

                            ((dyaw_cont*cos(euler_roll_cont)/(cos(euler_pitch_cont))) + 
                            dpitch_cont*(sin(euler_roll_cont)/cos(euler_pitch_cont)))
)

mpc_model.set_rhs('pos', dpos)
mpc_model.set_rhs('dpos', ddpos)
mpc_model.set_rhs('euler_ang', euler_ang_vel_cont)
mpc_model.set_rhs('dtheta', ddtheta)


f_bodyacc_cont = f_acc(T1_cont, T2_cont,T3_cont,T4_cont, tilt1_cont,tilt2_cont,tilt3_cont,tilt4_cont,droll_cont,dpitch_cont,dyaw_cont, euler_roll_cont,euler_pitch_cont,euler_yaw_cont )

w_euler_cont = vertcat(euler_ang_vel_cont)
droll_euler_cont = w_euler_cont[0] 
dpitch_euler_cont = w_euler_cont[1]
dyaw_euler_cont = w_euler_cont[2]

v_b_cont = vertcat(dx_cont,dy_cont,dz_cont)#dx,dy,dz
alpha_b_cont = vertcat(ddroll_cont, ddpitch_cont, ddyaw_cont)
r_b_cont = vertcat(xpos_cont, ypos_cont, zpos_cont)#pos



T_cont = T(euler_roll_cont, euler_pitch_cont, euler_yaw_cont)
T_dot_cont = T_dot(droll_euler_cont, dpitch_euler_cont, dyaw_euler_cont,euler_roll_cont, euler_pitch_cont, euler_yaw_cont)

alpha_euler_cont = T_cont@alpha_b_cont + T_dot_cont@vertcat(droll_cont, dpitch_cont, dyaw_cont)
rotEBMatrix_cont = rotEB(euler_roll_cont, euler_pitch_cont, euler_yaw_cont)

fspatial_linear_acc_cont = vertcat((rotEBMatrix_cont@(f_bodyacc_cont[0:3])) + 2 * skew(w_euler_cont)@v_b_cont + skew(alpha_euler_cont)@r_b_cont + skew(w_euler_cont)@(skew(w_euler_cont)@r_b_cont))
fspatial_rotation_acc_cont = vertcat(f_bodyacc_cont[3:6]) 
fspatial_acc_cont = vertcat(fspatial_linear_acc_cont, fspatial_rotation_acc_cont)

# TVP   - xyz pos, dx dy dz, and euler roll pitch yaw are spatial, while droll, dpitch, dyaw are body rates

#states
xpos_tvp = last_state[0]
ypos_tvp = last_state[1]
zpos_tvp = last_state[2]

euler_roll_tvp = last_state[3]
euler_pitch_tvp = last_state[4]
euler_yaw_tvp = last_state[5]

dx_tvp = last_state[6]
dy_tvp = last_state[7]
dz_tvp = last_state[8] 

droll_tvp = last_state[9]
dpitch_tvp = last_state[10]
dyaw_tvp = last_state[11]


#accelerations
ddx_tvp = last_acc[0]
ddy_tvp = last_acc[1]
ddz_tvp = last_acc[2]

ddroll_tvp = last_acc[3]
ddpitch_tvp = last_acc[4]
ddyaw_tvp = last_acc[5]


# Inputs
T1_tvp = last_input[0]
T2_tvp = last_input[1]
T3_tvp = last_input[2]
T4_tvp = last_input[3]
tilt1_tvp = last_input[4]
tilt2_tvp = last_input[5]
tilt3_tvp = last_input[6]
tilt4_tvp = last_input[7]

#would have to change to add roll and pitch for g term (still from last state input ig)
f_bodyacc_tvp = f_acc(T1_tvp, T2_tvp,T3_tvp,T4_tvp, tilt1_tvp,tilt2_tvp,tilt3_tvp,tilt4_tvp,droll_tvp,dpitch_tvp,dyaw_tvp, euler_roll_tvp,euler_pitch_tvp,euler_yaw_tvp )


euler_ang_vel_tvp = vertcat(
                                (droll_tvp + dyaw_tvp*cos(euler_roll_tvp)*tan(euler_pitch_tvp) + dpitch_tvp*sin(euler_roll_tvp)*tan(euler_pitch_tvp)),

                                (dpitch_tvp*cos(euler_roll_tvp) - dyaw_tvp*sin(euler_roll_tvp)),

                                ((dyaw_tvp*cos(euler_roll_tvp)/(cos(euler_pitch_tvp))) + dpitch_tvp*(sin(euler_roll_tvp)/cos(euler_pitch_tvp)))
)

w_euler_tvp = vertcat(euler_ang_vel_tvp)
droll_euler_tvp = w_euler_tvp[0] 
dpitch_euler_tvp = w_euler_tvp[1]
dyaw_euler_tvp = w_euler_tvp[2]

v_b_tvp = vertcat(dx_tvp,dy_tvp,dz_tvp)#dx,dy,dz
alpha_b_tvp = vertcat(ddroll_tvp, ddpitch_tvp, ddyaw_tvp)
r_b_tvp = vertcat(xpos_tvp, ypos_tvp, zpos_tvp)#pos



T_tvp = T(euler_roll_tvp, euler_pitch_tvp, euler_yaw_tvp)
T_dot_tvp = T_dot(droll_euler_tvp, dpitch_euler_tvp, dyaw_euler_tvp,euler_roll_tvp, euler_pitch_tvp, euler_yaw_tvp)

alpha_euler_tvp = T_tvp@alpha_b_tvp + T_dot_tvp@vertcat(droll_tvp,dpitch_tvp,dyaw_tvp)
rotEBMatrix_tvp = rotEB(euler_roll_tvp, euler_pitch_tvp, euler_yaw_tvp)

fspatial_linear_acc_tvp = vertcat((rotEBMatrix_tvp@(f_bodyacc_tvp[0:3])) + 2 * skew(w_euler_tvp)@v_b_tvp + skew(alpha_euler_tvp)@r_b_tvp + skew(w_euler_tvp)@(skew(w_euler_tvp)@r_b_tvp))
#spatial_linear_acc_tvp = vertcat((rotEBMatrix_tvp@(f_bodyacc_tvp[0:3])))
fspatial_rotation_acc_tvp = vertcat(f_bodyacc_tvp[3:6]) 
fspatial_acc_tvp = vertcat(fspatial_linear_acc_tvp, fspatial_rotation_acc_tvp)
#print(fspatial_acc_tvp)
#print("/n")
#print(fspatial_acc_cont)

u_vec_cont = vertcat(
    u_th,
    u_ti
)
state_vec_cont = vertcat(
    pos,
    euler_ang,
    dpos,
    dtheta
)
result_vec_cont = vertcat(ddx_cont, ddy_cont, ddz_cont, ddroll_cont, ddpitch_cont, ddyaw_cont)

A = jacobian(last_acc -fspatial_acc_tvp, last_state)
print((A.shape))
B = jacobian(last_acc - fspatial_acc_tvp, last_input)
print((B.shape))
C = jacobian(last_acc - fspatial_acc_tvp, last_acc)
print(C.shape)



#euler_lagrange =  (result_vec_cont -fspatial_acc_cont)
euler_lagrange = C@(result_vec_cont-last_acc) +(A@(state_vec_cont-last_state)) +(B@(u_vec_cont-last_input)) +(last_acc - fspatial_acc_tvp)  

mpc_model.set_alg('euler_lagrange', euler_lagrange)







#-----------------------Model Parameters----------

mpc_model.setup()



simulator = do_mpc.simulator.Simulator(mpc_model)

simulator_tvp_template = simulator.get_tvp_template()
def simulator_tvp_fun(t_now):
    
    simulator_tvp_template['last_state'] = tvp.x
    simulator_tvp_template['last_input'] = tvp.u
    simulator_tvp_template['last_acc'] = tvp.drone_accel


    return simulator_tvp_template

simulator.set_tvp_fun(simulator_tvp_fun)


params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 0.04
}

simulator.set_param(**params_simulator)

simulator.setup()

estimator = do_mpc.estimator.StateFeedback(mpc_model)

x0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
u0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
sim_acc = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).T
simulator.x0 = x0sim
estimator.x0 = x0sim

simulator.u0 = u0sim
simulator.z0 = sim_acc

estimator.u0 = u0sim
estimator.z0 = sim_acc

global_simulator.sim = simulator
global_simulator.est = estimator







