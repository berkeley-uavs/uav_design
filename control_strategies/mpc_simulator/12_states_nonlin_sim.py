import numpy as np
import do_mpc
from casadi import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


m = 1.8  # drone_mass
g = 9.81
arm_length = .2286
Ixx = 1.0
Iyy = 1.0
Izz = 1.0


model_type = "continuous"
mpc_modelsim = do_mpc.model.Model(model_type)
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
pos = mpc_modelsim.set_variable('_x',  'pos', (3, 1))
euler_ang = mpc_modelsim.set_variable('_x',  'euler_ang', (3, 1))
dpos = mpc_modelsim.set_variable('_x',  'dpos', (3, 1))
dtheta = mpc_modelsim.set_variable('_x',  'dtheta', (3, 1))
# INPUTS
u_th = mpc_modelsim.set_variable('_u',  'u_th', (4, 1))
u_ti = mpc_modelsim.set_variable('_u',  'u_ti', (4, 1))

# ALGEBRAIC TERMS
ddpos = mpc_modelsim.set_variable('_z',  'ddpos', (3, 1))
ddtheta = mpc_modelsim.set_variable('_z',  'ddtheta', (3, 1))



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

mpc_modelsim.set_rhs('pos', dpos)
mpc_modelsim.set_rhs('dpos', ddpos)
mpc_modelsim.set_rhs('euler_ang', euler_ang_vel_cont)
mpc_modelsim.set_rhs('dtheta', ddtheta)


f_bodyacc_cont = f_acc(T1_cont, T2_cont,T3_cont,T4_cont, tilt1_cont,tilt2_cont,tilt3_cont,tilt4_cont,droll_cont,dpitch_cont,dyaw_cont, euler_roll_cont,euler_pitch_cont,euler_yaw_cont )

w_euler_cont = vertcat(euler_ang_vel_cont)
droll_euler_cont = w_euler_cont[0] 
dpitch_euler_cont = w_euler_cont[1]
dyaw_euler_cont = w_euler_cont[2]

v_b_cont = vertcat(dx_cont,dy_cont,dz_cont)#dx,dy,dz
alpha_b_cont = vertcat(ddroll_cont, ddpitch_cont, ddyaw_cont)
r_b_tvp = vertcat(xpos_cont, ypos_cont, zpos_cont)#pos



T_cont = T(euler_roll_cont, euler_pitch_cont, euler_yaw_cont)
T_dot_cont = T_dot(droll_euler_cont, dpitch_euler_cont, dyaw_euler_cont,euler_roll_cont, euler_pitch_cont, euler_yaw_cont)

alpha_euler_cont = T_cont@alpha_b_cont + T_dot_cont@vertcat(droll_cont, dpitch_cont, dyaw_cont)
rotEBMatrix_cont = rotEB(euler_roll_cont, euler_pitch_cont, euler_yaw_cont)

fspatial_linear_acc_cont = vertcat((rotEBMatrix_cont@(f_bodyacc_cont[0:3])))
fspatial_rotation_acc_cont = vertcat(f_bodyacc_cont[3:6]) 
fspatial_acc_cont = vertcat(fspatial_linear_acc_cont, fspatial_rotation_acc_cont)



euler_lagrange = vertcat(ddx_cont, ddy_cont, ddz_cont, ddroll_cont, ddpitch_cont, ddyaw_cont) - fspatial_acc_cont

mpc_modelsim.set_alg('euler_lagrange', euler_lagrange)

#-------simulator parameters ----

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

x0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
u0sim = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
sim_acc = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).T
simulator.x0 = x0sim
estimator.x0 = x0sim

simulator.u0 = u0sim
simulator.z0 = sim_acc

estimator.u0 = u0sim
estimator.z0 = sim_acc





