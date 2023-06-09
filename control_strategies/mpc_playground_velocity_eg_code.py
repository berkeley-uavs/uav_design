import numpy as np
import sys
from casadi import *
import time

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

m = 1.8
L = 0.2286  # m,  length of tarm
Ixx = 1.2
Iyy = 1.1
Izz = 1.0

g = 9.80665 # m/s^2, Gravity

# h1 = m0 + m1 + m2
# h2 = m1*l1 + m2*L1
# h3 = m2*l2
# h4 = m1*l1**2 + m2*L1**2 + J1
# h5 = m2*l2*L1
# h6 = m2*l2**2 + J2
# h7 = (m1*l1 + m2*L1) * g
# h8 = m2*l2*g


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

pos = model.set_variable('_x',  'pos', (3,1))
theta = model.set_variable('_x',  'theta', (3,1))
dpos = model.set_variable('_x',  'dpos', (3,1))
dtheta = model.set_variable('_x',  'dtheta', (3,1))

u_thrust = model.set_variable('_u',  'thrust', (4,1))
u_tilt = model.set_variable('_u',  'tilt', (4,1))

ddpos = model.set_variable('_z', 'ddpos', (3,1))
ddtheta = model.set_variable('_z', 'ddtheta', (3,1))

model.set_rhs('pos', dpos)
model.set_rhs('theta', dtheta)
model.set_rhs('dpos', ddpos)
model.set_rhs('dtheta', ddtheta)

#second order taylor series approx of sin
def sinTE(x):
    return x - ((x)**3)/6
    #return sin(x)
def cosTE(x):
    return 1 -(x**2)/2
    #return cos(x)

#representing dynamics

T1 = u_thrust[0]
T2 = u_thrust[1]
T3 = u_thrust[2]
T4 = u_thrust[3]
theta1 = u_tilt[0]
theta2 = u_tilt[1]
theta3 = u_tilt[2]
theta4 = u_tilt[3]

x = pos[0]
y = pos[1]
z = pos[2]
roll = theta[0]
pitch = theta[1]
yaw = theta[2]

dx = dpos[0]
dy = dpos[1]
dz = dpos[2]
droll = dtheta[0]
dpitch = dtheta[1]
dyaw = dtheta[2]

ddx = ddpos[0]
ddy = ddpos[1]
ddz = ddpos[2]
ddroll = ddtheta[0]
ddpitch = ddtheta[1]
ddyaw = ddtheta[2]

f = vertcat(
    (T2*sin(theta2) - T4*sin(theta4))/m,
    # 2
    (T1*sin(theta1) - T3*sin(theta3))/m,
    # 3
    (T1*cos(theta1) + T2*cos(theta2) + T3*cos(theta3) + T4*cos(theta4))/m,
    # 4
    ((T2*cos(theta2)*L) - (T4*cos(theta4)*L) + (Iyy*dpitch*dy - Izz*dpitch*dy))/Ixx,
    # 5
    (T1*cos(theta1)*L - T3*cos(theta3)*L + (-Ixx*droll*dy + Izz*droll*dy))/Iyy,
    # 6
    (T1*sin(theta1)*L + T2*sin(theta2)*L + T3*sin(theta3)*L + T4*sin(theta4)*L + (Ixx*droll*dpitch - Iyy*droll*dpitch))/Izz,
 )

euler_lagrange = vertcat(ddx, ddy, ddz, ddroll, ddpitch, ddyaw) - rotEB(roll, pitch, yaw)@f + [0,0,g,0,0,0]
model.set_alg('euler_lagrange', euler_lagrange)

E_kin = m * (dx**2 + dy**2 + dz**2)/2


model.set_expression('E_kin', E_kin)

# Build the model
model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 100,
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
mpc.set_param(**setup_mpc)

mterm = model.aux['E_kin'] # terminal cost
lterm = model.aux['E_kin'] # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)
# Input force is implicitly restricted through the objective.
mpc.set_rterm(thrust=0.1)
mpc.set_rterm(tilt=0.001)

tilt_limit = pi/(2.2)
thrust_limit = 50
u_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
u_lower_limits =  np.array([0.00, 0.00, 0.00, 0.00])
u_ti_upper_limits = np.array([tilt_limit, tilt_limit, tilt_limit, tilt_limit])
u_ti_lower_limits =  np.array([-tilt_limit, -tilt_limit, -tilt_limit, -tilt_limit])

mpc.bounds['lower','_u','thrust'] = -u_lower_limits
mpc.bounds['upper','_u','thrust'] = u_upper_limits
mpc.bounds['lower','_u','tilt'] = u_ti_lower_limits
mpc.bounds['upper','_u','tilt'] = u_ti_upper_limits


mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 0.04
}

simulator.set_param(**params_simulator)

simulator.setup()

x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0])

simulator.x0 = x0
mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

fig = plt.figure(figsize=(16,9))

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1))
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

# ax2.set_ylabel('$E_{kin}$ [J]')
# ax3.set_ylabel('$E_{pot}$ [J]')
# ax4.set_ylabel('Angle  [rad]')
# ax5.set_ylabel('Input force [N]')

# Axis on the right.
for ax in [ax2, ax3, ax4, ax5]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if ax != ax5:
        ax.xaxis.set_ticklabels([])

ax5.set_xlabel('time [s]')

mpc_graphics.add_line(var_type='_aux', var_name='E_kin', axis=ax2)
mpc_graphics.add_line(var_type='_x', var_name='theta', axis=ax4)

ax1.axhline(0,color='black')

bar1 = ax1.plot([],[], '-o', linewidth=5, markersize=10)
bar2 = ax1.plot([],[], '-o', linewidth=5, markersize=10)

ax1.set_xlim(-1.8,1.8)
ax1.set_ylim(-1.2,1.2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()

u0 = mpc.make_step(x0)

# Quickly reset the history of the MPC data object.
mpc.reset_history()

n_steps = 100
for k in range(n_steps):
    start = time.time()
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    end = time.time()
    print("Computation time: ", end-start)

from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

# The function describing the gif:
x_arr = mpc.data['_x']

