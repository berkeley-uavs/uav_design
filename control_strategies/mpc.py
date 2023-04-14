import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import do_mpc
from casadi import *
import math

# xml file (assumes this is in the same folder as this file)
xml_path = '../quadrotor.xml'
simend = 200  # simulation time
print_camera_config = 0  # set to 1 to print camera config
# this is useful for initializing view of the model)


# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

curr_height = 0.0
Gain = .025
des_height = 0.5
desiredangle = 0

m = .5  # drone_mass
g = 9.81
arm_length = 1
Ixx = 1.2
Iyy = 1.1
Izz = 1.0


model_type = "discrete"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None
u = None
x = None




def init_controller(model, data):
    global  mpc_controller, mpc_model,waypoints, curr_waypoint,u_val

    pos = mpc_model.set_variable('states',  'pos', (3, 1))
    theta = mpc_model.set_variable('states',  'theta', (3, 1))

    dpos = mpc_model.set_variable('states',  'dpos', (3, 1))
    dtheta = mpc_model.set_variable('states',  'dtheta', (3, 1))

    u_th = mpc_model.set_variable('inputs',  'u_th', (4, 1))
    u_ti = mpc_model.set_variable('inputs',  'u_ti', (4, 1))


    ddpos = mpc_model.set_variable('algebraic',  'ddpos', (3, 1))
    ddtheta = mpc_model.set_variable('algebraic',  'ddtheta', (3, 1))
    target_point = mpc_model.set_variable(var_type='_tvp', var_name='target_point',shape=(3, 1))
    last_state = mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(12, 1))
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
   
    f = vertcat(
        (T2*sin(theta2) - T4*sin(theta4) - m*g*sin(pitch))/m,
        # 2
        (T1*sin(theta1) - T3*sin(theta3) - m*g*sin(roll))/m,
        # 3
        (T1*cos(theta1) + T2*cos(theta2) + T3*cos(theta3) + T4*cos(theta4) - m*g*cos(roll)*cos(pitch))/m,
        # 4
        ((T2*cos(theta2)*arm_length) - (T4*cos(theta4)*arm_length) + (Iyy*dpitch*dy + Izz*dpitch*dy))/Ixx,
        # 5
        (T1*cos(theta1)*arm_length - T3*cos(theta3)*arm_length + (-Ixx*droll*dy + Izz*droll*dy))/Iyy,
        # 6
        (T1*sin(theta1)*arm_length + T2*sin(theta2)*arm_length + T3*sin(theta3)*arm_length + T4*sin(theta4)*arm_length + (Ixx*droll*dpitch - Iyy*droll*dpitch))/Izz
    )


    #euler_lagrange = vertcat(
        # 1
        #m*ddx - T2*sin(theta2) + T4*sin(theta4) + m*g*sin(pitch),
        # 2
        #m*ddy - T1*sin(theta1) + T3*sin(theta3) + m*g*sin(roll),
        # 3
        #m*ddz - T1*cos(theta1) - T2*cos(theta2) - T3*cos(theta3) - T4*cos(theta4) + m*g*cos(roll)*cos(pitch),
        # 4
        #Ixx*ddroll - (T2*cos(theta2)*arm_length) + (T4*cos(theta4)*arm_length) - (Iyy*dpitch*dy - Izz*dpitch*dy),
        # 5
        #Iyy*ddpitch - T1*cos(theta1)*arm_length + T3*cos(theta3)*arm_length - (-Ixx*droll*dy + Izz*droll*dy),
        # 6
        #Izz*ddyaw - T1*sin(theta1)*arm_length - T2*sin(theta2)*arm_length - T3*sin(theta3)*arm_length - T4*sin(theta4)*arm_length - (Ixx*droll*dpitch - Iyy*droll*dpitch)

    #)

    u_vec = vertcat(
        u_th,
        u_ti
    )
    state_vec = vertcat(
        pos,
        theta,
        dpos,
        dtheta,
    )
   
    A = jacobian(f, last_state)
    B = jacobian(f, last_input)
    result_vec = vertcat(
        ddx,
        ddy,
        ddz,
        ddroll,
        ddpitch,
        ddyaw,
    )
    euler_lagrange = (result_vec-drone_acc) - A@(state_vec-last_state) - B@(u_vec-last_input)

    #print(euler_lagrange)
    

    mpc_model.set_alg('euler_lagrange', euler_lagrange)
    mpc_model.set_expression(expr_name='cost', expr=sum1(.9*sqrt((pos[0]-target_point[0])**2 + (pos[1]-target_point[1])**2 + (pos[2]-target_point[2])**2) +.0000000001*sqrt((u_th[0])**2 + (u_th[1])**2 + (u_th[2])**2 + (u_th[3])**2) ))
    mpc_model.set_expression(expr_name='mterm', expr=sum1(.9*sqrt((pos[0]-target_point[0])**2 + (pos[1]-target_point[1])**2 + (pos[2]-target_point[2])**2)))

    mpc_model.setup()


    mpc_controller = do_mpc.controller.MPC(mpc_model)
   

    setup_mpc = {
        'n_horizon': 7,
        'n_robust': 1,
        'open_loop': 0,
        't_step': 0.001,
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
    n_horizon = 7
    def tvp_fun(t_now):
        for k in range(n_horizon+1):
                tvp_template['_tvp',k,'target_point'] = [0.0,0.0,0.0]
                tvp_template['_tvp',k, 'last_state'] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                tvp_template['_tvp',k, 'last_input'] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                tvp_template['_tvp',k, 'drone_acc'] = [0.0,0.0,0.0,0.0,0.0,0.0]
        return tvp_template
    mpc_controller.set_tvp_fun(tvp_fun)

    mterm = mpc_model.aux['mterm']
    lterm = mpc_model.aux['cost']

    mpc_controller.set_objective(mterm=mterm, lterm=lterm)
    # Input force is implicitly restricted through the objective.
    mpc_controller.set_rterm(u_th=1e-4)
    mpc_controller.set_rterm(u_ti=1e-3)

    tilt_limit = pi/2
    thrust_limit = 50
    u_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
    u_lower_limits =  np.array([0, 0, 0, 0])
    u_ti_upper_limits = np.array([tilt_limit, tilt_limit, tilt_limit, tilt_limit])
    u_ti_lower_limits =  np.array([-tilt_limit, -tilt_limit, -tilt_limit, -tilt_limit])

    x_limits = np.array([inf, inf, inf, pi/2, pi/2, pi/2, .1, .1, .1, 1, 1, 1])

    mpc_controller.bounds['lower','_u','u_th'] = u_lower_limits
    mpc_controller.bounds['upper','_u','u_th'] = u_upper_limits
    mpc_controller.bounds['lower','_u','u_ti'] = u_ti_lower_limits
    mpc_controller.bounds['upper','_u','u_ti'] = u_ti_upper_limits

    mpc_controller.bounds['lower','_x','pos'] = -x_limits[0:3]
    mpc_controller.bounds['upper','_x','pos'] = x_limits[0:3]

    mpc_controller.bounds['lower','_x','theta'] = -x_limits[3:6]
    mpc_controller.bounds['upper','_x','theta'] = x_limits[3:6]

    mpc_controller.bounds['lower','_x','dpos'] = -x_limits[6:9]
    mpc_controller.bounds['upper','_x','dpos'] = x_limits[6:9]

    mpc_controller.bounds['lower','_x','dtheta'] = -x_limits[9:12]
    mpc_controller.bounds['upper','_x','dtheta'] = x_limits[9:12]

    mpc_controller.setup()
    

    mpc_controller.x0 = get_drone_state(data)
    mpc_controller.set_initial_guess()
    
    waypoints = []
    curr_waypoint = [0.0,0.0,1.0]
    waypoints.append([.5,.5,1.5])
    waypoints.append([.3,.3,2.0])

    u_val = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


def norm_vec(x1,x2):
    sum1 =0
    for i in range(len(x1)):
        sum1 = (x1[i] -x2[i])**2 + sum1
    sum1 = sqrt(sum1) 
    return sum1


def controller(model, data, ):
    global  mpc_controller, mpc_model, waypoints, curr_waypoint,u_val
    
    x = get_drone_state(data)
    curr_dist = norm_vec(x[0:3], curr_waypoint)
    if( curr_dist< .000000000000000000001):
        curr_waypoint = waypoints.pop(0)
    n_horizon = 7
    x_acc = get_drone_acc(data)
    print(x_acc)
    tvp_template = mpc_controller.get_tvp_template()
    def tvp_fun(t_now):
        for k in range(n_horizon+1):
                tvp_template['_tvp',k,'target_point'] = curr_waypoint
                tvp_template['_tvp',k, 'last_state'] = x
                tvp_template['_tvp',k, 'last_input'] = u_val
                tvp_template['_tvp',k, 'drone_acc'] = x_acc
        return tvp_template
    mpc_controller.set_tvp_fun(tvp_fun)


    u_val = mpc_controller.make_step(x)
    #apply_control(data, [.01, .01, .01, .01, pi/2, pi/2, pi/2, pi/2])
    # u[4] = 0
    # u[5] = 0
    # u[6] = 0
    # u[7] = 0
    print(x[0:6])
    print(u_val)
    print(curr_dist)
    print(curr_waypoint)
    apply_control(data, u_val)

    #print(x[0:6])

    
    

    pass


def RotToRPY(R):
    R=R.reshape(3,3) #to remove the last dimension i.e., 3,3,1
    phi = math.asin(R[1,2])
    psi = math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))
    theta = math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
    return phi,theta,psi

def get_drone_state(data):
    R = data.site_xmat[0].reshape(3,3)
    roll, pitch, yaw = RotToRPY(R)
    x = data.qpos[0]
    y = data.qpos[1]
    z = data.qpos[2]
    
    current_state = [x, y, z, roll, pitch, yaw]
    current_state.extend(get_sensor_data(data)[8:14])
    return np.array(current_state)

def get_drone_acc(data):
    drone_acc = [(data.sensordata[14]),(data.sensordata[15]),(data.sensordata[16]),0.0,0.0,0.0]
    R = data.site_xmat[0].reshape(3,3)
    drone_acc = drone_acc - vertcat(R@(np.array([0.0,0.0,9.81]).T), 0.0,0.0,0.0) 
    return np.array(drone_acc)
def get_sensor_data(data):
    tiltangle1 = data.sensordata[0]
    tiltvel1 = data.sensordata[1]
    tiltangle2 = data.sensordata[2]
    tiltvel2 = data.sensordata[3]
    tiltangle3 = data.sensordata[4]
    tiltvel3 = data.sensordata[5]
    tiltangle4 = data.sensordata[6]
    tiltvel4 = data.sensordata[7]
    x_vel = data.sensordata[8]
    y_vel = data.sensordata[9]
    z_vel = data.sensordata[10]
    pitch_vel = data.sensordata[11]
    roll_vel = data.sensordata[12]
    yaw_vel = data.sensordata[13]
    
    return [tiltangle1, tiltvel1,
            tiltangle2, tiltvel2,
            tiltangle3, tiltvel3,
            tiltangle4, tiltvel4,
            x_vel, y_vel, z_vel,
            pitch_vel, roll_vel, yaw_vel]


def apply_control(data, u):
    sensor_data = get_sensor_data(data)
    Kp = .005
    Kd = Kp/10
    
    tilt1 = -Kp*(sensor_data[0]-u[4]) - Kd*sensor_data[1]  # position control
    tilt2 = -Kp*(sensor_data[2]-u[5]) - Kd*sensor_data[3]  # position control
    tilt3 = -Kp*(sensor_data[4]-u[6]) - Kd*sensor_data[5]  # position control
    tilt4 = -Kp*(sensor_data[6]-u[7]) - Kd*sensor_data[7]  # position control
    data.ctrl[0] = u[0]
    data.ctrl[1] = u[1]
    data.ctrl[2] = u[2]
    data.ctrl[3] = u[3]
    data.ctrl[4] = tilt1
    data.ctrl[5] = tilt2
    data.ctrl[6] = tilt3
    data.ctrl[7] = tilt4





def keyboard(window, key, scancode, act, mods):
    global des_height, desiredangle
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)
    
    if (data.time >= simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    # if (print_camera_config==1):
    #     print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
    #     print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
