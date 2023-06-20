import os
import mujoco as mj
from mujoco.glfw import glfw
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
xml_path = 'quadrotor.xml' #xml file (assumes this is in the same folder as this file)
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

m = 2.  # drone_mass
g = 9.81
arm_length = .2212
Ixx = 1.
Iyy = 1.
Izz = 1.


model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None
u = None
x = None

with open("control_strategies/mpc/12_states_linear_controller.py") as f:
        exec(f.read())

with open("control_strategies/mpc/12_states_nonlin_sim.py") as f:
        exec(f.read())


def init_controller(model, data):
    global  mpc_controller, simulator, estimator, desired_velocities, last_x0_dot,x0
    
    


    mpc_controller = mpc_global_controller.controller
    simulator = global_simulator.sim
    estimator = global_simulator.est

    mpc_controller.set_initial_guess()
    simulator.set_initial_guess()
    last_x0_dot = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

    desired_velocities = np.array([[0.0,0.0,0.4],[0.0,0.0,0.0],[0.2, 0.0, 0.0],[0.5, 0.0, 0.0],[0.1, 0.0, 0.0],[0.0, -0.1, 0.0], [0.0,-0.5,0.0], [0.0,-0.1,0.0],[0.1,0.0,0.0], [0.5,0.0,0.0]  ])
    x0 = tvp.x


def norm_vec(x1,x2):
    sum1 =0
    for i in range(len(x1)):
        sum1 = (x1[i] -x2[i])**2 + sum1
    sum1 = sqrt(sum1) 
    return sum1


def controller(model, data ):
    global  mpc_controller, mpc_model, waypoints, curr_waypoint,u_val, last_x0_dot,x0
    dt = 0.04
    start = time.time()
        
    u0 = mpc_controller.make_step(x0)

    end = time.time()
    print("Computation time: ", end-start)
        
    ynext= simulator.make_step(u0)
    x0 = estimator.make_step(ynext)
    print("sim")
    # sim is pos, theta, dpos, dtheta
    # controller is dpos, dtheta, theta, pos 
    drone_acceleration = (np.array(x0[6:12]) - last_x0_dot )/dt
    tvp.x = x0
    tvp.u = u0
    tvp.drone_accel = drone_acceleration
    tvp.target_velocity = [0.0,0.0,0.2]
    #tvp.target_velocity = [0.3, 0.0, 0.0]
    print("target velocity is ", tvp.target_velocity)
        
    print("u")
    print(u0)
    apply_control(data,u0)
        # print("\n")
        # print("x")
        # print(x0)
        # print("\n")
        # print("a")
        # print(drone_acceleration)
    last_x0_dot = np.array(x0[6:12])

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
    
    if act == glfw.PRESS and key == glfw.KEY_UP:
        des_height += 0.1
    
    if act == glfw.PRESS and key == glfw.KEY_DOWN:
        des_height -= 0.1
    
    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        desiredangle += 0.01
    
    if act == glfw.PRESS and key == glfw.KEY_LEFT:
        desiredangle -= 0.01



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

#get the full path
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

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;
    


    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
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
