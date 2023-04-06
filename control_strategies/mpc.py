import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import do_mpc
from casadi import *

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

m = 1  # drone_mass
g = 9.81
arm_length = 1
Ixx = Iyy = Izz = 1


model_type = "continuous"
mpc_model = do_mpc.model.Model(model_type)
mpc_controller = None
estimator = None


def init_controller(model, data):
    model_type = "continuous"
    mpc_model = do_mpc.model.Model(model_type)

    pos = mpc_model.set_variable('states',  'pos', (3, 1))
    theta = mpc_model.set_variable('states',  'theta', (3, 1))

    dpos = mpc_model.set_variable('states',  'dpos', (3, 1))
    dtheta = mpc_model.set_variable('states',  'dtheta', (3, 1))

    u = mpc_model.set_variable('inputs',  'u', (8, 1))

    ddpos = mpc_model.set_variable('algebraic',  'ddpos', (3, 1))
    ddtheta = mpc_model.set_variable('algebraic',  'ddtheta', (3, 1))

    mpc_model.set_rhs('pos', dpos)
    mpc_model.set_rhs('theta', dtheta)
    mpc_model.set_rhs('dpos', ddpos)
    mpc_model.set_rhs('dtheta', ddtheta)

    T1 = u[0]
    T2 = u[1]
    T3 = u[2]
    T4 = u[3]
    theta1 = u[4]
    theta2 = u[5]
    theta3 = u[6]
    theta4 = u[7]

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

    euler_lagrange = vertcat(
        # 1
        m*ddx - T2*sin(theta2) + T4*sin(theta4) + g*sin(pitch),
        # 2
        m*ddy - T1*sin(theta1) + T3*sin(theta3) + g*sin(roll),
        # 3
        m*ddz - T1*cos(theta1) - T2*cos(theta2) - T3*cos(theta3) - T4*cos(theta4) - g*cos(roll)*cos(pitch),
        # 4
        Ixx*ddroll - (T2*cos(theta2)*arm_length) + (T4*cos(theta4)*arm_length) - (Iyy*dpitch*dy - Izz*dpitch*dy),
        # 5
        Iyy*ddpitch - T1*cos(theta1)*arm_length + T3*cos(theta3)*arm_length - (-Ixx*droll*dy + Izz*droll*dy),
        # 6
        Izz*ddyaw - T1*sin(theta1)*arm_length - T2*sin(theta2)*arm_length - T3*sin(theta3)*arm_length - T4*sin(theta4)*arm_length - (Ixx*droll*dpitch - Iyy*droll*dpitch)

    )

    mpc_model.set_alg('euler_lagrange', euler_lagrange)
    
    mpc_model.set_expression(expr_name='cost', expr=sum1((pos - [1, 5, 0])**2 + theta**2))

    mpc_model.setup()


    mpc_controller = do_mpc.controller.MPC(mpc_model)

    setup_mpc = {
        'n_horizon': 10,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.04,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    
    mpc_controller.set_param(**setup_mpc)


    mterm = mpc_model.aux['cost']
    lterm = mpc_model.aux['cost']

    mpc_controller.set_objective(mterm=mterm, lterm=lterm)
    # Input force is implicitly restricted through the objective.
    # mpc_controller.set_rterm(force=0.1)
    # mpc_controller.bounds['lower','_u','force'] = -4
    # mpc_controller.bounds['upper','_u','force'] = 4
    mpc_controller.setup()
    estimator = do_mpc.estimator.StateFeedback(mpc_model)
    
    x0 = np.array([[0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    mpc_controller.x0 = x0
    estimator.x0 = x0
    mpc_controller.set_initial_guess()

    u0 = mpc_controller.make_step(x0)
    print(u0)

    pass


def controller(model, data):

    # Kp = .005
    # Kd = Kp/10

    # control1 = -Kp*(tiltangle1-0) - Kd*tiltvel1  # position control
    # control2 = -Kp*(tiltangle2-0) - Kd*tiltvel2  # position control
    # control3 = -Kp*(tiltangle3-0) - Kd*tiltvel3  # position control
    # control4 = -Kp*(tiltangle4-0) - Kd*tiltvel4  # position control

    # data.ctrl[0] = u_val[0]
    # data.ctrl[1] = -u_val[1]
    # data.ctrl[2] = u_val[2]
    # data.ctrl[3] = -u_val[3]
    # data.ctrl[4] = control1
    # data.ctrl[5] = control2
    # data.ctrl[6] = control3
    # data.ctrl[7] = control4
    pass


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
