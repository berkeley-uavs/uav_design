import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import cvxpy as cp
import sympy as sp
from sympy.matrices import Matrix

import os

xml_path = 'quadrotor.xml' #xml file (assumes this is in the same folder as this file)
simend = 200 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)


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

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
     #initialize the controller here. This function is called once, in the beginning
    global f, xdot_des, xdot_val, angular_velocity, x_val,roll_angle, pitch_angle,r, p,x
    m = 1.0
    g = 9.81
    n = 8
    r =0.0
    p =0.0
    Ixx = Iyy = Izz = 1.0
    (w_1, w_2, w_3, roll_angle,pitch_angle) = sp.symbols('w_1, w_2, w_3, roll_angle, pitch_angle')
    x = cp.Variable(n)
    x_d2 = (x[1]* sp.sin(x[5]) - x[3]*sp.sin(x[7]) - m*g*sp.sin(pitch_angle))/m
    y_d2 = (x[0]* sp.sin(x[4]) - x[2]*sp.sin(x[6]) - m*g*sp.sin(roll_angle))/m
    z_d2 = (x[0]* sp.cos(x[4]) + x[1]*sp.cos(x[5]) + x[2]*sp.cos(x[6]) + x[3]*sp.cos(x[7])- m*g*sp.cos(roll_angle)*sp.cos(pitch_angle))/m
    n_r = (x[1]* sp.cos(x[5]) - x[3]*sp.cos(x[7]))
    n_p = (x[0]* sp.cos(x[4]) -x[2]*sp.cos(x[6]))
    n_y = (x[0]* sp.sin(x[5]) + x[3]*sp.sin(x[7])+x[2]*sp.sin(x[6]) + x[1]*sp.sin(x[5]))
    moments = Matrix([[n_r], [n_p], [n_y]])
    Inertia_matrix = Matrix([[Ixx, 0, 0], [ 0, Iyy, 0], [0, 0, Izz]])
    angular_velocity = Matrix([[w_1], [w_2], [w_3]])
    omega = Matrix([[0, -w_3, w_2], [w_3, 0, -w_1], [-w_2, w_1,0]])
    angular_accelerations = Inertia_matrix.inv() *(moments - omega*Inertia_matrix*angular_velocity )
    xdot_val = Matrix( [[0],[0],[0],[0],[0],[0]])
    xdot_des = Matrix( [[.0],[0.0] ,[.3] ,[0.0] ,[0.1] ,[0.0]])
    des_vel = Matrix( [[0.0],[0.0] ,[0.1] ,[0.0] ,[0.0] ,[0.0]])

    f = Matrix([[x_d2], [y_d2], [z_d2], [angular_accelerations[0]], [angular_accelerations[1]], [angular_accelerations[2]]])
    x_val = Matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ])

    pass

def controller(model, data):
    global f, xdot_des, xdot_val, angular_velocity, x_val,roll_angle, pitch_angle,r, p,x
    (w_1, w_2, w_3, roll_angle,pitch_angle) = sp.symbols('w_1, w_2, w_3, roll_angle, pitch_angle')

    m = 6
    n = 8
    A = np.random.randn(m,n)
    b = np.random.randn(m)
    f.subs([(roll_angle, r), (pitch_angle, p), (w_1, x_val[3]),(w_2, x_val[4]),(w_3, x_val[5])])
    func = .5 *(np.transpose(f - xdot_des ))*(f - xdot_des )
    cp.Problem(cp.Minimize(func), [A*x == b]).solve()
    print("Optimal value from CVXPY: {}".format(func.value))
    x_val = Matrix([[data.sensordata[11]],[data.sensordata[12]],[data.sensordata[13]],[data.sensordata[14]],[data.sensordata[15]],[data.sensordata[16]] ])


    
def set_torque_servo(actuator_no, flag):
    if (flag==0):
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1

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
