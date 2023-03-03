import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import sympy as sp
from sympy.matrices import Matrix
import math
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
    global f,u, u_val, xdot_des, xdot_val
    m = 1.0
    g = 9.81
    Ixx = Iyy = Izz = 1.0
    pitch_angle =0.0
    roll_angle =0.0
    (T1, T2, T3, T4, theta1, theta2, theta3, theta4) = sp.symbols('T1,T2,T3,T4,theta1,theta2,theta3,theta4')
    x_d2 = (T2* sp.sin(theta2) - T4*sp.sin(theta4) - m*g*sp.sin(pitch_angle))/m
    y_d2 = (T1* sp.sin(theta1) - T3*sp.sin(theta3) - m*g*sp.sin(roll_angle))/m
    z_d2 = (T1* sp.cos(theta1) + T2*sp.cos(theta2) + T3*sp.cos(theta3) + T4*sp.cos(theta4)- m*g*sp.cos(pitch_angle))/m
    r_d2 = (T2* sp.cos(theta2) - T4*sp.cos(theta4))/Ixx
    p_d2 = (T1* sp.cos(theta1) -T3*sp.cos(theta3))/Iyy
    yaw_d2 = (T1* sp.sin(theta1) + T4*sp.sin(theta4)+T3*sp.sin(theta3) + T2*sp.sin(theta2))/Izz
    xdot_val = Matrix( [[0],[0],[0],[0],[0],[0]])
    xdot_des = Matrix( [[0.1],[0] ,[0.3] ,[0] ,[0] ,[0]])
    u_val = Matrix([[0.0], [0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0]])
    u = Matrix([ [T1], [T2] ,[T3] ,[T4] ,[theta1] ,[theta2] ,[theta3] ,[theta4]])
    f = Matrix([[x_d2], [y_d2], [z_d2], [r_d2], [p_d2], [yaw_d2]])
    pass

def controller(model, data):
#put the controller here. This function is called inside the simulation.
    global f,u, u_val, xdot_des, xdot_val
    (T1, T2, T3, T4, theta1, theta2, theta3, theta4) = sp.symbols('T1,T2,T3,T4,theta1,theta2,theta3,theta4')
    J = f.jacobian(u).subs([(T1,u_val[0]), (T2,u_val[1]),(T3,u_val[2]), (T4,u_val[3]),(theta1,u_val[4]), (theta2,u_val[5]), (theta3,u_val[6]), (theta4,u_val[7])])
    J_inv = J.pinv()

    u_val = (J_inv*(xdot_des - xdot_val)) + u_val


    xdot_val = Matrix([[0.0], [0.0], [0.0] ,[0.0] ,[0.0] ,[0.0]])


    tiltangle1 = data.sensordata[0]
    tiltvel1 = data.sensordata[1]

    tiltangle2 = data.sensordata[2]
    tiltvel2 = data.sensordata[3]
    tiltangle3 = data.sensordata[4]
    tiltvel3 = data.sensordata[5]

    tiltangle4 = data.sensordata[6]
    tiltvel4 = data.sensordata[7]
    Kp = .005
    Kd = Kp/10
    control1 = -Kp*(tiltangle1-u_val[4]) - Kd*tiltvel1 # position control
    control2 = -Kp*(tiltangle2-u_val[5]) - Kd*tiltvel2 # position control
    control3 = -Kp*(tiltangle3-u_val[6]) - Kd*tiltvel3 # position control
    control4 = -Kp*(tiltangle4-u_val[7]) - Kd*tiltvel4 # position control

    data.ctrl[0] = u_val[0]
    data.ctrl[1] = u_val[1]
    data.ctrl[2] = u_val[2]
    data.ctrl[3] = u_val[3]
    data.ctrl[4] = u_val[4]
    data.ctrl[5] = u_val[5]
    data.ctrl[6] = u_val[6]
    data.ctrl[7] = u_val[7]
    

    # print(curr_height, des_vel, act_vel, Gain, cntrl)



    
   

    
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