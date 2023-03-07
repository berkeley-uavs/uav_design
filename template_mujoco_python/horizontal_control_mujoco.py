import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
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

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    global Gain_vert, Gain_yaw, Gain_pitch, Gain_roll, Gain_x, curr_height, curr_x, curr_y, ang0, ang2
    Gain_vert = .025
    Gain_x = .025
    Gain_y = 0.025
    curr_height = 0.0
    curr_x = 0.0
    curr_y = 0.0
    ang1 =0.0
    ang2 = 0.0
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    global Gain_vert, Gain_yaw, Gain_pitch, Gain_roll, Gain_x, curr_height, curr_x, ang1, ang2

    #keep raw pitch and roll at 0 -> then solve for linear motion

    #vertical cascade
    des_height = 0.5
    t_const_v = 1.0
    
    des_vel = (des_height - curr_height/t_const_v)
    act_vel = data.qvel[2]
    diff_old = (des_vel - act_vel)
    vert_cntrl = Gain_vert*diff_old
    if(des_vel <=0):
        vert_cntrl = 0
    total_vert_cntrl = 4* vert_cntrl

    # data.ctrl[4] = 10
    if abs((des_vel - act_vel)) - abs(diff_old):
        Gain_vert = Gain_vert - 0.001
    else:
        Gain_vert = Gain_vert + 0.001
    curr_height = data.qpos[2]
    print(curr_height, des_vel, act_vel, Gain_vert, vert_cntrl)

    #x_cascade 

    des_x = 0.5
    t_const_v = 1.0
    des_vel_x = (des_x - curr_x/t_const_v)
    act_vel_x = data.qvel[0]
    diff_old_x =(des_vel_x - act_vel_x)
    hor_cntrl = Gain_x*diff_old
    if(des_vel_x <=0):
        hor_cntrl = 0
    

    # data.ctrl[4] = 10
    if abs((des_vel_x - act_vel_x)) - abs(diff_old_x):
        Gain_x = Gain_x - 0.001
    else:
        Gain_x = Gain_x + 0.001
    curr_x = data.qpos[0]
    print(curr_x, des_vel_x, act_vel_x, Gain_x, hor_cntrl)


    #y_cascade 

    hor_cntrl_y = -hor_cntrl


    #roll_cascade
    cntrl_roll = 0
    cntrl_pitch = 0
    cntrl_yaw = 0

    

    des_cntrls = np.array([[hor_cntrl] [hor_cntrl_y] [vert_cntrl] [cntrl_roll] [cntrl_pitch] [cntrl_yaw]])


    mixer_matrix = np.array([(cos(np.pi/4)*cos(curr_ang0)),0, (cos(np.pi/4)*cos(curr_ang2)), 0, 0o ])

    data.ctrl[4] = (ang0 - curr_ang0)/.25
    data.ctrl[5] = (ang2 - curr_ang2)/.25
    pass

def keyboard(window, key, scancode, act, mods):
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
