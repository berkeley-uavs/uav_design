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
    global f,u, u_val, xdot_des, xdot_val, desired_pos, current_pos, angular_velocity, x_val, des_vel,roll_angle, pitch_angle,r,p,x
    m = 1.0
    g = 9.81
    # Ixx = Iyy = Izz = 1.0
  
    (Ixx, Ixy, Ixz, Iyy, Iyz, Izz, T1, T2, T3, T4, theta1, theta2, theta3, theta4, w_1, w_2, w_3, roll_angle,pitch_angle,dx,dy,dz) = sp.symbols('Ixx, Ixy, Ixz, Iyy, Iyz, Izz,T1,T2,T3,T4,theta1,theta2,theta3,theta4, w_1, w_2, w_3, roll_angle, pitch_angle, dx,dy,dz')
    x_d2 = (T2* sp.sin(theta2) - T4*sp.sin(theta4) - m*g*sp.sin(pitch_angle))/m
    y_d2 = (T1* sp.sin(theta1) - T3*sp.sin(theta3) - m*g*sp.sin(roll_angle))/m
    z_d2 = (T1* sp.cos(theta1) + T2*sp.cos(theta2) + T3*sp.cos(theta3) + T4*sp.cos(theta4)- m*g*sp.cos(roll_angle)*sp.cos(pitch_angle))/m
    n_r = (T2* sp.cos(theta2) - T4*sp.cos(theta4))
    n_p = (T1* sp.cos(theta1) -T3*sp.cos(theta3))
    n_y = (T1* sp.sin(theta1) + T4*sp.sin(theta4)+T3*sp.sin(theta3) + T2*sp.sin(theta2))
    moments = Matrix([[n_r], [n_p], [n_y]])
    Inertia_matrix = Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
    angular_velocity = Matrix([[w_1, w_2, w_3]]).T
    omega = Matrix([[0, -w_3, w_2], [w_3, 0, -w_1], [-w_2, w_1,0]])

    angular_accelerations = Inertia_matrix.inv() * (moments - omega*Inertia_matrix*angular_velocity)
    print(angular_accelerations[0])
    print(angular_accelerations[1])
    print(angular_accelerations[2])


    xdot_val = Matrix( [[0],[0],[0],[0],[0],[0]])
    xdot_des = Matrix( [[1],[.0] ,[5] ,[0.] ,[0.] ,[0.0]])
    des_vel = Matrix( [[0.0],[0.0] ,[0.5] ,[0.0] ,[0.] ,[0.]])
    x = Matrix([[dx], [dy],[dz],[w_1], [w_2], [w_3]])
    r = 0.0
    p =0.0
    u_val = Matrix([[0.0], [0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0] ,[0.0]])
    u = Matrix([ [T1], [T2] ,[T3] ,[T4] ,[theta1] ,[theta2] ,[theta3] ,[theta4]])
    f = Matrix([[x_d2], [y_d2], [z_d2], [angular_accelerations[0]], [angular_accelerations[1]], [angular_accelerations[2]]])
    desired_pos = Matrix([[0.0], [0.0], [0.5], [0.0], [0.0], [0.0] ])
    current_pos = Matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ])
    x_val = Matrix([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ])

def controller(model, data):
#put the controller here. This function is called inside the simulation.
    global f,u, u_val, xdot_des, xdot_val, angular_velocity, x_val, des_vel, roll_angle, pitch_angle,r,p,x
    T_v = 1
    T_a = .2
    g = 9.81
    dt = .01 #sampling rate
    (T1, T2, T3, T4, theta1, theta2, theta3, theta4, w_1, w_2, w_3,dx,dy,dz) = sp.symbols('T1,T2,T3,T4,theta1,theta2,theta3,theta4, w_1, w_2, w_3,dx,dy,dz')
    J = f.jacobian(u).subs([(T1,u_val[0]), (T2,u_val[1]),(T3,u_val[2]), (T4,u_val[3]),(theta1,u_val[4]), (theta2,u_val[5]), (theta3,u_val[6]), (theta4,u_val[7]), (w_1, x_val[3]),(w_2, x_val[4]),(w_3, x_val[5]), (roll_angle, r), (pitch_angle,p)])
    # print(J)
    A = f.jacobian(x).subs([(w_1, x_val[3]),(w_2, x_val[4]),(w_3, x_val[5])])
    J_inv = J.pinv()

    # print("u_val")

    # print(u_val)
    

    # print(u_val)
    feed_forward = (J_inv*(xdot_des- A*x_val) )
    K = Matrix([[0.5, 0.5 ,0.5, 0.5, .5,.5], [.5, .5 ,.5, .5, .5,.5],[.5, .5, .5, .5, .5,.5],[.5, .5, .5, .5, .5,.5],[.5, .5, .5, .5, .5,.5],[.5, .5 ,.5 ,.5 ,.5,.5],[.5, .5 ,.5 ,.5, .5,.5],[.5, .5 ,.5 ,.5 ,.5,.5]])
    Kv = K/4
    feed_back = K*(des_vel-x_val) + Kv*xdot_val
    u_val = feed_forward - feed_back
   
    i =0
    while i < len(u_val):
        if(abs(u_val[i]) <.0001):
            u_val[i] = 0.0
        #if(i<4):
            #%if((u_val[i]) < 0.0):
                #u_val[i] = 0
        if(i>= 4):
            u_val[i]=  np.sign(u_val[i])* abs(math.remainder(u_val[i], 2*math.pi))
            #if (abs(u_val[i]) > math.pi/2):
                #u_val[i] = np.sign(u_val[i]) * math.radians(math.pi/2)

        i = i+1

    
    

    #current_pos = Matrix([[data.qpos[0]* math.cos(math.pi/4) + data.qpos[1]* math.cos(math.pi/4)], [-data.qpos[0]* math.cos(math.pi/4) + data.qpos[1]* math.cos(math.pi/4)], [data.qpos[0]], [0.0], [0.0], [0.0]])
    #x represents the state which is linear and angular velocities, xdot is accelerations 
    #x_val = Matrix([[data.qvel[0]* math.cos(math.pi/4) + data.qvel[1]* math.cos(math.pi/4)], [-data.qvel[0]* math.cos(math.pi/4) + data.qvel[1]* math.cos(math.pi/4)], [data.qpos[0]], [0.0],[0.0], [0.0]])
    last_ang_vel = [x_val[3], x_val[4], x_val[5]]

    x_val = Matrix([[data.sensordata[11]],[data.sensordata[12]],[data.sensordata[13]],[data.sensordata[14]],[data.sensordata[15]],[data.sensordata[16]] ])
    
    #des_vel = (desired_pos - current_pos)/T_v
    xdot_des = (des_vel - x_val)/T_a
    xdot_val = Matrix([[data.sensordata[8]-g*sp.sin(pitch_angle)],[data.sensordata[9]-g*sp.sin(roll_angle)],[data.sensordata[10]- g*sp.cos(roll_angle)*sp.cos(pitch_angle) ],[(x_val[3] - last_ang_vel[0])/dt],[(x_val[4] - last_ang_vel[1])/dt],[(x_val[5] - last_ang_vel[2])/dt] ])
    xdot_val = xdot_val.subs([(roll_angle,r), (pitch_angle,r)])
    r = x_val[3]*dt + r
    p = x_val[4]*dt + p
    #need to find gyroscope angular accelerations
    i = 0

    while i < len(x_val):
        if(abs(x_val[i]) <.0001):
            x_val[i] = 0.0
        i = i+1
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

    #print(tiltangle1)
    #print(tiltangle2)

    #print(tiltangle3)

    #print(tiltangle4)

    #data.ctrl[0] = u_val[0]
    #data.ctrl[1] = u_val[1]
    #data.ctrl[2] = u_val[2]
    #data.ctrl[3] = u_val[3]
    data.ctrl[0] = u_val[0]
    data.ctrl[1] = u_val[1]
    data.ctrl[2] = u_val[2]
    data.ctrl[3] = u_val[3]
    data.ctrl[4] = control1
    data.ctrl[5] = control2
    data.ctrl[6] = control3
    data.ctrl[7] = control4
    #print("x_val")

    #print(x_val)

    # print(curr_height, des_vel, act_vel, Gain, cntrl)
    #print("pos")
    #print(desired_pos, current_pos)
    # print("vel")
    # print(des_vel, x_val)
    # print("acc")
    # print(xdot_des, xdot_val)
    # print("next")
    
   

    
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