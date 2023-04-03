import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import sympy as sp
from sympy import *
import math
import os
from scipy.optimize import minimize

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

m = 1 # drone_mass
g = 9.81


def lie_derivative(subscript, s, fn):
    # partial of fn with respect to s * subscript
    return fn.jacobian(s) * subscript

def get_beta_matrix(g, f, h, s):
    beta = zeros(h.shape[0], g.shape[1])
    for p in range(h.shape[0]):
        for m in range(g.shape[1]):
            beta[p, m] = lie_derivative(g.col(m), s, lie_derivative(f, s, h.row(p)))
    return beta




def init_controller(model,data):
    global h, v, beta, alpha, q,q_dot, theta1, theta2, theta3, theta4, Ixx, Iyy, Izz, x, y, z, roll, pitch, yaw, droll, dpitch, dyaw,f,g,s
    #initialize the controller here. This function is called once, in the beginning
    (Ixx, Iyy, Izz, theta1, theta2, theta3, theta4, theta1_dot, theta2_dot, theta3_dot, theta4_dot, T1, T2, T3, T4, x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw) = sp.symbols('Ixx, Iyy, Izz, theta1, theta2, theta3, theta4, theta1_dot, theta2_dot, theta3_dot, theta4_dot, T1, T2, T3, T4, x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw')
    q = Matrix([[x, y, z, roll, pitch, yaw]]).T
    q_dot = Matrix([[dx, dy, dz, droll, dpitch, dyaw]]).T
    s = Matrix([[q], [q_dot], [theta1], [theta2], [theta3], [theta4]])
    u = Matrix([[T1, T2, T3, T4, theta1_dot, theta2_dot, theta3_dot, theta4_dot]]).T
    g_mat = Matrix([[sp.zeros(6,8)], 
                [0, sin(theta2)/m, 0, -sin(theta4)/m, 0, 0, 0, 0],
                [sin(theta1)/m, 0, -sin(theta3)/m, 0, 0, 0, 0, 0],
                [cos(theta1)/m, cos(theta2)/m, cos(theta3)/m, cos(theta4)/m, 0, 0, 0, 0],
                [0, cos(theta2)/Ixx, 0, -cos(theta4)/Ixx, 0, 0, 0, 0],
                [cos(theta1)/Iyy, 0, -cos(theta3)/Iyy, 0, 0, 0, 0, 0],
                [sin(theta1)/Izz, sin(theta2)/Izz, sin(theta3)/Izz, sin(theta4)/Izz, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])


    temp = Matrix([[-g*sin(pitch), -g*sin(roll), -g*cos(roll)*cos(pitch), dpitch*dyaw*((Iyy-Izz)/Ixx), droll*dyaw*((Izz-Ixx)/Iyy), droll*dpitch*((Ixx-Iyy)/Izz), 0, 0, 0, 0]]).T
    f = q_dot[:, :]
    f = f.col_join(temp)
    init_q_desired = Matrix([[0, 0, 0, 0, 0, 0]]).T
    h = init_q_desired - q
    Kp = 0.5
    Kd = 0.05
    v = -Kp * h
    beta = get_beta_matrix(g_mat, f, h, s)
    alpha = lie_derivative(f, s, lie_derivative(f, s, h)) # lf_lf_h

    

    # print(alpha)
    # print(u)

def RotToRPY(R):
    R=R.reshape(3,3) #to remove the last dimension i.e., 3,3,1
    phi = math.asin(R[1,2])
    psi = math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))
    theta = math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
    return phi,theta,psi





def controller(model, data):
    global h, v, beta, alpha, q,q_dot, theta1, theta2, theta3, theta4, Ixx, Iyy, Izz, dpitch, droll, dyaw, roll, pitch, yaw, x, y, z,f,g,s
    

    
    tiltangle1 = data.sensordata[0]
    tiltvel1 = data.sensordata[1]
    tiltangle2 = data.sensordata[2]
    tiltvel2 = data.sensordata[3]
    tiltangle3 = data.sensordata[4]
    tiltvel3 = data.sensordata[5]
    tiltangle4 = data.sensordata[6]
    tiltvel4 = data.sensordata[7]
    pitch_vel = data.sensordata[8]
    roll_vel = data.sensordata[9]
    yaw_vel = data.sensordata[10]

    spatial_coords = data.qpos[0:3]
    R = data.site_xmat[0].reshape(3,3)
    body_coords = R@spatial_coords
    orientation = R@np.array(RotToRPY(R)).reshape(3,1)
    
    b = np.hstack((np.zeros((3, 3)), R.T))
    a = np.hstack((R.T, np.zeros((3, 3))))
    rot = np.vstack((a, b))
    q_desired = rot@np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q_desired = Matrix([q_desired]).T
    
    h = q_desired - q
    Kp = .005
    v = -(Kp * h)
    costs = Matrix([.1, .1, .1, .1, .5, .5, .5, .5])

    beta_val = beta.subs({theta1:tiltangle1, theta2:tiltangle2, theta3:tiltangle3, theta4:tiltangle4, Ixx:1, Iyy:1, Izz:1})
    v_val = v.subs({x:body_coords[0], y:body_coords[1], z:body_coords[2], roll:float(orientation[0]), pitch:float(orientation[1]), yaw:float(orientation[2])})
    #print(v_val)
    alpha_val = alpha.subs({roll:0, pitch:0, yaw:0, Ixx:1, Iyy:1, Izz:1, dpitch:pitch_vel, droll:roll_vel, dyaw:yaw_vel})
    
    #u_val = beta_val.pinv()*(v_val-alpha_val) 
    # #replace with optimization -> minimize some cost function W*u, subject to alpha + Beta*u = v
    
    def f(u):
        return costs[0]*u[0]+ costs[1]*u[1]+ costs[2]*u[2]+costs[3]*u[3]+costs[4]*u[4]+costs[5]*u[5]+ costs[6]*u[6]+costs[7]*u[7]

    def eq_constraint(u):
        return np.array(alpha_val).astype(np.float64) + np.matmul((np.array(beta_val).astype(np.float64)),(np.array([[u[0]],[u[1]],[u[2]],[u[3]],[u[4]],[u[5]],[u[6]],[u[7]]]))) - (np.array(v_val).astype(np.float64))
    def ineq_constraint(u):
        return np.array(alpha_val).astype(np.float64) + np.matmul((np.array(beta_val).astype(np.float64)),(np.array([[u[0]],[u[1]],[u[2]],[u[3]],[u[4]],[u[5]],[u[6]],[u[7]]]))) - (np.array(v_val).astype(np.float64))

    con1 = {'type': 'eq', 'fun': eq_constraint}
    con2 = {'type': 'ineq', 'fun': ineq_constraint}
    cons = [con1, con2]

    u0 = [0,0,0,0,0,0,0,0]
    b1 = (-1000, 1000)
    #b2 = (-math.pi/2,math.pi/2) doesnt work cuz our inputs are ang velocites now
    bnds = (b1,b1,b1,b1, b1, b1, b1, b1)
    print(eq_constraint(u0))
    u_val =  minimize(f, u0, method='SLSQP',bounds = bnds, constraints=cons)

    


    #ddoutput = (((h.jacobian(q))*q_dot).jacobian(s))*(f + g*u)
    #print(ddoutput)
    # control1 = 
    # control1 = -Kp*(tiltangle1-u[4]) - Kd*tiltvel1 # position control
    # control2 = -Kp*(tiltangle2-u[5]) - Kd*tiltvel2 # position control
    # control3 = -Kp*(tiltangle3-u[6]) - Kd*tiltvel3 # position control
    # control4 = -Kp*(tiltangle4-u[7]) - Kd*tiltvel4 # position control

    data.ctrl[0] = u_val[0]
    data.ctrl[1] = u_val[1]
    data.ctrl[2] = u_val[2]
    data.ctrl[3] = u_val[3]
    data.ctrl[4] = u_val[4]
    data.ctrl[5] = u_val[5]
    data.ctrl[6] = u_val[6]
    data.ctrl[7] = u_val[7]

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