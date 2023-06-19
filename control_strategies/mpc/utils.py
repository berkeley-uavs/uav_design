from casadi import *


m = 2.0  # drone_mass
g = 9.81
arm_length = .2212
Ixx = 1.0
Iyy = 1.0
Izz = 1.0

#second order taylor series approx of sin
def sinTE(x):
    return x - ((x)**3)/6 #+ ((x)**5)/120
    #return sinTE(x)
def cosTE(x):
    return 1 -(x**2)/2 #+ ((x)**4)/24
    #return cosTE(x)

def rotBE(r,p,y):    
    rotBErow1 = horzcat(
                            (cosTE(y)*cosTE(p)), 
                            (sinTE(y)*cosTE(p)), 
                            (-sinTE(p)))
    rotBErow2 = horzcat(
                            (cosTE(y)*sinTE(p) *sinTE(r) - sinTE(y)*cosTE(r)), 
                            (sinTE(y)*sinTE(p) * sinTE(r) + cosTE(y)*cosTE(r)),
                            (cosTE(p)*sinTE(r)))
    rotBErow3 = horzcat(
                            (cosTE(y)*sinTE(p) * cosTE(r) + sinTE(y)*sinTE(r)), 
                            (sinTE(y)*sinTE(p) * cosTE(r) - cosTE(y)*sinTE(r)), 
                            (cosTE(p)*cosTE(r)))
    
    rotBEm = vertcat(rotBErow1, rotBErow2,rotBErow3)
    return rotBEm
    

def rotEB(r,p,y):
    rotEBm = transpose(rotBE(r,p,y))
    return rotEBm


def f_acc(T1, T2, T3, T4, tilt1, tilt2, tilt3, tilt4,droll, dpitch,dyaw, euler_roll, euler_pitch, euler_yaw ):
    f_acc = vertcat(

    (T2*sinTE(tilt2) - T4*sinTE(tilt4) - m*g*sinTE(euler_pitch))/m, # 1
    
    (T1*sinTE(tilt1) - T3*sinTE(tilt3) - m*g*sinTE(euler_roll))/m, # 2
    
    (T1*cosTE(tilt1) + T2*cosTE(tilt2) + T3*cosTE(tilt3) + T4*cosTE(tilt4) - m*g*cosTE(euler_roll)*cosTE(euler_pitch))/m, # 3
    
    ((T2*cosTE(tilt2)*arm_length) - (T4*cosTE(tilt4)*arm_length) + (Iyy*dpitch*dyaw + Izz*dpitch*dyaw))/Ixx, # 4    
    
    (T1*cosTE(tilt1)*arm_length - T3*cosTE(tilt3)*arm_length + (-Ixx*droll*dyaw + Izz*droll*dyaw))/Iyy, # 5
    
    (T1*sinTE(tilt1)*arm_length + T2*sinTE(tilt2)*arm_length + T3*sinTE(tilt3)*arm_length + T4*sinTE(tilt4)*arm_length + (Ixx*droll*dpitch - Iyy*droll*dpitch))/Izz) # 6)
    return f_acc

#T_dot and T are for finding euler angle angular accelerations 

def T_dot(euler_roll, euler_pitch, euler_yaw, droll_euler, dpitch_euler, dyaw_euler):
    T_dot = vertcat(
        horzcat(0,      
            (cosTE(euler_roll)*droll_euler*tan(euler_pitch) + dpitch_euler*sinTE(euler_roll)*1/cosTE(euler_pitch)**2),
            (-sinTE(euler_roll)*droll_euler*tan(euler_pitch) + dpitch_euler*cosTE(euler_roll)*1/cosTE(euler_pitch)**2)),

        horzcat(0,      
            (droll_euler*-sinTE(euler_roll)), 
            (droll_euler*-cosTE(euler_roll))),
            
        horzcat(0,      
            (cosTE(euler_roll)*droll_euler*1/cosTE(euler_pitch) + tan(euler_pitch)*dpitch_euler*sinTE(euler_roll)*1/cosTE(euler_pitch)),      
            (sinTE(euler_roll)*droll_euler*1/cosTE(euler_pitch) + tan(euler_pitch)*dpitch_euler*cosTE(euler_roll)*1/cosTE(euler_pitch))))
    return T_dot


def T(euler_roll, euler_pitch, euler_yaw):
    T = vertcat(
    horzcat(1, sinTE(euler_roll)*tan(euler_pitch), cosTE(euler_roll)*tan(euler_pitch)),
    horzcat(0,cosTE(euler_roll), - sinTE(euler_roll)),
    horzcat(0, sinTE(euler_roll)/cosTE(euler_pitch), cosTE(euler_roll)/cosTE(euler_pitch)))
    return T