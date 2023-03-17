import sympy as sp
from sympy.diffgeom import *

M = Manifold("M", 5)
P = Patch("P", M)
coord = CoordSystem("coord", P, ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz", "T1", "T2", "T3", "T4", "theta1", "theta2", "theta3", "theta4", "w_1", "w_2", "w_3", "roll_angle","pitch_angle","dx","dy","dz"])
(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, T1, T2, T3, T4, theta1, theta2, theta3, theta4, w_1, w_2, w_3, roll_angle,pitch_angle,dx,dy,dz) = coord.coord_functions()
(e_Ixx, e_Ixy, e_Ixz, e_Iyy, e_Iyz, e_Izz, e_T1, e_T2, e_T3, e_T4, e_theta1, e_theta2, e_theta3, e_theta4, e_w_1, e_w_2, e_w_3, e_roll_angle,e_pitch_angle,e_dx,e_dy,e_dz) = coord.base_vectors()
