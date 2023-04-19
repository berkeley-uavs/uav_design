import numpy as np
import do_mpc
from casadi import *
import math


class UAV:

    def __init__(self, **drone_params):
        # init uav mpc controller here
        self.mass = drone_params["mass"]
        self.Ixx = drone_params["Ixx"]
        self.Iyy = drone_params["Iyy"]
        self.Izz = drone_params["Izz"]

        pass

    def __setup_mpc(self):
        model_type = "continuous"
        self.mpc_model = do_mpc.model.Model(model_type)
        self.mpc_controller = None

        pos = self.mpc_model.set_variable('states',  'pos', (3, 1))
        theta = self.mpc_model.set_variable('states',  'theta', (3, 1))

        dpos = self.mpc_model.set_variable('states',  'dpos', (3, 1))
        dtheta = self.mpc_model.set_variable('states',  'dtheta', (3, 1))

        u_th = self.mpc_model.set_variable('inputs',  'u_th', (4, 1))
        u_ti = self.mpc_model.set_variable('inputs',  'u_ti', (4, 1))

        ddpos = self.mpc_model.set_variable('algebraic',  'ddpos', (3, 1))
        ddtheta = self.mpc_model.set_variable('algebraic',  'ddtheta', (3, 1))
        target_point = self.mpc_model.set_variable(
            var_type='_tvp', var_name='target_point', shape=(3, 1))

        self.mpc_model.set_rhs('pos', dpos)
        self.mpc_model.set_rhs('theta', dtheta)
        self.mpc_model.set_rhs('dpos', ddpos)
        self.mpc_model.set_rhs('dtheta', ddtheta)

        T1 = u_th[0]
        T2 = u_th[1]
        T3 = u_th[2]
        T4 = u_th[3]
        theta1 = u_ti[0]
        theta2 = u_ti[1]
        theta3 = u_ti[2]
        theta4 = u_ti[3]

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
            self.mass*ddx - T2*sin(theta2) + T4*sin(theta4) + g*sin(pitch),
            # 2
            self.mass*ddy - T1*sin(theta1) + T3*sin(theta3) + g*sin(roll),
            # 3
            self.mass*ddz - T1*cos(theta1) - T2*cos(theta2) - T3*cos(theta3) - \
            T4*cos(theta4) - g*cos(roll)*cos(pitch),
            # 4
            Ixx*ddroll - (T2*cos(theta2)*arm_length) + (T4*cos(theta4)
                                                        * arm_length) - (Iyy*dpitch*dy - Izz*dpitch*dy),
            # 5
            Iyy*ddpitch - T1*cos(theta1)*arm_length + T3*cos(theta3) * \
            arm_length - (-Ixx*droll*dy + Izz*droll*dy),
            # 6
            Izz*ddyaw - T1*sin(theta1)*arm_length - T2*sin(theta2)*arm_length - T3*sin(
                theta3)*arm_length - T4*sin(theta4)*arm_length - (Ixx*droll*dpitch - Iyy*droll*dpitch)

        )

        mpc_model.set_alg('euler_lagrange', euler_lagrange)
        mpc_model.set_expression(expr_name='cost', expr=sum1(.9*sqrt((pos[0]-target_point[0])**2 + (pos[1]-target_point[1])**2 + (
            pos[2]-target_point[2])**2) + .00002*sqrt((u_th[0])**2 + (u_th[1])**2 + (u_th[2])**2 + (u_th[3])**2)))
        mpc_model.set_expression(
            expr_name='mterm', expr=sum1(.9*sqrt((pos[0]-.3)**2 + (pos[1]-.3)**2 + (pos[2]-1)**2)))

        mpc_model.setup()

        mpc_controller = do_mpc.controller.MPC(mpc_model)

        setup_mpc = {
            'n_horizon': 5,
            'n_robust': 1,
            'open_loop': 0,
            't_step': 0.001,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps', 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        }

        mpc_controller.set_param(**setup_mpc)

    def get_next_control(self):
        pass
