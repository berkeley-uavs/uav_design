import numpy as np
import do_mpc
from casadi import *
from global_vars_mpc import tvp
from utils import *


m = 2.0  # drone_mass
g = 9.81
arm_length = .2212
Ixx = 1.0
Iyy = 1.0
Izz = 1.0

# -------------------------------MODEL----------------------------------------


class MPC_Model:
    def __init__(self, is_linear) -> None:
        self.is_linear = is_linear
        self.model_type = "continuous"
        self.mpc_model = do_mpc.model.Model(self.model_type)

        # Continuous variables -xyz pos, dx dy dz, and euler roll pitch yaw are spatial, while droll, dpitch, dyaw are body rates
        self.pos, self.euler_ang, self.dpos, self.dtheta = self.init_states()
        self.u_th, self.u_ti = self.init_inputs()
        self.ddpos, self.ddtheta = self.init_algebraic_terms()
        self.last_state, self.last_input, self.last_acc, self.target_velocity = self.init_tvp()

        #states
        self.xpos_cont = self.pos[0]
        self.ypos_cont = self.pos[1]
        self.zpos_cont = self.pos[2]

        self.euler_roll_cont = self.euler_ang[0]
        self.euler_pitch_cont = self.euler_ang[1]
        self.euler_yaw_cont = self.euler_ang[2]

        self.dx_cont = self.dpos[0]
        self.dy_cont = self.dpos[1]
        self.dz_cont = self.dpos[2] 

        self.droll_cont = self.dtheta[0]
        self.dpitch_cont = self.dtheta[1]
        self.dyaw_cont = self.dtheta[2]

        #accelerations
        self.ddx_cont = self.ddpos[0]
        self.ddy_cont = self.ddpos[1]
        self.ddz_cont = self.ddpos[2]

        self.ddroll_cont = self.ddtheta[0]
        self.ddpitch_cont = self.ddtheta[1]
        self.ddyaw_cont = self.ddtheta[2]

        # Inputs
        self.T1_cont = self.u_th[0]
        self.T2_cont = self.u_th[1]
        self.T3_cont = self.u_th[2]
        self.T4_cont = self.u_th[3]
        self.tilt1_cont = self.u_ti[0]
        self.tilt2_cont = self.u_ti[1]
        self.tilt3_cont = self.u_ti[2]
        self.tilt4_cont = self.u_ti[3]

        self.euler_ang_vel_cont = vertcat(
                    (self.droll_cont + 
                    self.dyaw_cont*cosTE(self.euler_roll_cont)*tan(self.euler_pitch_cont) + 
                    self.dpitch_cont*sinTE(self.euler_roll_cont)*tan(self.euler_pitch_cont)),

                    (self.dpitch_cont*cosTE(self.euler_roll_cont) - 
                    self.dyaw_cont*sinTE(self.euler_roll_cont)),

                    ((self.dyaw_cont*cosTE(self.euler_roll_cont)/(cosTE(self.euler_pitch_cont))) + 
                    self.dpitch_cont*(sinTE(self.euler_roll_cont)/cosTE(self.euler_pitch_cont)))
                    )

        self.set_rhs()

        self.u_vec_cont = vertcat(
            self.u_th,
            self.u_ti
        )
        self.state_vec_cont = vertcat(
            self.pos,
            self.euler_ang,
            self.dpos,
            self.dtheta
        )
        self.result_vec_cont = vertcat(self.ddx_cont, self.ddy_cont, self.ddz_cont, self.ddroll_cont, self.ddpitch_cont, self.ddyaw_cont)

        if is_linear:
            self.euler_lagrange = self.linearise()
        else:
            self.euler_lagrange = self.result_vec_cont - self.find_continous_f_spatial_accel()

        self.setup_model()
        self.mpc_model.setup()

    # Called by default upon intialisation
    def init_states(self):
        # STATES
        #dtheta is in terms of BODY ANGULAR VELOCITIES, while euler_ang is in terms of SPATIAL EULER ANGLES
        pos = self.mpc_model.set_variable('_x',  'pos', (3, 1))
        euler_ang = self.mpc_model.set_variable('_x',  'euler_ang', (3, 1))
        dpos = self.mpc_model.set_variable('_x',  'dpos', (3, 1))
        dtheta = self.mpc_model.set_variable('_x',  'dtheta', (3, 1))

        return (pos, euler_ang, dpos, dtheta)

    # Called by default upon intialisation  
    def init_inputs(self):
        # INPUTS
        u_th = self.mpc_model.set_variable('_u',  'u_th', (4, 1))
        u_ti = self.mpc_model.set_variable('_u',  'u_ti', (4, 1))

        return (u_th, u_ti)
    
    # Called by default upon intialisation
    def init_algebraic_terms(self):
        # ALGEBRAIC TERMS
        ddpos = self.mpc_model.set_variable('_z',  'ddpos', (3, 1))
        ddtheta = self.mpc_model.set_variable('_z',  'ddtheta', (3, 1))

        return (ddpos, ddtheta)
    
    # Called by default upon intialisation
    def init_tvp(self):
        #TIME VARYING PARAMETERS
        last_state = self.mpc_model.set_variable(var_type='_tvp', var_name='last_state',shape=(12, 1))
        last_input = self.mpc_model.set_variable(var_type='_tvp', var_name='last_input',shape=(8, 1))
        last_acc = self.mpc_model.set_variable(var_type='_tvp', var_name='last_acc',shape=(6, 1))
        target_velocity = self.mpc_model.set_variable(var_type='_tvp', var_name='target_velocity', shape=(3, 1))

        return (last_state, last_input, last_acc, target_velocity)
    
    # Called by default upon intialisation
    def set_rhs(self):
        self.mpc_model.set_rhs('pos', self.dpos)
        self.mpc_model.set_rhs('dpos', self.ddpos)
        self.mpc_model.set_rhs('euler_ang', self.euler_ang_vel_cont)
        self.mpc_model.set_rhs('dtheta', self.ddtheta)

    # Called by default upon intialisation for Non Linear Models
    def find_continous_f_spatial_accel(self):

        f_bodyacc_cont = f_acc(self.T1_cont, self.T2_cont,self.T3_cont, self.T4_cont, 
                               self.tilt1_cont, self.tilt2_cont, self.tilt4_cont, 
                               self.droll_cont, self.dpitch_cont, self.dyaw_cont, 
                               self.euler_roll_cont, self.euler_pitch_cont, self.euler_yaw_cont)

        w_euler_cont = vertcat(self.euler_ang_vel_cont)
        droll_euler_cont = w_euler_cont[0] 
        dpitch_euler_cont = w_euler_cont[1]
        dyaw_euler_cont = w_euler_cont[2]

        v_b_cont = vertcat(self.dx_cont, self.dy_cont, self.dz_cont) #dx,dy,dz
        alpha_b_cont = vertcat(self.ddroll_cont, self.ddpitch_cont, self.ddyaw_cont)
        r_b_cont = vertcat(self.xpos_cont, self.ypos_cont, self.zpos_cont) #pos

        T_cont = T(self.euler_roll_cont, self.euler_pitch_cont, self.euler_yaw_cont)
        T_dot_cont = T_dot(droll_euler_cont, dpitch_euler_cont, dyaw_euler_cont, 
                           self.euler_roll_cont, self.euler_pitch_cont, self.euler_yaw_cont)

        alpha_euler_cont = T_cont@alpha_b_cont + T_dot_cont@vertcat(self.droll_cont, self.dpitch_cont, self.dyaw_cont)
        rotEBMatrix_cont = rotEB(self.euler_roll_cont, self.euler_pitch_cont, self.euler_yaw_cont)

        fspatial_linear_acc_cont = vertcat((rotEBMatrix_cont@(f_bodyacc_cont[0:3])) + 2 * skew(w_euler_cont)@v_b_cont + skew(alpha_euler_cont)@r_b_cont + skew(w_euler_cont)@(skew(w_euler_cont)@r_b_cont))
        fspatial_rotation_acc_cont = vertcat(f_bodyacc_cont[3:6]) 
        fspatial_acc_cont = vertcat(fspatial_linear_acc_cont, fspatial_rotation_acc_cont)

        return fspatial_acc_cont
    
    # Called in linearise() upon intialisation of Linear Models
    def find_tvp_f_spatial_accel(self):
        # TVP   - xyz pos, dx dy dz, and euler roll pitch yaw are spatial, while droll, dpitch, dyaw are body rates
        #states
        xpos_tvp = self.last_state[0]
        ypos_tvp = self.last_state[1]
        zpos_tvp = self.last_state[2]

        euler_roll_tvp = self.last_state[3]
        euler_pitch_tvp = self.last_state[4]
        euler_yaw_tvp = self.last_state[5]

        dx_tvp = self.last_state[6]
        dy_tvp = self.last_state[7]
        dz_tvp = self.last_state[8] 

        droll_tvp = self.last_state[9]
        dpitch_tvp = self.last_state[10]
        dyaw_tvp = self.last_state[11]


        #accelerations
        ddx_tvp = self.last_acc[0]
        ddy_tvp = self.last_acc[1]
        ddz_tvp = self.last_acc[2]

        ddroll_tvp = self.last_acc[3]
        ddpitch_tvp = self.last_acc[4]
        ddyaw_tvp = self.last_acc[5]


        # Inputs
        T1_tvp = self.last_input[0]
        T2_tvp = self.last_input[1]
        T3_tvp = self.last_input[2]
        T4_tvp = self.last_input[3]
        tilt1_tvp = self.last_input[4]
        tilt2_tvp = self.last_input[5]
        tilt3_tvp = self.last_input[6]
        tilt4_tvp = self.last_input[7]

        #would have to change to add roll and pitch for g term (still from last state input ig)
        f_bodyacc_tvp = f_acc(T1_tvp, T2_tvp,T3_tvp,T4_tvp, tilt1_tvp,tilt2_tvp,tilt3_tvp,tilt4_tvp,droll_tvp,dpitch_tvp,dyaw_tvp, euler_roll_tvp,euler_pitch_tvp,euler_yaw_tvp )


        euler_ang_vel_tvp = vertcat(
                                        (droll_tvp + dyaw_tvp*cosTE(euler_roll_tvp)*tan(euler_pitch_tvp) + dpitch_tvp*sinTE(euler_roll_tvp)*tan(euler_pitch_tvp)),

                                        (dpitch_tvp*cosTE(euler_roll_tvp) - dyaw_tvp*sinTE(euler_roll_tvp)),

                                        ((dyaw_tvp*cosTE(euler_roll_tvp)/(cosTE(euler_pitch_tvp))) + dpitch_tvp*(sinTE(euler_roll_tvp)/cosTE(euler_pitch_tvp)))
        )

        w_euler_tvp = vertcat(euler_ang_vel_tvp)
        droll_euler_tvp = w_euler_tvp[0] 
        dpitch_euler_tvp = w_euler_tvp[1]
        dyaw_euler_tvp = w_euler_tvp[2]

        v_b_tvp = vertcat(dx_tvp,dy_tvp,dz_tvp)#dx,dy,dz
        alpha_b_tvp = vertcat(ddroll_tvp, ddpitch_tvp, ddyaw_tvp)
        r_b_tvp = vertcat(xpos_tvp, ypos_tvp, zpos_tvp)#pos

        T_tvp = T(euler_roll_tvp, euler_pitch_tvp, euler_yaw_tvp)
        T_dot_tvp = T_dot(droll_euler_tvp, dpitch_euler_tvp, dyaw_euler_tvp,euler_roll_tvp, euler_pitch_tvp, euler_yaw_tvp)

        alpha_euler_tvp = T_tvp@alpha_b_tvp + T_dot_tvp@vertcat(droll_tvp,dpitch_tvp,dyaw_tvp)
        rotEBMatrix_tvp = rotEB(euler_roll_tvp, euler_pitch_tvp, euler_yaw_tvp)

        fspatial_linear_acc_tvp = vertcat((rotEBMatrix_tvp@(f_bodyacc_tvp[0:3])) + 2 * skew(w_euler_tvp)@v_b_tvp + skew(alpha_euler_tvp)@r_b_tvp + skew(w_euler_tvp)@(skew(w_euler_tvp)@r_b_tvp))
        #spatial_linear_acc_tvp = vertcat((rotEBMatrix_tvp@(f_bodyacc_tvp[0:3])))
        fspatial_rotation_acc_tvp = vertcat(f_bodyacc_tvp[3:6]) 
        fspatial_acc_tvp = vertcat(fspatial_linear_acc_tvp, fspatial_rotation_acc_tvp)

        return fspatial_acc_tvp

    # Called by default upon intialisation for Linear Models
    def linearise(self):
        fspatial_acc_tvp = self.find_tvp_f_spatial_accel()
        A = jacobian(self.last_acc - fspatial_acc_tvp, self.last_state)
        print((A.shape))
        B = jacobian(self.last_acc - fspatial_acc_tvp, self.last_input)
        print((B.shape))
        C = jacobian(self.last_acc - fspatial_acc_tvp, self.last_acc)
        print(C.shape)

        #euler_lagrange =  (result_vec_cont -fspatial_acc_cont)
        euler_lagrange = C@(self.result_vec_cont-self.last_acc) + (
            A@(self.state_vec_cont-self.last_state)) + (
            B@(self.u_vec_cont-self.last_input)) + (
            self.last_acc - fspatial_acc_tvp)

        return euler_lagrange
    
    # Called by default upon intialisation
    def setup_model(self):
        print("Enters setup_model")
        self.mpc_model.set_alg('euler_lagrange', self.euler_lagrange)
        # self.mpc_model.setup()




# -------------------------------CONTROLLER----------------------------------------





class MPC_Controller:
    def __init__(self, model):
        self.model = model
        self.controller = do_mpc.controller.MPC(self.model)
        self.n_horizon = 4

        self.x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
        self.u0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
        self.z0 = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]).T

        self.set_parameters()
        self.set_obj_func()
        self.setup_tvp_controller()
        self.setup_controller()

    # Called by default upon intialisation
    def set_parameters(self):
        setup_mpc = {
            'n_horizon': self.n_horizon,
            'n_robust': 0,
            'open_loop': 0,
            't_step': 0.04,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps', 'ipopt.print_level':0}
        }

        self.controller.set_param(**setup_mpc)

    # Called by default upon intialisation
    def set_obj_func(self):
        mterm = ((self.model.dx_cont - self.model.target_velocity[0])**2 + (self.model.dy_cont - self.model.target_velocity[1])**2 + (self.model.dz_cont - self.model.target_velocity[2])**2) # terminal cost

        lterm = ((self.model.dx_cont - self.model.target_velocity[0])**2 + (self.model.dy_cont - self.model.target_velocity[1])**2 + (self.model.dz_cont - self.model.target_velocity[2])**2) # stage cost

        self.controller.set_objective(mterm=mterm, lterm=lterm)
        # Input force is implicitly restricted through the objective.
        self.controller.set_rterm(u_th=0.1)
        self.controller.set_rterm(u_ti=0.01)

    # Called in main script
    def set_bounds(self, tilt_limit, thrust_limit):
        u_th_upper_limits = np.array([thrust_limit, thrust_limit, thrust_limit, thrust_limit])
        u_th_lower_limits =  np.array([0.00, 0.00, 0.00, 0.00])
        u_ti_upper_limits = np.array([tilt_limit, tilt_limit, tilt_limit, tilt_limit])
        u_ti_lower_limits =  np.array([-tilt_limit, -tilt_limit, -tilt_limit, -tilt_limit])

        # return (u_th_upper_limits, u_th_lower_limits, u_ti_upper_limits, u_ti_lower_limits)
        self.controller.bounds['lower','_u','u_th'] = u_th_upper_limits
        self.controller.bounds['upper','_u','u_th'] = u_th_lower_limits
        self.controller.bounds['lower','_u','u_ti'] = u_ti_lower_limits
        self.controller.bounds['upper','_u','u_ti'] = u_ti_upper_limits

    # Called in main script
    # X0 is a horizontal array
    def set_current_state(self, x0):
        self.x0 = x0.T
        self.controller.x0 = self.x0

    # Called in main script
    # U0 is a horizontal array
    def set_current_input(self, u0):
        self.u0 = u0.T
        self.controller.u0 = self.u0

    # Called in main script
    # Z0 is a horizontal array   
    def set_current_algebraic(self, z0):
        self.z0 = z0.T
        self.controller.z0 = self.z0
    
    def setup_tvp_controller(self):
        def controller_tvp_fun(self, t_now):
            controller_tvp_template = self.controller.get_tvp_template()
            for k in range(self.n_horizon + 1):
                controller_tvp_template['_tvp',k,'last_state'] = tvp.x
                controller_tvp_template['_tvp',k,'last_input'] = tvp.u
                controller_tvp_template['_tvp',k,'last_acc'] = tvp.drone_accel
                controller_tvp_template['_tvp',k,'target_velocity'] = tvp.target_velocity
            return controller_tvp_template
        self.controller.set_tvp_fun(controller_tvp_fun)

    # Called by default upon intialisation
    def setup_controller(self):
        self.controller.setup()

    # Call from main script
    def initial_guess(self):
        self.controller.set_initial_guess()



# -------------------------------SIMULATOR----------------------------------------




class MPC_Simulator:
    def __init__(self, model):
        self.model = model
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.set_parameters()
        self.x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
        self.u0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
        self.z0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).T

        self.setup_simulator()

    # Called by default upon intialisation
    def set_parameters(self):
        params_simulator = {
            # Note: cvode doesn't support DAE systems.
            'integration_tool': 'idas',
            'abstol': 1e-8,
            'reltol': 1e-8,
            't_step': 0.04
        }

        self.simulator.set_param(**params_simulator)

    # Called by default upon intialisation
    def setup_simulator(self):
        self.simulator.setup()

    # Called in main script
    # X0 is a horizontal array
    def set_current_state(self, x0):
        self.x0 = x0.T
        self.simulator.x0 = self.x0

    # Called in main script
    # U0 is a horizontal array
    def set_current_input(self, u0):
        self.u0 = u0.T
        self.simulator.u0 = self.u0

    # Called in main script
    # Z0 is a horizontal array   
    def set_current_algebraic(self, z0):
        self.z0 = z0.T
        self.simulator.z0 = self.z0

    # Call from main script
    def initial_guess(self):
        self.simulator.set_initial_guess()




# -------------------------------ESTIMATOR----------------------------------------





class MPC_Estimator:
    def __init__(self, model):
        self.model = model
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
        self.u0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
        self.z0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).T

    # Called in main script
    # X0 is a horizontal array
    def set_current_state(self, x0):
        self.x0 = x0.T
        self.estimator.x0 = self.x0

    # Called in main script
    # U0 is a horizontal array
    def set_current_input(self, u0):
        self.u0 = u0.T
        self.estimator.u0 = self.u0

    # Called in main script
    # Z0 is a horizontal array   
    def set_current_algebraic(self, z0):
        self.z0 = z0.T
        self.estimator.z0 = self.z0
    

class TVPData:
    def __init__(self, x, u, drone_accel, target_velocity):
        self.x = x
        self.u = u
        self.drone_accel = drone_accel
        self.target_velocity = target_velocity

