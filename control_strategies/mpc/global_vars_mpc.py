import numpy as np

m = .2286
g = 9.81

class TVPData:
    def __init__(self, x, u, drone_accel, target_velocity):
        self.x = x
        self.u = u
        self.drone_accel = drone_accel
        self.target_velocity = target_velocity

x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).T
u0 = np.array([m*g/4,m*g/4,m*g/4,m*g/4,0.0,0.0,0.0,0.0]).T
#u0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

drone_acceleration = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
target_velocity = np.array([0.01, 0.0, 0.0])
tvp = TVPData(x0, u0, drone_acceleration, target_velocity)


class MPCcont:
        def __init__(self,controller):
            self.controller = controller
        
mpc_global_controller = MPCcont(None)
class MPCsim:
        def __init__(self,sim, est):
            self.sim = sim
            self.est = est
        

global_simulator = MPCsim(None,None)
