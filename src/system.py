
# For solvers
from qpsolvers import solve_qp
from scipy.integrate import solve_ivp
from casadi import *

class ACC:
    def __init__(self, m, c_d, f0, f1, f2, v_lead, delta = 15000):
        self.m = m
        self.c_d = c_d
        self.f0 = f0
        self.f1 = f1
        self.f2 = f2
        self.v_lead = v_lead
        self.u = 0
        self.delta = delta
        
        
    def update(self, x0, u, t, dt):
        
        x = MX.sym('x',3) # Three states
        
        p = x[0]
        v = x[1]
        z = x[2]
        
        
        Fr = self.f0  + self.f1 * v + self.f2 * v**2
        
        f = vertcat(
            v,
            -1/self.m*Fr,
            self.v_lead - v           
        )
        g = vertcat([
            0,
            self.delta/self.m *u,
            0
        ])
        

        dx = f + g
        
        ode = {}         # ODE declaration
        ode['x']   = x   # states
        ode['ode'] = dx # right-hand side 

        # Construct a Function that integrates over 4s
        F = integrator('F','cvodes',ode,{'t0':t,'tf':t+dt})

        res = F(x0=x0)
        
        x_n = res['xf'].elements()
        
        return x_n 
        
        