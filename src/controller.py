from qpsolvers import solve_qp
from scipy.integrate import solve_ivp
from casadi import *

from qpsolvers import solve_qp
from scipy.integrate import solve_ivp
from casadi import *

class LCBF:
    def __init__(self, m_nom, ca_nom, cd_nom, f0_nom, f1_nom, f2_nom, v_lead_nom, v_des, Th, clf_rate, cbf_rate, p_slack, delta = 15000):
        self.g = 9.81
        self.m = m_nom
        self.ca = ca_nom
        self.cd = cd_nom
        self.f0 = f0_nom
        self.f1 = f1_nom
        self.f2 = f2_nom
        self.v_lead = v_lead_nom 
        self.v_des = v_des
        self.Th = Th
        
        self.p_slack = p_slack
        self.cbf_rate = cbf_rate
        self.clf_rate = clf_rate
        
        self.delta = delta
        self.k1 = 0
        
    def clf(self, x):
        v = x[1]        
        V = (v - self.v_des)**2
        return V

    def dclf(self, x, u):
        v = x[1]
        Fr = self.f0 + self.f1 * v + self.f2 * v**2 
        dV = (v - self.v_des)*(2/self.m*(u*self.delta - Fr))
        return dV
        
    def cbf(self, x, isMaxCD=False):
        v = x[1]
        z = x[2]
        if isMaxCD:
            h = z - self.Th * v - 0.5  * (self.v_lead - v)**2 / (self.cd * self.g)
        else:
            h = z - self.Th * v 
        return h
        
    def dcbf(self, x, u, isMaxCD=False):
        v = x[1]
        z = x[2]
        
        Fr = self.f0 + self.f1 * v + self.f2 * v**2 
        if isMaxCD:
            dh = 1/self.m * (self.Th + (v - self.v_lead)/self.cd/self.g ) * (Fr - u) + (self.v_lead - v)
        else:
            dh = self.Th/self.m * (Fr - u*self.delta) + self.v_lead - v
        return dh


        
    def compute_controller(self, x, u_ref, estimator, t = None, normalizer = None):
        # Symbolic values
        u = SX.sym('u')
        slack = SX.sym('slack')

        # CBF-CLF calculation                
        V = self.clf(x) # Numeric
        dV = self.dclf(x, u) # Symbolic

        h = self.cbf(x) # Numeric
        dh = self.dcbf(x, u) # Symbolic

        # Learning feed 
        if normalizer:
            x_n = normalizer['x'].update(x)
            u_n = normalizer['u'].normalize(u)
            
            dhe_n = estimator.forward(x_n, u_n, t)
            if dhe_n is None:
                dhe = 0
            else:
                dhe = normalizer['dhe'].denormalize(dhe_n)
        else:
            dhe_n = estimator.forward(x, u, t)
            if dhe_n is None:
                dhe = 0
            else:
                dhe = dhe_n
    
        # Estimator
        dS = dh + dhe
        
        # QP optimizer
        weight_input = 2/self.m**2
        fqp = (u_ref - u)**2 * weight_input + self.p_slack *self.delta *slack**2
        gqp = vertcat( -dV - self.clf_rate*V + slack*self.delta, dS + self.cbf_rate * h)     
        qp = {'x': vertcat(u,slack), 'f':fqp, 'g':gqp}
        S = nlpsol('S', 'ipopt', qp,{'verbose':False,'print_time':False, "ipopt": {"print_level": 0}})
        r = S(lbg=0, lbx = [-self.m*self.cd*self.g/self.delta,-10000], ubx = [self.m*self.ca*self.g/self.delta,10000])
        
        # Solutions
        if normalizer:
            k = r['x'].elements()[0]
            _ = normalizer['u'].update(k)
        else:
            k = r['x'].elements()[0]
            
        self.k1 = k

        slack_sol = r['x'].elements()[1]

        # Create a Function to evaluate expression
        dh_f = Function('f',[u],[dh])
        dhe_f = Function('f',[u],[dhe])
        dV_f = Function('f',[u],[dV])
        dS_f = Function('f',[u],[dS])

        # Evaluate numerically
        dh = dh_f(k).elements()[0]
        dhe = dhe_f(k).elements()[0]
        dS = dS_f(k).elements()[0]
        dV = dV_f(k).elements()[0]
        
        

        return k, slack_sol, V, dV, h, dh, dhe, dS



class PID:
    def __init__(self, x_dim, u_dim, Kp, Kd, Ki, dT):
        self.x_dim = x_dim
        self.u_dim = u_dim 

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.dT = dT
        
        self.e = np.zeros([x_dim,1])
        self.de = np.zeros([x_dim,1])
        self.ei = np.zeros([x_dim,1])
        
        # e(t-1) 
        self.e_1 = np.zeros([x_dim,1])
        
    def update(self, e):
        self.e = e
        
        # Compute derivative
        self.de = (self.e - self.e_1)/self.dT
        
        # Compute integral
        self.ei =  self.ei + self.dT*(self.e - self.e_1)/2
        
        u = self.Kp.dot(self.e) + self.Kd.dot(self.de)  + self.Ki.dot(self.ei)
        
        self.e_1 = self.e
        
        return u
        

    def reset(self):
        self.e = np.zeros([self.x_dim,1])
        self.de = np.zeros([self.x_dim,1])
        self.ei = np.zeros([self.x_dim,1])
        
        # e(t-1) 
        self.e_1 = np.zeros([self.x_dim,1])
        
        