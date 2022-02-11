import numpy as np


def sample_initial_state(p_min = 0, p_max = 2,
                         v_min = 18, v_max = 22,
                         z_min = 38, z_max = 42):
    p0 = np.random.uniform(p_min, p_max)
    v0 = np.random.uniform(v_min, v_max)
    z0 = np.random.uniform(z_min, z_max)
    return [p0, v0, z0] 


class Derivator:
    def __init__(self, dt = 0.01):
        self.x = 0
        self.dt = dt
        
    def update(self, x_n):
        dx = (x_n - self.x)/self.dt
        
        self.x = x_n
       
        return dx
    
   
def step(t, th = 5, A = 5):    
    return 0 if t > th else A

def square(t, w = 3, A = 5):
    return -A if np.sin(w*t) < 0 else A
    
def sin(t, w = 3, A = 5):
    return A*np.sin(w*t)


def c2l(d):
    if type(d) == float or type(d) == int or type(d) == np.float64:
        return [d]
    if type(d) == list:
        return d
    if type(d) == np.ndarray:
        return d.tolist()
    