from collections import deque, namedtuple
import random

class ELMDataset:
    def __init__(self, dt, features = ('x'), time_th = 0.5, maxlen = 5):
        self.time_th = time_th
        self._maxlen = maxlen
        self.D_pre = deque()
        self.D_post = deque(maxlen = self._maxlen)
        self.dt = dt
        
        self.trans = namedtuple('trans',
                                    features)
        
    def reset(self):
        self.D_pre = deque()
        self.D_post = deque(maxlen = self._maxlen)
    
    def update(self, t, *args):
        if t/self.dt < self.time_th:
            self.D_pre.append(self.trans(*args))
            self.D_post.append(self.trans(*args))       
        else:
            self.D_post.append(self.trans(*args))
        
    def shuffle(self):
        random.shuffle(self.D_post)
        
    def get_D(self, t):
        if t/self.dt < self.time_th:
            return self.trans(*zip(*self.D_pre))
        else:
            return self.trans(*zip(*self.D_post))
        
        
class NNDataset:
    def __init__(self, features = ('x'), maxlen = 5):
        self.D = deque(maxlen = maxlen)
        self.trans = namedtuple('trans',
                                    features) # ('x', 'k', 'dh', 'dh_e')
        
    def reset(self):
        self.D = deque()
    
    def update(self, *args):
        self.D.append(self.trans(*args))
        
    def shuffle(self):
        random.shuffle(self.D)
        
    def get_D(self):
        sample = self.trans(*zip(*self.D))
        return sample
    