import numpy as np


class WelfordNormalizerOne:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, input_size=()):
        self.mean = np.zeros(input_size, 'float32')
        self.var = np.ones(input_size, 'float32')
        self.M2 = np.ones(input_size, 'float32')
        self.count = 1

    def update(self, x):
        #x = np.array(x)
        self.count += 1
        
        delta = x - self.mean
        
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        res = self.normalize(x)
        return res
    
    def normalize(self, x):
        res = (x - self.mean) /  (self.M2 / (self.count ) )**(0.5)
        return res
    
    def denormalize(self, xn):
        res = xn * (self.M2 / (self.count ) )**(0.5) + self.mean
        return res
