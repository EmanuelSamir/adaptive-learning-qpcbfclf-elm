import math
import numpy as np
import matplotlib.pyplot as plt
import math 

C = 0.174
K = 11
f0 = 0
f1 = 0.182
f2 = -0.0004
speeds_time = np.loadtxt('speed_comp.csv')
# steers_time = np.loadtxt('steer_const.csv')[4:]
# stiffness_time = np.loadtxt('stiff_coeff.csv')
# x = np.array(steers_time[:,0]-steers_time[0,0])
# y = C*(1-np.exp(-K*x))
preds = f0 + f1*speeds_time[5:-15,1] + f2*speeds_time[5:-15,1]**2
plt.plot(speeds_time[5:-15,1],-speeds_time[5:-15,2],label='Speed vs Acceleration (Actual)')
plt.plot(speeds_time[5:-15,1],preds,label='Speed vs Acceleration (Parameterized)')
# plt.plot(steers_time[:,0],(math.pi/180)*steers_time[:,1]*3/steers_time[:,2],label='Wheel angle from simple kinematic model')
# plt.plot(steers_time[:,0],y,label='Wheel angle from equation (K=11)')
plt.legend()
plt.xlabel('Speed')
plt.ylabel('Acceleration')
# plt.plot(np.clip((stiffness_time[:,2]*1.5-stiffness_time[:,1])/(stiffness_time[:,3]*stiffness_time[:,0]),-10,10))
# plt.plot(x+steers_time[0,0],y)
plt.show()