#!/usr/bin/python3

from __future__ import print_function
from __future__ import division

'''
This code implements elm controller
'''


# Basic
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import pinv2
from tqdm import tqdm
import pandas as pd
import itertools
from argparse import ArgumentParser
import logging
import matplotlib.pyplot as plt
import math
from sys import path as sys_path
from os import path as os_path, times
    
# For dataset
from collections import deque, namedtuple
import random

# For solvers
from qpsolvers import solve_qp
from scipy.integrate import solve_ivp
from casadi import *
import casadi

# For estimators
from torch import nn
import torch
import torch.nn.functional as F

# Project packages
from system import ACC
from controller import LCBF, PID
from dataset import ELMDataset, NNDataset
from estimator import *
from normalizer import *
from functions import *

# For carla
from carla_utils import *

# Parameters
dt = 0.05
simTime = 40

# Real parameters
v_lead = 15
v_des = 18
# m  = 1650.0
print("mass : ", m)
g = 9.81

f0 = 0*m
f1 = 0.182*m
f2 = -0.0004*m

c_a = 0.8
c_d = 0.8
Th = 1.8

# Nominal parameters
f0_nom = 2*f0
f1_nom = 2*f1
f2_nom = 2*f2

m_nom = 0.75* m

# QP-CLF-CBF parameters
p_slack = 1e-2
clf_rate = 5
cbf_rate = 5

torch.manual_seed(42)

def game_loop(args):    

    ########################################
    #    System
    ########################################
    acc = ACC(m, c_d, f0, f1, f2, v_lead)
    derivator = Derivator(dt)

    ########################################
    #    Controller
    ########################################
    cont = LCBF(m_nom, c_a, c_d, f0_nom, f1_nom, f2_nom, v_lead, v_des, Th, clf_rate, cbf_rate, p_slack)

    ########################################
    #    Estimator parameters
    ########################################
    input_size = 3 
    hidden_size = 100
    output_size = 1

    learned_ratio = 1.1
    time_th = learned_ratio* hidden_size

    ########################################
    #    PID control reference
    ########################################
    x_dim = 3
    u_dim = 1

    kp = np.array([[0, 0.2, 0]])
    kd = np.array([[0, 0, 0]])
    ki = np.array([[0, 0.2, 0]])

    pid = PID(x_dim, u_dim, kp, kd, ki, dt)

    ########################################
    #    Training parameters or initial states
    ########################################
    lr_pres =  [1e-3]   #[1e-2, 1e-3]
    lr_posts =  [1e-2]  #[1e-2]
    z0s = [34] #[28,30,32,34,38] #[36]#[30,32,34,38]  #[30, 34, 38]
    v0s = [12]#,17,19,20] 
    funcs = [step, sin]#, square]# Square or sin


    # Path for saving data
    #data_dir = '../data/elm'
    data_dir = '../data/dummy_carla'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Total of cases
    cases = len(list(itertools.product(lr_pres, lr_posts, z0s, v0s, funcs)))
    pbar = tqdm(total=cases*simTime/dt)

    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        # world_load = client.load_world(args.map)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        print("here")
        controller = KeyboardControl(world, args.autopilot)
        settings = client.get_world().get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = dt
        client.get_world().apply_settings(settings)
        clock = pygame.time.Clock()
        
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                break
            world.tick(clock)
            world.render(display)
            client.get_world().tick()
            pygame.display.flip()
        
        print("Starting automated waypoint follower")
        vel_accs = []
        for lr_pre, lr_post, z0, v0, func in itertools.product(lr_pres, lr_posts, z0s, v0s, funcs):
            # Add waypoint markers
            world.restart(z0)
            t = world.player.get_transform()
            t_opp = world.opponent.get_transform()
            angle_heading = t.rotation.yaw * pi/ 180
            world.player.set_target_velocity(carla.Vector3D(float(v0*math.cos(angle_heading)),float(v0*math.sin(angle_heading)),0))
            world.player.apply_control(carla.VehicleControl(throttle=0, brake=0, steer=0, manual_gear_shift=True, gear=4))
            
            if SCENE == 'one_vehicle' :
                world.opponent.set_target_velocity(carla.Vector3D(float(v_lead*math.cos(angle_heading)),float(v_lead*math.sin(angle_heading)),0))
                world.opponent.apply_control(carla.VehicleControl(throttle=0, brake=0, steer=0, manual_gear_shift=True, gear=4))
            

            start_x = t.location.x
            start_y = t.location.y
            start_x_opp = t_opp.location.x
            start_y_opp = t_opp.location.y
            curr_speed = 0
            ####################################################
            ##############  Save data
            ####################################################
            fn = "lr_pre_{}_lr_post_{}_z0_{}_v0_{}_func_{}.csv".format(lr_pre, lr_post, z0, v0, func.__name__)
            column_names = ['p', 'v', 'z', 'u','u_ref','V','h','dhe_real','dhe','slack','v_lead']

            df = pd.DataFrame(columns=column_names,dtype=object)
            path = os.path.join(data_dir, fn)
            df.to_csv(path, index=False)        

            ####################################################
            ##############  Initialization
            ####################################################

            # Initial position
            x = [0, v0, z0]
            
            # Estimator
            estimator = EstimatorDummy()# EstimatorELM(input_size, hidden_size, output_size, time_th, dt, lr_pre, lr_post)
            
            ## Dataset
            dataset = ELMDataset(dt, ('x', 'k', 'dhe_real'), time_th)

            for t in np.arange(0, simTime, dt): 
                tr = world.player.get_transform()
                v = world.player.get_velocity()
                if SCENE == 'one_vehicle' or SCENE == 'one_vehicle_turn' :
                    t_opp = world.opponent.get_transform()
                    x_obst = t_opp.location.x
                
                px = tr.location.x
                angle_heading = tr.rotation.yaw * pi/ 180
                vx = v.x
                x[0] = (px-start_x) + vx*dt
                x[1] = vx + world.imu_sensor.accelerometer[0]*dt
                x[2] = x_obst-px + (acc.v_lead - vx)*dt
                # Get reference control input: u_ref
                e = np.array([[0], [v_des], [0]]) - np.expand_dims(x, axis = 1)
                u_ref = pid.update(e)
                u_ref = u_ref[0,0]
            
                # Simulate dynamic uncertainty
                unct = func(t)
                acc.v_lead = v_lead + unct  # lead_vehicle

                # Controller
                k, slack_sol, V, dV, h, dh, dhe, dS = cont.compute_controller(x, u_ref+(f0+f1*v_des+f2*v_des**2)/15000, estimator, t) 
                print(k)
                # System update
                # x_n = acc.update(x, k, t, dt)
                world.player.add_impulse(carla.Vector3D(float(15000*k*dt),0,0))
                # Obtaining label: dhe_real
                dh_real = derivator.update(h)
                dhe_real = dh_real - dh

                # Update dataset
                dataset.update(float(t), np.array(x).astype(float), float(k), float(dhe_real))

                # Update estimator: Training
                estimator.train(t, dataset)

                # Update data saved
                row = c2l(x) + c2l(k) + c2l(u_ref) + c2l(V) + c2l(h) + c2l(dhe_real) + c2l(dhe) + c2l(slack_sol) + c2l(acc.v_lead)
                df_row = pd.DataFrame(dict(zip(column_names, row)), index = [0])
                df.append(df_row, sort = False).to_csv(path, index=False, mode = 'a', header=False)
                # print("func : ", x[2],x[1],x_obst-px-Th*vx,x[2]-Th*x[1])
                # Update new state
                pbar.update(1)
                curr_loc_opp = t_opp.location
                curr_loc_opp.x = x_obst + acc.v_lead*dt
                world.opponent.set_location(curr_loc_opp)
                world.player.apply_control(carla.VehicleControl(throttle=0, brake=0, steer=-0.01*tr.rotation.yaw, manual_gear_shift=True, gear=4))
                # print(x,k,dt,float(15000*k*dt))
                # print("Yaw :", tr.rotation.yaw)
                v = world.player.get_velocity()
                vel_accs.append([hud.simulation_time,math.sqrt(v.x**2 + v.y**2),world.imu_sensor.accelerometer[0],world.imu_sensor.accelerometer[1]])
            
                # print("z :", u_ref+(f0+f1*v_des+f2*v_des**2)/15000, vx, x[1], k)
                world.tick(clock)
                client.get_world().tick()
                world.render(display)
                pygame.display.flip()
        
        np.savetxt('speed_comp.csv',np.array(vel_accs))
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()      
                    
    pbar.close()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "model3")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    
    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

