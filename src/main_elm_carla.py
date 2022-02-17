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
from system import AAC
from controller import LCBF, PID
from dataset import ELMDataset, NNDataset
from estimator import *
from normalizer import *
from functions import *

# For carla
from carla_utils import *

# Parameters
dt = 0.1
simTime = 20

# Real parameters
v_lead = 22
v_des = 24
m  = 1650.0
g = 9.81

f0 = 0.1
f1 = 5
f2 = 0.25

c_a = 0.3
c_d = 0.3
Th = 1.8

# Nominal parameters
f0_nom = 10*f0
f1_nom = 10*f1
f2_nom = 10*f2

m_nom = 0.8*m

# QP-CLF-CBF parameters
p_slack = 2e-2
clf_rate = 5
cbf_rate = 5.

torch.manual_seed(42)

def game_loop(args):    

    ########################################
    #    System
    ########################################
    aac = AAC(m, c_d, f0, f1, f2, v_lead)
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

    learned_ratio = 1.5
    time_th = learned_ratio* hidden_size

    ########################################
    #    PID control reference
    ########################################
    x_dim = 3
    u_dim = 1

    kp = np.array([[0, 0.2, 0]])
    kd = np.array([[0, 1e-3, 0]])
    ki = np.array([[0, 0.2, 0]])

    pid = PID(x_dim, u_dim, kp, kd, ki, dt)

    ########################################
    #    Training parameters or initial states
    ########################################
    lr_pres =  [1e-3]   #[1e-2, 1e-3]
    lr_posts =  [1e-3]  #[1e-2]
    z0s = [28] #[28,30,32,34,38] #[36]#[30,32,34,38]  #[30, 34, 38]
    v0s = [20,22,24,26] # [20]#[20,22,24,26]
    funcs = [step]#, sin, square] # Square or sin

    # Path for saving data
    data_dir = '../data/elm'
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
    
        for lr_pre, lr_post, z0, v0, func in itertools.product(lr_pres, lr_posts, z0s, v0s, funcs):
            # Add waypoint markers
            world.restart(z0)
            t = world.player.get_transform()
            t_opp = world.opponent.get_transform()
            angle_heading = t.rotation.yaw * pi/ 180
            world.player.set_velocity(carla.Vector3D(float(v0*math.cos(angle_heading)),float(v0*math.sin(angle_heading)),0))
            world.player.apply_control(carla.VehicleControl(throttle=1, brake=0, steer=0, manual_gear_shift=True, gear=4))
            
            if SCENE == 'one_vehicle' :
                world.opponent.set_velocity(carla.Vector3D(float(v_lead*math.cos(angle_heading)),float(v_lead*math.sin(angle_heading)),0))
                world.opponent.apply_control(carla.VehicleControl(throttle=1, brake=0, steer=0, manual_gear_shift=True, gear=4))
            

            start_x = t.location.x
            start_y = t.location.y
            start_x_opp = t_opp.location.x
            start_y_opp = t_opp.location.y
            
            curr_speed = 0
            ####################################################
            ##############  Save data
            ####################################################
            fn = "lr_pre_{}_lr_post_{}_z0_{}_v0_{}_func_{}.csv".format(lr_pre, lr_post, z0, v0, func.__name__)
            column_names = ['p', 'v', 'z', 'u','u_ref','V','h','dhe_real','dhe','slack']

            df = pd.DataFrame(columns=column_names,dtype=object)
            path = os.path.join(data_dir, fn)
            df.to_csv(path, index=False)        

            ####################################################
            ##############  Initialization
            ####################################################

            # Initial position
            x = [0, v0, z0]
            
            # Estimator
            estimator = EstimatorELM(input_size, hidden_size, output_size, time_th, dt, lr_pre, lr_post)
            
            ## Dataset
            dataset = ELMDataset(dt, ('x', 'k', 'dhe_real'), time_th)

            for t in np.arange(0, simTime, dt): #simTime
                t1 = hud.simulation_time
                
                tr = world.player.get_transform()
                v = world.player.get_velocity()
                if SCENE == 'one_vehicle' or SCENE == 'one_vehicle_turn' :
                    t_opp = world.opponent.get_transform()
                    v_opp = world.opponent.get_velocity()
                    x_obst = t_opp.location.x
                    y_obst = t_opp.location.y
                    vx_obst = v_opp.x
                    vy_obst = v_opp.y
                    yaw_obst = t_opp.rotation.yaw * pi/ 180
                
                px = tr.location.x
                py = tr.location.y
                angle_heading = tr.rotation.yaw * pi/ 180
                vx = v.x
                vy = v.y
                
        
                # Get reference control input: u_ref
                e = np.array([[0], [v_des], [0]]) - np.expand_dims(x, axis = 1)
                u_ref = pid.update(e)
                u_ref = u_ref[0,0]
            
                # Simulate dynamic uncertainty
                unct = func(t)
                aac.v_lead = v_lead + unct  # lead_vehicle

                # Controller
                k, slack_sol, V, dV, h, dh, dhe, dS = cont.compute_controller(x, u_ref, estimator, t) 
            
                # System update
                x_n = aac.update(x, k, t, dt)

                # Obtaining label: dhe_real
                dh_real = derivator.update(h)
                dhe_real = dh_real - dh

                # Update dataset
                dataset.update(t, x, k, dhe_real)

                # Update estimator: Training
                estimator.train(t, dataset)

                # Update data saved
                row = c2l(x) + c2l(k) + c2l(u_ref) + c2l(V) + c2l(h) + c2l(dhe_real) + c2l(dhe) + c2l(slack_sol)
                df_row = pd.DataFrame(dict(zip(column_names, row)), index = [0])
                df.append(df_row, sort = False).to_csv(path, index=False, mode = 'a', header=False)

                # Update new state
                x = x_n
                ego_pos = x[0]
                opp_pos = x[0] + x[2]
                pbar.update(1)
                curr_loc_ego = tr.location
                curr_loc_ego.x = start_x + ego_pos
                curr_loc_opp = t_opp.location
                curr_loc_opp.x = start_x + opp_pos
                world.player.set_location(curr_loc_ego)
                world.opponent.set_location(curr_loc_opp)
                print(x[2])
                world.tick(clock)
                client.get_world().tick()
                world.render(display)
                pygame.display.flip()
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

