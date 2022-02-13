#!/usr/bin/python3

# Basic
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import pinv2
from tqdm import tqdm
import pandas as pd
import itertools
from argparse import ArgumentParser
import logging


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

# Parameters
dt = 0.01
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

def main():
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
    z0s = [28,30,32,34,38] #[36]#[30,32,34,38]  #[30, 34, 38]
    v0s = [20,22,24,26] # [20]#[20,22,24,26]
    funcs = [step, sin, square] # Square or sin

    # Path for saving data
    data_dir = '../data/elm'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Total of cases
    cases = len(list(itertools.product(lr_pres, lr_posts, z0s, v0s, funcs)))
    pbar = tqdm(total=cases*simTime/dt)

    for lr_pre, lr_post, z0, v0, func in itertools.product(lr_pres, lr_posts, z0s, v0s, funcs):

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

            pbar.update(1)
                    
    pbar.close()


if __name__ == "__main__":
    # parser = ArgumentParser(description='Parameters for estimators')
    # parser.add_argument('--estimator', dest='estimator', type=str, help='Estimator type: NN or ELM')
    # parser.add_argument('--', dest='surname', type=str, help='Surname of the candidate')
    # parser.add_argument('--age', dest='age', type=int, help='Age of the candidate')

    # args = parser.parse_args()
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+c was pressed. Script run stopping.")