# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
# <codecell>
from __future__ import division
import csv
import numpy as np
import random
import pickle
import datetime
from NonSpatialFns import *

# <codecell>

#Script
#Script

#Number of state transitions to observe
M = int(3e7)
# time vector
time = np.zeros(M)

#Define parameters

init1=10 #10 #initial number of infected hepatocytes
init2 = 1 #initial number of infected extra-hepatic cells
v_init = 0#initial viral load
ALT_init = 100 #initial ALT level

rho = 8.18 #viral export rate
c = 22.3 #viral clearance rate
gamma = 1500 #scaling factor - 
R = 4.1825 #average HCV RNA in infected hepatocyte
N_liver = int(1e11) #Number of cells in liver
alpha = 1 #1/latent period (days)
alpha_x = 1.3e-2 #1/long-term latent period
nu_T = 1.4e-2 #death rate of healthy cells
nu_I = 1/7 #death rate of infected cells
phi_T = 10*nu_T #regeneration rate of dead healthy cells
phi_I = .8*phi_T #regeneration rate of dead infected cells
beta_V = .5e-8 #viral transmision rate
beta_L = R*1e-5/(60*24) #cell-cell transmission rate
eta = 0 #.01 #proportion of infected cells that go long-term latent
kappa = 0 #.1 #proportion of dead infected cells regenerated as infected cells
nu_T2 = 1.4e-2 #death rate/regeneration rate of second patch cells
beta_L2 = R*1e-5/(60*24) #cell-cell transmission rate for second patch cells
m12 = .6 #patch-sharing proportion between virions from patch 1 to patch 2
m21 = .2 #patch-sharing proportion between virions from patch 2 to patch 1
changes = 20;
delta = .33 #ALT degradation rate
N=N_liver/1e7 #initial number of hepatocytes
N2 = 1e3 #initial number of extra-hepatic target cells
eps = (delta*ALT_init)/(nu_T*N) #rate of ALT production


            
#Construct matrix of state transition vectors
trans_vecs = np.zeros([8, changes])
#state 0: infection of healthy cell by cell-> latent
trans_vecs[0,0] = -1;
trans_vecs[1,0] = 1;
#state 1: infection of healthy cell by virus -> latent
trans_vecs[0,1] = -1;
trans_vecs[1,1] = 1;
#state 2: infection of healthy cell by cell -> long-term latent
trans_vecs[0,2] = -1;
trans_vecs[2,2] = 1;
#state 3: infection of healthy cell by virus -> long-term latent
trans_vecs[0,3] = -1;
trans_vecs[2,3] = 1;
#state 4: death of healthy cell
trans_vecs[0,4] = -1;
trans_vecs[4,4] = 1;
#state 5: movement of latent cell into infected
trans_vecs[1,5] = -1;
trans_vecs[3,5] = 1;
#state 6: death of latent cell 
trans_vecs[1,6] = -1;
trans_vecs[4,6] = 1;
#state 7: movement of long-term latent cell into infected
trans_vecs[2,7] = -1;
trans_vecs[3,7] = 1;
#state 8: death of long-term latent cell 
trans_vecs[2,8] = -1;
trans_vecs[4,8] = 1;
#state 9: death of infected cell
trans_vecs[3,9] = -1;
trans_vecs[5,9] = 1;
#state 10: regeneration of dead healthy cell
trans_vecs[4,10] = -1;
trans_vecs[0,10] = 1;
#state 11: regeneration of dead infected cell into healthy cell
trans_vecs[5,11] = -1;
trans_vecs[0,11] = 1;
#state 12: Target cell type 1 infected by virus type 2 -> latent
trans_vecs[0,12] = -1
trans_vecs[1,12] = 1
#state 13: Target cell type 1 infected by virus type 2 -> long term latent
trans_vecs[0,13] = -1
trans_vecs[2,13] = 1
#state 14: infection of target cell 2 by cell
trans_vecs[6,14] = -1;
trans_vecs[7,14] = 1;
#state 15: infection of target cell 2 by virus type 2
trans_vecs[6,15] = -1;
trans_vecs[7,15] = 1;
#state 16: infection of target cell 2 by virus type 1
trans_vecs[6,16] = -1;
trans_vecs[7,16] = 1;
#state 17: generation of new target cell 2
trans_vecs[6,17] = 1;
#state 18: death of target cell 2
trans_vecs[6, 18] = -1
#state 19: death of infected target cell 2
trans_vecs[7,19] = -1

#Intialize random uniform numbers for distributions
time_vec_array = -np.log(np.random.random([M,changes]))
                         
#Initialize state variable vectors
T = np.zeros(M)
E = np.zeros(M)
Ex = np.zeros(M)
I = np.zeros(M)
Dt = np.zeros(M)
Di = np.zeros(M)
T2 = np.zeros(M)
I2 = np.zeros(M)
VL = np.zeros(M)
VL2 = np.zeros(M)
ALT = np.zeros(M)

#Initialize lists 
InfectedList = range(init1)
LatentList = []
LatentXList= []
DeadList = []
Infected2List = range(-init1, -(init1+init2),-1)

InfectionChain = []
Infecteds = []

lastCellID = init1+init2-1 #get last cellID


#Input initial conditions
I[0] = init1;
I2[0] = init2
T[0] = N-init1;
T2[0] = N2-init2
VL[0] = v_init 

j = 0
minKey = 0
finalFile = False
statesVec = [T[0], E[0], Ex[0], I[0], Dt[0], Di[0], T2[0], I2[0]] 

#define names for output files
now = datetime.datetime.now()
OutPrevFileName = 'Infecteds_'+ '1e' + str(int(np.log10(M))) + '_'+ now.strftime("%I%M%S%B%d") + 'TwoPatch.txt'
OutChainFileName = 'InfectionChain_'+ '1e' + str(int(np.log10(M))) + '_' +now.strftime("%I%M%S%B%d") + 'TwoPatch.txt'
WorkspaceFilename = 'WorkSpace_' + '1e' + str(int(np.log10(M))) + '_' +now.strftime("%I%M%S%B%d") + 'TwoPatch.txt'

############ Run Model ###################
while I[j] >= 0 and j<M-1:
    
    #Generate Transition Probabilities
    mult_vec = [T[j],T[j],T[j], T[j],T[j], E[j],E[j], Ex[j], Ex[j], I[j], Dt[j], Di[j], T[j],T[j], T2[j], T2[j],T2[j], T2[j]+I2[j], T2[j], I2[j]]
    Qij = GenTransitionProbsTwoPatch(changes, eta, beta_L, beta_V, nu_T, alpha, alpha_x, nu_I, phi_T, phi_I, nu_T2, beta_L2, m12, m21, I[j], VL[j], I2[j], VL2[j], mult_vec)
    
    #Calculate what the next state transition should be
    time[j+1], state_idx  = CalcNextState(Qij, time_vec_array[j], time[j])
    
    #Update the state vector
    statesVec = UpdateStateVector(statesVec, trans_vecs, state_idx)
    [T[j+1], E[j+1], Ex[j+1], I[j+1], Dt[j+1], Di[j+1], T2[j+1], I2[j+1]] = statesVec
    
    #Update Cell lists given appropriate state transition
    if state_idx in [0,1,2,3,12,13,14,15,16]:
        InfectedList, Infected2List,LatentList, LatentXList, Infector, lastCellID = UpdateInfectionListsTwoPatch(state_idx, InfectedList, Infected2List, LatentList, LatentXList, lastCellID)
        InfectionChain = UpdateInfectionChain(InfectionChain, Infector, lastCellID, time[j], minKey)
    elif state_idx in [5,7]:
        InfectedList, LatentList, LatentXList = UpdateLatent2Infectious(state_idx, InfectedList, LatentList, LatentXList)
    elif state_idx ==9:
        InfectedList = UpdateKillCellGen(state_idx, InfectedList)
    elif state_idx ==6:
        LatentList = UpdateKillCellGen(state_idx, LatentList)
    elif state_idx ==8:
        LatentXList = UpdateKillCellGen(state_idx, LatentXList)
    elif state_idx ==19:
        Infected2List = UpdateKillCellGen(state_idx, Infected2List)
        
    #Update list of Infecteds
    Infecteds = UpdateInfectedsTwoPatch(Infecteds,InfectedList, Infected2List, LatentList, LatentXList, time[j], minKey)
    

    #update viral load and ALT
    VL[j+1] = UpdateVL(rho, N_liver, N, R, gamma, c, I[j+1])
    VL2[j+1] = UpdateVL(rho, N, N, R, gamma, c, I2[j+1])
    ALT[j+1] = UpdateALT(eps, nu_T, nu_I, delta, ALT[j], T[j], E[j], Ex[j], I[j], time[j+1]-time[j])

    j+=1

    #write output to file every timestep
    if minKey < int(time[j]) or j == M-1:
        if j == M-1:
            finalFile = True
        Infecteds, InfectionChain, minKey = OutputTempFiles(Infecteds, InfectionChain, minKey, OutPrevFileName, OutChainFileName, finalFile)

    
    
#######################################

## Save workspace ####

kwargs = {'T' : T, 'E' : E, 'Ex': Ex, 'I': I, 'Dt':Dt, 'Di' : Di, 'T2': T2, 'I2':I2, 'time' :time, 'VL':VL, 'VL2':VL2, 'ALT' : ALT, 'Infecteds' :Infecteds, 'InfectionChain' : InfectionChain}
saveWorkspace(kwargs, WorkspaceFilename)