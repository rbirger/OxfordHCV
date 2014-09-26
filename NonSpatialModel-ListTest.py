# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext cythonmagic

# <codecell>

from __future__ import division
import numpy as np
import random

class HCVHepatocyte:
    def __init__(self, cellID, parentID, infType, tLat, cellType, tInf = None, tDead = None):
        self.cellID = cellID      #ID of cell
        self.parentID = parentID  #ID of infector, whether it is virus or infected cell
        self.infType = infType    #type of infection (from virus or from infected cell)
        self.tLat = tLat          #time of infection of cell (time cell became latently infected)
        self.cellType = cellType  #type of cell latent, longterm, infectious, infectious from longterm,
                                  #dead, dead from long term
        self.tInf = tInf          #time to become infectious
        self.tDead = tDead        #time of death
        
        if cellType in ('Infected', 'InfectedL'):
            if tInf == None:
                print("Error: Infectious cells must have time Infectious")
        elif cellType in ('Dead', 'DeadL'):
            if tInf == None:
                print("Error: Dead cells must have time of death")
    #define method for infecting a susceptible cell
    def InfectCell(self, newID, simTime, newInfType):
        ''' Method for infecting new cell'''
        if  self.cellType not in ['Infected', 'InfectedL']:
            print("Error: Latent Cell cannot infect")
        else:
            return HCVHepatocyte(newID, self.cellID, 'Cell', simTime, newInfType)
              

#Create function to randomly select one cell to infect
def CreateLatent(cellHandle, newID, state_idx, simTime):
    if state_idx in [0,1]:
        newLatent = cellHandle.InfectCell(newID, simTime, 'Latent')
        return newLatent
    elif state_idx in [2,3]:
        newLatent = cellHandle.InfectCell(newID, simTime, 'LatentL')
        return newLatent
    else:
        print("Error: State is not an infecting transition")

#Create function to Kill Infected cell            
def KillInfected(cellHandle, time):
    cellHandle.tDead = time
    if cellHandle.cellType == 'Infected':
        cellHandle.cellType = 'Dead'
    elif cellHandle.cellType == 'InfectedL':
        cellHandle.cellType = 'DeadL'
    else:
        print("Error: Cannot kill uninfected cell")
    return cellHandle

#Create function to move latent to infectious
def LatentInfectious(cellHandle, time):
    cellHandle.tInf = time
    if cellHandle.cellType == 'Latent':
        cellHandle.cellType = 'Infected'
    elif cellHandle.cellType == 'LatentL':
        cellHandle.cellType = 'InfectedL'
    else:
        print("Error: Cell not Latent")
    return cellHandle

# <codecell>

#Number of state transitions to observe
M = int(1e5)
# time vector
time = np.zeros(M)

#Define parameters

init=10 #10 #initial number of infected hepatocytes
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
eta = .01 #proportion of infected cells that go long-term latent
kappa = 0 #.1 #proportion of dead infected cells regenerated as infected cells
changes = 12;
delta = .33 #ALT degradation rate
N=N_liver/1e7 #initial number of hepatocytes
eps = (delta*ALT_init)/(nu_T*N) #rate of ALT production



Q = np.zeros(changes)
Q[0] = (1-eta)*(beta_L*init) #Infection of Target cell by cell-> latent
Q[1] = (1-eta)*beta_V*v_init #Infection of Target cell by virus -> latent
Q[2] = eta*beta_L*init  #Infection of Target cell by cell -> long-term latent
Q[3] = eta*beta_V*v_init #Infection of Target cell by virus -> long-term latent
Q[4] = nu_T; #Death of target cell

Q[5] = alpha; #latent cell becomes infected
Q[6] = nu_T; #latent cell dies

Q[7] = alpha_x #long-term latent cell becomes infected
Q[8] = nu_T #long-term latent cell dies

Q[9] = nu_I; #Infected cell dies

Q[10] = phi_T; #Healthy cell regenerates

Q[11] = (1-kappa)*phi_I; #Infected cell regenerates into healthy cell
#Q[12] = kappa*phi_I

#Construct matrix of state transition vectors
trans_vecs = np.zeros([6, changes])
#state 1: infection of healthy cell by cell-> latent
trans_vecs[0,0] = -1;
trans_vecs[1,0] = 1;
#state 2: infection of healthy cell by virus -> latent
trans_vecs[0,1] = -1;
trans_vecs[1,1] = 1;
#state 3: infection of healthy cell by cell -> long-term latent
trans_vecs[0,2] = -1;
trans_vecs[2,2] = 1;
#state 4: infection of healthy cell by virus -> long-term latent
trans_vecs[0,3] = -1;
trans_vecs[2,3] = 1;
#state 5: death of healthy cell
trans_vecs[0,4] = -1;
trans_vecs[4,4] = 1;
#state 6: movement of latent cell into infected
trans_vecs[1,5] = -1;
trans_vecs[3,5] = 1;
#state 7: death of latent cell 
trans_vecs[1,6] = -1;
trans_vecs[4,6] = 1;
#state 8: movement of long-term latent cell into infected
trans_vecs[2,7] = -1;
trans_vecs[3,7] = 1;
#state 9: death of long-term latent cell 
trans_vecs[2,8] = -1;
trans_vecs[4,8] = 1;
#state 10: death of infected cell
trans_vecs[3,9] = -1;
trans_vecs[5,9] = 1;
#state 11: regeneration of dead healthy cell
trans_vecs[4,10] = -1;
trans_vecs[0,10] = 1;
#state 12: regeneration of dead infected cell into healthy cell
trans_vecs[5,11] = -1;
trans_vecs[0,11] = 1;
#state 13: regeneration of dead infected cell into infected cell
#trans_vecs[5,12] = -1;
#trans_vecs[3,12] = 1;

#Intialize random uniform numbers for distributions
time_vec_array = -np.log(np.random.random([M,changes]))
                         
#Initialize state variable vectors
T = np.zeros(M)
E = np.zeros(M)
Ex = np.zeros(M)
I = np.zeros(M)
Dt = np.zeros(M)
Di = np.zeros(M)
VL = np.zeros(M)
ALT = np.zeros(M)
state_vec = np.zeros(M)
InfectionChain = []
Infecteds = []
#Initialize Infected Hepatocyte objects
InfectedDict = range(init)
#for i in range(0,int(init/2)):
#    x = HCVHepatocyte(i, None, 'Initial', -1, 'Infected', 0)
#    InfectedDict.append(x)

#for i in range(int(init/2),init):
#    x = HCVHepatocyte(i, None, 'Initial', -83, 'InfectedL', 0)
#    InfectedDict.append(x)
    
LatentDict = []
LatentLDict = []
DeadDict = []
lastCellID = init-1 #get last cellID


#Input initial conditions
I[0] = init;
T[0] = N-init;
VL[0] = v_init 

j =0
InfectionArray = []
while I[j] >= 0 and j<M-1:
    #print [T[j],E[j],I[j],Dt[j],Di[j]]
    #Update Q to reflect new number of infected cells and viruses
    Q[0] = (1-eta)*beta_L*I[j] 
    Q[1] = (1-eta)*beta_V*VL[j]
    Q[2] = eta*beta_L*I[j] 
    Q[3] = eta*beta_V*VL[j]
    #Calculate transition matrix
    Qij = Q*[T[j],T[j],T[j], T[j],T[j], E[j],E[j], Ex[j], Ex[j], I[j], Dt[j], Di[j]]
    #Draw from exponential distributions of waiting times
    time_vec = time_vec_array[j]/Qij
    #time_vec = np.random.exponential([1/Qij])[0]
    #
    #find minimum waiting time and obtain index to ascertain next state jump
    newTime = min(time_vec) 
    time_vecL = time_vec.tolist()
    state_idx = time_vecL.index(min(time_vecL))
    state_vec[j] = state_idx
    [T[j+1],E[j+1],Ex[j+1],I[j+1],Dt[j+1],Di[j+1]]=[T[j],E[j],Ex[j],I[j],Dt[j],Di[j]]+ trans_vecs[:,state_idx]
    #make adjustments to hepatocyte dictionaries according to state transition
     
    #Infection of healthy cell by cell or virus -> latent or longterm latent    
    if state_idx in [0,1,2,3]: 
        Infector = random.choice(InfectedDict) #choose random infector cell
        newCellID = lastCellID + 1
        lastCellID = newCellID
        newLatent = newCellID #CreateLatent(Infector, newCellID, state_idx, time[j])
        if state_idx in [0,1]:
            LatentDict.append(newLatent)
        elif state_idx in [2,3]:
            LatentLDict.append(newLatent)
        else:
            print('Incorrect State')
    
    #Latent cell becomes infectious        
    elif state_idx in [5,7]:
        if state_idx == 5:
            LatCell = random.choice(LatentDict)
            LatentDict.remove(LatCell) #remove cell from Latent Dict
        elif state_idx == 7:
            LatCell = random.choice(LatentLDict)
            LatentLDict.remove(LatCell) 
        else:
            print('Incorrect State')
        #x = LatentInfectious(LatCell, time[j]) #add cell to Infected Dict
        InfectedDict.append(LatCell)

    #Latent cell dies    
    elif state_idx == 6: 
        LatentDict.remove(random.choice(LatentDict))
        
    #LatentL cell dies
    elif state_idx == 8: 
        LatentLDict.remove(random.choice(LatentLDict))
    
    #Infected cell dies
    elif state_idx == 9:
        KilledCell = random.choice(InfectedDict) #choose random infector cell
        InfectedDict.remove(KilledCell)
        #KilledCell.cellType = 'Dead'
        #KilledCell.tDead = time[j]
        #newDead = KillInfected(KilledCell,time[j])
        #DeadDict[newDead.cellID] = newDead
        DeadDict.append(KilledCell)
        
    #Dead infected cell regenerates into healthy cell -- just delete from dead dict
    elif state_idx == 11:
       DeadDict.remove(random.choice(DeadDict))
        
    #Infected cell regenerated from Dead cell    
   
    #Output Infection chain and infecteds at each time step
     #check lengths of InfectionChain and Infecteds
    if len(InfectionChain)< int(time[j])+1:
        InfectionChain.append([])
    if len(Infecteds) < int(time[j])+1:
        Infecteds.append([])
        
    #add to array of infections with timestep
    if state_idx in [0,1,2,3]:  
        #if int(time[j]) in InfectionChain:
         #   InfectionChain[int(time[j])].append([Infector.cellID, newCellID])
        #else:
         #   InfectionChain[int(time[j])] = [[Infector.cellID, newCellID]]
         InfectionChain[int(time[j])].append([Infector, newCellID])
    #elif state_idx == 12:
        #if int(time[j]) in InfectionChain:
         #   InfectionChain[int(time[j])].append([DeadGen.cellID, newInfected.cellID])
        #else:
         #   InfectionChain[int(time[j])] = [DeadGen.cellID, newInfected.cellID]
        #InfectionChain[int(time[j])].append([DeadGen.cellID, newInfected.cellID])
    #else:
     #   InfectionChain.append([])
                               
    #Infecteds.append(int([time[j]),list(InfectedDict.keys())])
    #if int(time[j]) in Infecteds:
    InfectedIDs = InfectedDict+ LatentDict+LatentLDict
    Infecteds[int(time[j])] = list(set(Infecteds[int(time[j])] + InfectedIDs))
    
        
    #update viral load and ALT    
    VL[j+1] = np.floor(rho*N_liver*(I[j+1]/N)*R/(gamma*c))  #VL[j] + (I[j]/N)*rho*N_liver*newTime - c*gamma*VL[j]*newTime #
    ALT[j+1] = ALT[j] + (eps*(nu_T*(T[j] + E[j] + Ex[j]) + nu_I*I[j])-delta*ALT[j])*newTime
    time[j+1] = time[j] + newTime
    j+=1

# <codecell>

Dt

# <codecell>

%timeit np.log(np.random.random(1))/5

# <codecell>

#re-write with functions
#Function Definitions
import csv
from __future__ import division
import numpy as np
import random

# <codecell>

def GenTransitionProbs(changes, eta, beta_L, beta_V, nu_T, alpha, alpha_x, nu_I, phi_T, phi_I, iCells, vLoad, state_vec):
    ''' Generate matrix of transition probabilities given parameters and state vales'''
    Q = np.zeros(changes)
    Q[0] = (1-eta)*(beta_L*iCells) #Infection of Target cell by cell-> latent
    Q[1] = (1-eta)*beta_V*vLoad #Infection of Target cell by virus -> latent
    Q[2] = eta*beta_L*init  #Infection of Target cell by cell -> long-term latent
    Q[3] = eta*beta_V*v_init #Infection of Target cell by virus -> long-term latent
    Q[4] = nu_T; #Death of target cell
    
    Q[5] = alpha; #latent cell becomes infected
    Q[6] = nu_T; #latent cell dies
    
    Q[7] = alpha_x #long-term latent cell becomes infected
    Q[8] = nu_T #long-term latent cell dies
    
    Q[9] = nu_I; #Infected cell dies
    
    Q[10] = phi_T; #Healthy cell regenerates
    
    Q[11] = (1-kappa)*phi_I; #Infected cell regenerates into healthy cell
    
    Qij = Q*state_vec
    
    return Qij

def CalcNextState(Qij, randNums, oldTime):
    #Draw from exponential distributions of waiting times using pre-calculated random numbers
    time_vec = randNums/Qij
    #
    #find minimum waiting time and obtain index to ascertain next state jump
    newTime = min(time_vec) + oldTime
    time_vecL = time_vec.tolist()
    state_idx = time_vecL.index(min(time_vecL))
    return newTime, state_idx

def UpdateStateVector(oldStates, trans_vecs, state_idx):
    newStates = oldStates + trans_vecs[:,state_idx]
    return newStates

def UpdateInfectionLists(state_idx, InfectedList, LatentList, LatentXList, lastCellID):
    '''Susceptible cell becomes infected: update Latent Lists''' 
    Infector = random.choice(InfectedList) #choose random infector cell
    newCellID = lastCellID + 1
    lastCellID = newCellID
    newLatent = newCellID #CreateLatent(Infector, newCellID, state_idx, time[j])
    if state_idx in [0,1]:
        LatentList.append(newLatent)
    elif state_idx in [2,3]:
        LatentXList.append(newLatent)
    else:
        print('Incorrect State for Infecting Function')
    return InfectedList, LatentList, LatentXList, Infector, lastCellID

    
def UpdateLatent2Infectious(state_idx, InfectedList, LatentList, LatentXList):
    '''Latent cell becomes infectious'''        
    if state_idx == 5:
        LatCell = random.choice(LatentList)
        LatentList.remove(LatCell) #remove cell from Latent List
    elif state_idx == 7:
        LatCell = random.choice(LatentXList)
        LatentXList.remove(LatCell) 
    else:
        print('Incorrect State for Latent2Infectious')
    InfectedList.append(LatCell)
    return InfectedList, LatentList, LatentXList

def UpdateKillCell(state_idx, InfectedList, LatentList, LatentXList):  
    '''Latent, LatentX, or Infected Cell Dies'''
    #Latent cell dise
    if state_idx == 6: 
        LatentList.remove(random.choice(LatentList))        
    #LatentX cell dies
    elif state_idx == 8: 
        LatentXList.remove(random.choice(LatentXList))    
    #Infected cell dies
    elif state_idx == 9:
        InfectedList.remove(random.choice(InfectedList))
    else:
        print("State not a cell-death state")
    return InfectedList, LatentList, LatentXList

def UpdateInfectionChain(InfectionChain, Infector, lastCellID, time):
    '''Update Transmission Chain list'''
    if len(InfectionChain)< int(time)+1:
        InfectionChain.append([])
    InfectionChain[int(time)].append([Infector, lastCellID])
    return InfectionChain
    
    
def UpdateInfecteds(Infecteds, time):
    '''Update Infecteds List'''
    if len(Infecteds) < int(time)+1:
        Infecteds.append([])
    InfectedIDs = InfectedList+ LatentList+LatentXList
    Infecteds[int(time)] = list(set(Infecteds[int(time)] + InfectedIDs))
    return Infecteds

def UpdateVL(rho, N_liver, N, R, gamma, c, iCells):
    VL = np.floor(rho*N_liver*(iCells/N)*R/(gamma*c))
    return VL

def UpdateALT(eps, nu_T, nu_I, delta, oldALT, T, E, Ex, I, tStep):    
    newALT = (eps*(nu_T*(T + E+ Ex) + nu_I*I)-delta*oldALT)*tStep
    return newALT

def OutputPrevFile(Infecteds, FileName):
    #First, sort Infecteds so they are in order of cell ID
    InfectedsSort = dict()
    for key, item in enumerate(Infecteds):
        InfectedsSort[key] = sorted(Infecteds[i])
    f = open(FileName, 'w') 
    writer = csv.writer(f, delimiter = ' ')
    for key, value in InfectedsSort.iteritems():
        writer.writerow([key] + value)

def OutputChainFile(InfectionChain, FileName):
    #Sort Infecteds and Infection chain, and break up infection chain
    InfectionChainSort = dict()
    for key, item in enumerate(InfectionChain):
        a = sorted(list(InfectionChain[i]), key=lambda x: x[0])
        InfectionChainSort[key] = [b for c in a for b in c]
    f = open(FileName, 'w') 
    writer = csv.writer(f, delimiter = ' ')
    for key, value in InfectionChainSort.iteritems():
        writer.writerow([key] + value)


def saveWorkspace(kwargs, filename = None ):
    pickleDict = dict()
    for key, item in kwargs.iteritems():
        pickleDict[key] = item
    #picklelist = [T, E, Ex, I, Dt, Di, time, VL, ALT, Infecteds, InfectionChain]
    if filename == None:
        now = datetime.datetime.now()
        filename = 'Workspace_'+ now.strftime("%I%M%S%B%d")
    tempFile = open(filename, 'w')
    # now let's pickle picklelist
    pickle.dump(pickleDict,tempFile)
    # close the file, and your pickling is complete
    tempFile.close()
    return filename

def loadWorkspace(filename):
    unpicklefile = open(filename, 'r')
    # now load the list that we pickled into a new object
    unpickledDict = pickle.load(unpicklefile)
    # close the file, just for safety
    unpicklefile.close()
    return unpickledDict
    

# <codecell>

#Script

#Number of state transitions to observe
M = int(1e5)
# time vector
time = np.zeros(M)

#Define parameters

init=10 #10 #initial number of infected hepatocytes
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
eta = .01 #proportion of infected cells that go long-term latent
kappa = 0 #.1 #proportion of dead infected cells regenerated as infected cells
changes = 12;
delta = .33 #ALT degradation rate
N=N_liver/1e7 #initial number of hepatocytes
eps = (delta*ALT_init)/(nu_T*N) #rate of ALT production


            
#Construct matrix of state transition vectors
trans_vecs = np.zeros([6, changes])
#state 1: infection of healthy cell by cell-> latent
trans_vecs[0,0] = -1;
trans_vecs[1,0] = 1;
#state 2: infection of healthy cell by virus -> latent
trans_vecs[0,1] = -1;
trans_vecs[1,1] = 1;
#state 3: infection of healthy cell by cell -> long-term latent
trans_vecs[0,2] = -1;
trans_vecs[2,2] = 1;
#state 4: infection of healthy cell by virus -> long-term latent
trans_vecs[0,3] = -1;
trans_vecs[2,3] = 1;
#state 5: death of healthy cell
trans_vecs[0,4] = -1;
trans_vecs[4,4] = 1;
#state 6: movement of latent cell into infected
trans_vecs[1,5] = -1;
trans_vecs[3,5] = 1;
#state 7: death of latent cell 
trans_vecs[1,6] = -1;
trans_vecs[4,6] = 1;
#state 8: movement of long-term latent cell into infected
trans_vecs[2,7] = -1;
trans_vecs[3,7] = 1;
#state 9: death of long-term latent cell 
trans_vecs[2,8] = -1;
trans_vecs[4,8] = 1;
#state 10: death of infected cell
trans_vecs[3,9] = -1;
trans_vecs[5,9] = 1;
#state 11: regeneration of dead healthy cell
trans_vecs[4,10] = -1;
trans_vecs[0,10] = 1;
#state 12: regeneration of dead infected cell into healthy cell
trans_vecs[5,11] = -1;
trans_vecs[0,11] = 1;
#state 13: regeneration of dead infected cell into infected cell
#trans_vecs[5,12] = -1;
#trans_vecs[3,12] = 1;

#Intialize random uniform numbers for distributions
time_vec_array = -np.log(np.random.random([M,changes]))
                         
#Initialize state variable vectors
T = np.zeros(M)
E = np.zeros(M)
Ex = np.zeros(M)
I = np.zeros(M)
Dt = np.zeros(M)
Di = np.zeros(M)
VL = np.zeros(M)
ALT = np.zeros(M)

#Initialize lists 
InfectionChain = []
Infecteds = []
#Initialize Infected Hepatocyte objects
InfectedList = range(init)
#for i in range(0,int(init/2)):
#    x = HCVHepatocyte(i, None, 'Initial', -1, 'Infected', 0)
#    InfectedDict.append(x)

#for i in range(int(init/2),init):
#    x = HCVHepatocyte(i, None, 'Initial', -83, 'InfectedL', 0)
#    InfectedDict.append(x)
    
LatentList = []
LatentXList= []
DeadList = []
lastCellID = init-1 #get last cellID


#Input initial conditions
I[0] = init;
T[0] = N-init;
VL[0] = v_init 

j =0
InfectionArray = []
statesVec = [T[0], E[0], Ex[0], I[0], Dt[0], Di[0]] 
while I[j] >= 0 and j<M-1:
    
    #Generate Transition Probabilities
    mult_vec = [T[j],T[j],T[j], T[j],T[j], E[j],E[j], Ex[j], Ex[j], I[j], Dt[j], Di[j]]
    Qij = GenTransitionProbs(changes, eta, beta_L, beta_V, nu_T, alpha, alpha_x, nu_I, phi_T, phi_I, I[j], VL[j], mult_vec)
    
    #Calculate what the next state transition should be
    time[j+1], state_idx  = CalcNextState(Qij, time_vec_array[j], time[j])
    
    #Update the state vector
    statesVec = UpdateStateVector(statesVec, trans_vecs, state_idx)
    [T[j+1], E[j+1], Ex[j+1], I[j+1], Dt[j+1], Di[j+1]] = statesVec
    
    if state_idx in [0,1,2,3]:
        InfectedList, LatentList, LatentXList, Infector, lastCellID = UpdateInfectionLists(state_idx, InfectedList, LatentList, LatentXList, lastCellID)
        InfectionChain = UpdateInfectionChain(InfectionChain, Infector, lastCellID, time[j])
    elif state_idx in [5,7]:
        InfectedList, LatentList, LatentXList = UpdateLatent2Infectious(state_idx, InfectedList, LatentList, LatentXList)
    elif state_idx in [6,8,9]:
        InfectedList, LatentList, LatentXList = UpdateKillCell(state_idx, InfectedList, LatentList, LatentXList)
    
    #Update list of Infecteds
    Infecteds = UpdateInfecteds(Infecteds, time[j])
    
    #update viral load and ALT
    VL[j+1] = UpdateVL(rho, N_liver, N, R, gamma, c, I[j+1])
    ALT[j+1] = UpdateALT(eps, nu_T, nu_I, delta, oldALT, T[j], E[j], Ex[j], I[j], time[j+1]-time[j])
  
    j+=1


OutputPrevFile(Infecteds, 'InfectedsTest.txt')
OutputChainFile(InfectionChain, 'InfectionChainTest.txt')
kwargs = {'T' : T, 'E' : E, 'Ex': Ex, 'I': I, 'Dt':Dt, 'Di' : Di, 'time' :time, 'VL':VL, 'ALT' : ALT, 'Infecteds' :Infecteds, 'InfectionChain' : InfectionChain}
saveWorkspace(kwargs, 'testFilename')

# <codecell>

import pickle
import datetime

# <codecell>

str(int(np.log10(M)))

# <codecell>

kwargs = {'T' : T, 'E' : E, 'Ex': Ex, 'I': I, 'Dt':Dt, 'Di' : Di, 'time' :time, 'VL':VL, 'ALT' : ALT, 'Infecteds' :Infecteds, 'InfectionChain' : InfectionChain}

# <codecell>


# <codecell>

saveWorkspace(kwargs, 'testFilename')

# <codecell>

# lets create something to be pickled
# How about a list?
picklelist = [T, E, Ex, I, Dt, Di, time, VL, ALT]

# now create a file
# replace filename with the file you want to create
testFile = open('SavedStates', 'w')

# now let's pickle picklelist
pickle.dump(picklelist,testFile)

# close the file, and your pickling is complete
testFile.close()

# <codecell>

InfectionChain[int(time[j])]

# <codecell>

[Infector, lastCellID]

# <markdowncell>

# * write function for saving workspace
# * implement filename with date
# * create module of functions
# * make several different notebooks for different scenarios

# <codecell>


