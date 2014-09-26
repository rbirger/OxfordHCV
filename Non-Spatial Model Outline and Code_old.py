# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###Description and preliminary code for Continuous-Time Markov Chain Model
# 
# This model will test the importance of including a spatial component in the system. We will use ODEs to describe the dynamics of each lineage and competition between lineages.  
# The different states that each cell can move through are as follows
# 
# * Healthy Hepatocytes
# 
# * Latently Infected Hepatocytes
# 
# * Infected Hepatocytes
# 
# * Dead Infected Hepatocytes
# 
# * Dead Healthy Hepatocytes
# 
# Healthy cells are regenerated from Dead cells. Interacting with Infected cells, they become Latently Infected, and after the eclipse phase, Latent Infections become Infectious. Both Healthy and Infected Hepatocytes die, with Infected being eliminated by the immune response faster than natural death rates. Dead cells regenerate, but those dead after being infected with HCV have a lower probability of regenerating.
# 
# Adapting the Perelson/Neumann model, we have
# 
# $\begin{eqnarray*} 
# \frac{dT}{dt}& =& \phi_{DT} D_T + \phi_{DI} D_I  - (\lambda_{virions} + \lambda_{local} +\nu_T) T\\
# \frac{dE}{dt}& =& (\lambda_{virions} + \lambda_{local} )T - (\alpha +\nu_T)E\\
# \frac{dI}{dt}& =& \alpha E- \nu_I I\\
# \frac{dD_T}{dt}& =& \nu_T(T+E) - \phi_{DT} D_T\\
# \frac{dD_I}{dt}& =& \nu_I I - \phi_{DI} D_I\\\
# \end{eqnarray*}$
# 
# 
# 
# 
# To translate these equations into a continuous-time Markov Chain model, we can calculate the transition probabilities from the parameters above. Let $\vec{X(t)} = [T(t), E(t), I(t), D_T(t), D_I(t)]$, so the probability of state change is defined as Prob$\{\Delta \vec{X(t)} = (a, b, c, d, e)|\vec{X(t)}\}$, where $a$ represents the change in state $T$, $b$ in state $E$, etc. We assume that the time step is small enough that each change is only in one cell, so $a - e$ can only take the values 0 or $\pm 1$. The transition probabilities are as follows
# 
# 
# $$\begin{cases}
# (\lambda_{virions} + \lambda_{local}) T\ \Delta t  + o(\Delta t), & a = -1, b = 1\\
# \nu_T T \Delta t  + o(\Delta t), & a = -1, d = 1\\
# \alpha E \Delta t  + o(\Delta t),  & b = -1, c = 1\\
# \nu_T E \Delta t  + o(\Delta t),  & b = -1, d = 1\\
# \nu_I I \Delta t  + o(\Delta t), & c = -1, e = 1 \\
# \phi_{DT} D_T \Delta t  + o(\Delta t), & d = -1, a = 1\\
# \phi_{DI} D_I \Delta t  + o(\Delta t), & e = -1, a = 1\\
# \end{cases}$$
# 
# The generator matrix $\mathbf{Q}$ derived from these transition probabilities is thus as follows
# 
# <!--($$ \mathbf{Q} = 
# \left[ \begin{array}{ccccc}
# - (\beta I + \lambda +d) T & (\beta I + \lambda) T & 0 & 0 & dT \\
# 0 & -(\eta + d) L & \eta L &0 & dL \\
# 0 & 0 & -\delta I & \delta I & 0 \\
# \alpha_I D_I &0 &0 & -\alpha_I D_I&0\\
# \alpha_T D_T & 0 & 0& 0& -\alpha_T D_T\\
# \end{array} \right] $$ -->
# 
# $$ \mathbf{Q} = 
# \left[ \begin{array}{ccccc}
# 0& (\lambda_{virions} + \lambda_{local}) T& 0 & 0 & \nu_T T \\
# 0 & 0 & \alpha E & \nu_T E &0 \\
# 0 & 0 & 0  & 0 & \nu_I I\\
# \phi_{DT} D_T &0 &0 & 0&0\\
# \phi_{DI} D_I & 0 & 0& 0& 0\\
# \end{array} \right] $$

# <codecell>

%matplotlib inline
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

beta=.2
nu = .01
d = 2e-2
eta = 1
delta = 3*d
alpha_I = .8e-1
alpha_T = 2e-1

# <codecell>

from __future__ import division
import numpy as np
#Number of state transitions to observe
M = int(1e6)
# time vector
time = np.zeros(M)

#Define parameters
rho = 8.18 #viral export rate
c = 22.3 #viral clearance rate
gamma = 1500 #scaling factor
R = 4.1825 #average HCV RNA in infected hepatocyte
N_liver = int(8e10) #Number of cells in liver
alpha = 1 #1/latent period (days)
nu_T = 1.4e-2 #death rate of healthy cells
nu_I = 1/7 #death rate of infected cells
phi_T = 10*nu_T #regeneration rate of dead healthy cells
phi_I = .8*phi_T #regeneration rate of dead infected cells
beta_V = 1e-8 #viral transmision rate
beta_L = R*1e-5/(60*24) #cell-cell transmission rate


N=N_liver/1e6
init=10

v_init = 1e6
sim=3

Q = np.zeros(7)
Q[0] = (beta_L*init + beta_V*v_init); #Infection of Target cell
Q[1] = nu_T; #Death of target cell
Q[2] = alpha; #latent cell becomes infected
Q[3] = nu_T; #latent cell dies
Q[4] = nu_I; #Infected cell dies
Q[5] = phi_T; #Healthy cell regenerates
Q[6] = phi_I; #Infected cell regenerates

#Construct matrix of state transition vectors
trans_vecs = np.zeros([5,7])
#state 1: infection of healthy cell
trans_vecs[0,0] = -1;
trans_vecs[1,0] = 1;
#state 2: death of healthy cell
trans_vecs[0,1] = -1;
trans_vecs[3,1] = 1;
#state 3: movement of latent cell into infected
trans_vecs[1,2] = -1;
trans_vecs[2,2] = 1;
#state 4: death of latent cell 
trans_vecs[1,3] = -1;
trans_vecs[3,3] = 1;
#state 5: death of infected cell
trans_vecs[2,4] = -1;
trans_vecs[4,4] = 1;
#state 6: regeneration of dead healthy cell
trans_vecs[3,5] = -1;
trans_vecs[0,5] = 1;
#state 6: regeneration of dead infected cell
trans_vecs[4,6] = -1;
trans_vecs[0,6] = 1;

#Initialize state variable vectors
T = np.zeros(M)
E = np.zeros(M)
I = np.zeros(M)
Dt = np.zeros(M)
Di = np.zeros(M)
VL = np.zeros(M)
#Input initial conditions
I[0] = init;
T[0] = N-init;
VL[0] = v_init 
#Initialize state vector and index
#state_vec = np.vstack([S,E,I,Di,Dt])

j =0
while I[j] >0 and j<M-1:
    #print [T[j],E[j],I[j],Dt[j],Di[j]]
    #Update Q to reflect new number of infected cells and viruses
    Q[0] = (beta_L*I[j] +beta_V*VL[j]);
    #Calculate transition matrix
    Qij = Q*[T[j],T[j],E[j],E[j],I[j],Dt[j],Di[j]]
    #Draw from exponential distributions of waiting times
    time_vec = -np.log(np.random.random(7))/Qij 
    #np.random.exponential([1/Qij])[0]
    #
    #find minimum waiting time and obtain index to ascertain next state jump
    newTime = min(time_vec)
    time_vecL = time_vec.tolist()
    state_idx = time_vecL.index(min(time_vecL))
    [T[j+1],E[j+1],I[j+1],Dt[j+1],Di[j+1]]=[T[j],E[j],I[j],Dt[j],Di[j]]+ trans_vecs[:,state_idx]
    VL[j+1] = VL[0]+rho*I[j]*R/(gamma*c)
    time[j+1] = time[j] + newTime
    j+=1

# <codecell>

[T[j],E[j],I[j],Dt[j],Di[j]]
rho*I[j]*R/(gamma*c)

# <codecell>

%%timeit
np.random.exponential(y)

# <codecell>

y= np.ones(11)

# <codecell>

plt.plot(time[0:M-1],VL[0:M-1])

# <codecell>

plt.plot(time,T, label = 'Susc')
plt.plot(time,I, label = 'Infected')
plt.plot(time,Dt, label = 'Dead (healthy)')
plt.plot(time,Di, label = 'Dead (infected)')
plt.legend(loc = 'upper right')

# <markdowncell>

# An updated version of the model includes a second latent class that keeps cells latently infected for longer before becoming infectious, and also allows for proliferation of infected cells by allowing cells to be reborn into the latent class
# 
# * Healthy Hepatocytes
# 
# * Latently Infected Hepatocytes
# 
# * Long-lived Latently Infected Hepatocytes
# 
# * Infected Hepatocytes
# 
# * Dead Infected Hepatocytes
# 
# * Dead Healthy Hepatocytes
# 
# Healthy cells are regenerated from Dead cells. Interacting with Infected cells, they become Latently Infected, and after the eclipse phase, Latent Infections become Infectious. Both Healthy and Infected Hepatocytes die, with Infected being eliminated by the immune response faster than natural death rates. Dead cells regenerate, but those dead after being infected with HCV have a lower probability of regenerating. Some cells regenerate into infectious cells.
# 
# Adapting the Perelson/Neumann model, we have
# 
# $\begin{eqnarray*} 
# \frac{dT}{dt}& =& \phi_{DT} D_T + (1-\kappa)\phi_{DI} D_I  - (\lambda_{virions} + \lambda_{local} +\nu_T) T\\
# \frac{dE}{dt}& =& (1-\eta)(\lambda_{virions} + \lambda_{local} )T - (\alpha +\nu_T)E\\
# \frac{dEX}{dt}& =& \eta(\lambda_{virions} + \lambda_{local} )T - (\alpha_X +\nu_T)E\\
# \frac{dI}{dt}& =& \kappa\phi_{DI} D_I+ \alpha E- \nu_I I\\
# \frac{dD_T}{dt}& =& \nu_T(T+E+EX) - \phi_{DT} D_T\\
# \frac{dD_I}{dt}& =& \nu_I I - \phi_{DI} D_I\\\
# \end{eqnarray*}$
# 
# To translate these equations into a continuous-time Markov Chain model, we can calculate the transition probabilities from the parameters above. Let $\vec{X(t)} = [T(t), E(t), EX(t) I(t), D_T(t), D_I(t)]$, so the probability of state change is defined as Prob$\{\Delta \vec{X(t)} = (a, b, c, d, e, f)|\vec{X(t)}\}$, where $a$ represents the change in state $T$, $b$ in state $E$, etc. We assume that the time step is small enough that each change is only in one cell, so $a - f$ can only take the values 0 or $\pm 1$. The transition probabilities are as follows
# 
# 
# $$\begin{cases}
# (1-\eta)(\lambda_{virions} + \lambda_{local}) T\ \Delta t  + o(\Delta t), & a = -1, b = 1\\
# \eta(\lambda_{virions} + \lambda_{local}) T\ \Delta t  + o(\Delta t), & a = -1, c = 1\\
# \nu_T T \Delta t  + o(\Delta t), & a = -1, e = 1\\
# \alpha E \Delta t  + o(\Delta t),  & b = -1, d = 1\\
# \nu_T E \Delta t  + o(\Delta t),  & b = -1, e = 1\\
# \alpha_X EX \Delta t  + o(\Delta t),  & c = -1, d = 1\\
# \nu_T EX \Delta t  + o(\Delta t),  & c = -1, e = 1\\
# \nu_I I \Delta t  + o(\Delta t), & d = -1, f = 1 \\
# \phi_{DT} D_T \Delta t  + o(\Delta t), & d = -1, a = 1\\
# \kappa\phi_{DI} D_I \Delta t  + o(\Delta t), & f = -1, d = 1\\
# (1-\kappa)\phi_{DI} D_I \Delta t  + o(\Delta t), & f = -1, a = 1\\
# \end{cases}$$
# 
# The generator matrix $\mathbf{Q}$ derived from these transition probabilities is thus as follows
# 
# 
# $$ \mathbf{Q} = 
# \left[ \begin{array}{cccccc}
# 0& (1-\eta)(\lambda_{virions} + \lambda_{local}) T& \eta(\lambda_{virions} + \lambda_{local}) T& 0 & \nu_T T &0\\
# 0 & 0 & \alpha E &0 &\nu_T E  & 0\\
# 0 & 0 & \alpha_X EX &0 &\nu_T E  & 0\\
# 0 & 0 & 0 & 0 & 0&\nu_I I \\
# \phi_{DT} D_T &0 &0 & 0&0&0\\
# (1-\kappa)\phi_{DI} D_I & 0 & 0& \kappa \phi_{DI}& 0&0\\
# \end{array} \right] $$
# 

# <codecell>

%load_ext cythonmagic

# <codecell>

%%cython
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
        
class HCVVirion:
    def __init__(self, virusID, parentID):
        self.virusID = virusID
        self.parentID = parentID
    
    def InfectCell(self, newID, simTime, newInfType):
        return HCVHepatocyte(newID, self.virusID, 'Virus', simTime, newInfType)
                             
 
                                               
    
time = 0;        
cell1 = HCVHepatocyte(1, None, 'Virus', time, 'Latent')       

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

#Number of state transitions to observe
M = int(1e7)
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
changes = 13;
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
Q[12] = kappa*phi_I

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
trans_vecs[5,12] = -1;
trans_vecs[3,12] = 1;


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
InfectedDict = {}
for i in range(0,int(init/2)):
    x = HCVHepatocyte(i, None, 'Initial', -1, 'Infected', 0)
    InfectedDict[i] = x

for i in range(int(init/2),init):
    x = HCVHepatocyte(i, None, 'Initial', -83, 'InfectedL', 0)
    InfectedDict[i] = x
    
LatentDict = {}
LatentLDict = {}
DeadDict = {}
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
    Qij = Q*[T[j],T[j],T[j], T[j],T[j], E[j],E[j], Ex[j], Ex[j], I[j], Dt[j], Di[j], Di[j]]
    #Draw from exponential distributions of waiting times
    time_vec = -np.log(np.random.random(changes))/Qij
    #np.random.exponential([1/Qij])[0]
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
        Infector = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
        newCellID = lastCellID + 1
        lastCellID = newCellID
        newLatent = CreateLatent(Infector, newCellID, state_idx, time[j])
        if state_idx in [0,1]:
            LatentDict[newCellID] = newLatent
        elif state_idx in [2,3]:
            LatentLDict[newCellID] = newLatent
        else:
            print('Incorrect State')
    
    #Latent cell becomes infectious        
    elif state_idx in [5,7]:
        if state_idx == 5:
            LatCell = LatentDict[random.choice(list(LatentDict.keys()))]
            del LatentDict[LatCell.cellID] #remove cell from Latent Dict
        elif state_idx == 7:
            LatCell = LatentLDict[random.choice(list(LatentLDict.keys()))]
            del LatentLDict[LatCell.cellID] 
        else:
            print('Incorrect State')
        InfectedDict[LatCell.cellID] = LatentInfectious(LatCell, time[j]) #add cell to Infected Dict

    #Latent cell dies    
    elif state_idx == 6: 
        del LatentDict[random.choice(list(LatentDict.keys()))]
        
    #LatentL cell dies
    elif state_idx == 8: 
        del LatentLDict[random.choice(list(LatentLDict.keys()))]
    
    #Infected cell dies
    elif state_idx == 9:
        KilledCell = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
        del InfectedDict[KilledCell.cellID]
        KilledCell.cellType = 'Dead'
        KilledCell.tDead = time[j]
        #newDead = KillInfected(KilledCell,time[j])
        #DeadDict[newDead.cellID] = newDead
        DeadDict[KilledCell.cellID] = KilledCell
        
    #Dead infected cell regenerates into health cell -- just delete from dead dict
    elif state_idx == 11:
        del DeadDict[random.choice(list(DeadDict.keys()))]
        
    #Infected cell regenerated from Dead cell    
    elif state_idx == 12: 
        newCellID = lastCellID + 1
        lastCellID = newCellID
        DeadGen = DeadDict[random.choice(list(DeadDict.keys()))]
        del DeadDict[DeadGen.cellID]
        newInfected = HCVHepatocyte(newCellID,DeadGen.cellID,'DeadGen', DeadGen.tDead, 'Infected', time[j])
        InfectedDict[newInfected.cellID] = newInfected    
   
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
         InfectionChain[int(time[j])].append([Infector.cellID, newCellID])
    elif state_idx == 12:
        #if int(time[j]) in InfectionChain:
         #   InfectionChain[int(time[j])].append([DeadGen.cellID, newInfected.cellID])
        #else:
         #   InfectionChain[int(time[j])] = [DeadGen.cellID, newInfected.cellID]
        InfectionChain[int(time[j])].append([DeadGen.cellID, newInfected.cellID])
    #else:
     #   InfectionChain.append([])
                               
    #Infecteds.append(int([time[j]),list(InfectedDict.keys())])
    #if int(time[j]) in Infecteds:
    
    Infecteds[int(time[j])] = list(set(Infecteds[int(time[j])] + InfectedDict.keys() +LatentDict.keys() +LatentLDict.keys()))
    
        
    #update viral load and ALT    
    VL[j+1] = np.floor(rho*N_liver*(I[j+1]/N)*R/(gamma*c))  #VL[j] + (I[j]/N)*rho*N_liver*newTime - c*gamma*VL[j]*newTime #
    ALT[j+1] = ALT[j] + (eps*(nu_T*(T[j] + E[j] + Ex[j]) + nu_I*I[j])-delta*ALT[j])*newTime
    time[j+1] = time[j] + newTime
    j+=1

# <codecell>

#Sort Infecteds and Infection chain, and break up infection chain
InfectedsSort = dict()
for i in Infecteds.keys():
    InfectedsSort[i] = sorted(Infecteds[i])
InfectionChainSort = {}
for i in InfectionChain.keys():
    a = sorted(list(InfectionChain[i]), key=lambda x: x[0])
    InfectionChainSort[i] = [b for c in a for b in c]

# <codecell>

#Sort Infecteds and Infection chain, and break up infection chain
InfectedsSort = dict()
for key, item in enumerate(Infecteds):
    InfectedsSort[key] = sorted(item)
InfectionChainSort = dict()
for key, item in enumerate(InfectionChain):
    a = sorted(list(item), key=lambda x: x[0])
    InfectionChainSort[key] = [b for c in a for b in c]

# <codecell>


import csv
f = open('Infecteds1e7.txt', 'w') 
writer = csv.writer(f, delimiter = ' ')
for key, value in InfectedsSort.iteritems():
    writer.writerow([key] + value)
f = open('InfectionChain1e7.txt', 'w') 
writer = csv.writer(f, delimiter = ' ')
for key, value in InfectionChainSort.iteritems():
    writer.writerow([key] + value)
    

# <codecell>

f = open('Infecteds.txt', 'w') 
writer = csv.writer(f, delimiter = '\t')
for key, value in Infecteds.iteritems():
    writer.writerow([key] + [value])

# <codecell>

len(InfectionChainSort)

# <codecell>

InfectionChainSort[10]

# <codecell>

InfectionChain[10]

# <codecell>

plt.plot(time,T, label = 'Susc')
plt.plot(time,I, label = 'Infected')
plt.plot(time,Dt, label = 'Dead (healthy)')
plt.plot(time,Di, label = 'Dead (infected)')
plt.legend(loc = 'upper right')

# <codecell>

plt.plot(time,VL)

# <codecell>

random.choice(list(InfectedDict.keys()))
InfectedDict[8].cellType

# <codecell>

plt.plot(time,T, label = 'Susceptible')
plt.plot(time,I+Di, label = 'Ever Infected')
plt.legend(loc = 'upper right')

# <codecell>

HepatocyteDict = {}
for i in range(init):
    x = HCVHepatocyte(i, None, 'Initial', -1, 'Infected', 0)
    HepatocyteDict[i] = x

# <codecell>

InfectedDict = {}
for i in range(0,int(init/2)-2):
    x = HCVHepatocyte(i, None, 'Initial', -1, 'Infected', 0)
    InfectedDict[i] = x

for i in range(int(init/2)-1,init-1):
    x = HCVHepatocyte(i, None, 'Initial', -83, 'InfectedL', 0)
    InfectedDict[i] = x

# <codecell>

InfectedDict[53].cellType

# <codecell>

#Create Module for infection functions

#Build infected cell class
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
        
class HCVVirion:
    def __init__(self, virusID, parentID):
        self.virusID = virusID
        self.parentID = parentID
    
    def InfectCell(self, newID, simTime, newInfType):
        return HCVHepatocyte(newID, self.virusID, 'Virus', simTime, newInfType)
                             
 
                                               
    
time = 0;        
cell1 = HCVHepatocyte(1, None, 'Virus', time, 'Latent')       

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

    state_idx = 0
    time = np.zeros(1e3)
    j=1
    time[j] = 1
    Infector = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
    newCellID = lastCellID + 1
    lastCellID = newCellID
    newLatent = CreateLatent(Infector, newCellID, state_idx, time[j])
    if state_idx ==0:
        LatentDict[newCellID] = newLatent
    elif state_idx == 2:
        LatentLDict[newCellID] = newLatent
    else:
        print('Incorrect State')

# <codecell>

#Try numba
from numba import double
from numba.decorators import jit, autojit
import timeit

from __future__ import division
import numpy as np
import random
X = np.random.random((1000, 3))
D = np.empty((1000, 1000))

def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
%timeit pairwise_python(X)

# <codecell>


# <codecell>

@autojit
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
        
class HCVVirion:
    def __init__(self, virusID, parentID):
        self.virusID = virusID
        self.parentID = parentID
    
    def InfectCell(self, newID, simTime, newInfType):
        return HCVHepatocyte(newID, self.virusID, 'Virus', simTime, newInfType)
                             
 
                                               
    
time = 0;        
cell1 = HCVHepatocyte(1, None, 'Virus', time, 'Latent')       

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
CreateLatentNumba = autojit(CreateLatent)
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
KillInfected = autojit(KillInfected)
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
changes = 13;
delta = .33 #ALT degradation rate
N=N_liver/1e6 #initial number of hepatocytes
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
Q[12] = kappa*phi_I

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
trans_vecs[5,12] = -1;
trans_vecs[3,12] = 1;


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
InfectionChain = dict()
Infecteds = dict()
#Initialize Infected Hepatocyte objects
InfectedDict = {}
for i in range(0,int(init/2)):
    x = HCVHepatocyte(i, None, 'Initial', -1, 'Infected', 0)
    InfectedDict[i] = x

for i in range(int(init/2),init):
    x = HCVHepatocyte(i, None, 'Initial', -83, 'InfectedL', 0)
    InfectedDict[i] = x
    
LatentDict = {}
LatentLDict = {}
DeadDict = {}
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
    Qij = Q*[T[j],T[j],T[j], T[j],T[j], E[j],E[j], Ex[j], Ex[j], I[j], Dt[j], Di[j], Di[j]]
    #Draw from exponential distributions of waiting times
    time_vec = -np.log(np.random.random(changes))/Qij
    #np.random.exponential([1/Qij])[0]
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
        Infector = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
        newCellID = lastCellID + 1
        lastCellID = newCellID
        newLatent = CreateLatentNumba(Infector, newCellID, state_idx, time[j])
        if state_idx in [0,1]:
            LatentDict[newCellID] = newLatent
        elif state_idx in [2,3]:
            LatentLDict[newCellID] = newLatent
        else:
            print('Incorrect State')
    
    #Latent cell becomes infectious        
    elif state_idx in [5,7]:
        if state_idx == 5:
            LatCell = LatentDict[random.choice(list(LatentDict.keys()))]
            del LatentDict[LatCell.cellID] #remove cell from Latent Dict
        elif state_idx == 7:
            LatCell = LatentLDict[random.choice(list(LatentLDict.keys()))]
            del LatentLDict[LatCell.cellID] 
        else:
            print('Incorrect State')
        InfectedDict[LatCell.cellID] = LatentInfectious(LatCell, time[j]) #add cell to Infected Dict

    #Latent cell dies    
    elif state_idx == 6: 
        del LatentDict[random.choice(list(LatentDict.keys()))]
        
    #LatentL cell dies
    elif state_idx == 8: 
        del LatentLDict[random.choice(list(LatentLDict.keys()))]
    
    #Infected cell dies
    elif state_idx == 9:
        KilledCell = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
        del InfectedDict[KilledCell.cellID]
        KilledCell.cellType = 'Dead'
        KilledCell.tDead = time[j]
        #newDead = KillInfected(KilledCell,time[j])
        #DeadDict[newDead.cellID] = newDead
        DeadDict[KilledCell.cellID] = KilledCell
        
    #Dead infected cell regenerates into health cell -- just delete from dead dict
    elif state_idx == 11:
        del DeadDict[random.choice(list(DeadDict.keys()))]
        
    #Infected cell regenerated from Dead cell    
    elif state_idx == 12: 
        newCellID = lastCellID + 1
        lastCellID = newCellID
        DeadGen = DeadDict[random.choice(list(DeadDict.keys()))]
        del DeadDict[DeadGen.cellID]
        newInfected = HCVHepatocyte(newCellID,DeadGen.cellID,'DeadGen', DeadGen.tDead, 'Infected', time[j])
        InfectedDict[newInfected.cellID] = newInfected    
   
    #Output Infection chain and infecteds at each time step
    
    #add to array of infections with timestep
    if state_idx in [0,1,2,3]:  
        if int(time[j]) in InfectionChain:
            InfectionChain[int(time[j])].append([Infector.cellID, newCellID])
        else:
            InfectionChain[int(time[j])] = [[Infector.cellID, newCellID]]
    elif state_idx == 12:
        if int(time[j]) in InfectionChain:
            InfectionChain[int(time[j])].append([DeadGen.cellID, newInfected.cellID])
        else:
            InfectionChain[int(time[j])] = [DeadGen.cellID, newInfected.cellID]
    else:
        if int(time[j]) not in InfectionChain:
            InfectionChain[int(time[j])] = []
                               
    #Infecteds.append(int([time[j]),list(InfectedDict.keys())])
    if int(time[j]) in Infecteds:
        Infecteds[int(time[j])] = list(set(Infecteds[int(time[j])] + InfectedDict.keys() +LatentDict.keys() +LatentLDict.keys()))
    else:
        Infecteds[int(time[j])] = InfectedDict.keys() +LatentDict.keys() +LatentLDict.keys()
        
    #update viral load and ALT    
    VL[j+1] = np.floor(rho*N_liver*(I[j+1]/N)*R/(gamma*c))  #VL[j] + (I[j]/N)*rho*N_liver*newTime - c*gamma*VL[j]*newTime #
    ALT[j+1] = ALT[j] + (eps*(nu_T*(T[j] + E[j] + Ex[j]) + nu_I*I[j])-delta*ALT[j])*newTime
    time[j+1] = time[j] + newTime
    j+=1

# <codecell>

 #write out function patterns for each state transition
if state_idx in [0,1,2,3]: #Infection of healthy cell by cell or virus -> latent or longterm latent
    Infector = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
    newCellID = lastCellID + 1
    lastCellID = newCellID
    newLatent = CreateLatent(Infector, newCellID, state_idx, time[j])
    if state_idx ==0:
        LatentDict[newCellID] = newLatent
    elif state_idx == 2:
        LatentLDict[newCellID] = newLatent
    else:
        print('Incorrect State')
elif state_idx in [6,8]: #Latent cell becomes infectious
    if state_idx == 6:
        LatCell = LatentDict[random.choice(list(LatentDict.keys()))]
        del LatentDict[LatCell.cellID] #remove cell from Latent Dict
    elif state_idx == 8:
        LatCell = LatentLDict[random.choice(list(LatentLDict.keys()))]
        del LatentDict[LatCell.cellID] 
    else:
        print('Incorrect State')
    InfectedDict[LatCell.cellID] = LatentInfectious(LatCell, time[j]) #add cell to Infected Dict
elif state_idx == 7: #Latent cell dies
    del LatentDict[random.choice(list(LatentDict.keys()))]
elif state_idx == 8: #LatentL cell dies
    del LatentLDict[random.choice(list(LatentLDict.keys()))]
elif state_idx == 10:
    KilledCell = InfectedDict[random.choice(list(InfectedDict.keys()))] #choose random infector cell
    newDead = KillInfected(KilledCell,time[j])
    DeadDict[newDead.cellID] = newDead
elif state_idx == 13: #Infected cell regenerated from Dead cell
    newCellID = lastCellID + 1
    lastCellID = newCellID
    DeadGen = DeadDict[random.choice(list(InfectedDict.keys()))]
    newInfected = HCVHepatocyte(newCellID,DeadGen.cellID,'DeadGen', DeadGen.tDead, 'Infected', time[j])
    InfectedDict[newInfected.cellID] = newInfected
    
        

# <codecell>

 ##########       
 elif state_idx in [1,3]: #Infection of healthy cell by virus -> latent or longterm latent
    InfectorID = random.choice(virionList[j])
    Infector = InfectedDict[InfectorID]
    newCellID +=lastCellID
    newLatent = CreateLatent(Infector, newID, state_idx, newTime)
    if state_idx ==0:
        LatentDict[newCellID] = newLatent
    elif state_idx == 2:
        LatentLDict[newCellID] = newLatent
    else:
        print('Incorrect State')    
    
    #Create virion objects from infected cells
def GenerateVirions(cellDict, rho,R,gamma,c, N, N_liver): #lastVirusID,
    #newVirusID = lastVirusID #start ID count
    virionList = [] #initialize virion list
    nVirions = int(np.floor(rho*(N_liver/N)*R/(gamma*c)))
    for idx in cellDict.keys():
        newVirions = np.empty(nVirions)
        newVirions = newVirions.fill(cellDict[idx].cellID)
        virionList.extend(newVirions)
        #for i in range(nVirions):
         #   newVirion = [newVirusID,cellDict[idx].cellID]
          #  virionList.append(newVirion)
           # newVirusID += 1
    return virionList, newVirusID

# <markdowncell>

# Incorporating lineage dynamics:
# 
# create class InfectedCell
# 
# create class LatentCell
# 
# create class LongTermCell
# 
# if transition is infected cell, add new cell to latent class, pick one infected cell randomly and take its sequence, change it by one random step
# 
# if transition is latent cell becomes infectious, randomly choose latent cell and move it to infectious list
# 
# keep a snapshot of what sequences are around at each timestep
# 
# keep an id for each cell
# 
# Latent cell array
# 
# add in latent cells at each time step
# 
# 
# Infected cell class attributes:
# 
# id
# 
# parent id
# 
# virus or cell infection
# 
# time infectious
# 
# time infected
# 
# longterm
# 
# Latent cell class attributes
# 
# id
# 
# parent id
# 
# time infected
# 
# virus or cell infection
# 
# longterm
# 
# 
# virion class attributes
# 
# id
# 
# parent id
# 
# lists of infected cells and latent cells at each timestep
# 
# 
# pseudo code
# 
# create an array of latent cells 
# 
# create array of infected cells
# 
# create list of infected cell ids
# create list of latent cell ids
# create list of longterm cell ids
# 
# 
# export timestep and infections: which cell(s) infected which on each day

# <codecell>

cell2 = HCVHepatocyte(2, None, 'Virus', time, 'Infected', time+1) 

# <codecell>

newID = 3
newLatent = CreateLatent(cell2, newID, 0, time)

# <codecell>

xlist= []
xlist.append(1)

# <codecell>

np.floor((rho*(N_liver/N)*R/(gamma*c)))

# <codecell>

del cell2

# <codecell>

KillInfected(cell2,time)

# <codecell>

cell2.tDead

# <codecell>

cell2

# <codecell>


