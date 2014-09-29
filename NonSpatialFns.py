#This file contains all the functions for the NonSpatial version of the model
from __future__ import division
import csv
import numpy as np
import random
import pickle
import datetime
import zipfile
import zlib
import os

def GenTransitionProbs(changes, eta, beta_L, beta_V, nu_T, alpha, alpha_x, nu_I, phi_T, phi_I, iCells, vLoad, state_vec):
    ''' Generate matrix of transition probabilities given parameters and state values'''
    Q = np.zeros(changes)
    Q[0] = (1-eta)*(beta_L*iCells) #Infection of Target cell by cell-> latent
    Q[1] = (1-eta)*beta_V*vLoad #Infection of Target cell by virus -> latent
    Q[2] = eta*beta_L*iCells  #Infection of Target cell by cell -> long-term latent
    Q[3] = eta*beta_V*vLoad #Infection of Target cell by virus -> long-term latent
    Q[4] = nu_T; #Death of target cell
    
    Q[5] = alpha; #latent cell becomes infected
    Q[6] = nu_T; #latent cell dies
    
    Q[7] = alpha_x #long-term latent cell becomes infected
    Q[8] = nu_T #long-term latent cell dies
    
    Q[9] = nu_I; #Infected cell dies
    
    Q[10] = phi_T; #Healthy cell regenerates
    
    Q[11] = phi_I; #Infected cell regenerates into healthy cell
    
    Qij = Q*state_vec
    
    return Qij

def GenTransitionProbsTwoPatch(changes, eta, beta_L, beta_V, nu_T, alpha, alpha_x, nu_I, phi_T, phi_I, nu_T2, beta_L2, m12, m21, iCells, vLoad, iCells2, vLoad2,state_vec):
    ''' Generate matrix of transition probabilities given parameters and state values'''
    Q = np.zeros(changes)
    Q[0] = (1-eta)*(beta_L*iCells) #Infection of Target cell by cell-> latent
    Q[1] = (1-eta)*beta_V*vLoad #Infection of Target cell by virus -> latent
    Q[2] = eta*beta_L*iCells  #Infection of Target cell by cell -> long-term latent
    Q[3] = eta*beta_V*vLoad #Infection of Target cell by virus -> long-term latent
    Q[4] = nu_T; #Death of target cell
    
    Q[5] = alpha; #latent cell becomes infected
    Q[6] = nu_T; #latent cell dies
    
    Q[7] = alpha_x #long-term latent cell becomes infected
    Q[8] = nu_T #long-term latent cell dies
    
    Q[9] = nu_I; #Infected cell dies
    
    Q[10] = phi_T; #Healthy cell regenerates
    
    Q[11] = phi_I; #Infected cell regenerates into healthy cell

    Q[12] = (1-eta)*beta_V*vLoad2*m21 #Target cell 1 infected by virus from patch 2 -> latent
    Q[13] = eta*beta_V*vLoad2*m21 #Target cell 1 infected by virus from patch 2 -> long term latent

    Q[14] = beta_L2*iCells2  #Infection of target cell 2 by cell
    Q[15] = beta_V*vLoad2 #Infection of target cell type 2 by virus of type 2
    Q[16] = beta_V*m12*vLoad #Infection of target cell type 2 by virus type 1
    Q[17] = nu_T2 #generation of new target cell 2
    Q[18] = nu_T2 #death of target cell 2
    Q[19] = nu_T2 #death of infected target cell 2
    
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

def UpdateInfectionListsTwoPatch(state_idx, InfectedList, Infected2List, LatentList, LatentXList, lastCellID):
    '''Susceptible cell becomes infected: update Latent Lists''' 
    if state_idx in [0,1,2,3,16]:
        Infector = random.choice(InfectedList) #choose random infector cell
    elif state_idx in [12,13,14,15]:
        Infector = random.choice(Infected2List) #choose random infector cell
    else:
         print('Incorrect State for Infecting Function') 
    newCellID = abs(lastCellID) + 1
    lastCellID = newCellID
    if state_idx in [0,1,12]:
        LatentList.append(lastCellID)
    elif state_idx in [2,3,13]:
        LatentXList.append(lastCellID)
    elif state_idx in [14,15,16]:
        lastCellID = -lastCellID
        Infected2List.append(lastCellID)
    else:
        print('Incorrect State for Infecting Function')
    return InfectedList, Infected2List, LatentList, LatentXList, Infector, lastCellID

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

def UpdateKillCellGen(state_idx, cellList):  
    '''Latent, LatentX, or Infected Cell Dies'''
    if state_idx in [6,8,9,19]:
        cellList.remove(random.choice(cellList)) 
    else:
        print("State not a cell-death state")
    return cellList

def UpdateInfectionChain(InfectionChain, Infector, lastCellID, time, minKey):
    '''Update Transmission Chain list'''
    if len(InfectionChain)< int(time)+1-minKey:
        InfectionChain.append([])
    InfectionChain[int(time) -minKey].append([Infector, lastCellID])
    return InfectionChain
    
    
def UpdateInfecteds(Infecteds, InfectedList, LatentList, LatentXList, time, minKey):
    '''Update Infecteds List'''
    if len(Infecteds) < int(time)+1-minKey:
        Infecteds.append([])
    InfectedIDs = InfectedList+ LatentList+LatentXList
    Infecteds[int(time) -minKey] = list(set(Infecteds[int(time) -minKey] + InfectedIDs))
    return Infecteds

def UpdateInfectedsTwoPatch(Infecteds, InfectedList, Infected2List, LatentList, LatentXList, time, minKey):
    '''Update Infecteds List'''
    if len(Infecteds) < int(time)+1-minKey:
        Infecteds.append([])
    InfectedIDs = InfectedList+ LatentList+LatentXList+Infected2List
    Infecteds[int(time) - minKey] = list(set(Infecteds[int(time) -minKey] + InfectedIDs))
    return Infecteds

def UpdateVL(rho, N_liver, N, R, gamma, c, iCells):
    VL = np.floor(rho*N_liver*(iCells/N)*R/(gamma*c))
    return VL

def UpdateALT(eps, nu_T, nu_I, delta, oldALT, T, E, Ex, I, tStep):    
    newALT = (eps*(nu_T*(T + E+ Ex) + nu_I*I)-delta*oldALT)*tStep
    return newALT

def OutputPrevFile(Infecteds, filename = None):
    #First, sort Infecteds so they are in order of cell ID
    InfectedsSort = dict()
    if filename == None:
        now = datetime.datetime.now()
        filename = 'Infecteds_'+ now.strftime("%I%M%S%B%d")
    zipfilename = filename + '.zip'
    for key1, item in enumerate(Infecteds):
        item.sort(key = lambda x: abs(int(x)))
        InfectedsSort[key1] = item
    f = open(filename, 'w') 
    writer = csv.writer(f, delimiter = ' ')
    for key, value in InfectedsSort.iteritems():
        writer.writerow([key] + value)
    f.close()
    z = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED,allowZip64=True)
    z.write(filename)
    z.close()
    os.remove(filename)

def OutputChainFile(InfectionChain, filename = None):
    #Sort Infecteds and Infection chain, and break up infection chain
    InfectionChainSort = dict()
    if filename == None:
        now = datetime.datetime.now()
        filename = 'InfectionChain_'+ now.strftime("%I%M%S%B%d")
    zipfilename = filename + '.zip'
    for key1, item in enumerate(InfectionChain):
        a = sorted(list(item), key=lambda x: abs(x[0]))
        InfectionChainSort[key1] = [b for c in a for b in c]
    f = open(filename, 'w') 
    writer = csv.writer(f, delimiter = ' ')
    for key, value in InfectionChainSort.iteritems():
        writer.writerow([key] + value)
    f.close()
    z = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
    z.write(filename)
    z.close()
    os.remove(filename)


def saveWorkspace(kwargs, filename = None ):
    pickleDict = dict()
    for key, item in kwargs.iteritems():
        pickleDict[key] = item
    #picklelist = [T, E, Ex, I, Dt, Di, time, VL, ALT, Infecteds, InfectionChain]
    if filename == None:
        now = datetime.datetime.now()
        filename = 'Workspace_'+ now.strftime("%I%M%S%B%d")
    zipfilename = filename + '.zip'
    tempFile = open(filename, 'w')
    # now let's pickle picklelist
    pickle.dump(pickleDict,tempFile)
    # close the file, and your pickling is complete
    tempFile.close()
    z = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
    z.write(filename)
    z.close()
    os.remove(filename)
    return zipfilename

def loadWorkspace(filename):
    unpicklefile = open(filename, 'r')
    # now load the list that we pickled into a new object
    unpickledDict = pickle.load(unpicklefile)
    # close the file, just for safety
    unpicklefile.close()
    return unpickledDict

def OutputTempFiles(Infecteds, InfectionChain, minKey, OutPrevFileName, OutChainFileName, finalFile):
    OutputPrevFileMod(Infecteds, minKey, OutPrevFileName, finalFile)
    OutputChainFileMod(InfectionChain, minKey, OutChainFileName, finalFile)
    # if len(Infecteds) ==1:
    #     minKey = minKey
    # else:
    #     minKey = minKey + len(Infecteds)
    InfectionChain = []#[InfectionChain[-1]]
    Infecteds = []#[Infecteds[-1]]
    minKey +=1
    return Infecteds, InfectionChain , minKey


def OutputPrevFileMod(Infecteds, minKey, filename = None, finalFile=False):
    #First, sort Infecteds so they are in order of cell ID
    InfectedsSort = dict()
    if filename == None:
        now = datetime.datetime.now()
        filename = 'Infecteds_'+ now.strftime("%I%M%S%B%d")
    zipfilename = filename + '.zip'
    for key1, item in enumerate(Infecteds):
        item.sort(key = lambda x: abs(int(x)))
        InfectedsSort[key1] = item
    f = open(filename, 'ar') 
    writer = csv.writer(f, delimiter = ' ')
    #reader = csv.reader(f)
    for key, value in InfectedsSort.iteritems():
        # print "minKey is", minKey
        # print "key is", key
        # print "minKey plus key is", minKey + key
        writer.writerow([key+minKey] + value)
    f.close()
    if finalFile:
        z = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED,allowZip64=True)
        z.write(filename)
        z.close()
        os.remove(filename)

def OutputChainFileMod(InfectionChain, minKey, filename = None, finalFile=False):
    #Sort Infecteds and Infection chain, and break up infection chain
    InfectionChainSort = dict()
    if filename == None:
        now = datetime.datetime.now()
        filename = 'InfectionChain_'+ now.strftime("%I%M%S%B%d")
    zipfilename = filename + '.zip'
    for key1, item in enumerate(InfectionChain):
        a = sorted(list(item), key=lambda x: abs(x[0]))
        InfectionChainSort[key1] = [b for c in a for b in c]
    f = open(filename, 'a') 
    writer = csv.writer(f, delimiter = ' ')
    for key, value in InfectionChainSort.iteritems():
        writer.writerow([key+minKey] + value)
    f.close()
    if finalFile:
        z = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        z.write(filename)
        z.close()
        os.remove(filename)
    