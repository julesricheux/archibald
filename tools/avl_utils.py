# -*- coding: utf-8 -*-
"""
Created 04/07/2023
Last update: 24/12/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -
"""

#%% DEPENDENCIES

import os
import subprocess
import numpy as np

from tools.math_utils import *

from lifting_planes import *


#%% FUNCTIONS

def write_avl_rig_geometry(objectSet,
                           awa=45., beta=None,
                           IYsym=0, IZsym=0, Zsym=0.):
        
    if beta is None:
        beta = np.ones(objectSet.nSails) * (awa - 15)
    elif type(beta)==int or type(beta)==float:
        beta = np.ones(objectSet.nSails) * beta
    
    if os.path.exists(objectSet._AVLgeo):
        os.remove(objectSet._AVLgeo)
    
    f = open(objectSet._AVLgeo, 'w')
    
    Sref = objectSet.area
    Bref = np.max([sail.Bref for sail in objectSet.sails.values() if type(sail)!=Spi])
    Cref = Sref / Bref
    
    rotation = rotation_matrix(np.array([0,1,0]), -awa)
    
    f.write("{0:}\n".format(objectSet.name))
    f.write("#Mach\n0.0\n")
    f.write("#IYsym IZsym Zsym\n")
    f.write("{0:} {1:} {2:}\n".format(IYsym, IZsym, Zsym))
    f.write("#Sref   Cref    Bref\n")
    f.write("{0:.2f}  {1:.2f}  {2:.2f}\n".format(Sref, Cref, Bref))
    f.write("#Xref   Yref    Zref\n0.0     0.0     0.0\n\n")

    for i, sail in enumerate(objectSet.sails.values()):
        
        if type(sail) != Spi:
            
            avlLeadingEdge = np.transpose(np.vstack([-sail.le[:,0], sail.le[:,2], sail.le[:,1]]))
            rotLeadingEdge = np.dot(avlLeadingEdge, rotation)
            
            xle = rotLeadingEdge[:,0]
            yle = rotLeadingEdge[:,1]
            zle = rotLeadingEdge[:,2]
            
            f.write("#===================================================================\n")
            f.write("SURFACE\n")
            f.write("{0:}\n".format(sail.name))
            
            f.write("#Nchordwise Cspace Nspanwise Sspace\n")
            f.write("{0} 0 {1} 0\n\n".format(sail.nChordwise, sail.nSpanwise))
            
            f.write("ANGLE\n{0}\n".format(awa-beta[i]))
            
            f.write("#===================================================================\n")
            f.write("#Xle Yle Zle Chord Ainc Nspanwise Sspace\n\n")
            
            for i in range(sail.nSections):
                f.write("SECTION\n")
                f.write("{0} {1} {2} {3} {4} 1 0\n".format(xle[i], yle[i], zle[i], sail.chords[i], -sail.twists[i]))
                f.write("AFILE\n")
                f.write("{0}\n".format(sail.sections[i]))
            f.write("#===================================================================\n")
    
    f.close()
        
    
def write_avl_hull_geometry(objectSet,
                            IYsym=1, IZsym=0, Zsym=0.):
    
    Sref = objectSet.area
    Bref = np.max([appendage.Bref for appendage in objectSet.appendages.values()])
    Cref = Sref / Bref
    
    if os.path.exists(objectSet._AVLgeo):
        os.remove(objectSet._AVLgeo)

    f = open(objectSet._AVLgeo, 'w')

    f.write("{0:}\n".format(objectSet.name))
    f.write("#Mach\n0.0\n")
    f.write("#IYsym IZsym Zsym\n")
    f.write("{0:} {1:} {2:}\n".format(IYsym, IZsym, Zsym))
    f.write("#Sref   Cref    Bref\n")
    f.write("{0:.2f}  {1:.2f}  {2:.2f}\n".format(Sref, Cref, Bref))
    f.write("#Xref   Yref    Zref\n0.0     0.0     0.0\n\n")

    for appendage in objectSet.appendages.values():

        if type(appendage) == Rudder:
            rotation = rotation_matrix(appendage.shaft, appendage.angle)
            rotLeadingEdge = np.dot(appendage.le - appendage.shaftRoot, rotation) + appendage.shaftRoot
            avlLeadingEdge = np.transpose(np.vstack([-rotLeadingEdge[:,0], -rotLeadingEdge[:,2], rotLeadingEdge[:,1]]))
            angle = appendage.angle
            
        else:
            avlLeadingEdge = np.transpose(np.vstack([-appendage.le[:,0], appendage.le[:,2], appendage.le[:,1]]))
            angle = 0.0

        xle = avlLeadingEdge[:,0]
        yle = avlLeadingEdge[:,1]
        zle = avlLeadingEdge[:,2]

        f.write("#===================================================================\n")
        f.write("SURFACE\n")
        f.write("{0:}\n".format(appendage.name))
        
        f.write("#Nchordwise Cspace Nspanwise Sspace\n")
        f.write("{0} 0 {1} 0\n\n".format(appendage.nChordwise, appendage.nSpanwise))
        
        f.write("ANGLE\n{0}\n".format(angle))
        
        f.write("#===================================================================\n")
        f.write("#Xle Yle Zle Chord Ainc Nspanwise Sspace\n\n")
        
        for i in range(appendage.nSections):
            f.write("SECTION\n")
            f.write("{0} {1} {2} {3} {4} 1 0\n".format(xle[i], yle[i], zle[i], appendage.chords[i], -appendage.twists[i]))
            f.write("AFILE\n")
            f.write("{0}\n".format(appendage.sections[i]))
        f.write("#===================================================================\n")
    
    f.close()


def write_avl_input(objectSet, aoa=0.):
    
    if os.path.exists(objectSet._AVLin):
        os.remove(objectSet._AVLin)
    
    f = open(objectSet._AVLin, 'w')
    
    f.write("PLOP\n")
    f.write("G 0\n\n")
    
    f.write("LOAD {0}\n".format(objectSet._AVLgeo))
    
    f.write("OPER\n")
    f.write("A\n")
    f.write("A {0}\n".format(aoa))
    f.write("X\n")
    
    f.write("FS {0}\n".format(objectSet._AVLout))
    f.write("O\n")
    
    f.write("\n\n")
    
    f.write("quit\n")
    f.close()
    

def run_avl_analysis(objectSet):
    
    if os.path.exists(objectSet._AVLin):
        subprocess.call("avl.exe < "+objectSet._AVLin, shell=True)
    else:
        print('No AVL input file')     
    
    
def read_avl_rig_results(objectSet, awa):
    
    objectSet._globalData.clear()
    objectSet._localData.clear()
    
    skip = 0
    
    for i, sail in enumerate(objectSet.sails.values()):
        
        sail = objectSet.sails[i]
        
        if type(sail) == Spi:
            if objectSet.spiIsRaised:
                objectSet._globalData.append([sail.get_CL(awa), sail.get_CD(awa)])
            else:
                objectSet._globalData.append([0.0, 0.0])
                
            objectSet._localData.append(np.array([]))
            skip += 1
        
        else:
            _globalSkip = 18 + 17*i + sum([objectSet.sails[s].nSpanwise for s in range(i-skip)])
            _localSkip = 22 + 17*i + sum([objectSet.sails[s].nSpanwise for s in range(i-skip)])
            _localRows = sail.nSpanwise
            
            objectSet._globalData.append(list(np.loadtxt(objectSet._AVLout, skiprows=_globalSkip, max_rows=1, usecols=[2, 5], dtype='float')))
            objectSet._localData.append(np.loadtxt(objectSet._AVLout, skiprows=_localSkip, max_rows=_localRows, usecols=[1, 2, 3, 4, 5, 7, 9, 10, 11, 14]))
      

def read_avl_hull_results(objectSet):
    
    objectSet._globalData = list() 
    objectSet._localData = list()
    
    for i in range(objectSet.nAppendages):
        _globalSkip = 18 + 17*i + sum([objectSet.appendages[s].nSpanwise for s in range(i)])
        _localSkip = 22 + 17*i + sum([objectSet.appendages[s].nSpanwise for s in range(i)])
        _localRows = objectSet.appendages[i].nSpanwise
        
        objectSet._globalData.append(list(np.loadtxt(objectSet._AVLout, skiprows=_globalSkip, max_rows=1, usecols=[2, 5], dtype='float')))
        objectSet._localData.append(np.loadtxt(objectSet._AVLout, skiprows=_localSkip, max_rows=_localRows, usecols=[1, 2, 3, 4, 5, 7, 9, 10, 11, 14]))
        