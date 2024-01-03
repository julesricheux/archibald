# -*- coding: utf-8 -*-
"""
Created 04/07/2023
Last update: 03/01/2024

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -
"""

#%% DEPENDENCIES

import os
import subprocess
import numpy as np


#%% FUNCTIONS

def write_xfoil_geometry(name, x, y):
    
    if os.path.exists(name):
        os.remove(name)

    n = len(x)

    input_file = open(name, 'w')
    
    input_file.write(name+"\n")
    for i in range(n):
        input_file.write("{0} {1}\n".format(x[i], y[i]))

    input_file.close()
    
    
def write_xfoil_input(objectSet, profile, alpha, Re, nIter=200, N=200, t=0.5, r=0.2, nCrit=9, xtr_top=0.1, xtr_bot=0.1):

    geometry = profile
    
    if alpha > 20:
        print('Incidence > 20°')
        
        alpha = 20
        
    elif alpha < -15:
        print('Incidence < -15°')
        
        alpha = -15
    
    # Compute additionnal angles for convergence (WIP)
    n_cv = int((abs(alpha)/5))
    alpha_int = alpha * np.power((np.linspace(0,1,n_cv+3)[1:-1]), 3/4)
    
    if os.path.exists(objectSet._XFin):
        os.remove(objectSet._XFin)
    
    f = open(objectSet._XFin, 'w')
    
    f.write("PLOP\n")
    f.write("G 0\n\n")
    
    f.write("LOAD {0}\n".format('.'+geometry[len(objectSet._abs_dir):]))
    
    f.write("PPAR\n")
    f.write("N {0}\n".format(N))
    f.write("t {0}\n".format(t))
    f.write("r {0}\n".format(r))
    f.write("\n\n")
    
    f.write("OPER\n")
    f.write("VISC {0}\n".format(max(1000., Re)))
    
    f.write("VPAR\n")
    f.write("n {0}\n".format(nCrit))
    f.write("xtr\n")
    f.write("{0}\n".format(xtr_top))
    f.write("{0}\n\n".format(xtr_bot))
    
    f.write("ITER {0}\n".format(nIter))
    
    # Incidence convergence
    for a in alpha_int:
        f.write("A {0}\n".format(a))
    
    f.write("PACC\n")
    f.write("{0}\n\n".format('.'+objectSet._XFout[len(objectSet._abs_dir):]))
    
    f.write("A {0}\n".format(alpha))
    
    f.write("PACC\n")
    
    f.write("\n\n")
    
    f.write("quit\n")
    f.close()
    
    
def run_xfoil_analysis(objectSet):
    
    # if os.path.exists(objectSet._XFout):
    #     os.remove(objectSet._XFout)
    
    if os.path.exists(objectSet._XFin):
        cwd = os.getcwd()
        os.chdir(objectSet._abs_dir)
        
        subprocess.run("xfoil.exe < "+objectSet._XFin, shell=True)
        
        os.chdir(cwd)
    else:
        print('No XFOIL input file')


def read_xfoil_results(objectSet):
    
    data = np.loadtxt(objectSet._XFout, skiprows=12)

    # alpha = data[0]
    cd = data[2]
    
    return cd
