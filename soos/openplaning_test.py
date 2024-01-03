# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:50:41 2023

@author: Jules
"""

from openplaning import PlaningBoat

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

speed = 0.514 * 20 #kts to m/s
weight = 9.81 * (120 + 2*75) #kg to N
beam = 1. #m
length = 5.5 #m, vessel LOA
beta = 10.

#Vessel particulars
vcg = .1 #m, vert. center of gravity
r_g = 2. #m, radius of gyration

#Propulsion
epsilon = 0 #deg, thrust angle w.r.t. keel
vT = vcg #m, thrust vertical distance
lT = length/2 #m, thrust horizontal distance

#Trim tab particulars
sigma = 1.0 #flap span-hull beam ratio
delta = 5 #deg, flap deflection
Lf = 0.3048 #m, flap chord

#Seaway
H_sig = 1.402 #m, significant wave height

#%%

speed = 0.514 * 15 #kts to m/s
weight = 9.81 * (120 + 2*75) #kg to N
beam = 1. #m
length = 5 #m, vessel LOA
beta = 5
    
lcg = 1.3 #m, long. center of gravity

#Create boat object
boat = PlaningBoat(speed, weight, beam, lcg, vcg, r_g, beta,
                    epsilon, vT, lT, length,
                    wetted_lengths_type=3,
                    tau = 3.)


#Calculates the equilibrium trim and heave, and updates boat.tau and boat.z_wl
boat.get_steady_trim()

print(round(boat.hydrodynamic_force[0] + boat.skin_friction[0], 1))
print(round(boat.hydrodynamic_force[0], 1))
print(round(boat.hydrodynamic_force[1], 1))

#%%

speed = 0.514 * 15 #kts to m/s
beam = 1.2 #m
weight = 9.81 * (120 + 2*75) #kg to N
weight = 9.81 * (410) #kg to N
length = 5 #m, vessel LOA
beta = 5.5
lcg = .2

trim = 5

x0 = .1

def f(X, fullOutput = False):
    lcg = X[0]
    boat = PlaningBoat(speed, weight, beam, lcg, vcg, r_g, beta,
                        epsilon, vT, lT,
                        wetted_lengths_type=3)
    
    boat.get_steady_trim(tauLims=[0.1, 35])
    
    if fullOutput:
        print(boat.hydrodynamic_force[1])
        
        return boat.net_force[0], boat.hydrodynamic_force[0], boat.skin_friction[0]
    
    return np.abs(boat.net_force[0])

BEAMS = np.arange(0.8, 2.1, 0.1)
RX = []
RDYN = []
RF = []
TRIMS = []

for b in BEAMS:
    beam = b
    xOpt = opt.fmin(f, np.array([x0]))
    
    rx, rdyn, rf = f(xOpt, fullOutput = True)
    RX.append(rx)
    RDYN.append(rdyn)
    RF.append(rf)
    TRIMS.append(xOpt[0])
    

plt.plot(BEAMS, RX)
plt.plot(BEAMS, RDYN, ls='--')
plt.plot(BEAMS, RF)
plt.show()

plt.plot(BEAMS, TRIMS)
plt.show()
    

# boat.print_description()