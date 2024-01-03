# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:57:11 2023

@author: jrich
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

from archibald.tools.math_utils import *
from archibald.tools.dyn_utils import *
from archibald.tools.geom_utils import *

import aerosandbox as asb
import aerosandbox.numpy as asbnp
import matplotlib.pyplot as plt

import numpy as np

from gradient_vlm import GradientVortexLatticeMethod

import time

# mainAirfoil = asb.Airfoil("sd7037")
# jibAirfoil = asb.Airfoil("sd7037")
mainAirfoil = asb.Airfoil("naca6403")
jibAirfoil = asb.Airfoil("naca6403")


# Define rig initial geometry
nSections = 12

mainDxf = 'C:/Users/jrich/Documents/paki_aka/voilure/gv_rig2.dxf'
# jibDxf = 'C:/Users/jrich/Documents/paki_aka/voilure/jib1.dxf'

mainLe, mainChords = dxf_to_le_chords(mainDxf, nSections, method='bezier')

mainLe = np.array([[-1.4,  0. ,  4.4],
                   [-1.4,  0. ,  7.7],
                   [-1.4,  0. , 14. ],
                   [-1.4,  0. , 20.2],
                   [-1.4,  0. , 26.5],
                   [-1.4,  0. , 32.7],
                   [-1.4,  0. , 39. ],
                   [-1.4,  0. , 45.2],
                   [-1.4,  0. , 51.5],
                   [-1.4,  0. , 57.7],
                   [-1.4,  0. , 61.3],
                   [-1.4,  0. , 71.3]])

mainChords = np.array([20.6,
                       20.3,
                       19.8,
                       19.1,
                       18.3,
                       17.4,
                       16.3,
                       14.9,
                       13.4,
                       11.7,
                       10.6,
                       0.7])

# Jibsail rig 2
jibTackPt = np.array([19.6, 0.0,  4.5])
jibHeadPt = np.array([ 3.2, 0.0, 60.0])

jibFoot = 16.4
jibHead = 0.1

jibLe = np.linspace(jibTackPt, jibHeadPt, nSections)
jibChords = np.linspace(jibFoot, jibHead, nSections)


balAxis = np.array([[-22.2, 0, 2.6],
                    [-3.4/2, 0, 1.6],
                    [3.4/2, 0, 1.6],
                    [19.6, 0, 2.8]])

balRadius = np.array([1.4, 3.3, 3.3, 1.1])/2

# Apparent wind
aoa = 20

z0 = 10.
tws0 = 10.
twa0 = 90.
stw = 10.

aws0, awa0 = compute_AW(tws0, twa0, stw)

jibSheeting = 5


#Twisting
mainTwistAngle = -15.0
jibTwistAngle = mainTwistAngle * 6/5
twistPower = 0.912

mainTwist = np.linspace(0,1,nSections)**twistPower * mainTwistAngle
jibTwist = np.linspace(0,1,nSections)**twistPower * jibTwistAngle

solidsailCoords = np.array([[ 36.0, 0.0, 25.0],
                            [ 84.0, 0.0, 25.0],
                            [132.0, 0.0, 25.0]])

# solidsailCoords = np.array([[ 36.0, 0.0, 25.0]])

balSheeting = np.array([ 0.5, 0.8, 1.0]) * (awa0-aoa)
# balSheeting = np.array([ 1.0, 1.0, 1.0]) * (awa-aoa)

nSolidsail = len(solidsailCoords)



mainLe = np.array([mainLe for i in range(nSolidsail)])
jibLe = np.array([jibLe for i in range(nSolidsail)])
balAxis = np.array([balAxis for i in range(nSolidsail)])

rot_mat = rotation_matrix([0,0,1], balSheeting)


for j in range(nSolidsail):
    mainLe[j] = np.dot(mainLe[j], rot_mat[:,:,j])
    jibLe[j] = np.dot(jibLe[j], rot_mat[:,:,j])
    balAxis[j] = np.dot(balAxis[j], rot_mat[:,:,j])
    
    mainLe[j] = world_to_vlm(mainLe[j])
    jibLe[j] = world_to_vlm(jibLe[j])
    balAxis[j] = world_to_vlm(balAxis[j])

solidsailCoords = world_to_vlm(solidsailCoords)
    



#%% Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
solidsailRig = asb.Airplane(
    name="solidsail rig",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        asb.Wing(
            name="mainsail",
            symmetric=False,  # Should this wing be mirrored across the XZ plane?
            xsecs=[ asb.WingXSec(  # Root
                    xyz_le = mainLe[j,i],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord = mainChords[i],
                    twist = mainTwist[i] - balSheeting[j],  # degrees
                    airfoil = mainAirfoil
                    ).translate(solidsailCoords[j])  # Airfoils are blended between a given XSec and the next one
            for i in range(nSections)]
        )
        for j in range(nSolidsail)] +\
        [
        asb.Wing(
            name="jibsail",
            symmetric=False,
            xsecs=[ asb.WingXSec(  # Root
                    xyz_le = jibLe[j,i],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord = jibChords[i],
                    twist = jibTwist[i] - balSheeting[j] - jibSheeting,  # degrees
                    airfoil = jibAirfoil
                    ).translate(solidsailCoords[j])  # Airfoils are blended between a given XSec and the next one
            for i in range(nSections)]
        )
        for j in range(nSolidsail)
    ],
    #     [
    #     asb.Wing(
    #         name="mast",
    #         symmetric=False,
    #         xsecs=[ asb.WingXSec(  # Root
    #                 xyz_le = mastAxis[i],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
    #                 chord = mastRadius[i]*2,
    #                 twist = -balSheeting,  # degrees
    #                 airfoil = asb.Airfoil("naca0080")
    #                 ).translate(solidsailCoords[j])  # Airfoils are blended between a given XSec and the next one
    #         for i in range(len(mastRadius))]
    #     )
    #     for j in range(nSolidsail)
    # ],
    fuselages=[
        asb.Fuselage(
            name="balestron",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=balAxis[j,i],
                    radius=balRadius[i]
                ).translate(solidsailCoords[j])
            for i in range(len(balRadius))
            ]
        ) for j in range(nSolidsail)
    ]
)


solidsailRig.draw()





#%% MESH CONVERGENCE

nChordwiseMax = 5
AR = solidsailRig.wings[0].aspect_ratio()
span = solidsailRig.wings[0].span()


RES = []
TIME = []
N = np.arange(3, nChordwiseMax+1, 1)

t1 = time.time()

for nChordwise in N:
    t0 = t1
    
    nSpanwise = int(AR * nChordwise / nSections + 1)
    # nSpanwise = 4
    
    vlm = GradientVortexLatticeMethod(
        airplane=solidsailRig,
                     z0=z0,
                     tws0=tws0,
                     twa0=twa0,
                     stw=stw,
            spanwise_resolution=nSpanwise,
            chordwise_resolution=nChordwise,
            align_trailing_vortices_with_wind=True
    )
    
    aero = vlm.run() # Runs and prints results to console
    
    area = np.sum([wing.area() for wing in solidsailRig.wings])
    
    rho = 1.225
    nu = 1.81e-5
    chord = span/AR
    Re = aws0*0.514 * chord / nu
    
    CL = aero['CL']
    CDi = aero['CD']
    CDv = 1.328 / np.sqrt(Re)
    
    CX = CL * sind(awa0) - (CDi + CDv) * cosd(awa0)
    CY = CL * cosd(awa0) + (CDi + CDv) * sind(awa0)
    
    FX = 0.5*rho*area*CX*(aws0*.514)**2
    FY = 0.5*rho*area*CY*(aws0*.514)**2
    
    # FX, FZ, FY = aero["F_b"]
    
    t1 = time.time()
    
    RES.append(FY/FX)
    TIME.append(t1 - t0)
    
vlm.draw() # Creates an interactive display of the surface pressures and streamline
    
plt.plot(N, RES)
plt.show()

plt.plot(N, TIME)
plt.show()

# aero

#%% 
vlm = asb.VortexLatticeMethod(
    airplane=skiff,
    op_point=asb.OperatingPoint(
        velocity=25,  # m/s
        alpha=awa),  # degree
        spanwise_resolution=10,
        chordwise_resolution=5,
        align_trailing_vortices_with_wind=True
)

aero = vlm.run() # Runs and prints results to console
vlm.draw() # Creates an interactive display of the surface pressures and streamline

aero


#%%


# vlm.run()  # Returns a dictionary
# vlm.draw()

# for k, v in aero.items():
#     print(f"{k.rjust(4)} : {v}")


#%% NBVAL_SKIP
# # (This tells our unit tests to skip this cell, as it's a bit wasteful to run on every commit.)

# opti = asb.Opti()

# alpha = opti.variable(init_guess=5)

# vlm = asb.VortexLatticeMethod(
#     airplane=airplane,
#     op_point=asb.OperatingPoint(
#         velocity=25,
#         alpha=alpha
#     ),
#     align_trailing_vortices_with_wind=False,
# )

# aero = vlm.run()


#%%

# L_over_D = aero["CL"] / aero["CD"]

# opti.minimize(-L_over_D)

# sol = opti.solve()

#%% 

# best_alpha = sol.value(alpha)
# print(f"Alpha for max L/D: {best_alpha:.3f} deg")


