# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:57:11 2023

@author: jrich
"""

from tools.math_utils import *
from tools.dyn_utils import *
from tools.geom_utils import *

import aerosandbox as asb
import aerosandbox.numpy as asbnp

from gradient_vlm import GradientVortexLatticeMethod

def update_mast_position(dx,
                         mainLe, mainChords,
                         jibHeadPt, jibFoot):
    
    # Create copies of the input arrays
    newMainLe = np.copy(mainLe)
    newMainChords = np.copy(mainChords)
    newJibHeadPt = np.copy(jibHeadPt)
    newJibFoot = np.copy(jibFoot)

    # Add dx to the arrays
    newMainLe[:,0] += dx
    newMainChords += dx
    newJibHeadPt[0] += dx
    newJibFoot -= dx

    # Return the updated variables
    return newMainLe, newMainChords, newJibHeadPt, newJibFoot

# mainAirfoil = asb.Airfoil("sd7037")
# jibAirfoil = asb.Airfoil("sd7037")
mainAirfoil = asb.Airfoil("naca6403")
jibAirfoil = asb.Airfoil("naca6403")

# Define rig initial geometry
nSections = 10

mainDxf = 'D:/000Documents/Cours/DPEA/paki_aka/voilure/gv_rig2.dxf'
# jibDxf = 'C:/Users/jrich/Documents/paki_aka/voilure/jib1.dxf'

mainLe, mainChords = dxf_to_le_chords(mainDxf, nSections, method='bezier')

# Jibsail rig 2
jibTackPt = np.array([5.83, 0, 0.58])
jibHeadPt = np.array([2.24, 0, 6.77])

jibFoot = 3.05
jibHead = 0.20

jibLe = np.linspace(jibTackPt, jibHeadPt, nSections)
jibChords = np.linspace(jibFoot, jibHead, nSections)

nChordwise = 20
nSpanwise = 50




mastDx = 0

# Update mast position
mainLe, mainChords, newJibHeadPt, newJibFoot = update_mast_position(mastDx,
                                                                    mainLe, mainChords,
                                                                    jibHeadPt, jibFoot)

jibLe = np.linspace(jibTackPt, newJibHeadPt, nSections)
jibChords = np.linspace(newJibFoot, jibHead, nSections)

mainLe[:,2] *=1.083
mainChords *= .8

# mainLe[:,2] *= 0.9394
# mainChords *= 1.2


#Twisting
mainTwistAngle = 7.25
jibTwistAngle = mainTwistAngle * 1.2
twistPower = 0.912

mainAoa = 15
jibAoa = 10

tws = 15.
twa = 50.
stw = 10.
drift = 0.

aws, awa = compute_AW(tws, twa, stw)

mainSheeting = awa - drift - mainAoa
jibSheeting = awa - drift - jibAoa

mainTwist = - np.linspace(0,1,nSections)**twistPower * mainTwistAngle - mainSheeting
jibTwist = - np.linspace(0,1,nSections)**twistPower * jibTwistAngle - jibSheeting


mainLe = np.transpose(np.vstack([-mainLe[:,0], mainLe[:,2], mainLe[:,1]]))
jibLe = np.transpose(np.vstack([-jibLe[:,0], jibLe[:,2], jibLe[:,1]]))


#%% Define the 3D geometry you want to analyze/optimize.
# Here, all distances are in meters and all angles are in degrees.
skiff = asb.Airplane(
    name="18ft Skiff",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        asb.Wing(
            name="mainsail",
            symmetric=False,  # Should this wing be mirrored across the XZ plane?
            xsecs=[ asb.WingXSec(  # Root
                    xyz_le = mainLe[i],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord = mainChords[i],
                    twist = mainTwist[i],  # degrees
                    airfoil = mainAirfoil)  # Airfoils are blended between a given XSec and the next one
            for i in range(nSections)]
        ),
        asb.Wing(
            name="jibsail",
            symmetric=False,
            xsecs=[ asb.WingXSec(  # Root
                    xyz_le = jibLe[i],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord = jibChords[i],
                    twist = jibTwist[i],  # degrees
                    airfoil = jibAirfoil)  # Airfoils are blended between a given XSec and the next one
            for i in range(nSections)]
        )
    ]
)


skiff.draw()


#%% 
vlm = GradientVortexLatticeMethod(
        airplane=skiff,
        z0=10., # m
        tws0=tws, # kts
        twa0=twa, # deg
        stw=stw, #kts
        a=0.1,
        spanwise_resolution=4,
        chordwise_resolution=10,
        align_trailing_vortices_with_wind=True
)

aero = vlm.run() # Runs and prints results to console
# vlm.draw() # Creates an interactive display of the surface pressures and streamline

CL = aero['CL']
CD = aero['CD']

CX = CL * sind(awa) - CD * cosd(awa)
CY = CL * cosd(awa) + CD * sind(awa)

Fab = aero['F_g']
Mab = aero['M_g']

centroid = aero['centroid']

print(-Fab[0], Fab[2], Fab[2]/-Fab[0])
print(Fab[2]*centroid[1])


#%% 
vlm2 = GradientVortexLatticeMethod(
        airplane=skiff,
        z0=10., # m
        tws0=tws, # kts
        twa0=twa, # deg
        stw=stw, #kts
        a=0.,
        spanwise_resolution=4,
        chordwise_resolution=10,
        align_trailing_vortices_with_wind=True,
        # estimate_viscous=True
)

aero2 = vlm2.run() # Runs and prints results to console
# vlm.draw() # Creates an interactive display of the surface pressures and streamline

Fab2 = aero2['F_g']
Mab2 = aero2['M_g']

centroid2 = aero2['centroid']

print(-Fab2[0], Fab2[2], Fab2[2]/-Fab2[0])
print(Fab2[2]*centroid2[1])


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
#     airplane=skiff,
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


