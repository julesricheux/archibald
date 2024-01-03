# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:45:51 2024

@author: Jules
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

import matplotlib.pyplot as plt

from archibald.tools.geom_utils import *
from archibald.rig import Rig

# Paths to DXF drawings of the sail plan
# DXF file should contain exactly two lines or splines, being the leading and trailing edge of the sail

mainDxf = './assets/pakiaka_mainsail.dxf'
jibDxf = './assets/pakiaka_jibsail.dxf'

nSections = 10

mainLe, mainChords = dxf_to_le_chords(mainDxf, nSections)
jibLe, jibChords = dxf_to_le_chords(jibDxf, nSections)


#Twisting
mainTwistAngle = 7.25
jibTwistAngle = mainTwistAngle * 1.2
twistPower = 0.912

mainTwist = - np.linspace(0,1,nSections)**twistPower * mainTwistAngle
jibTwist = - np.linspace(0,1,nSections)**twistPower * jibTwistAngle

# Sheeting
mainAoa = 15
jibAoa = 10

awa = 45 # deg
aws = 12 # kts

mainSheeting = awa - mainAoa
jibSheeting = awa - jibAoa

beta = [mainSheeting, jibSheeting]

# Build rig
rig = Rig('pakiaka_rig')

rig.add_mainsail('main', nSections, mainLe, mainChords,
                 minCamber=0.03, maxCamber=0.13, maxTwist=mainTwist)

rig.add_jib('jib', nSections, jibLe, jibChords,
            minCamber=0.03, maxCamber=0.13, maxTwist=jibTwist)

# Compute rig aerodynamics
rig.compute_aerodynamics(awa, aws, beta, disp=True)


#%% Plotting

plt.figure(dpi=300)

for sail in rig.sails.values():
    plt.plot(sail.le[:,0], sail.le[:, 2], color='black', lw=1)
    plt.plot(sail.le[:,0]-sail.chords, sail.le[:, 2], color='black', lw=1)

plt.scatter(rig.centroid[0], rig.centroid[2], color='red')

plt.axis('equal')
plt.grid()
plt.show()
