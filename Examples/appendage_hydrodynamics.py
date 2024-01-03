# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:29:47 2024

@author: Jules
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

import matplotlib.pyplot as plt

from archibald.tools.geom_utils import *
from archibald.hull import Hull
from archibald.lifting_planes import Centreboard, Rudder

# Paths to DXF drawings of the appendage plan
# DXF file should contain exactly two lines or splines, being the leading and trailing edge of the appendage

# PAKI'AKA (wooden 18-foot skiff) appenage plan
# Drawn by Quentin Germe and Jules Richeux

daggerDxf = './assets/pakiaka_daggerboard.dxf'

nSections = 6
profile = 'e836.dat'

daggerLe, daggerChords = dxf_to_le_chords(daggerDxf, nSections)

# Rudder
rudderChords = np.array([0.33, 0.32, 0.30, 0.26, 0.17, 0.02])

rudderLe = np.array([[-0.03,  0.  , -0.01],
                     [-0.04,  0.  , -0.24],
                     [-0.06,  0.  , -0.47],
                     [-0.11,  0.  , -0.71],
                     [-0.2 ,  0.  , -0.87],
                     [-0.36,  0.  , -0.95]])

shaftRoot = np.array([-0.01, 0, 0.3])

hull = Hull('empty_hull')
hull.add_centreboard('daggerboard', nSections, daggerLe, daggerChords, profile, nSpanwise=15)
hull.add_rudder('rudder', nSections, rudderLe, rudderChords, profile, shaftRoot=shaftRoot)

V = 15 * 1852/3600

delta = 2.2 # drift angle in deg
rudder_angle = -delta * 0.81 # rudder angle in deg

hull.appendages[1].set_angle(rudder_angle)

hull.compute_appendage_hydrodynamics(delta, V, disp = True)

#%% Plotting

plt.figure(dpi=300)

for appendage in hull.appendages.values():
    plt.plot(appendage.le[:,0], appendage.le[:, 2], color='black', lw=1)
    plt.plot(appendage.le[:,0]-appendage.chords, appendage.le[:, 2], color='black', lw=1)

plt.scatter(hull.centroid[0], -hull.centroid[2], color='red')

plt.axis('equal')
plt.grid()
plt.show()
