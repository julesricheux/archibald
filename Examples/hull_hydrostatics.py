# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:01:54 2024

@author: Jules
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

import numpy as np
import matplotlib.pyplot as plt

from archibald.hull import *

# NEREUS (fast humanitarian supply ship) project hull
# Drawn by Pierre Meyer, Raphaël Gillet and Jules Richeux

mesh = './assets/nereus_hull.stl' # path to a STL mesh of the hull

displacement = 330e3 # kg
cog = np.array([19.8, 
                0.0,
                2.35]) # m

hull = Hull('nereus_hull', displacement, cog, mesh)

#%% GZ computation


HEELS = np.arange(0,80,2)
GZ =[]
for h in HEELS :
    # hull.free_trim_immersion(heel=h, disp=True)
    hull.free_immersion(heel=h, disp=True)
    GZ.append(hull.hydrostaticData['GZt'])
    
GZ = np.array(GZ)
        
#%% Plotting

font = {'fontname':'Arimo'}
fontsize = 11
resolution = 300

plt.figure(dpi=resolution)

plt.plot(HEELS, GZ)
plt.ylim((np.min(GZ), np.max(GZ)))
plt.xlim((np.min(HEELS), np.max(HEELS)))
plt.xlabel('Heel (°)', **font, size=fontsize)
plt.ylabel('GZ (m)', **font, size=fontsize)
plt.grid()
plt.show()

print('phi_GZ_max =', HEELS[np.where(GZ==np.max(GZ))][0], '°')
print('phi_capsize =', HEELS[np.where(GZ[1:]==np.sign(GZ[1:])*np.min(np.abs(GZ[1:])))][0], '°')