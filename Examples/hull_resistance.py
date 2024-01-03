# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:10:37 2024

@author: Jules
"""


import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))
    
import numpy as np
import matplotlib.pyplot as plt

from archibald.hull import Hull


# MOLENEZ II (sailing general cargo ship)

mesh = './assets/molenez2_hull.stl' # path to a STL mesh of the hull

# Ship properties
displacement = 140e3 # kg
cog = np.array([11.9,
                0.0,
                2.6]) # m

hull = Hull('molenez2_hull', displacement, cog, mesh)

#%%

vmin = 6 # kts
vmax = 16 # kts

V = np.linspace(vmin, vmax, 20)
R = []

hull.free_trim_immersion(heel=0) # puts the ship to hydrostatic equilibrium

for vkn in V:
    u = vkn * 1852/3600 # m/s
    R.append(hull.compute_hull_resistance(u, method='holtrop')) # computes bare hull resistance at speed u

R = np.array(R)

#%% Plotting

font = {'fontname':'Arimo'}
fontsize = 11
resolution = 300

plt.figure(dpi=resolution)

plt.plot(V, R/1e3, label='Holtrop78')

plt.xlabel("SHIP SPEED (kts)", **font, size=fontsize)
plt.ylabel("HULL RESISTANCE (kN)", **font, size=fontsize)

plt.legend(fontsize = fontsize)
plt.xlim((vmin, vmax))
plt.ylim((0, None))
plt.legend()
plt.grid()

plt.show()

plt.figure(dpi=resolution)

plt.plot(V, R*V*.514/1e3, label='Holtrop78')

plt.xlabel("SHIP SPEED (kts)", **font, size=fontsize)
plt.ylabel("BRAKE POWER (kW)", **font, size=fontsize)

plt.legend(fontsize = fontsize)
plt.xlim((vmin, vmax))
plt.ylim((0, None))
plt.legend()
plt.grid()

plt.show()
    