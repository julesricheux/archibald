# -*- coding: utf-8 -*-
"""
FULL SHIP EQUILIBRIUM EXAMPLE

@author: Jules Richeux
@university: ENSA Nantes, FRANCE

"""

#%% DEPENDENCIES

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

from archibald.boat import *
    
import matplotlib.pyplot as plt
    

#%% Sailboat definition

# PAKI'AKA (wooden 18-foot skiff) example
# Drawn by Quentin Germe and Jules Richeux

sailboat = Sailboat('pakiaka')

#Hull definition
mesh = './assets/pakiaka_hull.stl'
displacement = 405
cog = np.array([2.5, 0.0, 0.0])

sailboat.add_hull(displacement=displacement, cog=cog, mesh=mesh)

# Rig definition
nSections = 10

mainDxf = './assets/pakiaka_mainsail.dxf'
jibDxf = './assets/pakiaka_jibsail.dxf'

mainLe, mainChords = dxf_to_le_chords(mainDxf, nSections)
jibLe, jibChords = dxf_to_le_chords(jibDxf, nSections)

sailboat.add_rig()
sailboat.rig.add_mainsail('main', nSections, mainLe, mainChords)
sailboat.rig.add_jib('jib', nSections, jibLe, jibChords)

# Appendages definition
profile = 'e836.dat'
daggerDxf = './assets/pakiaka_daggerboard.dxf'
daggerLe, daggerChords = dxf_to_le_chords(daggerDxf, nSections)

sailboat.hull.add_centreboard('daggerboard', nSections, daggerLe, daggerChords, profile, nSpanwise=10)

#%% Response surfaces building

# Compute response surfaces
sailboat.build_hull_RS(n=10)
sailboat.build_rig_RS(n=20)
sailboat.build_appendage_RS(n=6)

# Save response surfaces for later use
filename = 'example_RS.pkl'
sailboat.save_RS(directory='./archibald_saves/', filename=filename)

# Save response surfaces for later use
# sailboat.load_RS(directory='./archibald_saves/', filename=filename)

#%% Fast equilibrium computation

tws = 10 * 0.514 # m/s
twa = 50. # deg
heel = 0.0 # deg
drift = 1.0 # deg

# V, heel, drift = sailboat.free_speed_heel_drift(tws, twa) # 3 DOF
V, drift = sailboat.free_speed_drift(tws, twa, heel) # 2 DOF
# V = sailboat.free_speed(tws, twa, heel, drift) # 1 DOF

aws, awa = compute_AW(tws, twa, V)

print('V = '+str(round(V/.514, 1))+' kts')
print('heel = '+str(round(heel, 1))+' deg')
print('drift = '+str(round(drift, 1))+' deg')
