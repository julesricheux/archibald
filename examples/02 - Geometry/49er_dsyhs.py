# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:20:18 2026

@author: jules
"""

import archibald.numpy as np

from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh
from archibald.geometry import Hull
from archibald.toolbox.mesh_utils import load_stl

from archibald.toolbox.math_utils import waterplane_normal

from archibald.dynamics.hydro import savitsky, holtrop, dsyhs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

#%% DATA

hullStl = r'data/49er_data/hull.stl'

hull = Hull(
    mesh=hullStl,
)


rho = 1025.
displacement = 122. + 2*90. # kg

cog = np.array([
    2.40,
    .1,
    0.14
]) # m

# hullStl = r'data/49er_data/hull.stl'

# mesh = ArchibaldMesh(
#     *load_stl(hullStl)
# )

T0, ref = 0., 0.290513
# T0, ref = 0.01, 0.234148

#%%

overhang = 3.775 # deg for 49er

opti = Opti()

T = opti.variable(init_guess=0.)
trim = 3

op_point = OperatingPoint(
    dz=-T,
    trim=-trim,
)

hull.mesh.transform(op_point)

Fb, Mb = hull.compute_buoyancy(op_point)

# opti.maximize(T)
opti.subject_to([(Fb[2] - ref * 1025 * 9.8066) == 0.])

sol = opti.solve()

hull = Hull(
    mesh=hullStl,
)

op_point = OperatingPoint(
    dz=-sol(T),
    trim=-trim,
)

hull.mesh.transform(op_point)

stw = np.linspace(0., 18., 50)

Rf = []
Rw = []
Rtr = []
R = []

for s in stw:

    op_point = OperatingPoint(
        stw=s,
    )
    
    # hull.mesh.draw(
    #     draw_plane=True,
    #     point=np.array([0, 0, 0]),
    #     # backend="matplotlib",
    #     # backend="plotly",
    # )
    
    hull.compute_hydrostatics_properties(op_point)
    
    hull.compute_resistance(
        op_point,
        method="dsyhs",
        # **{'Csternchoice': 1, 'Bulbchoice': 0}
        # **{'trim': trim, 'deadrise': 10, "use_Aws": True}
    )
    
    hrc = hull.resistance_components
    
    Rf.append(hrc["Rf"])
    Rw.append(hrc["Rw"])
    Rtr.append(hrc["Rtr"])
    R.append(hrc["_total"])

g = 9.8066

Rf = np.asarray(Rf) / g
Rw = np.asarray(Rw) / g
Rtr = np.asarray(Rtr) / g
R = np.asarray(R) / g


fig, ax = plt.subplots(figsize=(8, 5))

b = 1.1228946447372437
L = 4.8577

# stw = stw * 0.5144 / np.sqrt(L * 9.8066)
# stw = stw * 0.5144 / np.sqrt(b * 9.8066)

ax.fill_between(
    stw,
    0,
    Rf,
    alpha=0.5,
    label="Frictional resistance",
)

ax.fill_between(
    stw,
    Rf,
    Rf + Rw,
    alpha=0.5,
    label="Wave resistance",
)

ax.fill_between(
    stw,
    Rf + Rw,
    Rf + Rw + Rtr,
    alpha=0.5,
    label="Transom resistance",
)

ax.plot(
    stw,
    R,
    linewidth=2,
    label="Total resistance",
)

ax.set_xlabel("STW [kn]")
ax.set_ylabel("Resistance [N]")
ax.set_xlim(stw[0], stw[-1])
# ax.grid(True, alpha=0.2)
ax.legend()

plt.xlim(stw[0], stw[-1])
plt.ylim(0, 70)

fig.tight_layout()
plt.show()
