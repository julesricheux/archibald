# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:18:45 2026

@author: jrich
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


DEFAULT_RESISTANCE_METHODS = {
    # "dsyhs": {
    #     "Rf": dsyhs.compute_Rf_dsyhs,
    #     "Rw": dsyhs.compute_Rw_dsyhs,
    #     "Rtr": dsyhs.compute_Rtr_dsyhs,
    # },
    # "holtrop": {
    #     "Rf": holtrop.compute_Rf_holtrop,
    #     "Rw": holtrop.compute_Rw_holtrop,
    #     "Rb": holtrop.compute_Rb_holtrop,
    #     "Rtr": holtrop.compute_Rtr_holtrop,
    #     "Ra": holtrop.compute_Ra_holtrop,
    # },
    # "savitsky": {
    #     "Rf": savitsky.compute_Rf_savitsky,
    #     "Rp": savitsky.compute_Rp_savitsky,
    #     "Rsp": savitsky.compute_Rsp_savitsky,
    #     "Rtr": savitsky.compute_Rtr_savitsky,
    #     "Raw": savitsky.compute_Raw_savitsky,
    #     "Fz": savitsky.compute_Fz_savitsky,     # portance verticale (torseur)
    #     # "My": savitsky.compute_My_savitsky,     # moment de tangage (torseur)
    # },
    "composite": {
        "Rf": dsyhs.compute_Rf_dsyhs,
        # "Rf": holtrop.compute_Rf_holtrop,
        "Rw": dsyhs.compute_Rw_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
        
        "Rp": savitsky.compute_Rp_savitsky,
        "Fz": savitsky.compute_Fz_savitsky,
    }
}

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
hullStl = r'data/49er_data/hull.stl'

stw = np.arange(2, 19, 2)

Rf = []
Rw = []
Rtr = []
Rp = []
R = []

TRIM = []

for s in stw:

    opti = Opti()
    
    hull = Hull(
        mesh=hullStl,
    )
    
    T = 0.
    trim = 2. + overhang
    # trim = 0.
    
    T = opti.variable(init_guess=T)
    # trim = opti.variable(init_guess=trim, lower_bound=overhang + 2.)
    # trim = opti.variable(init_guess=trim, lower_bound=-1., upper_bound=s/2.)
    
    op_point = OperatingPoint(
        stw=s,
        dz=-T,
        trim=-trim,
    )
    
    hull.mesh.transform(op_point)
    
    # hull.mesh.draw(
    #     draw_plane=True,
    #     point=np.array([0, 0, 0]),
    #     # backend="matplotlib",
    #     # backend="plotly",
    # )
    
    hull.compute_hydrostatics_properties(op_point)
    
    volume = hull.mesh.hydrostatics()["volume"]
    
    hull.compute_resistance(
        op_point,
        method=DEFAULT_RESISTANCE_METHODS["composite"],
        **{'Csternchoice': 1, 'Bulbchoice': 0},
        **{'trim': trim - overhang, 'deadrise': 3, "use_Aws": True},
    )
        
    froude =  op_point._stw / np.sqrt(hull.hydrostatics_data["Lwl"] * 9.8066)
    
    # opti.minimize(
    #     np.sum([hull.resistance_components[c] for c in ["Rf", "Rp", "Rtr", "Rsp"]])
    # )
    hrc = hull.resistance_components
    
    val = savitsky.validity_factor(
        **{
            "trim": trim - overhang,
            "stw": s,
            "Bwl": hull.hydrostatics_data["Bwl"],
            "g": op_point.environment.gravity
        }
    )
    
    # val = 1.
    
    Fz = hrc["Fz"] * val
    Fx = hrc["Rp"] * val
    
    opti.minimize(
        hrc["Rf"] +
        hrc["Rw"] +
        hrc["Rtr"] + 
        # hrc["Rsp"] +
        Fx
    )
    
    opti.subject_to([
        ((volume - ref) * rho * 9.8066 + Fz) == 0.,
    ])
    
    sol = opti.solve(
        max_iter=300,
        behavior_on_failure="return_last",
        # verbose=False
    )
    
    hrc = sol(hull.resistance_components)
    
    TRIM.append(sol(trim))
    
    print()
    print(f"SPEED {s} kts")
    print()
    print(f"Trim {TRIM[-1]:.3} deg")
    print(f"Fac {sol(val) * 100:.1f} %")
    print(f"Dyn lift% =\t{sol(Fz / (ref * rho * 9.8066)) * 100:.3f} %")
    print()
    
    Rf.append(hrc["Rf"])
    Rw.append(hrc["Rw"])
    Rtr.append(hrc["Rtr"])
    Rp.append(sol(Fx))
    R.append(hrc["Rf"]+hrc["Rw"]+hrc["Rtr"]+sol(Fx))

g = 9.8066

Rf = np.asarray(Rf) / g
Rw = np.asarray(Rw) / g
Rtr = np.asarray(Rtr) / g
Rp = np.asarray(Rp) / g
R = np.asarray(R) / g

#%%

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

ax.fill_between(
    stw,
    Rf + Rw + Rtr,
    Rf + Rw + Rtr + Rp,
    alpha=0.5,
    label="Pressure resistance",
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
