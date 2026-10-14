# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:00:10 2026

@author: jules
"""

# -*- coding: utf-8 -*-
import archibald.numpy as np

from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh
from archibald.geometry import Hull
from archibald.toolbox.mesh_utils import load_stl

from archibald.toolbox.math_utils import waterplane_normal

from archibald.dynamics.hydro import savitsky, holtrop, dsyhs, michell

DEFAULT_RESISTANCE_METHODS = {
    "dsyhs": {
        "Rf": dsyhs.compute_Rf_dsyhs,
        "Rw": dsyhs.compute_Rw_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
    },
    "holtrop": {
        "Rf": holtrop.compute_Rf_holtrop,
        "Rw": holtrop.compute_Rw_holtrop,
        "Rb": holtrop.compute_Rb_holtrop,
        "Rtr": holtrop.compute_Rtr_holtrop,
        "Ra": holtrop.compute_Ra_holtrop,
    },
    "savitsky": {
        
        "Rf": dsyhs.compute_Rf_dsyhs,
        "Rw": dsyhs.compute_Rw_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
        
        # "Rw": michell.compute_Rw_michell,
        # "Rw": holtrop.compute_Rw_holtrop,
        # "Rw": lambda **kwargs: 0.,
        
        # "Rf": savitsky.compute_Rf_savitsky,
        # "Rtr": savitsky.compute_Rtr_savitsky,
        
        "Rp": savitsky.compute_Rp_savitsky,
        "Rsp": savitsky.compute_Rsp_savitsky,
        "Raw": savitsky.compute_Raw_savitsky,
        "Fz": savitsky.compute_Fz_savitsky,     # portance verticale (torseur)
        # "My": savitsky.compute_My_savitsky,     # moment de tangage (torseur)
    },
}

#%% DATA

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

# T0, ref = 0., 0.200513
T0, ref = 0., 0.290513
# T0, ref = 0.01, 0.234148


#%%

overhang = 3.775 # deg for 49er

opti = Opti()

hullStl = r'data/49er_data/hull.stl'

hull = Hull(
    mesh=hullStl,
)

stw = 18.
# T = stw / 100
# trim = stw / 2
T = 0.13
trim = 4.
# T = 0.
# trim = 0.

T = opti.variable(init_guess=T)
# trim = opti.variable(init_guess=trim, lower_bound=overhang + 2.)
trim = opti.variable(init_guess=trim)

op_point = OperatingPoint(
    stw=stw,
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

volume = hull.hydrostatics_data["volume"]

tau = hull.mesh.keel_line_angle(window=(0, 0.5))["keel_angle"]
deadrise = hull.mesh.deadrise_angles(locations=(0.66,))['deadrise']

# tau = trim - overhang
# deadrise = 5.0

# volume = hull.mesh.hydrostatics(mesh_scale=)["volume"]

hull.compute_resistance(
    op_point,
    method=DEFAULT_RESISTANCE_METHODS["savitsky"],
    # **{'Csternchoice': 1, 'Bulbchoice': 0}
    **{'trim': tau, 'deadrise': deadrise, "use_Aws": True}
)
    
froude =  op_point._stw / np.sqrt(hull.hydrostatics_data["Lwl"] * 9.8066)

# opti.minimize(
#     np.sum([hull.resistance_components[c] for c in ["Rf", "Rp", "Rtr", "Rsp"]])
# )
hrc = hull.resistance_components

opti.minimize(
    hrc["Rf"] +
    # hrc["Rw"] +
    hrc["Rtr"] + 
    hrc["Rp"] +
    hrc["Rsp"] +
    0.
)

opti.subject_to([
    (volume * rho * 9.8066 + hrc["Fz"] - ref * rho * 9.8066) == 0.,
    # (volume * rho * 9.8066 - ref * rho * 9.8066) == 0.,
])



sol = opti.solve()

#%% DRAWING

# Lwl = hull.hydrostatics_data["Bwl"]
# Bwl = hull.hydrostatics_data["Bwl"]

# print(savitsky.tau(sol(trim)), savitsky.lambda_(trim=sol(trim), Lwl=sol(Lwl), Bwl=sol(Bwl), deadrise=1.))

print(f"Trim =\t\t{sol(trim):.3f} deg")
print(f"Draft =\t\t{sol(T):.3f} m")
# print(f"R =\t\t\t{sol(opti.f):.3f} N")
print(f"R =\t\t\t{sol(opti.f)/9.8066:.3f} kg")
print(f"Dyn lift% =\t{sol(hull.resistance_components['Fz'] / (ref * rho * 9.8066)) * 100:.3f} %")
print(f"Disp. =\t\t{sol(volume * rho * 9.8066 + hrc["Fz"])} N")
print()
for r in ["Rf", "Rw", "Rp", "Rtr", "Rsp", "Fz"]:
    print(f"{r} = \t{sol(hrc[r]):.3f} N")

# print(f"Volume : {sol(volume)*1000:.1f} L")
# # print(f"Draft error : {sol(T-T0)/T0*100.:.1f} %")
# print(f"Volume error : {sol(volume-ref)/ref*100.:.1f} %")
