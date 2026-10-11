# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:07:57 2026

@author: jules
"""

import os

import archibald.numpy as np

from archibald.dynamics.hydro import holtrop, michell
from archibald.geometry import Hull, Sailboat
from archibald.performance import OperatingPoint

hulls = []

stl_files = [
    r'data/nacra15_data/hull_sbd.stl',
    r'data/nacra15_data/hull_prt.stl',
]

for file in stl_files:
    hulls.append(
        Hull(
            name=os.path.basename(file[:-4]),
            mesh=file,
        )
    )
    
nacra = Sailboat(
    name="nacra",
    hulls=hulls,
)

# nacra.draw()

op_point = OperatingPoint(
    stw=3,
    heel=10.,
)

# nacra.transform(op_point)

for hull in hulls:
    # hull.mesh.transform(op_point)
    hull.compute_hydrostatics_properties(op_point)
    
hull.mesh.draw(
    draw_plane=True,
    point=np.array([0, 0, 0]),
    backend="matplotlib",
    # backend="plotly",
)


# Individual resistance of each hull, then interaction between the two hulls,
# both through the Sailboat class. The hull positions are read from the STL
# files, which must therefore be expressed in the common boat frame.
# for interaction in ("michell", "michell_raw"):
    
#     nacra.compute_resistance(
#         op_point,
#         method="holtrop",
#         interaction=interaction,
#         **{'Csternchoice': 1, 'Bulbchoice': 0},
#     )
    
#     print(interaction)
#     print("components")
#     for hull in nacra.hulls:
#         print(hull.name, hull.resistance_components)
#     print("interactions")
#     print(nacra.hull_interaction['dR_hull'], nacra.hull_interaction['dR'])
#     # print(nacra.forces["Fh"])
#     print()

# # Slender hulls: Michell wave resistance for each hull (equivalent hull built
# # from the hydrostatics) with the Holtrop friction, as a custom process.
custom_process = {
    "Rf": holtrop.compute_Rf_holtrop,
    "Rw": michell.compute_Rw_michell,
}

nacra.compute_resistance(
    op_point,
    method=custom_process,
    interaction="michell_jamaluddin",
    **{'Csternchoice': 1, 'Bulbchoice': 0}
)
print("Michell")
print(nacra.forces["Fh"])

# # No interaction, for comparison
# nacra.compute_resistance(
#     op_point,
#     method="holtrop",
#     **{'Csternchoice': 1, 'Bulbchoice': 0}
# )
# print("Holtrop")
# print(nacra.forces["Fh"])

# nacra.draw()
