# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:07:57 2026

@author: jules
"""

import os

import archibald.numpy as np

from archibald.dynamics.hydro import holtrop, michell
from archibald.geometry import Hull, Sailboat, Rig, Sail, WingXSec, ThinAirfoil, Fin, Appendage, Airfoil
from archibald.performance import OperatingPoint

from archibald.toolbox.geom_utils import dxf_to_le_chords, read_dxf

#%% HULLS

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
    
#%% RIG

dxf_files = [
    r'data/nacra15_data/mainsail.dxf',
    r'data/nacra15_data/jib.dxf',
]

sails = []
for file in dxf_files:
    le, chords = dxf_to_le_chords(file, 10)
    
    sails.append(
        Sail(
            name=os.path.basename(file[:-4]),
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.4, mc=0.1),
                    twist=-i/10 * 20,
                )
            for i, (xyz, c) in enumerate(zip(le, chords))]
        )
    )


rig = Rig(
    name="nacra_rig",
    sails=sails,
)

#%% APPENDAGE

dxf_files = [
    r'data/nacra15_data/fin_prt.dxf',
    r'data/nacra15_data/rudder_sbd.dxf',
    r'data/nacra15_data/rudder_prt.dxf',
]

airfoils = ["e854", "e836", "e836"]

apps = []
for file, af in zip(dxf_files, airfoils):
    le, chords = dxf_to_le_chords(file, 10)
    
    apps.append(
        Fin(
            name=os.path.basename(file[:-4]),
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=Airfoil(af).scale(scale_y=-1.0),
                    twist=0.,
                )
            for i, (xyz, c) in enumerate(zip(le, chords))]
        )
    )

appendage = Appendage(
    name="nacra_appendage",
    fins=apps,
)

#%% SALBOAT
    
nacra = Sailboat(
    name="nacra",
    hulls=hulls,
    rigs=[rig],
    appendages=[appendage],
)

# nacra.draw()

op_point = OperatingPoint(
    stw=10,
    twa=55,
    tws0=20,
    heel=-6.,
    mainsail_trim=-10,
    jib_trim=-15,
)

main_le, main_chords = dxf_to_le_chords(r'data/nacra15_data/mainsail.dxf', 10)
jib_le, jib_chords = dxf_to_le_chords(r'data/nacra15_data/jib.dxf', 7)

rig["mainsail"] = rig["mainsail"].rotate_local(
    angle_deg=op_point.mainsail_trim,
    axis=jib_le[-1] - main_le[0],
    origin=main_le[0],
)

rig["jib"] = rig["jib"].rotate_local(
    angle_deg=op_point.jib_trim,
    axis=jib_le[-1] - jib_le[0],
    origin=jib_le[0],
)

# nacra.transform(op_point)

for hull in hulls:
    # hull.mesh.transform(op_point)
    hull.compute_hydrostatics_properties(op_point)
    
# hull.mesh.draw(
#     draw_plane=True,
#     point=np.array([0, 0, 0]),
#     # backend="matplotlib",
#     # backend="plotly",
# )


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
nacra.draw()

# # Slender hulls: Michell wave resistance for each hull (equivalent hull built
# # from the hydrostatics) with the Holtrop friction, as a custom process.
custom_process = {
    "Rf": holtrop.compute_Rf_holtrop,
    "Rw": michell.compute_Rw_michell,
    "Rtr": holtrop.compute_Rtr_holtrop,
}

nacra.compute_resistance(
    op_point,
    method=custom_process,
    interaction="michell_jamaluddin",
    **{'Csternchoice': 1, 'Bulbchoice': 0}
)
print("Michell")
print(nacra.forces["Fh"])
for hull in nacra.hulls:
    print(hull.name, hull.resistance_components)
    
nacra.compute_aerodynamics(op_point)
nacra.compute_hydrodynamics(op_point)

for k, v in nacra.forces.items():
    print(k, v)

# # No interaction, for comparison
# nacra.compute_resistance(
#     op_point,
#     method="holtrop",
#     **{'Csternchoice': 1, 'Bulbchoice': 0}
# )
# print("Holtrop")
# print(nacra.forces["Fh"])

# nacra.draw()
