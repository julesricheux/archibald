# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:49:48 2026

@author: jrich
"""

import os
from archibald.optimization import Opti

from archibald.dynamics.hydro import holtrop, dsyhs, savitsky

from archibald.geometry.hull import Hull
from archibald.performance import OperatingPoint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

stl = os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")
hull = Hull(mesh=stl)

T0, ref = 1.3, 116.487

# hull.draw(op_point, set_axis_visibility=True)

opti = Opti()

# T = opti.variable(init_guess = T0)
T = T0

op_point = OperatingPoint(
    stw=6.,
    dz=-T,
    # leeway=45.,
    # trim=0.1,
    # heel=45.
)

hull.compute_buoyancy(op_point)

volume = hull.hydrostatics_data['volume']

opti.minimize((volume - ref)**2.)

sol = opti.solve()

# print(sol(T))

hull.compute_resistance(
    op_point,
    method="dsyhs",
)
print(hull.resistance_components)

hull.compute_resistance(
    op_point,
    method="holtrop",
    **{'Csternchoice': 1, 'Bulbchoice': 0}
)
print(hull.resistance_components)

custom_process = {
    "Rf": holtrop.compute_Rf_holtrop,
    "Rw": dsyhs.compute_Rw_dsyhs,
}

hull.compute_resistance(
    op_point,
    method=custom_process,
    **{'Csternchoice': 1, 'Bulbchoice': 0}
)
print(hull.resistance_components)