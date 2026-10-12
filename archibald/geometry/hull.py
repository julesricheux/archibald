# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:23:04 2026

@author: jrich
"""

import archibald.numpy as np
import archibald.toolbox.units as u

import archibald.dynamics.hydro.dsyhs as dsyhs
import archibald.dynamics.hydro.holtrop as holtrop
import archibald.dynamics.hydro.michell as michell

import archibald.dynamics.aero.orc as orc
import archibald.dynamics.aero.day as day
import archibald.dynamics.aero.blendermann as blendermann

from typing import Union, List
from archibald.common import ArchibaldObject
from archibald.environment import Environment
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh


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
    # Slender hulls: Michell thin-ship wave resistance on an equivalent hull
    # built from the hydrostatics, with the Holtrop friction (needs the same
    # extra kwargs as "holtrop", e.g. Csternchoice).
    "michell": {
        "Rf": holtrop.compute_Rf_holtrop,
        "Rw": michell.compute_Rw_michell,
    },
}


# Windage methods. Each of them returns (Fx, Fy, zce) or (Fx, Fy, zce, dxce),
# see archibald.dynamics.aero.common. The element named "Crew" is not related to
# the hull: ``Sailboat.compute_windage`` evaluates it once for the whole boat.
DEFAULT_WINDAGE_METHODS = {
    "orc": {
        "Hull": orc.compute_Fhull_orc,
        "Crew": orc.compute_Fcrew_orc,
    },
    # Dinghies: Day (2017), anthropometric crew drag.
    "day": {
        "Hull": day.compute_Fhull_day,
        "Crew": day.compute_Fcrew_day,
    },
    # Ships and large powerboats: Blendermann wind load functions (needs a ship_type or its parameters).
    "blendermann": {
        "Hull": blendermann.compute_Fhull_blendermann,
    },
}


def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def get_air_density(op_point, default=1.225):
    """
    Air density [kg/m^3] of the operating point environment, if defined, else default.
    """
    air = getattr(op_point.environment, 'air', None)
    return getattr(air, 'density', default)


class Hull(ArchibaldObject):
    def __init__(
            self,
            name: str = 'hull',
            mesh: Union[ArchibaldMesh, str] = None,
        ):
        if type(mesh) == str:
            from archibald.toolbox.mesh_utils import load_stl
            mesh_path = mesh
            mesh = ArchibaldMesh(*load_stl(mesh_path))
        
        self.name = name
        self.mesh = mesh
        
        self.hydrostatics_data = {}
        self.aerostatics_data = {}
        self.resistance_components = {}
        self.windage_components = {}
        
    def __repr__(self):
        return f"Hull object '{self.name}'"
    
    def draw(
            self, 
            backend: str = "pyvista",
            **kwargs
        ):
        """
        Standardized Hull draw method to behave like a component of the Sailboat.
        """
        if self.mesh:
            # Pass all kwargs (like draw_plane, opacity, etc.) directly to the mesh
            return self.mesh.draw(backend=backend, **kwargs)
            
    def get_waterplane(
            self,
            op_point: OperatingPoint,
        ):
        """
        Point and normal of the water plane, expressed in the hull local frame.
        """
        
        xh = np.mean(self.mesh.vertices[:, 0])
        yh = np.mean(self.mesh.vertices[:, 1])
    
        # Bring the global water plane into the boat's local frame
        point = op_point.apply_transformations(
            # geometry=np.array([[0., 0., 0.]]),
            geometry=np.array([[xh, yh, 0.]]),
            inverse=True,
        )
        normal = op_point.apply_transformations(
            geometry=np.array([[0., 0., 1.]]), 
            inverse=True,
            is_vector=True,
        )
        
        # _temp = self.mesh.copy()
        # _temp.transform(op_point)
    
        # volume, cob = _temp.hydrostatics()
        
        return point, normal
    
    def compute_hydrostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
        
        point, normal = self.get_waterplane(op_point)
        
        self.hydrostatics_data = self.mesh.hydrostatics(point, normal)
        
    def compute_aerostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
        """
        Geometric properties of the hull above the water plane, see ``ArchibaldMesh.aerostatics``.
        """
        
        point, normal = self.get_waterplane(op_point)
        
        self.aerostatics_data = self.mesh.aerostatics(point, normal)
        
    def get_position(
            self,
            op_point: OperatingPoint,
        ):
        """
        Position (x, y) [m] of the hull centre of buoyancy in the water axes,
        i.e. with the heel, trim and leeway of the operating point applied.
        Used to compute the interaction between the hulls of a multihull.
        Hydrostatics must have been computed beforehand.
        """
        cob = op_point.apply_transformations(self.hydrostatics_data['cob'])
        cob = np.reshape(cob, (-1,))
        return cob[0], cob[1]
        
    def compute_buoyancy(
            self,
            op_point: OperatingPoint = OperatingPoint(),
            recompute_statics: bool = True,
        ):
        
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)
        
        center = op_point.apply_transformations(self.hydrostatics_data['cob'])
        volume = self.hydrostatics_data['volume']
        rho = op_point.environment.water.density
        g = op_point.environment.gravity
        
        Fb = wide(np.array([
            0.,
            0.,
            1.,
        ])) * volume*rho*g
        
        Mb = np.cross(center, Fb)
        
        return Fb, Mb
    
    
    def compute_resistance_components(
            self,
            op_point: OperatingPoint,
            method: str = str(),
            recompute_statics: bool = True,
            **kwargs,
        ):
        
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)
            
        V = op_point.stw * u.kt
        rho = op_point.environment.water.density
        nu = op_point.environment.water.kinematic_viscosity
        g = op_point.environment.gravity
            
        Re = V * self.hydrostatics_data["Lwl"] / nu
        Fr = V / np.sqrt(self.hydrostatics_data["Lwl"] * g + 1e-12)
        
        # Additionnal parameters needed to compute DSYHS resistance
        env_params = {
            'g': g,         # m/s^2
            'nu': nu,       # m^2/s
            'rho': rho,     # kg/m^3
        }
        
        state_params = {
            'stw': op_point.stw,
            'Re': np.softplus(Re, beta=1e3),
            'Fr': np.softplus(Fr, beta=1e3),
        }
        
        if type(method)==str and method in DEFAULT_RESISTANCE_METHODS.keys():
            process = DEFAULT_RESISTANCE_METHODS[method]
        elif type(method)==dict:
            process = method
        elif type(method)==str:
            raise NotImplementedError(
                f"Unknown method '{method}'. Should be in {list(DEFAULT_RESISTANCE_METHODS.keys())} or a custom dictionary of methods.")
        else:
            raise TypeError("method argument should be str or dict")
            
        r = {}
        t = 0.
        for name, func in process.items():
            r[name] = func(**self.hydrostatics_data, **env_params, **state_params, **kwargs)
            t += r[name]
        
        r["_total"] = t
        
        return r
    
    
    def compute_resistance(
            self,
            op_point: OperatingPoint,
            method: str = str(),
            recompute_statics: bool = True,
            **kwargs,
        ):
        
        self.resistance_components = self.compute_resistance_components(
            op_point=op_point,
            method=method,
            recompute_statics=recompute_statics,
            **kwargs
        )
        
        center = op_point.apply_transformations(self.hydrostatics_data['cow']) # TODO refine position
        
        R = self.resistance_components["_total"]
        
        Fh = wide(np.array([
            -1.,
            0.,
            0.,
        ])) * R
        
        Mh = np.cross(center, Fh)
        
        return Fh, Mh
        
        
    def compute_windage_components(
            self,
            op_point: OperatingPoint,
            tws: float,
            twa: float,
            method: Union[str, dict] = "orc",
            recompute_statics: bool = True,
            crew_position: Union[np.ndarray, List] = None,
            **kwargs,
        ):
        """
        Aerodynamic force of the wind on the above-water part of the hull and on the crew.
        
        Parameters
        ----------
        op_point : OperatingPoint, boat attitude and speed through water.
        tws : float, True wind speed [kt], at the reference height of the wind profile.
        twa : float, True wind angle [deg], positive when the wind comes from starboard.
        method : str or dict, key of ``DEFAULT_WINDAGE_METHODS`` or custom dict of methods.
        recompute_statics : bool, whether to recompute the aerostatics of the hull.
        rho_air : float or None, air density [kg/m^3]. Default from the environment, else 1.225.
        crew_position : (1, 3) or (3,) array or None, position (x, y, z) of the crew in the
            boat frame [m], y being non zero for a crew hiking or on a trapeze. It is brought
            in the water axes, and given to the windage methods as ``crew_position``, where the
            "Crew" element uses its z coordinate as the height of its centre of effort.
        **kwargs : parameters of the windage methods (crew_mass, n_crew, Cd_hull, ...).
        
        Returns
        -------
        dict, for each element of the method: ``Fx`` and ``Fy`` [N] along the boat
        x (forward) and y (port) axes, ``zce`` [m], height of its centre of effort
        above the waterplane, and ``dxce`` [m], longitudinal offset of its application
        point from the lateral centre of effort of the hull. ``_total`` gives the sum
        of ``Fx`` and ``Fy``.
        """
        
        if recompute_statics:
            self.compute_aerostatics_properties(op_point)
            
        g = op_point.environment.gravity
        rho_air = op_point.environment.air.density
            
        env_params = {
            'g': g,                 # m/s^2
            'rho_air': rho_air,     # kg/m^3
        }
        
        state_params = {
            'stw': op_point.stw,
            'tws': tws,
            'twa': twa,
            'heel': getattr(op_point, 'heel', 0.),
        }
        
        # Hydrostatic properties used by some windage methods
        hydro_params = {
            key: self.hydrostatics_data[key]
            for key in ("Cwp",)
            if key in self.hydrostatics_data
        }
        
        # Centroids are brought in the water axes, as done for the buoyancy: 
        # their z coordinate is then the height above the water plane.
        aero_data = dict(self.aerostatics_data)
        for key in ("cov", "cod", "cex", "cey"):
            aero_data[key] = op_point.apply_transformations(self.aerostatics_data[key])
        
        # The crew position is given only to the methods when defined, so that they can
        # fall back on their default height otherwise.
        if crew_position is not None:
            aero_data["crew_position"] = op_point.apply_transformations(wide(np.array(crew_position)))
        
        if type(method)==str and method in DEFAULT_WINDAGE_METHODS.keys():
            process = DEFAULT_WINDAGE_METHODS[method]
        elif type(method)==dict:
            process = method
        elif type(method)==str:
            raise NotImplementedError(
                f"Unknown method '{method}'. Should be in {list(DEFAULT_WINDAGE_METHODS.keys())} or a custom dictionary of methods.")
        else:
            raise TypeError("method argument should be str or dict")
            
        r = {}
        Fx_tot = 0.
        Fy_tot = 0.
        for name, func in process.items():
            out = func(**aero_data, **hydro_params, **env_params, **state_params, **kwargs)
            Fx, Fy, zce = out[:3]
            # optional longitudinal offset of the application point from the lateral centroid [m]
            dxce = out[3] if len(out) > 3 else 0.
            r[name] = {"Fx": Fx, "Fy": Fy, "zce": zce, "dxce": dxce}
            Fx_tot = Fx_tot + Fx
            Fy_tot = Fy_tot + Fy
        
        r["_total"] = {"Fx": Fx_tot, "Fy": Fy_tot}
        
        return r
    
    
    def compute_windage(
            self,
            op_point: OperatingPoint,
            tws: float,
            twa: float,
            method: Union[str, dict] = "orc",
            recompute_statics: bool = True,
            crew_position: Union[np.ndarray, List] = None,
            **kwargs,
        ):
        """
        Aerodynamic force and moment of the wind on the above-water part of the hull
        and on the crew, in the water axes. See ``compute_windage_components``.
        
        The hull elements are applied at the horizontal position of the lateral centre
        of effort of the hull (plus the offset ``dxce`` of the method, if any), and at
        their own height. The "Crew" element is applied at ``crew_position`` if given,
        otherwise like the hull elements.
        
        Parameters
        ----------
        crew_position : (1, 3) or (3,) array or None, position (x, y, z) of the crew in the
            boat frame [m]. The force is applied at this point, off the centreline if y
            is not zero, which gives the yaw moment term -y * Fx.
        
        Returns
        -------
        Fa, Ma : (1, 3) arrays, force [N] and moment [N.m] about the origin.
        """
        
        self.windage_components = self.compute_windage_components(
            op_point=op_point,
            tws=tws,
            twa=twa,
            method=method,
            recompute_statics=recompute_statics,
            crew_position=crew_position,
            **kwargs
        )
        
        cey = op_point.apply_transformations(self.aerostatics_data['cey'])
        
        if crew_position is not None:
            crew_center = op_point.apply_transformations(wide(np.array(crew_position)))
        
        Fa = wide(np.zeros(3))
        Ma = wide(np.zeros(3))
        
        for name, c in self.windage_components.items():
            if name == "_total":
                continue
            
            F = (
                wide(np.array([1., 0., 0.])) * c["Fx"]
                + wide(np.array([0., 1., 0.])) * c["Fy"]
            )
            # horizontal position of the application point
            if name == "Crew" and crew_position is not None:
                xy = crew_center
            else:
                xy = cey
            
            center = (
                xy * wide(np.array([1., 1., 0.]))
                + wide(np.array([1., 0., 0.])) * c["dxce"]
                + wide(np.array([0., 0., 1.])) * c["zce"]
            )
            
            Fa = Fa + F
            Ma = Ma + np.cross(center, F)
        
        return Fa, Ma
        
        
if __name__=="__main__":
    import os
    from archibald.optimization import Opti
    
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
    
    hull.compute_resistance(
        op_point,
        method="michell",
        **{'Csternchoice': 1, 'Bulbchoice': 0}
    )
    print(hull.resistance_components)
    
    print(hull.get_position(op_point))
    
    hull.compute_aerostatics_properties(op_point)
    print(hull.aerostatics_data)
    
    print(hull.compute_windage(op_point, tws=12., twa=45., method="orc", n_crew=1.))
    print(hull.windage_components)
    
    print(hull.compute_windage(op_point, tws=12., twa=45., method="day", crew_mass=80., crew_height=1.83))
    print(hull.windage_components)