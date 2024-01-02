# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:01:40 2023

@author: Jules Richeux
based on Peter D. Sharpe VLM formulation AeroSandbox
"""

import aerosandbox.numpy as asbnp
from aerosandbox import VortexLatticeMethod
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from typing import Dict, Any, List, Callable
import copy

from tools.geom_utils import *
from tools.math_utils import *
from tools.dyn_utils import *


### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return asbnp.reshape(array, (-1, 1))


def wide(array):
    return asbnp.reshape(array, (1, -1))


class GradientVortexLatticeMethod(VortexLatticeMethod):
    """
    An explicit (linear) vortex-lattice-method aerodynamics analysis.

    Usage example:
        >>> analysis = asb.VortexLatticeMethod(
        >>>     airplane=my_airplane,
        >>>     op_point=asb.OperatingPoint(
        >>>         velocity=100, # m/s
        >>>         alpha=5, # deg
        >>>         beta=4, # deg
        >>>         p=0.01, # rad/sec
        >>>         q=0.02, # rad/sec
        >>>         r=0.03, # rad/sec
        >>>     )
        >>> )
        >>> aero_data = analysis.run()
        >>> analysis.draw()
    """

    def __init__(self,
                 airplane: Airplane,
                 z0: float = 10., # m
                 tws0: float = 10., # kts
                 twa0: float = 90., # kts 
                 stw: float = 10., #kts
                 a: float = 0.,
                 xyz_ref: List[float] = None,
                 run_symmetric_if_possible: bool = False,
                 verbose: bool = False,
                 spanwise_resolution: int = 10,
                 spanwise_spacing_function: Callable[[float, float, float], asbnp.ndarray] = asbnp.cosspace,
                 chordwise_resolution: int = 10,
                 chordwise_spacing_function: Callable[[float, float, float], asbnp.ndarray] = asbnp.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = False,
                 ):
        
        kts2ms = 1852/3600
        
        aws0, awa0 = compute_AW(tws0, twa0, stw)
        
        self.z0 = z0
        self.tws0 = tws0 * kts2ms
        self.twa0 = twa0
        self.stw = stw * kts2ms
        self.a = a
        
        self.aws0 = aws0 * kts2ms
        self.awa0 = awa0
        
        super().__init__(airplane,
                         OperatingPoint(velocity = self.aws0,  # m/s
                                        alpha = self.awa0),
                         xyz_ref,
                         run_symmetric_if_possible,
                         verbose,
                         spanwise_resolution,
                         spanwise_spacing_function,
                         chordwise_resolution,
                         chordwise_spacing_function,
                         vortex_core_radius,
                         align_trailing_vortices_with_wind)
        
    
    def run(self) -> Dict[str, Any]:
        """
        Computes the aerodynamic forces.

        Returns a dictionary with keys:

            - 'F_g' : an [x, y, z] list of forces in geometry axes [N]
            - 'F_b' : an [x, y, z] list of forces in body axes [N]
            - 'F_w' : an [x, y, z] list of forces in wind axes [N]
            - 'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
            - 'M_b' : an [x, y, z] list of moments about body axes [Nm]
            - 'M_w' : an [x, y, z] list of moments about wind axes [Nm]
            - 'L' : the lift force [N]. Definitionally, this is in wind axes.
            - 'Y' : the side force [N]. This is in wind axes.
            - 'D' : the drag force [N]. Definitionally, this is in wind axes.
            - 'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
            - 'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
            - 'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
            - 'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
            - 'CY', the sideforce coefficient [-]. This is in wind axes.
            - 'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
            - 'Cl', the rolling coefficient [-], in body axes
            - 'Cm', the pitching coefficient [-], in body axes
            - 'Cn', the yawing coefficient [-], in body axes

        Nondimensional values are nondimensionalized using reference values in the VortexLatticeMethod.airplane object.
        """

        if self.verbose:
            print("Meshing...")

        ##### Make Panels
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        is_trailing_edge = []

        for wing in self.airplane.wings:
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing_function=self.chordwise_spacing_function,
                add_camber=True
            )
            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])
            is_trailing_edge.append(
                (asbnp.arange(len(faces)) + 1) % self.chordwise_resolution == 0
            )

        front_left_vertices = asbnp.concatenate(front_left_vertices)
        back_left_vertices = asbnp.concatenate(back_left_vertices)
        back_right_vertices = asbnp.concatenate(back_right_vertices)
        front_right_vertices = asbnp.concatenate(front_right_vertices)
        is_trailing_edge = asbnp.concatenate(is_trailing_edge)

        ### Compute panel statistics
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = asbnp.cross(diag1, diag2)
        cross_norm = asbnp.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2

        # Compute the location of points of interest on each panel
        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        collocation_points = (
                0.5 * (0.25 * front_left_vertices + 0.75 * back_left_vertices) +
                0.5 * (0.25 * front_right_vertices + 0.75 * back_right_vertices)
        )

        ### Save things to the instance for later access
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.is_trailing_edge = is_trailing_edge
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.collocation_points = collocation_points

        ##### Setup Operating Point
        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()  # Direction the wind is GOING TO, in geometry axes coordinates
        steady_freestream_direction = steady_freestream_velocity / asbnp.linalg.norm(steady_freestream_velocity)
        rotation_freestream_velocities = self.op_point.compute_rotation_velocity_geometry_axes(
            collocation_points)

        # freestream_velocities = asbnp.add(wide(steady_freestream_velocity), rotation_freestream_velocities)
        # Nx3, represents the freestream velocity at each panel collocation point (c)
        
        rot_mat = rotation_matrix(np.array([0.,1.,0.]), self.twa0)
        
        true_wind_velocities = np.zeros(self.collocation_points.shape)
        true_wind_velocities[:, 0] =  grad_wind(self.tws0, self.collocation_points[:,1], self.z0, self.a)
        true_wind_velocities = np.dot(true_wind_velocities, rot_mat)
        
        ship_speed_velocities = np.ones(self.collocation_points.shape) * np.array([self.stw, 0., 0.])
        
        freestream_velocities = true_wind_velocities + ship_speed_velocities

        freestream_influences = asbnp.sum(freestream_velocities * normal_directions, axis=1)

        ### Save things to the instance for later access
        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities
        
        self.true_wind_velocities = true_wind_velocities
        self.ship_speed_velocities = ship_speed_velocities

        ##### Setup Geometry
        ### Calculate AIC matrix
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        u_collocations_unit, v_collocations_unit, w_collocations_unit = calculate_induced_velocity_horseshoe(
            x_field=tall(collocation_points[:, 0]),
            y_field=tall(collocation_points[:, 1]),
            z_field=tall(collocation_points[:, 2]),
            x_left=wide(left_vortex_vertices[:, 0]),
            y_left=wide(left_vortex_vertices[:, 1]),
            z_left=wide(left_vortex_vertices[:, 2]),
            x_right=wide(right_vortex_vertices[:, 0]),
            y_right=wide(right_vortex_vertices[:, 1]),
            z_right=wide(right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                steady_freestream_direction
                if self.align_trailing_vortices_with_wind else
                asbnp.array([1, 0, 0])
            ),
            gamma=1.,
            vortex_core_radius=self.vortex_core_radius
        )

        AIC = (
                u_collocations_unit * tall(normal_directions[:, 0]) +
                v_collocations_unit * tall(normal_directions[:, 1]) +
                w_collocations_unit * tall(normal_directions[:, 2])
        )

        ##### Calculate Vortex Strengths
        if self.verbose:
            print("Calculating vortex strengths...")

        self.vortex_strengths = asbnp.linalg.solve(AIC, -freestream_influences)

        ##### Calculate forces
        ### Calculate Near-Field Forces and Moments
        # Governing Equation: The force on a straight, small vortex filament is F = rho * cross(V, l) * gamma,
        # where rho is density, V is the velocity vector, cross() is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating forces on each panel...")
        # Calculate the induced velocity at the center of each bound leg
        V_centers = self.get_velocity_at_points(vortex_centers)

        # Calculate forces_inviscid_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        Vi_cross_li = asbnp.cross(V_centers, vortex_bound_leg, axis=1)

        forces_geometry = self.op_point.atmosphere.density() * Vi_cross_li * tall(self.vortex_strengths)
        moments_geometry = asbnp.cross(
            asbnp.add(vortex_centers, -wide(asbnp.array(self.xyz_ref))),
            forces_geometry
        )
        
        # Calculate total forces and moments
        force_geometry = asbnp.sum(forces_geometry, axis=0)
        moment_geometry = asbnp.sum(moments_geometry, axis=0)
        
        centroid_geometry = asbnp.sum(forces_geometry*vortex_centers, axis=0) / force_geometry

        force_body = self.op_point.convert_axes(
            force_geometry[0], force_geometry[1], force_geometry[2],
            from_axes="geometry",
            to_axes="body"
        )
        force_wind = self.op_point.convert_axes(
            force_body[0], force_body[1], force_body[2],
            from_axes="body",
            to_axes="wind"
        )
        moment_body = self.op_point.convert_axes(
            moment_geometry[0], moment_geometry[1], moment_geometry[2],
            from_axes="geometry",
            to_axes="body"
        )
        moment_wind = self.op_point.convert_axes(
            moment_body[0], moment_body[1], moment_body[2],
            from_axes="body",
            to_axes="wind"
        )

        ### Save things to the instance for later access
        self.forces_geometry = forces_geometry
        self.moments_geometry = moments_geometry
        self.centroid = centroid_geometry
        self.force_geometry = force_geometry
        self.force_body = force_body
        self.force_wind = force_wind
        self.moment_geometry = moment_geometry
        self.moment_body = moment_body
        self.moment_wind = moment_wind

        # Calculate dimensional forces
        L = -force_wind[2]
        D = -force_wind[0]
        Y = force_wind[1]
        l_b = moment_body[0]
        m_b = moment_body[1]
        n_b = moment_body[2]

        # Calculate nondimensional forces
        q = self.op_point.dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        CL = L / q / s_ref
        CD = D / q / s_ref
        CY = Y / q / s_ref
        Cl = l_b / q / s_ref / b_ref
        Cm = m_b / q / s_ref / c_ref
        Cn = n_b / q / s_ref / b_ref

        return {
            "centroid": centroid_geometry,
            "F_g": force_geometry,
            "F_b": force_body,
            "F_w": force_wind,
            "M_g": moment_geometry,
            "M_b": moment_body,
            "M_w": moment_wind,
            "L"  : L,
            "D"  : D,
            "Y"  : Y,
            "l_b": l_b,
            "m_b": m_b,
            "n_b": n_b,
            "CL" : CL,
            "CD" : CD,
            "CY" : CY,
            "Cl" : Cl,
            "Cm" : Cm,
            "Cn" : Cn,
        }