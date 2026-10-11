# -*- coding: utf-8 -*-
"""
SAILBOAT DESCRIPTION

Created 19/04/2023
Last update: 22/10/2024

@author: Jules Richeux
@contributors: -

"""

import copy

from typing import List, Dict, Union, Optional, Tuple

from archibald.geometry.hull import Hull, DEFAULT_WINDAGE_METHODS
from archibald.geometry.planform import Rig, Appendage, Planform
from archibald.geometry.propeller import Propeller, BSeriesPropeller
from archibald.geometry.mesh import ArchibaldMesh

from archibald.performance.operating_point import OperatingPoint

from archibald.dynamics.aero_3D import HydroVortexLatticeMethod, AeroVortexLatticeMethod
from archibald.dynamics.hydro import multihull

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.common import ArchibaldObject

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))

#%%

class Sailboat(ArchibaldObject):
    
    def __init__(
            self,
            name: Optional[str] = None,
            xyz_ref: Union[np.ndarray, List] = None,
            displacement: Union[np.ndarray, float] = None,
            cog: Union[np.ndarray, List] = None,
            hulls: List[Hull] = [],
            rigs: List[Rig] = [],
            appendages: List[Appendage] = [],
            propellers: List[Propeller] = [],
            fittings: List[ArchibaldMesh] = [],
        ):
        
        self.hulls = hulls
        self.rigs = rigs
        self.appendages = appendages
        self.propellers = propellers
        self.fittings = fittings
        self.displacement = displacement
        self.cog = wide(np.array(cog))
        
        self.forces = {}
        self.moments = {}
        self.hull_interaction = None
        self.windage_components = {}
        
    def compute_aerodynamics(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        pass
    
    def compute_hydrodynamics(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        pass
        
    def compute_weight(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        
        # TODO: bring back cog in underway axes AND transform it with the rest
        
        weight = self.displacement
        g = op_point.environment.gravity
        
        center = op_point.apply_transformations(wide(self.cog))
        
        Fw = wide(np.array([
            0.,
            0.,
            -weight*g
        ]))
        
        Mw = np.cross(center, Fw)
        
        self.forces["Fw"] = Fw
        self.moments["Mw"] = Mw
        
        return Fw, Mw
    
    def compute_buoyancy(
            self,
            op_point: OperatingPoint,
            recompute_statics: bool = True,
        ):
        self.forces["Fb"] = wide(np.zeros(3))
        self.moments["Mb"] = wide(np.zeros(3))
        
        for i, hull in enumerate(self.hulls):
            Fb_i, Mb_i = hull.compute_buoyancy(
                op_point=op_point,
                recompute_statics=recompute_statics,
            )
            self.forces[f"Fb_{i}"] = Fb_i
            self.moments[f"Mb_{i}"] = Mb_i
            
            self.forces["Fb"] += Fb_i
            self.moments["Mb"] += Mb_i
            
            
    def compute_hull_interaction(
            self,
            op_point: OperatingPoint,
            method: Union[str, Dict] = "michell",
            positions: Optional[List] = None,
            **kwargs,
        ):
        """
        Interaction resistance between the hulls of the boat.
        
        The hydrostatics of every hull must have been computed beforehand
        (``compute_buoyancy`` or ``hull.compute_hydrostatics_properties``) and,
        for the calibrated / tabulated methods, so must the individual
        resistances (``hull.compute_resistance``).
        
        The interference of each hull is applied as a drag force along -x at the
        hull centre of water, as done in ``Hull.compute_resistance``.
        
        Parameters
        ----------
        op_point : OperatingPoint
        method : str or dict, key of ``multihull.INTERACTION_METHODS`` or custom dict.
        positions : list of (x, y) or None, hull positions, default from ``hull.get_position``.
        **kwargs : method parameters (``calibrate``, ``fn_grid``, ``tau_table``...).
        
        Returns
        -------
        dict, ``dRw`` (n x n), ``dRf``, ``dR_hull`` and ``dR`` [N].
        """
        self.hull_interaction = multihull.compute_hull_interaction(
            self.hulls,
            op_point,
            method=method,
            positions=positions,
            **kwargs,
        )
        
        self.forces["Fi"] = wide(np.zeros(3))
        self.moments["Mi"] = wide(np.zeros(3))
        
        for i, hull in enumerate(self.hulls):
            center = op_point.apply_transformations(hull.hydrostatics_data['cow'])
            
            Fi_i = wide(np.array([
                -1.,
                0.,
                0.,
            ])) * self.hull_interaction["dR_hull"][i]
            Mi_i = np.cross(center, Fi_i)
            
            self.forces[f"Fi_{i}"] = Fi_i
            self.moments[f"Mi_{i}"] = Mi_i
            
            self.forces["Fi"] += Fi_i
            self.moments["Mi"] += Mi_i
            
        return self.hull_interaction
            
            
    def compute_resistance(
            self,
            op_point: OperatingPoint,
            method: str,
            recompute_statics: bool = True,
            interaction: Optional[Union[str, Dict]] = None,
            interaction_kwargs: Optional[Dict] = None,
            **kwargs,
        ):
        """
        Resistance of the hulls, computed individually, plus their interaction.
        
        Parameters
        ----------
        method : str or dict, individual resistance method (see ``Hull.compute_resistance``).
        interaction : str, dict or None, interaction method between hulls (see
            ``compute_hull_interaction``). None: hulls are independent.
        interaction_kwargs : dict, parameters of the interaction method.
        **kwargs : parameters of the individual resistance method.
        """
        self.forces["Fh"] = wide(np.zeros(3))
        self.moments["Mh"] = wide(np.zeros(3))
        self.hull_interaction = None
        
        for i, hull in enumerate(self.hulls):
            Fb_i, Mb_i = hull.compute_resistance(
                op_point=op_point,
                method=method,
                recompute_statics=recompute_statics,
                **kwargs,
            )
            self.forces[f"Fh_{i}"] = Fb_i
            self.moments[f"Mh_{i}"] = Mb_i
            
            self.forces["Fh"] += Fb_i
            self.moments["Mh"] += Mb_i
            
        if interaction is not None and len(self.hulls) > 1:
            self.compute_hull_interaction(
                op_point=op_point,
                method=interaction,
                **(interaction_kwargs or {}),
            )
            self.forces["Fh"] += self.forces["Fi"]
            self.moments["Mh"] += self.moments["Mi"]
            
        
    def compute_windage(
            self,
            op_point: OperatingPoint,
            tws: float,
            twa: float,
            method: Union[str, Dict] = "orc",
            recompute_statics: bool = True,
            crew_hull: int = 0,
            crew_position: Optional[Union[np.ndarray, List]] = None,
            **kwargs,
        ):
        """
        Aerodynamic drag of the above-water part of the hulls, and of the crew.
        
        The hull elements of the method are evaluated for every hull. The element
        named "Crew" is not related to a hull, so it is evaluated only once for the
        whole boat, with the aerostatics of the hull ``crew_hull``.
        
        The aerostatics of every hull are (re)computed unless ``recompute_statics``
        is False. The hydrostatics must have been computed beforehand
        (``compute_buoyancy``) for the methods requiring ``Cwp``.
        
        The rig is not included: its aerodynamics is the purpose of ``compute_aerodynamics``.
        
        Parameters
        ----------
        op_point : OperatingPoint
        tws : float, True wind speed [kt].
        twa : float, True wind angle [deg], positive when the wind comes from starboard.
        method : str or dict, key of ``hull.DEFAULT_WINDAGE_METHODS`` or custom dict of methods.
        recompute_statics : bool, whether to recompute the aerostatics of the hulls.
        crew_hull : int, index of the hull whose geometry (freeboard, beam) gives the
            default height of the crew. Use ``zce_crew`` to set it directly.
        crew_position : (1, 3) or (3,) array or None, position of the crew in the boat
            frame. Default is the horizontal position of the lateral centre of effort
            of ``crew_hull``.
        **kwargs : parameters of the windage methods (crew_mass, n_crew, ship_type, ...).
        
        Returns
        -------
        Fa, Ma : (1, 3) arrays, total force [N] and moment [N.m] in the water axes.
        """
        if len(self.hulls) == 0:
            raise ValueError("The boat has no hull.")
        
        if type(method) == str and method in DEFAULT_WINDAGE_METHODS.keys():
            process = DEFAULT_WINDAGE_METHODS[method]
        elif type(method) == dict:
            process = method
        elif type(method) == str:
            raise NotImplementedError(
                f"Unknown method '{method}'. Should be in {list(DEFAULT_WINDAGE_METHODS.keys())} or a custom dictionary of methods.")
        else:
            raise TypeError("method argument should be str or dict")
        
        hull_process = {k: f for k, f in process.items() if k != "Crew"}
        crew_process = {k: f for k, f in process.items() if k == "Crew"}
        
        # clear the results of a previous call, e.g. with a different method
        for key in [k for k in self.forces if k.startswith("Fwd_")]:
            del self.forces[key]
        for key in [k for k in self.moments if k.startswith("Mwd_")]:
            del self.moments[key]
        
        self.forces["Fwd"] = wide(np.zeros(3))
        self.moments["Mwd"] = wide(np.zeros(3))
        self.windage_components = {}
        
        for i, hull in enumerate(self.hulls):
            Fa_i, Ma_i = hull.compute_windage(
                op_point=op_point,
                tws=tws,
                twa=twa,
                method=hull_process,
                recompute_statics=recompute_statics,
                **kwargs,
            )
            self.forces[f"Fwd_{i}"] = Fa_i
            self.moments[f"Mwd_{i}"] = Ma_i
            self.windage_components[f"hull_{i}"] = hull.windage_components
            
            self.forces["Fwd"] = self.forces["Fwd"] + Fa_i
            self.moments["Mwd"] = self.moments["Mwd"] + Ma_i
            
        if len(crew_process) > 0:
            # aerostatics of the hulls have just been computed above
            Fa_c, Ma_c = self.hulls[crew_hull].compute_windage(
                op_point=op_point,
                tws=tws,
                twa=twa,
                method=crew_process,
                recompute_statics=False,
                position=crew_position,
                **kwargs,
            )
            self.forces["Fwd_crew"] = Fa_c
            self.moments["Mwd_crew"] = Ma_c
            self.windage_components["crew"] = self.hulls[crew_hull].windage_components
            
            self.forces["Fwd"] = self.forces["Fwd"] + Fa_c
            self.moments["Mwd"] = self.moments["Mwd"] + Ma_c
        
        return self.forces["Fwd"], self.moments["Mwd"]
        
        
    def compute_torsor(
            self,
            op_point: OperatingPoint,
            method: str,
            recompute_statics: bool = True,
            interaction: Optional[Union[str, Dict]] = None,
            interaction_kwargs: Optional[Dict] = None,
            windage: Optional[Union[str, Dict]] = None,
            windage_kwargs: Optional[Dict] = None,
            **kwargs,
        ):
        """
        Total force and moment on the boat.
        
        Parameters
        ----------
        windage : str, dict or None, windage method of the hulls and crew (see
            ``compute_windage``). None: no windage.
        windage_kwargs : dict, parameters of the windage method, including the
            wind (``tws`` [kt] and ``twa`` [deg]).
        **kwargs : parameters of the individual resistance method.
        """
        self.compute_aerodynamics(op_point)
        self.compute_hydrodynamics(op_point)
        self.compute_weight(op_point)
        self.compute_buoyancy(op_point)
        self.compute_resistance(
            op_point=op_point,
            method=method,
            recompute_statics=recompute_statics,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            **kwargs,
        )
        
        if windage is not None:
            self.compute_windage(
                op_point=op_point,
                method=windage,
                recompute_statics=recompute_statics,
                **(windage_kwargs or {}),
            )
        else:
            self.forces["Fwd"] = wide(np.zeros(3))
            self.moments["Mwd"] = wide(np.zeros(3))
        
        self.forces["Ftot"] = self.forces["Fb"] + self.forces["Fw"] + self.forces["Fh"] + self.forces["Fwd"]
        self.moments["Mtot"] = self.moments["Mb"] + self.moments["Mw"] + self.moments["Mh"] + self.moments["Mwd"]
        
        return self.forces["Ftot"], self.moments["Mtot"]
    
    
    def draw(
        self,
        backend: str = "pyvista",
        thin_wings: bool = False,
        mesh_color: str = "lightgrey",
        show: bool = True,
        show_kwargs: Dict = None,
        **kwargs,
    ):
        """
        Visualizes the sailboat by aggregating all wings, appendages and hulls
        into a single scene.
        """
        if show_kwargs is None:
            show_kwargs = {}
    
        # =========================================================================
        # PYVISTA
        # =========================================================================
        if backend == "pyvista":
            import pyvista as pv
            import archibald.toolbox.mesh_utils as mesh_utils
    
            plotter = pv.Plotter()
    
            # --- Rigs ---
            for rig in self.rigs:
                p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
                mesh = pv.PolyData(
                    *mesh_utils.convert_mesh_to_polydata_format(p, f)
                )
                plotter.add_mesh(
                    mesh,
                    color=mesh_color,
                    show_edges=True,
                )
    
            # --- Appendages ---
            for app in self.appendages:
                p, f = app.mesh_body(method="quad", thin_wings=thin_wings)
                mesh = pv.PolyData(
                    *mesh_utils.convert_mesh_to_polydata_format(p, f)
                )
                plotter.add_mesh(
                    mesh,
                    color=mesh_color,
                    show_edges=True,
                )
    
            # --- Hulls ---
            for hull in self.hulls:
                hull.draw(
                    backend="pyvista",
                    show=False,
                    plotter=plotter,
                    **kwargs,
                )
                
            # --- Fittings ---
            for fitting in self.fittings:
                fitting.draw(
                    backend="pyvista",
                    show=False,
                    plotter=plotter,
                    **kwargs,
                )
    
            if show:
                plotter.show(**show_kwargs)
    
            return plotter
    
        # =========================================================================
        # PLOTLY
        # =========================================================================
        elif backend == "plotly":
            import plotly.graph_objects as go
            import plotly.offline as pyo
    
            fig = go.Figure()
    
            # --- Rigs ---
            for rig in self.rigs:
                p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
    
                # Split quads into triangles if necessary
                if f.shape[1] == 4:
                    faces = np.vstack([
                        f[:, [0, 1, 2]],
                        f[:, [0, 2, 3]],
                    ])
                else:
                    faces = f
    
                i, j, k = faces.T
    
                fig.add_trace(
                    go.Mesh3d(
                        x=p[:, 0],
                        y=p[:, 1],
                        z=p[:, 2],
                        i=i,
                        j=j,
                        k=k,
                        color=mesh_color,
                        opacity=1.0,
                        flatshading=True,
                    )
                )
    
            # --- Appendages ---
            for app in self.appendages:
                p, f = app.mesh_body(method="quad", thin_wings=thin_wings)
    
                if f.shape[1] == 4:
                    faces = np.vstack([
                        f[:, [0, 1, 2]],
                        f[:, [0, 2, 3]],
                    ])
                else:
                    faces = f
    
                i, j, k = faces.T
    
                fig.add_trace(
                    go.Mesh3d(
                        x=p[:, 0],
                        y=p[:, 1],
                        z=p[:, 2],
                        i=i,
                        j=j,
                        k=k,
                        color=mesh_color,
                        opacity=1.0,
                        flatshading=True,
                    )
                )
    
            # --- Hulls ---
            for hull in self.hulls:
                hull.draw(
                    backend="plotly",
                    show=False,
                    fig=fig,
                    **kwargs,
                )
                
            # --- Fittings ---
            for fitting in self.fittings:
                fitting.draw(
                    backend="plotly",
                    show=False,
                    fig=fig,
                    **kwargs,
                )
    
            fig.update_layout(
                scene=dict(
                    aspectmode="data",
                )
            )
    
            if show:
                pyo.plot(fig, auto_open=True)
    
            return fig
    
        # =========================================================================
        # MATPLOTLIB
        # =========================================================================
        elif backend == "matplotlib":
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
    
            vertices = []
    
            # --- Rigs ---
            for rig in self.rigs:
                p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
    
                polys = [p[face] for face in f]
    
                ax.add_collection3d(
                    Poly3DCollection(
                        polys,
                        facecolor=mesh_color,
                        edgecolor="k",
                        linewidth=0.3,
                    )
                )
    
                vertices.append(p)
    
            # --- Appendages ---
            for app in self.appendages:
                p, f = app.mesh_body(method="quad", thin_wings=thin_wings)
    
                polys = [p[face] for face in f]
    
                ax.add_collection3d(
                    Poly3DCollection(
                        polys,
                        facecolor=mesh_color,
                        edgecolor="k",
                        linewidth=0.3,
                    )
                )
    
                vertices.append(p)
    
            # --- Hulls ---
            for hull in self.hulls:
                hull.draw(
                    backend="matplotlib",
                    show=False,
                    ax=ax,
                    **kwargs,
                )
                
            # --- Fittings ---
            for fitting in self.fittings:
                fitting.draw(
                    backend="matplotlib",
                    show=False,
                    ax=ax,
                    **kwargs,
                )
    
            # Equal scaling
            if len(vertices):
                xyz = np.vstack(vertices)
    
                mins = xyz.min(axis=0)
                maxs = xyz.max(axis=0)
    
                center = 0.5 * (mins + maxs)
                radius = 0.5 * np.max(maxs - mins)
    
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
    
            ax.set_box_aspect((1, 1, 1))
    
            if show:
                plt.show()
    
            return ax
    
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                "Available backends are 'pyvista', 'plotly' and 'matplotlib'."
            )
    
    
    def _get_target_residual(
            self,
            target_name: str,
            Ftot: np.ndarray,
            Mtot: np.ndarray
        ):
        """Helper to map string targets to actual torsor arrays."""
        target_map = {
            'Fx': Ftot[0], 'Fy': Ftot[1], 'Fz': Ftot[2],
            'Mx': Mtot[0], 'My': Mtot[1], 'Mz': Mtot[2]
        }
        if target_name not in target_map:
            raise ValueError(f"Target '{target_name}' not recognized. Use one of: {list(target_map.keys())}")
        return target_map[target_name]


    def find_equilibrium(
            self,
            initial_op: OperatingPoint,
            free_variables: List[str] = ["dz", "heel", "trim"],
            targets: List[str] = ["Fz", "Mx", "My"],
            additional_constraints: Optional[List[callable]] = None,
            objective: Optional[callable] = None,
            verbose: bool = True
        ) -> Dict[str, any]:
        """
        Solves the static equilibrium of the sailboat.
        
        Parameters:
        -----------
        initial_op : OperatingPoint
            The starting operating point containing initial guesses and fixed parameters.
        free_variables : List[str]
            List of attributes on the OperatingPoint that the solver is allowed to vary.
        targets : List[str]
            List of forces/moments that must equal zero (e.g., ['Fz', 'Mx', 'My']).
        additional_constraints : List[callable], optional
            A list of functions: f(opti, Ftot, Mtot, vars_dict) -> constraint_expression.
        objective : callable, optional
            A function to minimize: f(opti, Ftot, Mtot, vars_dict) -> objective_expression.
            
        Returns:
        --------
        Dict containing the converged OperatingPoint, and the resulting forces/moments.
        """
        from archibald.optimization import Opti
        opti = Opti()
        
        # 1. Dynamically build the solver's OperatingPoint
        solver_kwargs = {}
        vars_dict = {}
        
        # Assuming OperatingPoint exposes its attributes (or defaults to 0)
        standard_attributes = ["dx", "dy", "dz", "heel", "trim", "leeway"]
        
        for attr in standard_attributes:
            init_val = getattr(initial_op, attr, 0.0)
            
            if attr in free_variables:
                var = opti.variable(init_guess=init_val)
                solver_kwargs[attr] = var
                vars_dict[attr] = var
            else:
                param = opti.parameter(init_val)
                solver_kwargs[attr] = param
                vars_dict[attr] = param
                
        solver_op = OperatingPoint(**solver_kwargs)
        
        # 2. Compute Torsor with symbolic/optimization variables
        Ftot, Mtot = self.compute_torsor(solver_op)
        
        # 3. Apply equilibrium constraints (Residuals == 0)
        for target in targets:
            residual = self._get_target_residual(target, Ftot, Mtot)
            opti.subject_to(residual == 0)
            
        # 4. Apply custom constraints
        if additional_constraints is not None:
            for constraint_func in additional_constraints:
                opti.subject_to(constraint_func(opti, Ftot, Mtot, vars_dict))
                
        # 5. Apply objective function
        if objective is not None:
            opti.minimize(objective(opti, Ftot, Mtot, vars_dict))
            
        # 6. Solve the system
        try:
            sol = opti.solve()
        except RuntimeError as e:
            if verbose: print(f"Solver failed to converge: {e}")
            raise
            
        # 7. Reconstruct the numerical OperatingPoint and Results
        resolved_kwargs = {}
        for attr in standard_attributes:
            resolved_kwargs[attr] = float(sol(solver_kwargs[attr]))
            
        converged_op = OperatingPoint(**resolved_kwargs)
        
        return {
            "converged_op": converged_op,
            "Ftot": np.array(sol(Ftot)).flatten(),
            "Mtot": np.array(sol(Mtot)).flatten(),
            "sol_object": sol, # Keep underlying sol object if user wants to query inner forces
            "success": True
        }