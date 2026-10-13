# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:38:18 2024

@author: jrich
"""
import casadi as ca
import archibald.numpy as np

from typing import Union, List
from archibald.common import ArchibaldObject
from archibald.performance import OperatingPoint
from archibald.toolbox.string_formatting import axis_string_to_array

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))

def smooth_clip(x, a, b, beta=10.0):
    r"""
    Smooth and differentiable clipping function bounded between [a, b] 
    using np.softplus, suitable for gradient-based optimization (IPOPT).
    
    Parameters
    ----------
    x : float or ndarray, Input value(s) to clip.
    a : float, Lower bound.
    b : float, Upper bound.
    beta : float, Steepness factor for the transition smoothing (higher = sharper).
    
    Returns
    -------
    float or ndarray, Smoothed and clipped values.
    """
    # 1. Smooth lower bound (smooth maximum with 'a')
    # Equivalent to: max(x, a) smoothly smoothed out
    x_lower = a + np.softplus(x - a, beta=beta)
    
    # 2. Smooth upper bound (smooth minimum with 'b')
    # Derived via symmetry: min(y, b) = -max(-y, -b)
    x_clipped = b - np.softplus(b - x_lower, beta=beta)
    
    return x_clipped

def complete_base_from_waterplane_normal(
        normal: Union[np.ndarray, List, str] = "z",
    ):
    """
    Compute the local ux and uy vectors from a waterplane normal, considered as uz, to complete the base.
    ux is taken in the global ZX plane to remain on the hull centerline.

    """
    if type(normal) is str:
        normal = axis_string_to_array(normal)
        
    # Normalize uz
    uz = np.array(normal, dtype=float)
    uz /= np.linalg.norm(uz)
    
    # ux is in the XZ-plane by construction: Y × uz gives a vector with no Y component
    # when uz has no Y component, and degrades gracefully otherwise
    ux = np.cross(np.array([[0.0, 1.0, 0.0]]), uz)
    ux /= np.linalg.norm(ux)
    
    # Complete the right-handed basis
    uy = np.cross(uz, ux)
    uy /= np.linalg.norm(uy)
    
    return wide(ux), wide(uy), wide(uz)


def wl_edge_intersection(
        Pa,
        Pb,
        da,
        db,
):
    """
    Compute the waterplane intersection point along the directed edge Pa→Pb.

    Parameters
    ----------
    Pa : (nF, 3) array — first endpoint of each edge
    Pb : (nF, 3) array — second endpoint of each edge
    da : (nF,)   array — signed distance of Pa to the waterplane (>0 wet)
    db : (nF,)   array — signed distance of Pb to the waterplane (>0 wet)

    Returns
    -------
    Q  : (nF, 3) array — intersection point (equals Pa when no crossing)
    """
    t = da / (da - db + 1e-12)              # (nF,)
    t = np.clip(t, 0., 1.)
    return Pa + tall(t) * (Pb - Pa)         # (nF, 3)


def wl_soft_crossing_weight(
        da,
        db,
        sharpness=1e3,
):
    """
    Soft indicator ≈ 1 when the edge crosses the waterplane, ≈ 0 otherwise.
    Differentiable everywhere.

    Parameters
    ----------
    da        : (nF,) signed distance of first endpoint  (>0 wet)
    db        : (nF,) signed distance of second endpoint (>0 wet)
    sharpness : steepness of the sigmoid transition

    Returns
    -------
    w : (nF,) soft crossing weight in [0, 1]
    """
    return (
          np.sigmoid( da * sharpness) * np.sigmoid(-db * sharpness)
        + np.sigmoid(-da * sharpness) * np.sigmoid( db * sharpness)
    )

def _weighted_slope(
        u,
        v,
        w,
):
    """
    Closed-form weighted least-squares slope for v = a*u + b.
    Returns a = cov(u,v) / var(u), both weighted.
    """
    W      = np.sum(w) + 1e-12
    u_mean = np.sum(w * u) / W
    v_mean = np.sum(w * v) / W
    cov_uv = np.sum(w * (u - u_mean) * (v - v_mean))
    var_u  = np.sum(w * (u - u_mean) ** 2) + 1e-12
    return cov_uv / var_u


def wl_soft_extremum(
        x,
        weights,
        alpha=100.,
        mode="max",
):
    """
    Differentiable soft maximum (or minimum) of x, masked by weights.
    Uses the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    x       : (N,) array of values
    weights : (N,) non-negative soft mask
    alpha   : sharpness — higher → closer to true max/min
    mode    : "max" or "min"

    Returns
    -------
    scalar soft extremum
    """
    sign = 1. if mode == "max" else -1.
    logits = sign * alpha * x + np.log(weights + 1e-12)
    logits_shifted = logits - np.max(logits)    # numerical stability
    exp_w = np.exp(logits_shifted)
    return np.sum(x * exp_w) / (np.sum(exp_w) + 1e-12)

def wl_intersection_points(
        mesh,
        point,
        normal,
        ux,
        uy,
        fdist,
):
    """
    Shared kernel: compute candidate waterplane intersection points for every
    face edge and project them onto the in-plane (ux, uy) frame.

    Called by all waterplane integral functions to avoid redundant computation.

    Parameters
    ----------
    mesh  : ArchibaldMesh
    point : (1, 3) waterplane reference point
    normal: (1, 3) waterplane normal (unit)
    ux    : (1, 3) longitudinal in-plane basis vector
    uy    : (1, 3) transverse  in-plane basis vector
    fdist : (nF, 3) signed distances of face vertices to the waterplane
            (positive = wet, i.e. below the waterplane)

    Returns
    -------
    u01, v01 : (nF,) in-plane coordinates of edge-01 intersection
    u12, v12 : (nF,) in-plane coordinates of edge-12 intersection
    u20, v20 : (nF,) in-plane coordinates of edge-20 intersection
    w01, w12, w20 : (nF,) soft crossing weights per edge
    Q01, Q12, Q20 : (nF, 3) 3-D intersection points
    """
    P  = mesh.vertices[mesh.faces]              # (nF, 3, 3)
    P0, P1, P2 = P[:, 0, :], P[:, 1, :], P[:, 2, :]   # each (nF, 3)

    d0 = fdist[:, 0]                            # (nF,)
    d1 = fdist[:, 1]
    d2 = fdist[:, 2]

    Q01 = wl_edge_intersection(P0, P1, d0, d1)  # (nF, 3)
    Q12 = wl_edge_intersection(P1, P2, d1, d2)
    Q20 = wl_edge_intersection(P2, P0, d2, d0)

    w01 = wl_soft_crossing_weight(d0, d1)       # (nF,)
    w12 = wl_soft_crossing_weight(d1, d2)
    w20 = wl_soft_crossing_weight(d2, d0)

    # Project onto in-plane frame.
    # ux, uy are (1, 3); Q__ are (nF, 3).
    # Q__ @ tall(ux) gives (nF, 1) — squeeze to (nF,).
    u01 = (Q01 @ tall(ux))            # (nF,)
    v01 = (Q01 @ tall(uy))
    u12 = (Q12 @ tall(ux))
    v12 = (Q12 @ tall(uy))
    u20 = (Q20 @ tall(ux))
    v20 = (Q20 @ tall(uy))

    return u01, v01, u12, v12, u20, v20, w01, w12, w20, Q01, Q12, Q20


def wl_extents(
        mesh,
        point,
        normal,
        ux,
        uy,
        fdist,
        alpha=50.,
):
    """
    Differentiable waterline length (Lwl), beam (Bwl), and fore/aft reference
    positions (u_fpp, u_app) computed via soft extrema over all waterplane
    intersection sample points.

    Parameters
    ----------
    mesh   : ArchibaldMesh
    point  : (1, 3) waterplane reference point
    normal : (1, 3) waterplane normal
    ux     : (1, 3) longitudinal in-plane basis vector
    uy     : (1, 3) transverse  in-plane basis vector
    fdist  : (nF, 3) signed vertex distances (positive = wet)
    alpha  : sharpness of soft extrema (higher → closer to true max/min)

    Returns
    -------
    Lwl   : scalar — waterline length
    Bwl   : scalar — waterline beam
    u_fpp : scalar — longitudinal coordinate of the fore perpendicular
    u_app : scalar — longitudinal coordinate of the aft  perpendicular
    u_all : (3*nF,) — pooled longitudinal coordinates of all sample points
    v_all : (3*nF,) — pooled transverse  coordinates of all sample points
    w_all : (3*nF,) — pooled soft crossing weights
    """
    u01, v01, u12, v12, u20, v20, w01, w12, w20, _, _, _ = \
        wl_intersection_points(mesh, point, normal, ux, uy, fdist)

    # Pool all candidate intersection points — shape (3*nF,)
    u_all = np.concatenate([u01, u12, u20], axis=0)
    v_all = np.concatenate([v01, v12, v20], axis=0)
    w_all = np.concatenate([w01, w12, w20], axis=0)

    # u_max = wl_soft_extremum(u_all, w_all, alpha, "max")
    # u_min = wl_soft_extremum(u_all, w_all, alpha, "min")
    # v_max = wl_soft_extremum(v_all, w_all, alpha, "max")
    # v_min = wl_soft_extremum(v_all, w_all, alpha, "min")
    
    # wl_extents — replace the four wl_soft_extremum calls:
    u_max = np.max(u_all)   # Lwl / u_fpp
    u_min = np.min(u_all)   # u_app
    v_max = np.max(v_all)   # Bwl
    v_min = np.min(v_all)

    Lwl   = u_max - u_min
    Bwl   = v_max - v_min
    u_fpp = u_max
    u_app = u_min

    return Lwl, Bwl, u_fpp, u_app, u_all, v_all, w_all

#%% CLASSES

class ArchibaldPolygon(ArchibaldObject):
    """
    Polygon description.
    
    Allows approximated though differentiable hydrostatics computations.
    
    """
    def __init__(
        self,
        vertices: Union[np.ndarray, List] = np.array([]),
        edges: List[int] = None
    ):
        
        if vertices is None or edges is None:
            vertices = np.zeros((2,3))
            # faces = wide(np.arange(2))
        
        self._vertices = vertices
        self._edges = edges
                
        self.reset_data() # Always needs to be called when vertices are modified in place
        
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def edges(self):
        return self._edges
    
    def reset_data(self):
        # Extract vertices of each face for later operations
        self._nV = self.vertices.shape[0]
        self._nE = self.edges.shape[0]
        self._c = np.mean(self.vertices, axis=0)
        
        self._v0 = self.vertices[self.edges[:, 0], :]
        self._v1 = self.vertices[self.edges[:, 1], :]
        
        self._v2 = np.tile(self._c, (self._nE, 1))
        
        self._data = {
            'cross_product': None,
            'triangle_areas': None,
            'triangle_centers': None,
            'edge_lengths': None,
            'edge_centers': None,
            'area': None,
            'area_centroid': None,
            'perimeter': None,
            'normals': None,
            'bounds': None,
        }
        
    def compute_cross_product(self):
        """
        Compute the cross products between the edges of all triangular faces.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        # Vectorized edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Cross product of edge vectors for all triangles at once
        self._data['cross_product'] = np.cross(edge1, edge2)
        
    def compute_triangle_normals(self):
        """
        Compute the normals of all triangular faces.

        """        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        self._data['normals'] = self._data['cross_product'] / 2.0
        
        
    def compute_edge_centers(self):
        """
        Compute the center of each polygon edge.

        """
        self._data['edge_centers'] = (self._v0 + self._v1) / 2.0
    
    
    def compute_edge_lengths(self):
        """
        Compute the length of each polygon edge.

        """
        v0, v1 = self._v0, self._v1
        
        self._data['edge_lengths'] = np.linalg.norm(v1 - v0, axis=1)
    

    def compute_perimeter(self):
        """
        Compute the polygon perimeter.

        """
        if self._data['edge_lengths'] is None:
            self.compute_edge_lengths()
            
        self._data['perimeter'] = np.sum(self._data['edge_lengths'])
    
    
    def compute_triangle_centers(self):
        """
        Compute the centers of all triangular faces.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        self._data['triangle_centers'] = (v0 + v1 + v2) / 3.0
        
        
    def compute_triangle_areas(self):
        """
        Compute the areas of all triangular faces.

        """        
        if self._data['normals'] is None:
            self.compute_triangle_normals()
            
        normals = self._data['normals']
        
        # Area of each triangular face
        self._data['triangle_areas'] = np.linalg.norm(normals + 1e-15, axis=1) # added a femto to avoid gradient error at x=0 for sqrt(0)
        
        
    def weighted_area(self, weight=1.):
        """
        Compute the mesh area. May be weighted.

        """
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        # Areas of each triangular faces
        tri_areas = self._data['triangle_areas'] * weight
        
        # Total area
        return np.sum(tri_areas)

    def weighted_area_centroid(self, weight=1.):
        """
        Compute the mesh volume centroid. May be weighted.

        """
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        tri_centers = self._data['triangle_centers']
        tri_areas = self._data['triangle_areas'] * weight
        
        # Weighted sum of centroids by volumes to get the total center of mass
        return np.sum(
            (wide(tri_areas) @ tri_centers / (np.sum(tri_areas) + 1e-12)),
            axis=0
        )
    
    @property
    def normals(self):
        if self._data['normals'] is None:
            self.compute_triangle_normals()
        
        return self._data['normals']
    
    @property
    def area(self):
        if self._data['area'] is None:
            self._data['area'] = self.weighted_area(weight=1.)
        
        return self._data['area']
    
    @property
    def area_centroid(self):
        if self._data['area_centroid'] is None:
            self._data['area_centroid'] = self.weighted_area_centroid(weight=1.)
        
        return self._data['area_centroid']
    
    @property
    def perimeter(self):
        if self._data['perimeter'] is None:
            self.compute_perimeter()
        
        return self._data['perimeter']
    
    def compute_bounds(self):
        """
        Compute the mesh bounding box in all 3 directions.

        """
        v = self.vertices
        
        x_min = np.min(v[:,0])
        x_max = np.max(v[:,0])
        y_min = np.min(v[:,1])
        y_max = np.max(v[:,1])
        z_min = np.min(v[:,2])
        z_max = np.max(v[:,2])
        
        self._data['bounds'] = np.array([
            [x_min, x_max],
            [y_min, y_max],
            [z_min, z_max],
        ])
        
    @property
    def bounds(self):
        """
        Return the mesh bounding box in all 3 directions.

        """
        if self._data['bounds'] is None:
            self.compute_bounds()
            
        return self._data['bounds']
    

    def draw(
            self,
            color = 'cyan',
            # mesh_color = 'grey',
            # plane_color = 'orange',
            # plane_opacity = 0.01,
            backend: str = 'pyvista',
            show: bool = True,
        ):
        
        if backend=="pyvista":
        
            import pyvista as pv
            
            # Create a PyVista plotter
            plotter = pv.Plotter()
                    
            for u, v in self.edges:
                # Extract endpoints
                pts = self.vertices[[u, v]]
                
                # Each line cell format: [number_of_points, id0, id1]
                # For 2-point line it's always [2, 0, 1]
                line = pv.PolyData(pts, lines=np.array([2, 0, 1]))
                
                # Add to plotter
                plotter.add_mesh(line, color=color, line_width=3)
                
                # Show axes triad and bounding box axes
            plotter.show_axes()            # 3D axes in corner
            plotter.show_grid()            # grid + bounding box
            
            # Display the plot
            if show:
                plotter.show()
            return plotter


class ArchibaldMesh(ArchibaldObject):
    """
    Differentiable mesh description. Vertices may be described by both numpy or CasADI objects.
    
    Allows approximated though differentiable hydrostatics computations.
    
    """
    
    # TODO: add a refine_mesh method to adapt coarser meshes
    
    def __init__(
        self,
        vertices: Union[np.ndarray, List] = np.array([]),
        faces: List[int] = None,
    ):
        
        if vertices is None or faces is None:
            vertices = np.zeros((3,3))
            faces = wide(np.arange(3))
            
        self._vertices = vertices
        self._faces = faces
                
        self.reset_data() # Always needs to be called when vertices are modified in place
        
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def faces(self):
        return self._faces
    
    @property
    def edges(self):
        if self._data['edges'] is None:
            self.compute_edges()
            
        return self._data['edges']
    
    @property
    def edges_lengths(self):
        if self._data['edges_lengths'] is None:
            self.compute_edges_lengths()
            
        return self._data['edges_lengths']
    
    @property
    def average_length(self):
        if self._data['average_length'] is None:
            self.compute_average_length()
            
        return self._data['average_length']
    
    @vertices.setter
    def vertices(self, value):
        # Check if value is a numpy array, CasADi MX, or CasADi DM
        if not (np.is_casadi_type(value) or type(value) == np.ndarray):
            raise ValueError(f"Vertices must be a numpy array or a CasADi array (MX or DM), not {type(value)}.")
        
        # Check if the shape matches the expected number of vertices and 3D coordinates
        if value.shape != (self.faces.max() + 1, 3):
            raise ValueError(f"Shape mismatch between vertices and faces. Vertices should have shape {(self.faces.max() + 1, 3)} to match the defined mesh topology.")
        
        # Assign the value and reset data
        self._vertices = value
        self.reset_data()
        
    def reset_data(self):
        # Extract vertices of each face for later operations
        self._v0 = self.vertices[self.faces[:, 0], :]
        self._v1 = self.vertices[self.faces[:, 1], :]
        self._v2 = self.vertices[self.faces[:, 2], :]
        
        self._data = {
            'cross_product': None,
            'edges': None,
            'edges_lengths': None,
            'triangle_centers': None,
            'triangle_areas': None,
            'tetrahedron_centers': None,
            'tetrahedron_volumes': None,
            'area': None,
            'area_centroid': None,
            'volume': None,
            'volume_centroid': None,
            'normals': None,
            'bounds': None,
            'average_length': None,
        }
        
    def compute_edges(self):
        # Generate all edges for each triangle (3 edges per face)
        edges = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ], axis=0)

        # Sort each edge so that [a, b] and [b, a] are treated the same
        edges = np.sort(edges, axis=1)
        
        # Remove duplicates
        self._data['edges'] = np.unique(edges, axis=0)
        
    def compute_edges_lengths(self):
        """
        Compute the length of each mesh edge.

        """
        if self._data['edges'] is None:
            self.compute_edges()
            
        # edges_pts = self.vertices[self.edges[:, 0]]
        v0, v1 = self.vertices[self.edges[:, 0]], self.vertices[self.edges[:, 1]]
        
        self._data['edges_lengths'] = np.linalg.norm(v1 - v0, axis=1)
        
    def compute_average_length(self):
        """
        Compute the average mesh edge length.

        """
        if self._data['edges_lengths'] is None:
            self.compute_edges_lengths()
        
        self._data['average_length'] = np.mean(self.edges_lengths)
        
    def compute_cross_product(self):
        """
        Compute the cross products between the edges of all triangular faces.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        # Vectorized edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Cross product of edge vectors for all triangles at once
        self._data['cross_product'] = np.cross(edge1, edge2)
        
    def compute_triangle_normals(self):
        """
        Compute the normals of all triangular faces.

        """        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        self._data['normals'] = self._data['cross_product'] / 2.0
        
        
    def compute_triangle_centers(self):
        """
        Compute the centers of all triangular faces.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        self._data['triangle_centers'] = (v0 + v1 + v2) / 3.0
        
        
    def compute_triangle_areas(self):
        """
        Compute the areas of all triangular faces.

        """        
        if self._data['normals'] is None:
            self.compute_triangle_normals()
            
        normals = self._data['normals']
        
        # Area of each triangular face
        self._data['triangle_areas'] = np.linalg.norm(normals + 1e-15, axis=1) # added a femto to avoid gradient error at x=0 for sqrt(0)
        
        
    def compute_tetrahedron_centers(self):
        """
        Compute the centers of all tetrahedrons formed by faces and origin.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        self._data['tetrahedron_centers'] = (v0 + v1 + v2) / 4.0
        
            
    def compute_tetrahedron_volumes(
        self,
        ref_point: Union[np.ndarray, List] = [0., 0., 0.],
    ):
        """
        Compute the signed volumes of all tetrahedrons formed by faces and origin.

        """
        if self._data['cross_product'] is None:
            self.compute_cross_product()
        
        # Signed volume of tetrahedron formed with origin for each face
        self._data['tetrahedron_volumes'] = np.sum(
            np.add(
                self._v0,
                -wide(ref_point)
            ) * self._data["cross_product"] / 6.0,
            axis=1
        )
    
    def vertices_distances_to_plane(
        self,
        point: Union[np.ndarray, List],
        normal: Union[np.ndarray, List, str],
    ):
        """
        Compute the oriented distances of each vertex to a given plane, represented by a point and a normal.

        """
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        # NB
        # if dist.all < 0 : fully immersed
        # if dist.all > 0 : fully emerged
        
        return np.add(
            self.vertices,
            -wide(point)
        ) @ tall(normal)
    
    def faces_distances_to_plane(
        self,
        point: Union[np.ndarray, List],
        normal: Union[np.ndarray, List, str],
    ):
        """
        Compute the oriented distances of each vertex to a given plane, represented by a point and a normal.

        """
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        # NB
        # if dist.all < 0 : fully immersed
        # if dist.all > 0 : fully emerged
        
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        
        return np.add(
            self._data['triangle_centers'],
            -wide(point)
        ) @ tall(normal)
    
    def frontal_area(
        self,
        direction: Union[str, np.ndarray],
        weight: Union[float, np.ndarray]=1.
    ):
        """
        Compute the frontal area of a mesh in a given direction.

        Parameters:
        - direction: str or (3,) array representing the frontal direction (str, numpy array or CasADi MX/SX).

        Returns:
        - frontal_area: Scalar representing the frontal area of the mesh.
        
        """
        if type(direction) is str:
            direction = axis_string_to_array(direction)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        cross_prod = self._data['cross_product'] # * weight
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Compute the product between the cosine of the angle between the triangle normal and the projection direction and the triangle area
        cos_theta_area = (cross_prod @ tall(direction))

        frontal_area = np.sum(np.fabs(cos_theta_area) * tall(weight)) / 4.0
        
        return frontal_area
        
    def frontal_area_centroid(
        self,
        direction: Union[str, np.ndarray],
        weight: Union[float, np.ndarray]=1.
    ):
        """
        Compute the centroid of the frontal area of a mesh in a given direction.
        
        Each face is located at its center and weighted by its contribution to the
        frontal area (same weights as in ``frontal_area``), so the result is the
        approximate centre of effort of the projected area.

        Parameters:
        - direction: str or (3,) array representing the frontal direction (str, numpy array or CasADi MX/SX).
        - weight: float or (n_faces,) array of face weights.

        Returns:
        - frontal_area_centroid: (1, 3) array.
        
        """
        if type(direction) is str:
            direction = axis_string_to_array(direction)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
            
        cross_prod = self._data['cross_product']
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)
        
        cos_theta_area = (cross_prod @ tall(direction))
        proj_areas = np.fabs(cos_theta_area) * tall(weight) / 4.0
        
        return wide(
            np.sum(
                self._data['triangle_centers'] * proj_areas / (np.sum(proj_areas) + 1e-12),
                axis=0
            )
        )
        
    def weighted_volume(
        self,
        weight: Union[np.ndarray, float] = 1.,
        ref_point: Union[np.ndarray, List] = [0., 0., 0.],
        recompute_tetrahedron_volumes: bool = False,
    ):
        """
        Compute the mesh volume. May be weighted.

        """
        if recompute_tetrahedron_volumes or self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes(ref_point)
        
        # Signed volume of tetrahedron formed with origin for each face
        tetra_volumes = tall(self._data['tetrahedron_volumes']) * tall(weight)
        
        # Total volume
        return np.sum(tetra_volumes)
        
    def weighted_volume_centroid(
        self,
        weight: Union[np.ndarray, float] = 1.,
        ref_point: Union[np.ndarray, List] = [0., 0., 0.],
        recompute_tetrahedron_volumes: bool = False,
    ):
        """
        Compute the mesh volume centroid. May be weighted.

        """
        if recompute_tetrahedron_volumes or self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes(ref_point)
            
        if self._data['tetrahedron_centers'] is None:
            self.compute_tetrahedron_centers()
        
        # compute_tetrahedron_centers() gives the centroid of the tetrahedra
        # whose apex is the origin: (v0 + v1 + v2) / 4. The volumes above are
        # signed with respect to ref_point, so the apex is ref_point and the
        # centroid is (ref_point + v0 + v1 + v2) / 4, hence the ref_point / 4 term.
        tetra_centers = np.add(self._data['tetrahedron_centers'], wide(ref_point) / 4.0)
        tetra_volumes = tall(self._data['tetrahedron_volumes']) * tall(weight)
        
        # Weighted sum of centroids by volumes to get the total center of mass
        # return np.sum(wide(tetra_volumes) @ tetra_centers / np.sum(tetra_volumes), axis=0)
    
        return np.sum(
            tetra_centers * tetra_volumes / (np.sum(tetra_volumes) + 1e-12),
            axis=0
        )
    
    def weighted_area(
        self,
        weight: Union[np.ndarray, float] = 1.,
    ):
        """
        Compute the mesh area. May be weighted.

        """
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        # Areas of each triangular faces
        tri_areas = tall(self._data['triangle_areas']) * tall(weight)
        
        # Total area
        return np.sum(tri_areas)

    def weighted_area_centroid(
            self,
            weight: Union[np.ndarray, float] = 1.,
        ):
        """
        Compute the mesh volume centroid. May be weighted.

        """
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        tri_centers = self._data['triangle_centers']
        tri_areas = tall(self._data['triangle_areas']) * tall(weight)
        
        # Weighted sum of centroids by volumes to get the total center of mass
        return np.sum(
            tri_centers * tri_areas / (np.sum(tri_areas) + 1e-12),
            axis=0
        )
    
    @property
    def volume(self):
        if self._data['volume'] is None:
            self._data['volume'] = self.weighted_volume(weight=1.)
        
        return self._data['volume']
    
    @property
    def volume_centroid(self):
        if self._data['volume_centroid'] is None:
            self._data['volume_centroid'] = self.weighted_volume_centroid(weight=1.)
        
        return self._data['volume_centroid']
    
    @property
    def area(self):
        if self._data['area'] is None:
            self._data['area'] = self.weighted_area(weight=1.)
        
        return self._data['area']
    
    @property
    def area_centroid(self):
        if self._data['area_centroid'] is None:
            self._data['area_centroid'] = self.weighted_area_centroid(weight=1.)
        
        return self._data['area_centroid']
    
    @property
    def normals(self):
        if self._data['normals'] is None:
            self.compute_triangle_normals()
        
        return self._data['normals']
    
    # @property
    # def area_Ixx(self):
    #     dist = self._data['triangle_centers'] - self.area_centroid
    #     Ixx = np.sum(
    #         self._data['triangle_areas'] * (dist[:,1]**2 + dist[:,2]**2)
    #     )
    #     return Ixx
    
    def compute_bounds(self):
        """
        Compute the mesh bounding box in all 3 directions.

        """
        v = self.vertices
        
        x_min = np.min(v[:,0])
        x_max = np.max(v[:,0])
        y_min = np.min(v[:,1])
        y_max = np.max(v[:,1])
        z_min = np.min(v[:,2])
        z_max = np.max(v[:,2])
        
        self._data['bounds'] = np.array([
            [x_min, x_max],
            [y_min, y_max],
            [z_min, z_max],
        ])
        
    @property
    def bounds(self):
        """
        Return the mesh bounding box in all 3 directions.

        """
        if self._data['bounds'] is None:
            self.compute_bounds()
            
        return self._data['bounds']     
    
    @property
    def n_vertices(self):
        return self.vertices.shape[0]
    
    @property
    def n_faces(self):
        return self.faces.shape[0]
    
    @property
    def n_edges(self):
        return self.edges.shape[0]
        
    def __repr__(self):
        
        return (
            f"{self.__class__.__name__} instance "+\
            f"with {self.n_vertices} vertices, "+\
            f"{self.n_edges} edges "+\
            f"and {self.n_faces} faces."
        )
    
    
    def _face_weights(
            self,
            point: Union[np.ndarray, List],
            normal: Union[np.ndarray, List],
        ):
        """
        Smooth wet weights of the faces with respect to a waterplane.

        Parameters
        ----------
        point  : (1, 3) reference point on the waterplane
        normal : (1, 3) waterplane outward normal

        Returns
        -------
        weights : (n_faces,) ~ 1 for wet faces, ~ 0 for dry faces
        vdist   : (n_vertices, 1) vertices signed distances, > 0 for wet, < 0 for dry
        fdist   : (n_faces, 3) vertices signed distances, shape similar to self.faces

        Shared by ``hydrostatics`` (weights) and ``aerostatics`` (1 - weights).

        """
        # vertices signed distances from the waterplane, shape similar to self.faces
        # > 0 for wet, < 0 for dry
        vdist = -self.vertices_distances_to_plane(
            point,
            normal
        )
        fdist = vdist[self.faces].reshape((self.n_faces, 3))
        
        # avg_dist = self.average_length
        avg_dist = 1.
        
        mix_weights = np.sigmoid(
            np.mean(
                fdist/avg_dist, 
                axis=1
            ) * 10./3.
            # ) * np.sqrt(10.)
        ) # smooth clipping weight
        # calibrated from the wet area variation of an equilateral triangle from
        # the average of its vertices signed distances
        
        highest_v = np.min(fdist, axis=1) # signed distance of the highest vertex of each face
        deepest_v = np.max(fdist, axis=1) # signed distance of the deepest vertex of each face
        
        corr = 1e3
        wet = (                                     # ≈ 1 for fully wet faces
            np.sigmoid(highest_v * corr)
            * np.sigmoid(deepest_v * corr)
        )
        dry = (                                     # ≈ 1 for fully dry faces
            np.sigmoid(-highest_v * corr)
            * np.sigmoid(-deepest_v * corr)
        )
        
        # weight = 1 for fully wet faces, 0 for fully dry faces,  
        weights = np.fmax(
            wet,
            mix_weights
        ) * (1-dry) 
        
        return weights, vdist, fdist
    
    def hydrostatics(
            self,
            point: Union[np.ndarray, List] = np.zeros(3),
            normal: Union[np.ndarray, List, str] = "z",
        ):
        """
        Compute hydrostatic quantities for the mesh at the given waterplane.

        Parameters
        ----------
        point            : (1, 3) or (3,) reference point on the waterplane
        normal           : (1, 3), (3,) or str — waterplane outward normal

        Returns
        -------
        dict with hydrostatics properties as its keys
            
        """
        
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        if self._data['tetrahedron_centers'] is None:
            self.compute_tetrahedron_centers()
            
        point = wide(np.array(point))
        normal = wide(np.array(normal))
            
        ux, uy, _ = complete_base_from_waterplane_normal(normal)
        
        weights, vdist, fdist = self._face_weights(point, normal)
        
        # compute tetrahedron volumes with reference point on the waterplane
        self.compute_tetrahedron_volumes(point)
        
        ### Submerged volume and centre of buoyancy
        
        # total volume
        volume = np.softplus(
            self.weighted_volume(
                weight=weights,
                ref_point=point,
                recompute_tetrahedron_volumes=False,
            ),
            beta=1e3,
        )
        volume = np.softplus(
            self.weighted_volume(
                weight=weights,
                ref_point=point,
                recompute_tetrahedron_volumes=False,
                
            ),
            beta=1e12,
        )
        
        # average weighted tetrahedron center i.e. center of buoyancy
        cob = self.weighted_volume_centroid(
            weight=weights,
            ref_point=point,
            recompute_tetrahedron_volumes=False,
        )
        
        ### Wetted surface area and centroid
        Aws = self.weighted_area(
            weight=weights
        )
        cow = wide(
            self.weighted_area_centroid(
                weight=weights
            )
        )
        
        ### Frontal areas
        Ax = self.frontal_area(
            direction=ux,
            weight=weights,
        )
        Ay = self.frontal_area(
            direction=uy,
            weight=weights,
        )
        
        ### Draft
        T = np.softplus(
            np.max(fdist),
            beta=1e3,
        )
        
        ### Waterline extents and FPP / APP
        
        ### Waterplane
        
        # Weight selection for waterplane vertices
        waterplane_vertices_weight = (
            np.sigmoid((vdist + self.average_length/2.) * 1e3)
            * np.sigmoid((-vdist + self.average_length/2.) * 1e3)
        )
        # Weight selection for waterplane faces
        waterplane_faces_weight = (
            np.sigmoid((weights + self.average_length/2.) * 1e3)
            * np.sigmoid((-weights + self.average_length/2.) * 1e3)
        )
        
        x_vdist = self.vertices_distances_to_plane(
            point=point,
            normal=ux,
        )
        y_vdist = self.vertices_distances_to_plane(
            point=point,
            normal=uy,
        )
        
        u_all = x_vdist * waterplane_vertices_weight
        v_all = y_vdist * waterplane_vertices_weight
        
        u_max = np.max(u_all)   # u_fpp
        u_min = np.min(u_all)   # u_app
        v_max = np.max(v_all)
        v_min = np.min(v_all)
        
        # Lwl, Bwl, u_fpp, u_app, u_wl, v_wl, w_wl = wl_extents(
        #     self,
        #     point,
        #     normal,
        #     ux,
        #     uy,
        #     fdist,
        #     alpha=soft_alpha,
        # )
        
        Lwl   = u_max - u_min
        Bwl   = v_max - v_min
        u_fpp = u_max
        u_app = u_min
        
        fpp_wp = point + u_fpp * ux # intersection between FPP and waterplane
        app_wp = point + u_app * ux # intersection between APP and waterplane
        
        dummy_cof = wide(self.weighted_area_centroid(weight=waterplane_faces_weight))
        
        scaled_vertices = np.add(
            np.add(self.vertices, -dummy_cof) * waterplane_vertices_weight,
            dummy_cof
        )
        
        waterplane_vertices = scaled_vertices - (
            (
                np.add(scaled_vertices, -point)
            ) @ normal.T
        ) @ normal
        
        wp = ArchibaldMesh(vertices=waterplane_vertices, faces=self.faces)
        self.wp = wp
        
        cof = wp.weighted_area_centroid(weight=1-waterplane_faces_weight)
        Awp = wp.frontal_area(direction=normal)
        
        ### Transom properties
        transom_vertices_weights = np.sigmoid(
            self.vertices_distances_to_plane(
            point = (app_wp + self.average_length * ux/2.), # point more or less 1/2 face forward of the aft-most waterplane point
            normal = -ux
        )*1e6) * vdist
        
        transom_faces_weights = np.sigmoid(
            self.faces_distances_to_plane(
            point = (app_wp + self.average_length * ux), # point more or less 1 face forward of the aft-most waterplane point
            normal = -ux
        )*1e6) * tall(weights)
        
        Ttr = np.max(transom_vertices_weights)
        Atr = self.frontal_area(direction=ux, weight=transom_faces_weights)
        
        ### Half entry angle
        
        fpp_vdist = self.vertices_distances_to_plane(
            point = fpp_wp,
            normal = ux
        )
        
        # Weight selection for bow vertices
        bow_vertices_weight = (
            np.sigmoid((fpp_vdist + self.average_length * 3.)*1e3)
            * np.sigmoid((-(fpp_vdist + self.average_length))*1e3)
        ) * waterplane_vertices_weight
        
        vectors_from_fpp = np.add(self.vertices, -fpp_wp) + tall(vdist) @ normal
        
        x = np.sum(vectors_from_fpp @ ux.T, axis=1)
        y = np.sum(vectors_from_fpp @ uy.T, axis=1)
        
        angles = tall(
            np.arctand(
                y/(x+1e-12)
            )
        )

        ie = np.sum(
            np.abs(angles)*tall(bow_vertices_weight),
        ) / (np.sum(bow_vertices_weight) + 1e-12)
        
        ### Dimensionless hull-form coefficients
        Cb  = volume / (Lwl * Bwl * T  + 1e-12)   # block coefficient
        Cp  = volume / (Ax  * Lwl      + 1e-12)   # prismatic coefficient
        Cx  = Ax     / (Bwl * T        + 1e-12)   # midship section coefficient
        Cy  = Ay     / (Lwl * T        + 1e-12)   # longitudinal plane coefficient
        Cwp = Awp    / (Lwl  * Bwl      + 1e-12)   # waterplane area coefficient
        
        eps = 1e-3
        beta = 1e3
        # Smoothly bounded coefficients for IPOPT stability
        Cb  = smooth_clip(Cb, eps, 1.0 - eps, beta=beta)  # block coefficient
        Cp  = smooth_clip(Cp, eps, 0.92, beta=beta)  # prismatic coefficient
        Cx  = smooth_clip(Cx, eps, 1.0 - eps, beta=beta)  # midship section coefficient
        Cy  = smooth_clip(Cy, eps, 1.0 - eps, beta=beta)  # longitudinal plane coefficient
        Cwp = smooth_clip(Cwp, eps, 1.0 - eps, beta=beta) # waterplane area coefficient
        
        ### Additional distances and lengths
        # u_fpp is measured from the waterplane reference point, so cob and cof
        # must be taken relative to it as well
        cob_u    = (np.add(wide(cob), -point) @ tall(ux))
        cof_u    = (np.add(wide(cof), -point) @ tall(ux))

        LCB_fpp  = u_fpp - cob_u # lcb from fpp / m
        LCF_fpp  = u_fpp - cof_u # lcf from fpp / m
        
        lcb = 1/2 - LCB_fpp/(Lwl + 1e-12) # longitudinal centre of buoyancy, as % of Lwl from midship
        
        h = {}
        h["volume"] = volume
        h["cob"] = wide(cob)
        h["Aws"] = Aws
        h["cow"] = cow
        h["cof"] = cof
        h["T"] = T
        h["Ttr"] = Ttr
        
        h["Lwl"] = Lwl
        h["Lbp"] = Lwl # approx : Lbp = Lwl
        h["Bwl"] = Bwl
        
        h["Ax"] = Ax
        h["Ay"] = Ay
        h["Atr"] = Atr
        h["Awp"] = Awp
        
        h["Cb"] = Cb
        h["Cp"] = Cp
        h["Cx"] = Cx
        h["Cy"] = Cy
        h["Cwp"] = Cwp
        
        h["ie"] = ie
        
        h["fpp"] = fpp_wp
        h["app"] = app_wp
        
        h["LCB_fpp"] = LCB_fpp
        h["LCF_fpp"] = LCF_fpp
        h["lcb"] = lcb
        
        # TODO refine computation for bulbous bow parameters
        h["Abt"] = 0.
        h["hB"] = T/2.
        
        return h
    
    
    def aerostatics(
            self,
            point: Union[np.ndarray, List] = np.zeros(3),
            normal: Union[np.ndarray, List, str] = "z",
        ):
        """
        Compute aerostatic quantities for the mesh above the given waterplane,
        i.e. the geometric properties of the emerged part (topsides, deck,
        superstructures) needed to evaluate its aerodynamic drag.
        
        Mirrors ``hydrostatics``: same waterplane definition, same smooth
        wet/dry weighting (dry weights = 1 - wet weights), and same key names
        whenever the quantity is the aerial counterpart of a hydrostatic one.
        
        Projected areas and their centroids are given along the waterplane axes
        (ux : forward, uy : transverse), so heel and trim of the operating point
        are already accounted for.

        Parameters
        ----------
        point            : (1, 3) or (3,) reference point on the waterplane
        normal           : (1, 3), (3,) or str — waterplane outward normal

        Returns
        -------
        dict with aerostatics properties as its keys. All centroids are (1, 3)
        arrays expressed in the mesh frame.
        
        Notes
        -----
        - ``volume`` and ``cov`` are meaningful only if the mesh is closed above
          the waterplane (deck included).
        - Projected areas use the same closed-surface convention as
          ``frontal_area``. They remain valid for front and side views of a mesh
          whose deck is left open (dinghy), but not for a top view.
        - No occlusion is modelled: concave shapes overestimate the projected areas.
            
        """
        
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        if self._data['tetrahedron_centers'] is None:
            self.compute_tetrahedron_centers()
            
        point = wide(np.array(point))
        normal = wide(np.array(normal))
            
        ux, uy, _ = complete_base_from_waterplane_normal(normal)
        
        weights, vdist, fdist = self._face_weights(point, normal)
        
        # weight = 1 for fully dry faces, 0 for fully wet faces
        dry_weights = 1. - weights
        
        ### Emerged volume and centroid
        self.compute_tetrahedron_volumes(point)
        
        volume = self.weighted_volume(
            weight=dry_weights,
            ref_point=point,
            recompute_tetrahedron_volumes=False,
        )
        
        cov = wide(
            self.weighted_volume_centroid(
                weight=dry_weights,
                ref_point=point,
                recompute_tetrahedron_volumes=False,
            )
        )
        
        ### Dry surface area and centroid
        Ads = self.weighted_area(
            weight=dry_weights
        )
        cod = wide(
            self.weighted_area_centroid(
                weight=dry_weights
            )
        )
        
        ### Frontal and lateral projected areas, and their centroids
        Ax = self.frontal_area(
            direction=ux,
            weight=dry_weights,
        )
        Ay = self.frontal_area(
            direction=uy,
            weight=dry_weights,
        )
        cex = self.frontal_area_centroid(
            direction=ux,
            weight=dry_weights,
        )
        cey = self.frontal_area_centroid(
            direction=uy,
            weight=dry_weights,
        )
        
        ### Air draft (highest point above the waterplane)
        H = np.softplus(
            -np.min(fdist),
            beta=1e3,
        )
        
        ### Overall extents of the emerged part
        # vertices up to half an edge below the waterplane are kept, to include the sheer line
        emerged_vertices_weight = np.sigmoid(
            (-vdist + self.average_length/2.) * 1e3
        )
        
        x_vdist = self.vertices_distances_to_plane(
            point=point,
            normal=ux,
        )
        y_vdist = self.vertices_distances_to_plane(
            point=point,
            normal=uy,
        )
        
        u_all = x_vdist * emerged_vertices_weight
        v_all = y_vdist * emerged_vertices_weight
        
        Loa = np.max(u_all) - np.min(u_all)
        Boa = np.max(v_all) - np.min(v_all)
        
        ### Equivalent mean freeboard, such as Ay = Fb * Loa
        Fb = Ay / (Loa + 1e-12)
        
        ### Dimensionless coefficients
        Cx = Ax / (Boa * H + 1e-12)   # frontal fullness coefficient
        Cy = Ay / (Loa * H + 1e-12)   # lateral fullness coefficient
        
        a = {}
        a["volume"] = volume
        a["cov"] = cov
        a["Ads"] = Ads
        a["cod"] = cod
        
        a["H"] = H
        a["Fb"] = Fb
        a["Loa"] = Loa
        a["Boa"] = Boa
        
        a["Ax"] = Ax
        a["Ay"] = Ay
        a["cex"] = cex
        a["cey"] = cey
        
        a["Cx"] = Cx
        a["Cy"] = Cy
        
        return a
    
    
    def slice_mesh(
            self,
            point,
            normal
        ):
        
        dist = self.vertices_distances_to_plane(point, normal)
        
        # edges = diff_mesh.edges
        edges_raw = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]],
        ], axis=0)
        
        edges = np.sort(edges_raw, axis=1)
        
        edge_dist = dist[edges]
        edge_prod = np.prod(edge_dist, axis=1)
        edge_sign = np.sign(edge_prod)
        edge_mask = ((1-edge_sign)/2)
        
        
        selected_edges = edges[edge_mask == 1.]
        
        if len(selected_edges) > 0:
        
            face_mask = np.sum(edge_mask.reshape((3,-1)).T, axis=1)//2 # sliced faces
            
            # Get coordinates of the selected edge endpoints
            v0 = self.vertices[selected_edges[:, 0]]  # shape: (n_crossing, 3)
            v1 = self.vertices[selected_edges[:, 1]]
            
            # Get distances of the endpoints to the plane
            d0 = edge_dist[edge_mask == 1.][:, 0]
            d1 = edge_dist[edge_mask == 1.][:, 1]
            
            # Compute interpolation factor t (clip to avoid numerical instability)
            t = d0 / (d0 - d1 + 1e-12)
            
            # Compute intersection points
            intersections = v0 + tall(t) * (v1 - v0)
        
            intersection_indices = (np.full(edges.shape[0], -1) +\
                np.cumsum(edge_mask)*edge_mask).reshape((3, -1)).T
            
            II = intersection_indices[np.where(face_mask, True, False)].astype(int)
            
            slice_edges = []
            
            for i in range(len(II)):
                I = II[i,:]
                
                shift = np.argwhere(I==-1)[0,0]
                i1 = (1 + shift) % 3
                i2 = (2 + shift) % 3
                
                slice_edges.append([I[i2], I[i1]])
                                   
            slice_edges = np.array(slice_edges)
            
            return ArchibaldPolygon(intersections, slice_edges)
        
        return ArchibaldPolygon(None, None)
    
    def transform(
            self,
            op_point: OperatingPoint,
            inverse: bool = False,
        ):
    
        self.vertices = op_point.apply_transformations(self.vertices, inverse=inverse)
            
    def draw(
        self,
        color = 'grey',
        opacity = 1.,
        show_edges = True,
        draw_plane = False,
        point = np.zeros(3),
        normal = np.array([0., 0., 1.]),
        mesh_color = 'grey',
        cmap = "RdYlGn",
        plane_color = 'blue',
        plane_opacity = 0.2,
        backend: str = 'pyvista',
        show: bool = True,
        plotter=None,
        fig=None,
        ax=None,
        set_axis_visibility = None,
    ):
        
        if backend == 'pyvista':
            import pyvista as pv
            
            # Create a PyVista plotter
            plotter = plotter if plotter is not None else pv.Plotter()
            
            # Create the mesh
            mesh = pv.PolyData(self.vertices, np.hstack([[3, *face] for face in self.faces]))
            
            if draw_plane:
                vdist = self.vertices_distances_to_plane(
                    point,
                    normal
                )
                weight = np.sigmoid(vdist * 10./3.)
                
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    scalars=weight,
                    cmap=cmap,
                )
                # Add the plane
                b = self.bounds
                diag = np.linalg.norm(b[:, 1] - b[:, 0]) * 1.2
                projected_centroid = (
                    self.area_centroid
                    - np.dot(
                        wide(self.area_centroid) - wide(point),
                        tall(normal),
                    )
                )
                plane = pv.Plane(
                    center=projected_centroid,
                    direction=normal,
                    i_size=diag, j_size=diag,
                )
                plotter.add_mesh(
                    plane,
                    color=plane_color,
                    opacity=plane_opacity
                )
            else:
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    color=color,
                    opacity=opacity,
                )
                
            if set_axis_visibility:
                plotter.add_axes()
                plotter.show_grid(color='gray')
            
            if show:
                # Display the plot
                plotter.show()
            return plotter
            
        elif backend == 'plotly':
            import plotly.graph_objects as go
            
            fig = fig if fig is not None else go.Figure()
            
            x, y, z = self.vertices.T
            i, j, k = self.faces.T
            
            if draw_plane:
                # Compute weights
                weight = np.sigmoid(
                    self.vertices_distances_to_plane(point, normal) * 10. / 3.
                )
                
                # Draw the main mesh mapped to the colorscale
                fig.add_trace(
                    go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        intensity=weight,
                        colorscale=cmap,
                        showscale=False,
                    )
                )
                
                # Calculate the plane boundaries
                b = self.bounds
                diag = np.linalg.norm(b[:, 1] - b[:, 0])
                projected_centroid = (
                    self.area_centroid
                    - np.dot(
                        wide(self.area_centroid) - wide(point),
                        tall(normal),
                    )
                ).flatten()
                
                if np.allclose(normal[:2], 0):
                    v1 = np.array([1., 0., 0.])
                else:
                    v1 = np.array([-normal[1], normal[0], 0.])
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 /= np.linalg.norm(v2)
                
                half_size = diag / 2.0 * 1.2
                p1 = projected_centroid - half_size * v1 - half_size * v2
                p2 = projected_centroid + half_size * v1 - half_size * v2
                p3 = projected_centroid + half_size * v1 + half_size * v2
                p4 = projected_centroid - half_size * v1 + half_size * v2
                
                # Draw the plane as two triangles (0,1,2 and 0,2,3)
                fig.add_trace(
                    go.Mesh3d(
                        x=[p1[0], p2[0], p3[0], p4[0]],
                        y=[p1[1], p2[1], p3[1], p4[1]],
                        z=[p1[2], p2[2], p3[2], p4[2]],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color=plane_color,
                        opacity=plane_opacity,
                    )
                )
            else:
                # Standard drawing without a plane
                fig.add_trace(
                    go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        opacity=opacity,
                        color=color,
                    )
                )
            
            # Setup scene aspect and axis visibility
            scene_layout = dict(aspectmode='data')
            
            if set_axis_visibility is False:
                scene_layout.update(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )
            elif set_axis_visibility is True:
                scene_layout.update(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True)
                )
                
            fig.update_layout(scene=scene_layout)

            if show:
                from plotly.offline import plot
                plot(fig)
            return fig
            
        elif backend == 'matplotlib':
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # 1. Setup or verify the 3D axis
            if ax is None:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                
            # Convert faces to a safe list of arrays for inhomogeneous shapes
            mesh_polys = [self.vertices[face] for face in self.faces]
            
            # Disable default shading to prevent immediate ValueError
            mesh_collection = Poly3DCollection(mesh_polys, shade=False)
            
            # 3. Handle Coloring and Shading Styles
            if draw_plane:
                weight = np.sigmoid(
                    self.vertices_distances_to_plane(point, normal) * 10. / 3.
                )
                # Compute face weights safely using list comprehension
                face_weights = np.array([np.mean(weight[face]) for face in self.faces])
                cmap_callable = plt.get_cmap(cmap)
                face_colors = cmap_callable(face_weights)
                mesh_collection.set_facecolor(face_colors)
            else:
                mesh_collection.set_facecolor(color)
                mesh_collection.set_alpha(opacity)
                
            if show_edges:
                mesh_collection.set_edgecolor('grey')
                mesh_collection.set_linewidth(0.5)
            else:
                mesh_collection.set_edgecolor('none')
                
            ax.add_collection3d(mesh_collection)
            
            # 4. Handle Waterplane Drawing (if requested)
            if draw_plane:
                b = self.bounds
                diag = np.linalg.norm(b[:, 1] - b[:, 0]) * 1.2
                projected_centroid = (
                    self.area_centroid
                    - np.dot(
                        wide(self.area_centroid) - wide(point),
                        tall(normal),
                    )
                )
                
                if np.allclose(normal[:2], 0):
                    v1 = np.array([1., 0., 0.])
                else:
                    v1 = np.array([-normal[1], normal[0], 0.])
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 /= np.linalg.norm(v2)
                
                half_size = diag / 2.0
                p1 = projected_centroid - half_size * v1 - half_size * v2
                p2 = projected_centroid + half_size * v1 - half_size * v2
                p3 = projected_centroid + half_size * v1 + half_size * v2
                p4 = projected_centroid - half_size * v1 + half_size * v2
                
                plane_collection = Poly3DCollection(
                    [p1, p2, p3, p4], 
                    facecolors=plane_color, 
                    alpha=plane_opacity,
                    shade=False
                )
                ax.add_collection3d(plane_collection)
            
            # 5. Apply view limits & visibility toggles requested by draw_three_view
            ax.auto_scale_xyz(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            
            # FORCE UNCONSTRAINED ASPECT RATIO
            ax.set_box_aspect(None)  # Resets any forced 1:1:1 box aspects
                
            # ax.auto_scale_xyz(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            
            if set_axis_visibility is False:
                ax.set_axis_off()
            elif set_axis_visibility is True:
                ax.set_axis_on()
            
            # set adequate zoom and scale
            b = self.bounds
            c = self.area_centroid
            main_dim = np.max(b[:, 1] - b[:, 0])
            
            ax.set_xlim(c[0] - main_dim/2., c[0] + main_dim/2.)
            ax.set_ylim(c[1] - main_dim/2., c[1] + main_dim/2.)
            ax.set_zlim(c[2] - main_dim/2., c[2] + main_dim/2.)
                
            if show:
                plt.show()
                
            return ax
            
        else:
            raise NotImplementedError(f'{backend} is not a supported drawing module.')