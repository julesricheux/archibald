# -*- coding: utf-8 -*-
"""
Created 04/07/2023
Last update: 03/01/2024

Useful geometrical tools for archibald

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

"""

#%% DEPENDENCIES

import numpy as np
import ezdxf
import math
import scipy.interpolate as itrp


#%% FUNCTIONS

_VALID_DXF_TYPES = [
    "POLYLINE",
    "SPLINE" ,
    "LINE",
    "LWPOLYLINE",
    "ARC",
]

def build_spline(control_points, degree, n=200):

    control_points = np.array(control_points)
    tck, u = itrp.splprep(control_points.T, k=degree)
    
    u = np.linspace(1, 0, n)
    points = itrp.splev(u, tck)
    
    return points


def build_bezier(ctrl, n=200):
    
    m = len(ctrl) - 1
    d = len(ctrl[0])
    
    t = np.linspace(0., 1., n)
    tt = np.vstack([t.T]*d).T
    
    points = np.zeros((n, d))
    
    for i in range(m+1):
        bernstein_i_m = math.comb(m, i) * tt**i * (1-tt)**(m-i)
        
        points += ctrl[i] * bernstein_i_m
        
    return np.transpose(points)


def evaluate_curve(points, z_coordinate):
    
    closest_point_idx = np.argmin(np.abs(points[2] - z_coordinate))
    closest_point = (points[0][closest_point_idx], points[1][closest_point_idx], points[2][closest_point_idx])
    
    return closest_point


def spline_to_le_chords(leControl, leDeg, teControl, teDeg, nSections, direction='Z'):
    
    leCurve = build_spline(leControl, leDeg)
    teCurve = build_spline(teControl, teDeg)
    
    if direction == 'Z':
        axis = 2
    elif direction == 'Y':
        axis = 1
    elif direction == 'X':
        axis = 0
    else:
        raise('Invalid axis')
        return None, None
        
    minZ = np.max([np.min(leControl[:,2]), np.min(teControl[:,2])])
    maxZ = np.min([np.max(leControl[:,2]), np.max(teControl[:,2])])
    
    Z = np.linspace(minZ, maxZ, nSections)
    
    le = []
    chords = []
    
    for z in Z:
        lePt = evaluate_curve(leCurve, z)
        tePt = evaluate_curve(teCurve, z)
        
        le.append(lePt)
        chords.append(lePt[0]-tePt[0])
        
    le, chords = np.array(le), np.array(chords)
    
    sortedIdx = np.argsort(le[:, axis])
    
    return le[sortedIdx], chords[sortedIdx]


def bezier_to_le_chords(leControl, teControl, nSections):
    
    leCurve = build_bezier(leControl)
    teCurve = build_bezier(teControl)
        
    minZ = np.max([np.min(leControl[:,2]), np.min(teControl[:,2])])
    maxZ = np.min([np.max(leControl[:,2]), np.max(teControl[:,2])])
    
    Z = np.linspace(minZ, maxZ, nSections)
    
    le = []
    chords = []
    
    for z in Z:
        lePt = evaluate_curve(leCurve, z)
        tePt = evaluate_curve(teCurve, z)
        
        le.append(lePt)
        chords.append(lePt[0]-tePt[0])
        
    return np.array(le), np.array(chords)


def read_dxf(filename, nSections, method='bezier'):
    
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    splines = []
    
    for entity in msp:
        if entity.dxftype() in _VALID_DXF_TYPES:
            splines.append(entity)
    
    if len(splines) == 2:
        
        def get_entity_data(entity):
            dtype = entity.dxftype()
            if dtype == 'SPLINE':
                control = np.array(entity.control_points[:])
                deg = entity.dxf.degree
            elif dtype == 'POLYLINE':
                control = np.array(list(entity.points()))
                deg = 1
            elif dtype == 'LWPOLYLINE':
                control = np.array(list(entity.vertices_in_wcs()))
                deg = 1
            elif dtype == 'ARC':
                # Convert the arc to a spline to preserve exact curvature data
                spline_ent = entity.to_spline(replace=False)
                control = np.array(spline_ent.control_points[:])
                deg = spline_ent.dxf.degree
            else:  # LINE or other entities with start/end attributes
                start = [entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z]
                end = [entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z]
                control = np.linspace(start, end, 3)
                deg = 2
            return control, deg

        leControl, leDeg = get_entity_data(splines[0])
        teControl, teDeg = get_entity_data(splines[1])
        
        if np.max(leControl[:, 0]) <= np.max(teControl[:, 0]): # if le is behind te
            # swap the edges
            tmpControl = leControl
            leControl = teControl
            teControl = tmpControl
            tmpDeg = leDeg
            leDeg = teDeg
            teDeg = tmpDeg
        
        return leControl, leDeg, teControl, teDeg
    
    else:
        return None, None, None, None
            

def dxf_to_le_chords(filename, nSections, method='bezier', direction='Z'):
    
    leControl, leDeg, teControl, teDeg = read_dxf(filename, nSections, method)
    
    if leControl is None:
        print('Invalid DXF file')
        return None, None

    else:
        if method == 'spline':
            return spline_to_le_chords(leControl, leDeg, teControl, teDeg, nSections, direction)
        elif method == 'bezier':
            return bezier_to_le_chords(leControl, teControl, nSections)
        else:
            print('Invalid method')


# DEPRECATED
def world_to_vlm(coords3D):
    return np.transpose(np.vstack([-coords3D[:,0], coords3D[:,2], coords3D[:,1]]))