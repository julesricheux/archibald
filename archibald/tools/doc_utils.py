# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:42:02 2023

@author: Jules
"""

#%% DEPENDENCIES

import numpy as np
import csv
import scipy.interpolate as itrp
import ezdxf


#%% FUNCTIONS

def read_coefs(name, delim=' '):
    TAB = list()
    
    with open(name, 'r') as file:
        reader = csv.reader(file, delimiter = delim)
        for row in reader:
            TAB.append(row)
            
    return np.array(TAB)[:,1:].astype('float32')


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
        bernstein_i_m = np.math.comb(m, i) * tt**i * (1-tt)**(m-i)
        
        points += ctrl[i] * bernstein_i_m
        
    return np.transpose(points)


def evaluate_curve(points, z_coordinate):
    
    closest_point_idx = np.argmin(np.abs(points[2] - z_coordinate))
    closest_point = (points[0][closest_point_idx], points[1][closest_point_idx], points[2][closest_point_idx])
    
    return closest_point


def spline_to_le_chords(leControl, leDeg, teControl, teDeg, nSections):
    
    leCurve = build_spline(leControl, leDeg)
    teCurve = build_spline(teControl, teDeg)
        
    minZ = np.max(np.min([leControl[:,2],teControl[:,2]]))
    maxZ = np.min(np.max([leControl[:,2],teControl[:,2]]))
    
    Z = np.linspace(minZ, maxZ, nSections)
    
    le = []
    chords = []
    
    for z in Z:
        lePt = evaluate_curve(leCurve, z)
        tePt = evaluate_curve(teCurve, z)
        
        le.append(lePt)
        chords.append(lePt[0]-tePt[0])
        
    return np.array(le), np.array(chords)


def bezier_to_le_chords(leControl, teControl, nSections):
    
    leCurve = build_bezier(leControl)
    teCurve = build_bezier(teControl)
        
    minZ = np.max(np.min([leControl[:,2],teControl[:,2]]))
    maxZ = np.min(np.max([leControl[:,2],teControl[:,2]]))
    
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
        if entity.dxftype() == 'SPLINE':
            splines.append(entity)
    
    if len(splines)==2:
        leControl = np.array(splines[0].control_points[:])
        teControl = np.array(splines[1].control_points[:])
        
        leDeg = splines[0].dxf.degree
        teDeg = splines[1].dxf.degree
        
        if np.sum(leControl[:,0]) <= np.sum(teControl[:,0]): # if le is behind te
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
            

def dxf_to_le_chords(filename, nSections, method='bezier'):
    
    leControl, leDeg, teControl, teDeg = read_dxf(filename, nSections, method)
    
    if leControl is None:
        print('Invalid DXF file')
        return None, None

    else:
        if method == 'spline':
            return spline_to_le_chords(leControl, leDeg, teControl, teDeg, nSections)
        elif method == 'bezier':
            return bezier_to_le_chords(leControl, teControl, nSections)
        else:
            print('Invalid method')