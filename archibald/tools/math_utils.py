# -*- coding: utf-8 -*-
"""
Created 04/07/2023
Last update: 10/09/2023

Useful math tools for archibald

@author: Jules Richeux
@university: ENSA Nantes, FRANCE

"""

#%% DEPENDENCIES

import csv
import numpy as np
import scipy.interpolate as itrp


#%% FUNCTIONS

def read_coefs(name, delim=' '):
    TAB = list()
    
    with open(name, 'r') as file:
        reader = csv.reader(file, delimiter = delim)
        for row in reader:
            TAB.append(row)
            
    return np.array(TAB)[:,1:].astype('float32')


def cosd(a):
    return np.cos(np.radians(a))


def sind(a):
    return np.sin(np.radians(a))


def tand(a):
    return sind(a)/cosd(a)


def rotation_matrix(vector, angle):
    """
    Computes the rotation matrix for given axis and angle.

    Parameters
    ----------
    vector ((3,) array): rotation axis
    angle (float): rotation angle in deg

    Returns
    -------
    (3,3) array: corresponding rotation matrix

    """
    
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)
    
    # Components of the vector
    x, y, z = vector
    
    # Compute the rotation matrix elements
    c = cosd(angle)
    s = sind(angle)
    t = 1 - c
    
    # Construct the rotation matrix
    rotation = np.array([[t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
                         [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
                         [t * x * z - s * y, t * y * z + s * x, t * z**2 + c]])
    
    return rotation


def rotate_x(vec3d, angle):
    """
    Rotates a 3D vector around the x-axis.

    Parameters
    ----------
    vec3d ((3,) array): vector to rotate
    angle (float): rotation angle in deg

    Returns
    -------
    (3,) array: rotated vector

    """
    
    rot_matrix = np.array([[1, 0, 0],
                           [0, cosd(angle), -sind(angle)],
                           [0, sind(angle), cosd(angle)]])
    
    return np.dot(rot_matrix, vec3d)


def set_normal(heel, trim):
    z0 = np.array([0,0,1])
    y0 = np.array([0,1,0])
    
    z1 = rotate_x(z0, heel)
    y1 = rotate_x(y0, heel)
    
    z2 = z1*cosd(trim) + np.cross(y1, z1)*sind(trim) + y1*np.dot(y1, z1)*(1 - cosd(trim))

    return -z2


def build_interpolation(coefs, method='cubic'):
    A = list()
    
    for a in range(1,len(coefs)):
        A.append(itrp.interp1d(coefs[0], coefs[a], kind=method))
        
    return A


def compute_AW(tws, twa, V):
    """
    Computes apparent wind from true wind.

    Parameters
    ----------
    tws : float. True wind speed in m/s
    twa : float. True wind angle in deg
    V : float. Boat speed in m/s

    Returns
    -------
    Apparent wind speed in m/s
    Apparent wind angle in deg

    """
    
    if type(V)!=int and type(V)!=np.float64 and type(V)!=float:
        V = V[0]
    
    TW = tws * np.array([cosd(twa), sind(twa)])
    SW = np.array([V, 0.])
    
    AW = TW + SW
    
    aws = np.linalg.norm(AW)
    awa = np.angle(AW[0]+1j*AW[1])
    
    return aws, np.rad2deg(awa)


def compute_TW(aws, awa, V):
    """
    Computes true wind from apparent wind.

    Parameters
    ----------
    tws : float. Apparent wind speed in m/s
    twa : float. Apparent wind angle in deg
    V : float. Boat speed in m/s

    Returns
    -------
    True wind speed in m/s
    True wind angle in deg

    """
    
    if type(V)!=int and type(V)!=np.float64 and type(V)!=float:
        V = V[0]
    
    AW = aws * np.array([cosd(awa), sind(awa)])
    SW = np.array([V, 0.])
    
    TW = AW - SW
    
    tws = np.linalg.norm(TW)
    twa = np.angle(TW[0]+1j*TW[1])
    
    return tws, np.rad2deg(twa)
