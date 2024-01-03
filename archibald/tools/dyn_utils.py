# -*- coding: utf-8 -*-
"""
Created 04/07/2023
Last update: 24/12/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -
"""

#%% DEPENDENCIES

import numpy as np
import scipy.interpolate as itrp


#%% FUNCTIONS

def Cf_hull(Re):
    """
    Computes the friction-drag coefficient of a bare hull
    following ITTC78
    https://www.ittc.info/media/8017/75-02-03-014.pdf

    Parameters
    ----------
    Re (float): Hull Reynolds number

    Returns
    -------
    float: Bare hull friction-drag coefficient

    """
    if Re < 0.0:
        print("Warning: Negative Reynolds number")
        return 0.0
    elif Re == 0.0:
        return 0.0
    
    return 0.075/((np.log10(Re) - 2)**2)


def Cd_cambered_plate(alpha, camber):
    """
    Interpolates the drag coefficient of a cambered plate
    following the data given by Glenn Research Center, NASA
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/foilsimstudent/#

    Parameters
    ----------
    alpha (float): Angle of attack of the plate in degrees. Should be between 0. and 20.
    camber (float): Relative camber of the plate. Should be between 0. and .15

    Returns
    -------
    float: Interpolated drag coefficient of the given plate in the given conditions.

    """
    
    aoas = np.array([0,5,10,15,20])
    cambers = np.array([0., .05, .10, .15])
    
    if alpha <= 0.0:
        alpha = 1e-8
    elif alpha >= 20.0:
        alpha = 20.0 - 1e-8
        
    if camber <= 0.0:
        camber = 1e-8
    elif alpha >= 0.15:
        camber = 0.15 - 1e-8
        
    values = np.array([[.0188, .0543, .0978, .2168],
                      [.0260, .0609, .1217, .3440],
                      [.0565, .0957, .1739, .5202],
                      [.2630, .1608, .2609, .6192],
                      [.2760, .2400, .3500, .4275]])

    values.reshape((20))

    f = itrp.RegularGridInterpolator((aoas, cambers), values, method='cubic')

    return f((alpha, camber))


# def Cf_airfoil(Re):
#     return 0.074/Re**0.2 - 1742/Re


def grad_wind(tws0, z, z0=10.0, a=0.10):
    """
    Computes true wind speed at a given height following : Heier, Siegfried (2005).
    Grid Integration of Wind Energy Conversion Systems. Chichester: John Wiley &
    Sons. p. 45. ISBN 978-0-470-86899-7.

    Parameters
    ----------
    tws0 (float): Reference true wind speed at z=z0
    z (float): Unit should be the same as z.
    z0 (float): Reference height. The default is 10.0 (in meters)
    a (float): Hellman exponent, governing the gradient intensity. The default is 0.10
    
    Other references (following "Renewable energy: technology, economics, and
                      environment" by Martin Kaltschmitt, Wolfgang Streicher,
                      Andreas Wiese, (Springer, 2007, ISBN 3-540-70947-9,
                      ISBN 978-3-540-70947-3), page 55) :
    Unstable air above open water surface : 0.06
    Neutral air above open water surface : 0.10
    Unstable air above flat open coast : 0.11
    Neutral air above flat open coast : 0.16
    Stable air above open water surface : 0.27
    Unstable air above human inhabited areas : 0.27
    Neutral air above human inhabited areas : 0.34
    Stable air above flat open coast : 0.40
    Stable air above human inhabited areas : 0.60

    Returns
    -------
    float: True wind speed at z. Unit is the same as tws0.
    
    """
    return tws0 * (z/z0)**a