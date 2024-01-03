# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:00:42 2023

@author: Jules
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

import warnings
import numpy as np
import scipy.interpolate as itrp
import scipy.optimize as opt

from archibald.hull import Hull
from archibald.environment import OffshoreEnvironment

from archibald.tools.math_utils import *
from archibald.tools.dyn_utils import *


## SAVISKY COMPUTATION DRAFT
# Needs to be investigated further to be integrated to the VPP.    


def get_hydrodynamic_force(V, Lk, Lc, beam, beta, trim, eta, environment=OffshoreEnvironment()):
    """Following Savitsky 1964.
    """
    
    rho = environment.water.rho
    g = environment.g
    
    #Beam Froude number
    Fn_B = V/np.sqrt(g*beam)
    
    lambda_W = (Lk + Lc) / (2*beam)
    
    #Warnings
    if Fn_B < 0.6 or Fn_B > 13:
        warnings.warn('Beam Froude number = {0:.3f}, outside of range of applicability (0.60 <= V/sqrt(g*b) <= 13.00) for planing lift equation. Results are extrapolations.'.format(Fn_B), stacklevel=2)
    if lambda_W > 4:
        warnings.warn('Mean wetted length-beam ratio = {0:.3f}, outside of range of applicability (lambda <= 4) for planing lift equation. Results are extrapolations.'.format(lambda_W), stacklevel=2)
    if (trim - eta) < 2 or (trim - eta) > 15:
        warnings.warn('Vessel trim = {0:.3f}, outside of range of applicability (2 deg <= trim <= 15 deg) for planing lift equation. Results are extrapolations.'.format(trim), stacklevel=2)

    #0-Deadrise lift coefficient
    C_L0 = (trim - eta)**1.1 * (0.012 * lambda_W**0.5 + 0.0055 * lambda_W**2.5 / Fn_B**2)

    #Lift coefficient with deadrise, C_Lbeta
    C_Lbeta = C_L0 - 0.0065 * beta * C_L0**0.6

    #Vertical force (lift)
    Ldyn = C_Lbeta * 0.5 * rho * V**2 * beam**2

    #Horizontal force
    Rdyn = Ldyn*tand(trim - eta)

    #Lift's Normal force w.r.t. keel
    Fdyn = Ldyn / cosd(trim - eta)

    #Longitudinal position of the center of pressure, l_p (Eq. 4.41, Doctors 1985)
    Xdyn = lambda_W * beam * (0.75 - 1 / (5.21 * (Fn_B / lambda_W)**2 + 2.39)) #Limits for this is (0.60 < Fn_B < 13.0, lambda < 4.0)
    
    lengths = {'Lk': Lk, 'Lc': Lc, 'Xp': Xdyn, 'lambda': lambda_W}
    
    #Vpdate values
    return Ldyn, Rdyn, Xdyn, lengths


def get_transom_resistance(V, Ttr, Atr, environment=OffshoreEnvironment()):
    
    rho = environment.water.rho
    g = environment.g
    
    # Transom additionnal resistance (Holtrop&Mennen, 1978)
    Fn_T = V / np.sqrt(g * Ttr)
    if Fn_T < 5:
        ctr = 0.2 * (1 - (0.2 * Fn_T))
    else:
        ctr = 0
        
    Rtr = 0.5 * rho * (V ** 2) * Atr * ctr
    
    return Rtr
    

def get_skin_friction(V, Lk, Lc, beam, beta, trim, eta, Wsa, Atr, environment=OffshoreEnvironment()):
    """This function outputs the frictional force of the vessel using ITTC 1957 and the Townsin 1985 roughness allowance.
    """

    rho = environment.water.rho
    nu = environment.water.nu
    g = environment.g
    
    #Beam Froude number
    Fn_B = V/np.sqrt(g*beam)
    
    lambda_W = (Lk + Lc) / (2*beam)
    
    #Warnings
    if Fn_B < 1.0 or Fn_B > 13:
        warnings.warn('Beam Froude number = {0:.3f}, outside of range of applicability (1.0 <= V/sqrt(g*b) <= 13.00) for average bottom velocity estimate. Results are extrapolations.'.format(Fn_B), stacklevel=2)

    #Mean bottom fluid velocity, Savitsky 1964 - derived to include deadrise effect
    V_m = V * np.sqrt(1 - (0.012 * trim**1.1 * np.sqrt(lambda_W) - 0.0065 * beta * (0.012 * np.sqrt(lambda_W) * trim**1.1)**0.6) / (lambda_W * cosd(trim)))
    
    S = Wsa - Atr
    
    #Surface area of the dry-chine region
    a1 = (Lk - Lc) * beam / (2 * cosd(beta)) 
    if Lk < (Lk - Lc):
        a1 = a1 * (Lk / (Lk - Lc))**2

    #Surface area of the wetted-chine region
    a2 = beam * Lc / cosd(beta)
        
    S1 = S * a1/(a1+a2)
    S2 = S * a2/(a1+a2)

    #Reynolds number (with bottom fluid velocity)
    Re = V_m * lambda_W * beam / nu

    #'Friction coefficient' ITTC 1957
    Cf = Cf_hull(Re)

    AHR = 1.5e-4 # roughness mean height (m)

    #Additional 'friction coefficient' due to skin friction, Townsin (1985) roughness allowance
    dCf = (44*((AHR/(lambda_W*beam))**(1/3) - 10*Re**(-1/3)) + 0.125)/10**3

    #Frictional force
    Ff = 0.5 * rho * (Cf + dCf) * S * V**2

    #Geometric vertical distance from keel
    Xf = (beam / 4 * tand(beta) * S1 + beam / 6 * tand(beta) * S1) / S

    #Horizontal force
    Rf = Ff * cosd(trim - eta)

    #Vertical force
    Lf = - Ff * sind(trim - eta)
        
    #Vpdate values
    return Lf, Rf, Xf


def get_forces_savitsky(z, hull, V, beta_func, heel, trim, eta):
    
        hull.compute_hydrostatics(z, heel, trim)
        
        Lk = hull.hydrostaticData['Lwl']
        beam = hull.hydrostaticData['Bwl']
        Wsa = hull.hydrostaticData['Wsa']
        Atr = hull.hydrostaticData['Atr']
        Ttr = hull.hydrostaticData['Ttr']
        
        beta_fore = beta_func(Lk/hull.Loa)
        beta_mid = beta_func(0.5*Lk/hull.Loa)
        
        Lc = max(0.0, Lk - beam/np.pi * tand(beta_fore)/tand(trim))
        
        Ldyn, Rdyn, Xdyn, lengthsDyn = get_hydrodynamic_force(V, Lk, Lc, beam, beta_mid, trim, eta, hull.environment)
        Lf, Rf, Xf = get_skin_friction(V, Lk, Lc, beam, beta_mid, trim, eta, Wsa, Atr, hull.environment)
        Rtr = get_transom_resistance(V, Ttr, Atr, hull.environment)
        
        return Lf, Rf, Xf, Ldyn, Rdyn, Xdyn, Rtr, lengthsDyn


def compute_resistance_savitsky(hull, V, heel, trim, eta):
    
    if (trim - eta) < 0.0:
        print('No planing. Increase trim angle.')

    # Physical data
    rho = hull.environment.water.rho
    nu = hull.environment.water.nu
    
    # Boat data
    b0 = 5
    b50 = 15.5
    b100 = 90
                
    beta_func = itrp.interp1d([0, 0.5, 1.0], [b0, b50, b100], kind='quadratic')
    
    def f(X, hull, V, beta_func, heel, trim, eta):
        z = X[0]
        g = hull.environment.g
        
        Lf, _, _, Ldyn, _, _, _, _ = get_forces_savitsky(z, hull, V, beta_func, heel, trim, eta)
        
        return abs(hull.displacement/(Ldyn/g + Lf/g + hull.volume * rho) - 1)
    
    fArgs = (hull, V, beta_func, heel, trim, eta)
    
    z0 = (hull.mesh.bounds[1,2] + hull.mesh.bounds[0,2]) / 10
    X0 = np.array([z0])
    Zeq = opt.fmin(f, X0, args=fArgs)[0]
    
    Lf, Rf, Xf, Ldyn, Rdyn, Xdyn, Rtr, lengthsDyn = get_forces_savitsky(Zeq, hull, V, beta_func, heel, trim, eta)
    
    print(lengthsDyn)
    print('Rf',round(Rf,1))
    print('Rtr',round(Rtr,1))
    print('Rdyn',round(Rdyn,1))
    print('Ldyn',round(Ldyn,1))
    print('Lf',round(Lf,1))
    print('Xdyn',round(Xdyn,1))
    print('Zeq',round(Zeq,3))
    
    
    return Rf + Rtr + Rdyn
    
#%%
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    
    mesh = "D:/000Documents/Cours/DPEA/paki_aka/carenes/18ft_25.stl"
    displacement = 210 + 3*80
    cog = np.array([2.5, 0.0, 0.3])
    profile = "naca0012.dat"
    
    eta = 2.
    
    hull = Hull('18ft_hull', displacement, cog, mesh)
    
    #%%
    
    heel = 0.
    
    TRIM1 = np.linspace(-1.5, 0, 2)
    VV1 = []
    RR1 = []
    DD1 = []
    
    vmin = 2
    vmax = 9
    
    for t in TRIM1:
        V = []
        R = []
        
        for vkn in np.linspace(vmin, vmax, 15):
            u = vkn * .514
            hull.free_immersion(0,0)
            r = hull.compute_hull_resistance(u, heel, trim=t, method="dsyhs")
            
            V.append(vkn)
            R.append(r)
        VV1.append(V)
        RR1.append(R)
    
    TRIM2 = np.linspace(3,3.8, 5)
    TRIM2 = [3., 3.5, 4.]
    VV2 = []
    RR2 = []
    DD2 = []
    
    vmin = 10
    vmax = 20
        
    for t in TRIM2:
        V = []
        R = []
        
        for vkn in np.arange(vmin, vmax+1, 2):
            u = vkn * .514
            
            r = compute_resistance_savitsky(hull, u, heel, t, eta)
            
            V.append(vkn)
            R.append(r)
        VV2.append(V)
        RR2.append(R)

    #%%
    del_axes = ['top', 'left', 'right']
    font = {'fontname':'Arimo'}
    fontsize = 11
    title_offset = -0.17
    resolution = 400
    aspect = (6, 4)
    lw = 1.2
    
    fig, ax = plt.subplots(figsize = aspect, dpi = resolution)
    
    ymin = 0
    ymax = 400
    
    color = [ 'firebrick','red', 'darkorange']
    lt = ['--', '-.', '-']
    lt = ['-', '--', '-.']
    
    for i in range(len(TRIM1)):
        div = 1.0
        if i == 1:
            div = 0.82
        ax.plot(VV1[i], np.array(RR1[i])*div,
                label='DHSYS \ trim='+str(TRIM1[i])+'°',
                color=color[i],
                linewidth = lw,
                linestyle= lt[i])
        
    for i in range(len(TRIM2)):
        div = 1.0
        if i == 1:
            div = 0.82
        ax.plot(VV2[i], np.array(RR2[i])*div,
                label='Savitsky \ trim='+str(TRIM2[i])+'°',
                color=color[i],
                linewidth = lw,
                linestyle= lt[i])
       
    

    plt.xlabel("VITESSE BATEAV (kts)", **font, size=fontsize)
    plt.ylabel("RÉSISTANCE CARÈNE (N)", **font, size=fontsize)
    
    plt.grid()
    plt.legend(fontsize = fontsize)
    # plt.title(mesh[39:])
    plt.xlim((2, 20))
    plt.ylim((0, 800))
    
    plt.show()