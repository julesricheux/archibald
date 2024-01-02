# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:02:55 2023

@author: Jules
"""

from aerosandbox import XFoil
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
from aerosandbox.geometry import Airfoil

import scipy.interpolate as itrp
from scipy.signal import argrelextrema

from tools.math_utils import *




if __name__ == '__main__':
    
    af = Airfoil("naca2410").repanel(n_points_per_side=100)
    # af = Airfoil("goe531").repanel(n_points_per_side=100)
    # af.coordinates[:, 1] *= .9
    af.coordinates[:, 1] += 4 * af.coordinates[:, 0] * (1 - af.coordinates[:, 0]) * .0

    xf = XFoil(
        airfoil=af,
        Re=1e7,
        n_crit=9,
        # hinge_point_x=0.75,
        xfoil_command = "./xfoil.exe",
        # xtr_upper = .1,
        # xtr_lower = .1,
        # max_iter = 100,
        # repanel = True,
        # verbose=True,
    )
    
    result = xf.alpha(np.linspace(0,30,31))  # Note: if a result does not converge, 
    

    #%%
    
    def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
        denom = (x1-x2) * (x1-x3) * (x2-x3)
        a = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
        b = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
        c = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
        
        return a, b, c
    
    CL = result['CL']
    CD = result['CD']
    alpha = result['alpha']
    
    iMaximaCL = argrelextrema(CL, np.greater)[0]
    iMinimaCL = argrelextrema(CL, np.less)[0]
    
    if len(iMaximaCL):
        iMaxCL = iMaximaCL[0]
    else:
        iMaxCL = -1
    
    if len(iMaximaCL) > 1:
        iMaxCL2 = iMaximaCL[-1]
    else:
        iMaxCL2 = -1
        
    if len(iMinimaCL):
        iMinCL = iMinimaCL[0]
    else:
        iMinCL = -1
        
    alpha = alpha[:iMinCL]
    CL = CL[:iMinCL]
    CD = CD[:iMinCL]

    maxCL = CL[iMaxCL]
    alphaMaxCL = alpha[iMaxCL]
    
    divCL = CL[-1]
    divCD = CD[-1]
    divAlpha = alpha[-1]
    
    maxCamber = af.max_camber()
    maxCD = -.92 * maxCamber**2 + 1.1 * maxCamber + 1.98
    
    AL = (maxCL + divCL)/2
    # AL = max(np.max(CL[4*len(CL)//5:]), maxCL)
    # AL = divCL/cosd(divAlpha)
    # AL = maxCL
    kL = 1/(divAlpha - 90) * (np.arccos(divCL/AL) * 180/np.pi - 90)
    # kL = 90 / (90 - 2*alphaMaxCL)
    
    AD = maxCD
    kD = 1/(divAlpha - 90) * (np.arcsin(np.sqrt(divCD/maxCD)) * 180/np.pi - 90)
    
    joinAlpha = int(min(2*alphaMaxCL, divAlpha))
    joinAlpha = divAlpha
    
    alphaStall = np.arange(alpha[-1]+1, 91, 1)
    alphaFull = np.hstack((alpha, alphaStall))

    
    divCLviterna = maxCD / 2 * sind(2*alphaStall) + maxCD * cosd(alphaStall)**2 / sind(alphaStall)    


    A1 = maxCD / 2
    B1 = maxCD
    
    A2 = (maxCL - maxCD*sind(alphaMaxCL)*cosd(alphaMaxCL)) * sind(alphaMaxCL) / cosd(alphaMaxCL)**2
    B2 = (maxCD - maxCD*sind(alphaMaxCL)**2) / cosd(alphaMaxCL)
    
    # divCLviterna = maxCD / 2 * sind(2*alphaStall[]) + maxCD * cosd(divAlpha)**2 / sind(divAlpha)  
    
    CLviterna = A1 * sind(2*alphaStall) + A2 * cosd(alphaStall)**2 / sind(alphaStall)
    CLviterna *= divCL / CLviterna[0]
    # CDviterna = B1 * sind(alphaStall)**2 #+ B2 * cosd(alphaStall)
    
    # a, b, c = calc_parabola_vertex(alpha[iMinCL], CL[iMinCL], alpha[iMaxCL2], CL[iMaxCL2], 90, 0.)
    # a, b, c = calc_parabola_vertex(alpha[-1], CL[-1], 2.5*alphaMaxCL, maxCL, 90, 0.)
    # a, b, c = calc_parabola_vertex(alpha[-2], CL[-2], alpha[-1], CL[-1], 90, 0.)
    

    
    CLfull = np.hstack((CL, np.zeros(len(alphaStall))))
    CDfull = np.hstack((CD, np.zeros(len(alphaStall))))
    # CLstall = AL * cosd(kL * (alphaStall - 90) + 90)
    # CLstall = AL * cosd(2*alphaStall-90)
    # CLstall = a * alphaStall**2 + b*alphaStall + c
    CLstall = CLviterna
    # CDstall = AD * sind(kD * (alphaFull - 90) + 90)
    # CLstall = a * alphaStall**2 + b*alphaStall + c
    CDstall = AD * sind(kD * (alphaStall - 90) + 90)**2
    
    alphaFull = np.hstack((alpha[np.where(alpha <= joinAlpha)], alphaStall))
    
    # CLfull = np.maximum(CLfull, CLstall)
    # CDfull = np.maximum(CDfull, CDstall)
    
    alphaFull = np.hstack((alpha, alphaStall))
    CLfull = np.hstack((CL, CLstall))
    CDfull = np.hstack((CD, CDstall))

    
    plt.plot(alphaFull, CLfull)
    plt.plot(alphaFull, CDfull)
    # plt.plot(alpha, CL)
    # plt.plot(alpha, CD)
    plt.plot(alphaStall, CLviterna)
    # plt.plot(alphaStall, CDviterna)
    plt.show()
    
    plt.plot(CDstall, CLviterna)
    plt.plot(CD, CL)
    # plt.plot(CDfull, CLfull)
    plt.show()
    
    awa = 30
    CX = CLfull * sind(awa) - CDfull * cosd(awa)
    CY = CLfull * cosd(awa) + CDfull * sind(awa)
    
    # red = np.where()
    
    # plt.plot(CY[np.where], CX)
