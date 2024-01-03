# -*- coding: utf-8 -*-
"""
ENTIRE SHIP DESCRIPTION AND PERFORMANCE COMPUTATION
NB: NOT YET READY FOR GENERAL USE

Created: 01/07/2023
Last update: 01/07/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -


Further development plans:
    
    - Automate the settings of response surfaces bounds for each parameter.
      Measurement points distribution should also be made adaptive to reduce 
      computation time
    
    - Stabilise the equilibrium computation from response surfaces. In particular,
      errors occur when a parameter goes outside the bounds of response surface.
      The interpolation method may be changed to an RBF interpolator to increase
      tolerance and allow for extrapolation. A private 
"""


#%% DEPENDENCIES

import os

from tqdm import tqdm
import numpy as np
import pickle

import scipy.interpolate as itrp
import scipy.optimize as opt

from archibald.hull import Hull
from archibald.rig import Rig
from archibald.environment import _Environment, OffshoreEnvironment, InshoreEnvironment

from archibald.tools.doc_utils import *
from archibald.tools.math_utils import *


#%% CLASSES

class _Boat():
    def __init__(self, 
                 name: str = 'boat',
                 env: _Environment = _Environment()):
        self.name = name
        self.hull = None
        self.hull_method = str()
        
        self.rig = None
        
        self.environment = env
        
        self.RS = dict()
        
        
    def add_hull(self,
                 name: str = 'hull',
                 displacement: float = 0.0,
                 cog: np.array = np.zeros(3),
                 mesh: str = None):
    
        self.hull = Hull(self.name+'_'+name, displacement, cog, mesh, self.environment)
        
        
    def add_rig(self,
                name: str = 'rig'):
        
        self.rig = Rig(self.name+'_'+name, self.environment)
        
        
    def build_RS(self, dim, paramRanges, paramN, function, fArgs, dataNames, msg):
        
        values = [np.zeros(paramN) for j in range(len(dataNames))]
        it = np.nditer(values, flags=['multi_index'])
        
        iterable = tqdm(range(np.prod(paramN)), desc=msg, position=0, leave=True)
        
        idx = []
        for x in it:
            idx.append(it.multi_index)
        
        for i in iterable:
            param = [paramRanges[k][idx[i][k]] for k in range(dim)]
            
            data = function(*param, *fArgs)
            
            for j in range(len(dataNames)):
                values[j][idx[i]] = data[j]
        
        RS = [itrp.RegularGridInterpolator(paramRanges, values[j], method='cubic') for j in range(len(dataNames))]
        
        for i, data in enumerate(dataNames):
            self.RS[data] = RS[i]
        
        # aws range ?
        # awa range from 0 to 180
        # speed range from Fn 0 to Fn 0.74 (dsyhs)
        # speed range from Fn 0 to Fn 0.53 (holtrop)
        # delta range from 0 to 45
        # heel range from 0 to 90
        
        
    def build_hull_RS(self, z=0, trim=0):
        
        def function(V, heel, delta, z, trim, method):
            self.hull.compute_hydrostatics(z, heel, trim)
            Rt = self.hull.compute_hull_resistance(V, z, heel, trim, delta, method)
            
            return Rt, self.hull.hydrostaticData['GZt']
        
        dataNames = ['FxHull', 'GZ']
        
        dim = 3
        
        paramN = np.ones(dim, dtype='int')
        paramN = np.array([4,4,4])
        
        paramExtrema = np.array([[0,4],
                                [0,80],
                                [0,10]])
        paramPowers = np.array([1, 2, 3])
        paramRanges = [np.linspace(0,1,paramN[i])**paramPowers[i] *\
                      (paramExtrema[i][1] - paramExtrema[i][0]) + paramExtrema[i][0]
                      for i in range(dim)]
        
        fArgs = (z, trim, self.method)
        
        msg = "Building hull response surfaces"
        
        self.build_RS(dim, paramRanges, paramN, function, fArgs, dataNames, msg)

        
    def build_rig_RS(self, beta=None):
        
        def function(aws, awa, beta):
            
            self.rig.compute_aerodynamics(awa, aws, beta)
            
            rho = self.environment.air.rho
            FxSail = 1/2 * rho * self.rig.area * aws**2 * self.rig.get_CX(awa)
            FySail = 1/2 * rho * self.rig.area * aws**2 * self.rig.get_CY(awa)
            
            return FxSail, FySail, *self.rig.centroid
        
        dataNames = ['FxRig', 'FyRig', 'xCE', 'yCE', 'zCE']
        
        dim = 2
        
        paramN = np.ones(dim, dtype='int')
        paramN = np.array([10,12])
        
        paramExtrema = np.array([[0,100],
                                 [0,120]])
        paramPowers = np.array([1, 1])
        paramRanges = [np.linspace(0,1,paramN[i])**paramPowers[i] *\
                      (paramExtrema[i][1] - paramExtrema[i][0]) + paramExtrema[i][0]
                      for i in range(dim)]
        
        fArgs = (None,)
        
        msg = "Building rig response surfaces"
        
        self.build_RS(dim, paramRanges, paramN, function, fArgs, dataNames, msg)
        
        
    def build_appendage_RS(self):
        
        def function(V, drift, neutralRudder=True, verbose=False):
            if neutralRudder:
                self.hull.set_rudder_angle(-0.88 * drift)
            else:
                self.hull.set_rudder_angle(0.0)
                
            self.hull.compute_appendage_hydrodynamics(drift, V, disp=verbose)
            
            rho = self.environment.water.rho
            FxApp = 1/2 * rho * self.hull.area * V**2 * self.hull.get_CD
            FyApp = 1/2 * rho * self.hull.area * V**2 * self.hull.get_CL
            
            return FxApp, FyApp, *self.hull.centroid
        
        dataNames = ['FxApp', 'FyApp', 'xCLR', 'yCLR', 'zCLR']
        
        dim = 2
        
        paramN = np.ones(dim, dtype='int')
        paramN = np.array([4,4])
        
        paramExtrema = np.array([[0,4],
                                [0,10]])
        paramPowers = np.array([1, 3])
        paramRanges = [np.linspace(0,1,paramN[i])**paramPowers[i] *\
                      (paramExtrema[i][1] - paramExtrema[i][0]) + paramExtrema[i][0]
                      for i in range(dim)]
        
        fArgs = (True, False)
        
        msg = "Building appendages response surfaces"
        
        self.build_RS(dim, paramRanges, paramN, function, fArgs, dataNames, msg)
        
    def save_RS(self, directory='./saves/', filename=None):
        # Create the directory if it doesn't exist        
        os.makedirs(directory, exist_ok=True)
        
        # Define the file path
        if filename == None:
            filename = self.name+'.pkl'
        filepath = os.path.join(directory, filename)
        
        # Save the variable to the file
        with open(filepath, 'wb') as f:
            pickle.dump(self.RS, f)
            
            
    def load_RS(self, directory='./saves/', filename=None):
        # Define the file path
        if filename == None:
            filename = self.name+'.pkl'
        filepath = os.path.join(directory, filename)
        
        if os.path.exists(filepath):
            # Open the file and retrieve the variable
            with open(filepath, 'rb') as f:
                self.RS = pickle.load(f)
        else:
            print('No such saved file or directory: '+filepath)
    
    
    def free_speed(self, tws, twa, heel, drift):
        
        def f(X, tws, twa, heel, drift):
            # initialize parameters
            V = X[0]
            aws, awa = compute_AW(tws, twa, V)
            
            Fx = self.RS['FxRig']((aws, awa))
            Rx = self.RS['FxHull']((V, heel, drift)) + self.RS['FxApp']((V, drift))
            
            return Fx - Rx
    
        X0 = np.ones(1)
        fArgs = (tws, twa, heel, drift)
        # xBounds = [(1e-3,4)]
        
        Xopt = opt.fsolve(f, X0, args=fArgs, xtol=1e-5)
        
        return Xopt
    
    def free_speed_heel(self, tws, twa, drift):
        g = self.environment.g
        
        def f(X, tws, twa, drift):
            # initialize parameters
            V, drift = X
            aws, awa = compute_AW(tws, twa, V)
            
            # CHECK IF INPUTS ARE INSIDE BOUNDS
            
            Fx = self.RS['FxRig']((aws, awa))
            Rx = self.RS['FxHull']((V, heel, drift)) + self.RS['FxApp']((V, drift))
            
            HA = self.RS['zCE']((V, drift)) - self.RS['zCLR']((V, drift))
            GZ = self.RS['GZ']((V, heel, drift))
            
            Fy = self.RS['FyRig']((aws, awa))
            
            D = self.hull.displacement
            
            eq = np.array([Fx - Rx,
                           Fy*HA - D*g*GZ])
            
            return eq
        
        X0 = np.ones(2)
        fArgs = (tws, twa, drift)
        
        Xopt = opt.root(f, X0, args=fArgs)
        
        return Xopt.x
    
    
    def free_speed_drift(self, tws, twa, heel):
        
        def f(X, tws, twa, heel):
            # initialize parameters
            V, drift = X
            aws, awa = compute_AW(tws, twa, V)
            
            Fx = self.RS['FxRig']((aws, awa))
            Rx = self.RS['FxHull']((V, heel, drift)) + self.RS['FxApp']((V, drift))
            
            Fy = self.RS['FyRig']((aws, awa))
            Ry = self.RS['FyApp']((V, drift))
            
            eq = np.array([Fx - Rx,
                           Fy - Ry])
            
            return eq
        
        X0 = np.ones(2)
        fArgs = (tws, twa, heel)
        
        Xopt = opt.root(f, X0, args=fArgs)
        
        return Xopt.x
    
    def free_speed_heel_drift(self, tws, twa):
        g = self.environment.g
        
        def f(X, tws, twa):
            # initialize parameters
            V, heel, drift = X
            aws, awa = compute_AW(tws, twa, V)
            
            print('V', V)
            # print('heel', heel)
            # print('drift', drift)
            # print('tws', tws)
            # print('twa', twa)
            # print('awa', awa)
            # print('aws', aws)
            
            # print(aws, awa)
            
            Fx = self.RS['FxRig']((aws, awa))
            Rx = self.RS['FxHull']((V, heel, drift)) + self.RS['FxApp']((V, drift))
            
            HA = self.RS['zCE']((V, drift)) - self.RS['zCLR']((V, drift))
            GZ = self.RS['GZ']((V, heel, drift))
            
            Fy = self.RS['FyRig']((aws, awa))
            Ry = self.RS['FyApp']((V, drift))
            
            D = self.hull.displacement
            
            eq = np.array([Fx - Rx,
                           Fy - Ry,
                           Fy*HA - D*g*GZ])
            
            return eq
        
        X0 = np.ones(3)
        fArgs = (tws, twa)
        # xBounds = [(1e-3,4), (1e-3,50), (1e-3,10)]
        
        Xopt = opt.root(f, X0, args=fArgs)
        
        return Xopt.x
        
        
class Sailboat(_Boat):
    def __init__(self, 
                 name: str = 'sailboat',
                 env: OffshoreEnvironment = OffshoreEnvironment()):
        super().__init__(name, env)
        self.method = 'dsyhs'


class CargoShip(_Boat):
    def __init__(self, 
                 name: str = 'cargoship',
                 env: OffshoreEnvironment = OffshoreEnvironment()):
        super().__init__(name, env)
        self.method = 'holtrop'
        
        
# need to differentiate mono and multihulls AND sail and motorboats
# Is there a general method to treat both ?

#%%
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    #%% Sailboat definition
    
    sailboat = Sailboat('soos')
    
    #Hull definition
    mesh = "D:/000Documents/Cours/DPEA/paki_aka/carenes/18ft_25.stl"
    displacement = 405
    cog = np.array([2.5, -2., 0.0])
    
    sailboat.add_hull(displacement=displacement, cog=cog, mesh=mesh)
    
    # Rig definition
    nSections = 5
    
    mainDxf = 'D:/000Documents/Cours/DPEA/paki_aka/voilure/gv_rig2.dxf'
    mainLe, mainChords = dxf_to_le_chords(mainDxf, nSections)
    
    sailboat.add_rig()
    sailboat.rig.add_mainsail('main', nSections, mainLe, mainChords)
    
    # Appendages definition
    profile = 'eppler836.dat'
    daggerDxf = 'D:/000Documents/Cours/DPEA/paki_aka/appendices/dagger.dxf'
    daggerLe, daggerChords = dxf_to_le_chords(daggerDxf, nSections)
    
    sailboat.hull.add_centreboard('daggerboard', nSections, daggerLe, daggerChords, profile, nSpanwise=10)
    
    #%% Response surfaces building
    
    # sailboat.load_RS(filename='example_RS.pkl')
    
    sailboat.build_hull_RS()
    sailboat.build_rig_RS()
    sailboat.build_appendage_RS()
    
    sailboat.save_RS(filename='example_RS.pkl')
    
    #%%
    tws = 1.
    twa = 100.
    heel = 0.0
    drift = 3.0
    
    V, heel, drift = sailboat.free_speed_heel_drift(tws, twa)
    # V, drift = sailboat.free_speed_drift(tws, twa, heel)
    V = sailboat.free_speed(tws, twa, heel, drift)
    
    aws, awa = compute_AW(tws, twa, V)
