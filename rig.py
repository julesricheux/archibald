# -*- coding: utf-8 -*-
"""
SAILBOAT RIG DESCRIPTION AND COMPUTATIONS
(adjustment and aerodynamics)

Created 30/05/2023
Last update: 24/12/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

Further development plans:
    
    - Migrate the vortex lattice method from AVL to AeroSandbox (see experimental
      features). AeroSandbox offers better geometry management and enhanced graphics.
      The modified version programmed as an experimental feature also offers to
      take into account a vertical wind gradient, and the vertical aerodynamic
      force. The main difficulty is to keep consistent the viscous XFoil coupling.
      
    - Search for a better way to compute spinnaker aerodynamics (currently being
      ORC polar data)
    
    - Implement a post-stall model to be able to evaluate sails working at high
      angle of attacks (this already has been investigated independently, see
      experimental features)
"""

#%% DEPENDENCIES

from tqdm import tqdm
import numpy as np

from lifting_planes import Mainsail, Jib, Spi, Wingsail

from environment import _Environment, OffshoreEnvironment, InshoreEnvironment

from tools.math_utils import *
from tools.dyn_utils import *
from tools.avl_utils import *
from tools.xfoil_utils import *


#%% CLASSES

class Rig():
    def __init__(self,
                 name: str = 'rig',
                 env: _Environment = OffshoreEnvironment()):
        self.name = name
        self.nSails = 0
        self.nSpi = 0
        self.sails = dict()
        self._spiIdx = list()
        self._flatIdx = list()
        
        self.spiIsRaised = False
        
        self.area = 0.0
        
        self.environment = env
        
        self._globalData = list() 
        self._localData = list()
        
        # Coefficients for additionnal drift resistance computation
        _beta_coefs = np.array([[0., 30., 90., 180.],
                                 [0., 18., 70, 160.]])
        self._beta = build_interpolation(_beta_coefs, method='linear')[0]
        
        self._AVLgeo = './temp/avl/'+self.name+'.avl'
        self._AVLin = './temp/avl/'+self.name+'.in'
        self._AVLout = './temp/avl/'+self.name+'.out'
    
        self._XFin = './temp/xfoil/'+self.name+'.in'
        self._XFout = './temp/xfoil/'+self.name+'.out'
        
    def add_mainsail(self, name, nSections, le, chords,
                     minCamber=0.02, maxCamber=0.10, xCamber=0.33, maxTwist=5.0,
                     nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        self.sails[self.nSails] = Mainsail(name, nSections, le, chords,
                                           minCamber, maxCamber, xCamber, maxTwist,
                                           nChordwise, nSpanwise, Sref, Cref, Bref)
                
        self._flatIdx.append(self.nSails)
        
        self.nSails += 1
        self.area += self.sails[self.nSails-1].Sref
        
        
    def add_jib(self, name, nSections, le, chords,
                minCamber=0.02, maxCamber=0.10, xCamber=0.33, maxTwist=6.0,
                nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        self.sails[self.nSails] = Jib(name, nSections, le, chords,
                                      minCamber, maxCamber, xCamber, maxTwist,
                                      nChordwise, nSpanwise, Sref, Cref, Bref)
        
        self._flatIdx.append(self.nSails)
        
        self.nSails += 1
        self.area += self.sails[self.nSails-1].Sref
        
    def add_spi(self, name, asymmetric, pole,
                slu, sle, shw, sfl,
                tackPoint, headPoint,
                Sref=None):
    
        self.sails[self.nSails] = Spi(name, asymmetric, pole,
                                      slu, sle, shw, sfl,
                                      tackPoint, headPoint,
                                      Sref)
    
        self._spiIdx.append(self.nSails)
        self.nSails += 1
        self.nSpi += 1
        
        
    def raise_spi(self):
        """
        Raises the spinnaker if one is rigged and lowered.
        It will be taken into account in the following computations.
        """
        
        if self._spiIdx:
            if self.spiIsRaised:
                print('Spinnaker already raised.')
            else:
                self.spiIsRaised = True
                for i in self._spiIdx:
                    self.area += self.sails[i].Sref
        else:
            print('No spinnaker rigged.')
            
    def lower_spi(self):
        """
        Lowers the spinnaker if one is rigged and raised.
        It will not be taken into account in the following computations.
        """
        
        if self._spiIdx:
            if not self.spiIsRaised:
                print('Spinnaker already lowered.')
            else:
                self.spiIsRaised = False
                for i in self._spiIdx:
                    self.area -= self.sails[i].Sref
        else:
            print('No spinnaker rigged.')
    
    
    def remove_sail(self, i):
        """
        Removes a rigged sail.

        Parameters
        ----------
        i (int): Index of the sail to remove.
        """
        
        if i in self.sails.keys():
            self.area -= self.sails[i].Sref
            self.nSails -= 1
            
            if type(self.sails[i]) == Spi:
                self._spiIdx.remove(i)
                self.nSpi -= 1
            else:
                self._flatIdx.remove(i)
            
            del self.sails[i]

    
    def compute_aerodynamics(self, awa, aws, beta=None, disp=False):
        
        if beta is None:
            beta = np.ones(self.nSails - self.nSpi) * self._beta(awa)
            
        nu = self.environment.air.nu
    
        # Call AVL resolution
        if disp:
            print('\nAVL computation of '+self.name+'...\n')
            
        write_avl_rig_geometry(self, awa, beta)
        write_avl_input(self)
        run_avl_analysis(self)
        read_avl_rig_results(self, awa)
        
        self.CL = self.get_CL
        self.CDi = self.get_CDi
        
        # Viscous resolution
        for a in range(self.nSails):

            sail = self.sails[a]
            
            if type(sail) != Spi:
                
                localData = self._localData[a]
                
                chords, areas, ai = localData[:,3], localData[:,4], localData[:,5]*180/np.pi
                x, y, z = -localData[:,0], localData[:,2], localData[:,1]
                cl_2D, cdi_2D = np.abs(localData[:,6]), localData[:,7] - localData[:,8]
                cp = localData[:,9]
                
                # print(ai)
                
                # Compute 2D viscous&pressure drag coefficients for each strip
                if disp:
                    iterable = tqdm(range(sail.nSpanwise), desc='Viscous computation of '+sail.name)
                else:
                    iterable = range(sail.nSpanwise)
                
                cdv_2D = list()
                
                for i in iterable:
                    index = int(i*sail.nSections/sail.nSpanwise)
                    
                    profile = sail.sections[index]
                    AoA = awa - beta[a] - ai[i] - sail.twists[index]
                    Re = aws*chords[i] / nu
                    
                    # print(AoA)
                    if type(sail)==Wingsail:
                        write_xfoil_input(self, profile, AoA, Re)
                        run_xfoil_analysis(self)
                        cdv = read_xfoil_results(self)
                        
                    elif type(sail)==Mainsail or type(sail)==Jib:
                        coords = np.loadtxt(profile, skiprows=2)
                        camber = np.max(coords[:,1])
                        
                        # cdv = Cd_cambered_plate(AoA, camber)
                        cdv = cdi_2D/sail.nSpanwise
                        
                    cdv_2D.append(cdv)
                    
                cdv_2D = np.array(cdv_2D)
                
                # plt.plot(y, cl_2D**2 + (cdi_2D+cdv_2D)**2)
                # plt.plot(y, ai)
                
                self._globalData[a].append(np.sum(cdv_2D * areas)/sail.Sref)
                
                # Compute the force centroid
                c_2D = np.sqrt(cl_2D**2 + (cdi_2D + cdv_2D)**2)
                
                xyz = np.transpose(np.vstack([x, y, z]))
                rotation = rotation_matrix(np.array([0,0,1]), awa)
                    
                xyzR = np.dot(xyz, rotation)
                x, y, z = xyzR[:,0], xyzR[:,1], xyzR[:,2]
                
                x_centroid = np.sum(c_2D * areas * (x - cp*chords)) / np.sum(c_2D*areas)
                y_centroid = np.sum(c_2D * areas * y) / np.sum(c_2D*areas)
                z_centroid = np.sum(c_2D * areas * z) / np.sum(c_2D*areas)
                
                centroid = np.array([x_centroid, y_centroid, z_centroid])
                
                self._globalData[a].append(centroid)
                
            else:
                self._globalData[a].append(0.0)
                self._globalData[a].append(sail.get_centroid)
        
        self.CDv = self.get_CDv
        self.centroid = self.get_centroid
            
    @property
    def get_CL(self):
        if self._globalData:
            cl = 0.0
            for i, data in enumerate(self._globalData):
                cl += data[0]*self.sails[i].Sref/self.area
                
            return cl
        else:
            return None
                
    @property
    def get_CDi(self):
        if self._globalData:
            cdi = 0.0
            for i, data in enumerate(self._globalData):
                cdi += data[1]*self.sails[i].Sref/self.area
            return cdi
        else:
            return None
        
    @property
    def get_CDv(self):
        if self._globalData:
            cdv = 0.0
            for i, data in enumerate(self._globalData):
                cdv += data[2]*self.sails[i].Sref/self.area
            return cdv
        else:
            return None
       
    @property
    def get_CD(self):
        if self._globalData:
            return self.get_CDv + self.get_CDi
        
    def get_CX(self, awa):
        if self._globalData:
            return self.get_CL * sind(awa) - self.get_CD * cosd(awa)
        else:
            return None
        
    def get_CY(self, awa):
        if self._globalData:
            return self.get_CL * cosd(awa) + self.get_CD * sind(awa)
        else:
            return None
        
    @property
    def get_centroid(self):
        if self._globalData:
            centroid = np.zeros(3)
            divider = 0.0
            
            for i, data in enumerate(self._globalData):
                iCentroid = data[3]
                
                iCL = data[0]
                iCD = data[1] + data[2]
                
                iC = np.sqrt(iCL**2 + iCD**2)
                iS = self.sails[i].Sref
                
                centroid += iCentroid * iC * iS
                divider += iC * iS
                
            return centroid/divider
        else:
            return None
          
 