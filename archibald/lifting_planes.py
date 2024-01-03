# -*- coding: utf-8 -*-
"""
SHIP SAILS AND APPENDAGES DEFINITION
(geometry and adjustment)

Created: 30/05/2023
Last update: 19/06/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

Further development plans:
    
    - Add a specific Hydrofoil class (would need a free-surface proximity model)
    
    - Improve and make the sails adjustment easier (add a flattening percentage
      and make twist and camber varying accordingly ?)
"""

#%% DEPENDENCIES

import numpy as np

import os
import archibald

from archibald.tools.math_utils import *
from archibald.tools.doc_utils import *
from archibald.tools.dyn_utils import *
from archibald.tools.xfoil_utils import *


ARCHIBALD_DIR = os.path.dirname(archibald.__file__)


#%% CLASSES

class LiftingPlane():
    def __init__(self, name, nSections, le, chords, sections, twists=0.0,
                 nChordwise=None, nSpanwise=10,
                 Sref=None, Cref=None, Bref=None):
        
        self.name = name
        self.nSections = nSections
        self.le = le
        
        if type(chords) == float or type(chords) == int:
            self.chords = np.ones(self.nSections) * chords
        else:
            self.chords = chords
            
        if type(sections) == str:
            self.sections = np.array([sections] * nSections)
        else:
            self.sections = sections
            
        if type(twists) == float or type(twists) == int:
            self.twists = np.linspace(0,1,self.nSections) * twists
        else:
            self.twists = twists
            
        if Sref:
            self.Sref = Sref
        else:
            self.Sref = np.sum([(self.chords[i+1] + self.chords[i])/2*(np.linalg.norm(self.le[i+1] - self.le[i])) 
                                for i in range(self.nSections-1)])
            
        if Cref:
            self.Cref = Cref
        else:
            self.Cref = np.mean(self.chords)
            
        if Bref:
            self.Bref = Bref
        else:
            self.Bref = np.linalg.norm(le[-1] - le[0])
            
        self.nSpanwise = nSpanwise
        if nChordwise == None:
            self.nChordwise = max(2, round(nSpanwise * self.Sref / self.Bref**2))
        else:
            self.nChordwise = nChordwise
            
            
class Sail(LiftingPlane):
    def __init__(self, name, nSections, le, chords, sections, twists=0.0,
                 nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        super().__init__(name, nSections, le, chords, sections, twists, nChordwise, nSpanwise, Sref, Cref, Bref)
        
        self.tackPoint = self.le[0]
        self.headPoint = self.le[-1]
        self.foot = self.chords[0]
        self.head = self.chords[-1]
        self._compute_luff()
        self._compute_leech()
        
    def _compute_leech(self):
        # headBackPoint = self.le[-1] - np.array([self.chords[-1], 0., 0.])
        self.leech = self.chords[-1] +\
                     np.sum([(np.linalg.norm(self.le[i+1] - np.array([1., 0., 0.])*self.chords[i+1] 
                                             - self.le[i] + np.array([1., 0., 0.])*self.chords[i] )) 
                             for i in range(self.nSections-1)])
        
    def _compute_luff(self):
        self.luff = np.sum([(np.linalg.norm(self.le[i+1] - self.le[i])) for i in range(self.nSections-1)])
        
        
class Flatsail(Sail):
    def __init__(self, name, nSections, le, chords,
                 minCamber, maxCamber, xCamber, maxTwist,
                 nChordwise=None, nSpanwise=10,
                 Sref=None, Cref=None, Bref=None):
        
        cambers = [(maxCamber-minCamber) * i/(nSections-1) + minCamber for i in range(nSections)]
        
        def f(x, xc, camber):
            a = np.log(1/2)/np.log(xc)
            xp = x**a
            
            return -4*camber*(xp-0.5)**2 + camber
        
        extrados = np.linspace(1.0, 0.0, 50)
        intrados = np.linspace(0.0, 1.0, 50)[1:]
        x = np.concatenate((extrados, intrados))
        
        sections = []
        
        for i in range(nSections):
            y = f(x, xCamber, cambers[i])
            folder = ARCHIBALD_DIR+'\\data\\airfoils\\thin_sections\\'
            file = 'camb_'+str(round(xCamber,3))+'_'+str(round(cambers[i],3))+'.dat'
            write_xfoil_geometry(folder+file, x, y)
            sections.append(folder+file)
            
        sections = np.array(sections)
        
        twists = np.linspace(0,1,nSections)**2/3 * maxTwist
        
        super().__init__(name, nSections, le, chords, sections,
                         twists,nChordwise, nSpanwise,
                         Sref, Cref, Bref)
        
        
class Mainsail(Flatsail):
    def __init__(self, name, nSections, le, chords,
                 minCamber=0.02, maxCamber=0.10, xCamber=0.33, maxTwist=5.0,
                 nChordwise=None, nSpanwise=10,
                 Sref=None, Cref=None, Bref=None):
        
        super().__init__(name, nSections, le, chords,
                         minCamber, maxCamber, xCamber, maxTwist,
                         nChordwise, nSpanwise,
                         Sref, Cref, Bref)
        
        
class Jib(Flatsail):
    def __init__(self, name, nSections, le, chords,
                 minCamber=0.02, maxCamber=0.10, xCamber=0.33, maxTwist=6.0,
                 nChordwise=None, nSpanwise=10,
                 Sref=None, Cref=None, Bref=None):
        
        super().__init__(name, nSections, le, chords,
                         minCamber, maxCamber, xCamber, maxTwist,
                         nChordwise, nSpanwise,
                         Sref, Cref, Bref)
        
        
class Spi(Sail):    
    def __init__(self, name, asymmetric, pole,
                 slu, sle, shw, sfl,
                 tackPoint, headPoint,
                 Sref=None):
        
        if asymmetric:
            sl = (slu + sle) / 2
        else:
            sl = slu
        
        if asymmetric:
            if pole:
                _coefs_file = "./data/spi-asym-pole_orc_2021.csv"
            else:
                _coefs_file = "./data/spi-asym-centre_orc_2021.csv"
        else:
            _coefs_file = "./data/spi-sym_orc_2021.csv"
            
        coefs = read_coefs(_coefs_file)
        self._orc = build_interpolation(coefs)
        

        if Sref == None:
            Sref = sl * (sfl + 4 * shw) / 6
        
        super().__init__(name, 1, np.zeros(1), np.zeros(1), np.zeros(1), Sref=Sref, Cref=shw, Bref=sl)
        
        self.leech = sle
        self.luff = slu
        self.foot = sfl
        self.halfWidth = shw
        
        self.tackPoint = tackPoint
        self.headPoint = headPoint
        
        
    def get_CL(self, awa):
        vec = self.headPoint - self.tackPoint
        angle = np.abs(np.arctan2(vec[0], vec[2])) * 180/np.pi * self.halfWidth/self.foot
        
        return self._orc[0](awa) * cosd(angle)
    
    def get_CD(self, awa):
        return self._orc[1](awa)
    
    def get_CZ(self, awa):
        vec = self.headPoint - self.tackPoint
        angle = np.abs(np.arctan2(vec[0], vec[2])) * 180/np.pi * self.halfWidth/self.foot
        
        return self._orc[0](awa) * sind(angle)
    
    def get_CX(self, awa):
        return sind(awa) * self.get_CL(awa) - cosd(awa) * self.get_CD(awa)
    
    def get_CY(self, awa):
        return cosd(awa) * self.get_CL(awa) + sind(awa) * self.get_CD(awa)
    
    @property
    def get_centroid(self):
        return self.tackPoint + 0.565 * (self.headPoint-self.tackPoint)
        

class Wingsail(Sail):
    def __init__(self, name, nSections, le, chords, sections, twists=0.0,
                 nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        folder = ARCHIBALD_DIR+'\\data\\airfoils\\thick_sections\\'
        if type(sections) == str:
            relativeSections = folder+sections
        else:
            relativeSections = [folder+sections[i] for i in range(nSections)]
            
        super().__init__(name, nSections, le, chords, relativeSections, twists, nChordwise, nSpanwise, Sref, Cref, Bref)
            
        
class Appendage(LiftingPlane):
    def __init__(self, name, nSections, le, chords, sections, twists=0.0,
                 nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        folder = ARCHIBALD_DIR+'\\data\\airfoils\\thick_sections\\'
        if type(sections) == str:
            relativeSections = folder+sections
        else:
            relativeSections = [folder+sections[i] for i in range(nSections)]
        
        super().__init__(name, nSections, le, chords, relativeSections, twists, nChordwise, nSpanwise, Sref, Cref, Bref)
        
        
class Centreboard(Appendage):
    def __init__(self, name, nSections, le, chords, sections, twists=0.0,
                 nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        super().__init__(name, nSections, le, chords, sections, twists, nChordwise, nSpanwise, Sref, Cref, Bref)
        
        
class Rudder(Appendage):
    def __init__(self, name, nSections, le, chords, sections, shaftRoot=None, shaftTip=None,
                 twists=0.0, nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        super().__init__(name, nSections, le, chords, sections, twists, nChordwise, nSpanwise, Sref, Cref, Bref)
        
        self.angle = 0.0
        
        if shaftRoot is None:
            self.shaftRoot = le[0]
        else:
            self.shaftRoot = shaftRoot
            
        if shaftTip is None:
            self.shaftTip = self.shaftRoot - np.array([0,0,1])
        else:
            self.shaftTip = shaftTip
            
        self.shaft = self.shaftTip - self.shaftRoot

    def set_angle(self, angle):
        self.angle = angle
