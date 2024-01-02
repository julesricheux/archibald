# -*- coding: utf-8 -*-
"""
SHIP HULLS DESCRIPTION AND COMPUTATION
(hydrostatics, resistance prediction, appendage hydrodynamics)

Created: 30/05/2023
Last update: 24/12/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

Further development plans:
    
    - Stabilise the hydrostatics computation. It can currently return an error
      when the immersion is initialised so that the hull has no waterplane (i.e.
      is fully dry or immersed, i.e. z is outside the mesh vertical bounds)
    
    - Implement a direct method to compute multihulls hull resistance. After
      hydrostatics computation, the mesh of each hull needs to be computed separately.
      Spacing and positionning between hulls should also be evaluated to apply
      interactions coefficients.
    
    - Implement Savisky planing and pre-planing methods for fast vessels (this
      already has been investigated independently, see experimental features)
    
    - Migrate the vortex lattice method from AVL to AeroSandbox (see experimental
      features). AeroSandbox offers better geometry management and enhanced graphics.
      The main difficulty is to keep consistent the viscous XFoil coupling.
"""

#%% DEPENDENCIES

import os
from tqdm import tqdm
import numpy as np

import scipy.optimize as opt

import trimesh
import trimesh.transformations as tf

import matplotlib.pyplot as plt

from lifting_planes import Centreboard, Rudder
from environment import _Environment, OffshoreEnvironment, InshoreEnvironment

from tools.math_utils import *
from tools.doc_utils import *
from tools.dyn_utils import *
from tools.avl_utils import *


#%% CLASSES

class Hull():
    def __init__(self,
                 name: str = 'hull',
                 displacement: float = 0.0,
                 cog: np.array = np.zeros(3),
                 mesh: str = None,
                 env: _Environment = OffshoreEnvironment()):
        
        self.name = name
        self.nAppendages = 0
        self.appendages = {}
        
        self.cog = cog
        self.displacement = displacement
        
        if mesh and os.path.exists(mesh):
            self.mesh = trimesh.load(mesh)
            self.Loa = self.mesh.bounds[1,0] - self.mesh.bounds[0,0]
            self.Boa = self.mesh.bounds[1,1] - self.mesh.bounds[0,1]
            self.D = self.mesh.bounds[1,2] - self.mesh.bounds[0,2]
        else:
            self.mesh = None
            self.Loa = None
            self.Boa = None
            self.D = None
        
        self.area = 0.0
        
        self.environment = env
        
        self._globalData = None
        self._localData = None
        
        # Coefficients for DSYHS computation
        current = os.path.dirname(os.path.realpath(__file__))
        _keunig_coefs_file = os.path.join(current, 'data/coefs_keunig_2008.csv')
        
        if os.path.exists(_keunig_coefs_file):
            _coefs = read_coefs(_keunig_coefs_file)
            self._keunig = build_interpolation(_coefs)
        else:
            self._keunig = None
            
        # Coefficients for additionnal drift resistance computation
        _drift_coefs = np.array([[0.0, np.pi/4, 1.0],
                                 [0.0, 0.47, 1.05]])
        self._drift = build_interpolation(_drift_coefs, method='linear')[0]
        
        self._AVLgeo = './temp/avl/'+self.name+'.avl'
        self._AVLin = './temp/avl/'+self.name+'.in'
        self._AVLout = './temp/avl/'+self.name+'.out'
    
        self._XFin = './temp/xfoil/'+self.name+'.in'
        self._XFout = './temp/xfoil/'+self.name+'.out'
        
        self.hydrostaticData = dict()
            
        # self.eta = 0.0 # 
            
        
    def add_centreboard(self, name, nSections, le, chords, sections, twists=0.0,
                        nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        self.appendages[self.nAppendages] = Centreboard(name, nSections, le, chords, sections,
                                                        twists, nChordwise, nSpanwise, Sref, Cref, Bref)
        self.nAppendages += 1
        self.area += self.appendages[self.nAppendages-1].Sref
        
        
    def add_rudder(self, name, nSections, le, chords, sections, shaftRoot=None, shaftTip=None,
                   twists=0.0, nChordwise=None, nSpanwise=10, Sref=None, Cref=None, Bref=None):
        
        self.appendages[self.nAppendages] = Rudder(name, nSections, le, chords, sections, shaftRoot, shaftTip,
                                                   twists, nChordwise, nSpanwise, Sref, Cref, Bref)
        self.nAppendages += 1
        self.area += self.appendages[self.nAppendages-1].Sref
        
        
    def set_rudder_angle(self, angle):
        for a in self.appendages.values():
            if type(a) == Rudder:
                a.set_angle(angle)
    
    
    def remove_appendage(self, i):
        if i in self.sails.keys():
            self.area -= self.appendages[i].Sref
            self.nAppendages -= 1
            del self.appendages[i]
    
    
    def compute_appendage_hydrodynamics(self, delta, V, disp=False):
        
        nu = self.environment.water.nu
        
        # Call AVL resolution
        if disp:
            print('\nAVL computation of '+self.name+' at '+str(round(delta,1))+'°...\n')
            
        write_avl_hull_geometry(self, IYsym=1)
        write_avl_input(self, delta)
        run_avl_analysis(self)
        read_avl_hull_results(self)
        
        self.CL = self.get_CL
        self.CDi = self.get_CDi
        
        # Call XFOIL resolution
        for a in range(self.nAppendages):

            appendage = self.appendages[a]
            
            localData = self._localData[a]
            
            chords, areas, ai = localData[:,3], localData[:,4], localData[:,5]*180/np.pi
            x, y, z = -localData[:,0], localData[:,2], -localData[:,1]
            cl_2D, cdi_2D = np.abs(localData[:,6]), localData[:,7] - localData[:,8]
            cp = localData[:,9]
            
            # print(ai)
            
            # Compute 2D viscous&pressure drag coefficients for each strip
            if disp:
                iterable = tqdm(range(appendage.nSpanwise), desc='XFOIL computation of '+appendage.name)
            else:
                iterable = range(appendage.nSpanwise)
            
            cdv_2D = []
            
            for i in iterable:
                index = int(i*appendage.nSections/appendage.nSpanwise)
                
                profile = appendage.sections[index]
                AoA = delta-ai[i]-appendage.twists[index]
                Re = V*chords[i] / nu
                
                # print(AoA)
                
                write_xfoil_input(self, profile, AoA, Re)
                run_xfoil_analysis(self)
                cdv = read_xfoil_results(self)
                
                cdv_2D.append(cdv)
                
            cdv_2D = np.array(cdv_2D)
            
            # self._globalData[a].append(np.sum(cdv_2D * areas/chords)/appendage.Bref * np.sum(areas)/appendage.Sref)
            # self._globalData[a].append(np.sum(cdv_2D * areas/chords) * np.sum(areas)/appendage.Sref)
            self._globalData[a].append(np.sum(cdv_2D * areas)/appendage.Sref)

            # Compute the force centroid
            c_2D = np.sqrt(cl_2D**2 + (cdi_2D + cdv_2D)**2)
            
            x_centroid = np.sum(c_2D * areas * (x - cp*chords)) / np.sum(c_2D*areas)
            y_centroid = np.sum(c_2D * areas * y) / np.sum(c_2D*areas)
            z_centroid = np.sum(c_2D * areas * z) / np.sum(c_2D*areas)
            
            centroid = np.array([x_centroid, y_centroid, z_centroid])
            
            l = 1/2 * self.environment.water.rho * areas * V**2 * cl_2D
            d = 1/2 * self.environment.water.rho * areas * V**2 * (cdi_2D + cdv_2D)
            l = cl_2D
            d = (cdi_2D + cdv_2D)
            
            self._globalData[a].append(centroid)
            
            # plt.figure()
            # plt.plot(l/np.max(l), label='l')
            # plt.plot(d/np.max(d), label='d')
            # plt.plot(np.sqrt(l**2 + d**2)/np.max(np.sqrt(l**2 + d**2)), label='f')
            # plt.plot(chords/np.max(chords), label='chords')
            # plt.legend()
            # plt.show()
            
        self.CDv = self.get_CDv
        self.centroid = self.get_centroid

    def compute_minimal_hydrostatics(self, z=0, heel=0, trim=0):        
        n = set_normal(heel, trim)

        # Extract the immersed hull
        underwater = self.mesh.slice_plane(plane_origin=[0,0,z], plane_normal=n, cap=True)

        # Compute main properties
        volume = underwater.volume #volume
        cob = underwater.center_mass #center of buoyancy
        
        return volume, cob

    def compute_hydrostatics(self, z=0, heel=0, trim=0):
        n = set_normal(heel, trim)
        
        try:
            # Extract the immersed hull
            underwater = self.mesh.slice_plane(plane_origin=[0,0,z], plane_normal=n, cap=True)
            ws = self.mesh.slice_plane(plane_origin=[0,0,z], plane_normal=n, cap=False)    
            floatation = self.mesh.section(plane_origin=[0,0,z], plane_normal=n)
            
            
            # Main hyrostatics computation
            volume = underwater.volume #volume
            
            uw_bounds = underwater.bounds
            f_bounds = floatation.bounds
            
            if uw_bounds is None:
                uw_bounds = np.zeros((3,3))
            if f_bounds is None:
                f_bounds = np.zeros((3,3))
            
            Lwl = (f_bounds[1,0] - f_bounds[0,0])/cosd(trim)
            Bwl = (f_bounds[1,1] - f_bounds[0,1])/cosd(heel)
            T = (uw_bounds[1,2] - uw_bounds[0,2])/cosd(heel)
            
            
            # BMt computation
            c = floatation.centroid
            v = floatation.vertices - c
            
            rot_matrix_trim = tf.rotation_matrix(np.radians(-trim), np.array([0,1,0]))
            rot_matrix_heel = tf.rotation_matrix(np.radians(-heel), np.array([1,0,0]))
    
            
            section = tf.transform_points(v, rot_matrix_trim)
            section = tf.transform_points(section, rot_matrix_heel)
            
            peak = section[np.argmax(section[:, 0]), :]
            
            section = section - peak + np.array([Lwl, 0, 0])
    
            # path2d = trimesh.path.path.Path2D(floatation.entities, vertices=section)
            planar = trimesh.path.Path2D(entities=floatation.entities.copy(),
                                         vertices=section[:, :2],
                                         metadata=floatation.metadata.copy(),
                                         process=False)
            
            # Half-entry angle computation
            x_interval = [Lwl*389/400, 399/400*Lwl]
            sel = (section[:, 0] >= x_interval[0]) & (section[:, 0] <= x_interval[1])
            bow_pts = section[sel] - np.array([Lwl, 0, 0])
            
            star_bow_pts = bow_pts[(bow_pts[:, 1] < 0.)]
            port_bow_pts = bow_pts[(bow_pts[:, 1] >= 0.)]
            
            star_offset = star_bow_pts[np.argmax(star_bow_pts[:, 0]), :]
            port_offset = port_bow_pts[np.argmax(port_bow_pts[:, 0]), :]
            
            star_bow_pts -= star_offset
            port_bow_pts -= port_offset
            
            star_bow_pts = star_bow_pts[(star_bow_pts[:, 0] != 0.)]
            port_bow_pts = port_bow_pts[(port_bow_pts[:, 0] != 0.)]
            
            star_ie = np.degrees(np.mean(np.arctan(star_bow_pts[:,1]/-star_bow_pts[:,0])))
            port_ie = np.degrees(np.mean(np.arctan(port_bow_pts[:,1]/-port_bow_pts[:,0])))
            
            ie = (port_ie - star_ie)/2
            
            Ixx = 0.0
            Iyy = 0.0
            for poly in planar.polygons_closed:
                Ixx += trimesh.path.polygons.second_moments(poly)[0]
                Iyy += trimesh.path.polygons.second_moments(poly)[1]
    
            # Compute main properties
            volume = underwater.volume #volume
            cob = underwater.center_mass #center of buoyancy
            BMt = Ixx / volume # transverse metacentric radius
            BMl = Iyy / volume # longitudinal metacentric radius
            
            # RMt computation
            yrot = np.array([0, cosd(heel), sind(heel)])
            zrot = np.array([0 ,-sind(heel), cosd(heel)])
            
            metaT = cob + BMt*zrot
            
            # BMtvec = metaT - cob
            
            GMtvec = metaT - self.cog
            GZtvec = GMtvec - np.dot(GMtvec, zrot.T) * zrot
            
            GMt = np.linalg.norm(GMtvec[1:]) * np.sign(np.dot(GMtvec[1:], zrot[1:]))
            GZt = np.linalg.norm(GZtvec[1:]) * np.sign(np.dot(GZtvec[1:], yrot[1:]))
            
            # RMl computation
            xrot = np.array([cosd(heel), 0, sind(heel)])
            zrot = np.array([-sind(heel), 0, cosd(heel)])
            
            metaL = cob + BMl*zrot
            
            # BMlvec = metaL - cob
    
            GMlvec = metaL - self.cog
            GZlvec = GMlvec - np.dot(GMlvec, zrot.T) * zrot
            
            GMl = np.linalg.norm(GMlvec[[0,2]]) * np.sign(np.dot(GMlvec[[0,2]], zrot[[0,2]]))
            GZl = np.linalg.norm(GZlvec[[0,2]]) * np.sign(np.dot(GZlvec[[0,2]], xrot[[0,2]]))
            
            # Compute floatation
            Uwa = underwater.area
            Wsa = ws.area
            Wpa = Uwa - Wsa
            
            Uw = underwater.centroid
            Ws = ws.centroid
            
            cof = (Uwa * Uw - Wsa * Ws) / Wpa #center of floatation
            
            # Compute midship area
            ms = underwater.section(plane_origin=cof, plane_normal=[1,0,0])
            if ms is None:
                Ax = 0.0
            else:
                ms = ms.to_planar()
                Ax = ms[0].area
                
            # Compute transversal area
            cl = underwater.section(plane_origin=cof, plane_normal=[0,1,0])
            if cl is None:
                Ay = 0.0
            else:
                cl = cl.to_planar()
                Ay = cl[0].area
                
            # Compute transom area
            xap = max(f_bounds[0,0], uw_bounds[0,0])
            tr = underwater.section(plane_origin=[xap+min(Lwl/100, .1),0,0], plane_normal=[1,0,0])
            if tr is None:
                Atr = 0.0
            else:
                tr = tr.to_planar()
                Atr = tr[0].area
                Ttr=np.min(np.abs(tr[0].bounds[1]-tr[0].bounds[0]))
                
            # Compute transverse bulb area
            xfp = f_bounds[1,1]
            bt = underwater.section(plane_origin=[xfp+min(Lwl/100, .1),0,0], plane_normal=[1,0,0])
            if bt is None:
                Abt = 0.0
            else:
                bt = bt.to_planar()
                Abt = bt[0].area
            
            # compute hydro coefficients
            Cb = volume / (Lwl*Bwl*T)
            Cp = volume / (Lwl*Ax)
            Cwp = Wpa / (Lwl*Bwl)
            Cx = Ax / (Bwl*T)
            
            # lengths = np.array([BMt, GMt, GZt, BMl, GMl, GZl, Lwl, Bwl, T])
            # areas = np.array([Wsa, Wpa, Amc, Atr, Abt])
            # coefs = np.array([Cb, Cp, Cwp, Cm])
            
            lengths = {'BMt': BMt, 'GMt': GMt, 'GZt': GZt, 'BMl': BMl, 'GMl': GMl, 'GZl': GZl, 'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': f_bounds[0,0]}
            areas = {'Wsa': Wsa, 'Wpa': Wpa, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': Abt}
            coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx}
            
            self.hydrostaticData = lengths | areas | coefs
            self.cob = cob
            self.cof = cof
            self.volume = volume
            
            self.hydrostaticData['immersion'] = z
            self.hydrostaticData['heel'] = heel
            self.hydrostaticData['trim'] = trim
            self.hydrostaticData['ie'] = ie
            
            return volume, cob, cof, lengths, areas, coefs
        
        except:
            self.hydrostaticData = {}
            return 0., np.array([0.,0.,0.]), np.array([0.,0.,0.]), {}, {}, {}
    
    
    def free_heel_trim_immersion(self):
        
        rho = self.environment.water.rho
        
        def heel_trim_immersion(X, rho):
            # initialize parameters
            z, heel, trim = X
            n = set_normal(heel, trim)

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            BG = self.cog - cob # CoB-CoG vector
            prod = np.cross(BG, n) # cross product to check colinearity
            prod = np.linalg.norm(prod)
            
            det1 = np.linalg.det([BG[1:], n[1:]])
            det2 = np.linalg.det([BG[[0,2]], n[[0,2]]])
            
            eq = np.array([volume*rho/self.displacement - 1,
                           det1,
                           det2])
            
            return np.linalg.norm(eq)
        
        z0 = (self.mesh.bounds[1,2] + self.mesh.bounds[0,2]) / 10
        x0 = np.array([z0, 1., 1.])
        param = (rho)
        
        # limits = [(0,10), (-60, 60), (-10,10)]
        
        Xopt = opt.minimize(heel_trim_immersion, x0, args=param, tol=1e-5)
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt.x[0], Xopt.x[1], Xopt.x[2])

        return volume, cob, cof, lengths, areas, coefs, Xopt.x


    def free_trim_immersion(self, heel=0):
        
        rho = self.environment.water.rho
        
        def trim_immersion_eq(X, heel, rho):
            # initialize parameters
            z, trim = X
            n = set_normal(heel, trim)

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            BG = self.cog - cob # CoB-CoG vector
            
            det = np.linalg.det([BG[[0,2]], n[[0,2]]])
            
            eq = np.array([volume*rho/self.displacement - 1,
                           det])
            
            return np.linalg.norm(eq)
        
        z0 = (self.mesh.bounds[1,2] + self.mesh.bounds[0,2]) / 10
        x0 = np.array([z0, 1.])
        param = (heel, rho)
        
        Xopt = opt.minimize(trim_immersion_eq, x0, args=param, tol=1e-5)        
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt.x[0], heel, Xopt.x[1])
        
        return volume, cob, cof, lengths, areas, coefs, Xopt.x


    def free_heel_immersion(self, trim=0):
        
        rho = self.environment.water.rho
        
        def heel_immersion_eq(X, trim, rho):
            # initialize parameters
            z, heel = X
            n = set_normal(heel, trim)

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            BG = self.cog - cob # CoB-CoG vector
            
            det = np.linalg.det([BG[1:], n[1:]])
            
            eq = np.array([volume*rho/self.displacement - 1,
                           det])
            
            return np.linalg.norm(eq)
        
        z0 = (self.mesh.bounds[1,2] + self.mesh.bounds[0,2]) / 10
        x0 = np.array([z0, 1.])
        param = (trim, rho)
        
        Xopt = opt.minimize(heel_immersion_eq, x0, args=param, tol=1e-5)
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt.x[0], Xopt.x[1], trim)
        
        return volume, cob, cof, lengths, areas, coefs, Xopt.x


    def free_immersion(self, heel=0, trim=0):
        
        rho = self.environment.water.rho
        
        def immersion_eq(X, heel, trim, rho):
            # initialize parameters
            z = X[0]

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            eq = volume*rho-self.displacement
            
            return eq
        
        z0 = (self.mesh.bounds[1,2] + self.mesh.bounds[0,2]) / 10
        x0 = np.array([z0])
        param = (heel, trim, rho)
        
        # xBounds = [(self.mesh.bounds[0,2], self.mesh.bounds[1,2])]
        
        # Xopt = opt.minimize(immersion_eq, x0, args=param, bounds=xBounds, tol=1e-5)
        Xopt = opt.fsolve(immersion_eq, x0, args=param, xtol=1e-5)
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt[0], heel, trim)
        
        return volume, cob, cof, lengths, areas, coefs, Xopt
        
        
    def compute_resistance_dsyhs(self, V, z=0, heel=0, trim=0):
        # Physical data
        rho = self.environment.water.rho
        nu = self.environment.water.nu
        g = self.environment.g
        
        volume = self.volume
        cob = self.cob
        cof = self.cof       
        
        displacement = volume * rho * g
        
        lcbfpp = self.mesh.bounds[1,0] - cob[0]
        LCFfpp = self.mesh.bounds[1,0] - cof[0]
        
        Lwl = self.hydrostaticData['Lwl']
        Bwl = self.hydrostaticData['Bwl']
        T = self.hydrostaticData['T']
        Ttr = self.hydrostaticData['Ttr']
        Cx = self.hydrostaticData['Cx']
        Cp = self.hydrostaticData['Cp']
        Wpa = self.hydrostaticData['Wpa']
        Wsa = self.hydrostaticData['Wsa']
        
        Atr = self.hydrostaticData['Atr']
        

        # Transom additionnal resistance (Holtrop&Mennen, 1978)
        Fr_T = V / np.sqrt(g * Ttr)
        if Fr_T < 5:
            ctr = 0.2 * (1 - (0.2 * Fr_T))
        else:
            ctr = 0
        Rtr = 0.5 * rho * (V ** 2) * Atr * ctr
        
        Fr = V / np.sqrt(g*Lwl)
        Re = V * Lwl / nu
        
        # Bare hull viscous resistance
        Rvh = 1/2 * rho * (Wsa - Atr) * Cf_hull(Re) * V**2
        
        # Bare hull upright residuary resistance DSYHS
        K = self._keunig
        
        Rrh = displacement * (K[0](Fr) + volume**(1/3)/Lwl * (K[1](Fr) * lcbfpp/Lwl + \
                                                              K[2](Fr) * Cp + \
                                                              K[3](Fr) * volume**(2/3)/Wpa + \
                                                              K[4](Fr) * Bwl/Lwl + \
                                                              K[5](Fr) * lcbfpp/LCFfpp + \
                                                              K[6](Fr) * Bwl/T + \
                                                              K[7](Fr) * Cx))
            
        return Rvh + Rrh + Rtr


    def compute_resistance_holtrop(self, V, z=0, heel=0, trim=0):
        """
        Calculate the total resistance of a ship given its velocity, geometric parameters and fluid properties
        """
        # Physical data
        rho = self.environment.water.rho
        nu = self.environment.water.nu
        g = self.environment.g
        
        Loa = self.mesh.bounds[1,0] - self.mesh.bounds[0,0]
        # Boa = self.mesh.bounds[1,1] - self.mesh.bounds[0,1]
        volume = self.volume
        cob = self.cob
        cof = self.cof   
        
        Lwl = self.hydrostaticData['Lwl']
        Bwl = self.hydrostaticData['Bwl']
        T = self.hydrostaticData['T']
        # Ttr = lengths['Ttr']
        Cx = self.hydrostaticData['Cx']
        Cp = self.hydrostaticData['Cp']
        Cb = self.hydrostaticData['Cb']
        Cwp = self.hydrostaticData['Cwp']
        Wpa = self.hydrostaticData['Wpa']
        Wsa = self.hydrostaticData['Wsa']
        
        Atr = self.hydrostaticData['Atr']
        Abt = self.hydrostaticData['Abt']
        
        origin = self.hydrostaticData['0L']
        ie = self.hydrostaticData['ie']
        
        Lbp = Loa # length between perpendiculars
        Lcb = cob[0]
        lcb = (Lcb-origin)/Lwl - .5
        
        # hB = cob[2] - self.mesh.bounds[0,2] # bulb center above keel line
        hB = T/2
        
        Csternchoice = max(1, round(Cp * 4))
        Bulbchoice = 0
        Sapp = self.area
        Appendage = self.nAppendages
        
        print(volume)
        
        def compute_Rv_holtrop(V, Lbp, Loa, Lwl, volume, Bwl, T, Wsa, Cp, lcb, Csternchoice, nu, rho, origin):
            """
            Calculate the frictional resistance of a ship using the ITTC '57 method.
            
            Args:
                V (float): speed [kts]
                Lbp (float): Length between perpendiculars (m)
                Loa (float): Length overall (m)
                Lwl (float): Length at waterline (m)
                T (float): Draft (m)
                S (float): Wetted surface area (m^2)
                V (float): Volume (m^3)
                Csternchoice (int): Choice of stern form (1=transom, 2=V-shaped, 3=U-shaped, 4=Spoon-shaped)
                M (float): Mass displacement (kg)
                rho (float): Density of water (kg/m^3)
                
            Returns:
                float: Frictional resistance of the ship (N)
            """

            if Csternchoice == 1:
                Cstern = -25
            elif Csternchoice == 2:
                Cstern = -10
            elif Csternchoice == 3:
                Cstern = 0
            elif Csternchoice == 4:
                Cstern = 10
            else:
                print("Invalid choice for the stern shape")
            

            c14 = 1 + 0.011*Cstern
            Lr = Lwl*(1 - Cp + 0.06*Cp*lcb/(4*Cp-1))

            # std deviation 4.6%
            k = .93 + .487118*c14*(Bwl/Lwl)**1.06806 * (T/Lwl)**.46106 * (Lwl/Lr)**.121563 *\
                   (Lwl**3/volume)**.36486 * (1-Cp)**(-.604247) - 1
            
            k *= (1 - .046*1.96)
            
            Re = (V * Lwl) / nu
            # ACF = 5.1e-4
            Rf = (1+k) * Cf_hull(Re) * (0.5 * rho * Wsa * (V ** 2))
            
            return Rf


        def compute_Rw_holtrop(V, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, ie, rho, g):
            """
            Calculates the wave-making resistance of a ship in calm water.
            
            params:
                V (float): speed [kts]
                Lwl (float): The length of the waterline in meters.
                Lbp (float): The length between perpendiculars in meters.
                B (float): The beam of the ship in meters.
                T (float): The draft of the ship in meters.
                Abt (float): The area of the bulbous bow in square meters.
                Cp (float): The prismatic coefficient.
                Cwp (float): The coefficient of the waterplane area.
                Atr (float): The transom area in square meters.
                lcb (float): lcb in %
                hB (float): Bulb height (m)
                Cx (float): The midship coefficient.
                rho (float): The density of water in kg/m^3.
                g (float): The acceleration due to gravity in m/s^2.
            
            Returns:
                float: The added resistance of the ship in calm water in (N)
            """
            
            
            def Rwa(Fr, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, rho, g):
                d = -0.9
                
                if (Bwl / Lwl) <= 0.11:
                    c7 = 0.229577 * (Bwl / Lwl)**(1/3)
                elif (Bwl / Lwl) <= 0.25:
                    c7 = Bwl / Lwl
                else:
                    c7 = 0.5 - 0.0625 * (Lwl / Bwl)
                    
                c1 = 2223105 * c7**3.78613 * (T / Bwl)**1.07961 * (90 - ie)**-1.37565
                
                c3 = (0.56 * Abt**1.5 / (Bwl * T * (0.31 * np.sqrt(Abt) + T - hB)))
                c2 = np.exp(-1.89 * np.sqrt(c3))
                
                c5 = 1.0 - 0.8 * Atr / (Bwl * T * Cx)
                
                if Cp < 0.8:
                    c16 = 8.07981 * Cp - 13.8673 * Cp**2.0 + 6.984388 * Cp**3.0
                else:
                    c16 = 1.73014 - 0.7067 * Cp
                    
                m1 = 0.0140407 * Lwl / T - 1.75254 * volume**(1/3) / Lwl - 4.79323 * Bwl / Lwl - c16
                
                if (Lwl**3 / volume) < 512:
                    c15 = -1.69385
                elif (Lwl**3 / volume) < 1726.91:
                    c15 = -1.69385 + ((Lwl / volume**(1.0 / 3.0)) - 8.0) / 2.36
                else:
                    c15 = 0.0

                m4 = c15 * 0.4 * np.exp(-0.034 * Fr**-3.29)
                
                if (Lwl / Bwl) < 12.0:
                    lam = 1.446 * Cp - 0.03 * (Lwl / Bwl)
                else:
                    lam = 1.446 * Cp - 0.36
                
                return (c1 * c2 * c5 * volume * rho * g *
                        np.exp(m1 * Fr**d + m4 * np.cos(lam * Fr**-2.0)))
            
            def Rwb(Fr, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, rho, g):
                d = -0.9
                
                c17 = (6919.3
                        * Cx**-1.3346
                        * ((volume / Lwl**3)**2.00977)
                        * ((Lwl / Bwl) - 2.0)**1.40692)
                
                c3 = (0.56 * Abt**1.5 / (Bwl * T * (0.31 * np.sqrt(Abt) + T - hB)))
                c2 = np.exp(-1.89 * np.sqrt(c3))
                
                c5 = 1.0 - 0.8 * Atr / (Bwl * T * Cx)
                
                m3 = (-7.2035 * (Bwl / Lwl)**0.326869 * (T / Bwl)**0.605375)
                
                if (Lwl**3 / volume) < 512:
                    c15 = -1.69385
                elif (Lwl**3 / volume) < 1726.91:
                    c15 = -1.69385 + ((Lwl / volume**(1.0 / 3.0)) - 8.0) / 2.36
                else:
                    c15 = 0.0

                m4 = c15 * 0.4 * np.exp(-0.034 * Fr**-3.29)
                
                if (Lwl / Bwl) < 12.0:
                    lam = 1.446 * Cp - 0.03 * (Lwl / Bwl)
                else:
                    lam = 1.446 * Cp - 0.36
                
                return (c17 * c2 * c5 * volume * rho * g *
                        np.exp(m3 * Fr**d + m4 * np.cos(lam * Fr**-2.0)))
            
            Fr = V / (np.sqrt(g * Lwl))
            
            if Fr < 0.4:
                return Rwa(Fr, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, rho, g)  * 1e3
            elif Fr < 0.55:
                Rwa40 = Rwa(.40, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, rho, g)
                Rwb55 = Rwb(.55, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, rho, g)
                
                return (Rwa40 + (10.0 * Fr - 4) * Rwb55 - Rwa40 / 1.5)  * 1e3
            else:
                return Rwb(Fr, Lwl, Lbp, Bwl, T, volume, Abt, Cp, Cwp, Atr, lcb, hB, Cx, rho, g) * 1e3
        

        def compute_Rb_holtrop(V, T, hB, Abt, Bulbchoice, rho, g):
            """
            Calculates the resistance due to bulbous bow using Holtrop's method.

            Parameters:
            hB (float): height of bulbous bow [m]
            Bulbchoice (int): 1 for bulbous bow, 0 for no bulbous bow
            """
            
            if Bulbchoice == 1:
                Fri = V / (np.sqrt((g * (T - hB - (0.25 * (np.sqrt(Abt))))) + (0.15 * (V ** 2))))
                pb = (0.56 * (np.sqrt(Abt))) / (T - (1.5 * hB))
                Rb = 0.11 * (np.exp(((-3) * (pb ** (-2)))) * (Fri ** 3) * (Abt ** 1.5) * rho * g) / (1 + (Fri ** 2))
                
                return Rb
            
            elif Bulbchoice == 0:
                return 0
            
            else:
                print("Invalid choice for the bulbous bow")


        def compute_Rtr(V, Atr, Bwl, Cwp, rho, g):
            """
            Calculate transom resistance using the Holtrop-Mennen method.
            """
            
            if Atr == 0.0:
                return 0.0
            
            else : 
                V = V * 0.5144
                FRT = V / (np.sqrt((2 * g * Atr) / (Bwl + (Bwl * Cwp))))
                if FRT < 5:
                    c6 = 0.2 * (1 - (0.2 * FRT))
                else:
                    c6 = 0
                Rtr = 0.5 * rho * (V ** 2) * Atr * c6
                return Rtr


        def compute_Ra_holtrop(V, Lbp, Bwl, T, Cb, hB, rho, Wsa, Abt):
            """
            Calculates the model-ship correlation resistance RA.
            """
            
            # if T / Lbp <= 0.04:
            #     c4 = T / Lbp
            # else:
            #     c4 = 0.04
            # c3 = ((0.56 * Abt) ** 1.5) / ((Bwl * T) * ((0.31 * (np.sqrt(Abt))) + (T - hB)))
            # c2 = np.exp(-1.89 * (np.sqrt(c3)))
            
            # div = (7.5 * (Cb ** 4) * c2 * (0.04 - c4))
            
            # if div == 0.0:
            #     CA = (0.006 * ((Lbp + 100) ** -0.16)) - 0.00205
            # else :
            #     CA = (0.006 * ((Lbp + 100) ** -0.16)) - 0.00205 + (0.003 * (np.sqrt(Lbp / div)))
                
            # CA = (0.006 * (Lwl + 100.0)**-0.16
            #       - 0.00205
            #       + 0.003 * (Lwl / 7.5)**0.5 * Cb**4 * c2 * (0.04 - c4))
            
            # std deviation 0.00021
            CA = 0.00675 * (Lwl +100)**(-1/3) - 0.00064
            
            CA -= 1.96*0.00021
                
            Ra = 0.5 * rho * Wsa * (V ** 2) * CA
            
            return Ra
        
        Rf = compute_Rv_holtrop(V, Lbp, Loa, Lwl, volume, Bwl, T, Wsa, Cp, lcb, Csternchoice, nu, rho, origin)
        Rw = compute_Rw_holtrop(V, Lwl, Lbp, Bwl, T, Bwl, Abt, Cp, Cwp, Atr, lcb, hB, Cx, ie, rho, g)
        Rb = compute_Rb_holtrop(V, T, hB, Abt, Bulbchoice, rho, g)
        Rtr = compute_Rtr(V, Atr, Bwl, Cwp, rho, g)
        Ra = compute_Ra_holtrop(V, Lbp, Bwl, T, Cb, hB, rho, Wsa, Abt)
        
        print(Rf, Rw, Rb, Rtr, Ra)
        
        return Rf + Rw + Rb + Rtr + Ra
    
    
    def compute_hull_resistance(self, V, z=0, heel=None, trim=None, delta=0, method=str()):
        # Physical data
        rho = self.environment.water.rho
        
        # if heel is None and trim is None:
        #     volume, cob, cof, lengths, areas, coefs, _ = self.free_heel_trim_immersion()
        # elif heel is None:
        #     volume, cob, cof, lengths, areas, coefs, _ = self.free_heel_immersion(trim)
        # elif trim is None:
        #     volume, cob, cof, lengths, areas, coefs, _ = self.free_trim_immersion(heel)
        # else:
        #     volume, cob, cof, lengths, areas, coefs, _ = self.free_immersion(heel, trim)
        
        Vx = V * cosd(delta)
        Vy = V * sind(delta)
        Cb = self.hydrostaticData['Cb']
        Ay = self.hydrostaticData['Ay']
        
        if method == "dsyhs":
            Rhull = self.compute_resistance_dsyhs(V, z, heel, trim)
        elif method == "holtrop":
            Rhull = self.compute_resistance_holtrop(V, z, heel, trim)
        else:
            print("Warning: invalid method given for the hull resistance computation")
            Rhull = 0.0
        
        Rdrift = 1/2 * rho * Ay * Vy**2 * self._drift(Cb)
        
        return np.sqrt(Rhull**2 + Rdrift**2)
        
        
    @property
    def get_CL(self):
        if self._globalData:
            cl = 0.0
            for i, data in enumerate(self._globalData):
                cl += data[0]*self.appendages[i].Sref/self.area
                
            return cl
        else:
            return None
            
        
    @property
    def get_CDi(self):
        if self._globalData:
            cdi = 0.0
            for i, data in enumerate(self._globalData):
                cdi += data[1]*self.appendages[i].Sref/self.area
            return cdi
        else:
            return None
        
        
    @property
    def get_CDv(self):
        if self._globalData:
            cdv = 0.0
            for i, data in enumerate(self._globalData):
                cdv += data[2]*self.appendages[i].Sref/self.area
            return cdv
        else:
            return None
        
        
    @property
    def get_CD(self):
        if self._globalData:
            return self.get_CDv + self.get_CDi
        
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
                iS = self.appendages[i].Sref
                
                centroid += iCentroid * iC * iS
                divider += iC * iS
                
            return centroid/divider
        else:
            return None
        
        
#%%
if __name__ == "__main__":
    
    #%% 18ft APPENDAGES
    
    Vkts = 9 # kts
    V = Vkts * 0.5144 # m/s
    
    
    mesh = "D:/000Documents/Cours/DPEA/paki_aka/carenes/18ft_25.stl"
    displacement = 405
    cog = np.array([2.5, 0.0, 0.3])
    profile = "naca0012.dat"
    # # profile = "eppler377_t3_c4.dat"
    
    # MOLENEZ II test
    # mesh = 'D:/000Documents/Cours/DPEA/vpp/molenez6.stl'
    
    # Ship properties
    # displacement = 140e3 # kg
    # cog = np.array([11.9,
    #                 0.0,
    #                 2.6])
    
    nSections = 6
    
    # Centreboard
    dagger_chords = np.array([0.40, 0.40, 0.38, 0.33, 0.25, 0.02])
    
    dagger_xle = np.array([2.95, 2.93, 2.89, 2.83, 2.73, 2.51])
    dagger_yle = np.zeros(nSections)
    dagger_zle = np.array([-0.13, -0.46, -0.78, -1.09, -1.29, -1.41])
    
    dagger_le = np.transpose(np.vstack((dagger_xle, dagger_yle, dagger_zle)))
    
    # Rudder
    rudder_chords = np.array([0.33, 0.32, 0.30, 0.26, 0.17, 0.02])
    
    rudder_xle = np.array([-0.03, -0.04, -0.06, -0.11, -0.20, -0.36])
    rudder_yle = np.zeros(nSections)
    rudder_zle = np.array([-0.01, -0.24, -0.47, -0.71, -0.87, -0.95])
    
    rudder_le = np.transpose(np.vstack((rudder_xle, rudder_yle, rudder_zle)))
    
    shaft_root = np.array([-0.01, 0, 0.3]) 
    shaft_tip = np.array([-0.01, 0, 0.1])
    
    hull = Hull('18ft_hull', displacement, cog, mesh)
    hull.add_centreboard('daggerboard', nSections, dagger_le, dagger_chords, profile)
    hull.add_rudder('rudder', nSections, rudder_le, rudder_chords, profile, shaftRoot=shaft_root)
    
    delta = 2.0 # drift angle in deg
    rudder_angle = -delta * 0.8 # rudder angle in deg
    
    #%%
    
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
            r = hull.compute_hull_resistance(u, heel=5, trim=t, method="dsyhs")
            
            V.append(vkn)
            R.append(r)
        VV1.append(V)
        RR1.append(R)
    
    #%%
    R = []
    for i in range(0,15):
        hull.free_immersion(0,0)
        R.append(hull.compute_hull_resistance(5,delta=i, method='dsyhs'))
    plt.plot(R)
    plt.ylim((0., None))

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
       
    

    plt.xlabel("VITESSE BATEAU (kts)", **font, size=fontsize)
    plt.ylabel("RÉSISTANCE CARÈNE (N)", **font, size=fontsize)
    
    
    plt.legend(fontsize = fontsize)
    # plt.title(mesh[39:])
    plt.xlim((2, 18))
    plt.ylim((0, 800))
    
    plt.show()
        
    