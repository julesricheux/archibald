# -*- coding: utf-8 -*-
"""
MARINE ENVIRONMENT DESCRIPTION
(air and water properties)

Created 30/05/2023
Last update: 19/06/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

Further development plans:
    
    - Implement enhanced air description

"""

#%% DEPENDENCIES

import os

from archibald.tools.math_utils import build_interpolation
from archibald.tools.doc_utils import read_coefs


#%% CLASSES

class _Water():
    def __init__(self, _water_data, T):
        
        measurements = read_coefs(_water_data)
        self._data = build_interpolation(measurements)
        
        self.set_temperature(T)
        
    def set_temperature(self, T):
        self.rho = self._data[0](T)
        self.mu = self._data[1](T)
        self.nu = self._data[2](T)
        
        
class Seawater(_Water):
    def __init__(self, T=15):
        current = os.path.dirname(os.path.realpath(__file__))
        _water_data = os.path.join(current, 'data/seawater_ittc_2011.csv')
        
        super().__init__(_water_data, T)
    
        
class Freshwater(_Water):
    def __init__(self, T=15):
        current = os.path.dirname(os.path.realpath(__file__))
        _water_data = os.path.join(current, 'data/freshwater_ittc_2011.csv')
        
        super().__init__(_water_data, T)
        

class Air():
    def __init__(self, rho=1.225, mu=1.81e-5, nu=1.48e-5):
        self.rho = rho
        self.mu = mu
        self.nu = nu


class _Environment():
    def __init__(self, water=None, air=Air()):
        self.water = water
        self.air = air
        self.g = 9.8066
        

class OffshoreEnvironment(_Environment):
    def __init__(self, waterT=15):
        super().__init__(water=Seawater(waterT))
        

class InshoreEnvironment(_Environment):
    def __init__(self, waterT=15):
        super().__init__(water=Freshwater(waterT))


if __name__ == '__main__':
    env = OffshoreEnvironment()