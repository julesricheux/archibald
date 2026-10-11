# -*- coding: utf-8 -*-
"""
MIYAZAWA VISCOUS INTERFERENCE BETWEEN HULLS (BODY INTERFERENCE)

Closed-form estimate of the increase of the viscous resistance of a hull due to
the velocity augmentation between hulls. No tank data needed, but it comes
from a potential flow analysis of a pair of cylinders: use it as an order of
magnitude when no measured factors are available.
"""

#%% DEPENDENCIES

import archibald.numpy as np

from archibald.dynamics.hydro.multihull_common import EPS

#%% RESISTANCE COMPONENTS

def compute_dRf_miyazawa(
        Bwl,
        y,
        Rf,
        **kwargs,
    ):
    r"""Viscous interference of each hull due to the presence of the other hulls.

    According to the review of :cite:`insel1992investigation` of the work of
    Miyazawa, the viscous (wake traverse) resistance of a catamaran is

    .. math::
        (C_{WT})_{cat} = (1 + \varepsilon^2)\,(C_{WT})_{demihull},
        \qquad \varepsilon = \frac{B}{S}

    where B is the demihull beam and S the separation between the centrelines.
    The wake traverse resistance is the viscous resistance (friction and
    viscous pressure), so the factor is applied to the individual ``Rf``,
    which already holds the form factor. For n hulls, the contributions of the
    other hulls are added pairwise (assumption):

    .. math::
        \Delta R_{f,i} = R_{f,i}\sum_{j\neq i}\left(\frac{B_i}{S_{ij}}\right)^2

    Valid for S/B larger than about 2 (the factor tends to 1 when the hulls
    touch). Insel & Molland measured a velocity augmentation of up to about
    10 % between demihulls, hence a friction increase of the same order.

    Parameters
    ----------
    Bwl : list of float, Waterline beam of each hull [m].
    y : list of float, Transverse position of each hull [m].
    Rf : list of float, Individual viscous (frictional) resistance of each hull [N].

    Returns
    -------
    list of float, Viscous interference of each hull dRf[i] [N].
    """
    n = len(Bwl)
    dRf = [0. for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                S2 = (y[i] - y[j]) ** 2 + EPS
                dRf[i] = dRf[i] + Rf[i] * Bwl[i] ** 2 / S2
    return dRf
