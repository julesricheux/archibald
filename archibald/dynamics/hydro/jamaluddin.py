# -*- coding: utf-8 -*-
"""
EMPIRICAL CATAMARAN INTERFERENCE REGRESSIONS (JAMALUDDIN, AFTER MOLLAND ET AL.)

Closed-form regressions of the viscous and wave interference of catamaran
demihulls, function of the separation to length ratio S/L (and of the Froude
number for the wave part). Fitted on the Southampton experiments of the NPL
round bilge catamaran series.

The equations were found in a secondary source (Iqbal & Trimulyono, KAPAL 2014),
which quotes Jamaluddin (2012, PhD thesis, ITS, in Indonesian) as the origin of
the S/L dependent terms, and Molland, Wellicome & Couser (1996, Trans. RINA 138A)
for the base form factor. They reproduce the tables of that paper exactly, but
the primary sources have not been checked: validate before relying on them.
"""

#%% DEPENDENCIES

import archibald.numpy as np

from archibald.dynamics.hydro.multihull_common import (
    EPS, froude_number, separation_to_length, interp_nodes, smooth_clip,
)

#%% COEFFICIENTS

# Wave interference tau = A * (S/L)**B, fitted at six Froude numbers
FN_TAU = [0.19, 0.28, 0.37, 0.47, 0.56, 0.65]
A_TAU = [0.068, 0.359, 0.574, 0.790, 0.504, 0.501]
B_TAU = [-1.38, -0.87, -0.33, -0.14, -0.31, -0.18]

# Validity range of the regressions
SL_MIN, SL_MAX = 0.2, 0.4       # fitted S/L range
FN_MAX = 0.65                   # highest fitted Froude number
SL_ZERO = 0.6                   # S/L above which the wave interference is assumed to vanish
FN_ZERO = 0.75                  # Fn above which the wave interference is assumed to vanish

#%% INDIVIDUAL COEFFICIENTS

def tau_jamaluddin(
        Fn,
        SL,
        **kwargs,
    ):
    r"""Wave resistance interference factor of a catamaran.

    According to the regressions of Jamaluddin, fitted at six Froude numbers
    (Fn = 0.19 to 0.65) and S/L = 0.2 to 0.4:

    .. math::
        \tau = A(F_n)\,(S/L)^{B(F_n)}

    with the coefficients ``A_TAU`` and ``B_TAU`` linearly interpolated in Fn.
    Outside the fitted range, the interference is assumed to vanish smoothly
    (assumption, the regression power law would not tend to 1 for large S/L):

    .. math::
        \tau_{eff} = 1 + (\tau - 1)\,
            \text{clip}\left(\frac{0.6 - S/L}{0.2}\right)\,
            \text{clip}\left(\frac{0.75 - F_n}{0.10}\right)

    The factor is defined with respect to the wave resistance of the demihull
    computed by a slender body (thin ship) method, so it also absorbs the
    difference between that theory and the experiments: it is consistent with
    ``michell`` individual wave resistances, not with regression ones.

    Parameters
    ----------
    Fn : float, Froude number [-].
    SL : float, Separation to length ratio [-].

    Returns
    -------
    float, Wave interference factor tau [-].
    """
    SLc = np.maximum(SL, SL_MIN)
    taus = [A_TAU[k] * SLc ** B_TAU[k] for k in range(len(FN_TAU))]
    tau = interp_nodes(Fn, FN_TAU, taus)
    w_sl = np.minimum(np.maximum((SL_ZERO - SL) / (SL_ZERO - SL_MAX), 0.), 1.)
    w_fn = np.minimum(np.maximum((FN_ZERO - Fn) / (FN_ZERO - FN_MAX), 0.), 1.)
    return 1. + (tau - 1.) * w_sl * w_fn


def viscous_interference_ratio(
        Lwl,
        volume,
        SL,
        **kwargs,
    ):
    r"""Relative increase of the form factor due to the other demihull.

    According to the regression of Jamaluddin, who added an S/L term to the
    form factor of Molland et al.:

    .. math::
        1 + \beta k = 3.03\left(\frac{L_{WL}}{\nabla^{1/3}}\right)^{-0.40}
            + 0.016\left(\frac{S}{L}\right)^{-0.65}

    The first term does not depend on S/L, it is the form factor without
    interference, so only the second term is interference:

    .. math::
        \frac{\Delta(1+k)}{1+k} =
            \frac{0.016\,(S/L)^{-0.65}}{3.03\,(L_{WL}/\nabla^{1/3})^{-0.40}}

    S/L is bounded below by 0.2 (lower limit of the fit). Fitted for
    L/∇^(1/3) between 6.3 and 9.5.

    Parameters
    ----------
    Lwl : float, Waterline length [m].
    volume : float, Displaced volume [m³].
    SL : float, Separation to length ratio [-].

    Returns
    -------
    float, Relative viscous interference [-].
    """
    SLc = np.maximum(SL, SL_MIN)
    # bounded: a hull that is hardly wet has a vanishing volume
    Vp = 0.5 * (volume + np.sqrt(volume ** 2 + 1e-12)) + EPS    # smooth ReLU: V >= 0
    slenderness = smooth_clip(Lwl / Vp ** (1. / 3.), 5., 14.)
    return 0.016 * SLc ** (-0.65) / (3.03 * slenderness ** (-0.40))

#%% RESISTANCE COMPONENTS

def compute_dRw_jamaluddin(
        stw,
        Lwl,
        y,
        g,
        Rw,
        **kwargs,
    ):
    r"""Wave interference matrix from the regressions of Jamaluddin.

    .. math::
        \Delta R_{w,ij} = R_{w,i}\,\big(\tau_{eff}(Fn_i, S_{ij}/L_{ij}) - 1\big)

    See :func:`tau_jamaluddin` for the definition and the limits of validity:
    ``Rw`` should come from a thin ship method (``"michell"``), and the
    interference fades out above Fn = 0.65 and S/L = 0.4.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl : list of float, Waterline length of each hull [m].
    y : list of float, Transverse position of each hull [m].
    g : float, Gravitational acceleration [m/s²].
    Rw : list of float, Individual (thin ship) wave resistance of each hull [N].

    Returns
    -------
    list of list, Interference matrix dRw[i][j] [N], zero diagonal.
    """
    n = len(Lwl)
    dRw = [[0. for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Fn = froude_number(stw=stw, L=Lwl[i], g=g)
        for j in range(n):
            if i != j:
                SL = separation_to_length(y[i], y[j], Lwl[i], Lwl[j])
                dRw[i][j] = Rw[i] * (tau_jamaluddin(Fn=Fn, SL=SL) - 1.)
    return dRw


def compute_dRf_jamaluddin(
        Lwl,
        volume,
        y,
        Rf,
        **kwargs,
    ):
    r"""Viscous interference from the regression of Jamaluddin.

    .. math::
        \Delta R_{f,i} = R_{f,i}\sum_{j\neq i}
            \frac{\Delta(1+k)}{1+k}\bigg|_{S_{ij}/L_{ij}}

    see :func:`viscous_interference_ratio`. For S/L = 0.2, 0.3, 0.4 and
    L/∇^(1/3) = 7.4 this is +3.4 %, +2.6 % and +2.1 % of the viscous
    resistance, close to the Miyazawa estimate (see ``miyazawa``). A CFD study
    of a slender catamaran reports a larger form factor increase (1.254 to
    1.420), so these figures may be on the low side.

    Parameters
    ----------
    Lwl : list of float, Waterline length of each hull [m].
    volume : list of float, Displaced volume of each hull [m³].
    y : list of float, Transverse position of each hull [m].
    Rf : list of float, Individual viscous (frictional) resistance of each hull [N].

    Returns
    -------
    list of float, Viscous interference of each hull dRf[i] [N].
    """
    n = len(Lwl)
    dRf = [0. for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                SL = separation_to_length(y[i], y[j], Lwl[i], Lwl[j])
                dRf[i] = dRf[i] + Rf[i] * viscous_interference_ratio(
                    Lwl=Lwl[i], volume=volume[i], SL=SL)
    return dRf
