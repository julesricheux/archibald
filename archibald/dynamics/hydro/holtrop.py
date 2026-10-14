# -*- coding: utf-8 -*-
"""
HOLTROP-MENNEN 1984 METHOD FUNCTIONS
"""

#%% DEPENDENCIES

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.toolbox.math_utils import ReLU
from archibald.dynamics.hydro.common import Cf_hull, transom_resistance

#%% HELPER

#TODO move to share in toolbox, or NumPy

def smooth_clip(x, a, b, beta=10.0):
    r"""
    Smooth and differentiable clipping function bounded between [a, b] 
    using np.softplus, suitable for gradient-based optimization (IPOPT).
    
    Parameters
    ----------
    x : float or ndarray, Input value(s) to clip.
    a : float, Lower bound.
    b : float, Upper bound.
    beta : float, Steepness factor for the transition smoothing (higher = sharper).
    
    Returns
    -------
    float or ndarray, Smoothed and clipped values.
    """
    # 1. Smooth lower bound (smooth maximum with 'a')
    # Equivalent to: max(x, a) smoothly smoothed out
    x_lower = a + np.softplus(x - a, beta=beta)
    
    # 2. Smooth upper bound (smooth minimum with 'b')
    # Derived via symmetry: min(y, b) = -max(-y, -b)
    x_clipped = b - np.softplus(b - x_lower, beta=beta)
    
    return x_clipped

#%% INDIVIDUAL COEFFICIENTS

def c_14(
        Cstern,
        **kwargs,
    ):
    r"""Coefficient c14 used in form factor prediction.

    According to :cite:`holtrop1982approximate`, p. 166:

    .. math::
        c_{14} = 1 + 0.011\,C_{\text{stern}}

    Parameters
    ----------
    Cstern : float, Stern shape parameter [-].

    Returns
    -------
    float, Coefficient c14 [-].
    """
    return 1.0 + 0.011 * Cstern


def L_R(
        Lwl,
        Cp,
        lcb,
        **kwargs,
    ):
    r"""Stern run length LR used in form factor prediction.

    According to :cite:`holtrop1982approximate`, p. 166:

    .. math::
        L_R = L_{WL} \left(1 - C_P + \frac{0.06\,C_P\,\text{lcb}}{4\,C_P - 1}\right)

    Parameters
    ----------
    Lwl : float, Waterline length [m].
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lwl from midship [-].

    Returns
    -------
    float, Stern run length LR [m].
    """
    return Lwl * (1 - Cp + 0.06 * Cp * lcb / (4 * Cp - 1))


def one_plus_k(
        Lwl,
        Bwl,
        T,
        volume,
        Cp,
        Cstern,
        lcb,
        **kwargs,
    ):
    r"""Form factor (1+k) for frictional resistance prediction.

    According to :cite:`holtrop1982approximate`, p. 166:

    .. math::
        1 + k = 0.93 + 0.487118\,c_{14}
            \left(\frac{B}{L}\right)^{1.06806}
            \left(\frac{T}{L}\right)^{0.46106}
            \left(\frac{L}{L_R}\right)^{0.121563}
            \left(\frac{L^3}{\nabla}\right)^{0.36486}
            \left(1 - C_P\right)^{-0.604247}

    Parameters
    ----------
    Lwl : float, Waterline length [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    volume : float, Displaced volume [m³].
    Cp : float, Prismatic coefficient [-].
    Cstern : float, Stern shape parameter [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lwl from midship [-].

    Returns
    -------
    float, Form factor (1+k) [-].
    """
    _c14 = c_14(Cstern=Cstern)
    _Lr  = L_R(Lwl=Lwl, Cp=Cp, lcb=lcb)
    return (
        0.93
        + 0.487118 * _c14
        * (Bwl / (Lwl + 1e-12)) ** 1.06806
        * (T / (Lwl + 1e-12)) ** 0.46106
        * (Lwl / (_Lr + 1e-12)) ** 0.121563
        * (Lwl ** 3 / (volume + 1e-12)) ** 0.36486
        * (1 - Cp) ** (-0.604247)
    )


def C_A(
        Lwl,
        **kwargs,
    ):
    r"""Model-ship correlation allowance coefficient CA.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        C_A = 0.00675\,(L + 100)^{-1/3} - 0.00064

    Parameters
    ----------
    Lwl : float, Waterline length [m].

    Returns
    -------
    float, Correlation allowance coefficient CA [-].
    """
    return 0.00675 * (Lwl + 100) ** (-1/3) - 0.00064


def Re(
        stw,
        Lwl,
        nu,
        **kwargs,
    ):
    r"""Reynolds number for frictional resistance prediction.

    .. math::
        \text{Re} = \frac{V\,L}{\nu}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl : float, Waterline length [m].
    nu : float, Kinematic viscosity of water [m²/s].

    Returns
    -------
    float, Reynolds number [-].
    """
    return (np.softplus(stw * u.kt, beta=1e3) * Lwl) / nu


def c_7(
        Lbp,
        Bwl,
        **kwargs,
    ):
    r"""Coefficient c7 used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_7 = \begin{cases}
            0.229577 \left(\frac{B}{L}\right)^{1/3} & \text{if } \frac{B}{L} < 0.11 \\
            \frac{B}{L} & \text{if } 0.11 \leq \frac{B}{L} \leq 0.25 \\
            0.5 - 0.0625 \frac{L}{B} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].

    Returns
    -------
    float, Coefficient c7 [-].
    """
    
    Bwl_to_Lbp = Bwl / (Lbp + 1e-12)
    Lbp_to_Bwl = Lbp / (Bwl + 1e-12)
    
    return np.where(
        Bwl_to_Lbp < 0.11,
        0.229577 * Bwl_to_Lbp ** 0.3333,
        np.where(
            Bwl_to_Lbp <= 0.25,
            Bwl_to_Lbp,
            0.5 - 0.0625 * Lbp_to_Bwl
        )
    )


def c_1(
        Lbp,
        Bwl,
        T,
        ie,
        **kwargs,
    ):
    r"""Coefficient c1 used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_1 = 2223105\,c_7^{3.78613}
            \left(\frac{T}{B}\right)^{1.07961}
            \left(90 - i_E\right)^{-1.37565}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    ie : float, Half angle of entrance of waterline [°].

    Returns
    -------
    float, Coefficient c1 [-].
    """
    _c7 = c_7(Lbp=Lbp, Bwl=Bwl)
    return (
        2223105
        * _c7 ** 3.78613
        * (T / (Bwl + 1e-12)) ** 1.07961
        * (90 - ie) ** (-1.37565)
    )


def c_3(
        Bwl,
        T,
        Abt,
        hB,
        **kwargs,
    ):
    r"""Coefficient c3 accounting for bulbous bow effect on wave resistance.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_3 = \frac{0.56\,A_{BT}^{1.5}}{B\,T\,\left(0.31\sqrt{A_{BT}} + T - h_B\right)}

    Parameters
    ----------
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    Abt : float, Transverse area of bulbous bow [m²].
    hB : float, Centre of bulbous bow above keel [m].

    Returns
    -------
    float, Coefficient c3 [-].
    """
    return (
        (0.56 * Abt) ** 1.5
        / (Bwl * T * (0.31 * np.sqrt(Abt + 1e-12) + T - hB) + 1e-12)
    )


def c_2(
        Bwl,
        T,
        Abt,
        hB,
        **kwargs,
    ):
    r"""Coefficient c2 accounting for bulbous bow effect on wave resistance.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_2 = \exp\left(-1.89\,\sqrt{c_3}\right)

    Parameters
    ----------
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    Abt : float, Transverse area of bulbous bow [m²].
    hB : float, Centre of bulbous bow above keel [m].

    Returns
    -------
    float, Coefficient c2 [-].
    """
    return np.exp(-1.89 * np.sqrt(c_3(Bwl=Bwl, T=T, Abt=Abt, hB=hB)))


def c_5(
        Bwl,
        T,
        Cx,
        Atr,
        **kwargs,
    ):
    r"""Coefficient c5 accounting for transom stern effect on wave resistance.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_5 = 1 - 0.8 \frac{A_{TR}}{B\,T\,C_X}

    Parameters
    ----------
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    Cx : float, Midship section coefficient [-].
    Atr : float, Transom area at rest waterline [m²].

    Returns
    -------
    float, Coefficient c5 [-].
    """
    return 1 - 0.8 * (Atr / (Bwl * T * Cx + 1e-12))


def c_15(
        Lbp,
        volume,
        **kwargs,
    ):
    r"""Coefficient c15 used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_{15} = \begin{cases}
            -1.69385 & \text{if } \frac{L^3}{\nabla} < 512 \\
            -1.69385 + \frac{L / \nabla^{1/3} - 8}{2.36}
            & \text{if } 512 \leq \frac{L^3}{\nabla} < 1726.91 \\
            0 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    volume : float, Displaced volume [m³].

    Returns
    -------
    float, Coefficient c15 [-].
    """
    L3_V = Lbp ** 3 / (volume + 1e-12)
    return np.where(
        L3_V < 512,
        -1.69385,
        np.where(
            L3_V < 1726.91,
            -1.69385 + ((Lbp / (volume + 1e-12) ** (1/3)) - 8) / 2.36,
            0.
        )
    )


def c_16(
        Cp,
        **kwargs,
    ):
    r"""Coefficient c16 used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        c_{16} = \begin{cases}
            8.07981\,C_P - 13.8673\,C_P^2 + 6.984388\,C_P^3
            & \text{if } C_P < 0.8 \\
            1.73014 - 0.7067\,C_P & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].

    Returns
    -------
    float, Coefficient c16 [-].
    """
    return np.where(
        Cp < 0.8,
        8.07981 * Cp - 13.8673 * Cp ** 2 + 6.984388 * Cp ** 3,
        1.73014 - 0.7067 * Cp
    )


def c_17(
        Lbp,
        Bwl,
        volume,
        Cx,
        **kwargs,
    ):
    r"""Coefficient c17 used in high-speed wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        c_{17} = 6919.3\,C_X^{-1.3346}
            \left(\frac{\nabla}{L^3}\right)^{2.00977}
            \left(\frac{L}{B} - 2\right)^{1.40692}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    volume : float, Displaced volume [m³].
    Cx : float, Midship section coefficient [-].

    Returns
    -------
    float, Coefficient c17 [-].
    """
    safe_Lbp_to_Bwl = 1e-12 + np.softplus(Lbp / (Bwl + 1e-12) - 2, beta=1e3)
    safe_Lbp = Lbp + 1e-12
    return (
        6919.3
        * (Cx) ** (-1.3346)
        * (volume / safe_Lbp ** 3) ** 2.00977
        * safe_Lbp_to_Bwl ** 1.40692
    )

# def c_17(
#         Lbp,
#         Bwl,
#         volume,
#         Cx,
#         **kwargs,
#     ):
#     r"""Coefficient c17 used in high-speed wave-making resistance prediction.

#     According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

#     .. math::
#         c_{17} = 6919.3\,C_X^{-1.3346}
#             \left(\frac{\nabla}{L^3}\right)^{2.00977}
#             \left(\frac{L}{B} - 2\right)^{1.40692}

#     Parameters
#     ----------
#     Lbp : float, Length between perpendiculars [m].
#     Bwl : float, Waterline beam [m].
#     volume : float, Displaced volume [m³].
#     Cx : float, Midship section coefficient [-].

#     Returns
#     -------
#     float, Coefficient c17 [-].
#     """
#     # 1. Guarantee strictly positive bases using softplus. 
#     # Even if Cx, volume, or dimensions become negative or zero during extreme 
#     # optimizer iterations, softplus smoothly floors them to a safe positive value (> 0).
#     safe_Cx = 0.05 + np.softplus(Cx - 0.05, beta=50.0)
#     safe_volume = volume
#     safe_Lbp = 0.1 + np.softplus(Lbp - 0.1, beta=50.0)
#     safe_Bwl = 0.05 + np.softplus(Bwl - 0.05, beta=50.0)

#     # 2. Safely bound the L/B - 2 term above a positive epsilon
#     lbp_to_bwl_ratio = safe_Lbp / safe_Bwl - 2.0
#     safe_lbp_to_bwl = 1e-6 + np.softplus(lbp_to_bwl_ratio - 1e-6, beta=100.0)
    
#     # 3. Evaluate with guaranteed positive bases (no negative-power explosions)
#     return (
#         6919.3
#         * safe_Cx ** (-1.3346)
#         * (safe_volume / safe_Lbp ** 3) ** 2.00977
#         * safe_lbp_to_bwl ** 1.40692
#     )


def m_1(
        Lbp,
        Bwl,
        T,
        volume,
        Cp,
        **kwargs,
    ):
    r"""Coefficient m1 used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        m_1 = 0.014047 \frac{L}{T} - \frac{1.75254\,\nabla^{1/3}}{L}
            - 4.79323 \frac{B}{L} - c_{16}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    volume : float, Displaced volume [m³].
    Cp : float, Prismatic coefficient [-].

    Returns
    -------
    float, Coefficient m1 [-].
    """
    Lbp_to_T = (Lbp / (T + 1e-1))
    return (
        0.014047 * Lbp_to_T
        - (1.75254 * (volume + 1e-3) ** (1/3)) / (Lbp + 1e-12)
        - 4.79323 * ((Bwl + 1e-12) / (Lbp + 1e-12))
        - c_16(Cp=Cp)
    )


def m_3(
        Lbp,
        Bwl,
        T,
        **kwargs,
    ):
    r"""Coefficient m3 used in high-speed wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        m_3 = -7.2035
            \left(\frac{B}{L}\right)^{0.326869}
            \left(\frac{T}{B}\right)^{0.605375}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].

    Returns
    -------
    float, Coefficient m3 [-].
    """
    return (
        -7.2035
        * (Bwl / (Lbp + 1e-12)) ** 0.326869
        * (T / (Bwl + 1e-12)) ** 0.605375
    )


def m_4(
        Fr,
        Lbp,
        volume,
        **kwargs,
    ):
    r"""Coefficient m4 used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        m_4 = c_{15} \cdot 0.4\,\exp\left(-0.034\,F_r^{-3.29}\right)

    Parameters
    ----------
    Fr : float, Froude number [-].
    Lbp : float, Length between perpendiculars [m].
    volume : float, Displaced volume [m³].

    Returns
    -------
    float, Coefficient m4 [-].
    """
    return c_15(Lbp=Lbp, volume=volume) * 0.4 * np.exp(-0.034 * Fr ** (-3.29))


def lambda_(
        Lbp,
        Bwl,
        Cp,
        **kwargs,
    ):
    r"""Coefficient lambda used in wave-making resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 271:

    .. math::
        \lambda = \begin{cases}
            1.446\,C_P - 0.03 \frac{L}{B} & \text{if } \frac{L}{B} < 12 \\
            1.446\,C_P - 0.36 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    Cp : float, Prismatic coefficient [-].

    Returns
    -------
    float, Coefficient lambda [-].
    """
    Lbp_to_Bwl = Lbp / (Bwl + 1e-12)
    return np.where(
        Lbp_to_Bwl < 12,
        1.446 * Cp - 0.03 * Lbp_to_Bwl,
        1.446 * Cp - 0.36
    )


def Fr_i(
        stw,
        T,
        hB,
        Abt,
        g,
        **kwargs,
    ):
    r"""Froude number based on bulb immersion for bulbous bow resistance.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        F_{r_i} = \frac{V}{\sqrt{g\,(T - h_B - 0.25\,\sqrt{A_{BT}})
            + 0.15\,V^2}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    T : float, Mean moulded draught [m].
    hB : float, Centre of bulbous bow above keel [m].
    Abt : float, Transverse area of bulbous bow [m²].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Bulb Froude number Fri [-].
    """
    Vms = stw * u.kt
    return Vms / np.sqrt(g * (T - hB - 0.25 * np.sqrt(Abt)) + 0.15 * Vms ** 2)


def p_b(
        T,
        hB,
        Abt,
        **kwargs,
    ):
    r"""Emergence coefficient pb for bulbous bow resistance.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        p_b = \frac{0.56\,\sqrt{A_{BT}}}{T - 1.5\,h_B}

    Parameters
    ----------
    T : float, Mean moulded draught [m].
    hB : float, Centre of bulbous bow above keel [m].
    Abt : float, Transverse area of bulbous bow [m²].

    Returns
    -------
    float, Emergence coefficient pb [-].
    """
    return 0.56 * np.sqrt(Abt) / ((T - 1.5 * hB) + 1e-12)


def Fr_T(
        stw,
        Ttr,
        g,
        **kwargs,
    ):
    r"""Transom Froude number for transom resistance prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        F_{r_T} = \frac{V}{\sqrt{g\,T_{TR}}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Ttr : float, Transom draught [m].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Transom Froude number FrT [-].
    """
    return (stw * u.kt) / np.sqrt(g * Ttr)


def c_tr(
        stw,
        Ttr,
        g,
        **kwargs,
    ):
    r"""Transom resistance coefficient ctr.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        c_{tr} = \text{ReLU}\left(0.2\,\left(1 - 0.2\,F_{r_T}\right)\right)

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Ttr : float, Transom draught [m].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Transom resistance coefficient ctr [-].
    """
    return ReLU(0.2 * (1 - 0.2 * Fr_T(stw=stw, Ttr=Ttr, g=g)))


def c_8(
        Lbp,
        Bwl,
        T,
        D,
        Aws,
        Atr,
        **kwargs,
    ):
    r"""Coefficient c8 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_8 = \begin{cases}
            \frac{B \cdot S}{L \cdot D \cdot T} & \text{if } \frac{B}{T} < 5 \\
            \frac{S \left(7 \frac{B}{T} - 25\right)}{L \cdot D \left(\frac{B}{T} - 3\right)}
            & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Aws : float, Wetted surface area [m²].
    Aws : float, Transom area [m²].

    Returns
    -------
    float, Coefficient c8 [-].
    """
    A = Aws - Atr
    
    return np.where(
        Bwl / T < 5.0,
        Bwl * A / (Lbp * D * T),
        A * (7.0 * (Bwl / T) - 25.0) / (Lbp * D * ((Bwl / T) - 3.0))
    )


def c_9(
        Lbp,
        Bwl,
        T,
        D,
        Aws,
        Atr,
        **kwargs,
    ):
    r"""Coefficient c9 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_9 = \begin{cases}
            c_8 & \text{if } c_8 < 28 \\
            32 - \frac{16}{c_8 - 24} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Aws : float, Wetted surface area [m²].
    Atr : float, Transom area [m²].

    Returns
    -------
    float, Coefficient c9 [-].
    """
    _c8 = c_8(Lbp=Lbp, Bwl=Bwl, T=T, D=D, Aws=Aws, Atr=Atr)
    return np.where(
        _c8 < 28.0,
        _c8,
        32.0 - 16.0 / (_c8 - 24.0)
    )


def c_11(
        T,
        D,
        **kwargs,
    ):
    r"""Coefficient c11 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_{11} = \begin{cases}
            \frac{T}{D} & \text{if } \frac{T}{D} < 2 \\
            \frac{1}{12} \left(\frac{T}{D}\right)^3 + \frac{4}{3} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].

    Returns
    -------
    float, Coefficient c11 [-].
    """
    return np.where(
        T / D < 2.0,
        T / D,
        0.0833333 * (T / D) ** 3.0 + 1.33333
    )


def c_19(
        Cp,
        Cb,
        Cx,
        **kwargs,
    ):
    r"""Coefficient c19 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_{19} = \begin{cases}
            \frac{0.12997}{0.95 - C_B} - \frac{0.11056}{0.95 - C_P}
            & \text{if } C_P < 0.7 \\
            \frac{0.18567}{1.3571 - C_X} - 0.71276 + 0.38648\,C_P
            & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    Cb : float, Block coefficient [-].
    Cx : float, Midship section coefficient [-].

    Returns
    -------
    float, Coefficient c19 [-].
    """
    return np.where(
        Cp < 0.7,
        0.12997 / (0.95 - Cb) - 0.11056 / (0.95 - Cp),
        0.18567 / (1.3571 - Cx) - 0.71276 + 0.38648 * Cp
    )


def c_20(
        Cstern,
        **kwargs,
    ):
    r"""Coefficient c20 used in wake and thrust deduction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_{20} = 1 + 0.015\,C_{\text{stern}}

    Parameters
    ----------
    Cstern : float, Stern shape parameter [-].

    Returns
    -------
    float, Coefficient c20 [-].
    """
    return 1.0 + 0.015 * Cstern


def C_P1(
        Cp,
        lcb,
        **kwargs,
    ):
    r"""Modified prismatic coefficient Cp1 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        C_{P1} = 1.45\,C_P - 0.315 - 0.0225\,\text{lcb}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].

    Returns
    -------
    float, Modified prismatic coefficient Cp1 [-].
    """
    return 1.45 * Cp - 0.315 - 0.0225 * lcb


def C_V(
        stw,
        Lbp,
        Bwl,
        T,
        volume,
        Cp,
        Cstern,
        lcb,
        rho,
        g,
        **kwargs,
    ):
    r"""Viscous resistance coefficient CV.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        C_V = \left(1 + k\right) C_F + C_A

    where the form factor :math:`(1+k)` is given by
    :cite:`holtrop1982approximate`, p. 166:

    .. math::
        1 + k = 0.93 + 0.487118\,c_{14}
            \left(\frac{B}{L}\right)^{1.06806}
            \left(\frac{T}{L}\right)^{0.46106}
            \left(\frac{L}{L_R}\right)^{0.121563}
            \left(\frac{L^3}{\nabla}\right)^{0.36486}
            \left(1 - C_P\right)^{-0.604247}

    and the correlation allowance :math:`C_A` by
    :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        C_A = 0.00675\,(L + 100)^{-1/3} - 0.00064

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    volume : float, Displaced volume [m³].
    Cp : float, Prismatic coefficient [-].
    Cstern : float, Stern shape parameter [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Viscous resistance coefficient CV [-].
    """
    c14 = c_14(Cstern)
    Lr = Lbp * (1 - Cp + 0.06 * Cp * lcb / (4 * Cp - 1))
    one_plus_k = (
        0.93
        + 0.487118 * c14
        * (Bwl / Lbp) ** 1.06806
        * (T / Lbp) ** 0.46106
        * (Lbp / Lr) ** 0.121563
        * (Lbp ** 3 / volume) ** 0.36486
        * (1 - Cp) ** (-0.604247)
    )
    CA = C_A(Lbp)
    Re = (np.softplus(stw * u.kt, beta=1e3) * Lbp) / 1.2e-6
    return one_plus_k * Cf_hull(Re + 10.0) + CA

#%% RESISTANCE COMPONENTS

def compute_Rf_holtrop(
        stw,
        Lbp,
        Lwl,
        volume,
        Bwl,
        T,
        Aws,
        Atr,
        Cp,
        lcb,
        Csternchoice,
        nu,
        rho,
        uInterval=False,
        **kwargs,
    ):
    """
    Calculate the frictional resistance of a ship using the ITTC '57 method
    and Holtrop form factor (1+k).

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Lwl : float, Waterline length [m].
    volume : float, Displaced volume [m³].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    Aws : float, Wetted surface area [m²].
    Atr : float, Transom area [m²].
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lwl from midship [-].
    Csternchoice : int, Stern form (1=transom, 2=V-shaped, 3=U-shaped, 4=Spoon).
    nu : float, Kinematic viscosity of water [m²/s].
    rho : float, Water density [kg/m³].
    uInterval : bool, Whether to return min/max uncertainty intervals.

    Returns
    -------
    float or tuple, Frictional resistance [N], or (RfMin, RfMax) if uInterval=True.
    """
    Vms = stw * u.kt

    if Csternchoice == 1:
        Cstern = -25.
    elif Csternchoice == 2:
        Cstern = -10.
    elif Csternchoice == 3:
        Cstern = 0.
    elif Csternchoice == 4:
        Cstern = 10.
    else:
        Cstern = 0.
        
    A = Aws - Atr

    _1pk    = one_plus_k(Lwl=Lwl, Bwl=Bwl, T=T, volume=volume, Cp=Cp, Cstern=Cstern, lcb=lcb)
    _CA     = C_A(Lwl=Lwl)
    _Re     = Re(stw=stw, Lwl=Lwl, nu=nu)
    sigma_k = (_1pk - 1) * 0.046

    dynamic_pressure = 0.5 * rho * A * Vms ** 2
    Rf = _1pk * Cf_hull(_Re + 1e1) * dynamic_pressure
    
    if not np.is_casadi_type(Rf):
        Rf = np.squeeze(Rf)
        
    return Rf

    # TODO implement correctly
    # if uInterval:
    #     RfMin = (_1pk - 2 * sigma_k) * Cf_hull(_Re + 10.) * dynamic_pressure
    #     RfMax = (_1pk + 2 * sigma_k) * Cf_hull(_Re + 10.) * dynamic_pressure
    #     return RfMin, RfMax

    


def compute_Rw_holtrop(
        stw,
        Lwl,
        Lbp,
        Bwl,
        T,
        volume,
        Abt,
        Cp,
        Cwp,
        Atr,
        lcb,
        hB,
        Cx,
        ie,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculates the wave-making resistance of a ship in calm water.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl : float, Waterline length [m].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    volume : float, Displaced volume [m³].
    Abt : float, Transverse area of bulbous bow [m²].
    Cp : float, Prismatic coefficient [-].
    Cwp : float, Waterplane area coefficient [-].
    Atr : float, Transom area at rest waterline [m²].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    hB : float, Centre of bulbous bow above keel [m].
    Cx : float, Midship section coefficient [-].
    ie : float, Half angle of entrance of waterline [°].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wave-making resistance [N].
    """
    Vms = stw * u.kt
    Fr   = np.softplus(Vms, beta=1e3) / (np.sqrt(g * Lbp + 1e-12) + 1e-12) + 1e-12
    d_   = -0.9
    
    _c1  = 0.
    _c2  = 0.
    _c5  = 0.
    _c17 = 0.
    _m1  = 0.
    _m3  = 0.
    _m4  = 0.
    _l   = 0.

    _c1  = c_1(Lbp=Lbp, Bwl=Bwl, T=T, ie=ie)
    _c2  = c_2(Bwl=Bwl, T=T, Abt=Abt, hB=hB)
    _c5  = c_5(Bwl=Bwl, T=T, Cx=Cx, Atr=Atr)
    _c17 = c_17(Lbp=Lbp, Bwl=Bwl, volume=volume, Cx=Cx)
    _m1  = m_1(Lbp=Lbp, Bwl=Bwl, T=T, volume=volume, Cp=Cp)
    _m3  = m_3(Lbp=Lbp, Bwl=Bwl, T=T)
    _m4  = m_4(Fr=Fr, Lbp=Lbp, volume=volume)
    _l   = lambda_(Lbp=Lbp, Bwl=Bwl, Cp=Cp)

    base = _c2 * _c5 * volume * rho * g

    Rw_to_040  = _c1  * base * np.exp(_m1 * Fr ** d_  + _m4 * np.cos(_l * Fr ** (-2)))
    Rw_from_055 = _c17 * base * np.exp(_m3 * Fr ** d_ + _m4 * np.cos(_l * Fr ** (-2)))

    _m4_040 = m_4(Fr=0.40, Lbp=Lbp, volume=volume)
    _m4_055 = m_4(Fr=0.55, Lbp=Lbp, volume=volume)

    Rw_anchor_low  = _c1  * base * np.exp(_m1 * 0.40 ** d_ + _m4_040 * np.cos(_l * 0.40 ** (-2)))
    Rw_anchor_high = _c17 * base * np.exp(_m3 * 0.55 ** d_ + _m4_055 * np.cos(_l * 0.55 ** (-2)))

    Rw_from_040_to_055 = Rw_anchor_low + ((Fr - 0.40) / (0.55 - 0.40)) * (Rw_anchor_high - Rw_anchor_low)

    is_below_040 = np.where(Fr <= 0.40, 1., 0.)
    is_below_055 = np.where(Fr <= 0.55, 1., 0.)
    
    Rw = Rw_to_040 * is_below_040 + \
         Rw_from_040_to_055 * (1. - is_below_040) * is_below_055 + \
         Rw_from_055 * (1. - is_below_040) * (1. - is_below_055)
         
    return Rw


def compute_Rb_holtrop(
        stw,
        T,
        hB,
        Abt,
        Bulbchoice,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculates the resistance due to bulbous bow using Holtrop's method.
    """
    Vms = stw * u.kt
    
    if Bulbchoice == 1:
        Fri = Fr_i(stw, T, hB, Abt, g)
        pb = p_b(T, hB, Abt)
        Fri = Vms / (np.sqrt((g * (T - hB - (0.25 * (np.sqrt(Abt))))) + (0.15 * (Vms ** 2))))
        Rb = 0.11 * (np.exp(((-3) * (pb ** (-2)))) * (Fri ** 3) * (Abt ** 1.5) * rho * g) / (1 + (Fri ** 2))
        return Rb
    else:
        return 0.


def compute_Rtr_holtrop(
        stw,
        Ttr,
        Atr,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculate transom resistance using the Holtrop-Mennen method.
    """
    
    Vms = stw * u.kt
    
    Fr_T = Vms / (np.sqrt(g * Ttr + 1e-12) + 1e-12)
    
    return transom_resistance(
            Vms,
            Fr_T,
            Atr,
            rho,
        )


def compute_Ra_holtrop(
        stw,
        Lwl,
        Bwl,
        T,
        Cb,
        hB,
        rho,
        Aws,
        Atr,
        Abt,
        uInterval=False,
        **kwargs,
    ):
    """
    Calculates the model-ship correlation resistance RA.
    """
    Vms = stw * u.kt
    
    A = Aws - Atr
    
    CA = C_A(Lwl)
    sigmaCA = 0.00021
        
    # TODO implement correctly
    # if uInterval:
    #     RaMin = 0.5 * rho * Aws * (Vms ** 2) * (CA - 2*sigmaCA)
    #     RaMax = 0.5 * rho * Aws * (Vms ** 2) * (CA + 2*sigmaCA)
    #     return RaMin, RaMax
    
    Ra = 0.5 * rho * A * (Vms ** 2) * CA
    return Ra

#%% PROPELLER COEFFICIENTS

def w_single(
        stw,
        Lbp,
        Bwl,
        T,
        T_A,
        Aws,
        Atr,
        Cp,
        Cb,
        Cx,
        D,
        Cstern,
        volume,
        lcb,
        rho,
        g,
        **kwargs,
    ):
    r"""Wake fraction prediction for single screw ships.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        w = c_9 c_{20} C_V \frac{L}{T_A} \left( 0.050776 + 0.93405 c_{11}
            \frac{C_V}{1 - C_{P1}} \right) + 0.27915 c_{20}
            \sqrt{\frac{B}{L \left(1 - C_{P1}\right)}} + c_{19} c_{20}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    T_A : float, Draught at aft perpendicular [m].
    Cp : float, Prismatic coefficient [-].
    Cb : float, Block coefficient [-].
    Cx : float, Midship section coefficient [-].
    D : float, Propeller diameter [m].
    Cstern : float, Stern shape parameter [-].
    volume : float, Displaced volume [m³].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wake fraction w [-].
    """
    _CV  = C_V(stw=stw, Lbp=Lbp, Bwl=Bwl, T=T, volume=volume, Cp=Cp, Cstern=Cstern, lcb=lcb, rho=rho, g=g)
    _c9  = c_9(Lbp=Lbp, Bwl=Bwl, T=T, D=D, Aws=Aws, Atr=Atr)
    _c11 = c_11(T=T, D=D)
    _c19 = c_19(Cp=Cp, Cb=Cb, Cx=Cx)
    _c20 = c_20(Cstern=Cstern)
    _CP1 = C_P1(Cp=Cp, lcb=lcb)

    return (
        _c9
        * _c20
        * _CV
        * (Lbp / T_A)
        * (0.050776 + 0.93405 * _c11 * (_CV / (1 - _CP1)))
        + 0.27915 * _c20 * np.sqrt(Bwl / (Lbp * (1 - _CP1)))
        + _c19 * _c20
    )


def eta_R_single(
        Cp,
        lcb,
        Ae_Ao,
        **kwargs,
    ):
    r"""Relative rotative efficiency prediction for single screw ships.

    According to :cite:`holtrop1982approximate`, p. 168:

    .. math::
        \eta_R = 0.9922 - 0.05908 \frac{A_E}{A_O}
            + 0.07424 \left(C_P - 0.0225\,\text{lcb}\right)

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    Ae_Ao : float, Expanded blade area ratio [-].

    Returns
    -------
    float, Relative rotative efficiency eta_R [-].
    """
    return (
        0.9922
        - 0.05908 * (Ae_Ao)
        + 0.07424 * (Cp - 0.0225 * lcb)
    )


def w_single_open_stern(
        stw,
        Lbp,
        Bwl,
        T,
        D,
        Cb,
        rho,
        g,
        **kwargs,
    ):
    r"""Wake fraction prediction for single screw ships with open stern.

    Applied to slender, fast ships. According to :cite:`holtrop1982approximate`, p. 169:

    .. math::
        w = 0.3 C_B + 10 C_V C_B - 0.23 \frac{D}{\sqrt{BT}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Cb : float, Block coefficient [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wake fraction w [-].
    """
    return (
        0.3 * Cb
        + 10.0 * C_V(stw, Lbp, Bwl, T, rho, g) * Cb
        - 0.23 * D / np.sqrt(Bwl * T)
    )


def t_single_open_stern(**kwargs):
    r"""Thrust deduction fraction for single screw ships with open stern.

    Applied to slender, fast ships. According to :cite:`holtrop1982approximate`, p. 169,
    a constant value is used:

    .. math::
        t = 0.10

    Returns
    -------
    float, Thrust deduction fraction t = 0.1 [-].
    """
    return 0.1


def eta_R_single_open_stern(**kwargs):
    r"""Relative rotative efficiency for single screw ships with open stern.

    Applied to slender, fast ships. According to :cite:`holtrop1982approximate`, p. 168,
    a constant value is used:

    .. math::
        \eta_R = 0.98

    Returns
    -------
    float, Relative rotative efficiency eta_R = 0.98 [-].
    """
    return 0.98


def w_twin(
        stw,
        Lbp,
        Bwl,
        T,
        D,
        Cb,
        rho,
        g,
        **kwargs,
    ):
    r"""Wake fraction prediction for twin screw ships.

    According to :cite:`holtrop1982approximate`, p. 169:

    .. math::
        w = 0.3095 C_B + 10 C_V C_B - 0.23 \frac{D}{\sqrt{BT}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Cb : float, Block coefficient [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wake fraction w [-].
    """
    return (
        0.3095 * Cb
        + 10.0 * C_V(stw, Lbp, Bwl, T, rho, g) * Cb
        - 0.23 * D / np.sqrt(Bwl * T)
    )


def t_twin(
        Bwl,
        T,
        D,
        Cb,
        **kwargs,
    ):
    r"""Thrust deduction fraction prediction for twin screw ships.

    According to :cite:`holtrop1982approximate`, p. 169:

    .. math::
        t = 0.325 C_B - 0.1885 \frac{D}{\sqrt{BT}}

    Parameters
    ----------
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float ,Propeller diameter [m].
    Cb : float ,Block coefficient [-].

    Returns
    -------
    float, Thrust deduction fraction t [-].
    """
    return (
        0.325 * Cb
        - 0.1885 * D / np.sqrt(Bwl * T)
    )


def eta_R_twin(
        Cp,
        lcb,
        P,
        D,
        **kwargs,
    ):
    r"""Relative rotative efficiency prediction for twin screw ships.

    According to :cite:`holtrop1982approximate`, p. 168:

    .. math::
        \eta_R = 0.9737 + 0.111 \left(C_P - 0.0225\,\text{lcb}\right)
            + 0.06325 \frac{P}{D}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    P : float, Propeller pitch [m].
    D : float, Propeller diameter [m].

    Returns
    -------
    float, Relative rotative efficiency eta_R [-].
    """
    return (
        0.9737
        + 0.111 * (Cp - 0.0225 * lcb)
        + 0.06325 * P / D
    )

#%% EXAMPLE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # VERIFICATION & PLOTTING (MOCK COMMERCIAL HULL)
    # ---------------------------------------------------------
    
    env_params = {
        'g': 9.81,         # m/s^2
        'rho': 1025.0,     # kg/m^3
        'nu': 1.189e-6,    # m^2/s (15 deg C seawater)
    }
    
    # Mock dimensions for a general cargo vessel / tanker
    hull_params = {
        'Lbp': 150.0,
        'Lwl': 155.0,
        'Bwl': 22.0,
        'T': 8.5,
        'volume': 19000.0,
        'Aws': 4500.0,
        'Cp': 0.65,
        'Cwp': 0.75,
        'Cx': 0.98,
        'Cb': 0.637,
        'lcb': -1.5,       # % of L forward of midships (negative means aft)
        'ie': 15.0,        # Half angle of entrance (degrees)
        'Csternchoice': 1, # Transom
        'Atr': 6.0,        # Transom immersed area [m^2]
        'Ttr': 0.5,        # Transom draft [m]
        'Bulbchoice': 1,
        'Abt': 12.0,       # Bulb cross-sectional area [m^2]
        'hB': 3.0,         # Height of centroid of bulb area [m]
        'uInterval': False
    }

    # Define Froude sweep (Holtrop generally valid for Fn < 0.40)
    fr_array = np.linspace(0.0, 1.0, 1000)
    
    Rf_list, Rw_list, Rb_list, Rtr_list, Ra_list = [], [], [], [], []
    
    for fr in fr_array:
        # Update speed dependent variables
        v_ms = fr * np.sqrt(env_params['g'] * hull_params['Lbp'])
        hull_params['stw'] = v_ms / u.kt
        
        # Compute individual components
        Rf  = compute_Rf_holtrop(**hull_params, **env_params)
        Rw  = compute_Rw_holtrop(**hull_params, **env_params)
        Rb  = compute_Rb_holtrop(**hull_params, **env_params)
        Rtr = compute_Rtr_holtrop(**hull_params, **env_params)
        Ra  = compute_Ra_holtrop(**hull_params, **env_params)
        
        Rf_list.append(Rf)
        Rw_list.append(Rw)
        Rb_list.append(Rb)
        Rtr_list.append(Rtr)
        Ra_list.append(Ra)
        
    Rf_arr  = np.array(Rf_list)
    Rw_arr  = np.array(Rw_list)
    Rb_arr  = np.array(Rb_list)
    Rtr_arr = np.array(Rtr_list)
    Ra_arr  = np.array(Ra_list)

    # ---------------------------------------------------------
    # STACKED AREA PLOT
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    layer1 = Rf_arr
    layer2 = layer1 + Rw_arr
    layer3 = layer2 + Rtr_arr
    layer4 = layer3 + Rb_arr
    layer5 = layer4 + Ra_arr
    
    ax.fill_between(fr_array, 0, layer1, color='#1f77b4', alpha=0.6, label='Frictional ($R_f + 1+k$)')
    ax.plot(fr_array, layer1, color='#1f77b4', linewidth=1)
    
    ax.fill_between(fr_array, layer1, layer2, color='#ff7f0e', alpha=0.6, label='Wave ($R_w$)')
    ax.plot(fr_array, layer2, color='#ff7f0e', linewidth=1)

    ax.fill_between(fr_array, layer2, layer3, color='#d62728', alpha=0.6, label='Transom ($R_{tr}$)')
    ax.plot(fr_array, layer3, color='#d62728', linewidth=1)
    
    ax.fill_between(fr_array, layer3, layer4, color='#9467bd', alpha=0.6, label='Bulbous Bow ($R_b$)')
    ax.plot(fr_array, layer4, color='#9467bd', linewidth=1)
    
    ax.fill_between(fr_array, layer4, layer5, color='#8c564b', alpha=0.6, label='Model Correlation ($R_a$)')
    ax.plot(fr_array, layer5, color='#8c564b', linewidth=1)
    
    ax.set_title("Holtrop & Mennen\nComponents of Resistance vs. Froude Number", fontsize=14, pad=15)
    ax.set_xlabel('Froude Number ($F_n$)', fontsize=12)
    ax.set_ylabel('Total Resistance (N)', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.xlim([fr_array.min(), fr_array.max()])
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()