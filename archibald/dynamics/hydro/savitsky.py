# -*- coding: utf-8 -*-
"""
SAVITSKY 1964 / SAVITSKY & BROWN 1976 PLANING METHOD FUNCTIONS -- DYNAMIC COMPONENTS ONLY

No hydrostatic component is computed here :
  - the buoyancy-like lift term of Savitsky's C_L0 (the one in lambda^2.5 / Cv^2, which carries g)
    is removed : only the dynamic lift  C_L0 = 0.012 tau^1.1 lambda^0.5  remains ;
  - the centre of pressure is the high speed (purely dynamic) limit  l_cp = 0.75 lambda b.
The hydrostatic force, moment and trim drag come from the hydrostatics module ;
Fz and My below are ADDED to them in the equilibrium (IPOPT).

Open-loop evaluation : nothing is solved here. The attitude (trim, sinkage) and the wetted
keel length at that attitude (Lwl, from the hydrostatics module) are inputs.

Wetted geometry (kwarg `wetted_lengths`) :
    "76" (default) Savitsky & Brown 1976, Eq. 12, 14, 15 : improved fit of the chine wetted length
    "64"           Savitsky 1964, Eq. 3
At the same attitude the '76 geometry gives about +4 % lambda and +8 % lift on the published
example : use "64" only to reproduce 1964 results.

Conventions : SI units, speeds in knots, ALL ANGLES IN DEGREES (like `ie`).
Resistance components (to be summed), each from its own independent function :
    Rf  = Df / cos(tau)          frictional   (depends on the wetted geometry and Vm only)
    Rw  = L_dyn tan(tau)         pressure     (depends on the dynamic lift only)
    Rsp                          whisker spray (viscous, 2007 extension)
    Raw                          added resistance in waves (Savitsky & Brown 1976), 0 if no swh
    Rtr = 0                      dry transom
Rf + Rw is Savitsky's  D = Delta tan(tau) + Df / cos(tau)  (the vertical friction component
Df sin(tau) is carried by the pressure force, which is why Rf has the 1/cos(tau)).
Hydrodynamic wrench   : Fz, My       (NOT resistances, never to be summed with them).

Range of validity of the original method (total lift) :
0.6 <= Cv <= 13, 2 deg <= trim <= 15 deg, lambda <= 4, deadrise <= 30 deg.
The dynamic-only lift is the Cv -> infinity limit ; at low Cv it under-predicts the lift
unless the hydrostatics module supplies the missing buoyancy.
"""

#%% DEPENDENCIES

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.dynamics.hydro.common import Cf_hull

#%% CONSTANTS

_DEG = np.pi / 180.   # [rad/deg]
TAU_MIN = 0.1         # Smooth lower bound of the trim angle [deg]
LAMBDA_MIN = 0.01     # Smooth lower bound of the wetted length ratio [-]
TAU_SPRAY_MIN = 2.0   # Trim angle defining the lower bound of the spray line angle [deg]

#%% HELPER

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


def smooth_ramp(x, a, b, beta=10.0):
    """
    Smooth and differentiable ramp function from 0 to 1 over the interval [a, b],
    suitable for gradient-based optimization (IPOPT).
    """
    clipped = smooth_clip(x, a, b, beta=beta)
    return (clipped - a) / (b - a)


def validity_factor(
        stw,
        Bwl,
        g,
        trim,
        **kwargs,
    ):
    
    _Cv = C_v(stw=stw, Bwl=Bwl, g=g)
    _tau = tau(trim)
    
    beta = 20.
    rules = np.array([
        smooth_ramp(_Cv, 0.6, 1.0, beta=beta), # Cv > 0.60, better for Cv > 1.0
        # smooth_ramp(_tau, 0.0, 2.0, beta=beta), # tau > 2.0 deg
    ])
    
    # print(rules)
    
    return np.prod(rules)

#%% INDIVIDUAL COEFFICIENTS

def tau(
        trim,
        **kwargs,
    ):
    r"""Effective trim angle, smoothly bounded below to avoid the singularity at zero trim.

    .. math::
        \tau = \tau_{min} + \text{softplus}\left(\text{trim} - \tau_{min}\right)

    Parameters
    ----------
    trim : float, Trim angle, bow up positive [deg].

    Returns
    -------
    float, Effective trim angle tau [deg].
    """
    return TAU_MIN + np.softplus(trim - TAU_MIN, beta=10.)


def C_v(
        stw,
        Bwl,
        g,
        **kwargs,
    ):
    r"""Speed coefficient based on the chine beam (diagnostic : the dynamic lift does not use it).

    .. math::
        C_V = \frac{V}{\sqrt{g\,b}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Bwl : float, Beam at the chines [m].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Speed coefficient Cv [-].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    return Vms / np.sqrt(g * Bwl)


def lambda_K(
        Lwl,
        Bwl,
        rise_factor=1.,
        **kwargs,
    ):
    r"""Wetted keel length to chine beam ratio.

    .. math::
        \lambda_K = \frac{f_{rise}\,L_{WL}}{b}

    Parameters
    ----------
    Lwl : float, Wetted keel length at the current attitude (hydrostatics) [m].
    Bwl : float, Beam at the chines [m].
    rise_factor : float, Dynamic water rise-up factor applied to Lwl [-] (calibration).

    Returns
    -------
    float, Keel wetted length ratio [-].
    """
    return rise_factor * Lwl / Bwl


def w_wetted(
        trim,
        deadrise,
        **kwargs,
    ):
    r"""Chine wetted length offset of Savitsky & Brown (1976), Eq. 12.

    .. math::
        w = \left(0.57 + \frac{\beta}{1000}\right)
            \left(\frac{\tan\beta}{2\tan\tau} - \frac{\beta}{167}\right)

    Parameters
    ----------
    trim : float, Trim angle [deg].
    deadrise : float, Deadrise angle beta [deg].

    Returns
    -------
    float, Offset w [-].
    """
    _tau = tau(trim)
    return (0.57 + deadrise / 1000.) * (np.tan(deadrise * _DEG) / (2. * np.tan(_tau * _DEG)) - deadrise / 167.)


def lambda_C(
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        wetted_lengths="76",
        **kwargs,
    ):
    r"""Chine wetted length to chine beam ratio, smoothly bounded below by zero (dry chines).

    '76 (Savitsky & Brown 1976, Eq. 14) :

    .. math::
        \lambda_C = (\lambda_K - w) - 0.2\,e^{-(\lambda_K - w)/0.3}

    '64 (Savitsky 1964, Eq. 3) :

    .. math::
        \lambda_C = \lambda_K - \frac{\tan\beta}{\pi\tan\tau}

    Parameters
    ----------
    trim : float, Trim angle [deg].
    Lwl : float, Wetted keel length [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    rise_factor : float, Dynamic water rise-up factor [-].
    wetted_lengths : str, "76" or "64".

    Returns
    -------
    float, Chine wetted length ratio [-].
    """
    _lK = lambda_K(Lwl=Lwl, Bwl=Bwl, rise_factor=rise_factor)

    if wetted_lengths == "64":
        _raw = _lK - np.tan(deadrise * _DEG) / (np.pi * np.tan(tau(trim) * _DEG))
    else:
        _x = _lK - w_wetted(trim=trim, deadrise=deadrise)
        _raw = _x - 0.2 * np.exp(-_x / 0.3)

    return np.softplus(_raw, beta=50.)


def lambda_(
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        wetted_lengths="76",
        **kwargs,
    ):
    r"""Mean wetted length to chine beam ratio.

    Purely geometric : Lwl is the wetted keel length supplied by the hydrostatics module at the
    current attitude (no hydrostatic force is involved).

    '76 (Eq. 15) :

    .. math::
        \lambda = \frac{\lambda_K + \lambda_C}{2} + 0.03

    '64 :

    .. math::
        \lambda = \frac{\lambda_K + \lambda_C}{2}

    Parameters
    ----------
    trim : float, Trim angle, bow up positive [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    rise_factor : float, Dynamic water rise-up factor applied to Lwl [-] (calibration).
    lam : float, Optional imposed value of lambda (external variable) [-].
    wetted_lengths : str, "76" (default) or "64".

    Returns
    -------
    float, Mean wetted length ratio lambda [-].
    """
    if lam is not None:
        return lam

    _lK = lambda_K(Lwl=Lwl, Bwl=Bwl, rise_factor=rise_factor)
    _lC = lambda_C(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, rise_factor=rise_factor,
                   wetted_lengths=wetted_lengths)
    _lam = 0.5 * (_lK + _lC) + (0.03 if wetted_lengths != "64" else 0.)

    return LAMBDA_MIN + np.softplus(_lam - LAMBDA_MIN, beta=20.)


def x_s(
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        wetted_lengths="76",
        **kwargs,
    ):
    r"""Distance from the keel / waterline intersection to the start of the wetted chine.

    .. math::
        x_s = L_K - L_C \qquad (\text{imposed } \lambda : x_s = \tfrac{b}{\pi}\tan\beta/\tan\tau)

    Returns
    -------
    float, Distance x_s [m].
    """
    if lam is not None:
        return Bwl / np.pi * np.tan(deadrise * _DEG) / np.tan(tau(trim) * _DEG)

    _lK = lambda_K(Lwl=Lwl, Bwl=Bwl, rise_factor=rise_factor)
    _lC = lambda_C(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, rise_factor=rise_factor,
                   wetted_lengths=wetted_lengths)
    return Bwl * (_lK - _lC)


def A_wetted(
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        wetted_lengths="76",
        **kwargs,
    ):
    r"""Wetted bottom area : chine dry triangle + chine wet rectangle.

    .. math::
        S = \frac{b\,(L_K + L_C)}{2\cos\beta} \qquad (\text{imposed } \lambda : S = \lambda b^2 / \cos\beta)

    Returns
    -------
    float, Wetted bottom area [m²].
    """
    if lam is not None:
        return lam * Bwl ** 2 / np.cos(deadrise * _DEG)

    _lK = lambda_K(Lwl=Lwl, Bwl=Bwl, rise_factor=rise_factor)
    _lC = lambda_C(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, rise_factor=rise_factor,
                   wetted_lengths=wetted_lengths)
    return Bwl ** 2 * (_lK + _lC) / (2. * np.cos(deadrise * _DEG))


def chines_dry(
        stw,
        g,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Chines dry planing margin of Savitsky & Brown (1976), Eq. 3 (diagnostic / constraint).

    .. math::
        m = C_V^2 - \frac{\lambda - 0.16\tan\beta/\tan\tau}{3\sin\tau}

    m >= 0 : chines dry ; m < 0 : the chines are wetted along a side length L_C2.

    Returns
    -------
    float, Margin m [-].
    """
    _tau = tau(trim)
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _Cv = C_v(stw=stw, Bwl=Bwl, g=g)
    return _Cv ** 2 - (_lam - 0.16 * np.tan(deadrise * _DEG) / np.tan(_tau * _DEG)) / (3. * np.sin(_tau * _DEG))


def L_C2(
        stw,
        g,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Side chine wetted length of Savitsky & Brown (1976), Eq. 1 (zero when the chines are dry).

    .. math::
        L_{C2} = L_C - 3\,\frac{V^2 \sin\tau}{g}

    Returns
    -------
    float, Side wetted length [m].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lC = Bwl * lambda_C(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _L2 = np.fmax(0., _lC - 3. * Vms ** 2 * np.sin(tau(trim) * _DEG) / g)
    _m = chines_dry(stw=stw, g=g, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    return np.where(_m >= 0., 0., _L2)


def C_L0(
        trim,
        lam,
        **kwargs,
    ):
    r"""Dynamic lift coefficient of the flat plate (zero deadrise).

    Savitsky's :math:`C_{L_0} = \tau^{1.1}(0.012\lambda^{0.5} + 0.0055\lambda^{2.5}/C_V^2)`
    without its second (gravity dependent, hydrostatic) term:

    .. math::
        C_{L_0} = 0.012\,\tau^{1.1}\,\lambda^{0.5}

    Parameters
    ----------
    trim : float, Trim angle [deg].
    lam : float, Mean wetted length ratio [-].

    Returns
    -------
    float, Dynamic flat plate lift coefficient [-].
    """
    _tau = tau(trim)
    return 0.012 * _tau ** 1.1 * np.sqrt(lam)


def C_Lbeta(
        trim,
        lam,
        deadrise,
        **kwargs,
    ):
    r"""Dynamic lift coefficient of the deadrise surface.

    According to :cite:`savitsky1964hydrodynamic`:

    .. math::
        C_{L_\beta} = C_{L_0} - 0.0065\,\beta\,C_{L_0}^{0.6} \quad (\geq 0, \text{smooth floor})

    Parameters
    ----------
    trim : float, Trim angle [deg].
    lam : float, Mean wetted length ratio [-].
    deadrise : float, Deadrise angle beta [deg].

    Returns
    -------
    float, Dynamic deadrise lift coefficient [-].
    """
    _CL0 = C_L0(trim=trim, lam=lam)
    # smooth floor at zero : the empirical formula turns negative outside its validity range
    return np.softplus(_CL0 - 0.0065 * deadrise * _CL0 ** 0.6, beta=2000.)


def lcp(
        lam,
        Bwl,
        **kwargs,
    ):
    r"""Centre of pressure of the dynamic lift, measured from the transom along the keel line.

    Savitsky's :math:`l_{cp} = \lambda b\,(0.75 - 1/(5.21 C_V^2/\lambda^2 + 2.39))` moves aft at
    low speed because of the hydrostatic lift ; the purely dynamic limit is kept:

    .. math::
        l_{cp} = 0.75\,\lambda\,b

    Parameters
    ----------
    lam : float, Mean wetted length ratio [-].
    Bwl : float, Beam at the chines [m].

    Returns
    -------
    float, Centre of pressure abscissa from the transom [m].
    """
    return 0.75 * lam * Bwl


def V_m_ratio(
        trim,
        lam,
        deadrise,
        **kwargs,
    ):
    r"""Mean bottom velocity to speed ratio.

    According to :cite:`savitsky1964hydrodynamic`:

    .. math::
        \frac{V_m}{V} = \sqrt{1 - \frac{C_{L_\beta}}{\lambda\cos\tau}}

    Parameters
    ----------
    trim : float, Trim angle [deg].
    lam : float, Mean wetted length ratio [-].
    deadrise : float, Deadrise angle beta [deg].

    Returns
    -------
    float, Ratio Vm / V [-].
    """
    _tau = tau(trim)
    _CLb = C_Lbeta(trim=trim, lam=lam, deadrise=deadrise)
    return np.sqrt(np.fmax(0.05, 1. - _CLb / (lam * np.cos(_tau * _DEG))))


def L_K(
        Lwl,
        rise_factor=1.,
        **kwargs,
    ):
    r"""Wetted keel length (dynamic) : :math:`L_K = f_{rise}\,L_{WL}`.

    Returns
    -------
    float, Wetted keel length [m].
    """
    return rise_factor * Lwl


def L_C(
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Wetted chine length : :math:`L_C = \lambda_C\,b`.

    Returns
    -------
    float, Wetted chine length [m].
    """
    return Bwl * lambda_C(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)


def alpha_spray(
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Angle of the stagnation / spray root line relative to the keel (plan view).

    .. math::
        \alpha = \arctan\left(\frac{b}{2\,x_s}\right)

    With the '64 geometry this is the Wagner relation
    :math:`\alpha = \arctan(\frac{\pi}{2}\tan\tau/\tan\beta)`. It is evaluated with a trim of at
    least 2 deg (validity range of the spray model, below it A_as ~ 1 / sin(2 alpha) diverges)
    and smoothly bounded to 45 deg (the model is meant for deep-V hulls at small trim).

    Returns
    -------
    float, Spray root line angle alpha [deg].
    """
    _xs = np.fmax(x_s(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs), 1e-3 * Bwl)
    a = np.arctan(Bwl / (2. * _xs)) / _DEG
    a_min = np.arctan(np.pi / 2. * np.tan(TAU_SPRAY_MIN * _DEG) / np.tan(deadrise * _DEG)) / _DEG
    a = a_min + np.softplus(a - a_min, beta=5.)
    return 45. - np.softplus(45. - a, beta=0.5)


def A_spray(
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Whisker spray wetted area (forward of the stagnation line).

    .. math::
        A_{as} = \frac{b^2}{4\,\sin(2\alpha)\cos\beta}

    !! Trigonometric form read from secondary sources (see compute_Rsp_savitsky).

    Returns
    -------
    float, Whisker spray area [m²].
    """
    _alpha = alpha_spray(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    return Bwl ** 2 / (4. * np.sin(2. * _alpha * _DEG) * np.cos(deadrise * _DEG))


def phi_spray(
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Angle between the spray viscous force and the keel line.

    .. math::
        \varphi = \arctan\left(\frac{\tan\theta}{\cos\beta}\right), \qquad \theta = 2\alpha

    (the whisker spray leaves the stagnation line at about twice its angle).

    Returns
    -------
    float, Spray force angle phi [deg].
    """
    _alpha = alpha_spray(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    return np.arctan(np.tan(2. * _alpha * _DEG) / np.cos(deadrise * _DEG)) / _DEG


def blount_fox_M_prime(
        bf_M,
        bf_K=0.5,
        **kwargs,
    ):
    r"""Blount & Fox semi-planing correction factor.

    .. math::
        M' = K\,(M - 1) + 1

    Parameters
    ----------
    bf_M : float, Blount & Fox factor M [-] (user supplied).
    bf_K : float, Correlation factor [-] (0 = Savitsky, 0.5 = realistic hump, 1 = Blount & Fox).

    Returns
    -------
    float, Corrected factor M' [-].
    """
    return bf_K * (bf_M - 1.) + 1.


def L_dynamic(
        stw,
        rho,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Dynamic vertical lift (no hydrostatic part).

    .. math::
        L = C_{L_\beta}\,\tfrac{1}{2}\rho V^2 b^2

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    trim : float, Trim angle [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    Optional (kwargs) : rise_factor, lam, wetted_lengths.

    Returns
    -------
    float, Dynamic vertical lift [N].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _CLb = C_Lbeta(trim=trim, lam=_lam, deadrise=deadrise)

    return _CLb * 0.5 * rho * Vms ** 2 * Bwl ** 2


def D_f(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        dCf=0.,
        Aws=None,
        use_Aws=False,
        **kwargs,
    ):
    r"""Frictional force along the keel line (ITTC '57 on the mean wetted length).

    .. math::
        D_f = \tfrac{1}{2}\rho\,C_f(Re_m)\,V_m^2\,S, \qquad Re_m = \frac{V_m\,\lambda\,b}{\nu}

    Parameters
    ----------
    nu : float, Kinematic viscosity [m²/s].
    dCf : float, Roughness allowance [-].
    Aws : float, Hydrostatic wetted surface [m²] (used if use_Aws is True).
    use_Aws : bool, Use the hydrostatic wetted surface instead of the Savitsky wetted area.

    Returns
    -------
    float, Frictional force along the keel [N].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    Vm = Vms * V_m_ratio(trim=trim, lam=_lam, deadrise=deadrise)

    if use_Aws and Aws is not None:
        S, Lw = Aws, Lwl
    else:
        S = A_wetted(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
        Lw = _lam * Bwl

    Re = Vm * Lw / nu
    return 0.5 * rho * (Cf_hull(Re + 10.) + dCf) * Vm ** 2 * S


def N_normal(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Dynamic force normal to the keel line.

    .. math::
        N = \frac{L + D_f \sin\tau}{\cos\tau}

    Returns
    -------
    float, Normal force [N].
    """
    _tau = tau(trim)
    _L = L_dynamic(stw=stw, rho=rho, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _Df = D_f(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    return (_L + _Df * np.sin(_tau * _DEG)) / np.cos(_tau * _DEG)


# Added resistance in waves, Savitsky & Brown (1976) : regression in imperial units (ft, lbf, LT)
_FT = 3.28084     # [ft/m]
_LBF = 4.44822    # [N/lbf]
_LT = 9964.016    # [N/long ton]


def Raw_parameters(
        stw,
        rho,
        g,
        trim,
        Bwl,
        deadrise,
        swh,
        Loa,
        displacement,
        **kwargs,
    ):
    r"""Non dimensional parameters of the added resistance regression (imperial units).

    Applicability range of the regression (Savitsky & Brown 1976, p. 395) :
    100 <= P1 = Delta_LT / (0.01 L)^3 <= 250 ; 3 <= P2 = L/b <= 5 ; 3 <= trim <= 7 deg ;
    10 <= beta <= 30 deg ; 0.2 <= P5 = swh/b <= 0.7 ; 2 <= P6 = Vk/sqrt(L) <= 6.

    Returns
    -------
    dict, C_Delta, P1, P2, P5, P6, and the imperial quantities Delta, L, b, H, w.
    """
    Delta_LT = displacement / _LT
    Delta = Delta_LT * 2240.
    L = Loa * _FT
    b = Bwl * _FT
    H = swh * _FT
    w = rho * g / (_LBF * _FT ** 3)
    return dict(
        C_Delta=Delta / (w * b ** 3), P1=Delta_LT / (0.01 * L) ** 3, P2=L / b, P5=H / b,
        P6=stw / np.sqrt(L), Delta=Delta, L=L, b=b, H=H, w=w,
    )


def R_AW_interp(
        stw,
        rho,
        g,
        trim,
        Bwl,
        deadrise,
        swh,
        Loa,
        displacement,
        **kwargs,
    ):
    r"""Added resistance in waves : regression at Vk/sqrt(L) = 2, 4, 6 and parabolic interpolation.

    .. math::
        R_{AW,2} = w b^3\left[66\cdot10^{-6}\,(H/b + 0.5)\frac{(L/b)^3}{C_\Delta} + 0.0043(\tau - 4)\right]

        R_{AW,4} = \Delta\,\frac{0.3\,H/b}{1 + 2H/b}\left(1.76 - \frac{\tau}{6} - 2\tan^3\beta\right)

        R_{AW,6} = w b^3\,\frac{0.158\,H/b}{1 + (H/b)\left(0.12\beta - 21 C_\Delta (5.6 - L/b) + 7.5(6 - L/b)\right)}

    Returns
    -------
    float, Added resistance [N].
    """
    P = Raw_parameters(stw=stw, rho=rho, g=g, trim=trim, Bwl=Bwl, deadrise=deadrise,
                       swh=swh, Loa=Loa, displacement=displacement)
    wb3 = P["w"] * P["b"] ** 3
    Hb, Lb, Cd = P["H"] / P["b"], P["L"] / P["b"], P["C_Delta"]

    R2 = wb3 * (66e-6 * (Hb + 0.5) * Lb ** 3 / Cd + 0.0043 * (trim - 4.))
    R4 = P["Delta"] * (0.3 * Hb) / (1. + 2. * Hb) * (1.76 - trim / 6. - 2. * np.tan(deadrise * _DEG) ** 3)
    R6 = wb3 * (0.158 * Hb) / (1. + Hb * (0.12 * deadrise - 21. * Cd * (5.6 - Lb) + 7.5 * (6. - Lb)))

    x = P["P6"]
    R = R2 * (x - 4.) * (x - 6.) / 8. - R4 * (x - 2.) * (x - 6.) / 4. + R6 * (x - 2.) * (x - 4.) / 8.

    return R * _LBF

#%% RESISTANCE COMPONENTS

def compute_Rf_savitsky(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""
    Calculate the frictional resistance using the Savitsky method.

    Independent of the lift : it only depends on the wetted geometry and on the mean
    bottom velocity.

    .. math::
        R_f = \frac{D_f}{\cos\tau}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    nu : float, Kinematic viscosity of water [m²/s].
    trim : float, Trim angle, bow up positive [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    Optional (kwargs) : wetted_lengths ("76" or "64"), rise_factor, lam, dCf, Aws, use_Aws.

    Returns
    -------
    float, Frictional resistance [N].
    """
    _tau = tau(trim)
    _Df = D_f(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    Rf = _Df / np.cos(_tau * _DEG)

    if not np.is_casadi_type(Rf):
        Rf = np.squeeze(Rf)

    return Rf


def compute_Rp_savitsky(
        stw,
        rho,
        trim,
        Lwl,
        Bwl,
        deadrise,
        bf_M=None,
        bf_K=0.5,
        **kwargs,
    ):
    r"""
    Calculate the pressure (induced) resistance of the DYNAMIC lift using the Savitsky method.

    Independent of friction : it only depends on the dynamic lift and on the trim.

    .. math::
        R_p = L_{dyn} \tan\tau

    The trim drag of the hydrostatic (buoyant) part is NOT included : it belongs to the
    hydrostatics module.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    trim : float, Trim angle, bow up positive [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    bf_M : float, Optional Blount & Fox factor M (user supplied) [-].
    bf_K : float, Blount & Fox correlation factor K [-].

    Returns
    -------
    float, Pressure resistance [N].
    """
    _tau = tau(trim)
    _L = L_dynamic(stw=stw, rho=rho, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    Rp = _L * np.tan(_tau * _DEG)

    if bf_M is not None:
        Rp = Rp * blount_fox_M_prime(bf_M, bf_K)

    if not np.is_casadi_type(Rp):
        Rp = np.squeeze(Rp)

    return Rp


# Registry name (same slot as the residuary / wave component of the other methods)
compute_Rw_savitsky = compute_Rp_savitsky


def compute_Rsp_savitsky(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        dCf=0.,
        **kwargs,
    ):
    r"""
    Calculate the whisker spray (viscous) resistance, Savitsky, DeLorme & Datla (2007).

    The 1964 method only contains the friction and pressure drag aft of the stagnation
    line ; the thin spray sheet forward of it adds viscous drag (10-15 % of the total at
    planing speeds):

    .. math::
        F_s = \tfrac{1}{2}\rho V^2 C_f A_{as}, \qquad R_{sp} = F_s \cos\varphi \cos\tau

    Parameters
    ----------
    (same as compute_Rf_savitsky)

    Returns
    -------
    float, Whisker spray resistance [N].

    Notes
    -----
    !! The original paper could not be consulted. The root-line angle alpha, F_s, phi and
       R_sp = F_s cos(phi) cos(tau) come from secondary sources ; the area
       A_as = b^2 / (4 sin(2 alpha) cos(beta)) and theta = 2 alpha are the readings
       consistent with the published magnitude (spray = 10-15 % of the total). Check them
       against the original (Marine Technology 44(1), 2007) before normative use.
       Cf is evaluated at the free stream speed on the mean wetted length (+ dCf).
       The spray force moment / lift are neglected.
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _tau = tau(trim)
    Cf = Cf_hull(Vms * _lam * Bwl / nu + 10.) + dCf

    Fs = 0.5 * rho * Vms ** 2 * Cf * A_spray(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    Rsp = Fs * np.cos(phi_spray(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs) * _DEG) * np.cos(_tau * _DEG)

    if not np.is_casadi_type(Rsp):
        Rsp = np.squeeze(Rsp)

    return Rsp


def compute_Raw_savitsky(
        stw,
        rho,
        g,
        trim,
        Bwl,
        deadrise,
        swh=None,
        Loa=None,
        Lbp=None,
        displacement=None,
        mass=None,
        volume=None,
        **kwargs,
    ):
    r"""
    Calculate the added resistance in waves using Savitsky & Brown (1976).

    Zero in calm water (swh is None or 0). The regression is expressed in terms of the vessel
    displacement (use the real displacement, constant in the NLP : `displacement` or `mass`) ; if neither is given
    rho g volume is used, which then follows the immersed volume (not recommended).

    !! The regression was fitted on heavy, large planing craft : 100 <= Delta_LT/(0.01 L)^3 <= 250,
       3 <= L/b <= 5, 3 <= trim <= 7 deg, 10 <= beta <= 30 deg, 0.2 <= swh/b <= 0.7,
       2 <= Vk/sqrt(L[ft]) <= 6. A dinghy is usually OUTSIDE this range (extrapolation) :
       see Raw_parameters() to check the margins.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].
    trim : float, Trim angle [deg].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    swh : float, Significant wave height [m].
    Loa : float, Overall length [m] (default Lbp, else 3 b).
    displacement : float, Vessel displacement [N] (or mass [kg]).

    Returns
    -------
    float, Added resistance in waves [N].
    """
    if swh is None or (isinstance(swh, (int, float)) and swh == 0):
        return 0. * stw

    if displacement is None:
        displacement = mass * g if mass is not None else rho * g * volume
    if Loa is None:
        Loa = Lbp if Lbp is not None else 3. * Bwl

    Raw = R_AW_interp(stw=stw, rho=rho, g=g, trim=trim, Bwl=Bwl, deadrise=deadrise,
                      swh=swh, Loa=Loa, displacement=displacement)

    if not np.is_casadi_type(Raw):
        Raw = np.squeeze(Raw)

    return Raw


def compute_Rtr_savitsky(
        stw,
        **kwargs,
    ):
    """
    Transom resistance : the transom is dry in the planing regime.
    """
    return 0. * stw

#%% HYDRODYNAMIC WRENCH (not resistances)

def compute_Fz_savitsky(
        stw,
        rho,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    """
    Dynamic vertical force, upward positive [N] (no hydrostatic part : add it to the
    buoyancy supplied by the hydrostatics module).
    """
    Fz = L_dynamic(stw=stw, rho=rho, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)

    if not np.is_casadi_type(Fz):
        Fz = np.squeeze(Fz)

    return Fz


def compute_My_savitsky(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        lcg_tr,
        vcg_keel=None,
        **kwargs,
    ):
    r"""
    Dynamic pitching moment about the reduction point, bow up positive [N.m].

    .. math::
        M_y = N\,(l_{cp} - l_{cg}) - D_f\left(v_{cg} - \frac{b}{4}\tan\beta\right)

    Parameters
    ----------
    lcg_tr : float, Reduction point abscissa from the transom, along the keel [m].
    vcg_keel : float, Reduction point height above the keel, normal to it [m]
        (default : friction lever arm set to zero).
    """
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _N = N_normal(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _Df = D_f(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)

    f_arm = 0. if vcg_keel is None else vcg_keel - Bwl / 4. * np.tan(deadrise * _DEG)
    My = _N * (lcp(lam=_lam, Bwl=Bwl) - lcg_tr) - _Df * f_arm

    if not np.is_casadi_type(My):
        My = np.squeeze(My)

    return My
