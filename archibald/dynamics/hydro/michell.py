# -*- coding: utf-8 -*-
"""
MICHELL THIN-SHIP THEORY FOR SLENDER HULLS AND MULTIHULLS

Wave resistance of one or n slender hulls, with the interference between
hulls, from the Michell integral applied to an equivalent parametric hull
built from the hydrostatics (volume, Cp, Cx, Lwl, Bwl, T).
"""

#%% DEPENDENCIES

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.dynamics.hydro.multihull_common import (
    EPS, nodes_unit, simpson_weights, k_0, smooth_clip,
)

#%% VALIDITY RANGE OF THE EQUIVALENT HULL

# Bounds of the shape coefficients. Hydrostatics of a hull that is about to leave
# the water (or just left it) give degenerate values (Cp < 0, Cp = 0, Cx > 1...),
# for which the shape exponents p, q and r are not defined or diverge.
CP_MIN, CP_MAX = 0.001, 0.999
CX_MIN, CX_MAX = 0.001, 0.999
T_FLOOR = 0.001      # [m] lowest draught used for the truncation of the integral
S_MAX = 1.          # largest rescaling of the Kochin amplitude by the calibration
# CP_MIN, CP_MAX = 0.15, 0.92
# CX_MIN, CX_MAX = 0.40, 0.95
# T_FLOOR = 0.01      # [m] lowest draught used for the truncation of the integral
# S_MAX = 3.          # largest rescaling of the Kochin amplitude by the calibration

#%% INDIVIDUAL COEFFICIENTS

def p_exp(
        Cp,
        **kwargs,
    ):
    r"""Longitudinal exponent of the equivalent Michell hull.

    The half-beam of the equivalent hull is separable:

    .. math::
        y(\xi, \zeta) = \frac{B}{2}\,(1 - |\xi|^p)\,(1 - \zeta^q),
        \qquad \xi = \frac{2x}{L_{WL}},\quad \zeta = \frac{|z|}{T}

    Its prismatic coefficient is :math:`C_P = p/(p+1)`, hence

    .. math::
        p = \frac{C_P}{1 - C_P}

    (Wigley hull: :math:`p = 2`).

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].

    Returns
    -------
    float, Longitudinal exponent p [-].
    """
    return Cp / (1. - Cp + EPS)


def q_exp(
        Cx,
        **kwargs,
    ):
    r"""Vertical exponent of the equivalent Michell hull.

    The midship section coefficient of the equivalent hull is
    :math:`C_X = q/(q+1)`, hence

    .. math::
        q = \frac{C_X}{1 - C_X}

    Together with :func:`p_exp` this reproduces the displaced volume
    :math:`\nabla = C_P\,C_X\,L_{WL}\,B_{WL}\,T` of the real hull.

    Parameters
    ----------
    Cx : float, Midship section coefficient [-].

    Returns
    -------
    float, Vertical exponent q [-].
    """
    return Cx / (1. - Cx + EPS)


def r_exp(
        Cp,
        **kwargs,
    ):
    r"""Longitudinal exponent of the equivalent hull for fine hulls (Cp < 0.5).

    The family :math:`1 - |\xi|^p` of :func:`p_exp` has a cusp at midship when
    :math:`p < 1`, i.e. :math:`C_P < 0.5` (typically a hull partly lifted out
    of the water by heel). For these hulls the half-beam is

    .. math::
        f(\xi) = (1 - |\xi|)^r, \qquad C_P = \frac{1}{r + 1}
        \quad\Rightarrow\quad r = \frac{1}{C_P} - 1

    which is continuous with the first family at :math:`C_P = 0.5`
    (:math:`p = r = 1`, a diamond waterline).

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].

    Returns
    -------
    float, Exponent r [-].
    """
    return 1. / (Cp + EPS) - 1.


def u_nodes_michell(
        stw,
        T,
        g,
        n_u=401,
        depth_decay=16.,
        **kwargs,
    ):
    r"""Quadrature nodes and weights of the Michell integral in :math:`u=\tan\theta`.

    Using :math:`u = \tan\theta`, :math:`\sec^3\theta\,d\theta = \sqrt{1+u^2}\,du`.
    The integrand decays as :math:`e^{-k_0(1+u^2)T}`, so the integral is truncated at

    .. math::
        u_{max} = \sqrt{\frac{D}{k_0\,T_{max}}}

    with D = ``depth_decay``. The grid scales smoothly with speed
    (differentiable w.r.t. ``stw`` and ``T``).

    Parameters
    ----------
    stw : float, Speed through water [kt].
    T : float, Largest draught of the hulls [m].
    g : float, Gravitational acceleration [m/s²].
    n_u : int, Number of nodes in u.
    depth_decay : float, Depth decay exponent at the truncation [-].

    Returns
    -------
    tuple (u, w), Nodes [-] and Simpson weights including du [-].
    """
    _k0 = k_0(stw=stw, g=g)
    u_max = np.sqrt(depth_decay / (_k0 * T + EPS))
    return u_max * nodes_unit(n_u), u_max * simpson_weights(n_u)


def kochin_amplitude(
        Lwl,
        Bwl,
        T,
        Cp,
        Cx,
        stw,
        g,
        u_nodes,
        n_x=61,
        n_z=25,
        **kwargs,
    ):
    r"""Real Kochin amplitude :math:`A(u)` of one equivalent hull centred at the origin.

    According to :cite:`michell1898wave`, the wave resistance involves

    .. math::
        P + iQ = \iint \frac{\partial y}{\partial x}\,
            e^{k_0 \sec^2\theta\, z + i k_0 \sec\theta\, x}\,dx\,dz

    For the equivalent hull of :func:`p_exp` (:func:`r_exp` if Cp < 0.5) and
    :func:`q_exp`, integrating by
    parts in x (pointed ends) leaves a purely imaginary factor common to all
    hulls, so that only the real amplitude is kept:

    .. math::
        A = \frac{B_{WL}}{2}\,\omega\,T\,C(\omega)\,G(a)

    .. math::
        C = 2\int_0^1 f(\xi)\cos(\omega\xi)\,d\xi, \qquad
        G = \int_0^1 (1-s^q)\,e^{-a T s}\,ds

    with :math:`k = k_0\sqrt{1+u^2}`, :math:`\omega = k L_{WL}/2` and
    :math:`a = k_0(1+u^2)`.

    Parameters
    ----------
    Lwl, Bwl, T : float, Waterline length, beam and draught [m].
    Cp, Cx : float, Prismatic and midship section coefficients [-].
    stw : float, Speed through water [kt].
    g : float, Gravitational acceleration [m/s²].
    u_nodes : array, Nodes in u = tan(theta) [-].
    n_x, n_z : int, Number of nodes of the longitudinal and vertical quadratures.

    Returns
    -------
    array, Kochin amplitude A(u) [m²].
    """
    _k0 = k_0(stw=stw, g=g)
    
    Lwl = np.softplus(Lwl, beta=1e3)
    Bwl = np.softplus(Bwl, beta=1e3)
    T = np.softplus(T, beta=1e3)
    
    p, q, r = p_exp(Cp), q_exp(Cx), r_exp(Cp)
    k = _k0 * np.sqrt(1. + u_nodes ** 2)
    a = _k0 * (1. + u_nodes ** 2)
    omega = k * Lwl / 2.

    xi, wx = nodes_unit(n_x), simpson_weights(n_x)
    C = 0.
    for i in range(len(xi)):
        shape = np.where(Cp >= 0.5, 1. - (xi[i] + 1e-12) ** p, (1. - xi[i] + 1e-9) ** r)
        C = C + wx[i] * shape * np.cos(omega * xi[i])
    C = 2. * C

    s, wz = nodes_unit(n_z), simpson_weights(n_z)
    G = 0.
    for i in range(len(s)):
        G = G + wz[i] * (1. - (s[i] + 1e-12) ** q) * np.exp(-a * T * s[i])

    return (Bwl / 2.) * omega * T * C * G

#%% RESISTANCE COMPONENTS

def compute_Rw_matrix_michell(
        stw,
        Lwl,
        Bwl,
        T,
        Cp,
        Cx,
        x,
        y,
        rho,
        g,
        n_u=401,
        n_x=61,
        n_z=25,
        **kwargs,
    ):
    r"""Michell wave resistance matrix of n hulls (self and interaction terms).

    According to :cite:`michell1898wave`, a single hull has

    .. math::
        R_w = \frac{4\rho g^2}{\pi U^2}\int_0^{\pi/2} (P^2 + Q^2)\sec^3\theta\,d\theta

    In linear theory the wave spectra of n hulls located at :math:`(x_j, y_j)`
    superpose with a phase :math:`k(x_j + y_j\tan\theta)` (see
    :cite:`tuck1998optimum` and :cite:`day2001multihull`), which gives

    .. math::
        R_{ij} = \frac{2\rho g^2}{\pi U^2}\int_0^\infty A_i A_j
            \Big[\cos k(\Delta x + \Delta y\,u) + \cos k(\Delta x - \Delta y\,u)\Big]
            \sqrt{1+u^2}\,du

    with :math:`\Delta x = x_i - x_j`, :math:`\Delta y = y_i - y_j` and :math:`A`
    from :func:`kochin_amplitude`. The diagonal is the wave resistance of each
    hull alone, and :math:`R_w = \sum_{ij} R_{ij}`.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl, Bwl, T, Cp, Cx : list of float, Hydrostatic parameters of each hull.
    x, y : list of float, Longitudinal and transverse position of each hull's
        centre of buoyancy in the boat frame [m].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].
    n_u, n_x, n_z : int, Number of quadrature nodes (u, x, z).

    Returns
    -------
    list of list, Matrix R[i][j] [N].
    """
    n = len(Lwl)
    T_ref = T[0]
    for Ti in T[1:]:
        T_ref = np.maximum(T_ref, Ti)
    T_ref = np.maximum(T_ref, T_FLOOR)

    un, w = u_nodes_michell(stw=stw, T=T_ref, g=g, n_u=n_u)
    _k0 = k_0(stw=stw, g=g)
    k = _k0 * np.sqrt(1. + un ** 2)
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    pref = 2. * rho * g ** 2 / (np.pi * Vms ** 2)
    weight = w * np.sqrt(1. + un ** 2)

    A = [
        kochin_amplitude(
            Lwl=Lwl[i], Bwl=Bwl[i], T=T[i], Cp=Cp[i], Cx=Cx[i],
            stw=stw, g=g, u_nodes=un, n_x=n_x, n_z=n_z,
        )
        for i in range(n)
    ]

    R = [[0. for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            phase = np.cos(k * (dx + dy * un)) + np.cos(k * (dx - dy * un))
            R[i][j] = pref * np.sum(A[i] * A[j] * phase * weight)
            R[j][i] = R[i][j]
    return R


def compute_Rw_michell(
        stw,
        Lwl,
        Bwl,
        T,
        Cp,
        Cx,
        rho,
        g,
        **kwargs,
    ):
    r"""Wave resistance of a single slender hull by Michell thin-ship theory.

    Equivalent-hull formulation of :func:`compute_Rw_matrix_michell` with one
    hull. It can be used as the ``"Rw"`` entry of a custom resistance process
    for slender hulls, outside the validity range of the Holtrop-Mennen regression.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl : float, Waterline length [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    Cp : float, Prismatic coefficient [-].
    Cx : float, Midship section coefficient [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wave resistance [N].
    """
    return compute_Rw_matrix_michell(
        stw=stw, Lwl=[Lwl], Bwl=[Bwl], T=[T], Cp=[Cp], Cx=[Cx],
        x=[0.], y=[0.], rho=rho, g=g, **kwargs,
    )[0][0]


def compute_dRw_michell(
        stw,
        Lwl,
        Bwl,
        T,
        Cp,
        Cx,
        x,
        y,
        rho,
        g,
        Rw=None,
        calibrate=True,
        s_max=S_MAX,
        **kwargs,
    ):
    r"""Wave interference matrix between n hulls from Michell theory.

    Thin-ship theory is better at capturing the *relative* effect of spacing
    and stagger (phase) than the absolute wave resistance. With
    ``calibrate=True`` the Kochin amplitude of each hull is rescaled so that
    its self wave resistance matches the individual (empirical) one:

    .. math::
        s_i = \sqrt{R_{w,i} / R^{M}_{ii}}, \qquad
        \Delta R_{w,ij} = s_i\,s_j\,R^{M}_{ij}, \quad i \neq j

    For identical hulls this is exactly :math:`R_w\,(\tau - 1)`, with the
    interference factor :math:`\tau` of :cite:`insel1992investigation`
    evaluated by Michell theory. With ``calibrate=False``, the raw Michell
    cross terms are returned. ``dRw`` is symmetric and its row i is the share
    of the interference attributed to hull i (half of each pair).

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl, Bwl, T, Cp, Cx : list of float, Hydrostatic parameters of each hull.
    x, y : list of float, Positions of each hull's centre of buoyancy [m].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].
    Rw : list of float or None, Individual wave resistance of each hull [N].
    calibrate : bool, Whether to rescale on the individual Rw.
    s_max : float, The rescaling factor s_i is kept in [1/s_max, s_max]. It is
        needed when a hull is hardly wet: its Michell self resistance vanishes
        while the empirical one does not, and the ratio would diverge.

    Returns
    -------
    list of list, Interference matrix dRw[i][j] [N], zero diagonal.
    """
    n = len(Lwl)
    R = compute_Rw_matrix_michell(
        stw=stw, Lwl=Lwl, Bwl=Bwl, T=T, Cp=Cp, Cx=Cx, x=x, y=y,
        rho=rho, g=g, **kwargs,
    )
    s = [1. for _ in range(n)]
    if calibrate and Rw is not None:
        s = [
            smooth_clip(np.sqrt(Rw[i] / (R[i][i] + EPS)), 1. / s_max, s_max)
            for i in range(n)
        ]
    return [
        [0. if i == j else s[i] * s[j] * R[i][j] for j in range(n)]
        for i in range(n)
    ]

#%% EXAMPLE
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------
    # WAVE RESISTANCE AND INTERFERENCE FACTOR OF A SLENDER CATAMARAN
    # ---------------------------------------------------------

    env_params = {'g': 9.81, 'rho': 1025.0}
    demihull = {'Lwl': 12.0, 'Bwl': 0.6, 'T': 0.5, 'Cp': 0.62, 'Cx': 0.70}

    fr_array = np.linspace(0.2, 1.0, 120)
    sl_list = [0.2, 0.3, 0.4, 0.6, 1.0]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for SL in sl_list:
        S = SL * demihull['Lwl']
        tau = []
        for fr in fr_array:
            stw = fr * np.sqrt(env_params['g'] * demihull['Lwl']) / u.kt
            R = compute_Rw_matrix_michell(
                stw=stw,
                Lwl=[demihull['Lwl']] * 2, Bwl=[demihull['Bwl']] * 2,
                T=[demihull['T']] * 2, Cp=[demihull['Cp']] * 2,
                Cx=[demihull['Cx']] * 2,
                x=[0., 0.], y=[-S / 2, S / 2], **env_params,
            )
            tau.append((R[0][0] + R[0][1] + R[1][0] + R[1][1])
                       / (R[0][0] + R[1][1]))
        ax.plot(fr_array, tau, label=f'S/L = {SL}')

    ax.axhline(1., color='k', linewidth=0.8)
    ax.set_title("Michell wave interference factor of a catamaran\n"
                 "$\\tau = R_w / (2 R_{w,demihull})$ vs. Froude Number",
                 fontsize=14, pad=15)
    ax.set_xlabel('Froude Number ($F_n$)', fontsize=12)
    ax.set_ylabel('$\\tau$', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()
