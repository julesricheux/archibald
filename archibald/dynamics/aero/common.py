# -*- coding: utf-8 -*-
"""
Common tools for the aerodynamic drag of the above-water part of a boat
(topsides, superstructures, crew).

Conventions
-----------
- Boat axes : x forward, y to port, z up (right-handed).
- Wind angles (twa, awa) are in degrees, measured from the bow.
  They are positive when the wind comes from starboard.
- Wind speeds (tws) and boat speed (stw) are in knots, as ``stw`` elsewhere.
- Every windage method returns ``(Fx, Fy, zce)``:
    Fx  : force along x [N], positive forward (negative in a headwind)
    Fy  : force along y [N], positive to port (positive for a wind from starboard)
    zce : height of the centre of effort above the waterplane [m]
"""

import archibald.numpy as np
import archibald.toolbox.units as u


def apparent_wind(
        tws,
        twa,
        stw,
        z,
        z_ref=10.,
        wind_alpha=1./7.,
    ):
    """
    Apparent wind at a given height above the water, from the true wind
    (power-law profile) and the boat speed through the water.

    Parameters
    ----------
    tws : float, True wind speed at the reference height z_ref [kt].
    twa : float, True wind angle [deg].
    stw : float, Speed through water [kt].
    z : float, Height above the waterplane [m].
    z_ref : float, Reference height of tws [m].
    wind_alpha : float, Power-law exponent of the wind profile (0 for a uniform wind).

    Returns
    -------
    aws : Apparent wind speed at height z [m/s].
    awa : Apparent wind angle at height z [deg].
    """
    # smooth floor to keep the power law differentiable at the water surface
    z_eff = np.softplus(z, beta=1e2) + 1e-2
    
    Vt = tws * u.kt * (z_eff / z_ref) ** wind_alpha
    Vb = stw * u.kt
    
    # components of the apparent wind, pointing toward its source
    Vx = Vt * np.cosd(twa) + Vb
    Vy = Vt * np.sind(twa)
    
    aws = np.sqrt(Vx ** 2 + Vy ** 2 + 1e-12)
    awa = np.arctan2d(Vy, Vx)
    
    return aws, awa


def smooth_abs_cosd(angle):
    """|cos(angle)|, smoothed around 90 deg to remain differentiable."""
    return np.sqrt(np.cosd(angle) ** 2 + 1e-6)


def smooth_abs_sind(angle):
    """|sin(angle)|, smoothed around 0 deg to remain differentiable."""
    return np.sqrt(np.sind(angle) ** 2 + 1e-6)


def resolve_wind_drag(
        D,
        awa,
    ):
    """
    Resolve a drag force D, acting along the apparent wind direction, in the boat axes.

    Returns
    -------
    Fx, Fy : Force components [N], see module conventions.
    """
    Fx = -D * np.cosd(awa)
    Fy = D * np.sind(awa)
    
    return Fx, Fy


def windage_element(
        CdA_front,
        CdA_side,
        zce,
        tws,
        twa,
        stw,
        rho_air,
        z_ref=10.,
        wind_alpha=1./7.,
    ):
    """
    Windage force of an element, whose drag area varies with the apparent
    wind angle between a frontal and a side value (ORC VPP formulation):

        CdA(awa) = CdA_front * |cos(awa)| + CdA_side * |sin(awa)|

    The dynamic pressure is evaluated with the apparent wind at the height of
    the centre of effort of the element.

    Parameters
    ----------
    CdA_front : float, Drag area for a head wind [m^2].
    CdA_side : float, Drag area for a beam wind [m^2].
    zce : float, Height of the centre of effort above the waterplane [m].
    tws, twa, stw : True wind speed [kt], true wind angle [deg], speed through water [kt].
    rho_air : float, Air density [kg/m^3].
    z_ref, wind_alpha : Wind profile, see ``apparent_wind``.

    Returns
    -------
    Fx, Fy, zce : see module conventions.
    """
    aws, awa = apparent_wind(
        tws=tws,
        twa=twa,
        stw=stw,
        z=zce,
        z_ref=z_ref,
        wind_alpha=wind_alpha,
    )
    
    CdA = CdA_front * smooth_abs_cosd(awa) + CdA_side * smooth_abs_sind(awa)
    D = 0.5 * rho_air * aws ** 2 * CdA
    
    Fx, Fy = resolve_wind_drag(D, awa)
    
    return Fx, Fy, zce


def crew_zce(
        Fb,
        Boa,
        heel,
    ):
    """
    Default height of the crew centre of effort above the waterplane [m]
    (ORC VPP): 0.5 m above the deck edge, raised by the heel since the crew sits on the rail.
    """
    return Fb + 0.5 + 0.5 * Boa * smooth_abs_sind(heel)
