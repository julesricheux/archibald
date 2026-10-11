# -*- coding: utf-8 -*-
"""
Windage of the hull and crew of a sailing dinghy, after Day (2017),
"Performance Prediction for Sailing Dinghies".

The hull follows the ORC VPP, with a projected area varying sinusoidally
between its frontal and side values. The crew drag area is derived from
anthropometric data: Du Bois surface area, and the projected areas and drag
coefficients measured by Penwarden et al. (1978).
"""

from archibald.dynamics.aero.common import (
    apparent_wind,
    resolve_wind_drag,
    windage_element,
    smooth_abs_sind,
    crew_zce,
)


def hull_areas_day(
        Fb,
        Boa,
        Loa,
        Cwp,
        heel,
    ):
    """
    Parametric areas of the topsides, with the heel correction of Day (2017), eq. 7.
    
    Only meant for an UPRIGHT freeboard ``Fb``: with an aerostatics computed
    on a heeled mesh, the heel is already in the areas.
    
    Parameters
    ----------
    Fb : float, Mean freeboard [m].
    Boa : float, Overall beam [m].
    Loa : float, Overall length [m].
    Cwp : float, Waterplane area coefficient (mean beam = Cwp * Boa).
    heel : float, Heel angle [deg].

    Returns
    -------
    Af, As, zce : Frontal area [m^2], side area [m^2], centre of effort height [m].
    """
    heeled_freeboard = Fb + 0.5 * Boa * Cwp * smooth_abs_sind(heel)
    
    Af = Boa * Fb
    As = Loa * heeled_freeboard
    zce = 0.66 * heeled_freeboard
    
    return Af, As, zce


def crew_areas_day(
        crew_mass,
        crew_height,
        g,
    ):
    """
    Frontal and side areas of a person, from Du Bois (1916) and Penwarden et al. (1978).

    Parameters
    ----------
    crew_mass : float, Mass of the person [kg].
    crew_height : float, Height of the person [m].
    g : float, Gravity [m/s^2].

    Returns
    -------
    A_front, A_side : Frontal and side projected areas [m^2].
    """
    weight = crew_mass * g  # Du Bois formula of Day (2017) takes the weight in N
    A_Du = 0.0769 * weight ** 0.425 * crew_height ** 0.725
    
    return 0.326 * A_Du, 0.219 * A_Du


def compute_Fhull_day(
        stw,
        tws,
        twa,
        rho_air,
        Ax,
        Ay,
        cex,
        cey,
        Fb,
        Boa,
        Loa,
        Cd_hull=0.816,
        parametric=False,
        Cwp=None,
        heel=0.,
        z_ref=10.,
        wind_alpha=1./7.,
        **kwargs,
    ):
    """
    Windage of the topsides, Day (2017) model.
    
    The drag is evaluated with a projected area varying sinusoidally with the
    apparent wind angle between the frontal and the side areas:

        A(awa) = Af + (As - Af) * |sin(awa)|
    
    The drag coefficient is not given by Day, who follows the ORC VPP (0.816).

    Parameters
    ----------
    stw, tws, twa, rho_air : see ``compute_Fhull_orc``.
    Ax, Ay, cex, cey : Frontal and lateral areas and their centroids from aerostatics.
    Fb, Boa, Loa : Mean freeboard, overall beam and length from aerostatics [m].
    Cd_hull : float, Drag coefficient of the hull.
    parametric : bool, If True, areas come from ``hull_areas_day`` (upright
        freeboard and heel correction) instead of the mesh areas. Requires Cwp.
    Cwp : float, Waterplane area coefficient, from hydrostatics. Used if parametric.
    heel : float, Heel angle [deg]. Used if parametric.
    z_ref, wind_alpha : Wind profile, see ``apparent_wind``.

    Returns
    -------
    Fx, Fy, zce : Force components [N] and height of the centre of effort [m].
    """
    
    if parametric:
        if Cwp is None:
            raise ValueError("Cwp is required for the parametric Day hull model.")
        Af, As, zce = hull_areas_day(Fb=Fb, Boa=Boa, Loa=Loa, Cwp=Cwp, heel=heel)
    else:
        Af, As = Ax, Ay
        zce = (cex[0, 2] * Ax + cey[0, 2] * Ay) / (Ax + Ay + 1e-12)
    
    aws, awa = apparent_wind(
        tws=tws,
        twa=twa,
        stw=stw,
        z=zce,
        z_ref=z_ref,
        wind_alpha=wind_alpha,
    )
    
    A = Af + (As - Af) * smooth_abs_sind(awa)
    D = 0.5 * rho_air * aws ** 2 * Cd_hull * A
    
    Fx, Fy = resolve_wind_drag(D, awa)
    
    return Fx, Fy, zce


def compute_Fcrew_day(
        stw,
        tws,
        twa,
        rho_air,
        g,
        Fb,
        Boa,
        heel,
        crew_mass=80.,
        crew_height=1.83,
        n_crew=1.,
        Cd_front=1.075,
        Cd_side=0.954,
        clothing_factor=0.9,
        shielding_factor=0.8,
        zce_crew=None,
        z_ref=10.,
        wind_alpha=1./7.,
        **kwargs,
    ):
    """
    Windage of the crew, Day (2017) model.
    
    Areas from Du Bois and Penwarden, drag coefficients from the lowest values
    measured by Penwarden et al., reduced by 10% for the clothing and the drag
    area by 20% for the shielding by the cockpit (or the other crew member).
    
    Day does not specify how the drag area varies with the wind angle, nor the
    height of the centre of effort: the ORC interpolation and height are used.

    Parameters
    ----------
    stw, tws, twa, rho_air : see ``compute_Fhull_orc``.
    g : float, Gravity [m/s^2].
    Fb, Boa : Mean freeboard and overall beam from aerostatics [m].
    heel : float, Heel angle [deg].
    crew_mass : float, Mass of one crew member [kg].
    crew_height : float, Height of one crew member [m].
    n_crew : float, Number of crew members.
    Cd_front, Cd_side : float, Drag coefficients on frontal and side areas.
    clothing_factor : float, Multiplier on the drag coefficients.
    shielding_factor : float, Multiplier on the drag areas.
    zce_crew : float or None, Height of the centre of effort above the waterplane [m].
    z_ref, wind_alpha : Wind profile, see ``apparent_wind``.

    Returns
    -------
    Fx, Fy, zce : Force components [N] and height of the centre of effort [m].
    """
    
    A_front, A_side = crew_areas_day(
        crew_mass=crew_mass,
        crew_height=crew_height,
        g=g,
    )
    
    k = n_crew * clothing_factor * shielding_factor
    
    if zce_crew is None:
        zce_crew = crew_zce(Fb=Fb, Boa=Boa, heel=heel)
    
    return windage_element(
        CdA_front=k * Cd_front * A_front,
        CdA_side=k * Cd_side * A_side,
        zce=zce_crew,
        tws=tws,
        twa=twa,
        stw=stw,
        rho_air=rho_air,
        z_ref=z_ref,
        wind_alpha=wind_alpha,
    )
