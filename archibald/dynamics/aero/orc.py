# -*- coding: utf-8 -*-
"""
Windage of the above-water part of a boat, from the ORC VPP documentation (2023), section 5.3.

Each windage element is given its own dynamic pressure, based on the apparent
wind at the height of its centre of effort, and a drag area interpolated
between a frontal and a side value.

Unlike the original ORC formulation, where areas are parametric (average
freeboard, beam, length, heel correction), the hull areas and centres of effort
are taken here from the mesh aerostatics (``ArchibaldMesh.aerostatics``). They
are measured in the waterplane axes, hence the effect of heel and trim is
already included.
"""

from archibald.dynamics.aero.common import windage_element, crew_zce


def compute_Fhull_orc(
        stw,
        tws,
        twa,
        rho_air,
        Ax,
        Ay,
        cex,
        cey,
        Cd_hull=0.816,
        z_ref=10.,
        wind_alpha=1./7.,
        **kwargs,
    ):
    """
    Windage of the hull (topsides and superstructures), ORC VPP model.
    
    Parameters
    ----------
    stw : float, Speed through water [kt].
    tws : float, True wind speed at z_ref [kt].
    twa : float, True wind angle [deg].
    rho_air : float, Air density [kg/m^3].
    Ax : float, Frontal area above the waterplane [m^2].
    Ay : float, Lateral area above the waterplane [m^2].
    cex : (1, 3) array, Centroid of Ax, in the water axes [m].
    cey : (1, 3) array, Centroid of Ay, in the water axes [m].
    Cd_hull : float, Drag coefficient of the hull, 0.816 in the ORC VPP.
    z_ref, wind_alpha : Wind profile, see ``apparent_wind``.

    Returns
    -------
    Fx, Fy, zce : Force components [N] and height of the centre of effort [m].
    """
    
    # single centre of effort height for the hull, as in the ORC model
    zce = (cex[0, 2] * Ax + cey[0, 2] * Ay) / (Ax + Ay + 1e-12)
    
    return windage_element(
        CdA_front=Cd_hull * Ax,
        CdA_side=Cd_hull * Ay,
        zce=zce,
        tws=tws,
        twa=twa,
        stw=stw,
        rho_air=rho_air,
        z_ref=z_ref,
        wind_alpha=wind_alpha,
    )


def compute_Fcrew_orc(
        stw,
        tws,
        twa,
        rho_air,
        Fb,
        Boa,
        heel,
        n_crew=1.,
        Cd_crew=1.08,
        A_front=0.25,
        A_side_per_crew=0.5,
        zce_crew=None,
        z_ref=10.,
        wind_alpha=1./7.,
        **kwargs,
    ):
    """
    Windage of the movable crew, ORC VPP model.
    
    The reference areas are those of the ORC table (to be checked against the
    original document): a constant frontal area for the whole crew, and a side
    area proportional to the number of movable crew.
    
    Parameters
    ----------
    stw, tws, twa, rho_air : see ``compute_Fhull_orc``.
    Fb : float, Mean freeboard [m], from aerostatics.
    Boa : float, Overall beam [m], from aerostatics.
    heel : float, Heel angle [deg].
    n_crew : float, Number of movable crew.
    Cd_crew : float, Drag coefficient of the crew, 1.08 in the ORC VPP.
    A_front : float, Frontal area of the crew [m^2].
    A_side_per_crew : float, Side area per crew member [m^2].
    zce_crew : float or None, Height of the centre of effort above the waterplane [m].
        Default is the ORC value, see ``crew_zce``.
    z_ref, wind_alpha : Wind profile, see ``apparent_wind``.

    Returns
    -------
    Fx, Fy, zce : Force components [N] and height of the centre of effort [m].
    """
    
    if zce_crew is None:
        zce_crew = crew_zce(Fb=Fb, Boa=Boa, heel=heel)
    
    return windage_element(
        CdA_front=Cd_crew * A_front,
        CdA_side=Cd_crew * A_side_per_crew * n_crew,
        zce=zce_crew,
        tws=tws,
        twa=twa,
        stw=stw,
        rho_air=rho_air,
        z_ref=z_ref,
        wind_alpha=wind_alpha,
    )
