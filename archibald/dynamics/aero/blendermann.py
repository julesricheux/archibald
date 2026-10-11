# -*- coding: utf-8 -*-
"""
Wind load on the above-water part of a ship, after Blendermann (1994),
"Parameter identification of wind loads on ships", J. Wind Eng. Ind. Aerodyn. 51.

Wind tunnel based loading functions, with four parameters identified for each
type of ship: transverse resistance coefficient CDt, longitudinal resistance
coefficient CDlAF (referred to the frontal area, for a head and a following wind),
cross-force parameter delta and rolling-moment factor kappa.

With e the apparent wind angle (0 = head wind), AF the frontal area, AL the
lateral area, Loa the overall length and q = 0.5 * rho_air * U^2:

    CDl = CDlAF * AF / AL
    den = 1 - delta / 2 * (1 - CDl / CDt) * sin(2e)^2
    X   = q * AF * ( -CDlAF * cos(e) / den )
    Y   = q * AL * (  CDt   * sin(e) / den )
    N   = Y * Loa * ( sL / Loa - 0.18 * (e - pi/2) )
    K   = Y * kappa * sH

with sL, sH the position of the centroid of the lateral area from the main
section and above the waterline. The longitudinal force is that of a ship
sailing in a head wind when negative.

Unlike for Isherwood, no perimeter nor centroid of the silhouette is needed:
AF, AL, sL and sH are given by the mesh aerostatics (``ArchibaldMesh.aerostatics``).

NOTE: the formulas are those of the loading functions as reported by Turk and
Prpic-Orsic (2009) and in later papers. Check the ship type parameters against
the tables of Blendermann (1994) before using a type other than the ones listed.
"""

import archibald.numpy as np

from archibald.dynamics.aero.common import apparent_wind


# Ship type parameters. Reference wind tunnel data of Blendermann (1994).
# TODO: complete from the original tables. Parameters of other types can be
# given directly to the method.
BLENDERMANN_SHIP_TYPES = {
    # CDt and CDlAF from Blendermann (1994) as reported for a loaded container
    # ship (CDt = 0.9, CDlAF = 0.55). delta and kappa to be checked.
    "container_ship_loaded": {
        "CDt": 0.90,
        "CDlAF_head": 0.55,
        "CDlAF_stern": 0.55,
        "delta": 0.40,
        "kappa": 1.40,
    },
}


def get_ship_parameters(
        ship_type=None,
        CDt=None,
        CDlAF_head=None,
        CDlAF_stern=None,
        delta=None,
        kappa=None,
    ):
    """
    Blendermann parameters of a ship, from a type of ``BLENDERMANN_SHIP_TYPES``
    and/or explicit values, which have priority over the type.

    Parameters
    ----------
    ship_type : str or None, key of ``BLENDERMANN_SHIP_TYPES``.
    CDt : float, Transverse resistance coefficient (referred to the lateral area).
    CDlAF_head : float, Longitudinal resistance coefficient in a head wind (referred to the frontal area).
    CDlAF_stern : float, Same for a following wind. Default CDlAF_head.
    delta : float, Cross-force parameter.
    kappa : float, Rolling-moment factor.

    Returns
    -------
    dict with keys CDt, CDlAF_head, CDlAF_stern, delta, kappa.
    """
    p = {}
    if ship_type is not None:
        if ship_type not in BLENDERMANN_SHIP_TYPES:
            raise NotImplementedError(
                f"Unknown ship type '{ship_type}'. Should be in {list(BLENDERMANN_SHIP_TYPES.keys())} or given explicitly.")
        p.update(BLENDERMANN_SHIP_TYPES[ship_type])
    
    explicit = {
        "CDt": CDt,
        "CDlAF_head": CDlAF_head,
        "CDlAF_stern": CDlAF_stern,
        "delta": delta,
        "kappa": kappa,
    }
    p.update({k: v for k, v in explicit.items() if v is not None})
    
    if "CDlAF_stern" not in p and "CDlAF_head" in p:
        p["CDlAF_stern"] = p["CDlAF_head"]
    
    missing = [k for k in explicit if k not in p]
    if missing:
        raise ValueError(f"Blendermann parameters {missing} are missing: give a ship_type or the values.")
    
    return p


def compute_Fhull_blendermann(
        stw,
        tws,
        twa,
        rho_air,
        Ax,
        Ay,
        Loa,
        cey,
        ship_type=None,
        CDt=None,
        CDlAF_head=None,
        CDlAF_stern=None,
        delta=None,
        kappa=None,
        z_ref=10.,
        wind_alpha=1./7.,
        **kwargs,
    ):
    """
    Wind load on the above-water part of a ship, Blendermann (1994) model.
    
    The apparent wind is evaluated at the height of the centroid of the lateral
    area. Blendermann coefficients come from uniform flow wind tunnel tests:
    use ``wind_alpha=0`` for a consistent uniform wind.
    
    The force is applied at the centroid of the lateral area, shifted
    longitudinally by -0.18 * Loa * (e - pi/2) as in the yaw moment function
    (forward for a bow wind, aft for a stern wind), and at the height kappa * sH
    above the waterplane, which gives the rolling moment of the model.
    
    Parameters
    ----------
    stw, tws, twa, rho_air : see ``compute_Fhull_orc``.
    Ax : float, Frontal area above the waterplane AF [m^2].
    Ay : float, Lateral area above the waterplane AL [m^2].
    Loa : float, Overall length [m].
    cey : (1, 3) array, Centroid of the lateral area, in the water axes [m].
    ship_type, CDt, CDlAF_head, CDlAF_stern, delta, kappa : see ``get_ship_parameters``.
    z_ref, wind_alpha : Wind profile, see ``apparent_wind``.

    Returns
    -------
    Fx, Fy, zce, dxce : Force components [N], height of the application point above
        the waterplane [m], and its longitudinal offset from the lateral centroid [m].
    """
    
    p = get_ship_parameters(
        ship_type=ship_type,
        CDt=CDt,
        CDlAF_head=CDlAF_head,
        CDlAF_stern=CDlAF_stern,
        delta=delta,
        kappa=kappa,
    )
    
    sH = cey[0, 2]
    
    aws, awa = apparent_wind(
        tws=tws,
        twa=twa,
        stw=stw,
        z=sH,
        z_ref=z_ref,
        wind_alpha=wind_alpha,
    )
    
    # smooth switch between the head wind and following wind longitudinal coefficients
    w_head = np.sigmoid(20. * np.cosd(awa))
    CDlAF = w_head * p["CDlAF_head"] + (1. - w_head) * p["CDlAF_stern"]
    CDl = CDlAF * Ax / (Ay + 1e-12)
    
    den = 1. - p["delta"] / 2. * (1. - CDl / p["CDt"]) * np.sind(2. * awa) ** 2
    
    CX = -CDlAF * np.cosd(awa) / den
    CY = p["CDt"] * np.sind(awa) / den
    
    q = 0.5 * rho_air * aws ** 2
    
    Fx = q * Ax * CX
    Fy = q * Ay * CY
    
    # yaw moment arm from the lateral centroid, awa in rad
    eps = np.sqrt((awa * np.pi / 180.) ** 2 + 1e-6)
    dxce = -0.18 * Loa * (eps - np.pi / 2.)
    
    zce = p["kappa"] * sH
    
    return Fx, Fy, zce, dxce
