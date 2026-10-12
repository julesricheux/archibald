# -*- coding: utf-8 -*-
"""
MULTIHULL INTERACTION: COORDINATION OF THE METHODS

The resistance of each hull is computed alone first (``hull.compute_resistance``).
The interaction between the n hulls is then evaluated at the Sailboat level
by :func:`compute_hull_interaction`, which returns an n x n matrix of wave
interference ``dRw`` and a vector of viscous interference ``dRf``:

    R_total = sum_i R_i + sum_ij dRw[i][j] + sum_i dRf[i]

``dRw`` is symmetric for Michell methods; its row i is the share of the
interference attributed to hull i (half of each pair).
"""

#%% DEPENDENCIES

from archibald.dynamics.hydro import michell, insel_molland, miyazawa, jamaluddin
from archibald.dynamics.hydro.multihull_common import immersion_weight

#%% METHODS

# method name -> {component: function}. As for ``hull.compute_resistance``, a
# custom dict {"Rw": callable, "Rf": callable} can be given instead of a name.
# "Rw" callables return an n x n matrix, "Rf" callables a list of length n.
INTERACTION_METHODS = {
    "none": {},
    "michell": {
        "Rw": michell.compute_dRw_michell,
    },
    "michell_raw": {
        "Rw": lambda **kw: michell.compute_dRw_michell(**{**kw, "calibrate": False}),
    },
    "miyazawa": {
        "Rf": miyazawa.compute_dRf_miyazawa,
    },
    "michell_miyazawa": {
        "Rw": michell.compute_dRw_michell,
        "Rf": miyazawa.compute_dRf_miyazawa,
    },
    "jamaluddin": {
        "Rw": jamaluddin.compute_dRw_jamaluddin,
        "Rf": jamaluddin.compute_dRf_jamaluddin,
    },
    "michell_jamaluddin": {
        "Rw": michell.compute_dRw_michell,
        "Rf": jamaluddin.compute_dRf_jamaluddin,
    },
    "insel_molland": {
        "Rw": insel_molland.compute_dRw_insel_molland,
        "Rf": insel_molland.compute_dRf_insel_molland,
    },
}

#%% SAILBOAT-LEVEL ENTRY POINT

def compute_hull_interaction(
        hulls,
        op_point,
        method="michell",
        positions=None,
        rho=None,
        g=None,
        nu=None,
        T_kiss=0.01,
        **kwargs,
    ):
    """Interaction resistance between the hulls of a multihull.

    Called by ``Sailboat.compute_hull_interaction`` with ``self.hulls``.

    Requires, for every hull, ``hull.hydrostatics_data`` (hydrostatics already
    computed) and, for the calibrated Michell and the tabulated methods,
    ``hull.resistance_components`` (individual resistance already computed).

    Parameters
    ----------
    hulls : list of Hull, Hulls with computed hydrostatics.
    op_point : OperatingPoint, Operating point (``stw`` and ``environment`` are used).
    method : str or dict, A key of ``INTERACTION_METHODS`` or a custom dict
        ``{"Rw": callable, "Rf": callable}``.
    positions : list of (x, y) or None, Hull positions [m]; if None, read with
        ``hull.get_position(op_point)``.
    rho : float or None, Water density [kg/m³]; default from ``op_point.environment``.
    g : float or None, Gravitational acceleration [m/s²]; default from ``op_point.environment``.
    nu : float or None, Kinematic viscosity [m²/s]; default from ``op_point.environment``.
    T_kiss : float, Draught [m] at which the interference of a hull is halved. The
        interference of a hull is weighted by ``immersion_weight`` so that it
        vanishes smoothly when the hull lifts out of the water (heel).
    **kwargs : Method parameters (``calibrate``, ``fn_grid``, ``sl_grid``, ``tau_table``...).

    Returns
    -------
    dict with
        ``dRw`` : n x n wave interference matrix [N],
        ``dRf`` : viscous interference per hull [N],
        ``dR_hull`` : total interference attributed to each hull [N],
        ``dR`` : total interference [N].
    """
    n = len(hulls)
    process = INTERACTION_METHODS[method] if isinstance(method, str) else method

    env = op_point.environment
    rho = env.water.density if rho is None else rho
    nu = env.water.kinematic_viscosity if nu is None else nu
    g = env.gravity if g is None else g

    # Flat per-hull lists of every hydrostatic value shared by all hulls
    common = set.intersection(*[set(h.hydrostatics_data) for h in hulls])
    args = {k: [h.hydrostatics_data[k] for h in hulls] for k in common}

    if positions is None:
        positions = [h.get_position(op_point) for h in hulls]
    args["x"] = [p[0] for p in positions]
    args["y"] = [p[1] for p in positions]

    # Individual resistances, if already computed
    comps = [getattr(h, "resistance_components", None) for h in hulls]
    for key in ("Rw", "Rf"):
        if all(c is not None and key in c for c in comps):
            args[key] = [c[key] for c in comps]

    args.update(stw=op_point.stw, rho=rho, g=g, nu=nu)
    args.update(kwargs)

    dRw = [[0. for _ in range(n)] for _ in range(n)]
    dRf = [0. for _ in range(n)]
    if "Rw" in process:
        dRw = process["Rw"](**args)
    if "Rf" in process:
        dRf = process["Rf"](**args)

    # A hull that is hardly wet (or out of the water) neither makes waves nor
    # disturbs the flow around the others: smooth weights of the pairs
    w = [immersion_weight(args["T"][i], T_kiss=T_kiss) for i in range(n)]
    dRw = [[dRw[i][j] * w[i] * w[j] for j in range(n)] for i in range(n)]
    for i in range(n):
        others = sum(w[j] for j in range(n) if j != i) / max(n - 1, 1)
        dRf[i] = dRf[i] * w[i] * others

    dR_hull = [sum(dRw[i]) + dRf[i] for i in range(n)]
    return {"dRw": dRw, "dRf": dRf, "dR_hull": dR_hull, "dR": sum(dR_hull)}
