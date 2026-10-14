# -*- coding: utf-8 -*-
import io
import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.modeling import InterpolatedModel
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

#%%

# DSYHS residuary resistance coefs
# Coefficients a0 to a7
_keunig_coefs = {
    'Fr': np.array([0.,   0.15,    0.20,    0.25,    0.30,    0.35,    0.40,    0.45,    0.50,    0.55,    0.60,    0.65,    0.70,    0.75]),
    'a0': np.array([0.,-0.0005, -0.0003, -0.0002, -0.0009, -0.0026, -0.0064, -0.0218, -0.0388, -0.0347, -0.0361, +0.0008, +0.0108, +0.1023]),
    'a1': np.array([0.,+0.0023, +0.0059, -0.0156, +0.0016, -0.0567, -0.4034, -0.5261, -0.5986, -0.4764, +0.0037, +0.3728, -0.1238, +0.7726]),
    'a2': np.array([0.,-0.0086, -0.0064, +0.0031, +0.0337, +0.0446, -0.1250, -0.2945, -0.3038, -0.2361, -0.2960, -0.3667, -0.2026, +0.5040]),
    'a3': np.array([0.,-0.0015, +0.0070, -0.0021, -0.0285, -0.1091, +0.0273, +0.2485, +0.6033, +0.8726, +0.9661, +1.3957, +1.1282, +1.7867]),
    'a4': np.array([0.,+0.0061, +0.0014, -0.0070, -0.0367, -0.0707, -0.1341, -0.2428, -0.0430, +0.4219, +0.6123, +1.0343, +1.1836, +2.1934]),
    'a5': np.array([0.,+0.0010, +0.0013, +0.0148, +0.0218, +0.0914, +0.3578, +0.6293, +0.8332, +0.8990, +0.7534, +0.3230, +0.4973, -1.5479]),
    'a6': np.array([0.,+0.0001, +0.0005, +0.0010, +0.0015, +0.0021, +0.0045, +0.0081, +0.0106, +0.0096, +0.0100, +0.0072, +0.0038, -0.0115]),
    'a7': np.array([0.,+0.0052, -0.0020, -0.0043, -0.0172, -0.0078, +0.1115, +0.2086, +0.1336, -0.2272, -0.3352, -0.4632, -0.4477, -0.0977])
}

# _keunig_coefs = {
#     'Fr': np.array([0.,   0.15,    0.20,    0.25,    0.30,    0.35,    0.40,    0.45,    0.50,    0.55,    0.60,    0.65,    0.70,]),
#     'a0': np.array([0.,-0.0005, -0.0003, -0.0002, -0.0009, -0.0026, -0.0064, -0.0218, -0.0388, -0.0347, -0.0361, +0.0008, +0.0108,]),
#     'a1': np.array([0.,+0.0023, +0.0059, -0.0156, +0.0016, -0.0567, -0.4034, -0.5261, -0.5986, -0.4764, +0.0037, +0.3728, -0.1238,]),
#     'a2': np.array([0.,-0.0086, -0.0064, +0.0031, +0.0337, +0.0446, -0.1250, -0.2945, -0.3038, -0.2361, -0.2960, -0.3667, -0.2026,]),
#     'a3': np.array([0.,-0.0015, +0.0070, -0.0021, -0.0285, -0.1091, +0.0273, +0.2485, +0.6033, +0.8726, +0.9661, +1.3957, +1.1282,]),
#     'a4': np.array([0.,+0.0061, +0.0014, -0.0070, -0.0367, -0.0707, -0.1341, -0.2428, -0.0430, +0.4219, +0.6123, +1.0343, +1.1836,]),
#     'a5': np.array([0.,+0.0010, +0.0013, +0.0148, +0.0218, +0.0914, +0.3578, +0.6293, +0.8332, +0.8990, +0.7534, +0.3230, +0.4973,]),
#     'a6': np.array([0.,+0.0001, +0.0005, +0.0010, +0.0015, +0.0021, +0.0045, +0.0081, +0.0106, +0.0096, +0.0100, +0.0072, +0.0038,]),
#     'a7': np.array([0.,+0.0052, -0.0020, -0.0043, -0.0172, -0.0078, +0.1115, +0.2086, +0.1336, -0.2272, -0.3352, -0.4632, -0.4477,])
# }

_keunig_end_slopes = {
    k: float((_keunig_coefs[k][-1] - _keunig_coefs[k][-2]) /
             (_keunig_coefs["Fr"][-1] - _keunig_coefs["Fr"][-2]))
    for k in (f"a{i}" for i in range(8))
}

_keunig_interpolators = {}

for i in range(8):
    _keunig_interpolators[f"a{i}"] = InterpolatedModel(
        x_data_coordinates=_keunig_coefs["Fr"],
        y_data_structured=_keunig_coefs[f"a{i}"],
        # method="linear",
        method="bspline",
    )


# DSYHS added wave resistance coefs
_wave_coefs = np.genfromtxt(
    io.StringIO(
"""
2.0 100 0.0064 1.8012 0.0094 1.6076 0.0131 1.4260 0.0170 1.2701 0.0000 0.0000
2.0 115 0.0702 1.2351 0.1253 1.0662 0.2020 0.7105 0.2958 0.4759 0.0000 0.0000
2.0 125 0.1778 0.9700 0.2924 0.7191 0.4691 0.4367 0.6982 0.1580 0.0000 0.0000
2.0 135 0.3367 0.7520 0.4790 0.5351 0.7962 0.2292 0.0000 0.0000 0.0000 0.0000
2.0 145 0.5349 0.5876 0.7386 0.3868 1.1418 0.0633 0.0000 0.0000 0.0000 0.0000
2.5 100 0.0022 2.1441 0.0031 1.9747 0.0043 1.7986 0.0055 1.6561 0.0071 1.4768
2.5 115 0.0217 1.6837 0.0383 1.4654 0.0609 1.2603 0.0876 1.0757 0.1283 0.8442
2.5 125 0.0537 1.4670 0.0884 1.2926 0.1408 1.0782 0.2054 0.8712 0.3089 0.5981
2.5 135 0.0999 1.3090 0.1515 1.1733 0.2403 0.9466 0.3544 0.7159 0.5456 0.4009
2.5 145 0.1360 1.2874 0.2196 1.0797 0.3444 0.8444 0.5117 0.5931 0.8044 0.2401
3.0 100 0.0010 2.3044 0.0014 2.1272 0.0019 1.9711 0.0025 1.7965 0.0031 1.6447
3.0 115 0.0092 1.8811 0.0161 1.6875 0.0255 1.5033 0.0365 1.3381 0.0523 1.1408
3.0 125 0.0226 1.6904 0.0369 1.5517 0.0585 1.3686 0.0847 1.1929 0.1234 0.9741
3.0 135 0.0418 1.5562 0.0630 1.4652 0.0993 1.2809 0.1450 1.0916 0.2145 0.8488
3.0 145 0.0642 1.4592 0.0908 1.4015 0.1423 1.2141 0.2087 1.0130 0.3126 0.7488
3.5 100 0.0005 2.3809 0.0007 2.2141 0.0010 2.0250 0.0013 1.8954 0.0016 1.7004
3.5 115 0.0046 1.9846 0.0082 1.7948 0.0129 1.6215 0.0184 1.4634 0.0261 1.2793
3.5 125 0.0115 1.7995 0.0186 1.6757 0.0294 1.5105 0.0422 1.3514 0.0609 1.1546
3.5 135 0.0210 1.6782 0.0317 1.6078 0.0497 1.4467 0.0724 1.2761 0.1050 1.0676
3.5 145 0.0324 1.5891 0.0456 1.5598 0.0712 1.3989 0.1040 1.2222 0.1522 0.9995
4.0 100 0.0003 2.4046 0.0004 2.2345 0.0006 2.0808 0.0007 1.9466 0.0010 1.7326
4.0 115 0.0026 2.0357 0.0046 1.8532 0.0073 1.6824 0.0103 1.5325 0.0148 1.3493
4.0 125 0.0065 1.8550 0.0105 1.7445 0.0166 1.5870 0.0238 1.4336 0.0341 1.2495
4.0 135 0.0119 1.7434 0.0180 1.6810 0.0281 1.5238 0.0408 1.3743 0.0588 1.1803
4.0 145 0.0183 1.6596 0.0258 1.6418 0.0401 1.4981 0.0586 1.3336 0.0850 1.1309
4.5 100 0.0002 2.4482 0.0003 2.2604 0.0004 2.0584 0.0005 1.9708 0.0006 1.7525
4.5 115 0.0017 2.0487 0.0028 1.8848 0.0044 1.7216 0.0064 1.5621 0.0090 1.3874
4.5 125 0.0040 1.8939 0.0064 1.7856 0.0101 1.6294 0.0147 1.4736 0.0209 1.2971
4.5 135 0.0073 1.7805 0.0109 1.7283 0.0171 1.5848 0.0249 1.4305 0.0359 1.2425
4.5 145 0.0112 1.6989 0.0157 1.6910 0.0246 1.5529 0.0359 1.3961 0.0518 1.2037
5.0 100 0.0001 2.0466 0.0002 2.3517 0.0002 2.1274 0.0003 1.9235 0.0003 1.8738
5.0 115 0.0011 2.0596 0.0019 1.8990 0.0029 1.7384 0.0042 1.5856 0.0059 1.4108
5.0 125 0.0026 1.9051 0.0042 1.8020 0.0066 1.6550 0.0094 1.5094 0.0135 1.3302
5.0 135 0.0047 1.8004 0.0071 1.7548 0.0112 1.6107 0.0161 1.4648 0.0232 1.2797
5.0 145 0.0072 1.7275 0.0102 1.7217 0.0159 1.5872 0.0232 1.4363 0.0336 1.2460
5.5 100 6.0e-5 2.6609 0.0001 2.3753 0.0002 2.0611 0.0002 2.0358 0.0002 1.8756
5.5 115 0.0007 2.0844 0.0012 1.9306 0.0019 1.7600 0.0028 1.5994 0.0039 1.4301
5.5 125 0.0018 1.9197 0.0028 1.8161 0.0044 1.6773 0.0064 1.5267 0.0091 1.3506
5.5 135 0.0032 1.8225 0.0048 1.7664 0.0075 1.6315 0.0109 1.4841 0.0157 1.3046
5.5 145 0.0050 1.7307 0.0069 1.7372 0.0107 1.6102 0.0158 1.4579 0.0229 1.2703
6.0 100 5.0e-5 2.5121 6.0e-5 2.4574 9.0e-5 2.2831 0.0001 2.0042 0.0002 1.6893
6.0 115 0.0005 2.0700 0.0009 1.9220 0.0014 1.7578 0.0020 1.6112 0.0028 1.4332
6.0 125 0.0012 1.9246 0.0020 1.8259 0.0031 1.6867 0.0046 1.5244 0.0064 1.3600
6.0 135 0.0022 1.8258 0.0034 1.7814 0.0053 1.6457 0.0077 1.4950 0.0111 1.3136
6.0 145 0.0034 1.7506 0.0049 1.7446 0.0076 1.6232 0.0111 1.4746 0.0161 1.2867
"""
    )
)

# Generate structured coordinates for the 3D function: f(T1_prime, mu, Fn)
_wave_coords = {
    "T1_prime": np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]), # dimensionless relative wave period
    "mu": np.array([100, 115, 125, 135, 145]), # 
    "Fr": np.array([0.15, 0.25, 0.35, 0.45, 0.60]), # Froude number
}

_wave_interpolators = {}

# Reshape the raw columns into a structured grid (9x5x5)
# Using standard "ij" indexing equivalent, the data naturally unpacks exactly as the coordinates above
grid_shape = (
    len(_wave_coords["T1_prime"]),
    len(_wave_coords["mu"]),
    len(_wave_coords["Fr"]),
)

# Columns [2, 4, 6, 8, 10] map to 'a' at Fn=0.15, 0.25, 0.35, 0.45, 0.60 respectively
a_data_structured = _wave_coefs[:, [2, 4, 6, 8, 10]].reshape(grid_shape)

# Columns [3, 5, 7, 9, 11] map to 'b' at Fn=0.15, 0.25, 0.35, 0.45, 0.60 respectively
b_data_structured = _wave_coefs[:, [3, 5, 7, 9, 11]].reshape(grid_shape)

# 3. Create the interpolators using your custom object
# Note: You need two distinct objects since a and b are separate physical coefficients
_wave_interpolators["a"] = InterpolatedModel(
    x_data_coordinates=_wave_coords,
    y_data_structured=a_data_structured,
    method="bspline"
)

_wave_interpolators["b"] = InterpolatedModel(
    x_data_coordinates=_wave_coords,
    y_data_structured=b_data_structured,
    method="bspline"
)
    

def compute_Rf_dsyhs(
        Re,
        rho,
        Aws,
        Atr,
        stw,
        **kwargs,
    ):
    """
    Calculates the frictional resistance (Rf) based on the 
    Delft Systematic Yacht Hull series methodology.
    
    Parameters:
    -----------
    Re : float, Reynolds number (dimensionless).
    rho : float, Density of the water (typically in kg/m^3).
    Aws : float, Wetted surface area of the hull (typically in m^2).
    Atr : float, Transom area of the hull (typically in m^2).
    stw : float,  Speed through water in knots.
        
    Returns:
    --------
    Rf : float
        Frictional resistance force (typically in Newtons).
    """
    V = stw * u.kt # speed through water in m/s
    
    A = Aws - Atr
    
    Rf = 0.5 * Cf_hull(Re) * rho * A * V**2
    return Rf


def compute_Rw_dsyhs(
        Fr,
        rho,
        g,
        volume,
        Lwl,
        Bwl,
        T,
        Awp,
        Cp,
        Cx,
        LCB_fpp,
        LCF_fpp,
        **kwargs,
    ):
    """
    Calculates the residuary resistance (Rw) based on the 
    Delft Systematic Yacht Hull series polynomial equation.
    
    Parameters:
    -----------
    Fr      : float, Froude number (valid range roughly 0.15 - 0.75)
    volume  : float, Canoe body volume displacement (∇) [m^3]
    rho     : float, Density of water [kg/m^3]
    g       : float, Acceleration due to gravity [m/s^2]
    LCB_fpp : float, Longitudinal Center of Buoyancy from fpp [m]
    Lwl     : float, Waterline length [m]
    Cp      : float, Prismatic coefficient
    Awp     : float, Waterplane area [m^2]
    Bwl     : float, Waterline beam [m]
    LCF_fpp : float, Longitudinal Center of Flotation from fpp [m]
    T       : float, Canoe body draft [m]
    Cx      : float, Midship section coefficient
    
    Returns:
    --------
    Rw     : float, Residuary resistance [N]
    """
    
    # Fr_max = _keunig_coefs["Fr"][-1]
    Fr_max = 0.6
    
    excess_Fr = np.softplus(Fr - Fr_max, 10)
    Fr = smooth_clip(Fr, 0, Fr_max, beta=10)
    
    a = []

    for i in range(8):
        a.append(
            _keunig_interpolators[f"a{i}"](Fr)
        )
    
    # Calculate dimensional ratios
    vol_ratio = (volume**(1/3)) / (Lwl + 1e-12)
    
    term1 = (
        (a[1] * LCB_fpp / (Lwl + 1e-12))
        + (a[2] * Cp)
        + (a[3] * (volume**(2/3)) / (Awp + 1e-12))
        + (a[4] * Bwl / (Lwl + 1e-12))
    )
    term2 = (
        (a[5] * LCB_fpp / (LCF_fpp + 1e-12))
        + (a[6] * Bwl / (T + 1e-12))
        + (a[7] * Cx)
    )
    
    # Calculate the non-dimensional resistance coefficient
    resistance_coeff = a[0] + (term1 * vol_ratio) + (term2 * vol_ratio)
    
    # print(extrap_Fr)
    
    # print(Fr, (a[6] * Bwl / (T + 1e-12)))
    # print(1 + excess_Fr)
    
    resistance_coeff *= (1 + excess_Fr)
    
    if not np.is_casadi_type(resistance_coeff):
        resistance_coeff = np.squeeze(resistance_coeff)
    
    # Convert back to dimensional resistance in Newtons
    Rw = resistance_coeff * volume * rho * g
    
    return Rw


def compute_Raw_dsyhs(
        rho,
        g,
        Lwl,
        swh,
        volume,
        kyy,
        T1,
        mu,
        Fr,
        **kwargs,
    ):
    """
    Calculates the added resistance in waves (R_aw) based on the 
    Delft Systematic Yacht Hull series empirical formula, utilizing
    custom 3D interpolators for the 'a' and 'b' coefficients.
    
    Parameters:
    -----------
    rho      : float or ndarray
        Density of the water [kg/m^3] (e.g., 1025 for seawater).
    g        : float
        Acceleration due to gravity [m/s^2] (e.g., 9.81).
    Lwl     : float or ndarray
        Waterline length of the hull [m].
    swh     : float or ndarray
        Significant wave height [m].
    volume    : float or ndarray
        Volume displacement of the canoe body (∇c) [m^3].
    kyy     : float or ndarray
        Longitudinal radius of gyration [m].
    T1      : float or ndarray
        Wave period [s].
    mu       : float or ndarray
        Wave heading angle [degrees] (100 to 145 per DSYHS limits).
    Fn       : float or ndarray
        Froude number (0.15 to 0.60 per DSYHS limits).
    interp_a : InterpolatedModel
        Instantiated 3D interpolator object for the 'a' coefficient.
    interp_b : InterpolatedModel
        Instantiated 3D interpolator object for the 'b' coefficient.
        
    Returns:
    --------
    R_aw : float or ndarray
        Added resistance in waves [N].
    """
    
    # 1. Calculate the non-dimensional wave period (T1')
    T1_prime = T1 * np.sqrt(g / Lwl)
    
    # 2. Query the interpolators for the coefficients
    # Note: If your custom InterpolatedModel does not support vectorized NumPy arrays 
    # as coordinate inputs, you may need to wrap this step in a loop or np.vectorize
    wave_query = {
        "T1_prime": T1_prime,
        "mu": mu,
        "Fr": Fr,
    }
    a = _wave_interpolators["a"](wave_query)
    b = _wave_interpolators["b"](wave_query)
    
    # 3. Compute the structural terms of the equation
    vol_ratio = (volume**(1/3)) / Lwl
    gyration_ratio = kyy / Lwl
    
    bracket_term = 100 * vol_ratio * gyration_ratio
    
    # 4. Calculate final added wave resistance
    rhs = a * (bracket_term ** b)
    R_aw = (rhs * rho * g * Lwl * (swh**2)) / 100
    
    return R_aw


def compute_Rrr_dsyhs(
        **kwargs,
    ):
    pass #TODO implement roughness influence
    
    
def compute_Rtr_dsyhs(
        stw,
        Ttr,
        Atr,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculate transom resistance.
    """
    
    Vms = stw * u.kt
    
    Fr_T = Vms / (np.sqrt(g * Ttr + 1e-12) + 1e-12)
    
    return transom_resistance(
            Vms,
            Fr_T,
            Atr,
            rho,
        )


if __name__=="__main__":
    # VERIFICATION
    # --- Example Usage based on the image's check values ---
    # The image notes: Rb / (∇ * p * g) = 0.00649 at Fr = 0.35
    # Assuming arbitrary hull dimensions that yield exactly this coefficient to prove the math:
    
    # Mock dimensions # TODO find correct mock dimensions for verification
    env_params = {
        'g': 9.81,         # m/s^2
        'rho': 1025.0,     # kg/m^3 (seawater)
        'swh': 0.4,
        'T1': 2.8,
        'mu': 135.,
    }
    
    # hull_params = {
    #     'Fr': 0.35,
    #     # 'Fr': np.linspace(0.35, 0.5, 10),
    #     'volume': 6.0,      # m^3 (approx 6 tons)
    #     'LCB_fpp': 5.2,    # m
    #     'Lwl': 10.0,      # m
    #     'Cp': 0.55,
    #     'Awp': 18.0,       # m^2
    #     'Bwl': 3.0,       # m
    #     'LCF_fpp': 5.4,    # m
    #     'T': 0.8,        # m
    #     'Cx': 0.70,
    #     'kyy': 2.685, # m
        
    #     'Aws': 15.,
    # }
    
    # MOLENEZ II
    hull_params = {
        'Fr': 0.3,
    
        'volume': 116.7926694549928,
        'cob': np.array([12.2, 0.003, 0.494]),
        'Aws': 157.1220345584358,
        'cow': np.array([11.8, 0.00324, 0.510]),
        'cof': np.array([10.9, 0.0065, 1.30]),
    
        'T': 1.299766705208458,
        'Ttr': -0.0,
    
        'Lwl': 25.393312454223633,
        'Lbp': 25.393312454223633,
        'Bwl': 7.2994184494018555,
    
        'Ax': 7.2594194330079045,
        'Ay': 26.872448496209117,
        'Atr': 0.5045104756899772,
        'Awp': 138.4459100965084,
    
        'Cb': 0.48477761712401507,
        'Cp': 0.6335696116907322,
        'Cx': 0.7651528864054807,
        'Cy': 0.8141838335623939,
        'Cwp': 0.7469172905442572,
    
        'ie': 18.520232193238826,
    
        'fpp': np.array([25.39, 0.0, 1.3]),
        'app': np.array([0.0, 0.0, 1.3]),
    
        'LCB_fpp': 13.163823206145308,
        'LCF_fpp': 14.520357168645651,
        'lcb': -0.018397244545218405,
    
        'Abt': 0.0,
        'hB': 0.649883352604229,
    
        # To be supplied separately:
        # 'kyy': ...,
    }
        
    # TRIMMED 49er
    # hull_params = {'volume': 0.1393596784036322,
    #      'cob': np.array([ 5.26e-01,  1.51e-07, -1.45e-02]),
    #      'Aws': 2.087198803536013,
    #      'cow': np.array([ 4.60e-01,  2.44e-08, -3.75e-02]),
    #      'cof': np.array([ 1.23e+00,  7.90e-08, -2.61e-02]),
    #      'T': 0.27538997584829705,
    #      'Ttr': 0.27538997584829705,
    #      'Lwl': 1.61396370576692,
    #      'Lbp': 1.61396370576692,
    #      'Bwl': 1.4763862903018294,
    #      'Ax': 0.3189070527300096,
    #      'Ay': 0.18171187951249143,
    #      'Atr': 0.1786702866868349,
    #      'Awp': 2.977836578024428,
    #      'Cb': 0.21237101346426612,
    #      'Cp': 0.2707567218574761,
    #      'Cx': 0.784361001295639,
    #      'Cy': 0.4088287492963387,
    #      'Cwp': 0.999,
    #      'ie': 0.0020676786524235916,
    #      'fpp': np.array([ 1.81e+00,  7.33e-06, -1.04e-01]),
    #      'app': np.array([2.15e-01, 7.33e-06, 1.11e-01]),
    #      'LCB_fpp': 1.2885814676990712,
    #      'LCF_fpp': 0.5852325387107029,
    #      'lcb': -0.29839556682346036,
    #      'Abt': 0.0,
    #      'hB': 0.13769498792414853}
    
    # hull_params["stw"] = hull_params['Fr']*np.sqrt(env_params['g']*hull_params['Lwl'])/u.kt
    # hull_params["Re"] = hull_params['stw']*u.kt / (1.220e-6 * hull_params['Lwl'])
    
    # Rf = compute_Rf_dsyhs(**hull_params, **env_params)
    # Rw = compute_Rw_dsyhs(**hull_params, **env_params)
    # # Raw = compute_Raw_dsyhs(**hull_params, **env_params)
    # Raw = 0.
    # print(f"Boat speed: {hull_params['Fr']*np.sqrt(env_params['g']*hull_params['Lwl'])/u.kt:.2f} kts")
    # print(f"Calculated Frictionnal Resistance: {Rf:.2f} N")
    # print(f"Calculated Residuary Resistance: {Rw:.2f} N")
    # print(f"Calculated Added wave Resistance: {Raw:.2f} N")
    
    import matplotlib.pyplot as plt
    
    # Assuming env_params, hull_params, and your compute functions are already defined 
    # above this code block, along with `archibald.toolbox.units as u`.
    
    # 1. Define the range of Froude numbers to sweep (e.g., 0.15 to 0.60)
    fr_array = np.linspace(0.0, 1., 50)
    
    # Arrays to store the calculated resistances
    Rf_list = []
    Rw_list = []
    Raw_list = []
    
    # Kinematic viscosity of seawater approx 1.18e-6 to 1.22e-6 m^2/s depending on temp
    nu = 1.220e-6 
    
    # 2. Iterate through Froude numbers and calculate components
    for fr in fr_array:
        # Update Fr dependent variables
        hull_params['Fr'] = fr
        
        # Calculate speed in m/s, then convert to knots for your dictionary
        v_ms = fr * np.sqrt(env_params['g'] * hull_params['Lwl'])
        hull_params['stw'] = v_ms / u.kt
        
        # Corrected Reynolds number: Re = (V * L) / nu
        hull_params['Re'] = (v_ms * hull_params['Lwl']) / nu 
        
        # Compute resistances (make sure you pass the interpolator objects if required 
        # by your compute_Raw_dsyhs signature, e.g., interp_a=interp_a_3d, etc.)
        Rf = compute_Rf_dsyhs(**hull_params, **env_params)
        Rw = compute_Rw_dsyhs(**hull_params, **env_params)
        # Raw = compute_Raw_dsyhs(**hull_params, **env_params)
        Raw = 0.
        
        Rf_list.append(Rf)
        Rw_list.append(Rw)
        Raw_list.append(Raw)
    
    # Convert to numpy arrays for element-wise addition
    Rf_arr = np.array(Rf_list)
    Rw_arr = np.array(Rw_list)
    Raw_arr = np.array(Raw_list)
    
    # 3. Create the stacked area plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # Define cumulative sums for the stacked fill_between
    layer1 = Rf_arr
    layer2 = Rf_arr + Rw_arr
    layer3 = Rf_arr + Rw_arr + Raw_arr
    
    # Base layer: Frictional Resistance
    ax.fill_between(fr_array, 0, layer1, color='#1f77b4', alpha=0.5, label='Frictional ($R_f$)')
    ax.plot(fr_array, layer1, color='#1f77b4', linewidth=1.5)
    
    # Second layer: Residuary Resistance
    ax.fill_between(fr_array, layer1, layer2, color='#ff7f0e', alpha=0.5, label='Residuary ($R_w$)')
    ax.plot(fr_array, layer2, color='#ff7f0e', linewidth=1.5)
    
    # Third layer: Added Wave Resistance
    ax.fill_between(fr_array, layer2, layer3, color='#2ca02c', alpha=0.5, label='Added Wave ($R_{aw}$)')
    ax.plot(fr_array, layer3, color='#2ca02c', linewidth=1.5)
    
    # 4. Beautify the plot
    ax.set_title('Delft Systematic Yacht Hull Series\nComponents of Resistance vs. Froude Number', fontsize=14, pad=15)
    ax.set_xlabel('Froude Number ($F_n$)', fontsize=12)
    ax.set_ylabel('Resistance (N)', fontsize=12)
    
    # Add a grid that stays behind the plotted elements
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    # Add legend and adjust layout
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.xlim([fr_array.min(), fr_array.max()])
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    # PLOT
    import matplotlib.pyplot as plt
    x_coords = _keunig_coefs["Fr"]
    x_fine = np.linspace(
        _keunig_coefs["Fr"].min(), 
        _keunig_coefs["Fr"].max(), 
        100
    )
    
    for i in range(8):
        # 3. Evaluate the interpolator
        
        y_data = _keunig_coefs[f"a{i}"]
        y_interp = _keunig_interpolators[f"a{i}"](x_fine)
    
        # 4. Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(x_coords, y_data, 'ok', label="Original Data Points")
        plt.plot(x_fine, y_interp, '-', label="B-Spline Interpolation")
        plt.legend()
        plt.grid()
        plt.title(f"1D Interpolation for coef a{i}")
        plt.xlabel("Froude")
        # plt.ylabel("y")
        plt.show()