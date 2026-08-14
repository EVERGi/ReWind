"""
mass_formulas.py

Closed-form per-component mass formulas, extracted verbatim from
fleet_evaluation_lca_algebraic.py (line numbers refer to that file). Shared by
generate_material_importance.py (decomposes these into raw materials) and
generate_feature_importance.py (uses the component totals directly as RF features), so the
formulas live in exactly one place instead of two copies drifting apart.

Every formula here is pure algebra, not a new LCA run: fleet_evaluation_lca_algebraic.py already
builds each of these as a SymPy expression while constructing its Input-phase exchange dict; this
module reproduces the same closed-form coefficients directly in numpy.
"""
import numpy as np


def M_tower(d, h):
    """Tower steel mass, kg. fleet_evaluation_lca_algebraic.py:423."""
    return (3.03584782e-04 * d ** 2 * h + 9.68652909e+00) * 1e3


def M_nacelle_onshore(P):
    """fleet_evaluation_lca_algebraic.py:424."""
    return (1.66691134e-06 * P ** 2 + 3.20700974e-02 * P) * 1e3


def M_nacelle_offshore(P):
    """Offshore-specific coefficients. fleet_evaluation_lca_algebraic.py:1138."""
    return (2.15668283e-06 * P ** 2 + 3.24712680e-02 * P) * 1e3


def M_rotor_onshore(d):
    """Rotor (blades+hub) mass, kg. fleet_evaluation_lca_algebraic.py:425."""
    return (0.00460956 * d ** 2 + 0.11199577 * d) * 1e3


def M_rotor_offshore(d):
    """Offshore-specific coefficients. fleet_evaluation_lca_algebraic.py:1139."""
    return (0.0088365 * d ** 2 + -0.16435292 * d) * 1e3


def M_elec(P):
    """Nacelle electronics/electrical mass, kg (same table onshore and offshore).
    fleet_evaluation_lca_algebraic.py:430."""
    return np.interp(P, [30, 150, 600, 800, 2000], [150, 300, 862, 1112, 3946])


def M_reinf(P):
    """Foundation reinforcement steel, kg (onshore only). fleet_evaluation_lca_algebraic.py:429."""
    return np.interp(P, [750, 2000, 4500], [10210, 27000, 51900])


def M_found_onshore(d, h):
    """Onshore gravity-base foundation total mass, kg. fleet_evaluation_lca_algebraic.py:426."""
    return 1696e3 * h / 80 * d ** 2 / 10000


def M_cable_onshore(P, dist_to_grid_m):
    """fleet_evaluation_lca_algebraic.py:437."""
    return (300.0 * P / 21516.0) * 1e-6 * dist_to_grid_m * 8960.0 * (617.0 / 220.0) * 0.5


def M_cable_offshore(P, dist_to_grid_m, park_size):
    """Two-segment offshore cable (turbine->transformer + shared park->shore export cable).
    fleet_evaluation_lca_algebraic.py:1206-1232."""
    DIST_TRANSFO_M = 1000.0  # 1 km, fixed, never given a real per-turbine value anywhere
    cross_section1 = 300.0 * P / 21516.0
    m_copper = cross_section1 * 1e-6 * DIST_TRANSFO_M * 8960.0
    df33_P = [11616, 13167, 14718, 16566, 19173, 21516, 23958, 26763, 29832, 32769]
    df33_idx = [95, 120, 150, 185, 240, 300, 400, 500, 630, 800]
    df150_P = [106500, 122250, 138750, 156750, 174000, 200250, 213750, 234000]
    df150_idx = [400, 500, 630, 800, 1000, 1200, 1600, 2000]
    park_power = P * park_size
    cross_section2 = np.where(
        park_power <= 30000,
        np.interp(park_power, df33_P, df33_idx),
        np.interp(park_power, df150_P, df150_idx),
    )
    m_copper = m_copper + cross_section2 * 1e-6 * (dist_to_grid_m / park_size) * 8960.0
    return (m_copper * 617.0 / 220.0) * 0.5


def foundation_masses_offshore(bucket, P, sea_depth_m):
    """fleet_evaluation_lca_algebraic.py:_foundation_mass_formulas, lines 1068-1091.
    Spar buoy reproduces a known unit bug in the baseline (tonnes used as kg) faithfully,
    for comparability; see that function's docstring.

    Returns {component: mass_kg}; callers that want ONE total foundation mass (e.g. as a
    single RF feature) should sum the dict's values."""
    if bucket == "offshore_monopile":
        m_grout = 2.44094476537839 * P + 2585.70316709778 * sea_depth_m + 15406.5926965901
        m_monopile = 5.28641521017598 * P + 5599.92210615505 * sea_depth_m + 102643.279628469
        return {"grout": m_grout, "material": m_monopile}
    elif bucket == "offshore_semi-submersible":
        m_semi_sub = 289.473684210526 * P + 17828.5714285714 * sea_depth_m + 80000.0
        return {"material": m_semi_sub}
    elif bucket == "offshore_spar":
        m_spar_steel = 0.383333333333333 * P + 6.84 * sea_depth_m + 300.0
        m_spar_iron = 0.833333333333333 * P
        return {"steel": m_spar_steel, "iron": m_spar_iron}
    raise ValueError(bucket)
