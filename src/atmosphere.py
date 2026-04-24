"""Atmospheric corrections for GNSS pseudoranges.

Two standard models for single-frequency receivers:

- Klobuchar (IS-GPS-200, ICD) for the ionosphere
- Saastamoinen (1972) for the troposphere

The outputs are delays (seconds for ionosphere, meters for troposphere) that
must be subtracted from measured pseudorange:

    rho_corrected = rho_measured − c·T_iono − T_tropo

Typical mid-latitude daytime values at moderate elevation:
  T_iono ~ 5-15 m   (up to ~30 m at tropical zenith in high solar activity)
  T_tropo ~ 2-10 m  (up to ~30 m at low elevation)
"""
from __future__ import annotations

import math
import numpy as np

C = 299_792_458.0  # speed of light [m/s]


# ─── Klobuchar (L1 ionosphere) ───────────────────────────────────────────────

def klobuchar_delay(
    alpha: np.ndarray,
    beta:  np.ndarray,
    lat_rx_rad: float,
    lon_rx_rad: float,
    elev_rad:   float,
    azim_rad:   float,
    gps_tow_s:  float,
) -> float:
    """Compute L1 ionospheric delay in seconds with Klobuchar model.

    IS-GPS-200 / ICD-GPS-200C algorithm, section 20.3.3.5.2.5.

    Parameters
    ----------
    alpha, beta : Klobuchar coefficients (4 each) from nav header.
    lat_rx_rad, lon_rx_rad : receiver WGS-84 position [rad].
    elev_rad, azim_rad     : satellite direction [rad].
    gps_tow_s              : GPS time-of-week [s] (0 to 604800).

    Returns
    -------
    Delay in seconds. Multiply by c for meters.
    """
    # Convert to semicircles (pi rad = 1 semicircle)
    SEMI = 1.0 / math.pi
    elev_sc = elev_rad * SEMI
    azim    = azim_rad

    # Earth-centered angle (semicircles)
    psi = 0.0137 / (elev_sc + 0.11) - 0.022

    # Sub-ionospheric latitude (semicircles)
    phi_u_sc = lat_rx_rad * SEMI
    phi_i = phi_u_sc + psi * math.cos(azim)
    phi_i = max(-0.416, min(0.416, phi_i))

    # Sub-ionospheric longitude (semicircles)
    lam_u_sc = lon_rx_rad * SEMI
    lam_i = lam_u_sc + psi * math.sin(azim) / math.cos(phi_i * math.pi)

    # Geomagnetic latitude (semicircles)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)

    # Local time (seconds, bound 0–86400)
    t = 43200.0 * lam_i + gps_tow_s
    t = t % 86400.0
    if t < 0:
        t += 86400.0

    # Delay amplitude and period
    AMP = alpha[0] + phi_m * (alpha[1] + phi_m * (alpha[2] + phi_m * alpha[3]))
    PER = beta[0]  + phi_m * (beta[1]  + phi_m * (beta[2]  + phi_m * beta[3]))
    AMP = max(AMP, 0.0)
    PER = max(PER, 72000.0)

    # Phase (rad)
    X = 2.0 * math.pi * (t - 50400.0) / PER

    # Obliquity factor
    F = 1.0 + 16.0 * (0.53 - elev_sc) ** 3

    # Vertical delay model
    if abs(X) < 1.57:
        T_iono = F * (5.0e-9 + AMP * (1.0 - X**2 / 2.0 + X**4 / 24.0))
    else:
        T_iono = F * 5.0e-9

    return T_iono


# ─── Saastamoinen (troposphere) ──────────────────────────────────────────────

def saastamoinen_delay(
    lat_rx_rad: float,
    alt_rx_m:   float,
    elev_rad:   float,
    P_hPa:      float = 1013.25,
    T_K:        float = 288.15,
    e_hPa:      float = 11.691,
) -> float:
    """Compute total tropospheric delay in meters (Saastamoinen 1972).

    Full model with wet term and zenith-to-slant mapping.
    Valid for elevation >= 3 degrees.

    Default meteo parameters use ISA sea-level atmosphere.
    Local meteo data would improve precision but is usually unavailable here.

    Parameters
    ----------
    lat_rx_rad : receiver latitude [rad]
    alt_rx_m   : receiver ellipsoidal altitude [m]
    elev_rad   : satellite elevation [rad]
    P_hPa      : atmospheric pressure (1013.25 hPa standard sea level)
    T_K        : temperature [K] (288.15 K = 15 C)
    e_hPa      : water vapor pressure [hPa] (~11.7 hPa at 15 C, 50% RH)

    Returns
    -------
    Delay in meters (to subtract from pseudorange).
    """
    # Gravity correction for latitude and altitude
    f_lat = 1.0 - 0.00266 * math.cos(2 * lat_rx_rad) - 0.00028e-3 * alt_rx_m

    # Zenith angle
    z = math.pi / 2.0 - max(elev_rad, math.radians(3.0))
    cos_z = math.cos(z)
    tan_z = math.tan(z)

    # Zenith delay (Saastamoinen)
    # ZHD (dry) + ZWD (wet), mapped to slant by 1/cos(z)
    zenith_delay = (0.002277 / f_lat) * (P_hPa + (1255.0 / T_K + 0.05) * e_hPa
                                          - tan_z ** 2)

    return zenith_delay / cos_z


# ─── Klobuchar parameter extraction ───────────────────────────────────────────

def get_klobuchar_params(nav: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract GPS Klobuchar (alpha, beta) from a nav dict (nav_loader).

    Special key ``_klobuchar`` is injected by ``nav_loader.load_nav_files``
    from ``IONOSPHERIC CORR`` header lines in RINEX 3 nav files.

    Returns None if coefficients are absent.
    """
    if not isinstance(nav, dict):
        return None
    if "_klobuchar" not in nav:
        return None
    alpha, beta = nav["_klobuchar"]
    return np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
