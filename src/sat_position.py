"""Compute GPS satellite state from broadcast ephemerides (RINEX nav).

Implements IS-GPS-200 equations (Table 20-IV).

Input  : `nav` dict = {sv: [eph_record, ...]} from nav_loader,
         satellite ID, receive time, measured pseudorange.
Output : (pos, dt_sat, vel) -> ECEF [m], [s], [m/s]
         or None if no valid ephemeris is available.
"""
from __future__ import annotations

import numpy as np

from nav_loader import find_ephemeris

# ─── Constants ────────────────────────────────────────────────────────────────
# mu (GM) and Earth rate: common WGS-84 values.
MU      = 3.986005e14        # Earth GM                [m^3/s^2]
OMEGA_E = 7.2921151467e-5    # Earth angular rate      [rad/s]
C       = 299_792_458.0      # speed of light          [m/s]
F_REL   = -4.442807633e-10   # relativistic correction [s/sqrt(m)]

# GPS epoch (1980-01-06 00:00:00 UTC)
_GPS_EPOCH_NP = np.datetime64("1980-01-06T00:00:00", "ns")


# ─── Time helpers ─────────────────────────────────────────────────────────────

def _dt64_to_gps_s(t: np.datetime64) -> float:
    """Convert numpy.datetime64 to absolute GPS seconds."""
    return float((np.datetime64(t, "ns") - _GPS_EPOCH_NP) / np.timedelta64(1, "s"))


def _wrap_tk(tk: float) -> float:
    """Wrap tk into [-302400, +302400] to handle GPS week rollover."""
    if tk >  302400.0:
        tk -= 604800.0
    if tk < -302400.0:
        tk += 604800.0
    return tk


# ─── Kepler equations (IS-GPS-200 Table 20-IV) ───────────────────────────────

def _kepler_position(eph: dict, t_tx_s: float) -> tuple[float, float, float, float]:
    """Compute (X, Y, Z, dt_sat) in ECEF for one GPS ephemeris record."""
    sqrtA    = eph["sqrtA"]
    A        = sqrtA ** 2
    e        = eph["e"]
    M0       = eph["M0"]
    delta_n  = eph["DeltaN"]
    Omega0   = eph["Omega0"]
    OmegaDot = eph["OmegaDot"]
    i0       = eph["Io"]
    IDOT     = eph["IDOT"]
    omega    = eph["omega"]
    Crc, Crs = eph["Crc"], eph["Crs"]
    Cuc, Cus = eph["Cuc"], eph["Cus"]
    Cic, Cis = eph["Cic"], eph["Cis"]

    toe_abs  = eph["toe_abs"]
    toe_sow  = eph["Toe"]

    tk = _wrap_tk(t_tx_s - toe_abs)

    # ── Orbital motion ───────────────────────────────────────────────────────
    n0 = np.sqrt(MU / A ** 3)
    n  = n0 + delta_n
    M  = M0 + n * tk

    # Eccentric anomaly (Newton iterations)
    E = M
    for _ in range(10):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))

    # True anomaly
    nu  = np.arctan2(np.sqrt(1.0 - e ** 2) * np.sin(E), np.cos(E) - e)
    Phi = nu + omega

    # Harmonic corrections
    sin2, cos2 = np.sin(2 * Phi), np.cos(2 * Phi)
    du  = Cuc * cos2 + Cus * sin2
    dr  = Crc * cos2 + Crs * sin2
    di  = Cic * cos2 + Cis * sin2

    u   = Phi + du
    r   = A * (1.0 - e * np.cos(E)) + dr
    inc = i0 + di + IDOT * tk

    # Position in orbital plane
    xp = r * np.cos(u)
    yp = r * np.sin(u)

    # Corrected longitude of ascending node
    Omega = Omega0 + (OmegaDot - OMEGA_E) * tk - OMEGA_E * toe_sow

    # ECEF
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc),   np.sin(inc)
    X = xp * cO - yp * ci * sO
    Y = xp * sO + yp * ci * cO
    Z = yp * si

    # ── Satellite clock correction ───────────────────────────────────────────
    dt_clk = t_tx_s - eph["toc_abs"]
    af0 = eph["af0"]; af1 = eph["af1"]; af2 = eph["af2"]
    dt_sv = af0 + af1 * dt_clk + af2 * dt_clk ** 2

    # Relativistic correction
    dt_rel = F_REL * e * sqrtA * np.sin(E)

    # Group delay correction (GPS TGD on L1)
    TGD = eph.get("TGD", 0.0)
    if not np.isfinite(TGD):
        TGD = 0.0

    dt_sat = dt_sv + dt_rel - TGD

    return X, Y, Z, dt_sat


# ─── Public interface ─────────────────────────────────────────────────────────

def compute_sat_state(
    nav: dict,
    sv: str,
    t_rx_dt64: np.datetime64,
    pseudorange: float,
    x_rx: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Compute (pos, dt_sat, vel) for a GPS satellite.

    Parameters
    ----------
    nav        : dict {sv: [records]} from nav_loader.load_nav_files()
    sv         : satellite ID, e.g. 'G01'
    t_rx_dt64  : receive time numpy.datetime64 (GPS time)
    pseudorange: measured C1C pseudorange in meters
    x_rx       : approximate receiver ECEF position (optional, improves
                 transmit-time refinement)

    Returns
    -------
    (pos, dt_sat, vel)
        pos    : np.ndarray([X, Y, Z]) ECEF meters (with Sagnac correction)
        dt_sat : float satellite clock offset [s]
        vel    : np.ndarray([vX, vY, vZ]) ECEF m/s (finite difference, dt=0.5 s)
    or None if sv is not GPS, no ephemeris is found, or pseudorange is invalid.
    """
    if sv[0] != "G":
        return None

    if not np.isfinite(pseudorange) or pseudorange <= 0:
        return None

    t_rx_s = _dt64_to_gps_s(t_rx_dt64)
    t_tx_s = t_rx_s - pseudorange / C

    # ── Transmit-time refinement loop ────────────────────────────────────────
    for _ in range(3):
        eph = find_ephemeris(nav, sv, t_tx_s)
        if eph is None:
            return None
        X, Y, Z, _ = _kepler_position(eph, t_tx_s)
        if x_rx is not None:
            geom = float(np.linalg.norm(np.array([X, Y, Z]) - x_rx))
        else:
            geom = pseudorange
        t_tx_s = t_rx_s - geom / C

    # ── Final position and clock offset ──────────────────────────────────────
    eph = find_ephemeris(nav, sv, t_tx_s)
    if eph is None:
        return None

    X, Y, Z, dt_sat = _kepler_position(eph, t_tx_s)

    # Sagnac correction: Earth rotation during signal flight time
    dt_flight = pseudorange / C
    theta = OMEGA_E * dt_flight
    ct, st = np.cos(theta), np.sin(theta)
    pos = np.array([
         X * ct + Y * st,
        -X * st + Y * ct,
         Z,
    ])

    # ── Satellite velocity via centered finite differences (dt = 0.5 s) ─────
    half = 0.5
    Xp, Yp, Zp, _ = _kepler_position(eph, t_tx_s + half)
    Xm, Ym, Zm, _ = _kepler_position(eph, t_tx_s - half)
    vel = np.array([
        (Xp - Xm) / (2 * half),
        (Yp - Ym) / (2 * half),
        (Zp - Zm) / (2 * half),
    ])

    return pos, dt_sat, vel
