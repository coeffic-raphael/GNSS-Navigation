"""WGS-84 geodetic coordinate conversions.

ecef_to_lla         : ECEF (X, Y, Z) -> (lat_deg, lon_deg, alt_m) (Bowring)
elev_azim_from_ecef : satellite elevation/azimuth from receiver position
"""
from __future__ import annotations

import math

# WGS-84 ellipsoid
_A  = 6_378_137.0          # semi-major axis [m]
_F  = 1.0 / 298.257223563  # flattening
_B  = _A * (1.0 - _F)      # semi-minor axis [m]
_E2 = 2 * _F - _F ** 2     # first eccentricity squared


def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF -> (latitude_deg, longitude_deg, altitude_m) on WGS-84.

    Uses iterative Bowring method (converges in 3 iterations).
    Accuracy: < 0.1 mm globally.
    """
    lon = math.atan2(y, x)
    p   = math.sqrt(x * x + y * y)

    # Bowring initialization
    theta = math.atan2(z * _A, p * _B)
    lat   = math.atan2(
        z + (_E2 / (1.0 - _E2)) * _B * math.sin(theta) ** 3,
        p - _E2 * _A * math.cos(theta) ** 3,
    )

    # Three iterations are enough in practice
    for _ in range(3):
        N   = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        lat = math.atan2(z + _E2 * N * math.sin(lat), p)

    N   = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) / math.sin(lat) - N * (1.0 - _E2)

    return math.degrees(lat), math.degrees(lon), alt


def elev_azim_from_ecef(
    sat_pos: tuple[float, float, float],
    rx_pos:  tuple[float, float, float],
) -> tuple[float, float]:
    """Satellite elevation and azimuth as seen from receiver [rad].

    Returns (elevation, azimuth). Azimuth: 0=north, pi/2=east in local ENU.
    Elevation: 0=horizon, pi/2=zenith.
    """
    sx, sy, sz = sat_pos
    rx, ry, rz = rx_pos
    dx, dy, dz = sx - rx, sy - ry, sz - rz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1.0:
        return math.pi / 2.0, 0.0

    lat_deg, lon_deg, _ = ecef_to_lla(rx, ry, rz)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl   = math.sin(lat), math.cos(lat)
    slo, clo = math.sin(lon), math.cos(lon)

    # ECEF to local ENU vector at receiver location
    e = -slo * dx + clo * dy
    n = -sl * clo * dx - sl * slo * dy + cl * dz
    u =  cl * clo * dx + cl * slo * dy + sl * dz

    sin_elev = max(-1.0, min(1.0, u / dist))
    elev = math.asin(sin_elev)
    azim = math.atan2(e, n)
    if azim < 0:
        azim += 2 * math.pi
    return elev, azim
