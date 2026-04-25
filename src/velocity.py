"""Receiver velocity estimation from GNSS Doppler measurements.

════════════════════════════════════════════════════════════════════
THEORY

The measured Doppler shift D1C [Hz] on the L1 carrier is related to the
instantaneous pseudorange rate ρ̇_i [m/s] by:

    ρ̇_i = −λ_L1 · D1C_i

where λ_L1 = c / f_L1 ≈ 0.19029 m  (GPS L1 wavelength).

The pseudorange rate decomposes as:

    ρ̇_i = ê_i · (v_sat_i − v_rx) + c · ḋ_rx

where:
    ê_i     = unit LOS vector from receiver to satellite   (dimensionless)
    v_sat_i = satellite ECEF velocity [m/s]                (from sat_position.py)
    v_rx    = receiver ECEF velocity [m/s]                 (3 unknowns)
    ḋ_rx    = receiver clock drift rate [s/s → m/s here]   (1 unknown)

Rearranging into a linear system for x = [vx, vy, vz, d_dot]ᵀ:

    −ê_i · v_rx + d_dot = ρ̇_i − ê_i · v_sat_i

    ⎡ −ê_1x  −ê_1y  −ê_1z   1 ⎤ ⎡ vx    ⎤   ⎡ b_1 ⎤
    ⎢ −ê_2x  −ê_2y  −ê_2z   1 ⎥ ⎢ vy    ⎥ = ⎢ b_2 ⎥
    ⎢  ...                   . ⎥ ⎢ vz    ⎥   ⎢ ... ⎥
    ⎣ −ê_Nx  −ê_Ny  −ê_Nz   1 ⎦ ⎣ d_dot ⎦   ⎣ b_N ⎦

    H · x = b

with  b_i = ρ̇_i − ê_i · v_sat_i

N = 4  → square system   : x = H⁻¹ · b
N > 4  → Gauss-Newton LS : x = (HᵀH)⁻¹ · Hᵀ · b   (analogous to PDF Part 6)

This formulation is linear, so no iteration is needed once the receiver
position pos_rx (from solve_position) and satellite velocities (from
sat_position.compute_sat_state) are available.

════════════════════════════════════════════════════════════════════
ADVANTAGES OVER FINITE DIFFERENCING

    Finite diff: v ≈ (pos_t − pos_{t-1}) / Δt
        • Differentiates noisy position → amplifies position errors
        • Fails on position outliers and bad epoch pairs
        • Typical accuracy: 0.5–2 m/s on a pedestrian trace

    Doppler: ρ̇ = −λ · D1C  (direct measurement, no time difference)
        • D1C is an instantaneous measurement with ~0.003 Hz noise
        • Equivalent to ~0.6 mm/s velocity noise at L1
        • Independent of position solver quality
        • Works even if the previous epoch was invalid (NaN)
        • Typical accuracy: 0.05–0.1 m/s on a pedestrian trace
════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

# ─── GPS L1 constants ─────────────────────────────────────────────────────────
C          = 299_792_458.0           # speed of light        [m/s]
L1_FREQ    = 1_575_420_000.0         # GPS L1 carrier        [Hz]
L1_LAMBDA  = C / L1_FREQ            # GPS L1 wavelength     [m] ≈ 0.19029 m


def solve_velocity(
    vel_data: list[tuple[np.ndarray, np.ndarray, float]],
    pos_rx:   np.ndarray,
) -> np.ndarray | None:
    """Solve receiver ECEF velocity from GPS Doppler measurements.

    Parameters
    ----------
    vel_data : list of (pos_sat, vel_sat, doppler_hz)
        pos_sat    : np.ndarray([X, Y, Z])        satellite ECEF position  [m]
                     (Sagnac-corrected, from compute_sat_state)
        vel_sat    : np.ndarray([vX, vY, vZ])     satellite ECEF velocity  [m/s]
                     (finite-diff at ±0.5 s, from compute_sat_state)
        doppler_hz : float D1C observation                                 [Hz]
                     (positive = satellite approaching receiver)
    pos_rx   : np.ndarray([X, Y, Z]) receiver ECEF position [m]
               (from solve_position — needed to build LOS unit vectors)

    Returns
    -------
    np.ndarray([vx, vy, vz, d_dot]) or None on failure.
        vx, vy, vz : receiver ECEF velocity                               [m/s]
        d_dot      : receiver clock drift rate (in distance units)        [m/s]
                     (c × clock_drift_s_per_s)

    Notes
    -----
    Requires N ≥ 4 valid Doppler measurements (same minimum as position).
    The returned velocity is in ECEF — convert to ENU/NED as needed.
    """
    N = len(vel_data)
    if N < 4:
        return None

    # ── Build linear system H · x = b ────────────────────────────────────────
    H = np.empty((N, 4))    # geometry matrix
    b = np.empty(N)         # right-hand side (observed pseudorange rates)

    for i, (pos_sat, vel_sat, doppler_hz) in enumerate(vel_data):

        # ── LOS unit vector  ê_i = (r_sat − r_rx) / ‖r_sat − r_rx‖ ─────────
        los  = pos_sat - pos_rx[:3]
        dist = float(np.linalg.norm(los))
        if dist < 1e3:
            # Sanity: satellite < 1 km away is physically impossible
            return None
        e_los = los / dist          # unit vector receiver → satellite

        # ── Pseudorange rate from Doppler ─────────────────────────────────────
        # RINEX D1C sign convention:
        #   D1C > 0 → satellite approaching → pseudorange decreasing → ρ̇ < 0
        # Hence:  ρ̇ = −λ · D1C
        rho_dot = -L1_LAMBDA * doppler_hz

        # ── Row of H (observation equation) ──────────────────────────────────
        # −ê_i · v_rx + d_dot = ρ̇_i − ê_i · v_sat_i
        H[i, 0] = -e_los[0]    # coefficient of vx
        H[i, 1] = -e_los[1]    # coefficient of vy
        H[i, 2] = -e_los[2]    # coefficient of vz
        H[i, 3] =  1.0         # coefficient of d_dot [m/s]

        b[i] = rho_dot - float(np.dot(e_los, vel_sat))

    # ── Solve linear system ───────────────────────────────────────────────────
    # Analogous to the position solver (PDF Part 1 / Part 6):
    #   N = 4 : square system  → H · x = b
    #   N > 4 : least-squares  → HᵀH · x = Hᵀ · b
    try:
        if N == 4:
            x = np.linalg.solve(H, b)
        else:
            # Normal equations (same structure as Gauss-Newton in solver.py)
            HTH = H.T @ H       # 4×4
            HTb = H.T @ b       # 4×1
            x   = np.linalg.solve(HTH, HTb)
    except np.linalg.LinAlgError:
        # Singular geometry (degenerate satellite configuration)
        return None

    return x   # [vx, vy, vz, d_dot]  all in [m/s]
