"""GNSS navigation solver using multivariate Newton on GPS equations.

Implementation faithful to "Solving the GPS Equations"
(Harnam Arneja, Andrew Bender, Sam Jugus, Tim Reid).

════════════════════════════════════════════════════════════════════
EQUATIONS (PDF Part 1) — for each satellite i:

    F_i(x, y, z, d) = (x − A_i)² + (y − B_i)² + (z − C_i)² − (c·(t_i − d))² = 0

Unknowns:
    (x, y, z) = receiver ECEF position                      [m]
    d         = receiver clock bias                         [s]
Known per satellite:
    (A_i, B_i, C_i) = satellite ECEF position               [m]   (via sat_position.py)
    t_i             = signal flight time                     [s]
                    = rho_i / c + dt_sat_i  (rho_i = measured pseudorange)
    c               = speed of light                         [m/s]

Geometric interpretation: the euclidean distance to satellite i must
equal c·(t_i − d). Four unknowns, so N ≥ 4 satellites required.

JACOBIAN (partial derivatives of F_i, PDF Part 1):
    ∂F_i/∂x = 2·(x − A_i)
    ∂F_i/∂y = 2·(y − B_i)
    ∂F_i/∂z = 2·(z − C_i)
    ∂F_i/∂d = 2·c²·(t_i − d)

════════════════════════════════════════════════════════════════════
METHOD (multivariate Newton):

    N = 4  :  square system        D · Δg = −F              (PDF Part 1)
    N > 4  :  least-squares        DᵀD · Δg = −Dᵀ·F         (PDF Part 6)

Δg = (Δx, Δy, Δz, Δd) is the correction at each iteration; we update
g ← g + Δg until ‖Δ(x,y,z)‖ < tol.

Initial seed: (0, 0, 6_370_000 m, 0 s) — from PDF Part 4
(center of the Earth shifted by its average radius along Z).

The PDF states: "This method usually converges for the GPS equations to
7 correct digits in less than 20 iterations." In practice with this seed
we converge in 2-3 iterations.

════════════════════════════════════════════════════════════════════
TYPICAL CONVERGENCE (see reference Excel table):

    iter 0 : g = (0, 0, 6 370 000, 0)  →  RMS(F) ≈ 255 km
    iter 1 : after 1st Δg              →  RMS(F) ≈ 9.53 m
    iter 2 : after 2nd Δg              →  RMS(F) ≈ 0     → break
════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

C = 299_792_458.0     # speed of light [m/s]


def solve_position(
    sat_data: list[tuple[np.ndarray, float, float]],
    x0: np.ndarray | None = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> np.ndarray | None:
    """Solve (x, y, z, d) with multivariate Newton on the PDF system.

    Parameters
    ----------
    sat_data : list of (pos_sat, dt_sat, rho_meas)
        pos_sat  : np.ndarray([A_i, B_i, C_i]) satellite ECEF position [m]
        dt_sat   : satellite clock correction [s]
        rho_meas : measured pseudorange [m] (t_i = rho_meas/c + dt_sat)
    x0       : initial seed (x, y, z, d). None -> (0, 0, 6_370_000, 0).
    max_iter : maximum iterations (PDF: "less than 20").
    tol      : convergence threshold on ||Delta(x,y,z)|| in meters.

    Returns
    -------
    np.ndarray([x, y, z, d]) or None if N < 4 or numerical divergence.
    """
    # ── Safeguard: 4 unknowns, need at least 4 equations ─────────────────────
    # (PDF Part 1 starts with exactly N=4; Part 6 generalizes to N>4)
    N = len(sat_data)
    if N < 4:
        return None

    # ── Arrays (A_i, B_i, C_i, t_i) in PDF notation ──────────────────────────
    # Transform sat_data into 4 numpy vectors so we can compute F and D
    # in a vectorized fashion (single pass over the N satellites).
    A  = np.empty(N)          # PDF: A_i, satellite X ECEF component
    B  = np.empty(N)          # PDF: B_i, satellite Y ECEF component
    Cv = np.empty(N)          # PDF: C_i (renamed Cv to avoid collision with C=c)
    t  = np.empty(N)          # PDF: t_i, flight time corrected for dt_sat
    for i, (pos_sat, dt_sat, rho) in enumerate(sat_data):
        A[i]  = pos_sat[0]
        B[i]  = pos_sat[1]
        Cv[i] = pos_sat[2]
        # t_i = rho/c + dt_sat : "true" flight time after sat clock correction.
        # (the sat clock has already been computed in sat_position.py; here
        #  we just add it to the raw propagation time.)
        t[i]  = rho / C + dt_sat

    # ── Initial seed g = (x, y, z, d) ────────────────────────────────────────
    # PDF Part 4: "a good starting point is the center of the Earth shifted
    # up by its average radius along Z, and a small clock bias". We use
    # d=0 instead of 0.0001 s (negligible effect on convergence).
    # => iter 0 of the Excel table: RMS(F) ≈ 255 km before correction.
    if x0 is None:
        g = np.array([0.0, 0.0, 6_370_000.0, 0.0])
    else:
        g = np.asarray(x0, dtype=float).copy()

    # ── Newton loop ──────────────────────────────────────────────────────────
    # Each iteration: evaluate F(g), evaluate D(g), solve the linear system,
    # update g ← g + Δg. Exits when the spatial correction < tol.
    for _ in range(max_iter):
        x, y, z, d = g

        # ── F(g): vector of N residuals (PDF Part 1) ─────────────────────────
        #   F_i = (x−A_i)² + (y−B_i)² + (z−C_i)² − (c·(t_i − d))²
        # Interpretation: |pos_sat − pos_rx|² − (distance inferred from time)².
        # F = 0 for all i <=> we are at the correct position AND bias d
        # exactly explains the measured flight times.
        ct = C * (t - d)                                          # c·(t_i − d)
        F  = (x - A) ** 2 + (y - B) ** 2 + (z - Cv) ** 2 - ct ** 2

        # ── D(g): (N, 4) Jacobian of partial derivatives of F ────────────────
        # PDF Part 1: the 4 columns are ∂F/∂x, ∂F/∂y, ∂F/∂z, ∂F/∂d.
        D = np.empty((N, 4))
        D[:, 0] = 2.0 * (x - A)            # ∂F_i/∂x = 2·(x − A_i)
        D[:, 1] = 2.0 * (y - B)            # ∂F_i/∂y = 2·(y − B_i)
        D[:, 2] = 2.0 * (z - Cv)           # ∂F_i/∂z = 2·(z − C_i)
        D[:, 3] = 2.0 * C ** 2 * (t - d)   # ∂F_i/∂d = 2·c²·(t_i − d)

        # ── Linear system resolution ─────────────────────────────────────────
        # PDF Part 1 (N=4): square system, classical Newton.
        # PDF Part 6 (N>4): over-determined system, normal equations
        #   (Gauss-Newton least-squares).
        try:
            if N == 4:
                # Exact Newton: D · Δg = −F
                delta = np.linalg.solve(D, -F)
            else:
                # Gauss-Newton (normal equations): DᵀD · Δg = −Dᵀ·F
                DTD  = D.T @ D                  # 4×4 matrix (invertible unless
                                                # satellites are coplanar)
                DTF  = D.T @ F                  # 4×1 vector
                delta = np.linalg.solve(DTD, -DTF)
        except np.linalg.LinAlgError:
            # Degenerate geometry (D singular): bail out; caller will drop
            # this epoch or add/remove a satellite.
            return None

        # ── Update g ← g + Δg ────────────────────────────────────────────────
        # Corresponds to the "iter 1", "iter 2"... rows of the Excel table.
        g = g + delta

        # ── Stopping criterion ───────────────────────────────────────────────
        # We test ONLY the norm of the spatial correction Δ(x,y,z), not Δd,
        # because d is in seconds (different unit). 1e-4 m = 0.1 mm, well
        # below pseudorange noise (~1 m).
        # => iter 2 of the Excel table: ‖Δ(x,y,z)‖ < tol, break.
        if np.linalg.norm(delta[:3]) < tol:
            break

    return g
