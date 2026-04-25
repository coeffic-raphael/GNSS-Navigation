"""Main GNSS navigation pipeline (RINEX → positions/velocities → CSV + KML).

Usage:
    python src/main.py                    # phase1 by default (120 epochs)
    python src/main.py --session phase2   # medium session (280 epochs)
    python src/main.py --session phase3   # long session (2590 epochs)
    python src/main.py --session all      # all 3 sessions

Outputs per session:
    output/<prefix>.csv
    output/<prefix>.kml

════════════════════════════════════════════════════════════════════
ALGORITHM (faithful to "Solving the GPS Equations" + ICD extensions)

For each epoch t:
    a. Read C1C (pseudorange), S1C (SNR), D1C (Doppler) for every visible
       GPS satellite.
    b. Quality filters: SNR ≥ 25 dB-Hz and |Doppler| ≥ 200 Hz. A real GPS
       satellite always shows significant radial velocity from the ground;
       a near-zero Doppler is typically an artifact (multipath,
       cross-correlation).
    c. Compute (A_i, B_i, C_i, dt_sat_i, vel_sat_i) via IS-GPS-200 Kepler
       equations + Sagnac correction (see sat_position.py).
    d. PASS 1 — solver.py without atmospheric correction → rough fix (~15 m).
       => Excel table iter 0 (RMS ≈ 255 km) → iter 1 (9.53 m) → iter 2 (~0).
    e. Klobuchar (ionosphere) + Saastamoinen (troposphere) atmospheric
       corrections computed from the rough fix.
    f. PASS 2 — solver.py on corrected pseudoranges → final fix (~2 m).
    g. Post-hoc outlier rejection: while any residual
         r_i = rho_meas_i − (||r − r_sat_i|| + c·d − c·dt_sat_i)
       exceeds 500 m, drop the worst satellite and resolve.
       (≤ 3 rejections per epoch).
    h. "On-Earth" filter: PDF Part 1 notes the system has two roots, one
       on the surface and one in space. Reject solutions outside
       [-500, +1500] m altitude.

2. Receiver velocity from Doppler measurements (velocity.py):
       ρ̇_i = −λ_L1 · D1C_i        (pseudorange rate from Doppler [m/s])
       H · [vx, vy, vz, ḋ]ᵀ = b   (linear system, same structure as PDF Part 6)
   This is a direct instantaneous measurement (~0.05 m/s accuracy) vs the
   finite-difference method (v = Δpos/Δt, ~1 m/s accuracy). The Doppler
   approach is independent of successive position fixes and works even when
   the previous epoch was invalid.

3. Export CSV + KML (Google Earth, with timestamps for chronological playback).
════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path

import numpy as np

from parser import load_observations
from nav_loader import load_nav_files
from sat_position import compute_sat_state
from solver import solve_position, C as C_LIGHT
from velocity import solve_velocity
from coordinates import ecef_to_lla, elev_azim_from_ecef
from atmosphere import klobuchar_delay, saastamoinen_delay, get_klobuchar_params
from export import write_csv, write_kml

# ─── Pipeline constants ──────────────────────────────────────────────────────

# Minimum geometry: 4 unknowns (x,y,z,d), so N ≥ 4 (PDF Part 1).
# We require 5 to have 1 degree of redundancy and enable Gauss-Newton
# (PDF Part 6: least-squares on the normal equations DᵀD·Δg = −DᵀF).
MIN_SAT            = 5

# Quality filters on RINEX observables
MIN_SNR_DB         = 25.0     # minimum SNR (S1C) [dB-Hz]
MIN_DOPPLER_HZ     = 200.0    # minimum |D1C| [Hz] — a Doppler near 0 usually
                              # indicates multipath / cross-correlation

# Post-hoc outlier rejection on pseudorange residuals
OUTLIER_THRESH_M   = 500.0    # tolerated residual threshold [m]
MAX_OUTLIER_REJECT = 3        # max satellites dropped per epoch

# "On-Earth" solution filter (PDF Part 1: the system has 2 roots, one at
# the surface and one in orbit). We bound altitude to what is expected for
# a pedestrian / vehicle trajectory from an Android phone.
ALT_MIN_M          = -500.0
ALT_MAX_M          = 1_500.0

LEAP_SECONDS = 18   # GPS-UTC offset valid for 2026-03-21/22

# ─── Sessions ────────────────────────────────────────────────────────────────

SESSIONS: dict[str, tuple[str, str, str]] = {
    "phase1": (
        "gnss_log_2026_03_21_17_14_34 (1).26o",
        "gnss_log_2026_03_21_17_14_34 (1).nmea",
        "session_21_03_short",
    ),
    "phase2": (
        "gnss_log_2026_03_21_17_17_57.26o",
        "gnss_log_2026_03_21_17_17_57.nmea",
        "session_21_03_long",
    ),
    "phase3": (
        "gnss_log_2026_03_22_08_44_21.26o",
        "gnss_log_2026_03_22_08_44_20.nmea",
        "session_22_03",
    ),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gps_to_utc(t_dt64: np.datetime64) -> str:
    """Convert numpy.datetime64 GPS time to UTC ISO string."""
    t_s = float((np.datetime64(t_dt64, "ns") -
                 np.datetime64("1980-01-06T00:00:00", "ns")) / np.timedelta64(1, "s"))
    utc_s = t_s - LEAP_SECONDS
    epoch = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
    utc_dt = epoch + datetime.timedelta(seconds=utc_s)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _gps_seconds(t_dt64: np.datetime64) -> float:
    """Convert numpy.datetime64 GPS time to absolute GPS seconds."""
    return float((np.datetime64(t_dt64, "ns") -
                  np.datetime64("1980-01-06T00:00:00", "ns")) / np.timedelta64(1, "s"))


def _residuals(
    pos_data: list[tuple[np.ndarray, float, float]],
    sol: np.ndarray,
) -> np.ndarray:
    """Compute pseudorange residuals after the solver has converged.

    residual_i = rho_meas_i − (||r_rx − r_sat_i|| + c·d_rx − c·dt_sat_i)

    Ideally all residuals ≈ 0 (noise + unmodeled errors ≈ 1-2 m).
    A |residual_i| >> 10 m reveals an outlier (multipath, cycle slip,
    bad ephemeris) that we drop in _solve_with_outlier_rejection.
    """
    x, y, z, d = sol
    res = np.empty(len(pos_data))
    for i, (pos_sat, dt_sat, rho) in enumerate(pos_data):
        # Geometric receiver-satellite distance
        r = float(np.sqrt((x - pos_sat[0]) ** 2
                         + (y - pos_sat[1]) ** 2
                         + (z - pos_sat[2]) ** 2))
        # Reconstructed pseudorange vs measured
        res[i] = rho - (r + C_LIGHT * d - C_LIGHT * dt_sat)
    return res


def _fill_gaps_with_nan(records: list[dict]) -> list[dict]:
    """Insert NaN-filled records for each missing second in the trajectory.

    Guarantees a strict 1 Hz output cadence from the first to the last fix.
    Missing seconds (epochs skipped by the solver) get a row with NaN
    position/velocity and ``n_sat=0`` — honest placeholder, not interpolation.

    Some Android RINEX captures drift slightly in their sub-second offsets
    (e.g. .416, .431, .491). To keep the exported trajectory strictly 1 Hz,
    we snap every output row onto the 1-second grid defined by the first fix
    and map each solved epoch to its nearest integer-second slot.
    """
    if not records:
        return records

    def _parse(utc_str: str) -> datetime.datetime:
        # Example: "2026-03-21T15:14:41.418Z"
        return datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00"))

    sorted_recs = sorted(records, key=lambda r: r["utc_time"])
    start_dt = _parse(sorted_recs[0]["utc_time"])
    end_dt   = _parse(sorted_recs[-1]["utc_time"])

    # Index real fixes by their integer-second offset from the first epoch.
    # If several raw epochs fall in the same slot, keep the latest one; the
    # final output timestamps are regenerated on the strict 1 Hz grid below.
    by_sec: dict[int, dict] = {}
    for r in sorted_recs:
        offset = (_parse(r["utc_time"]) - start_dt).total_seconds()
        by_sec[int(round(offset))] = r

    total_seconds = int(round((end_dt - start_dt).total_seconds()))
    filled: list[dict] = []
    for s in range(total_seconds + 1):
        t = start_dt + datetime.timedelta(seconds=s)
        if s in by_sec:
            row = dict(by_sec[s])
            row["utc_time"] = t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            filled.append(row)
        else:
            filled.append({
                "utc_time": t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "lat_deg":  float("nan"),
                "lon_deg":  float("nan"),
                "alt_m":    float("nan"),
                "vx_mps":   float("nan"),
                "vy_mps":   float("nan"),
                "vz_mps":   float("nan"),
                "n_sat":    0,
            })
    return filled


def _solve_with_outlier_rejection(
    pos_data: list[tuple[np.ndarray, float, float]],
    seed: np.ndarray | None,
) -> tuple[np.ndarray | None, int]:
    """Solve (x, y, z, d) while iteratively dropping the worst outlier.

    Loop: solve → compute residuals → if max |residual| > OUTLIER_THRESH_M,
    drop that satellite and retry. Stops after MAX_OUTLIER_REJECT drops
    or when all residuals are acceptable.

    This is a practical extension of the PDF: the PDF assumes F → 0, but
    on an Android phone a single satellite can keep an abnormal residual
    even after convergence (multipath). Removing it realigns the others.
    """
    current = list(pos_data)

    for _ in range(MAX_OUTLIER_REJECT + 1):
        # Keep N ≥ MIN_SAT so the system stays over-determined
        if len(current) < MIN_SAT:
            return None, 0

        # One call = one full Newton (2-3 iterations inside the solver)
        sol = solve_position(current, x0=seed)
        if sol is None:
            return None, 0

        res = _residuals(current, sol)
        i_worst = int(np.argmax(np.abs(res)))

        if abs(res[i_worst]) < OUTLIER_THRESH_M:
            # All residuals acceptable → solution validated
            return sol, len(current)

        # Otherwise drop the worst satellite and resolve
        current.pop(i_worst)

    # MAX_OUTLIER_REJECT reached: return the latest solution
    return sol, len(current)


# ─── Session processing ──────────────────────────────────────────────────────

def _apply_atmosphere(
    pos_data: list[tuple[np.ndarray, float, float]],
    rx_pos:   np.ndarray,
    klob:     tuple[np.ndarray, np.ndarray] | None,
    gps_tow_s: float,
) -> list[tuple[np.ndarray, float, float]]:
    """Correct pseudoranges for ionospheric and tropospheric effects.

    Models:
        - Klobuchar    (IS-GPS-200 §20.3.3.5.2.5) → L1 ionospheric delay
        - Saastamoinen (1972)                     → total tropospheric delay

    These corrections depend on the receiver position (satellite elev/azim,
    and lat/alt for Saastamoinen) — hence the 2-pass scheme in run_session:
    rough fix → compute corrections → final fix.
    """
    # Receiver latitude / longitude / altitude for the atmo models
    lat_deg, lon_deg, alt_m = ecef_to_lla(rx_pos[0], rx_pos[1], rx_pos[2])
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    corrected = []
    for pos_sat, dt_sat, rho in pos_data:
        # Satellite → receiver direction in local ENU coordinates
        elev, azim = elev_azim_from_ecef(
            (pos_sat[0], pos_sat[1], pos_sat[2]),
            (rx_pos[0], rx_pos[1], rx_pos[2]),
        )
        if elev < math.radians(3.0):
            # Avoid Saastamoinen's 1/cos(z) singularity at grazing elevation
            elev = math.radians(3.0)

        # ── Ionospheric delay (Klobuchar) ────────────────────────────────────
        # klobuchar_delay returns a delay in seconds → convert to meters via c.
        iono_m = 0.0
        if klob is not None:
            alpha, beta = klob
            T_iono = klobuchar_delay(alpha, beta, lat_rad, lon_rad,
                                     elev, azim, gps_tow_s)
            iono_m = C_LIGHT * T_iono

        # ── Tropospheric delay (Saastamoinen, already in meters) ─────────────
        tropo_m = saastamoinen_delay(lat_rad, alt_m, elev)

        # Corrected pseudorange (to feed back into solve_position)
        rho_corr = rho - iono_m - tropo_m
        corrected.append((pos_sat, dt_sat, rho_corr))

    return corrected


def run_session(obs_path: Path, nav: object, out_prefix: str) -> None:
    """Process one RINEX observation file and export CSV + KML."""
    print(f"\n{'='*60}")
    print(f"Session: {obs_path.name}")
    print(f"{'='*60}")

    obs = load_observations(obs_path, use=["G"])
    n_epochs = obs.time.size
    print(f"Loaded epochs: {n_epochs}")

    # Klobuchar parameters (loaded once at startup)
    klob = get_klobuchar_params(nav)
    if klob is None:
        print("  (no Klobuchar parameters found - ionosphere correction disabled)")
    else:
        print(f"  Klobuchar alpha={klob[0]} beta={klob[1]}")

    records   = []
    last_fix  = None     # last valid fix, used for x_rx in sat-state refinement
    n_ok      = 0
    n_skip    = 0

    for epoch_idx, t in enumerate(obs.time.values):
        epoch_obs = obs.sel(time=t)

        # ── Collect satellite measurements ───────────────────────────────────
        # pos_data : fed into solve_position  → (pos_sat, dt_sat, rho)
        # vel_data : fed into solve_velocity  → (pos_sat, vel_sat, doppler_hz)
        pos_data: list[tuple[np.ndarray, float, float]] = []
        vel_data: list[tuple[np.ndarray, np.ndarray, float]] = []

        for sv_val in epoch_obs.sv.values:
            sv = str(sv_val)
            if sv[0] != "G":
                continue

            rho     = float(epoch_obs["C1C"].sel(sv=sv_val).values) if "C1C" in obs else np.nan
            snr     = float(epoch_obs["S1C"].sel(sv=sv_val).values) if "S1C" in obs else np.nan
            doppler = float(epoch_obs["D1C"].sel(sv=sv_val).values) if "D1C" in obs else np.nan

            if not np.isfinite(rho) or rho <= 0:
                continue
            # Quality filters: reject noisy / physically implausible measurements.
            if np.isfinite(snr) and snr < MIN_SNR_DB:
                continue
            if np.isfinite(doppler) and abs(doppler) < MIN_DOPPLER_HZ:
                continue

            x_rx = last_fix[:3] if last_fix is not None else None
            state = compute_sat_state(nav, sv, t, rho, x_rx)
            if state is None:
                continue
            pos_sat, dt_sat, vel_sat = state
            pos_data.append((pos_sat, dt_sat, rho))

            # Doppler entry: only when D1C is a valid finite measurement
            if np.isfinite(doppler):
                vel_data.append((pos_sat, vel_sat, doppler))

        if len(pos_data) < MIN_SAT:
            n_skip += 1
            continue

        # ── Two-pass resolution (PDF Part 1/6 + atmo corrections) ────────────
        # PASS 1: multivariate Newton without atmospheric correction.
        # solve_position starts from the PDF seed (0, 0, 6370 km, 0) and
        # converges in 2-3 iterations (see Excel reference table:
        #   iter 0: RMS ≈ 255 km, iter 1: 9.53 m, iter 2: ~0).
        # The resulting fix is good to ~5-15 m (limited by residual ionosphere).
        pos_rough, _ = _solve_with_outlier_rejection(pos_data, None)
        if pos_rough is None:
            n_skip += 1
            continue

        # PASS 2: use pos_rough to compute Klobuchar + Saastamoinen (which
        # depend on the receiver position), then resolve on the corrected
        # pseudoranges. Typical gain: 15 m → 2-5 m.
        gps_tow = _gps_seconds(t) % 604_800.0
        pos_data_corr = _apply_atmosphere(pos_data, pos_rough, klob, gps_tow)
        pos, n_used = _solve_with_outlier_rejection(pos_data_corr, None)
        if pos is None:
            n_skip += 1
            continue

        # ECEF → WGS-84 geodetic (Bowring, see coordinates.py)
        lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])

        # "On-Earth" filter (PDF Part 1): the F_i=0 system admits two
        # solutions, one real at the surface and one "mirror" in orbit.
        # If Newton converged to the wrong root (rare with our seed), we
        # reject it via a simple altitude gate.
        if not (ALT_MIN_M <= alt <= ALT_MAX_M):
            n_skip += 1
            continue

        last_fix = pos.copy()

        # ── Receiver velocity from Doppler measurements (velocity.py) ────────
        # Build the linear system H·[vx,vy,vz,ḋ] = b where:
        #   ρ̇_i = −λ_L1 · D1C_i   (pseudorange rate from Doppler)
        #   ρ̇_i = ê_i·(v_sat_i − v_rx) + ḋ
        # This is a direct, instantaneous measurement (~0.05 m/s accuracy)
        # independent of successive position fixes.
        vx = vy = vz = 0.0
        vel_sol = solve_velocity(vel_data, pos) if len(vel_data) >= 4 else None
        if vel_sol is not None:
            vx, vy, vz = float(vel_sol[0]), float(vel_sol[1]), float(vel_sol[2])
            # vel_sol[3] = receiver clock drift rate [m/s] — not exported but available

        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        records.append({
            "utc_time": _gps_to_utc(t),
            "lat_deg":  lat,
            "lon_deg":  lon,
            "alt_m":    alt,
            "vx_mps":   float(vx),
            "vy_mps":   float(vy),
            "vz_mps":   float(vz),
            "n_sat":    n_used,
        })
        n_ok += 1

        if (epoch_idx + 1) % 50 == 0 or epoch_idx == 0:
            n_dropped = len(pos_data) - n_used
            extra = f" (-{n_dropped} outliers)" if n_dropped else ""
            vel_src = "D" if vel_sol is not None else "-"
            print(f"  [{epoch_idx+1:4d}/{n_epochs}]  "
                  f"lat={lat:.5f}°  lon={lon:.5f}°  alt={alt:.1f} m  "
                  f"v={speed:.2f} m/s [{vel_src}]  n_sat={n_used}{extra}")

    print(f"\nSolved epochs: {n_ok}/{n_epochs}  (skipped: {n_skip})")

    if not records:
        print("No valid fixes computed - check ephemerides and observation data.")
        return

    # Pad gaps with NaN rows to guarantee a strict 1 Hz output cadence.
    # KML/CSV writers ignore NaN coordinates for mapping but keep the
    # timestamps in the CSV, so coverage is documented explicitly.
    records_1hz = _fill_gaps_with_nan(records)
    n_filled = len(records_1hz) - len(records)
    if n_filled:
        print(f"Padded with {n_filled} NaN rows to reach strict 1 Hz cadence "
              f"({len(records_1hz)} total rows).")

    out_dir = Path("output")
    write_csv(records_1hz, out_dir / f"{out_prefix}.csv")
    write_kml(records_1hz, out_dir / f"{out_prefix}.kml")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline GNSS RINEX → KML/CSV")
    parser.add_argument(
        "--session",
        default="phase1",
        choices=["phase1", "phase2", "phase3", "all"],
        help="Session(s) to process (default: phase1)",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    nav_dir  = data_dir / "brdc" / "rinex" / "nav"
    nav_files = sorted(nav_dir.glob("*.rnx"))

    if not nav_files:
        raise FileNotFoundError(
            f"No BRDC file found in {nav_dir}\n"
            "Run first: python src/download_brdc.py"
        )

    print(f"Loading ephemerides ({len(nav_files)} files)...")
    nav = load_nav_files(nav_files)
    sat_keys = [k for k in nav if not k.startswith("_")]
    n_recs = sum(len(nav[k]) for k in sat_keys)
    systems = sorted({k[0] for k in sat_keys})
    print(f"Ephemerides loaded: {len(sat_keys)} satellites, "
          f"{n_recs} records - systems={systems}")

    sessions_to_run = list(SESSIONS.keys()) if args.session == "all" else [args.session]

    for key in sessions_to_run:
        obs_file, _, prefix = SESSIONS[key]
        obs_path = data_dir / obs_file
        if not obs_path.exists():
            print(f"WARNING: {obs_path} not found, skipping this session.")
            continue
        run_session(obs_path, nav, prefix)


if __name__ == "__main__":
    main()
