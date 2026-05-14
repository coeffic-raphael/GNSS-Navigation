"""Compare altitude smoothing against NMEA GGA fixes.

Usage:
    python src/compare_alt_smoothing.py \
        --csv output/session_21_03_short.csv \
        --nmea "data/gnss_log_2026_03_21_17_14_34 (1).nmea" \
        --input-is-smoothed

By default, --csv is treated as an unsmoothed trajectory and the script compares:
    - without_smoothing: original alt_m
    - with_smoothing   : same lat/lon, alt_m filtered as
                         0.5 * current + 0.5 * previous

If the CSV was already produced by the current pipeline, pass
--input-is-smoothed. The script first reconstructs the raw altitude with:
    raw_alt = 2 * smoothed_alt - previous_smoothed_alt
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from validate import haversine_m, parse_gngga


def _time_of_day_s(utc_time: pd.Series) -> pd.Series:
    return (
        utc_time.dt.hour * 3600
        + utc_time.dt.minute * 60
        + utc_time.dt.second
        + utc_time.dt.microsecond / 1e6
    )


def smooth_altitude(alt: pd.Series, previous_weight: float = 0.5) -> pd.Series:
    """Apply the same valid-fix-only altitude smoothing as main.py."""
    current_weight = 1.0 - previous_weight
    smoothed = alt.copy()
    last_alt = None

    for idx, value in alt.items():
        if not np.isfinite(value):
            continue
        if last_alt is not None:
            value = current_weight * float(value) + previous_weight * last_alt
        smoothed.loc[idx] = value
        last_alt = float(value)

    return smoothed


def unsmooth_altitude(smoothed_alt: pd.Series, previous_weight: float = 0.5) -> pd.Series:
    """Reconstruct raw altitude from the 1-pole smoother used in main.py."""
    current_weight = 1.0 - previous_weight
    raw = smoothed_alt.copy()
    last_smoothed = None

    for idx, value in smoothed_alt.items():
        if not np.isfinite(value):
            continue
        if last_smoothed is not None:
            value = (float(value) - previous_weight * last_smoothed) / current_weight
        raw.loc[idx] = value
        last_smoothed = float(smoothed_alt.loc[idx])

    return raw


def build_matches(calc: pd.DataFrame, nmea: pd.DataFrame, max_dt_s: float) -> pd.DataFrame:
    errors: list[dict[str, float | str]] = []

    for _, row in calc.iterrows():
        if not (
            np.isfinite(row["lat_deg"])
            and np.isfinite(row["lon_deg"])
            and np.isfinite(row["alt_m"])
        ):
            continue

        idx = (nmea["time_s"] - row["time_s"]).abs().idxmin()
        dt = abs(float(nmea.loc[idx, "time_s"] - row["time_s"]))
        if dt > max_dt_s:
            continue

        lat_calc = float(row["lat_deg"])
        lon_calc = float(row["lon_deg"])
        alt_calc = float(row["alt_m"])
        lat_nmea = float(nmea.loc[idx, "lat"])
        lon_nmea = float(nmea.loc[idx, "lon"])
        alt_nmea = float(nmea.loc[idx, "alt"])
        horiz_err = haversine_m(lat_calc, lon_calc, lat_nmea, lon_nmea)
        alt_err = alt_calc - alt_nmea
        errors.append({
            "utc_time": row["utc_time"].isoformat(),
            "dt_s": dt,
            "lat_calc_deg": lat_calc,
            "lon_calc_deg": lon_calc,
            "alt_calc_m": alt_calc,
            "lat_nmea_deg": lat_nmea,
            "lon_nmea_deg": lon_nmea,
            "alt_nmea_m": alt_nmea,
            "horiz_err_m": horiz_err,
            "alt_err_m": alt_err,
            "abs_alt_err_m": abs(alt_err),
            "err_3d_m": sqrt(horiz_err ** 2 + alt_err ** 2),
        })

    return pd.DataFrame(errors)


def summarize(label: str, matches: pd.DataFrame) -> dict[str, float | str]:
    if matches.empty:
        return {"variant": label, "n": 0}

    err = matches["alt_err_m"]
    abs_err = matches["abs_alt_err_m"]
    horiz = matches["horiz_err_m"]
    err_3d = matches["err_3d_m"]
    alt = matches["alt_calc_m"]
    jump = alt.diff().abs().dropna()

    return {
        "variant": label,
        "n": int(len(matches)),
        "horiz_median_m": float(horiz.median()),
        "horiz_rms_m": float(np.sqrt((horiz ** 2).mean())),
        "horiz_p95_m": float(horiz.quantile(0.95)),
        "horiz_max_m": float(horiz.max()),
        "alt_bias_m": float(err.mean()),
        "alt_mae_m": float(abs_err.mean()),
        "alt_median_abs_m": float(abs_err.median()),
        "alt_rms_m": float(np.sqrt((err ** 2).mean())),
        "alt_p95_abs_m": float(abs_err.quantile(0.95)),
        "alt_max_abs_m": float(abs_err.max()),
        "err_3d_median_m": float(err_3d.median()),
        "err_3d_rms_m": float(np.sqrt((err_3d ** 2).mean())),
        "err_3d_p95_m": float(err_3d.quantile(0.95)),
        "alt_median_jump_m": float(jump.median()) if not jump.empty else 0.0,
        "alt_p95_jump_m": float(jump.quantile(0.95)) if not jump.empty else 0.0,
    }


def compare_altitude(
    csv_path: str | Path,
    nmea_path: str | Path,
    max_dt_s: float,
    previous_weight: float,
    input_is_smoothed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calc = pd.read_csv(csv_path, parse_dates=["utc_time"])
    calc["time_s"] = _time_of_day_s(calc["utc_time"])

    nmea = parse_gngga(str(nmea_path))
    if nmea.empty:
        raise ValueError(f"No valid GNGGA/GPGGA frames found in {nmea_path}")

    raw = calc.copy()
    if input_is_smoothed:
        raw["alt_m"] = unsmooth_altitude(raw["alt_m"], previous_weight)

    smoothed = raw.copy()
    smoothed["alt_m"] = smooth_altitude(smoothed["alt_m"], previous_weight)

    raw_matches = build_matches(raw, nmea, max_dt_s)
    smoothed_matches = build_matches(smoothed, nmea, max_dt_s)

    summary = pd.DataFrame([
        summarize("without_smoothing", raw_matches),
        summarize(f"with_smoothing_prev_{previous_weight:g}", smoothed_matches),
    ])

    details = pd.concat(
        [
            raw_matches.assign(variant="without_smoothing"),
            smoothed_matches.assign(variant=f"with_smoothing_prev_{previous_weight:g}"),
        ],
        ignore_index=True,
    )

    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixes with/without altitude smoothing against NMEA GGA."
    )
    parser.add_argument("--csv", default="output/session_21_03_short.csv", help="Trajectory CSV")
    parser.add_argument(
        "--nmea",
        default="data/gnss_log_2026_03_21_17_14_34 (1).nmea",
        help="NMEA file containing GNGGA/GPGGA altitude",
    )
    parser.add_argument("--max-dt", type=float, default=2.0, help="Maximum time delta for matching (s)")
    parser.add_argument(
        "--previous-weight",
        type=float,
        default=0.5,
        help="Weight of previous altitude in the smoother",
    )
    parser.add_argument(
        "--input-is-smoothed",
        action="store_true",
        help="Use this when the input CSV already includes altitude smoothing from main.py",
    )
    parser.add_argument("--out-details", help="Optional CSV path for per-epoch comparison rows")
    args = parser.parse_args()

    summary, details = compare_altitude(
        args.csv,
        args.nmea,
        max_dt_s=args.max_dt,
        previous_weight=args.previous_weight,
        input_is_smoothed=args.input_is_smoothed,
    )

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out_details:
        out_path = Path(args.out_details)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(out_path, index=False)
        print(f"\nDetails written: {out_path} ({len(details)} rows)")


if __name__ == "__main__":
    main()
