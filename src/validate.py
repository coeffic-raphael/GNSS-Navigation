"""Validate GNSS trajectory against NMEA (GNGGA).

Usage:
    python src/validate.py
    python src/validate.py --csv output/session_21_03_long.csv --nmea data/gnss_log_2026_03_21_17_17_57.nmea
"""

from __future__ import annotations

import argparse
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd


def parse_gngga(nmea_path: str) -> pd.DataFrame:
    """Extract (time_s, lat, lon, alt) from $GNGGA/$GPGGA frames."""
    records: list[dict[str, float]] = []
    with open(nmea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # Android logger format: "NMEA,$GNGGA,...,<unix_ms>"
            if line.startswith("NMEA,"):
                line = line[5:]

            if not (line.startswith("$GNGGA") or line.startswith("$GPGGA")):
                continue
            parts = line.split(",")
            if len(parts) < 10 or parts[2] == "" or parts[4] == "":
                continue

            # UTC time HHMMSS.ss
            t = parts[1]
            if len(t) < 6:
                continue
            h, m, s = int(t[0:2]), int(t[2:4]), float(t[4:])
            time_s = h * 3600 + m * 60 + s

            # Latitude DDMM.MMMM
            raw_lat = float(parts[2])
            lat = int(raw_lat / 100) + (raw_lat % 100) / 60
            if parts[3] == "S":
                lat = -lat

            # Longitude DDDMM.MMMM
            raw_lon = float(parts[4])
            lon = int(raw_lon / 100) + (raw_lon % 100) / 60
            if parts[5] == "W":
                lon = -lon

            alt = float(parts[9]) if parts[9] else 0.0
            records.append({"time_s": time_s, "lat": lat, "lon": lon, "alt": alt})

    return pd.DataFrame(records)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Horizontal distance in meters between two geodetic points."""
    r_earth = 6_371_000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return r_earth * 2 * atan2(sqrt(a), sqrt(1 - a))


def compare(trajectory_csv: str, nmea_path: str, max_dt_s: float = 2.0) -> pd.DataFrame:
    calc = pd.read_csv(trajectory_csv, parse_dates=["utc_time"])
    calc["time_s"] = (
        calc["utc_time"].dt.hour * 3600
        + calc["utc_time"].dt.minute * 60
        + calc["utc_time"].dt.second
        + calc["utc_time"].dt.microsecond / 1e6
    )

    nmea = parse_gngga(nmea_path)
    if nmea.empty:
        raise ValueError(f"No valid GNGGA/GPGGA frames found in {nmea_path}")

    errors: list[dict[str, float]] = []
    for _, row in calc.iterrows():
        idx = (nmea["time_s"] - row["time_s"]).abs().idxmin()
        dt = abs(float(nmea.loc[idx, "time_s"] - row["time_s"]))
        if dt > max_dt_s:
            continue

        d_horiz = haversine_m(
            float(row["lat_deg"]),
            float(row["lon_deg"]),
            float(nmea.loc[idx, "lat"]),
            float(nmea.loc[idx, "lon"]),
        )
        d_vert = abs(float(row["alt_m"]) - float(nmea.loc[idx, "alt"]))
        errors.append({"dt_s": dt, "d_horiz": d_horiz, "dalt": d_vert, "d_3d": sqrt(d_horiz**2 + d_vert**2)})

    df = pd.DataFrame(errors)
    if df.empty:
        print("No matched points found (time alignment failed).")
        return df

    print(f"Compared points: {len(df)}")
    print(
        "Horizontal error - "
        f"median: {df['d_horiz'].median():.1f} m  "
        f"RMS: {np.sqrt((df['d_horiz'] ** 2).mean()):.1f} m  "
        f"max: {df['d_horiz'].max():.1f} m"
    )
    print(
        "3D error         - "
        f"median: {df['d_3d'].median():.1f} m  "
        f"RMS: {np.sqrt((df['d_3d'] ** 2).mean()):.1f} m"
    )
    print(f"Median time delta: {df['dt_s'].median():.2f} s")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare computed trajectory against NMEA GNGGA.")
    parser.add_argument("--csv", default="output/session_22_03.csv", help="Computed trajectory CSV")
    parser.add_argument("--nmea", default="data/gnss_log_2026_03_22_08_44_20.nmea", help="NMEA file (GNGGA/GPGGA)")
    parser.add_argument("--max-dt", type=float, default=2.0, help="Maximum time delta for matching (s)")
    args = parser.parse_args()

    compare(args.csv, args.nmea, max_dt_s=args.max_dt)


if __name__ == "__main__":
    main()
