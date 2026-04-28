"""Export computed trajectories to CSV and KML.

write_csv : writes a CSV file with position, velocity, satellite count.
write_kml : writes a KML file (Google Earth) for the 3D trajectory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import simplekml


def write_csv(records: list[dict], path: str | Path) -> None:
    """Write trajectory to CSV.

    Columns: utc_time, lat_deg, lon_deg, alt_m, vx_mps, vy_mps, vz_mps,
             speed_mps, n_sat

    Rows with NaN position/velocity are preserved (they document epochs
    where the solver could not produce a fix) — speed_mps is NaN in that
    case, and n_sat is 0.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    # Speed propagates NaN naturally when any velocity component is NaN
    df["speed_mps"] = np.sqrt(
        df["vx_mps"] ** 2 + df["vy_mps"] ** 2 + df["vz_mps"] ** 2
    )
    cols = ["utc_time", "lat_deg", "lon_deg", "alt_m",
            "vx_mps", "vy_mps", "vz_mps", "speed_mps", "n_sat"]
    df[cols].to_csv(path, index=False)
    n_valid = int(df["lat_deg"].notna().sum())
    print(f"CSV written: {path}  ({len(df)} rows, {n_valid} valid fixes)")


def write_kml(records: list[dict], path: str | Path) -> None:
    """Write trajectory to KML (Google Earth, absolute altitude).

    Creates:
    - a red LineString for the full trajectory
    - periodic placemarks with speed and satellite count
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kml = simplekml.Kml()

    # ── Full trajectory ──────────────────────────────────────────────────────
    coords = [
        (r["lon_deg"], r["lat_deg"], r["alt_m"])
        for r in records
        if np.isfinite(r["lat_deg"]) and np.isfinite(r["lon_deg"])
    ]
    if coords:
        ls = kml.newlinestring(name="GPS Trajectory", coords=coords)
        ls.altitudemode = simplekml.AltitudeMode.absolute
        ls.style.linestyle.color = simplekml.Color.red
        ls.style.linestyle.width = 3

    # ── Spaced placemarks with UTC + speed visible on map ────────────────────
    # One placemark every STRIDE points. Label = "HH:MM:SS • v m/s" for quick
    # readability. TimeStamp enables chronological playback in Google Earth.
    STRIDE = max(5, len(records) // 30)  # target ~30 placemarks, at least every 5 s

    folder = kml.newfolder(name=f"Points (1 every {STRIDE}s)")
    for i, r in enumerate(records):
        if i % STRIDE != 0 and i != len(records) - 1:
            continue
        if not (np.isfinite(r["lat_deg"]) and np.isfinite(r["lon_deg"])):
            continue
        speed = float(np.sqrt(
            r.get("vx_mps", 0) ** 2
            + r.get("vy_mps", 0) ** 2
            + r.get("vz_mps", 0) ** 2
        ))
        # Extract "HH:MM:SS" from "2026-03-21T17:15:30.416Z"
        utc_short = str(r["utc_time"])[11:19]
        pt = folder.newpoint(
            name=f"{utc_short} • {speed:.1f} m/s",
            coords=[(r["lon_deg"], r["lat_deg"], r["alt_m"])],
        )
        pt.altitudemode = simplekml.AltitudeMode.absolute
        pt.timestamp.when = str(r["utc_time"])
        pt.description = (
            f"UTC: {str(r['utc_time'])}\n"
            f"Position: {r['lat_deg']:.6f}°, {r['lon_deg']:.6f}°, "
            f"{r['alt_m']:.1f} m\n"
            f"Speed: {speed:.2f} m/s ({speed*3.6:.1f} km/h)\n"
            f"  vx = {r['vx_mps']:+.2f} m/s\n"
            f"  vy = {r['vy_mps']:+.2f} m/s\n"
            f"  vz = {r['vz_mps']:+.2f} m/s\n"
            f"Satellites: {r['n_sat']}"
        )

    kml.save(str(path))
    print(f"KML written: {path}  ({len(coords)} points)")
