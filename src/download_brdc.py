"""Download broadcast ephemerides (BRDC) for recording dates.

Uses gnss_lib_py.utils.ephemeris_downloader.load_ephemeris (v1.0.4) to query
public IGS mirrors and download RINEX navigation files (BRDC).

Covered sessions:
- 21/03/2026 15:14 UTC (DoY 080)
- 21/03/2026 15:18 UTC (DoY 080)
- 22/03/2026 06:44 UTC (DoY 081)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from gnss_lib_py.utils.ephemeris_downloader import load_ephemeris
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis


OUT = Path(__file__).resolve().parent.parent / "data" / "brdc"
OUT.mkdir(parents=True, exist_ok=True)

# One timestamp per day (each BRDC file covers 24 hours)
timestamps = [
    dt.datetime(2026, 3, 21, 15, 16, tzinfo=dt.timezone.utc),  # DoY 080
    dt.datetime(2026, 3, 22,  6, 44, tzinfo=dt.timezone.utc),  # DoY 081
]

# Convert to GPS milliseconds (expected by load_ephemeris)
gps_millis = np.array([datetime_to_gps_millis(ts) for ts in timestamps])

print(f"Downloading for {len(timestamps)} timestamps")
for ts in timestamps:
    print(f"   - {ts.isoformat()}")

files = load_ephemeris(
    file_type="rinex_nav",
    gps_millis=gps_millis,
    constellations=["gps"],
    download_directory=str(OUT),
    verbose=True,
)

print("\n=== Downloaded files ===")
for f in files:
    print(f"  {f}")

print("\n=== Contents of data/brdc/ ===")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        rel = p.relative_to(OUT)
        print(f"  {str(rel):60s} {p.stat().st_size:>10d} bytes")
