# GNSS Positioning Pipeline — Intro to Navigation (Ex0)

Offline GPS positioning pipeline for Android RINEX 4 raw measurements.
Given a RINEX 4 observation file, it computes a **strict 1 Hz** trajectory
with 3D position, velocity and UTC time, and exports it as CSV and KML.

The solver is a faithful implementation of the Newton/Gauss-Newton method
described in *"Solving the GPS Equations"* (Arneja, Bender, Jugus, Reid),
with the standard IS-GPS-200 extensions (Kepler orbit propagation, Sagnac
correction, Klobuchar ionosphere, Saastamoinen troposphere).

## Requirements

- Python **3.11** or newer
- An internet connection is only required once, to download the broadcast
  ephemerides (BRDC) from the IGS mirrors via `gnss_lib_py`.

## Installation

```bash
git clone <your-repo-url> ex0
cd ex0

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The pipeline needs two kinds of inputs:

1. **Android RINEX 4 observation files** (`.26o`) produced by Google's
   *GNSS Logger*. They live in `data/`. The code ships already configured
   for three sessions recorded on 2026-03-21 and 2026-03-22 (see
   `SESSIONS` in `src/main.py`).

2. **Broadcast navigation files** (BRDC, `.rnx`) covering the session
   dates. They live in `data/brdc/rinex/nav/` and can be re-fetched by
   the script below.

NMEA (`.nmea`) files are optional — they are only used by `src/validate.py`
to compare the computed trajectory against the phone's own solution.
They are **not** used by the positioning algorithm.

**What's in the repo already.** For convenience, the three session
observation files, the matching BRDC ephemerides and the reference
outputs (`output/*.csv`, `output/*.kml`) are all committed to the
repository. You can open them straight after cloning, no download or
computation required. The instructions below are for regenerating them
or for running the pipeline on your own RINEX captures.

## How to run

### 1. (Optional) Refresh the broadcast ephemerides

The BRDC files for the three shipped sessions are already in
`data/brdc/rinex/nav/`. If you delete them, or want to run the pipeline
on a different date, regenerate them with:

```bash
python src/download_brdc.py
```

This fetches the BRDC files (≈ 17 MB per day) from the IGS mirrors and
stores them in `data/brdc/rinex/nav/`. Requires an internet connection
and the `gnss_lib_py` package (listed in `requirements.txt`).

### 2. Run the positioning pipeline

The reference CSV/KML outputs are already in `output/`. To regenerate
them (or to run against your own RINEX files):

```bash
python src/main.py --session phase1    # short session (~2 min, 120 epochs)
python src/main.py --session phase2    # medium session (~5 min, 280 epochs)
python src/main.py --session phase3    # long session (~43 min, 2590 epochs)
python src/main.py --session all       # run all three sequentially
```

Outputs are written to `output/`:

| File                        | Contents                                                              |
|-----------------------------|-----------------------------------------------------------------------|
| `session_XX.csv`            | Strict 1 Hz trajectory: UTC time, lat, lon, alt, vx, vy, vz, speed, n_sat |
| `session_XX.kml`            | Trajectory for Google Earth: red LineString + ~30 timestamped placemarks |

Rows with no valid fix are kept in the CSV with empty lat/lon/alt/velocity
fields and `n_sat = 0`, so the 1 Hz cadence is preserved. The KML ignores
these rows.

### 3. (Optional) Validate against the phone's NMEA solution

```bash
python src/validate.py \
    --csv  output/session_22_03.csv \
    --nmea data/gnss_log_2026_03_22_08_44_20.nmea
```

Prints horizontal / vertical / 3D error statistics (median, RMS, max) and
the median time delta between the two streams.


## Project layout

```
.
├── README.md                 # this file
├── requirements.txt
├── src/
│   ├── main.py               # pipeline entry point
│   ├── parser.py             # RINEX 3/4 observation parser (xarray output)
│   ├── nav_loader.py         # RINEX 3 navigation parser (GPS only)
│   ├── sat_position.py       # Kepler propagation + Sagnac + relativistic clock
│   ├── atmosphere.py         # Klobuchar ionosphere + Saastamoinen troposphere
│   ├── coordinates.py        # WGS-84 ECEF ↔ geodetic conversions
│   ├── solver.py             # Newton / Gauss-Newton GPS solver
│   ├── export.py             # CSV and KML writers
│   ├── validate.py           # optional comparison against NMEA GGA
│   └── download_brdc.py      # one-time BRDC downloader
├── data/                     # RINEX observations + BRDC (committed, regenerable)
└── output/                   # Generated CSV + KML (committed, regenerable)
```

## Algorithm (one-paragraph summary)

For each 1 Hz epoch of the RINEX file: (a) read the C1C pseudoranges, S1C
(SNR) and D1C (Doppler) for each visible GPS satellite; (b) reject
measurements with SNR < 25 dB-Hz or |Doppler| < 200 Hz; (c) propagate each
satellite's position from its broadcast ephemeris using the IS-GPS-200
Kepler equations (with Sagnac and relativistic clock corrections);
(d) solve the four GPS equations *F_i = (x − A_i)² + (y − B_i)² + (z − C_i)²
− (c·(t_i − d))² = 0* with multivariate Newton, starting from the PDF seed
*(0, 0, 6 370 000 m, 0 s)* — the system is square for N = 4 and
over-determined (Gauss-Newton normal equations) for N > 4; (e) apply
Klobuchar + Saastamoinen corrections from the rough fix and re-solve
(two-pass scheme); (f) drop outlier satellites whose post-fit residual
exceeds 500 m and re-solve (up to 3 rejections). Receiver velocity is
estimated by finite differencing the ECEF positions of consecutive epochs.

## Credits

- Solver faithful to *"Solving the GPS Equations"* — Harnam Arneja, Andrew
  Bender, Sam Jugus, Tim Reid.
- Orbit propagation and time-system conventions follow **IS-GPS-200**.
- Ionospheric model: **Klobuchar** (IS-GPS-200 §20.3.3.5.2.5).
- Tropospheric model: **Saastamoinen** (1972).
- BRDC fetching: [`gnss_lib_py`](https://github.com/Stanford-NavLab/gnss_lib_py).
