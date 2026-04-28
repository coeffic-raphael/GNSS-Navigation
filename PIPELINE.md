# GNSS Pipeline Explanation

This document explains how the project transforms Android RINEX raw
measurements into a 1 Hz trajectory with UTC time, 3D position and velocity.

## 1. Goal

The goal is to compute an offline GNSS path from a RINEX 4 observation file.
For each measurement time, called an epoch, the pipeline computes:

- UTC time
- 3D position: latitude, longitude, altitude
- velocity: `vx`, `vy`, `vz`, and speed
- number of satellites used

The final outputs are:

- `output/*.csv`: numerical 1 Hz trajectory
- `output/*.kml`: trajectory for Google Earth

The NMEA files are not used to compute the position. They are only used by
`src/validate.py` to compare the computed path with the phone's own solution.

## 2. Input Files

The pipeline uses two kinds of RINEX files.

### Observation RINEX

Example:

```text
data/gnss_log_2026_03_21_17_14_34 (1).26o
```

This file comes from the Android GNSS Logger. It contains the raw measurements
recorded by the phone.

Important observables:

```text
C1C = pseudorange
D1C = Doppler
S1C = signal strength / SNR
```

Meaning:

- `C1C` gives the apparent distance between the phone and the satellite.
- `D1C` gives the Doppler shift, used to estimate receiver velocity.
- `S1C` gives the signal quality, used to reject weak measurements.

### Navigation RINEX / BRDC

Examples:

```text
data/brdc/rinex/nav/BRDM00DLR_S_20260800000_01D_MN.rnx
data/brdc/rinex/nav/BRDM00DLR_S_20260810000_01D_MN.rnx
```

These files contain the satellite ephemerides.

Ephemerides are the broadcast orbital and clock parameters of the satellites.
They allow the code to compute where each GPS satellite was at a given time.

Without ephemerides, we would know the measured distances to satellites, but
not where the satellites were. So we could not solve the receiver position.

## 3. Main Entry Point

The main command is:

```bash
python src/main.py --session phase1
```

Available sessions are defined in `src/main.py`:

```python
SESSIONS = {
    "phase1": (
        "gnss_log_2026_03_21_17_14_34 (1).26o",
        "gnss_log_2026_03_21_17_14_34 (1).nmea",
        "session_21_03_short",
    ),
    ...
}
```

For `phase1`, the input observation file is:

```text
data/gnss_log_2026_03_21_17_14_34 (1).26o
```

The outputs are:

```text
output/session_21_03_short.csv
output/session_21_03_short.kml
```

## 4. Full Pipeline

The complete flow is:

```text
RINEX observation file
        |
        v
parser.py
        |
        v
measurements per epoch: C1C, D1C, S1C
        |
        v
sat_position.py + BRDC ephemerides
        |
        v
satellite positions and velocities
        |
        v
solver.py
        |
        v
receiver ECEF position + receiver clock bias
        |
        v
atmosphere.py
        |
        v
corrected pseudoranges
        |
        v
solver.py again
        |
        v
final receiver ECEF position
        |
        v
coordinates.py
        |
        v
latitude, longitude, altitude
        |
        v
velocity.py
        |
        v
receiver velocity from Doppler
        |
        v
export.py
        |
        v
CSV + KML
```

## 5. Step By Step

### Step 1: Load Navigation Files

In `src/main.py`, the navigation files are loaded with:

```python
nav_files = sorted(nav_dir.glob("*.rnx"))
nav = load_nav_files(nav_files)
```

The function `load_nav_files()` is in `src/nav_loader.py`.

It parses the BRDC navigation files and returns a dictionary like:

```python
nav["G03"] = [ephemeris_record_1, ephemeris_record_2, ...]
nav["G08"] = [ephemeris_record_1, ephemeris_record_2, ...]
```

Only GPS satellites are parsed. Their IDs start with `G`, for example `G03`.

### Step 2: Load Observation File

In `run_session()`:

```python
obs = load_observations(obs_path, use=["G"])
```

The function `load_observations()` is in `src/parser.py`.

It reads the Android RINEX `.26o` file and returns an `xarray.Dataset` with:

```text
time x sv
```

where:

- `time` is the list of epochs
- `sv` is the list of satellites

An epoch is one measurement time. For example:

```text
2026-03-21 15:14:41.418
```

At each epoch, the dataset contains the measurements for all visible
satellites.

### Step 3: Loop Over Epochs

The main loop is in `src/main.py`:

```python
for epoch_idx, t in enumerate(obs.time.values):
    epoch_obs = obs.sel(time=t)
```

For each epoch, the code tries to compute one receiver position and velocity.

### Step 4: Read Measurements For Each Satellite

For every visible satellite, the code reads:

```python
rho     = C1C
snr     = S1C
doppler = D1C
```

In code:

```python
rho = float(epoch_obs["C1C"].sel(sv=sv_val).values)
snr = float(epoch_obs["S1C"].sel(sv=sv_val).values)
doppler = float(epoch_obs["D1C"].sel(sv=sv_val).values)
```

Here:

- `rho` is the pseudorange
- `snr` is the signal quality
- `doppler` is used for velocity

The code rejects invalid or weak measurements:

```python
if not np.isfinite(rho) or rho <= 0:
    continue

if np.isfinite(snr) and snr < MIN_SNR_DB:
    continue

if np.isfinite(doppler) and abs(doppler) < MIN_DOPPLER_HZ:
    continue
```

The constants are:

```python
MIN_SNR_DB = 25.0
MIN_DOPPLER_HZ = 200.0
```

### Step 5: Compute Satellite Position

For each valid satellite, the code calls:

```python
state = compute_sat_state(nav, sv, t, rho, x_rx)
```

The function is in `src/sat_position.py`.

It computes:

```python
pos_sat, dt_sat, vel_sat
```

Meaning:

- `pos_sat`: satellite position in ECEF coordinates, in meters
- `dt_sat`: satellite clock correction, in seconds
- `vel_sat`: satellite velocity in ECEF coordinates, in m/s

The satellite position is computed from the ephemerides using Kepler orbit
propagation and GNSS corrections such as satellite clock correction,
relativistic correction and Sagnac correction.

The result is stored for the position solver:

```python
pos_data.append((pos_sat, dt_sat, rho))
```

And for the velocity solver:

```python
vel_data.append((pos_sat, vel_sat, doppler))
```

### Step 6: Solve Receiver Position

The receiver position has four unknowns:

```text
x, y, z, d
```

where:

- `x, y, z` are the receiver ECEF coordinates
- `d` is the receiver clock bias

The receiver clock bias is necessary because a phone does not have a perfect
GPS clock. A very small timing error creates a large distance error because
GPS signals travel at the speed of light.

The code requires at least 5 satellites:

```python
MIN_SAT = 5
```

Four satellites are the mathematical minimum because there are four unknowns.
Five satellites give redundancy and make outlier rejection possible.

The first position estimate is computed with:

```python
pos_rough, _ = _solve_with_outlier_rejection(pos_data, None)
```

This function calls `solve_position()` in `src/solver.py`.

The equation used for each satellite is:

```text
F_i = (x - A_i)^2 + (y - B_i)^2 + (z - C_i)^2 - (c(t_i - d))^2
```

where:

- `A_i, B_i, C_i` are the satellite coordinates
- `x, y, z` are the receiver coordinates
- `c` is the speed of light
- `t_i` is related to the measured pseudorange
- `d` is the receiver clock bias

The system is non-linear, so `solve_position()` cannot solve it directly with
one simple formula. Instead, it uses Newton / Gauss-Newton iterations.

The idea of the iteration is:

```text
start with an approximate receiver position
measure how wrong this position is
compute a correction
update the position
repeat until the correction becomes very small
```

In `src/solver.py`, the current estimate is called `g`:

```python
g = np.array([0.0, 0.0, 6_370_000.0, 0.0])
```

This is the initial guess:

```text
x = 0
y = 0
z = 6 370 000 m
d = 0
```

It is not the real receiver position. It is only a reasonable starting point:
roughly one Earth radius away from the center of the Earth.

For each iteration, the solver does four things.

#### Iteration Step 1: Compute The Current Error

Using the current guess `g = [x, y, z, d]`, the solver evaluates the GPS
equation for every satellite:

```python
F = (x - A) ** 2 + (y - B) ** 2 + (z - Cv) ** 2 - ct ** 2
```

If the current position were perfect, every value in `F` would be close to
zero.

If `F` is large, it means:

```text
the current receiver position does not explain the measured pseudoranges yet
```

So `F` is the vector of residual errors for the current guess.

#### Iteration Step 2: Build The Jacobian

The solver then builds the Jacobian matrix `D`:

```python
D[:, 0] = 2.0 * (x - A)
D[:, 1] = 2.0 * (y - B)
D[:, 2] = 2.0 * (z - Cv)
D[:, 3] = 2.0 * C ** 2 * (t - d)
```

The Jacobian says how the errors `F` change if we slightly change:

```text
x, y, z, d
```

In other words, it tells the solver:

```text
if I move the receiver estimate a little in x, y, z,
how much do the pseudorange errors improve or worsen?
```

This is what makes Newton's method work: it locally approximates the
non-linear equations by a linear system.

#### Iteration Step 3: Solve For The Correction

The solver looks for a correction:

```text
delta = [delta_x, delta_y, delta_z, delta_d]
```

If there are exactly 4 satellites, the system is square:

```python
delta = np.linalg.solve(D, -F)
```

If there are more than 4 satellites, the system is over-determined. There are
more equations than unknowns, so the code uses Gauss-Newton least squares:

```python
DTD = D.T @ D
DTF = D.T @ F
delta = np.linalg.solve(DTD, -DTF)
```

This finds the correction that best reduces the residuals across all
satellites.

#### Iteration Step 4: Update The Estimate

The current estimate is updated:

```python
g = g + delta
```

So if the solver started with a rough position, after one iteration it is much
closer to the real receiver position. After two or three iterations, it is
usually close enough.

The loop stops when the spatial correction is tiny:

```python
if np.linalg.norm(delta[:3]) < tol:
    break
```

In this project:

```python
tol = 1e-4
```

So the iteration stops when the update in `x, y, z` is below `0.0001 m`.

Conceptually, the position solve looks like this:

```text
iteration 0:
    guess is rough
    residuals are large

iteration 1:
    solver applies a large correction
    guess becomes close to the receiver

iteration 2:
    solver applies a small correction
    residuals become very small

iteration 3:
    correction is tiny
    stop
```

So the receiver position is not found in one shot. It is found progressively:
each iteration uses the current error to compute a better estimate of
`x, y, z` and the receiver clock bias `d`.

### Step 7: Apply Atmospheric Corrections

The first position is used to estimate atmospheric delays:

```python
pos_data_corr = _apply_atmosphere(pos_data, pos_rough, klob, gps_tow)
```

This uses functions from `src/atmosphere.py`:

```python
klobuchar_delay()
saastamoinen_delay()
```

The models are:

- Klobuchar for ionospheric delay
- Saastamoinen for tropospheric delay

The corrected pseudorange is:

```python
rho_corr = rho - iono_m - tropo_m
```

Then the position is solved again:

```python
pos, n_used = _solve_with_outlier_rejection(pos_data_corr, None)
```

This gives the final receiver position in ECEF coordinates.

### Step 8: Reject Outliers

The function `_solve_with_outlier_rejection()` solves the position, computes
the residuals, and removes the worst satellite if its residual is too large.

The threshold is:

```python
OUTLIER_THRESH_M = 500.0
```

The code removes at most:

```python
MAX_OUTLIER_REJECT = 3
```

This helps when one satellite measurement is affected by multipath or another
large error.

### Step 9: Convert ECEF To Latitude, Longitude, Altitude

The solver returns ECEF coordinates:

```text
x, y, z
```

These are useful for GNSS equations, but not ideal for maps.

The code converts them with:

```python
lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
```

The function `ecef_to_lla()` is in `src/coordinates.py`.

The exported 3D position is:

```text
latitude, longitude, altitude
```

This is the format used in the CSV and KML.

### Step 10: Compute Velocity From Doppler

The receiver velocity is computed with:

```python
vel_sol = solve_velocity(vel_data, pos)
```

The function is in `src/velocity.py`.

The key Doppler relation is:

```text
rho_dot = -lambda_L1 * D1C
```

where:

- `rho_dot` is the pseudorange rate, in m/s
- `lambda_L1` is the GPS L1 wavelength
- `D1C` is the Doppler measurement, in Hz

The velocity solver estimates:

```text
vx, vy, vz, clock drift
```

The exported values are:

```text
vx_mps, vy_mps, vz_mps, speed_mps
```

The speed is:

```python
speed = sqrt(vx * vx + vy * vy + vz * vz)
```

Using Doppler is better than finite-differencing positions because Doppler is
an instantaneous measurement and does not depend on the previous epoch.

### Step 11: Build One Output Record

For each valid epoch, `src/main.py` appends:

```python
records.append({
    "utc_time": _gps_to_utc(t),
    "lat_deg": lat,
    "lon_deg": lon,
    "alt_m": alt,
    "vx_mps": float(vx),
    "vy_mps": float(vy),
    "vz_mps": float(vz),
    "n_sat": n_used,
})
```

This is one trajectory point.

Example output row:

```csv
2026-03-21T15:14:41.418Z,32.168450583149365,34.81323454021795,33.590019593946636,-0.6960910031639498,-0.9016214540783412,-0.5339953050681365,1.2580202371086677,5
```

### Step 12: Keep Strict 1 Hz Output

Some epochs may fail because there are not enough valid satellites or because
the solver rejects the solution.

To keep a strict 1 Hz CSV, the code calls:

```python
records_1hz = _fill_gaps_with_nan(records)
```

If a second has no valid fix, the CSV still contains a row, but with `NaN`
position and velocity fields and `n_sat = 0`.

This preserves the 1 Hz timeline without inventing fake positions.

### Step 13: Export CSV And KML

The final export is:

```python
write_csv(records_1hz, out_dir / f"{out_prefix}.csv")
write_kml(records_1hz, out_dir / f"{out_prefix}.kml")
```

These functions are in `src/export.py`.

The CSV contains the complete numerical trajectory:

```text
utc_time, lat_deg, lon_deg, alt_m, vx_mps, vy_mps, vz_mps, speed_mps, n_sat
```

The KML contains the trajectory as longitude, latitude, altitude coordinates
for Google Earth.

## 6. Concrete Example

For one epoch, suppose the phone sees five GPS satellites:

```text
G03, G08, G14, G22, G27
```

The observation file provides:

```text
G03: C1C = 21400000 m, D1C = -1200 Hz, S1C = 35
G08: C1C = 23100000 m, D1C =   900 Hz, S1C = 38
G14: C1C = 20800000 m, D1C =  -600 Hz, S1C = 30
G22: C1C = 24000000 m, D1C =   700 Hz, S1C = 32
G27: C1C = 22500000 m, D1C = -1000 Hz, S1C = 36
```

The pipeline does:

1. `parser.py` reads these measurements.
2. `main.py` keeps the satellites because their SNR and Doppler are valid.
3. `sat_position.py` computes each satellite position from the BRDC files.
4. `solver.py` solves the receiver `x, y, z` and clock bias.
5. `atmosphere.py` corrects the pseudoranges.
6. `solver.py` computes the final receiver position.
7. `coordinates.py` converts ECEF to latitude, longitude, altitude.
8. `velocity.py` computes receiver velocity from Doppler.
9. `export.py` writes the CSV and KML records.

The final CSV row contains:

```text
UTC time
latitude
longitude
altitude
vx
vy
vz
speed
number of satellites
```

## 7. Short Presentation Version

For each RINEX epoch, the code reads the raw GPS observables: `C1C`
pseudorange, `D1C` Doppler and `S1C` signal strength. The BRDC navigation
files provide the satellite ephemerides, which are used to compute satellite
positions. Then the solver estimates the receiver ECEF position and clock bias
using Newton / Gauss-Newton. After atmospheric corrections, the position is
converted to latitude, longitude and altitude. Finally, Doppler measurements
are used to estimate velocity, and the result is exported as a strict 1 Hz CSV
and a KML trajectory.
