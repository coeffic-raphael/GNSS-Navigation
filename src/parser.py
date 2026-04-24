"""RINEX 3/4 observation parser (.26o).

Exposes `load_observations(path, use=None)` and returns an xarray.Dataset
(dimensions: time x sv) with measured variables (C1C, D1C, S1C, ...).

For navigation files, use `nav_loader.load_nav_files()`.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import xarray as xr


PathLike = Union[str, Path]


def _parse_rinex4_obs_header(lines: list[str]) -> tuple[dict[str, list[str]], int]:
    """Read RINEX observation header and return (obs_types_per_sys, end_line_idx).

    obs_types_per_sys : dict {'G': ['C1C','D1C','S1C',...], 'E': [...], ...}
    end_line_idx      : index of first data line
    """
    obs_types: dict[str, list[str]] = {}
    for i, line in enumerate(lines):
        label = line[60:].strip() if len(line) > 60 else ""
        if "SYS / # / OBS TYPES" in label:
            sys_char = line[0].strip()
            n_obs = int(line[3:6])
            types = line[7:60].split()[:n_obs]
            if sys_char:
                obs_types[sys_char] = types
            else:
                # Continuation line (rare but possible when >13 obs types)
                last_sys = list(obs_types.keys())[-1]
                obs_types[last_sys].extend(types)
        elif "END OF HEADER" in label:
            return obs_types, i + 1
    return obs_types, len(lines)


def load_observations(
    path: PathLike,
    use: Iterable[str] | None = None,
) -> xr.Dataset:
    """Load a RINEX 3/4 observation file (.26o).

    Parameters
    ----------
    path : file path
    use  : constellations to keep, e.g. ['G'] (None = all)

    Returns
    -------
    xarray.Dataset with dimensions (time, sv) and obs variables (C1C, D1C...)
    """
    path = Path(path)
    use_set = set(use) if use is not None else None

    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    obs_types, data_start = _parse_rinex4_obs_header(lines)

    # Accumulators: {obs_type: {sv: [value_per_epoch]}}
    epochs: list[np.datetime64] = []
    data: dict[str, dict[str, list[float]]] = {}
    sv_set: set[str] = set()

    i = data_start
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i]
        if not line.startswith(">"):
            i += 1
            continue

        # Epoch line: > YYYY MM DD HH MM ss.sssssss  flag  n_sv
        try:
            year  = int(line[2:6])
            month = int(line[7:9])
            day   = int(line[10:12])
            hour  = int(line[13:15])
            minu  = int(line[16:18])
            sec   = float(line[19:29])
            n_sv  = int(line[32:35])
        except (ValueError, IndexError):
            i += 1
            continue

        sec_int  = int(sec)
        microsec = int(round((sec - sec_int) * 1e6))
        t_dt = datetime.datetime(year, month, day, hour, minu, sec_int,
                                 microsec, tzinfo=datetime.timezone.utc)
        t_np = np.datetime64(t_dt.replace(tzinfo=None), "ns")

        epoch_data: dict[str, dict[str, float]] = {}

        for j in range(n_sv):
            sat_line = lines[i + 1 + j] if (i + 1 + j) < n_lines else ""
            if len(sat_line) < 3:
                continue
            sv = sat_line[:3].strip()
            if not sv:
                continue
            sys = sv[0]
            if use_set is not None and sys not in use_set:
                continue
            if sys not in obs_types:
                continue

            obs_list = obs_types[sys]
            sv_obs: dict[str, float] = {}
            for k, obs_name in enumerate(obs_list):
                start = 3 + k * 16
                end   = start + 14
                raw   = sat_line[start:end] if len(sat_line) >= end else ""
                try:
                    sv_obs[obs_name] = float(raw)
                except ValueError:
                    sv_obs[obs_name] = np.nan
            epoch_data[sv] = sv_obs
            sv_set.add(sv)

        epochs.append(t_np)
        epoch_idx = len(epochs) - 1

        all_obs: set[str] = set()
        for sv_obs in epoch_data.values():
            all_obs.update(sv_obs.keys())

        for obs_name in all_obs:
            if obs_name not in data:
                data[obs_name] = {}
            for sv_key in epoch_data:
                if sv_key not in data[obs_name]:
                    data[obs_name][sv_key] = [np.nan] * epoch_idx
                data[obs_name][sv_key].append(epoch_data[sv_key].get(obs_name, np.nan))

        # Pad satellites absent in this epoch
        for obs_name in data:
            for sv_key in data[obs_name]:
                if len(data[obs_name][sv_key]) < epoch_idx + 1:
                    data[obs_name][sv_key].append(np.nan)

        i += 1 + n_sv

    if not epochs or not sv_set:
        raise ValueError(f"No parsable observation data found in {path}")

    sv_list   = sorted(sv_set)
    times_arr = np.array(epochs)

    data_vars = {}
    for obs_name, sv_dict in data.items():
        arr = np.full((len(epochs), len(sv_list)), np.nan)
        for ci, sv_key in enumerate(sv_list):
            if sv_key in sv_dict:
                vals = sv_dict[sv_key]
                n = min(len(vals), len(epochs))
                arr[:n, ci] = vals[:n]
        data_vars[obs_name] = (["time", "sv"], arr)

    return xr.Dataset(data_vars, coords={"time": times_arr, "sv": sv_list})
