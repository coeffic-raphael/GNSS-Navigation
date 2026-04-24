"""RINEX 3 navigation parser for GPS broadcast ephemerides.

Implemented manually because georinex 1.16.2 has issues on some multi-GNSS
DLR BRDC files (Galileo fields become NaN). This pipeline uses GPS only, so
only constellation 'G' is parsed here.

RINEX 3 GPS navigation format: 8 lines per record (Table A2 / IS-GPS-200).

Returns: dict {sv: [eph_record, ...]} where eph_record contains:
    sv,              # satellite identifier, e.g. 'G01'
    toe_abs,         # absolute GPS seconds (gps_week * 604800 + toe_sow)
    toc_abs,         # absolute GPS seconds of toc (= clock epoch)
    # orbital parameters
    sqrtA, e, M0, DeltaN, Omega0, OmegaDot, Io, IDOT, omega,
    Crc, Crs, Cuc, Cus, Cic, Cis, Toe,
    # satellite clock
    af0, af1, af2,
    TGD,             # inter-frequency group delay for L1 [s]
)

Special key '_klobuchar' stores (alpha, beta) parsed from header.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]

# GPS epoch
_GPS_EPOCH = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)


def _rinex_float(s: str) -> float:
    """Convert RINEX float ('D' or 'E' notation); NaN if empty/invalid."""
    s = s.strip().replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _f4(line: str) -> tuple[float, float, float, float]:
    """Extract 4 floats (columns 4-23, 23-42, 42-61, 61-80)."""
    return (_rinex_float(line[4:23]),
            _rinex_float(line[23:42]),
            _rinex_float(line[42:61]),
            _rinex_float(line[61:80]))


def _toc_to_gps_s(y: int, mo: int, d: int, h: int, mi: int, se: int) -> float:
    """Convert toc date to absolute GPS seconds (RINEX nav toc is GPS time)."""
    dt = datetime.datetime(y, mo, d, h, mi, se, tzinfo=datetime.timezone.utc)
    return (dt - _GPS_EPOCH).total_seconds()


def _parse_gps_record(lines: list[str], i: int) -> tuple[dict, int]:
    """Parse 8 RINEX-3 GPS nav lines from i. Return (record, i_next).

    Lines (indexed 0-7):
      0 : SV YYYY MM DD HH MM SS  af0 af1 af2
      1 :      IODE  Crs  DeltaN  M0
      2 :      Cuc  e  Cus  sqrtA
      3 :      Toe  Cic  Omega0  Cis
      4 :      Io  Crc  omega  OmegaDot
      5 :      IDOT  CodesL2  GPSWeek  L2Pflag
      6 :      SVacc  health  TGD  IODC
      7 :      TransTime  FitIntvl  spare  spare
    """
    l0 = lines[i]
    sv = l0[0:3].strip()
    y  = int(l0[4:8])
    mo = int(l0[9:11])
    d  = int(l0[12:14])
    h  = int(l0[15:17])
    mi = int(l0[18:20])
    se = int(l0[21:23])
    af0 = _rinex_float(l0[23:42])
    af1 = _rinex_float(l0[42:61])
    af2 = _rinex_float(l0[61:80])

    _, Crs, DeltaN, M0              = _f4(lines[i + 1])
    Cuc, e, Cus, sqrtA              = _f4(lines[i + 2])
    Toe, Cic, Omega0, Cis           = _f4(lines[i + 3])
    Io,  Crc, omega, OmegaDot       = _f4(lines[i + 4])
    IDOT, _, Week, _                = _f4(lines[i + 5])
    _, _, TGD, _                    = _f4(lines[i + 6])

    toc_abs = _toc_to_gps_s(y, mo, d, h, mi, se)
    toe_abs = Week * 604_800.0 + Toe

    return {
        "sv":        sv,
        "toe_abs":   toe_abs,
        "toc_abs":   toc_abs,
        "Toe":       Toe,          # TOW in seconds (0..604800)
        "sqrtA":     sqrtA,
        "e":         e,
        "M0":        M0,
        "DeltaN":    DeltaN,
        "Omega0":    Omega0,
        "OmegaDot":  OmegaDot,
        "Io":        Io,
        "IDOT":      IDOT,
        "omega":     omega,
        "Crc": Crc, "Crs": Crs,
        "Cuc": Cuc, "Cus": Cus,
        "Cic": Cic, "Cis": Cis,
        "af0": af0, "af1": af1, "af2": af2,
        "TGD":       TGD,
    }, i + 8


def load_nav_files(paths: Iterable[PathLike]) -> dict[str, list[dict]]:
    """Parse one or more RINEX 3 nav files and return {sv: [records]}.

    Only GPS records (prefix 'G') are parsed; other constellations are ignored.

    Special key '_klobuchar' stores (alpha, beta) read from the first header
    that provides GPSA/GPSB IONOSPHERIC CORR terms.
    """
    out: dict[str, list[dict]] = {}
    klobuchar_alpha = None
    klobuchar_beta  = None

    for path in paths:
        path = Path(path)
        lines = path.read_text(errors="replace").splitlines()

        # Skip header and extract Klobuchar coefficients
        start = 0
        for i, l in enumerate(lines):
            if "IONOSPHERIC CORR" in l:
                key = l[0:4].strip()
                vals = [_rinex_float(l[5:17]), _rinex_float(l[17:29]),
                        _rinex_float(l[29:41]), _rinex_float(l[41:53])]
                if key == "GPSA" and klobuchar_alpha is None:
                    klobuchar_alpha = vals
                elif key == "GPSB" and klobuchar_beta is None:
                    klobuchar_beta = vals
            if "END OF HEADER" in l:
                start = i + 1
                break

        i = start
        n = len(lines)
        while i < n:
            l = lines[i]
            if not l or len(l) < 23:
                i += 1
                continue
            # Keep GPS records only. For others, skip to next record header.
            if l[0] != "G":
                i += 1
                while i < n and not (lines[i] and lines[i][0].isalpha()
                                     and lines[i][1:3].strip().isdigit()):
                    i += 1
                continue
            try:
                rec, i = _parse_gps_record(lines, i)
                out.setdefault(rec["sv"], []).append(rec)
            except (ValueError, IndexError):
                i += 1

    # Sort records by increasing toe_abs for each satellite
    for sv in out:
        out[sv].sort(key=lambda r: r["toe_abs"])

    # Store Klobuchar parameters (alpha, beta) under a special key
    if klobuchar_alpha is not None and klobuchar_beta is not None:
        out["_klobuchar"] = (klobuchar_alpha, klobuchar_beta)  # type: ignore
    return out


def find_ephemeris(nav: dict[str, list[dict]], sv: str, t_tx_s: float,
                   max_age_s: float = 7200.0) -> dict | None:
    """Return ephemeris record nearest to t_tx_s for `sv`.

    Return None if satellite is missing or if nearest record is older than max_age_s.
    """
    recs = nav.get(sv)
    if not recs:
        return None
    best = min(recs, key=lambda r: abs(t_tx_s - r["toe_abs"]))
    if abs(t_tx_s - best["toe_abs"]) > max_age_s:
        return None
    return best
