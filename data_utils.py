from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "date", "duration_min", "distance_km", "pace_min_km", "avg_hr_bpm",
    "elevation_m", "power_w", "cadence_spm", "rpe"
]

ALIASES = {
    "date": ["date", "data", "datetime", "start_date", "activity_date"],
    "duration_min": ["duration_min", "durata", "duration", "elapsed_time", "moving_time", "time"],
    "distance_km": ["distance_km", "distanza", "distance", "km"],
    "pace_min_km": ["pace_min_km", "passo", "pace", "avg_pace", "average_pace"],
    "avg_hr_bpm": ["avg_hr_bpm", "fc_media", "fc", "heart_rate", "avg_hr", "average_heartrate", "average_heart_rate"],
    "elevation_m": ["elevation_m", "dislivello", "elevation", "elevation_gain", "total_elevation_gain", "ascent"],
    "power_w": ["power_w", "potenza", "power", "avg_power", "average_watts"],
    "cadence_spm": ["cadence_spm", "cadenza", "cadence", "avg_cadence", "average_cadence"],
    "rpe": ["rpe", "perceived_exertion", "sforzo_percepito"],
}


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def parse_pace(value) -> float:
    if pd.isna(value) or value == "":
        return np.nan
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        return x if 2 <= x <= 20 else np.nan
    s = str(value).strip().lower().replace("min/km", "").replace("/km", "")
    m = re.match(r"^(\d{1,2})[:'](\d{1,2})(?:\")?$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60
    try:
        x = float(s.replace(",", "."))
        return x if 2 <= x <= 20 else np.nan
    except ValueError:
        return np.nan


def parse_duration_min(value) -> float:
    if pd.isna(value) or value == "":
        return np.nan
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        # Heuristic: Strava exports seconds; explicit canonical duration_min is handled earlier.
        return x / 60 if x > 300 else x
    s = str(value).strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = map(float, parts)
            return 60*h + m + sec/60
        if len(parts) == 2:
            m, sec = map(float, parts)
            return m + sec/60
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def _coerce_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")


def canonicalize_dataframe(df: pd.DataFrame, derive_pace: bool = False) -> pd.DataFrame:
    src = df.copy()
    src.columns = [_norm_col(c) for c in src.columns]
    out = pd.DataFrame(index=src.index)
    chosen = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            a = _norm_col(alias)
            if a in src.columns:
                chosen[canonical] = a
                out[canonical] = src[a]
                break
        if canonical not in out:
            out[canonical] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True, format="mixed")
    # Duration: canonical numeric duration_min means minutes; colon strings are parsed as time.
    if chosen.get("duration_min") == "duration_min":
        raw_duration = out["duration_min"].copy()
        colon_mask = raw_duration.astype(str).str.contains(":", regex=False, na=False)
        parsed = _coerce_number(raw_duration).astype(float)
        parsed.loc[colon_mask] = raw_duration.loc[colon_mask].map(parse_duration_min)
        out["duration_min"] = parsed
    else:
        out["duration_min"] = out["duration_min"].map(parse_duration_min)
    out["pace_min_km"] = out["pace_min_km"].map(parse_pace)
    for c in ["distance_km", "avg_hr_bpm", "elevation_m", "power_w", "cadence_spm", "rpe"]:
        out[c] = _coerce_number(out[c])

    # Heuristic for meter-based distances from exports.
    med_dist = out["distance_km"].median(skipna=True)
    if pd.notna(med_dist) and med_dist > 200:
        out["distance_km"] = out["distance_km"] / 1000

    # Keep extracted/imported fields missing by default. Optional deterministic derivation is explicit.
    if derive_pace:
        missing_pace = out["pace_min_km"].isna() & out["duration_min"].notna() & out["distance_km"].gt(0)
        out.loc[missing_pace, "pace_min_km"] = out.loc[missing_pace, "duration_min"] / out.loc[missing_pace, "distance_km"]

    return out[CANONICAL_COLUMNS].sort_values("date", na_position="last").reset_index(drop=True)


def read_uploaded_csv(file_bytes: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    # Try separators commonly found in European exports.
    last_error = None
    for sep in [None, ",", ";", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python")
            if df.shape[1] > 1 or sep is None:
                return canonicalize_dataframe(df)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Impossibile leggere {filename}: {last_error}")


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # Deterministic pace derivation is allowed only after dataset confirmation and is tracked.
    x["pace_derived"] = False
    can_derive = x["pace_min_km"].isna() & x["duration_min"].notna() & x["distance_km"].gt(0)
    x.loc[can_derive, "pace_min_km"] = x.loc[can_derive, "duration_min"] / x.loc[can_derive, "distance_km"]
    x.loc[can_derive, "pace_derived"] = True
    x = x.dropna(subset=["date", "distance_km", "pace_min_km", "avg_hr_bpm"])
    x = x[(x["distance_km"] > 0) & (x["pace_min_km"] > 0)]
    x = x.sort_values("date").reset_index(drop=True)
    if x.empty:
        return x
    x["speed_kmh"] = 60.0 / x["pace_min_km"]
    t0 = x["date"].min()
    x["t_days"] = (x["date"] - t0).dt.total_seconds() / 86400
    return x


def pace_str_from_speed(speed_kmh: float) -> str:
    if speed_kmh is None or not np.isfinite(speed_kmh) or speed_kmh <= 0:
        return "—"
    pace = 60.0 / speed_kmh
    mins = int(np.floor(pace))
    secs = int(round((pace - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{mins}:{secs:02d}/km"


def detect_duplicates(df: pd.DataFrame) -> pd.Series:
    keys = ["date", "distance_km", "duration_min"]
    temp = df.copy()
    for c in keys:
        if c not in temp:
            temp[c] = np.nan
    return temp.duplicated(subset=keys, keep=False)
