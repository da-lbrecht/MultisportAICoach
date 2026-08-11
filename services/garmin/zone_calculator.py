"""Derives concrete power/HR zone tables from live Garmin training thresholds.

Uses standard, published percentage bands (not reverse-engineered from any Garmin API):
- Power: Coggan's 7-zone %FTP model.
- Heart rate: a 5-zone %LTHR model (Friel-style), anchored on lactate threshold HR rather
  than max HR, consistent with how the power zones are threshold-anchored.
"""

# name, min_percent_of_ftp, max_percent_of_ftp (None = unbounded above)
POWER_ZONE_BANDS: dict[str, tuple[str, float, float | None]] = {
    "z1_recovery": ("Active Recovery", 0.0, 0.55),
    "z2_endurance": ("Endurance", 0.55, 0.75),
    "z3_tempo": ("Tempo (Sweet Spot)", 0.75, 0.90),
    "z4_threshold": ("Threshold", 0.90, 1.05),
    "z5_vo2max": ("VO2Max", 1.05, 1.20),
    "z6_anaerobic": ("Anaerobic", 1.20, 1.50),
    "z7_neuromuscular": ("Neuromuscular Power", 1.50, None),
}

# name, min_percent_of_lthr, max_percent_of_lthr (None = unbounded above)
HR_ZONE_BANDS: dict[str, tuple[str, float, float | None]] = {
    "z1_recovery": ("Recovery", 0.0, 0.85),
    "z2_aerobic": ("Aerobic Base", 0.85, 0.89),
    "z3_tempo": ("Tempo", 0.89, 0.94),
    "z4_threshold": ("Threshold", 0.94, 0.99),
    "z5_maximal": ("Maximal", 0.99, None),
}

_UNBOUNDED_PERCENT = 999
_UNBOUNDED_MULTIPLIER = 3.0  # Used only to compute a display "max" value for the top zone.


def compute_power_zones(ftp_watts: float) -> dict[str, dict]:
    return {
        key: {
            "name": name,
            "min_percent": round(lo * 100),
            "max_percent": round(hi * 100) if hi is not None else _UNBOUNDED_PERCENT,
            "min_watts": round(ftp_watts * lo),
            "max_watts": round(ftp_watts * (hi if hi is not None else _UNBOUNDED_MULTIPLIER)),
        }
        for key, (name, lo, hi) in POWER_ZONE_BANDS.items()
    }


def compute_hr_zones(lthr_bpm: float) -> dict[str, dict]:
    return {
        key: {
            "name": name,
            "min_percent": round(lo * 100),
            "max_percent": round(hi * 100) if hi is not None else _UNBOUNDED_PERCENT,
            "min_bpm": round(lthr_bpm * lo),
            "max_bpm": round(lthr_bpm * (hi if hi is not None else _UNBOUNDED_MULTIPLIER)),
        }
        for key, (name, lo, hi) in HR_ZONE_BANDS.items()
    }
