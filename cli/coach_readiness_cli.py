#!/usr/bin/env python3
"""Readiness snapshot for coach chats.

Prints today's Garmin readiness markers (sleep, HRV, resting HR, training load, Garmin's
training readiness score) plus the last few days of sleep and activities as compact
Markdown — the input for "should I adapt today's session?" questions. Garmin only, no LLM
call, so it costs no API tokens. Non-interactive when cached Garmin tokens are valid, so it
can be run from a Claude Code chat.
"""

import argparse
import getpass
import logging
import os
import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from cli.garmin_ai_coach_cli import ConfigParser
from services.garmin import TriathlonCoachDataExtractor
from services.garmin.models import Activity, RecoveryIndicators

# The coach CLI module configures INFO logging on import; keep this snapshot's output readable.
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _connect(config_parser: ConfigParser) -> TriathlonCoachDataExtractor:
    """Logs in via cached Garmin tokens first; only asks for a password if those are expired."""
    _, email = config_parser.get_athlete_info()
    password = config_parser.config.get("credentials", {}).get("password", "") or os.getenv("GARMIN_PASSWORD", "")
    try:
        return TriathlonCoachDataExtractor(email, password)
    except Exception:
        if password or not sys.stdin.isatty():
            raise RuntimeError(
                "Garmin login needs a fresh password (cached tokens expired). Run "
                "`pixi run coach-readiness --config <config>` once in your own terminal to log in, "
                "or set GARMIN_PASSWORD."
            ) from None
        return TriathlonCoachDataExtractor(email, getpass.getpass("Enter Garmin Connect password: "))


def _fmt(value: Any, unit: str = "", digits: int = 0) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        value = round(value, digits) if digits else round(value)
    return f"{value}{unit}"


def _training_readiness(extractor: TriathlonCoachDataExtractor, day: date) -> str:
    raw = extractor._call_api(
        extractor.garmin.client.get_training_readiness,
        day.isoformat(),
        default=None,
        what=f"get_training_readiness({day})",
    )
    entry = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(entry, Mapping) or entry.get("score") is None:
        return "n/a"
    level = entry.get("level") or ""
    feedback = entry.get("feedbackShort") or ""
    return " ".join(part for part in (str(entry["score"]), f"({level})" if level else "", feedback) if part)


def _sleep_row(night: RecoveryIndicators) -> str:
    sleep = night.sleep or {}
    stress = night.stress or {}
    duration = (sleep.get("duration") or {}).get("total")
    score = (sleep.get("quality") or {}).get("overall_score")
    return (
        f"| {night.date} | {_fmt(duration, ' h', 1)} | {_fmt(score)} | {_fmt(sleep.get('avg_overnight_hrv'), ' ms')} "
        f"| {_fmt(sleep.get('resting_heart_rate'), ' bpm')} | {_fmt(stress.get('avg_level'))} |"
    )


def _activity_row(activity: Activity) -> str:
    summary = activity.summary
    duration_min = summary.duration / 60 if summary and summary.duration else None
    distance_km = summary.distance / 1000 if summary and summary.distance else None
    return (
        f"| {(activity.start_time or '')[:10]} | {activity.activity_type or 'n/a'} | {activity.activity_name or 'n/a'} "
        f"| {_fmt(duration_min, ' min')} | {_fmt(distance_km, ' km', 1)} "
        f"| {_fmt(summary.average_hr if summary else None, ' bpm')} "
        f"| {_fmt(summary.activity_training_load if summary else None)} |"
    )


def build_readiness_snapshot(extractor: TriathlonCoachDataExtractor, today: date, days: int = 7) -> str:
    start = today - timedelta(days=days - 1)

    markers = extractor.get_physiological_markers(start, today)
    nights = extractor.get_recovery_indicators(start, today)
    status = extractor.get_training_status(today)
    activities = extractor.get_recent_activities(start, today)

    hrv = markers.hrv or {}
    baseline = hrv.get("baseline") or {}
    load = status.acute_training_load or {}
    last_night = nights[-1] if nights else RecoveryIndicators(date=today.isoformat())
    last_sleep = last_night.sleep or {}

    lines = [
        f"# Readiness snapshot — {today.isoformat()} ({today.strftime('%A')})",
        "",
        "## Today",
        f"- Garmin training readiness: {_training_readiness(extractor, today)}",
        f"- Last night: sleep {_fmt((last_sleep.get('duration') or {}).get('total'), ' h', 1)} "
        f"(score {_fmt((last_sleep.get('quality') or {}).get('overall_score'))}), "
        f"overnight HRV {_fmt(last_sleep.get('avg_overnight_hrv'), ' ms')}, "
        f"resting HR {_fmt(markers.resting_heart_rate or last_sleep.get('resting_heart_rate'), ' bpm')}",
        f"- HRV: last night {_fmt(hrv.get('last_night_avg'), ' ms')}, 7-day avg {_fmt(hrv.get('weekly_avg'), ' ms')}, "
        f"balanced baseline {_fmt(baseline.get('balanced_low'))}-{_fmt(baseline.get('balanced_upper'), ' ms')}",
        f"- Training load: acute {_fmt(load.get('acute_load'))}, chronic {_fmt(load.get('chronic_load'))}, "
        f"ACWR {_fmt(load.get('acwr'), '', 2)}",
        "",
        f"## Last {days} nights",
        "| Date | Sleep | Score | Overnight HRV | RHR | Avg stress |",
        "|---|---|---|---|---|---|",
        *(_sleep_row(night) for night in nights),
        "",
        f"## Activities, last {days} days",
    ]
    if activities:
        lines += [
            "| Date | Type | Name | Duration | Distance | Avg HR | Load |",
            "|---|---|---|---|---|---|---|",
            *(_activity_row(activity) for activity in sorted(activities, key=lambda a: a.start_time or "")),
        ]
    else:
        lines.append("No activities recorded.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Print today's Garmin readiness snapshot for coach chats (no AI tokens used).",
        epilog="Example: python coach_readiness_cli.py --config my_config.yaml",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--days", type=int, default=7, help="How many days of sleep/activities to include (default 7)")
    args = parser.parse_args()

    try:
        extractor = _connect(ConfigParser(args.config))
        print(build_readiness_snapshot(extractor, date.today(), args.days))
    except Exception as e:
        logger.error("❌ Readiness snapshot failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
