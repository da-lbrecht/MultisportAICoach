#!/usr/bin/env python3
"""Standalone Garmin workout push.

Reads the latest weekly plan written by a Garmin AI Coach run (or a plan file passed with
--plan), extracts every day's primary session from it, and — after a toggleable checklist —
schedules them as Garmin Connect workouts. Runs independently of the analysis pipeline, so a
plan can be (re)pushed at any time without regenerating it.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from cli.garmin_ai_coach_cli import ConfigParser, resolve_zones
from core.config import reload_config
from services.ai.ai_settings import ai_settings
from services.ai.session_extractor import PlannedSession, extract_planned_sessions
from services.ai.utils.plan_storage import FilePlanStorage
from services.ai.utils.scheduled_workouts_store import ScheduledWorkoutsStore
from services.garmin import TriathlonCoachDataExtractor
from services.garmin.client import GarminConnectClient
from services.garmin.workout_builder import build_garmin_workout
from services.garmin.workout_client import GarminWorkoutPublisher

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


PUSH_WINDOW_DAYS = 28  # Matches the weekly planner's own 28-day (4-week) horizon.
USER_ID = "cli_user"  # The coach run stores its latest plan under this user id.


def load_weekly_plan(plan_path: Path | None = None, user_id: str = USER_ID) -> tuple[str | None, str]:
    """Returns (plan markdown, human-readable source). Defaults to the coach's latest stored plan."""
    if plan_path is not None:
        if not plan_path.exists():
            return None, str(plan_path)
        return plan_path.read_text(encoding="utf-8"), str(plan_path)

    storage = FilePlanStorage()
    source_path = storage.base_dir / user_id / "weekly_plan.md"
    return storage.load_plan(user_id, "weekly_plan"), str(source_path)


def _describe_plan_source(plan_markdown: str, source: str) -> str:
    path = Path(source)
    modified = (
        datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        if path.exists()
        else "unknown"
    )
    title = next((line.lstrip("# ").strip() for line in plan_markdown.splitlines() if line.strip()), "")
    return f"{source} (saved {modified})" + (f" — {title}" if title else "")


def _filter_sessions_to_window(
    sessions: list[PlannedSession], start_date_iso: str, days: int = PUSH_WINDOW_DAYS
) -> list[PlannedSession]:
    start = datetime.strptime(start_date_iso, "%Y-%m-%d").date()
    end = start + timedelta(days=days - 1)

    in_window: list[PlannedSession] = []
    excluded = 0
    for session in sessions:
        try:
            session_date = datetime.strptime(session.date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Skipping extracted session with unparseable date %r", session.date)
            continue
        if start <= session_date <= end:
            in_window.append(session)
        else:
            excluded += 1

    if excluded:
        logger.info("Excluded %d extracted session(s) outside the %d-day push window", excluded, days)
    return in_window


def _session_duration_estimate_min(session: PlannedSession) -> float:
    return sum((step.duration_min or 0.0) * max(step.repeat, 1) for step in session.steps)


def _describe_session_steps(session: PlannedSession) -> str:
    parts: list[str] = []
    steps = session.steps
    i = 0
    while i < len(steps):
        step = steps[i]
        if step.repeat > 1:
            j = i
            group_labels = []
            while j < len(steps) and steps[j].repeat == step.repeat:
                group_labels.append(steps[j].label)
                j += 1
            parts.append(f"{step.repeat}x({', '.join(group_labels)})")
            i = j
        else:
            parts.append(step.label)
            i += 1
    return ", ".join(parts)


def _format_session_line(index: int, checked: bool, session: PlannedSession, conflict_note: str = "") -> str:
    mark = "x" if checked else " "
    weekday = datetime.strptime(session.date, "%Y-%m-%d").strftime("%a")
    duration = _session_duration_estimate_min(session)
    header = f"  [{mark}] {index}. {session.date} ({weekday})  {session.sport:<8} {session.name}  (~{duration:.0f} min)"
    if conflict_note:
        header += f"  {conflict_note}"
    return f"{header}\n         {_describe_session_steps(session)}"


def review_sessions_to_push(
    sessions: list[PlannedSession], conflict_notes: dict[str, str] | None = None
) -> list[PlannedSession]:
    if not sessions:
        return []

    conflict_notes = conflict_notes or {}
    checked = [True] * len(sessions)

    while True:
        print(f"\n{'='*60}")
        print(f"GARMIN WORKOUT PUSH — {len(sessions)} structured session(s) parsed from the plan")
        for i, session in enumerate(sessions, start=1):
            print(_format_session_line(i, checked[i - 1], session, conflict_notes.get(session.date, "")))
        print(f"{'='*60}")
        raw = input(
            "Toggle numbers (space-separated) to check/uncheck, Enter to push the checked "
            "session(s) above, or 'q' to cancel: "
        ).strip()

        if raw.lower() in ("q", "quit", "cancel"):
            print("Cancelled — nothing pushed to Garmin.")
            return []

        if not raw:
            selected = [s for s, is_checked in zip(sessions, checked, strict=True) if is_checked]
            if not selected:
                print("Nothing checked — nothing pushed to Garmin.")
            return selected

        toggled_any = False
        for token in raw.split():
            if token.isdigit() and 1 <= int(token) <= len(sessions):
                checked[int(token) - 1] = not checked[int(token) - 1]
                toggled_any = True
        if not toggled_any:
            print("Didn't recognize that input — use session numbers, Enter, or 'q'.")


def resolve_schedule_conflict(session: PlannedSession, existing_entries: list[dict[str, str]]) -> str:
    """Returns 'keep_existing', 'replace', or 'keep_both'."""
    print(
        f"\nGarmin already has {len(existing_entries)} session(s) scheduled for {session.date} "
        "from a previous push:"
    )
    for entry in existing_entries:
        print(
            f"  Existing: {entry.get('name', 'unknown')} ({entry.get('sport', 'unknown')}), "
            f"pushed {entry.get('scheduled_at', 'unknown time')}"
        )
    print(f"  New:      {session.name} ({session.sport})")

    while True:
        choice = input("Keep [e]xisting / [r]eplace with new / keep [b]oth? ").strip().lower()
        if choice in ("e", "existing"):
            return "keep_existing"
        if choice in ("r", "replace"):
            return "replace"
        if choice in ("b", "both"):
            return "keep_both"
        print("Please enter 'e', 'r', or 'b'.")


def _resolve_existing_entries(
    session: PlannedSession,
    existing: list[dict[str, str]],
    entries_by_date: dict[str, list[dict[str, str]]],
    publisher: GarminWorkoutPublisher,
) -> bool:
    """Handles a same-day conflict. Returns False if the new session should NOT be pushed."""
    decision = resolve_schedule_conflict(session, existing)
    if decision == "keep_existing":
        print(f"  ↷ Kept the existing session for {session.date}; new one not pushed.")
        return False

    if decision == "replace":
        for entry in existing:
            if not (workout_id := entry.get("workout_id")):
                continue
            try:
                publisher.delete(int(workout_id))
            except Exception:
                logger.exception("Failed to delete previous workout %s on %s", workout_id, session.date)
        entries_by_date[session.date] = []

    # "keep_both" falls through without touching the existing entries.
    return True


def push_sessions_to_garmin(
    sessions: list[PlannedSession],
    garmin_client: GarminConnectClient,
    power_zones: dict[str, dict],
    hr_zones: dict[str, dict],
    user_id: str = USER_ID,
    store: ScheduledWorkoutsStore | None = None,
) -> None:
    if not sessions:
        return

    store = store or ScheduledWorkoutsStore()
    entries_by_date = store.load(user_id)
    publisher = GarminWorkoutPublisher(garmin_client)

    for session in sessions:
        existing = entries_by_date.get(session.date, [])
        if existing and not _resolve_existing_entries(session, existing, entries_by_date, publisher):
            continue

        workout = build_garmin_workout(session, power_zones, hr_zones)
        if workout is None:
            print(f"  ⚠️  Skipped {session.date} {session.name} — couldn't build a valid workout (see logs).")
            continue

        try:
            workout_id = publisher.publish(workout, session.date)
        except Exception:
            logger.exception("Failed to push session '%s' (%s) to Garmin", session.name, session.date)
            print(f"  ❌ Failed to push {session.date} {session.name} — see logs.")
            continue

        print(f"  ✅ Pushed {session.date} {session.name} to Garmin Connect.")
        entries_by_date.setdefault(session.date, []).append({
            "workout_id": str(workout_id),
            "name": session.name,
            "sport": session.sport,
            "scheduled_at": datetime.now().isoformat(timespec="seconds"),
        })

    store.save(user_id, entries_by_date)


async def collect_and_push_garmin_workouts(
    weekly_plan_markdown: str,
    power_zones: dict[str, dict],
    hr_zones: dict[str, dict],
    garmin_client: GarminConnectClient,
    current_date_iso: str,
    user_id: str = USER_ID,
    store: ScheduledWorkoutsStore | None = None,
) -> None:
    print("\nExtracting structured sessions from the weekly plan for Garmin push...")
    sessions = await extract_planned_sessions(weekly_plan_markdown, power_zones, hr_zones)
    sessions = _filter_sessions_to_window(sessions, current_date_iso)

    if not sessions:
        print(f"No sessions found from {current_date_iso} through the next {PUSH_WINDOW_DAYS} days to push.")
        return

    store = store or ScheduledWorkoutsStore()
    entries_by_date = store.load(user_id)
    conflict_notes = {
        session.date: f"⚠ already has {len(entries_by_date[session.date])} session(s) scheduled"
        for session in sessions
        if entries_by_date.get(session.date)
    }

    selected = review_sessions_to_push(sessions, conflict_notes)
    if not selected:
        return

    push_sessions_to_garmin(selected, garmin_client, power_zones, hr_zones, user_id=user_id, store=store)


async def run_push_from_config(config_path: Path, plan_path: Path | None = None) -> None:
    config_parser = ConfigParser(config_path)
    _, email = config_parser.get_athlete_info()

    plan_markdown, plan_source = load_weekly_plan(plan_path)
    if not plan_markdown or not plan_markdown.strip():
        raise FileNotFoundError(
            f"No weekly plan found at {plan_source} — run the coach first (pixi run coach-cli) "
            "or pass a plan file with --plan."
        )
    print(f"\nUsing weekly plan: {_describe_plan_source(plan_markdown, plan_source)}")

    # Session extraction is an LLM call, so it uses the same AI mode as the coach run.
    os.environ["AI_MODE"] = config_parser.get_extraction_config().get("ai_mode", "development")
    reload_config()
    ai_settings.reload()
    logger.info("AI Mode: %s", os.environ["AI_MODE"])

    password = config_parser.get_password()
    logger.info("Connecting to Garmin Connect...")
    extractor = TriathlonCoachDataExtractor(email, password)

    static_power_zones, static_hr_zones = config_parser.get_athlete_zones()
    power_zones, hr_zones = resolve_zones(
        extractor.get_training_thresholds(), static_power_zones, static_hr_zones
    )

    await collect_and_push_garmin_workouts(
        plan_markdown,
        power_zones,
        hr_zones,
        extractor.garmin,
        date.today().isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Garmin Workout Push — schedule the latest Garmin AI Coach weekly plan as Garmin "
            f"Connect workouts (today through the next {PUSH_WINDOW_DAYS} days)."
        ),
        epilog="Example: python garmin_workout_push_cli.py --config my_config.yaml",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configuration file (YAML or JSON)")
    parser.add_argument(
        "--plan",
        type=Path,
        help="Weekly plan markdown to push (default: the latest plan saved by the coach run)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_push_from_config(args.config, args.plan))
    except KeyboardInterrupt:
        logger.info("❌ Push cancelled by user")
    except Exception as e:
        logger.error("❌ Push failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
