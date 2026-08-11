"""Low-stakes manual test for the Garmin workout push feature.

Pushes ONE simple, clearly-labeled test session to your real Garmin account so you can
verify the (partly reverse-engineered) upload/schedule/description mechanics actually
work before trusting a full 4-week batch push. Does NOT run the AI pipeline at all —
this only exercises services/garmin/workout_builder.py + services/garmin/workout_client.py.

Usage:
    python examples/garmin/test_workout_push.py                  # push a test session (tomorrow, cycling)
    python examples/garmin/test_workout_push.py --sport running  # test a specific sport
    python examples/garmin/test_workout_push.py --sport mobility # test the unverified yoga sportTypeId guess
    python examples/garmin/test_workout_push.py --delete 123456  # remove a previously pushed test workout
"""

import argparse
import logging
import os
from datetime import date, timedelta

from dotenv import load_dotenv

from services.ai.session_extractor import PlannedSession, SessionStep
from services.garmin.client import GarminConnectClient
from services.garmin.workout_builder import build_garmin_workout
from services.garmin.workout_client import GarminWorkoutPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SPORT_CHOICES = ["running", "cycling", "swimming", "walking", "hiking", "strength", "mobility"]


def _connect() -> GarminConnectClient:
    load_dotenv()
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")
    if not email or not password:
        raise ValueError("Set GARMIN_EMAIL and GARMIN_PASSWORD in environment or .env file")
    client = GarminConnectClient()
    client.connect(email, password)
    return client


def _test_session(sport: str, test_date: str) -> PlannedSession:
    return PlannedSession(
        date=test_date,
        sport=sport,
        name="COACH TEST -- safe to delete",
        steps=[SessionStep(label="Coach test - easy", duration_min=10, zone_key=None)],
        description=(
            "This is a low-stakes test push from the Garmin AI Coach's workout-push feature. "
            "If you can see this text on your device/app, the description field round-trips "
            "correctly. Safe to delete."
        ),
    )


def push(sport: str) -> None:
    client = _connect()
    test_date = (date.today() + timedelta(days=1)).isoformat()
    session = _test_session(sport, test_date)

    workout = build_garmin_workout(session, power_zones={}, hr_zones={})
    if workout is None:
        logger.error("Could not build a workout for sport '%s' -- check the logs above", sport)
        return

    publisher = GarminWorkoutPublisher(client)
    workout_id = publisher.publish(workout, test_date)

    logger.info("Pushed test workout id=%s for %s (sport=%s)", workout_id, test_date, sport)
    logger.info("Check Garmin Connect (web/app/watch sync) on %s to verify:", test_date)
    logger.info("  - it appears on the calendar under the expected sport")
    logger.info("  - the description text shows up somewhere in the UI")
    logger.info("  - it's startable as a workout on your watch/head unit")
    logger.info("To remove it: python examples/garmin/test_workout_push.py --delete %s", workout_id)


def delete(workout_id: int) -> None:
    client = _connect()
    GarminWorkoutPublisher(client).delete(workout_id)
    logger.info("Deleted workout id=%s", workout_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sport", default="cycling", choices=SPORT_CHOICES, help="Sport to test (default: cycling)")
    parser.add_argument("--delete", type=int, metavar="WORKOUT_ID", help="Delete a previously pushed test workout")
    args = parser.parse_args()

    if args.delete is not None:
        delete(args.delete)
    else:
        push(args.sport)


if __name__ == "__main__":
    main()
