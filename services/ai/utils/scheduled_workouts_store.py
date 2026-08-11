import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ScheduledWorkoutsStore:
    """Tracks Garmin workouts the AI coach has scheduled, keyed by user_id -> date -> entries.

    This is the coach's own record of what IT pushed — used to detect same-day conflicts
    across reruns without depending on an unverified Garmin "list scheduled workouts" endpoint.
    It won't see workouts scheduled manually in Garmin Connect or by a prior run whose record
    was lost (e.g. this file deleted, or run from a different machine).
    """

    def __init__(self, base_dir: str = "data/storage"):
        self.base_dir = Path(base_dir)

    def _get_path(self, user_id: str) -> Path:
        user_dir = self.base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "scheduled_workouts.json"

    def load(self, user_id: str) -> dict[str, list[dict[str, str]]]:
        path = self._get_path(user_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load scheduled workouts for user %s", user_id)
            return {}

    def save(self, user_id: str, entries_by_date: dict[str, list[dict[str, str]]]) -> None:
        path = self._get_path(user_id)
        try:
            path.write_text(
                json.dumps(entries_by_date, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("Saved scheduled-workout records for %d date(s) for user %s", len(entries_by_date), user_id)
        except OSError:
            logger.exception("Failed to save scheduled workouts for user %s", user_id)
