import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ActivityNotesStore:
    """Persists per-activity session notes across CLI runs, keyed by Garmin activity_id."""

    def __init__(self, base_dir: str = "data/storage"):
        self.base_dir = Path(base_dir)

    def _get_path(self, user_id: str) -> Path:
        user_dir = self.base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "activity_notes.json"

    def load(self, user_id: str) -> dict[str, dict[str, str]]:
        path = self._get_path(user_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load activity notes for user %s", user_id)
            return {}

    def save(self, user_id: str, notes: dict[str, dict[str, str]]) -> None:
        path = self._get_path(user_id)
        try:
            path.write_text(
                json.dumps(notes, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("Saved %d activity note(s) for user %s to %s", len(notes), user_id, path)
        except OSError:
            logger.exception("Failed to save activity notes for user %s", user_id)
