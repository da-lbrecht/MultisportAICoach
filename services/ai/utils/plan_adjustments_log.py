import logging
import re
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Each entry starts with a level-2 heading carrying its date, e.g. "## 2026-11-03 — Swapped Thu/Sat".
_ENTRY_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})\b", re.MULTILINE)


class PlanAdjustmentsLog:
    """Markdown log of plan changes agreed in coach chats (written by the `/coach` Claude Code skill).

    Lives next to the stored plans so the next full coach run can read what the athlete
    changed — and why — since the plan was generated.
    """

    def __init__(self, base_dir: str = "data/storage"):
        self.base_dir = Path(base_dir)

    def path(self, user_id: str) -> Path:
        return self.base_dir / user_id / "plan_adjustments.md"

    def recent_entries(self, user_id: str, since: date) -> list[str]:
        """Returns the log entries dated on or after `since`, oldest first, each as its full Markdown text."""
        path = self.path(user_id)
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Failed to read plan adjustments log for user %s", user_id)
            return []

        headings = list(_ENTRY_HEADING.finditer(text))
        entries: list[str] = []
        for i, heading in enumerate(headings):
            try:
                entry_date = datetime.strptime(heading.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
            if entry_date < since:
                continue
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            entries.append(text[heading.start():end].strip())
        return entries


def render_adjustments_context(entries: list[str], days: int) -> str:
    if not entries:
        return ""
    return "\n\n".join([
        f"PLAN ADJUSTMENTS FROM THE LAST {days} DAYS (agreed between athlete and coach in chat after "
        "a previous coach run — the athlete trained to the adjusted plan, so interpret recent "
        "activities against these changes and carry forward any lasting preferences or constraints "
        "they reveal):",
        *entries,
    ])
