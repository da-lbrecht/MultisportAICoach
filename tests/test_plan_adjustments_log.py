from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from services.ai.utils.plan_adjustments_log import PlanAdjustmentsLog, render_adjustments_context
from services.garmin.models import GarminData

LOG = """# Plan Adjustments Log

## 2026-10-01 — Old change
- **Change:** something long ago

## 2026-11-03 — Swapped Thu intervals with Sat long ride
- **Affected dates:** 2026-11-05, 2026-11-07
- **Reason:** work trip Thursday

## 2026-11-04 — Shortened swim
- **Change:** 2,000 m → 1,500 m
"""


def _write_log(tmp_path, text=LOG):
    log = PlanAdjustmentsLog(base_dir=str(tmp_path))
    log.path("cli_user").parent.mkdir(parents=True)
    log.path("cli_user").write_text(text, encoding="utf-8")
    return log


class TestPlanAdjustmentsLog:
    def test_returns_entries_on_or_after_cutoff_oldest_first(self, tmp_path):
        log = _write_log(tmp_path)

        entries = log.recent_entries("cli_user", since=date(2026, 11, 3))

        assert len(entries) == 2
        assert entries[0].startswith("## 2026-11-03 — Swapped Thu intervals")
        assert "work trip Thursday" in entries[0]
        assert "Shortened swim" not in entries[0]  # entries are split at the next heading
        assert entries[1].startswith("## 2026-11-04 — Shortened swim")

    def test_missing_log_returns_no_entries(self, tmp_path):
        assert PlanAdjustmentsLog(base_dir=str(tmp_path)).recent_entries("cli_user", date(2026, 1, 1)) == []

    def test_title_and_undated_text_are_ignored(self, tmp_path):
        log = _write_log(tmp_path, "# Plan Adjustments Log\n\nSome intro text.\n")

        assert log.recent_entries("cli_user", date(2000, 1, 1)) == []

    def test_render_is_empty_without_entries(self):
        assert render_adjustments_context([], 50) == ""

    def test_render_includes_header_and_entries(self):
        rendered = render_adjustments_context(["## 2026-11-03 — A change"], 50)

        assert rendered.startswith("PLAN ADJUSTMENTS FROM THE LAST 50 DAYS")
        assert "## 2026-11-03 — A change" in rendered


@pytest.mark.asyncio
@patch("cli.garmin_ai_coach_cli.run_complete_analysis_and_planning", new_callable=AsyncMock)
@patch("cli.garmin_ai_coach_cli.TriathlonCoachDataExtractor")
@patch("cli.garmin_ai_coach_cli.OutsideApiGraphQlClient")
async def test_coach_run_includes_recent_adjustments_in_contexts(
    mock_outside_client, mock_extractor_class, mock_workflow, tmp_path
):
    from datetime import datetime, timedelta

    from cli.garmin_ai_coach_cli import run_analysis_from_config

    recent = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    stale = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    # The conftest fixture runs each test from tmp_path, so the default store location is isolated.
    _write_log(tmp_path / "data" / "storage", (
        f"# Plan Adjustments Log\n\n## {stale} — Stale change\n- old\n\n"
        f"## {recent} — Moved long ride to Sunday\n- **Reason:** family visit\n"
    ))

    mock_workflow.return_value = {"cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0}}
    mock_extractor_class.return_value.extract_data.return_value = GarminData()
    mock_outside_client.return_value.get_competitions.return_value = []

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
athlete:
  email: "user@example.com"
context:
  analysis: "Analysis context"
  planning: "Planning context"
extraction:
  activities_days: 7
  hitl_enabled: false
output:
  directory: "{(tmp_path / 'out').as_posix()}"
credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    await run_analysis_from_config(config_path)

    kwargs = mock_workflow.call_args.kwargs
    for context in (kwargs["analysis_context"], kwargs["planning_context"]):
        assert "Moved long ride to Sunday" in context
        assert "family visit" in context
        assert "Stale change" not in context  # older than the 7-day activities window
