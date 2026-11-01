from datetime import date
from unittest.mock import Mock, patch

import pytest

from services.garmin.models import (
    Activity,
    ActivitySummary,
    PhysiologicalMarkers,
    RecoveryIndicators,
    TrainingStatus,
)


def _extractor(readiness=None, activities=None):
    extractor = Mock()
    extractor.get_physiological_markers.return_value = PhysiologicalMarkers(
        resting_heart_rate=45,
        hrv={"weekly_avg": 55.0, "last_night_avg": 51.0, "baseline": {"balanced_low": 50.0, "balanced_upper": 62.0}},
    )
    extractor.get_recovery_indicators.return_value = [
        RecoveryIndicators(
            date="2026-11-01",
            sleep={
                "duration": {"total": 7.4},
                "quality": {"overall_score": 81},
                "avg_overnight_hrv": 51.0,
                "resting_heart_rate": 45,
            },
            stress={"avg_level": 24},
        )
    ]
    extractor.get_training_status.return_value = TrainingStatus(
        acute_training_load={"acute_load": 310.0, "chronic_load": 420.0, "acwr": 0.738}
    )
    extractor.get_recent_activities.return_value = activities or []
    extractor._call_api.return_value = readiness
    return extractor


class TestBuildReadinessSnapshot:
    def test_renders_today_markers_sleep_and_activities(self):
        from cli.coach_readiness_cli import build_readiness_snapshot

        activity = Activity(
            activity_type="cycling", activity_name="Z2 Ride", start_time="2026-10-31 09:00:00",
            summary=ActivitySummary(duration=5400, distance=45000.0, average_hr=132, activity_training_load=95),
        )
        snapshot = build_readiness_snapshot(
            _extractor(readiness=[{"score": 68, "level": "MODERATE", "feedbackShort": "GOOD_SLEEP"}], activities=[activity]),
            date(2026, 11, 1),
        )

        assert "# Readiness snapshot — 2026-11-01 (Sunday)" in snapshot
        assert "Garmin training readiness: 68 (MODERATE) GOOD_SLEEP" in snapshot
        assert "sleep 7.4 h (score 81)" in snapshot
        assert "last night 51 ms, 7-day avg 55 ms, balanced baseline 50-62 ms" in snapshot
        assert "ACWR 0.74" in snapshot
        assert "| 2026-10-31 | cycling | Z2 Ride | 90 min | 45.0 km | 132 bpm | 95 |" in snapshot

    def test_missing_readiness_and_activities_degrade_gracefully(self):
        from cli.coach_readiness_cli import build_readiness_snapshot

        snapshot = build_readiness_snapshot(_extractor(readiness=None), date(2026, 11, 1))

        assert "Garmin training readiness: n/a" in snapshot
        assert "No activities recorded." in snapshot


class TestConnect:
    def _parser(self, password=""):
        parser = Mock()
        parser.get_athlete_info.return_value = ("Athlete", "a@b.c")
        parser.config = {"credentials": {"password": password}}
        return parser

    def test_uses_cached_tokens_without_prompting(self, monkeypatch):
        from cli.coach_readiness_cli import _connect

        monkeypatch.delenv("GARMIN_PASSWORD", raising=False)
        with patch("cli.coach_readiness_cli.TriathlonCoachDataExtractor") as mock_cls, \
             patch("cli.coach_readiness_cli.getpass.getpass") as mock_getpass:
            _connect(self._parser())

        mock_cls.assert_called_once_with("a@b.c", "")
        mock_getpass.assert_not_called()

    def test_non_interactive_login_failure_explains_how_to_log_in(self, monkeypatch):
        from cli.coach_readiness_cli import _connect

        monkeypatch.delenv("GARMIN_PASSWORD", raising=False)
        with patch("cli.coach_readiness_cli.TriathlonCoachDataExtractor", side_effect=RuntimeError("expired")), \
             patch("cli.coach_readiness_cli.sys.stdin") as mock_stdin, \
             pytest.raises(RuntimeError, match="in your own terminal"):
            mock_stdin.isatty.return_value = False
            _connect(self._parser())

    def test_interactive_login_failure_prompts_for_password(self, monkeypatch):
        from cli.coach_readiness_cli import _connect

        monkeypatch.delenv("GARMIN_PASSWORD", raising=False)
        with patch("cli.coach_readiness_cli.TriathlonCoachDataExtractor", side_effect=[RuntimeError("expired"), Mock()]) as mock_cls, \
             patch("cli.coach_readiness_cli.sys.stdin") as mock_stdin, \
             patch("cli.coach_readiness_cli.getpass.getpass", return_value="pw"):
            mock_stdin.isatty.return_value = True
            _connect(self._parser())

        assert mock_cls.call_args.args == ("a@b.c", "pw")
