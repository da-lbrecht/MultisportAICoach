from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.ai.session_extractor import PlannedSession, SessionStep
from services.ai.utils.scheduled_workouts_store import ScheduledWorkoutsStore
from services.garmin.models import TrainingThresholds


def _session(date: str, sport: str = "cycling", name: str = "Ride") -> PlannedSession:
    return PlannedSession(
        date=date, sport=sport, name=name,
        steps=[SessionStep(label="Main", duration_min=30)],
        description=f"{name} plan text",
    )


class TestResolveZones:
    def test_prefers_live_thresholds_when_present(self):
        from cli.garmin_ai_coach_cli import resolve_zones

        power_zones, hr_zones = resolve_zones(
            TrainingThresholds(ftp_watts=250.0, lactate_threshold_hr=165, lactate_threshold_speed_ms=None),
            static_power_zones={"z2_endurance": {"min_watts": 1}},
            static_hr_zones={"z2_aerobic": {"min_bpm": 1}},
        )

        assert power_zones["z4_threshold"]["min_watts"] == round(250.0 * 0.90)
        assert hr_zones["z4_threshold"]["min_bpm"] == round(165 * 0.94)

    def test_falls_back_to_static_config_when_thresholds_missing(self):
        from cli.garmin_ai_coach_cli import resolve_zones

        static_power = {"z2_endurance": {"min_watts": 100, "max_watts": 150}}
        static_hr = {"z2_aerobic": {"min_bpm": 110, "max_bpm": 130}}

        power_zones, hr_zones = resolve_zones(
            TrainingThresholds(ftp_watts=None, lactate_threshold_hr=None, lactate_threshold_speed_ms=None),
            static_power, static_hr,
        )

        assert power_zones == static_power
        assert hr_zones == static_hr

    def test_falls_back_to_static_when_thresholds_object_is_none(self):
        from cli.garmin_ai_coach_cli import resolve_zones

        static_power = {"z2_endurance": {"min_watts": 100}}
        static_hr = {"z2_aerobic": {"min_bpm": 110}}

        power_zones, hr_zones = resolve_zones(None, static_power, static_hr)

        assert power_zones == static_power
        assert hr_zones == static_hr

    def test_resolves_power_and_hr_independently(self):
        from cli.garmin_ai_coach_cli import resolve_zones

        static_power = {"z2_endurance": {"min_watts": 100}}
        static_hr = {"z2_aerobic": {"min_bpm": 110}}

        power_zones, hr_zones = resolve_zones(
            TrainingThresholds(ftp_watts=250.0, lactate_threshold_hr=None, lactate_threshold_speed_ms=None),
            static_power, static_hr,
        )

        assert power_zones != static_power  # live FTP used
        assert hr_zones == static_hr  # no live LTHR — fell back


class TestFilterSessionsToWindow:
    def test_keeps_only_sessions_within_28_days(self):
        from cli.garmin_ai_coach_cli import _filter_sessions_to_window

        sessions = [
            _session("2026-08-09"),   # day 0 — in window
            _session("2026-09-05"),   # day 27 — in window (last day)
            _session("2026-09-06"),   # day 28 — outside window
            _session("2026-08-01"),   # before start — outside window
        ]

        result = _filter_sessions_to_window(sessions, "2026-08-09")

        assert [s.date for s in result] == ["2026-08-09", "2026-09-05"]

    def test_skips_sessions_with_unparseable_dates(self):
        from cli.garmin_ai_coach_cli import _filter_sessions_to_window

        sessions = [_session("not-a-date"), _session("2026-08-10")]
        result = _filter_sessions_to_window(sessions, "2026-08-09")

        assert [s.date for s in result] == ["2026-08-10"]


class TestReviewSessionsToPush:
    def test_default_confirm_pushes_all_checked(self):
        from cli.garmin_ai_coach_cli import review_sessions_to_push

        sessions = [_session("2026-08-11"), _session("2026-08-13")]
        with patch("builtins.input", return_value=""):
            selected = review_sessions_to_push(sessions)

        assert selected == sessions

    def test_toggle_then_confirm_excludes_toggled_session(self):
        from cli.garmin_ai_coach_cli import review_sessions_to_push

        sessions = [_session("2026-08-11"), _session("2026-08-13")]
        with patch("builtins.input", side_effect=["2", ""]):
            selected = review_sessions_to_push(sessions)

        assert selected == [sessions[0]]

    def test_cancel_returns_empty_list(self):
        from cli.garmin_ai_coach_cli import review_sessions_to_push

        sessions = [_session("2026-08-11")]
        with patch("builtins.input", return_value="q"):
            selected = review_sessions_to_push(sessions)

        assert selected == []

    def test_empty_sessions_returns_immediately_without_prompting(self):
        from cli.garmin_ai_coach_cli import review_sessions_to_push

        with patch("builtins.input") as mock_input:
            selected = review_sessions_to_push([])

        assert selected == []
        mock_input.assert_not_called()


class TestResolveScheduleConflict:
    @pytest.mark.parametrize("choice,expected", [("e", "keep_existing"), ("r", "replace"), ("b", "keep_both")])
    def test_recognizes_shorthand_choices(self, choice, expected):
        from cli.garmin_ai_coach_cli import resolve_schedule_conflict

        session = _session("2026-08-11")
        existing = [{"name": "Old Ride", "sport": "cycling", "scheduled_at": "2026-08-01T10:00:00"}]

        with patch("builtins.input", return_value=choice):
            assert resolve_schedule_conflict(session, existing) == expected

    def test_reprompts_on_invalid_input(self):
        from cli.garmin_ai_coach_cli import resolve_schedule_conflict

        session = _session("2026-08-11")
        existing = [{"name": "Old Ride", "sport": "cycling", "scheduled_at": "2026-08-01T10:00:00"}]

        with patch("builtins.input", side_effect=["nonsense", "r"]) as mock_input:
            result = resolve_schedule_conflict(session, existing)

        assert result == "replace"
        assert mock_input.call_count == 2


class TestPushSessionsToGarmin:
    def test_pushes_new_session_and_records_it_in_store(self, tmp_path):
        from cli.garmin_ai_coach_cli import push_sessions_to_garmin

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
        session = _session("2026-08-11", sport="cycling", name="Sweet Spot Ride")

        with patch("cli.garmin_ai_coach_cli.build_garmin_workout", return_value=Mock()), \
             patch("cli.garmin_ai_coach_cli.GarminWorkoutPublisher") as mock_publisher_cls:
            mock_publisher_cls.return_value.publish.return_value = 555
            push_sessions_to_garmin([session], garmin_client=Mock(), power_zones={}, hr_zones={}, store=store)

        saved = store.load("cli_user")
        assert saved["2026-08-11"][0]["workout_id"] == "555"
        assert saved["2026-08-11"][0]["name"] == "Sweet Spot Ride"

    def test_skips_session_when_workout_cannot_be_built(self, tmp_path):
        from cli.garmin_ai_coach_cli import push_sessions_to_garmin

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
        session = _session("2026-08-11")

        with patch("cli.garmin_ai_coach_cli.build_garmin_workout", return_value=None), \
             patch("cli.garmin_ai_coach_cli.GarminWorkoutPublisher") as mock_publisher_cls:
            push_sessions_to_garmin([session], garmin_client=Mock(), power_zones={}, hr_zones={}, store=store)

        mock_publisher_cls.return_value.publish.assert_not_called()
        assert store.load("cli_user") == {}

    def test_conflict_keep_existing_skips_the_new_push(self, tmp_path):
        from cli.garmin_ai_coach_cli import push_sessions_to_garmin

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
        store.save("cli_user", {
            "2026-08-11": [{"workout_id": "111", "name": "Old Ride", "sport": "cycling", "scheduled_at": "x"}]
        })
        session = _session("2026-08-11", name="New Ride")

        with patch("cli.garmin_ai_coach_cli.resolve_schedule_conflict", return_value="keep_existing"), \
             patch("cli.garmin_ai_coach_cli.build_garmin_workout") as mock_build, \
             patch("cli.garmin_ai_coach_cli.GarminWorkoutPublisher") as mock_publisher_cls:
            push_sessions_to_garmin([session], garmin_client=Mock(), power_zones={}, hr_zones={}, store=store)

        mock_build.assert_not_called()
        mock_publisher_cls.return_value.publish.assert_not_called()
        saved = store.load("cli_user")
        assert saved["2026-08-11"][0]["name"] == "Old Ride"  # untouched

    def test_conflict_replace_deletes_old_and_pushes_new(self, tmp_path):
        from cli.garmin_ai_coach_cli import push_sessions_to_garmin

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
        store.save("cli_user", {
            "2026-08-11": [{"workout_id": "111", "name": "Old Ride", "sport": "cycling", "scheduled_at": "x"}]
        })
        session = _session("2026-08-11", name="New Ride")

        with patch("cli.garmin_ai_coach_cli.resolve_schedule_conflict", return_value="replace"), \
             patch("cli.garmin_ai_coach_cli.build_garmin_workout", return_value=Mock()), \
             patch("cli.garmin_ai_coach_cli.GarminWorkoutPublisher") as mock_publisher_cls:
            mock_publisher = mock_publisher_cls.return_value
            mock_publisher.publish.return_value = 222
            push_sessions_to_garmin([session], garmin_client=Mock(), power_zones={}, hr_zones={}, store=store)

        mock_publisher.delete.assert_called_once_with(111)
        saved = store.load("cli_user")
        assert len(saved["2026-08-11"]) == 1
        assert saved["2026-08-11"][0]["workout_id"] == "222"
        assert saved["2026-08-11"][0]["name"] == "New Ride"

    def test_conflict_keep_both_leaves_old_and_adds_new(self, tmp_path):
        from cli.garmin_ai_coach_cli import push_sessions_to_garmin

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
        store.save("cli_user", {
            "2026-08-11": [{"workout_id": "111", "name": "Old Ride", "sport": "cycling", "scheduled_at": "x"}]
        })
        session = _session("2026-08-11", name="New Ride")

        with patch("cli.garmin_ai_coach_cli.resolve_schedule_conflict", return_value="keep_both"), \
             patch("cli.garmin_ai_coach_cli.build_garmin_workout", return_value=Mock()), \
             patch("cli.garmin_ai_coach_cli.GarminWorkoutPublisher") as mock_publisher_cls:
            mock_publisher = mock_publisher_cls.return_value
            mock_publisher.publish.return_value = 222
            push_sessions_to_garmin([session], garmin_client=Mock(), power_zones={}, hr_zones={}, store=store)

        mock_publisher.delete.assert_not_called()
        saved = store.load("cli_user")
        assert {entry["name"] for entry in saved["2026-08-11"]} == {"Old Ride", "New Ride"}


class TestCollectAndPushGarminWorkouts:
    @pytest.mark.asyncio
    async def test_filters_window_then_reviews_then_pushes(self, tmp_path):
        from cli.garmin_ai_coach_cli import collect_and_push_garmin_workouts

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
        in_window = _session("2026-08-11", name="In Window")
        out_of_window = _session("2026-12-01", name="Out Of Window")

        with patch(
            "cli.garmin_ai_coach_cli.extract_planned_sessions",
            new=AsyncMock(return_value=[in_window, out_of_window]),
        ), patch("cli.garmin_ai_coach_cli.review_sessions_to_push", return_value=[in_window]) as mock_review, \
           patch("cli.garmin_ai_coach_cli.push_sessions_to_garmin") as mock_push:
            await collect_and_push_garmin_workouts(
                "plan markdown", {}, {}, garmin_client=Mock(), current_date_iso="2026-08-09", store=store,
            )

        reviewed_sessions = mock_review.call_args.args[0]
        assert [s.name for s in reviewed_sessions] == ["In Window"]  # out-of-window already excluded
        mock_push.assert_called_once()
        assert mock_push.call_args.args[0] == [in_window]

    @pytest.mark.asyncio
    async def test_no_sessions_found_skips_review_and_push(self, tmp_path):
        from cli.garmin_ai_coach_cli import collect_and_push_garmin_workouts

        store = ScheduledWorkoutsStore(base_dir=str(tmp_path))

        with patch(
            "cli.garmin_ai_coach_cli.extract_planned_sessions", new=AsyncMock(return_value=[])
        ), patch("cli.garmin_ai_coach_cli.review_sessions_to_push") as mock_review, \
           patch("cli.garmin_ai_coach_cli.push_sessions_to_garmin") as mock_push:
            await collect_and_push_garmin_workouts(
                "plan markdown", {}, {}, garmin_client=Mock(), current_date_iso="2026-08-09", store=store,
            )

        mock_review.assert_not_called()
        mock_push.assert_not_called()
