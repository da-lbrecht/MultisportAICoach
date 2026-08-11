from services.ai.utils.scheduled_workouts_store import ScheduledWorkoutsStore


def test_round_trip(tmp_path):
    store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
    assert store.load("cli_user") == {}

    entries = {
        "2026-08-11": [{"workout_id": "123", "name": "Ride", "sport": "cycling", "scheduled_at": "2026-08-09T10:00:00"}]
    }
    store.save("cli_user", entries)

    reloaded = ScheduledWorkoutsStore(base_dir=str(tmp_path))
    assert reloaded.load("cli_user") == entries


def test_load_returns_empty_dict_for_malformed_file(tmp_path):
    store = ScheduledWorkoutsStore(base_dir=str(tmp_path))
    path = store._get_path("cli_user")
    path.write_text("not valid json", encoding="utf-8")

    assert store.load("cli_user") == {}
