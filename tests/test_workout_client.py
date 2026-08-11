from unittest.mock import MagicMock, Mock

import pytest
from garminconnect.workout import CyclingWorkout, WorkoutSegment

from services.garmin.client import GarminConnectClient
from services.garmin.workout_client import GarminWorkoutPublisher, GarminWorkoutPublishError


def _make_workout() -> CyclingWorkout:
    return CyclingWorkout(
        workoutName="Test Ride",
        estimatedDurationInSecs=1800,
        workoutSegments=[WorkoutSegment(segmentOrder=1, sportType={"sportTypeId": 2, "sportTypeKey": "cycling"}, workoutSteps=[])],
    )


@pytest.fixture
def mock_garmin_connect_client():
    client = GarminConnectClient.__new__(GarminConnectClient)
    client._client = Mock()
    return client


def test_upload_calls_library_upload_workout(mock_garmin_connect_client):
    mock_garmin_connect_client.client.upload_workout.return_value = {"workoutId": 999}
    publisher = GarminWorkoutPublisher(mock_garmin_connect_client)

    result = publisher.upload(_make_workout())

    assert result == {"workoutId": 999}
    mock_garmin_connect_client.client.upload_workout.assert_called_once()
    uploaded_payload = mock_garmin_connect_client.client.upload_workout.call_args.args[0]
    assert uploaded_payload["workoutName"] == "Test Ride"


def test_schedule_posts_to_the_schedule_url_with_the_date(mock_garmin_connect_client):
    garmin = mock_garmin_connect_client.client
    garmin.garmin_workouts_schedule_url = "/workout-service/schedule"
    garmin.garth.post.return_value = MagicMock(json=lambda: {"scheduleId": 42})

    publisher = GarminWorkoutPublisher(mock_garmin_connect_client)
    result = publisher.schedule(999, "2026-08-11")

    assert result == {"scheduleId": 42}
    garmin.garth.post.assert_called_once_with(
        "connectapi", "/workout-service/schedule/999", json={"date": "2026-08-11"}, api=True
    )


def test_publish_uploads_then_schedules_and_returns_workout_id(mock_garmin_connect_client):
    garmin = mock_garmin_connect_client.client
    garmin.garmin_workouts_schedule_url = "/workout-service/schedule"
    garmin.upload_workout.return_value = {"workoutId": 123}
    garmin.garth.post.return_value = MagicMock(json=lambda: {"scheduleId": 1})

    publisher = GarminWorkoutPublisher(mock_garmin_connect_client)
    workout_id = publisher.publish(_make_workout(), "2026-08-11")

    assert workout_id == 123
    garmin.garth.post.assert_called_once_with(
        "connectapi", "/workout-service/schedule/123", json={"date": "2026-08-11"}, api=True
    )


def test_publish_raises_clear_error_when_upload_response_has_no_workout_id(mock_garmin_connect_client):
    mock_garmin_connect_client.client.upload_workout.return_value = {"unexpected": "shape"}
    publisher = GarminWorkoutPublisher(mock_garmin_connect_client)

    with pytest.raises(GarminWorkoutPublishError):
        publisher.publish(_make_workout(), "2026-08-11")


def test_delete_sends_delete_request_to_the_workout_url(mock_garmin_connect_client):
    garmin = mock_garmin_connect_client.client
    garmin.garmin_workouts = "/workout-service"

    publisher = GarminWorkoutPublisher(mock_garmin_connect_client)
    publisher.delete(123)

    garmin.garth.request.assert_called_once_with("DELETE", "connectapi", "/workout-service/workout/123", api=True)
