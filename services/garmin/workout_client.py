import logging
from typing import Any

from garminconnect.workout import BaseWorkout

from .client import GarminConnectClient

logger = logging.getLogger(__name__)


class GarminWorkoutPublishError(RuntimeError):
    pass


class GarminWorkoutPublisher:
    """Uploads structured workouts to Garmin Connect and schedules them on the calendar.

    Scheduling isn't wrapped by the `garminconnect` library, so this reuses the same
    authenticated `garth` session the library already sets up for `upload_workout`.
    """

    def __init__(self, client: GarminConnectClient):
        self._client = client

    def upload(self, workout: BaseWorkout) -> dict[str, Any]:
        return self._client.client.upload_workout(workout.to_dict())

    def schedule(self, workout_id: int, date_iso: str) -> dict[str, Any]:
        garmin = self._client.client
        url = f"{garmin.garmin_workouts_schedule_url}/{workout_id}"
        response = garmin.garth.post("connectapi", url, json={"date": date_iso}, api=True)
        return response.json()

    def publish(self, workout: BaseWorkout, date_iso: str) -> int:
        """Uploads and schedules the workout, returning its Garmin workoutId."""
        uploaded = self.upload(workout)
        workout_id = uploaded.get("workoutId") if isinstance(uploaded, dict) else None
        if not workout_id:
            raise GarminWorkoutPublishError(f"Garmin upload response missing workoutId: {uploaded!r}")

        self.schedule(workout_id, date_iso)
        return workout_id

    def delete(self, workout_id: int) -> None:
        """Deletes a workout entirely — removes it from the library and any calendar schedule."""
        garmin = self._client.client
        url = f"{garmin.garmin_workouts}/workout/{workout_id}"
        garmin.garth.request("DELETE", "connectapi", url, api=True)
