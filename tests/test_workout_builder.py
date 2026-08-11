from garminconnect.workout import (
    BaseWorkout,
    CyclingWorkout,
    ExecutableStep,
    FitnessEquipmentWorkout,
    RepeatGroup,
    RunningWorkout,
    SwimmingWorkout,
    WalkingWorkout,
)

from services.ai.session_extractor import PlannedSession, SessionStep
from services.garmin.workout_builder import build_garmin_workout

POWER_ZONES = {
    "z2_endurance": {"min_watts": 149, "max_watts": 203},
    "z4_threshold": {"min_watts": 244, "max_watts": 285},
}
HR_ZONES = {
    "z2_aerobic": {"min_bpm": 115, "max_bpm": 134},
    "z4_threshold": {"min_bpm": 154, "max_bpm": 173},
}


def _session(sport: str, name: str, steps: list[SessionStep], description: str = "Plan text verbatim") -> PlannedSession:
    return PlannedSession(date="2026-08-11", sport=sport, name=name, steps=steps, description=description)


def test_build_cycling_workout_maps_zone_to_power_target():
    session = _session("cycling", "Sweet Spot Ride", [
        SessionStep(label="Warmup", duration_min=10, zone_key="z2_endurance"),
        SessionStep(label="Main set", duration_min=30, zone_key="z4_threshold"),
        SessionStep(label="Cooldown", duration_min=10, zone_key=None),
    ])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, CyclingWorkout)
    assert workout.workoutName == "Sweet Spot Ride"
    assert workout.estimatedDurationInSecs == 50 * 60
    steps = workout.workoutSegments[0].workoutSteps
    assert len(steps) == 3
    main_set = steps[1]
    assert isinstance(main_set, ExecutableStep)
    assert main_set.targetType["workoutTargetTypeKey"] == "power.zone"
    assert main_set.targetValueOne == 244
    assert main_set.targetValueTwo == 285
    assert main_set.endConditionValue == 30 * 60


def test_build_running_workout_maps_zone_to_heart_rate_target():
    session = _session("running", "Easy Run", [SessionStep(label="Steady state", duration_min=45, zone_key="z2_aerobic")])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, RunningWorkout)
    step = workout.workoutSegments[0].workoutSteps[0]
    assert step.targetType["workoutTargetTypeKey"] == "heart.rate.zone"
    assert step.targetValueOne == 115
    assert step.targetValueTwo == 134


def test_build_swimming_workout_always_uses_open_target():
    session = _session(
        "swimming", "Endurance Swim",
        # zone_key set despite swimming having no configured zones — must be ignored.
        [SessionStep(label="Main set", distance_m=2000, zone_key="z4_threshold")],
    )

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, SwimmingWorkout)
    step = workout.workoutSegments[0].workoutSteps[0]
    assert step.targetType["workoutTargetTypeKey"] == "no.target"
    assert step.endCondition["conditionTypeKey"] == "distance"
    assert step.endConditionValue == 2000


def test_unknown_zone_key_falls_back_to_open_target_instead_of_crashing():
    session = _session("cycling", "Ride", [SessionStep(label="Main set", duration_min=20, zone_key="not_a_real_zone")])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    step = workout.workoutSegments[0].workoutSteps[0]
    assert step.targetType["workoutTargetTypeKey"] == "no.target"


def test_repeated_steps_are_grouped_into_one_repeat_group_not_flattened():
    session = _session("cycling", "VO2max Intervals", [
        SessionStep(label="Warmup", duration_min=10, zone_key="z2_endurance"),
        SessionStep(label="Interval", duration_min=4, repeat=4, zone_key="z4_threshold"),
        SessionStep(label="Recovery", duration_min=3, repeat=4, zone_key=None),
        SessionStep(label="Cooldown", duration_min=10, zone_key="z2_endurance"),
    ])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)
    steps = workout.workoutSegments[0].workoutSteps

    assert len(steps) == 3  # warmup, ONE repeat group, cooldown — not 4 separate steps
    assert isinstance(steps[0], ExecutableStep)
    repeat_group = steps[1]
    assert isinstance(repeat_group, RepeatGroup)
    assert repeat_group.numberOfIterations == 4
    assert len(repeat_group.workoutSteps) == 2  # interval + recovery, repeated together
    assert isinstance(steps[2], ExecutableStep)

    # Total duration accounts for the group repeating as a whole.
    assert workout.estimatedDurationInSecs == (10 + 4 * (4 + 3) + 10) * 60


def test_step_with_neither_duration_nor_distance_causes_session_to_be_skipped():
    session = _session("running", "Broken Session", [SessionStep(label="Mystery step")])

    assert build_garmin_workout(session, POWER_ZONES, HR_ZONES) is None


def test_description_is_carried_onto_the_built_workout():
    session = _session(
        "running", "Easy Run", [SessionStep(label="Steady state", duration_min=30)],
        description="FOCUS: Recovery\nWORKOUT: 30 min easy Z1/Z2",
    )

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert workout.to_dict()["description"] == "FOCUS: Recovery\nWORKOUT: 30 min easy Z1/Z2"


def test_walking_session_uses_walking_workout_class_with_open_target():
    session = _session("walking", "Recovery Walk", [SessionStep(label="Walk", duration_min=30)])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, WalkingWorkout)
    step = workout.workoutSegments[0].workoutSteps[0]
    assert step.targetType["workoutTargetTypeKey"] == "no.target"


def test_strength_session_uses_fitness_equipment_workout_class():
    session = _session("strength", "Lower Body Strength", [SessionStep(label="Circuit", duration_min=35)])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, FitnessEquipmentWorkout)


def test_mobility_session_uses_generic_workout_tagged_as_yoga():
    session = _session("mobility", "Yoga Flow", [SessionStep(label="Flow", duration_min=20)])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, BaseWorkout)
    assert workout.workoutSegments[0].sportType["sportTypeKey"] == "yoga"


def test_unrecognized_sport_falls_back_to_generic_other_workout():
    session = _session("meditation", "Breathwork", [SessionStep(label="Session", duration_min=15)])

    workout = build_garmin_workout(session, POWER_ZONES, HR_ZONES)

    assert isinstance(workout, BaseWorkout)
    assert workout.workoutSegments[0].sportType["sportTypeKey"] == "other"
