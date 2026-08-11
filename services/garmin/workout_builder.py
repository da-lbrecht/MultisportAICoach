import logging

from garminconnect.workout import (
    BaseWorkout,
    ConditionType,
    CyclingWorkout,
    ExecutableStep,
    FitnessEquipmentWorkout,
    HikingWorkout,
    RepeatGroup,
    RunningWorkout,
    SportType,
    StepType,
    SwimmingWorkout,
    TargetType,
    WalkingWorkout,
    WorkoutSegment,
    create_repeat_group,
)

from services.ai.session_extractor import PlannedSession, SessionStep

logger = logging.getLogger(__name__)

# running/cycling/swimming get a full multi-step, zone-targeted breakdown (see _target_for).
# Everything else still gets pushed — just as a single simple block — using the closest
# matching Garmin sport type, or a generic "other" fallback for anything unrecognized.
_WORKOUT_CLASSES: dict[str, type[BaseWorkout]] = {
    "running": RunningWorkout,
    "cycling": CyclingWorkout,
    "swimming": SwimmingWorkout,
    "walking": WalkingWorkout,
    "hiking": HikingWorkout,
    "strength": FitnessEquipmentWorkout,
    # sportTypeId 9 ("yoga") was tested against a live account and pushed successfully, but
    # Garmin rendered it as a generic/unrecognized activity rather than a proper Yoga icon —
    # no benefit over the generic fallback, so mobility/yoga sessions share the verified,
    # correctly-rendering Fitness Equipment mapping instead.
    "mobility": FitnessEquipmentWorkout,
}

_SPORT_TYPES: dict[str, dict] = {
    "running": {"sportTypeId": SportType.RUNNING, "sportTypeKey": "running", "displayOrder": 1},
    "cycling": {"sportTypeId": SportType.CYCLING, "sportTypeKey": "cycling", "displayOrder": 2},
    "swimming": {"sportTypeId": SportType.SWIMMING, "sportTypeKey": "swimming", "displayOrder": 3},
    "walking": {"sportTypeId": SportType.WALKING, "sportTypeKey": "walking", "displayOrder": 4},
    "hiking": {"sportTypeId": SportType.HIKING, "sportTypeKey": "hiking", "displayOrder": 7},
    "strength": {"sportTypeId": SportType.FITNESS_EQUIPMENT, "sportTypeKey": "fitness_equipment", "displayOrder": 6},
    "mobility": {"sportTypeId": SportType.FITNESS_EQUIPMENT, "sportTypeKey": "fitness_equipment", "displayOrder": 6},
}

_DEFAULT_WORKOUT_CLASS = BaseWorkout
_DEFAULT_SPORT_TYPE = {"sportTypeId": SportType.OTHER, "sportTypeKey": "other", "displayOrder": 8}

_NO_TARGET = {"workoutTargetTypeId": TargetType.NO_TARGET, "workoutTargetTypeKey": "no.target", "displayOrder": 1}


class IncompleteStepError(ValueError):
    """A session step has neither a duration nor a distance — can't build a Garmin step from it."""


def _classify_step_type(label: str) -> dict:
    lowered = label.lower()
    if "warm" in lowered:
        return {"stepTypeId": StepType.WARMUP, "stepTypeKey": "warmup", "displayOrder": 1}
    if "cool" in lowered:
        return {"stepTypeId": StepType.COOLDOWN, "stepTypeKey": "cooldown", "displayOrder": 2}
    if "recover" in lowered or "rest" in lowered or "easy" in lowered:
        return {"stepTypeId": StepType.RECOVERY, "stepTypeKey": "recovery", "displayOrder": 4}
    return {"stepTypeId": StepType.INTERVAL, "stepTypeKey": "interval", "displayOrder": 3}


def _end_condition(step: SessionStep) -> tuple[dict, float]:
    if step.duration_min is not None:
        condition = {
            "conditionTypeId": ConditionType.TIME,
            "conditionTypeKey": "time",
            "displayOrder": 2,
            "displayable": True,
        }
        return condition, step.duration_min * 60.0

    if step.distance_m is None:
        raise IncompleteStepError(f"step '{step.label}' has neither duration nor distance")

    condition = {
        "conditionTypeId": ConditionType.DISTANCE,
        "conditionTypeKey": "distance",
        "displayOrder": 1,
        "displayable": True,
    }
    return condition, step.distance_m


def _target_for(sport: str, zone_key: str | None, power_zones: dict, hr_zones: dict) -> dict:
    if not zone_key:
        return {"targetType": _NO_TARGET}

    if sport == "cycling":
        zone = power_zones.get(zone_key)
        if not zone or zone.get("min_watts") is None or zone.get("max_watts") is None:
            logger.warning("Unknown/incomplete power zone '%s' — falling back to open target", zone_key)
            return {"targetType": _NO_TARGET}
        return {
            "targetType": {"workoutTargetTypeId": TargetType.POWER, "workoutTargetTypeKey": "power.zone", "displayOrder": 5},
            "targetValueOne": zone["min_watts"],
            "targetValueTwo": zone["max_watts"],
        }

    if sport == "running":
        zone = hr_zones.get(zone_key)
        if not zone or zone.get("min_bpm") is None or zone.get("max_bpm") is None:
            logger.warning("Unknown/incomplete HR zone '%s' — falling back to open target", zone_key)
            return {"targetType": _NO_TARGET}
        return {
            "targetType": {
                "workoutTargetTypeId": TargetType.HEART_RATE,
                "workoutTargetTypeKey": "heart.rate.zone",
                "displayOrder": 2,
            },
            "targetValueOne": zone["min_bpm"],
            "targetValueTwo": zone["max_bpm"],
        }

    return {"targetType": _NO_TARGET}  # swimming has no configured zones


def _build_executable_step(
    step: SessionStep, sport: str, power_zones: dict, hr_zones: dict, step_order: int
) -> ExecutableStep:
    end_condition, end_value = _end_condition(step)
    return ExecutableStep(
        stepOrder=step_order,
        stepType=_classify_step_type(step.label),
        endCondition=end_condition,
        endConditionValue=end_value,
        **_target_for(sport, step.zone_key, power_zones, hr_zones),
    )


def _build_steps(
    steps: list[SessionStep], sport: str, power_zones: dict, hr_zones: dict
) -> list[ExecutableStep | RepeatGroup]:
    for step in steps:
        if step.duration_min is None and step.distance_m is None:
            raise IncompleteStepError(f"step '{step.label}' has neither duration nor distance")

    built: list[ExecutableStep | RepeatGroup] = []
    order = 1
    i, n = 0, len(steps)
    while i < n:
        step = steps[i]
        if step.repeat and step.repeat > 1:
            # Group the whole contiguous run sharing this repeat count into ONE RepeatGroup —
            # Garmin repeats the entire contained sequence, not each step individually. This is
            # what turns e.g. [interval(repeat=4), recovery(repeat=4)] into a proper
            # "4x(interval, recovery)" block rather than "interval x4" followed by "recovery x4".
            run = [step]
            j = i + 1
            while j < n and steps[j].repeat == step.repeat:
                run.append(steps[j])
                j += 1

            child_steps = [
                _build_executable_step(s, sport, power_zones, hr_zones, child_order)
                for child_order, s in enumerate(run, start=1)
            ]
            built.append(create_repeat_group(iterations=step.repeat, workout_steps=child_steps, step_order=order))
            order += 1
            i = j
        else:
            built.append(_build_executable_step(step, sport, power_zones, hr_zones, order))
            order += 1
            i += 1
    return built


def _total_duration_seconds(steps: list[ExecutableStep | RepeatGroup]) -> float:
    total = 0.0
    for step in steps:
        if isinstance(step, RepeatGroup):
            total += _total_duration_seconds(step.workoutSteps) * step.numberOfIterations
        elif step.endCondition and step.endCondition.get("conditionTypeKey") == "time":
            total += step.endConditionValue or 0.0
    return total


def build_garmin_workout(
    session: PlannedSession, power_zones: dict, hr_zones: dict
) -> BaseWorkout | None:
    """Maps an extracted PlannedSession into a Garmin Connect workout, or None if it can't be built.

    running/cycling/swimming get their typed, zone-targeted workout class; any other sport still
    gets pushed as a single-block workout under the closest matching (or generic "other") sport type.
    """
    try:
        steps = _build_steps(session.steps, session.sport, power_zones, hr_zones)
    except IncompleteStepError:
        logger.exception("Skipping session '%s' (%s)", session.name, session.date)
        return None

    if not steps:
        logger.warning("Session '%s' (%s) has no steps — skipping", session.name, session.date)
        return None

    segment = WorkoutSegment(
        segmentOrder=1,
        sportType=_SPORT_TYPES.get(session.sport, _DEFAULT_SPORT_TYPE),
        workoutSteps=steps,
    )

    workout_cls = _WORKOUT_CLASSES.get(session.sport, _DEFAULT_WORKOUT_CLASS)
    return workout_cls(
        workoutName=session.name,
        sportType=_SPORT_TYPES.get(session.sport, _DEFAULT_SPORT_TYPE),
        estimatedDurationInSecs=int(_total_duration_seconds(steps)),
        workoutSegments=[segment],
        description=session.description or None,
    )
