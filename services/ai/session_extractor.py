import json
import logging

from pydantic import BaseModel, Field

from services.ai.ai_settings import AgentRole
from services.ai.model_config import ModelSelector
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

logger = logging.getLogger(__name__)

# These get a full multi-step breakdown with zone targets. Any other sport (strength,
# mobility, walking, hiking, etc.) still gets pushed, just as a single simple block —
# see workout_builder.py's broader sport-to-workout-class mapping.
STRUCTURED_SPORTS = ("running", "cycling", "swimming")


class SessionStep(BaseModel):
    """One ordered step of a session.

    Warmup, interval, recovery, cooldown, or — for a non-structured sport — the single
    block covering the whole session.
    """

    label: str = Field(..., description="Short step label, e.g. 'Warmup', 'Interval 1', 'Cooldown'")
    repeat: int = Field(1, description="How many times this step repeats back-to-back (e.g. 4 for '4x8min')")
    duration_min: float | None = Field(None, description="Step duration in minutes, if time-based")
    distance_m: float | None = Field(None, description="Step distance in meters, if distance-based")
    zone_key: str | None = Field(
        None,
        description=(
            "The single closest-matching zone key from the athlete's configured zones "
            "(given below), or null for an open/no-target step (always null for swimming "
            "and for any sport outside running/cycling)."
        ),
    )


class PlannedSession(BaseModel):
    """A single pushable training session extracted from the weekly plan."""

    date: str = Field(..., description="ISO date YYYY-MM-DD")
    sport: str = Field(
        ...,
        description=(
            f"One of {', '.join(STRUCTURED_SPORTS)} for a full structured workout, or "
            "'walking', 'hiking', 'strength', 'mobility' (yoga/stretching/mobility work), "
            "or 'other' for a simple single-block entry."
        ),
    )
    name: str = Field(..., description="Short workout name, e.g. 'VO2max Intervals'")
    steps: list[SessionStep] = Field(
        ...,
        description=(
            "Ordered steps. For running/cycling/swimming, break the session into warmup/main-set/"
            "cooldown steps with repeat/zone_key as appropriate. For any other sport, a single step "
            "covering the whole session with your best-effort duration_min is enough."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "The verbatim text describing this session from the plan (e.g. its FOCUS/WORKOUT/PURPOSE/"
            "ADAPTATION lines), copied as-is — do not summarize or paraphrase. This is shown to the "
            "athlete on their Garmin device/app."
        ),
    )


class PlannedSessionList(BaseModel):
    sessions: list[PlannedSession] = Field(
        ..., description="Every day's primary session found in the plan. Skip only pure rest days "
        "with no prescribed activity at all — every other day (including strength/mobility/walking) "
        "gets an entry."
    )


SESSION_EXTRACTOR_SYSTEM_PROMPT = """You convert a written training plan into structured, machine-readable
training sessions suitable for upload as Garmin Connect calendar entries/workouts.
## Principles
- Faithful: Preserve the plan's intent (duration, structure, intensity, and wording) exactly — do not
  invent or embellish. The `description` field must be copied verbatim from the plan.
- Complete: Extract every day's primary session, regardless of discipline. Only pure rest days with no
  prescribed activity at all are skipped.
- Precise: Map each step's intensity to the single closest configured zone key. Never invent a zone key."""

SESSION_EXTRACTOR_USER_PROMPT = """## Task
Extract every day's primary session from the training plan below into the required schema. Preserve
dates exactly as given in the plan.

## Athlete's Configured Zones
Use these exact zone keys when setting `zone_key` on running/cycling steps (or null for open/no-target):

### Power zones (cycling, watts)
```json
{power_zones}
```

### Heart rate zones (running, bpm)
```json
{hr_zones}
```

Swimming, and any sport outside running/cycling, has no configured zones — always set `zone_key` to null.

## Training Plan
```markdown
{weekly_plan}
```

## Rules
- Running/cycling/swimming: break the session into warmup/main-set/cooldown steps as the plan describes.
- Everything else (strength, mobility, walking, hiking, etc.): a single step is enough — set its
  duration_min to your best-effort estimate of the whole session's length from the plan text.
- Skip a day ONLY if it is pure rest with no prescribed activity whatsoever.
- Copy the `description` field verbatim from the plan's text for that day — do not paraphrase.
- Use the plan's "IF TIRED" alternative only if it's the primary prescribed session; otherwise use the main workout.
- Compress repeated intervals with `repeat` (e.g. "4x8min Z4, 4min recovery" -> one step with repeat=4 for the
  work interval, one recovery step with repeat=4), rather than writing out each repetition individually.
"""


def _format_zones(zones: dict[str, dict]) -> str:
    return json.dumps(zones or {}, indent=2)


async def extract_planned_sessions(
    weekly_plan_markdown: str,
    power_zones: dict[str, dict],
    hr_zones: dict[str, dict],
) -> list[PlannedSession]:
    if not weekly_plan_markdown or not weekly_plan_markdown.strip():
        return []

    messages = [
        {"role": "system", "content": SESSION_EXTRACTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SESSION_EXTRACTOR_USER_PROMPT.format(
                power_zones=_format_zones(power_zones),
                hr_zones=_format_zones(hr_zones),
                weekly_plan=weekly_plan_markdown,
            ),
        },
    ]

    llm = ModelSelector.get_llm(AgentRole.WORKOUT).with_structured_output(PlannedSessionList)

    async def call_extraction():
        return await llm.ainvoke(messages)

    result = await retry_with_backoff(call_extraction, AI_ANALYSIS_CONFIG, "Session Extraction")

    return result.sessions
