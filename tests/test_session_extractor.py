from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.ai.session_extractor import PlannedSession, PlannedSessionList, SessionStep, extract_planned_sessions


def _mock_llm(response):
    structured = Mock()
    structured.ainvoke = AsyncMock(return_value=response)
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured)
    return llm


@pytest.mark.asyncio
async def test_extract_planned_sessions_returns_parsed_sessions():
    response = PlannedSessionList(sessions=[
        PlannedSession(
            date="2026-08-11", sport="cycling", name="Sweet Spot Ride",
            steps=[SessionStep(label="Main set", duration_min=40, zone_key="z4_threshold")],
            description="Sweet Spot Ride, 40min Z4",
        ),
    ])

    with patch("services.ai.session_extractor.ModelSelector.get_llm", return_value=_mock_llm(response)):
        sessions = await extract_planned_sessions("## Mon\nSweet Spot Ride 40min Z4", {}, {})

    assert len(sessions) == 1
    assert sessions[0].name == "Sweet Spot Ride"


@pytest.mark.asyncio
async def test_extract_planned_sessions_passes_through_every_sport_unfiltered():
    response = PlannedSessionList(sessions=[
        PlannedSession(
            date="2026-08-11", sport="cycling", name="Ride",
            steps=[SessionStep(label="Main", duration_min=30)], description="Ride text",
        ),
        PlannedSession(
            date="2026-08-12", sport="mobility", name="Yoga Flow",
            steps=[SessionStep(label="Flow", duration_min=30)], description="Yoga text",
        ),
    ])

    with patch("services.ai.session_extractor.ModelSelector.get_llm", return_value=_mock_llm(response)):
        sessions = await extract_planned_sessions("plan text", {}, {})

    assert [s.sport for s in sessions] == ["cycling", "mobility"]


@pytest.mark.asyncio
async def test_extract_planned_sessions_short_circuits_on_blank_plan():
    mock_get_llm = Mock()
    with patch("services.ai.session_extractor.ModelSelector.get_llm", mock_get_llm):
        sessions = await extract_planned_sessions("   ", {}, {})

    assert sessions == []
    mock_get_llm.assert_not_called()
