import json
from unittest.mock import AsyncMock, patch

import pytest

from services.garmin.models import Activity, ActivitySummary, GarminData


@pytest.mark.asyncio
@patch("services.ai.langgraph.workflows.planning_workflow.run_complete_analysis_and_planning", new_callable=AsyncMock)
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
async def test_cli_e2e_smoke_with_mocks(
    mock_outside_client,
    mock_extractor_class,
    mock_workflow,
    tmp_path,
):
    """Test CLI end-to-end with all external dependencies mocked."""
    # Configure workflow mock
    mock_workflow.return_value = {
        "analysis_html": "<html><body>Analysis OK</body></html>",
        "planning_html": "<html><body>Plan OK</body></html>",
        "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "execution_id": "test-exec",
        "execution_metadata": {"trace_id": "trace-1", "root_run_id": "root-1"},
    }

    # Configure extractor mock
    mock_instance = mock_extractor_class.return_value
    mock_instance.extract_data.return_value = GarminData()

    # Configure outside client mock
    mock_outside_instance = mock_outside_client.return_value
    mock_outside_instance.get_competitions.return_value = []

    # Import after patches are in place
    from cli.garmin_ai_coach_cli import run_analysis_from_config

    output_directory = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test A"
  email: "user@example.com"

context:
  analysis: "Analysis context"
  planning: "Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: false

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    await run_analysis_from_config(config_path)

    analysis_path = output_directory / "analysis.html"
    planning_path = output_directory / "planning.html"
    summary_path = output_directory / "summary.json"
    assert analysis_path.exists()
    assert planning_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["athlete"] == "Test A"
    assert summary["total_cost_usd"] == 0.0


@pytest.mark.asyncio
@patch("services.ai.langgraph.workflows.planning_workflow.run_complete_analysis_and_planning", new_callable=AsyncMock)
@patch("services.garmin.TriathlonCoachDataExtractor")
@patch("services.outside.client.OutsideApiGraphQlClient")
@patch("getpass.getpass", return_value="dummy")
@patch("builtins.input", side_effect=["My goal is to complete a marathon"])
async def test_cli_e2e_with_hitl_enabled(
    mock_input,
    mock_getpass,
    mock_outside_client,
    mock_extractor_class,
    mock_workflow,
    tmp_path,
):
    """Test CLI with HITL enabled to ensure user interactions work."""
    # Configure workflow mock
    mock_workflow.return_value = {
        "analysis_html": "<html><body>Analysis with HITL</body></html>",
        "planning_html": "<html><body>Plan with HITL</body></html>",
       "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.05, "total_tokens": 1000},
        "execution_id": "test-exec-hitl",
        "execution_metadata": {"trace_id": "trace-hitl", "root_run_id": "root-hitl"},
    }

    # Configure extractor mock
    mock_instance = mock_extractor_class.return_value
    mock_instance.extract_data.return_value = GarminData()

    # Configure outside client mock
    mock_outside_instance = mock_outside_client.return_value
    mock_outside_instance.get_competitions.return_value = []

    # Import after patches are in place
    from cli.garmin_ai_coach_cli import run_analysis_from_config

    output_directory = tmp_path / "out_hitl"
    config_path = tmp_path / "config_hitl.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test Athlete HITL"
  email: "user@example.com"

context:
  analysis: "HITL Analysis context"
  planning: "HITL Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: true

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    await run_analysis_from_config(config_path)

    analysis_path = output_directory / "analysis.html"
    planning_path = output_directory / "planning.html"
    summary_path = output_directory / "summary.json"

    assert analysis_path.exists()
    assert planning_path.exists()
    assert summary_path.exists()

    # Verify the basic structure is correct
    assert analysis_path.read_text(encoding="utf-8").startswith("<html>")
    assert planning_path.read_text(encoding="utf-8").startswith("<html>")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["athlete"] == "Test Athlete HITL"
    assert "total_cost_usd" in summary
    assert "total_tokens" in summary


def _make_activity(activity_type: str, activity_id: int, distance: float | None = None) -> Activity:
    return Activity(
        activity_id=activity_id,
        activity_type=activity_type,
        activity_name=f"{activity_type} session",
        start_time="2026-08-01 07:00:00",
        summary=ActivitySummary(distance=distance),
    )


def test_collect_equipment_annotations_filters_and_prompts_only_relevant_activities():
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[
        _make_activity("open_water_swimming", 1, distance=3000),
        _make_activity("road_biking", 2, distance=100000),
        _make_activity("strength_training", 3),  # not swim/bike — must not prompt
    ])

    with patch("builtins.input", side_effect=["Wetsuit, open water", ""]) as mock_input:
        result = collect_equipment_annotations(garmin_data)

    assert mock_input.call_count == 2  # only the swim + bike activities prompt
    assert "EQUIPMENT LOG" in result
    assert "Wetsuit, open water" in result
    assert "strength_training" not in result


def test_collect_equipment_annotations_returns_empty_when_nothing_relevant():
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[_make_activity("strength_training", 1)])

    with patch("builtins.input") as mock_input:
        result = collect_equipment_annotations(garmin_data)

    mock_input.assert_not_called()
    assert result == ""


@pytest.mark.asyncio
@patch("cli.garmin_ai_coach_cli.run_complete_analysis_and_planning", new_callable=AsyncMock)
@patch("cli.garmin_ai_coach_cli.TriathlonCoachDataExtractor")
@patch("cli.garmin_ai_coach_cli.OutsideApiGraphQlClient")
@patch("builtins.input", side_effect=["Neoprene jammers, pool"])
async def test_cli_e2e_equipment_annotation_reaches_contexts(
    mock_input,
    mock_outside_client,
    mock_extractor_class,
    mock_workflow,
    tmp_path,
):
    """Equipment notes collected after import must be merged into both AI contexts."""
    mock_workflow.return_value = {
        "analysis_html": "<html><body>Analysis OK</body></html>",
        "planning_html": "<html><body>Plan OK</body></html>",
        "metrics_outputs": None,
        "activity_outputs": None,
        "physiology_outputs": None,
        "season_plan": {"output": "Season OK"},
        "weekly_plan": {"output": "Weekly OK"},
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "execution_id": "test-exec-equipment",
        "execution_metadata": {"trace_id": "trace-eq", "root_run_id": "root-eq"},
    }

    mock_instance = mock_extractor_class.return_value
    mock_instance.extract_data.return_value = GarminData(
        recent_activities=[_make_activity("pool_swimming", 1, distance=2500)]
    )

    mock_outside_instance = mock_outside_client.return_value
    mock_outside_instance.get_competitions.return_value = []

    from cli.garmin_ai_coach_cli import run_analysis_from_config

    output_directory = tmp_path / "out_equipment"
    config_path = tmp_path / "config_equipment.yaml"
    config_path.write_text(
        f"""
athlete:
  name: "Test Equipment"
  email: "user@example.com"

context:
  analysis: "Analysis context"
  planning: "Planning context"

extraction:
  activities_days: 7
  metrics_days: 14
  ai_mode: "development"
  hitl_enabled: false
  equipment_annotation_enabled: true

output:
  directory: "{output_directory.as_posix()}"

credentials:
  password: "dummy"
""",
        encoding="utf-8",
    )

    await run_analysis_from_config(config_path)

    _, call_kwargs = mock_workflow.call_args
    assert "Neoprene jammers, pool" in call_kwargs["analysis_context"]
    assert "Neoprene jammers, pool" in call_kwargs["planning_context"]
