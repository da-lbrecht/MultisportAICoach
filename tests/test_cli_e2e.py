import json
from unittest.mock import AsyncMock, patch

import pytest

from services.garmin.models import Activity, ActivitySummary, GarminData, Gear


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


def _make_activity(
    activity_type: str,
    activity_id: int,
    distance: float | None = None,
    gear: list[Gear] | None = None,
) -> Activity:
    return Activity(
        activity_id=activity_id,
        activity_type=activity_type,
        activity_name=f"{activity_type} session",
        start_time="2026-08-01 07:00:00",
        summary=ActivitySummary(distance=distance),
        gear=gear,
    )


def _make_notes_store(tmp_path):
    from services.ai.utils.activity_notes_store import ActivityNotesStore

    return ActivityNotesStore(base_dir=str(tmp_path / "notes_storage"))


def test_collect_equipment_annotations_prompts_every_activity_including_non_swim_bike(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[
        _make_activity("open_water_swimming", 1, distance=3000),
        _make_activity("road_biking", 2, distance=100000),
        _make_activity("strength_training", 3),  # no longer excluded — must prompt too
    ])

    with patch("builtins.input", side_effect=["Wetsuit, open water", "", "Resistance bands"]) as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    assert mock_input.call_count == 3  # every imported activity prompts
    assert "SESSION NOTES" in result
    assert "Wetsuit, open water" in result
    assert "Resistance bands" in result


def test_collect_equipment_annotations_returns_empty_when_no_activities(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[])

    with patch("builtins.input") as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    mock_input.assert_not_called()
    assert result == ""


def test_collect_equipment_annotations_skips_activity_when_note_left_blank(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[_make_activity("strength_training", 1)])

    with patch("builtins.input", return_value="") as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    mock_input.assert_called_once()  # still prompted, just skipped when left blank
    assert result == ""


def test_collect_equipment_annotations_gear_not_fetched_for_strength_still_prompts(tmp_path):
    # Gear extraction is gated to swim/bike/run activity types (data_extractor.py),
    # but the annotation prompt itself must not be gated the same way.
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[
        _make_activity("strength_training", 1, gear=None),
    ])

    with patch("builtins.input", return_value="Dumbbells + bands") as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    mock_input.assert_called_once()
    assert "Dumbbells + bands" in result
    assert "Garmin gear" not in result  # no gear was ever fetched for this activity type


def test_collect_equipment_annotations_running_gear_shown_as_default(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[
        _make_activity(
            "running", 1, distance=10000,
            gear=[Gear(uuid="g2", display_name="Nike Pegasus 40", gear_type="Shoes")],
        ),
    ])

    with patch("builtins.input", return_value="") as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    mock_input.assert_called_once()  # every activity prompts now, including running
    prompt_text = mock_input.call_args.args[0]
    assert "Nike Pegasus 40 (Shoes)" in prompt_text  # shown as default in the prompt
    assert "Garmin gear — Nike Pegasus 40 (Shoes)" in result  # kept as-is on blank Enter


def test_collect_equipment_annotations_offers_garmin_gear_as_editable_default(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[
        _make_activity(
            "road_biking", 1, distance=100000,
            gear=[Gear(uuid="g1", display_name="Canyon Endurace", gear_type="Bike")],
        ),
    ])

    with patch("builtins.input", return_value="") as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    mock_input.assert_called_once()  # bike still prompts, gear shown as default
    prompt_text = mock_input.call_args.args[0]
    assert "Canyon Endurace (Bike)" in prompt_text  # default shown in the prompt itself
    assert "Garmin gear — Canyon Endurace (Bike)" in result  # accepting default keeps it as-is


def test_collect_equipment_annotations_appends_typed_notes_to_garmin_default(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    garmin_data = GarminData(recent_activities=[
        _make_activity(
            "road_biking", 1, distance=100000,
            gear=[Gear(uuid="g1", display_name="Canyon Endurace", gear_type="Bike")],
        ),
        _make_activity("open_water_swimming", 2, distance=3000),  # no Garmin gear for swim
    ])

    with patch("builtins.input", side_effect=["new tires fitted", "Wetsuit, open water"]) as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=_make_notes_store(tmp_path))

    assert mock_input.call_count == 2  # both swim/bike sessions prompt
    assert "Garmin gear — Canyon Endurace (Bike); new tires fitted" in result
    assert "Wetsuit, open water" in result


def test_collect_equipment_annotations_prefills_note_from_previous_run(tmp_path):
    """The whole point: an activity seen in a prior CLI run must offer its note back."""
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    store = _make_notes_store(tmp_path)
    garmin_data = GarminData(recent_activities=[
        _make_activity("strength_training", 42),
    ])

    # First "run" — athlete types a note, which gets persisted.
    with patch("builtins.input", return_value="Dumbbells + resistance bands"):
        first_result = collect_equipment_annotations(garmin_data, notes_store=store)
    assert "Dumbbells + resistance bands" in first_result

    # Second "run" — same activity_id reappears (still inside the import window).
    with patch("builtins.input", return_value="") as mock_input:
        second_result = collect_equipment_annotations(garmin_data, notes_store=store)

    prompt_text = mock_input.call_args.args[0]
    assert "Dumbbells + resistance bands" in prompt_text  # offered as the default
    assert "Dumbbells + resistance bands" in second_result  # kept as-is via blank Enter


def test_collect_equipment_annotations_typed_note_replaces_previous_note(tmp_path):
    from cli.garmin_ai_coach_cli import collect_equipment_annotations

    store = _make_notes_store(tmp_path)
    garmin_data = GarminData(recent_activities=[_make_activity("strength_training", 42)])

    with patch("builtins.input", return_value="Dumbbells + resistance bands"):
        collect_equipment_annotations(garmin_data, notes_store=store)

    with patch("builtins.input", return_value="Bodyweight only, no equipment") as mock_input:
        result = collect_equipment_annotations(garmin_data, notes_store=store)

    mock_input.assert_called_once()
    assert "Bodyweight only, no equipment" in result
    assert "Dumbbells + resistance bands" not in result  # replaced, not appended

    # And the replacement itself is now what gets persisted for next time.
    assert store.load("cli_user")["42"]["note"] == "Bodyweight only, no equipment"


def test_activity_notes_store_round_trip(tmp_path):
    from services.ai.utils.activity_notes_store import ActivityNotesStore

    store = ActivityNotesStore(base_dir=str(tmp_path))
    assert store.load("cli_user") == {}

    store.save("cli_user", {"123": {"note": "new tires fitted", "date": "2026-08-01", "sport": "road_biking"}})

    reloaded = ActivityNotesStore(base_dir=str(tmp_path))
    assert reloaded.load("cli_user") == {
        "123": {"note": "new tires fitted", "date": "2026-08-01", "sport": "road_biking"}
    }


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
    request,
):
    """Equipment notes collected after import must be merged into both AI contexts."""
    from services.ai.utils.activity_notes_store import ActivityNotesStore

    notes_dir = tmp_path / "notes_storage"
    activity_notes_patch = patch(
        "cli.garmin_ai_coach_cli.ActivityNotesStore",
        return_value=ActivityNotesStore(base_dir=str(notes_dir)),
    )
    activity_notes_patch.start()
    request.addfinalizer(activity_notes_patch.stop)

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
