import yaml

from cli.garmin_ai_coach_cli import ConfigParser


def _write_config(tmp_path, competitions):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"competitions": competitions}), encoding="utf-8")
    return path


def test_get_competitions_forwards_completed_result_and_notes(tmp_path):
    config_path = _write_config(tmp_path, [
        {
            "name": "Cancelled Race",
            "date": "2026-10-03",
            "race_type": "Ultra Cycling",
            "priority": "A",
            "completed": False,
            "result": "Cancelled",
            "notes": "Cancelled by organizers.",
        }
    ])

    competitions = ConfigParser(config_path).get_competitions()

    assert competitions == [{
        "name": "Cancelled Race",
        "date": "2026-10-03",
        "race_type": "Ultra Cycling",
        "priority": "A",
        "target_time": "",
        "completed": False,
        "result": "Cancelled",
        "notes": "Cancelled by organizers.",
    }]


def test_get_competitions_defaults_when_fields_missing(tmp_path):
    config_path = _write_config(tmp_path, [{"name": "Bare Entry"}])

    competitions = ConfigParser(config_path).get_competitions()

    assert competitions[0]["completed"] is False
    assert competitions[0]["result"] == ""
    assert competitions[0]["notes"] == ""
