import pytest


@pytest.fixture(autouse=True)
def _isolate_working_directory(tmp_path, monkeypatch):
    """Run every test from its own temp dir.

    Plan, notes and scheduled-workout stores default to the relative path ``data/storage``,
    so a test that exercises the CLI from the repo root would otherwise overwrite the
    athlete's real saved plans.
    """
    monkeypatch.chdir(tmp_path)
