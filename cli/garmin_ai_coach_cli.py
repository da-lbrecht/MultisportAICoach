#!/usr/bin/env python3

import argparse
import asyncio
import getpass
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from core.config import reload_config
from services.ai.ai_settings import ai_settings
from services.ai.langgraph.workflows.planning_workflow import (
    run_complete_analysis_and_planning,
)
from services.ai.utils.activity_notes_store import ActivityNotesStore
from services.ai.utils.plan_storage import FilePlanStorage
from services.garmin import Activity, ExtractionConfig, GarminData, TriathlonCoachDataExtractor
from services.outside.client import OutsideApiGraphQlClient

sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ConfigParser:

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        content = self.config_path.read_text(encoding="utf-8")

        if self.config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(content)
        elif self.config_path.suffix == ".json":
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")

    def get_athlete_info(self) -> tuple[str, str]:
        if not (email := self.config.get("athlete", {}).get("email")):
            raise ValueError("Athlete email is required in config file")

        return self.config.get("athlete", {}).get("name", "Athlete"), email

    def get_contexts(self) -> tuple[str, str]:
        return (
            self.config.get("context", {}).get("analysis", "").strip(),
            self.config.get("context", {}).get("planning", "").strip()
        )

    def get_extraction_config(self) -> dict[str, Any]:
        extraction = self.config.get("extraction", {})
        return {
            "activities_days": extraction.get("activities_days", 7),
            "metrics_days": extraction.get("metrics_days", 14),
            "ai_mode": extraction.get("ai_mode", "development"),
            "enable_plotting": extraction.get("enable_plotting", False),
            "hitl_enabled": extraction.get("hitl_enabled", True),
            "skip_synthesis": extraction.get("skip_synthesis", False),
            "include_long_term_trends": extraction.get("include_long_term_trends", True),
            "long_term_range": extraction.get("long_term_range", 360),
            "long_term_interval": extraction.get("long_term_interval", 7),
            "equipment_annotation_enabled": extraction.get("equipment_annotation_enabled", False),
        }

    def get_competitions(self) -> list[dict[str, Any]]:
        competitions = self.config.get("competitions", [])
        return [
            {
                "name": comp.get("name", ""),
                "date": comp.get("date", ""),
                "race_type": comp.get("race_type", ""),
                "priority": comp.get("priority", "B"),
                "target_time": comp.get("target_time", ""),
            }
            for comp in competitions
        ]

    def get_output_directory(self) -> Path:
        return Path(self.config.get("output", {}).get("directory", "./data"))

    def get_password(self) -> str:
        return (
            self.config.get("credentials", {}).get("password", "") or
            getpass.getpass("Enter Garmin Connect password: ")
        )


def fetch_outside_competitions_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    client = OutsideApiGraphQlClient()

    if isinstance(outside_cfg := config.get("outside"), dict) and any(
        isinstance(value, list) for value in outside_cfg.values()
    ):
        return client.get_competitions(outside_cfg)

    aggregate: list[dict[str, Any]] = []

    if isinstance(legacy_bikereg := config.get("bikereg", []), list) and legacy_bikereg:
        aggregate.extend(client.get_competitions(legacy_bikereg))

    if legacy_all := {
        key: entries
        for key in ("runreg", "trireg", "skireg")
        if isinstance(entries := config.get(key, []), list) and entries
    }:
        aggregate.extend(client.get_competitions(legacy_all))

    return aggregate


def _describe_activity(activity: Activity) -> str:
    date = (activity.start_time or "")[:10]
    sport = activity.activity_type or "activity"
    name = activity.activity_name or sport
    distance_km = (
        f"{activity.summary.distance / 1000:.1f} km"
        if activity.summary and activity.summary.distance
        else ""
    )
    return f"{date}  {sport:<14} {name} {distance_km}".strip()


def _describe_garmin_gear(activity: Activity) -> str:
    if not activity.gear:
        return ""
    return ", ".join(
        (item.display_name or "unknown gear") + (f" ({item.gear_type})" if item.gear_type else "")
        for item in activity.gear
    )


def _render_equipment_log(entries: list[dict[str, str]]) -> str:
    if not entries:
        return ""

    lines = [
        "SESSION NOTES (bikes/shoes tagged in Garmin are read automatically as a gear "
        "default; anything else Garmin can't capture — wetsuit vs jammers, open water "
        "vs pool, indoor vs outdoor, terrain, equipment used, etc — is provided directly "
        "by the athlete for this run; use it to correctly interpret each session and to "
        "prescribe context-appropriate targets):"
    ]
    lines.extend(f"- {entry['date']} ({entry['sport']}): {entry['notes']}" for entry in entries)

    return "\n".join(lines)


def collect_equipment_annotations(
    garmin_data: GarminData,
    user_id: str = "cli_user",
    notes_store: ActivityNotesStore | None = None,
) -> str:
    activities = garmin_data.recent_activities or []

    if not activities:
        return ""

    notes_store = notes_store or ActivityNotesStore()
    previous_notes = notes_store.load(user_id)

    print(f"\n{'='*60}")
    print(f"SESSION NOTES — {len(activities)} session(s) imported from Garmin")
    print("Where Garmin has gear tagged, or you noted this exact session in a previous")
    print("run, it's shown as a default — press Enter to keep it, or type to replace it.")
    print("Enter alone with no default skips a session entirely.")
    print(f"{'='*60}")

    entries: list[dict[str, str]] = []
    updated_notes: dict[str, dict[str, str]] = {}
    for activity in activities:
        activity_key = str(activity.activity_id) if activity.activity_id is not None else None
        previous_note = previous_notes.get(activity_key, {}).get("note", "") if activity_key else ""

        garmin_gear = _describe_garmin_gear(activity)
        hint_parts = [part for part in (garmin_gear, previous_note) if part]
        default_hint = f" [{'; '.join(hint_parts)}]" if hint_parts else ""

        typed = input(f"  {_describe_activity(activity)}\n    Notes{default_hint}: ").strip()
        effective_note = typed or previous_note  # Enter keeps the prior note; typing replaces it.

        if garmin_gear:
            combined = f"Garmin gear — {garmin_gear}" + (f"; {effective_note}" if effective_note else "")
        else:
            combined = effective_note

        if combined:
            entries.append({
                "date": (activity.start_time or "")[:10],
                "sport": activity.activity_type or "activity",
                "notes": combined,
            })

        if activity_key and effective_note:
            updated_notes[activity_key] = {
                "note": effective_note,
                "date": (activity.start_time or "")[:10],
                "sport": activity.activity_type or "activity",
            }

    print(f"{'='*60}\n")

    if updated_notes:
        notes_store.save(user_id, {**previous_notes, **updated_notes})

    return _render_equipment_log(entries)


def _save_html_outputs(output_dir: Path, result: dict[str, Any]) -> list[str]:
    files_generated: list[str] = []

    for filename, key in [
        ("analysis.html", "analysis_html"),
        ("planning.html", "planning_html"),
    ]:
        if content := result.get(key):
            if isinstance(content, dict):
                content = content.get("content", "")

            output_path = output_dir / filename
            output_path.write_text(content, encoding="utf-8")
            files_generated.append(filename)
            logger.info("Saved: %s", output_path)

    return files_generated


def _save_expert_outputs(output_dir: Path, result: dict[str, Any]) -> list[str]:
    files_generated: list[str] = []

    for filename, key in [
        ("metrics_expert.json", "metrics_outputs"),
        ("activity_expert.json", "activity_outputs"),
        ("physiology_expert.json", "physiology_outputs"),
    ]:
        if output := result.get(key):
            output_path = output_dir / filename
            output_path.write_text(
                json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            files_generated.append(filename)
            logger.info("Saved: %s", output_path)

    return files_generated


def _save_plan_outputs(output_dir: Path, result: dict[str, Any]) -> list[str]:
    files_generated: list[str] = []

    storage = FilePlanStorage()
    user_id = result.get("user_id", "cli_user")

    for filename, key in [
        ("season_plan.md", "season_plan"),
        ("weekly_plan.md", "weekly_plan"),
    ]:
        if plan_dict := result.get(key):
            output = plan_dict.get("output", plan_dict) if isinstance(plan_dict, dict) else plan_dict
            if isinstance(output, str):
                output_path = output_dir / filename
                output_path.write_text(output, encoding="utf-8")
                files_generated.append(filename)
                logger.info("Saved: %s", output_path)
                storage.save_plan(user_id, key, output)

    return files_generated


async def run_analysis_from_config(config_path: Path, output_dir_override: Path | None = None) -> None:
    config_parser = ConfigParser(config_path)
    athlete_name, email = config_parser.get_athlete_info()
    analysis_context, planning_context = config_parser.get_contexts()
    extraction_settings = config_parser.get_extraction_config()

    competitions = config_parser.get_competitions()
    outside_competitions = fetch_outside_competitions_from_config(config_parser.config)
    if outside_competitions:
        competitions.extend(outside_competitions)

    output_dir = output_dir_override or config_parser.get_output_directory()

    logger.info("Starting analysis for %s", athlete_name)
    logger.info("Output directory: %s", output_dir)

    password = config_parser.get_password()

    os.environ["AI_MODE"] = extraction_settings.get("ai_mode", "development")

    # Reload config and settings to pick up the new AI_MODE
    reload_config()
    ai_settings.reload()

    logger.info("AI Mode: %s", os.environ["AI_MODE"])


    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Extracting Garmin Connect data...")
        extractor = TriathlonCoachDataExtractor(email, password)

        extraction_config = ExtractionConfig(
            activities_range=extraction_settings["activities_days"],
            metrics_range=extraction_settings["metrics_days"],
            include_detailed_activities=True,
            include_metrics=True,
            include_long_term_trends=extraction_settings["include_long_term_trends"],
            long_term_range=extraction_settings["long_term_range"],
            long_term_interval=extraction_settings["long_term_interval"],
        )

        garmin_data = extractor.extract_data(extraction_config)
        logger.info("Data extraction completed")

        if extraction_settings.get("equipment_annotation_enabled", False):
            if equipment_log := collect_equipment_annotations(garmin_data):
                analysis_context = f"{analysis_context}\n\n{equipment_log}".strip()
                planning_context = f"{planning_context}\n\n{equipment_log}".strip()

        now = datetime.now()
        plotting_enabled = extraction_settings.get("enable_plotting", False)
        hitl_enabled = extraction_settings.get("hitl_enabled", True)
        skip_synthesis = extraction_settings.get("skip_synthesis", False)

        logger.info("Plotting enabled: %s", plotting_enabled)
        logger.info("HITL enabled: %s", hitl_enabled)
        logger.info("Skip synthesis: %s", skip_synthesis)

        current_date = {"date": now.strftime("%Y-%m-%d"), "day_name": now.strftime("%A")}
        week_dates = [
            {"date": (now + timedelta(days=offset)).strftime("%Y-%m-%d"),
             "day_name": (now + timedelta(days=offset)).strftime("%A")}
            for offset in range(14)
        ]

        logger.info("Running AI analysis and planning...")

        result = await run_complete_analysis_and_planning(
            user_id="cli_user",
            athlete_name=athlete_name,
            garmin_data=asdict(garmin_data),
            analysis_context=analysis_context,
            planning_context=planning_context,
            competitions=competitions,
            current_date=current_date,
            week_dates=week_dates,
            plotting_enabled=plotting_enabled,
            hitl_enabled=hitl_enabled,
            skip_synthesis=skip_synthesis,
        )

        logger.info("Saving results...")

        files_generated: list[str] = []
        files_generated.extend(_save_html_outputs(output_dir, result))
        files_generated.extend(_save_expert_outputs(output_dir, result))
        files_generated.extend(_save_plan_outputs(output_dir, result))

        cost_total = float(
            result.get("cost_summary", {}).get("total_cost_usd", 0.0) or
            result.get("execution_metadata", {}).get("total_cost_usd", 0.0) or
            sum(cost.get("total_cost", 0) for cost in result.get("costs", []))
        )
        total_tokens = int(
            result.get("cost_summary", {}).get("total_tokens", 0) or
            result.get("execution_metadata", {}).get("total_tokens", 0)
        )

        (output_dir / "summary.json").write_text(
            json.dumps({
                "athlete": athlete_name,
                "analysis_date": datetime.now().isoformat(),
                "competitions": competitions,
                "total_cost_usd": cost_total,
                "total_tokens": total_tokens,
                "execution_id": result.get("execution_id", ""),
                "trace_id": result.get("execution_metadata", {}).get("trace_id", ""),
                "root_run_id": result.get("execution_metadata", {}).get("root_run_id", ""),
                "files_generated": files_generated,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        logger.info("✅ Analysis completed successfully!")
        if outside_competitions:
            logger.info("✅  Added %d Outside competitions from config", len(outside_competitions))
        logger.info("📁 Results saved to: %s", output_dir)
        logger.info("💰 Total cost: $%.2f (%d tokens)", cost_total, total_tokens)
    except Exception as e:
        logger.error("❌ Analysis failed: %s", e)
        raise


def create_config_template(output_path: Path) -> None:
    template_path = Path(__file__).parent / "coach_config_template.yaml"

    if template_path.exists():
        output_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("✅ Config template created: %s", output_path)
        logger.info("Edit this file with your settings and run analysis with --config")
    else:
        logger.error("❌ Template file not found")


def main():
    parser = argparse.ArgumentParser(
        description="Garmin AI Coach CLI - AI Triathlon Coach",
        epilog="Example: python garmin_ai_coach_cli.py --config my_config.yaml",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to configuration file (YAML or JSON)")
    group.add_argument("--init-config", type=Path, help="Create a configuration template file")

    parser.add_argument("--output-dir", type=Path, help="Override output directory from config")

    args = parser.parse_args()

    if args.init_config:
        create_config_template(args.init_config)
        return

    if args.config:
        try:
            asyncio.run(run_analysis_from_config(args.config, args.output_dir))
        except KeyboardInterrupt:
            logger.info("❌ Analysis cancelled by user")
        except Exception as e:
            logger.error("❌ Analysis failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
