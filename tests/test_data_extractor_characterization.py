
from datetime import date
from typing import Any
from unittest.mock import Mock, patch

import pytest

from services.garmin.data_extractor import DataExtractor, TriathlonCoachDataExtractor
from services.garmin.models import Activity, ActivitySummary, ExtractionConfig, Gear, TrainingThresholds, UserProfile


class TestDataExtractorCharacterization:

    def test_safe_divide_and_round_normal_case(self):
        result = DataExtractor.safe_divide_and_round(100.0, 3.0, 2)
        assert result == 33.33

    def test_safe_divide_and_round_none_input(self):
        result = DataExtractor.safe_divide_and_round(None, 3.0, 2)
        assert result is None

    def test_extract_start_time_from_summary(self):
        activity_data = {
            "summaryDTO": {
                "startTimeLocal": "2025-01-01T10:00:00"
            }
        }
        result = DataExtractor.extract_start_time(activity_data)
        assert result == "2025-01-01T10:00:00"

    def test_extract_start_time_fallback_chain(self):
        activity_data = {
            "startTimeLocal": "2025-01-01T11:00:00"
        }
        result = DataExtractor.extract_start_time(activity_data)
        assert result == "2025-01-01T11:00:00"

    def test_extract_activity_type_from_nested_dto(self):
        activity_data = {
            "activityType": {
                "typeKey": "cycling"
            }
        }
        result = DataExtractor.extract_activity_type(activity_data)
        assert result == "cycling"

    def test_extract_activity_type_fallback_unknown(self):
        activity_data: dict[str, Any] = {}
        result = DataExtractor.extract_activity_type(activity_data)
        assert result == "unknown"

    def test_convert_lactate_threshold_speed_conversion(self):
        result = DataExtractor.convert_lactate_threshold_speed(10.0)
        assert result == 100.0

    def test_convert_lactate_threshold_speed_none_input(self):
        result = DataExtractor.convert_lactate_threshold_speed(None)
        assert result is None

    def test_get_date_ranges_calculation(self):
        ranges = DataExtractor.get_date_ranges(ExtractionConfig(activities_range=21, metrics_range=56))

        assert "activities" in ranges
        assert "metrics" in ranges
        assert isinstance(ranges["activities"]["start"], date)
        assert isinstance(ranges["activities"]["end"], date)


class TestTriathlonCoachDataExtractorCharacterization:

    @patch("services.garmin.data_extractor.GarminConnectClient")
    def test_initialization_connects_to_garmin(self, mock_client):
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        TriathlonCoachDataExtractor("test@example.com", "password")

        mock_client.assert_called_once()
        mock_instance.connect.assert_called_once_with("test@example.com", "password")

    @patch("services.garmin.data_extractor.GarminConnectClient")
    def test_extract_data_base_data_always_included(self, mock_client):
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")

        mock_instance.client.get_user_profile.return_value = {
            "userData": {"gender": "male", "weight": 70000},
            "userSleep": {"sleepTime": "22:00"}
        }
        mock_instance.client.get_stats.return_value = {
            "calendarDate": "2025-01-01",
            "totalSteps": 10000
        }
        mock_instance.client.get_sleep_data.return_value = {
            "dailySleepDTO": {"sleepTimeSeconds": 28800}
        }

        result = extractor.extract_data(ExtractionConfig(include_detailed_activities=False, include_metrics=False))

        assert hasattr(result, "user_profile")
        assert hasattr(result, "daily_stats")
        assert result.user_profile is not None
        assert result.user_profile.gender == "male"
        assert result.daily_stats is not None
        assert result.daily_stats.total_steps == 10000

    def test_activity_summary_extraction_structure(self):
        result = TriathlonCoachDataExtractor.__new__(TriathlonCoachDataExtractor)._extract_activity_summary({
            "distance": 10000.0,
            "duration": 3600,
            "averageSpeed": 2.78,
            "maxSpeed": 5.0,
            "calories": 400,
            "averageHR": 150,
            "maxHR": 180,
            "avgPower": 250,
            "maxPower": 400
        })

        assert isinstance(result, ActivitySummary)
        assert result.distance == 10000.0
        assert result.duration == 3600
        assert result.avg_power == 250
        assert result.max_power == 400

    def test_weather_data_extraction_none_input(self):
        extractor = TriathlonCoachDataExtractor.__new__(TriathlonCoachDataExtractor)

        result = extractor._extract_weather_data(None)

        assert result.temp is None
        assert result.apparent_temp is None
        assert result.relative_humidity is None
        assert result.wind_speed is None
        assert result.weather_type is None


@pytest.fixture
def mock_garmin_client():
    with patch("services.garmin.data_extractor.GarminConnectClient") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


class TestDataExtractorIntegrationBehavior:

    def test_multisport_activity_processing_structure(self, mock_garmin_client):
        mock_garmin_client.client.get_activity.side_effect = [
            {"activityId": 12346, "activityName": "Swim Leg", "startTimeLocal": "2025-01-01T06:00:00", "summaryDTO": {"distance": 1500.0, "duration": 1800}},
            {"activityId": 12347, "activityName": "Bike Leg", "startTimeLocal": "2025-01-01T06:30:00", "summaryDTO": {"distance": 40000.0, "duration": 3600, "avgPower": 200}},
            {"activityId": 12348, "activityName": "Run Leg", "startTimeLocal": "2025-01-01T07:30:00", "summaryDTO": {"distance": 10000.0, "duration": 2400}}
        ]
        mock_garmin_client.client.get_activity_details.return_value = {}
        mock_garmin_client.client.get_activity_weather.return_value = None

        result = TriathlonCoachDataExtractor("test@example.com", "password")._process_multisport_activity({
            "activityId": 12345,
            "activityName": "Morning Triathlon",
            "isMultiSportParent": True,
            "startTimeLocal": "2025-01-01T06:00:00",
            "summaryDTO": {"distance": 15000.0, "duration": 5400},
            "metadataDTO": {"childIds": [12346, 12347, 12348], "childActivityTypes": ["swimming", "cycling", "running"]}
        })

        assert isinstance(result, Activity)
        assert result.activity_type == "multisport"
        assert result.laps is not None
        assert len(result.laps) == 3
        assert result.laps[0]["activityType"] == "swimming"
        assert result.laps[1]["activityType"] == "cycling"
        assert result.laps[2]["activityType"] == "running"

    def test_cycling_power_data_extraction_priority(self, mock_garmin_client):
        # We need an instance with initialized client for _process_single_sport_activity
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")

        # Mock API calls made by _process_single_sport_activity
        mock_garmin_client.client.get_activity_details.return_value = {}
        mock_garmin_client.client.get_activity_weather.return_value = None

        result = extractor._process_single_sport_activity({
            "activityId": 123,
            "startTimeLocal": "2025-01-01T10:00:00",
            "activityType": {"typeKey": "cycling"},
            "summaryDTO": {"avgPower": 250, "normPower": 260},
            "averagePower": 240,
            "normalizedPower": 250
        })

        assert result is not None
        assert result.summary is not None
        assert result.summary.avg_power == 250
        assert result.summary.normalized_power == 260

    def test_gear_attached_for_relevant_activity_type(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_activity_details.return_value = {}
        mock_garmin_client.client.get_activity_weather.return_value = None
        mock_garmin_client.client.get_activity_gear.return_value = [
            {"uuid": "g1", "displayName": "Canyon Endurace", "gearTypeName": "Bike"}
        ]

        result = extractor._process_single_sport_activity({
            "activityId": 123,
            "startTimeLocal": "2025-01-01T10:00:00",
            "activityType": {"typeKey": "cycling"},
            "summaryDTO": {},
        })

        assert result is not None
        assert result.gear == [Gear(uuid="g1", display_name="Canyon Endurace", gear_type="Bike")]
        mock_garmin_client.client.get_activity_gear.assert_called_once_with(123)

    def test_gear_not_fetched_for_irrelevant_activity_type(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_activity_details.return_value = {}
        mock_garmin_client.client.get_activity_weather.return_value = None

        result = extractor._process_single_sport_activity({
            "activityId": 124,
            "startTimeLocal": "2025-01-01T10:00:00",
            "activityType": {"typeKey": "strength_training"},
            "summaryDTO": {},
        })

        assert result is not None
        assert result.gear == []
        mock_garmin_client.client.get_activity_gear.assert_not_called()

    def test_multisport_gear_aggregated_and_deduplicated(self, mock_garmin_client):
        mock_garmin_client.client.get_activity.side_effect = [
            {"activityId": 12346, "activityName": "Bike Leg", "startTimeLocal": "2025-01-01T06:00:00", "summaryDTO": {}},
            {"activityId": 12347, "activityName": "Run Leg", "startTimeLocal": "2025-01-01T07:00:00", "summaryDTO": {}},
        ]
        mock_garmin_client.client.get_activity_details.return_value = {}
        mock_garmin_client.client.get_activity_weather.return_value = None
        mock_garmin_client.client.get_activity_gear.side_effect = [
            [{"uuid": "g1", "displayName": "Canyon Endurace", "gearTypeName": "Bike"}],
            [{"uuid": "g2", "displayName": "Nike Pegasus 40", "gearTypeName": "Shoes"}],
        ]

        result = TriathlonCoachDataExtractor("test@example.com", "password")._process_multisport_activity({
            "activityId": 12345,
            "activityName": "Duathlon",
            "isMultiSportParent": True,
            "startTimeLocal": "2025-01-01T06:00:00",
            "summaryDTO": {},
            "metadataDTO": {"childIds": [12346, 12347], "childActivityTypes": ["cycling", "running"]},
        })

        assert result is not None
        assert result.gear is not None
        assert {g.display_name for g in result.gear} == {"Canyon Endurace", "Nike Pegasus 40"}


class TestGetActivityGear:

    def test_parses_list_response(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_activity_gear.return_value = [
            {"uuid": "g1", "displayName": "Canyon Endurace", "gearTypeName": "Bike"},
            {"uuid": "g2", "customMakeModel": "Nike Pegasus 40", "gearTypeDto": {"displayName": "Shoes"}},
        ]

        result = extractor.get_activity_gear(123)

        assert result == [
            Gear(uuid="g1", display_name="Canyon Endurace", gear_type="Bike"),
            Gear(uuid="g2", display_name="Nike Pegasus 40", gear_type="Shoes", make_model="Nike Pegasus 40"),
        ]

    def test_parses_dict_wrapped_response(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_activity_gear.return_value = {
            "gear": [{"uuid": "g1", "displayName": "Canyon Endurace", "gearTypeName": "Bike"}]
        }

        result = extractor.get_activity_gear(123)

        assert result == [Gear(uuid="g1", display_name="Canyon Endurace", gear_type="Bike")]

    def test_returns_empty_list_for_unexpected_response(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_activity_gear.return_value = None

        assert extractor.get_activity_gear(123) == []


class TestGetTrainingThresholds:

    def test_extracts_ftp_and_reuses_given_profile_lthr(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_cycling_ftp.return_value = {"functionalThresholdPower": 271}
        profile = UserProfile(lactate_threshold_heart_rate=165, lactate_threshold_speed=3.5)

        result = extractor.get_training_thresholds(profile)

        assert result == TrainingThresholds(
            ftp_watts=271.0, lactate_threshold_hr=165, lactate_threshold_speed_ms=3.5
        )
        mock_garmin_client.client.get_user_profile.assert_not_called()

    def test_handles_list_shaped_ftp_response(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_cycling_ftp.return_value = [{"functionalThresholdPower": 250}]

        result = extractor.get_training_thresholds(UserProfile())

        assert result.ftp_watts == 250.0

    def test_missing_ftp_field_falls_back_to_none(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_cycling_ftp.return_value = {"unexpected": "shape"}

        result = extractor.get_training_thresholds(UserProfile())

        assert result.ftp_watts is None

    def test_fetches_profile_itself_when_none_provided(self, mock_garmin_client):
        extractor = TriathlonCoachDataExtractor("test@example.com", "password")
        mock_garmin_client.client.get_cycling_ftp.return_value = {"functionalThresholdPower": 271}
        mock_garmin_client.client.get_user_profile.return_value = {
            "userData": {"lactateThresholdHeartRate": 160},
        }

        result = extractor.get_training_thresholds()

        assert result.lactate_threshold_hr == 160
        mock_garmin_client.client.get_user_profile.assert_called_once()
