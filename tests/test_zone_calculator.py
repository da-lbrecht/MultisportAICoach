from itertools import pairwise

from services.garmin.zone_calculator import compute_hr_zones, compute_power_zones


def test_compute_power_zones_boundaries_scale_with_ftp():
    zones = compute_power_zones(200.0)

    assert zones["z1_recovery"] == {
        "name": "Active Recovery", "min_percent": 0, "max_percent": 55,
        "min_watts": 0, "max_watts": 110,
    }
    assert zones["z4_threshold"] == {
        "name": "Threshold", "min_percent": 90, "max_percent": 105,
        "min_watts": 180, "max_watts": 210,
    }


def test_compute_power_zones_top_zone_is_unbounded_but_has_a_display_max():
    zones = compute_power_zones(200.0)

    top = zones["z7_neuromuscular"]
    assert top["min_percent"] == 150
    assert top["max_percent"] == 999
    assert top["min_watts"] == 300
    assert top["max_watts"] > top["min_watts"]  # display value only, not a real ceiling


def test_compute_power_zones_covers_all_seven_zones_contiguously():
    zones = compute_power_zones(250.0)

    assert list(zones.keys()) == [
        "z1_recovery", "z2_endurance", "z3_tempo", "z4_threshold",
        "z5_vo2max", "z6_anaerobic", "z7_neuromuscular",
    ]
    for lower, upper in pairwise(zones.values()):
        assert lower["max_watts"] == upper["min_watts"]  # zone boundaries are shared, not gapped


def test_compute_hr_zones_boundaries_scale_with_lthr():
    zones = compute_hr_zones(170.0)

    assert zones["z1_recovery"]["max_bpm"] == round(170.0 * 0.85)
    assert zones["z5_maximal"]["min_bpm"] == round(170.0 * 0.99)


def test_compute_hr_zones_covers_all_five_zones():
    zones = compute_hr_zones(160.0)
    assert list(zones.keys()) == ["z1_recovery", "z2_aerobic", "z3_tempo", "z4_threshold", "z5_maximal"]
