"""Contract tests for device-side launch-boundary gaps in the timeline."""

import json
from pathlib import Path

import pytest

from src.serving.host_timing import (
    RuntimeGraphSegmentDescriptor,
    load_host_launch_profile,
)
from src.serving.kernel_lowering.execution_timeline import (
    _DeviceLaunchGapTracker,
)


_PROFILES = "configs/host_timing/host_launch_profiles"


def _segment(segment_id):
    return RuntimeGraphSegmentDescriptor(
        segment_id, "main", 0, (("fixture", 0),))


def test_gap_classes_follow_the_launch_boundary_on_one_lane():
    profile = load_host_launch_profile(f"{_PROFILES}/metax_c500.json")
    tracker = _DeviceLaunchGapTracker({"metax_c500": profile})
    lane = ("main", 0)
    first, second = _segment("graph-segment-0"), _segment("graph-segment-1")
    sequence = (None, None, first, first, None, second)
    gaps = [tracker.observe("metax_c500", ((lane, segment),))
            for segment in sequence]

    gap = profile.device_launch_gaps.gap_s
    assert gaps == [
        0.0, gap["eager"], gap["graph_entry"], gap["graph_node"],
        gap["graph_exit"], gap["graph_entry"],
    ]
    # Entry/exit are Graph-launch services; a node inside one replay and an
    # eager dispatch are the cheaper steady-state boundaries.
    assert gap["graph_node"] < gap["eager"] < gap["graph_exit"]
    assert gap["graph_node"] < gap["graph_entry"]


def test_a800_gap_classes_follow_the_launch_boundary():
    profile = load_host_launch_profile(f"{_PROFILES}/nvidia_a800.json")
    tracker = _DeviceLaunchGapTracker({"nvidia_a800": profile})
    lane = ("main", 0)
    first, second = _segment("graph-segment-0"), _segment("graph-segment-1")
    sequence = (None, None, first, first, None, second)
    gaps = [tracker.observe("nvidia_a800", ((lane, segment),))
            for segment in sequence]

    gap = profile.device_launch_gaps.gap_s
    assert gaps == [
        0.0, gap["eager"], gap["graph_entry"], gap["graph_node"],
        gap["graph_exit"], gap["graph_entry"],
    ]
    assert gap["graph_node"] < gap["eager"] < gap["graph_entry"]
    assert gap["graph_node"] < gap["graph_exit"]


def test_lanes_track_their_own_previous_launch():
    profile = load_host_launch_profile(f"{_PROFILES}/metax_c500.json")
    tracker = _DeviceLaunchGapTracker({"metax_c500": profile})
    assert tracker.observe("metax_c500", ((("main", 0), None),)) == 0.0
    assert tracker.observe("metax_c500", ((("main", 1), None),)) == 0.0


@pytest.mark.parametrize("hardware_id", ("ascend_910b",))
def test_profiles_without_gap_evidence_charge_no_device_idle(hardware_id):
    profile = load_host_launch_profile(f"{_PROFILES}/{hardware_id}.json")
    assert profile.device_launch_gaps is None
    tracker = _DeviceLaunchGapTracker({hardware_id: profile})
    lane = ("main", 0)
    for segment in (None, None, _segment("graph-segment-0"), None):
        assert tracker.observe(hardware_id, ((lane, segment),)) == 0.0


def _write_mutated_profile(tmp_path, mutate):
    payload = json.loads(open(f"{_PROFILES}/metax_c500.json").read())
    mutate(payload["device_launch_gap"])
    # Keep the retained measurement paths resolvable from the tmp copy.
    for section in (payload["measurement"], payload["device_launch_gap"][
            "measurement"]):
        section["artifact"] = str(Path(section["artifact"]).resolve())
    payload["source"]["path"] = str(Path(payload["source"]["path"]).resolve())
    (tmp_path / "AGENTS.md").write_text("fixture\n")
    path = tmp_path / "metax_c500.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("mutate, message", (
    (lambda gap: gap["measurement"].update(boundary="host_submit"),
     "boundary"),
    (lambda gap: gap["gap_s"].pop("graph_exit"), "fields differ"),
    (lambda gap: gap["measurement"].update(sha256="0" * 64), "SHA256"),
    (lambda gap: gap["gap_s"].update(eager=0.0), "positive"),
))
def test_gap_section_fails_closed(tmp_path, mutate, message):
    path = _write_mutated_profile(tmp_path, mutate)
    with pytest.raises((ValueError, TypeError), match=message):
        load_host_launch_profile(path)
