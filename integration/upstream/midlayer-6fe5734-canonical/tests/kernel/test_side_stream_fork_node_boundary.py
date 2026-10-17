"""H200 side-stream-fork graph node boundary: HWSpec contract."""

import pytest

from src.kernel.hardware_spec import HWSpec


def test_h200_publishes_the_fork_spacing_increment():
    increment = HWSpec(
        "configs/hardware/nvidia_h200.yaml"
    ).side_stream_fork_graph_node_increment_s()
    # Mean main-chain cadence at fork spacing 33 minus the pure chain.
    assert increment == pytest.approx(217.9e-9, abs=0.1e-9)


def test_cards_without_the_probe_fail_closed():
    with pytest.raises(NotImplementedError, match="side-stream-fork"):
        HWSpec(
            "configs/hardware/nvidia_a800.yaml"
        ).side_stream_fork_graph_node_increment_s()
