import json

import pytest

from dataset_pipeline.collective_selection.rule_audit import (
    audit_selection_observations,
)


def _payload(algorithm: str) -> dict[str, object]:
    return {
        "communication_library": "NCCL",
        "hardware_id": "nvidia_a800",
        "library_version": "2.27.3",
        "observations": [
            {
                "algorithm": algorithm,
                "channels": 16,
                "message_bytes": 524288,
                "operation": "all_reduce",
                "protocol": "ll",
                "repeat": 0,
            }
        ],
        "rank_mapping": ["GPU-0", "GPU-1"],
        "topology": "single_node_nvlink",
    }


def test_rule_audit_accepts_matching_normalized_observation(tmp_path):
    path = tmp_path / "selection_observations.json"
    path.write_text(json.dumps(_payload("ring")), encoding="utf-8")

    assert audit_selection_observations((path,)) == {
        "manifest_count": 1,
        "observation_count": 1,
        "unique_input_count": 1,
    }


def test_rule_audit_rejects_selector_mismatch(tmp_path):
    path = tmp_path / "selection_observations.json"
    path.write_text(json.dumps(_payload("tree")), encoding="utf-8")

    with pytest.raises(ValueError, match="selector differs"):
        audit_selection_observations((path,))


def test_rule_audit_rejects_empty_manifest_set():
    with pytest.raises(ValueError, match="at least one manifest"):
        audit_selection_observations(())
