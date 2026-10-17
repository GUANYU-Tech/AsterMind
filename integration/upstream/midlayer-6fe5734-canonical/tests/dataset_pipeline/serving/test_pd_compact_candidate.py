import json

from dataset_pipeline.serving_dataset import pd_compact_candidate


def test_pd_candidate_requires_complete_passing_observer_report(tmp_path):
    report_path = tmp_path / "observer_effect.json"
    report_path.write_text(json.dumps({
        "status": "inherited_from_representative_case",
        "validation_case_id": "nvidia_a800/qwen3-8b/tp1_qps4",
        "baseline_run_id": "trace-off",
        "instrumented_run_id": "trace-on",
        "threshold_percent": 3.0,
        "metric_delta_percent": {
            "request_throughput": 0.1,
            "output_token_throughput": 0.1,
            "ttft": 0.1,
            "tpot": 0.1,
            "e2e": 0.1,
        },
        "passed": True,
    }))

    report = pd_compact_candidate._load_observer_effect_report(report_path)

    assert report["passed"] is True


def test_pd_candidate_accepts_one_complete_aggregate_per_decode_rank():
    def event(*, event_name, process_id, timestamp_ns):
        return {
            "connector": "P2pNcclConnector",
            "instance_role": "decode",
            "process_pid": process_id,
            "timestamp_ns": timestamp_ns,
        }

    transfer_starts = (
        event(event_name="transfer_start", process_id=101, timestamp_ns=10),
        event(event_name="transfer_start", process_id=102, timestamp_ns=11),
    )
    transfer_ends = (
        event(event_name="transfer_end", process_id=101, timestamp_ns=20),
        event(event_name="transfer_end", process_id=102, timestamp_ns=21),
    )
    load_starts = (
        event(event_name="consumer_load_start", process_id=101, timestamp_ns=12),
        event(event_name="consumer_load_start", process_id=102, timestamp_ns=13),
    )
    load_ends = (
        event(event_name="consumer_load_end", process_id=101, timestamp_ns=22),
        event(event_name="consumer_load_end", process_id=102, timestamp_ns=23),
    )

    pd_compact_candidate._validate_handoff_event_group(
        request_id="request-0",
        producer={"connector": "P2pNcclConnector", "instance_role": "prefill"},
        load_scheduled={"connector": "P2pNcclConnector", "instance_role": "decode"},
        transfer_starts=transfer_starts,
        transfer_ends=transfer_ends,
        load_starts=load_starts,
        load_ends=load_ends,
    )
