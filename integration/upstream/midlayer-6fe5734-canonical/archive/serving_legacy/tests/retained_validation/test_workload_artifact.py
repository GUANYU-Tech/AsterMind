import json
from pathlib import Path
import random

import pytest

from src.serve_predict.experiment_manifest import (
    ManifestValidationError,
    create_workload_artifact,
    load_workload_artifact,
    materialize_workload_commands,
    save_workload_artifact,
)


def _mock_tokenizer(text, add_special_tokens=False):
    return {"input_ids": text.split()}


def _sampling(max_tokens=8):
    return {
        "temperature": 0.0,
        "min_tokens": max_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }


def _write_artifact(tmp_path, artifact):
    path = tmp_path / "workload.json"
    save_workload_artifact(artifact, path)
    return path


def test_generated_workload_is_deterministic_for_same_seed():
    first = create_workload_artifact(
        name="long-prefill",
        num_requests=4,
        prompt_len=6,
        output_len=8,
        sampling=_sampling(8),
        seed=123,
        tokenizer=_mock_tokenizer,
    )
    second = create_workload_artifact(
        name="long-prefill",
        num_requests=4,
        prompt_len=6,
        output_len=8,
        sampling=_sampling(8),
        seed=123,
        tokenizer=_mock_tokenizer,
    )

    assert first.prompts == second.prompts
    assert first.prompt_lengths == [6, 6, 6, 6]
    assert first.request_ids == ["req-000000", "req-000001", "req-000002", "req-000003"]
    assert first.hash == second.hash


def test_seed_and_output_token_controls_change_workload_hash():
    base = create_workload_artifact(
        name="long-prefill",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=1,
        tokenizer=_mock_tokenizer,
    )
    different_seed = create_workload_artifact(
        name="long-prefill",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=2,
        tokenizer=_mock_tokenizer,
    )
    different_tokens = create_workload_artifact(
        name="long-prefill",
        num_requests=2,
        prompt_len=4,
        output_len=16,
        sampling=_sampling(16),
        seed=1,
        tokenizer=_mock_tokenizer,
    )

    assert base.hash != different_seed.hash
    assert base.hash != different_tokens.hash


def test_prompt_file_reuse_preserves_prompts_and_records_actual_count():
    prompts = ["alpha beta", "gamma delta epsilon"]

    artifact = create_workload_artifact(
        name="from-file",
        num_requests=5,
        prompt_len=128,
        output_len=8,
        sampling=_sampling(8),
        seed=99,
        tokenizer=_mock_tokenizer,
        prompts=prompts,
    )

    assert artifact.prompts == prompts
    assert artifact.prompt_lengths == [2, 3]
    assert artifact.num_requests == 2
    assert artifact.requested_num_requests == 5
    assert artifact.workload["prompt_len"] == 128


def test_saved_artifact_round_trips_with_stable_hash(tmp_path):
    artifact = create_workload_artifact(
        name="roundtrip",
        num_requests=3,
        prompt_len=5,
        output_len=8,
        sampling=_sampling(8),
        seed=7,
        tokenizer=_mock_tokenizer,
    )
    path = _write_artifact(tmp_path, artifact)

    loaded = load_workload_artifact(path)

    assert loaded.path == path
    assert loaded.prompts == artifact.prompts
    assert loaded.request_ids == artifact.request_ids
    assert loaded.sampling == artifact.sampling
    assert loaded.hash == artifact.hash


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda raw: raw.update({"prompts": []}), "prompts"),
        (lambda raw: raw.update({"request_ids": ["req-0", "req-0"]}), "duplicate request_ids"),
        (lambda raw: raw.update({"prompt_lengths": [1]}), "prompt_lengths"),
        (lambda raw: raw.update({"num_requests": 99}), "num_requests"),
        (lambda raw: raw["sampling"].pop("min_tokens"), "sampling"),
        (lambda raw: raw["sampling"].update({"min_tokens": 1}), "min_tokens"),
        (lambda raw: raw["sampling"].update({"ignore_eos": False}), "ignore_eos"),
        (lambda raw: raw["sampling"].update({"top_p": 0.9}), "unsupported sampling"),
        (lambda raw: raw.pop("token_controls"), "token_controls"),
        (lambda raw: raw.update({"hash": "bad"}), "hash"),
    ],
)
def test_load_workload_artifact_rejects_malformed_content(tmp_path, mutation, message):
    artifact = create_workload_artifact(
        name="strict",
        num_requests=2,
        prompt_len=3,
        output_len=8,
        sampling=_sampling(8),
        seed=11,
        tokenizer=_mock_tokenizer,
    )
    raw = artifact.to_dict()
    mutation(raw)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ManifestValidationError, match=message):
        load_workload_artifact(path)


def test_command_materialization_uses_one_workload_artifact_for_all_paths(tmp_path):
    artifact = create_workload_artifact(
        name="shared",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=3,
        tokenizer=_mock_tokenizer,
    )
    path = _write_artifact(tmp_path, artifact)

    commands = materialize_workload_commands(
        workload_path=path,
        model="/models/dense",
        deployment="disaggregated",
        colocated_hw_config="configs/hw/A800.yaml",
        disaggregated_prefill_hw_config="configs/hw/A800.yaml",
        disaggregated_decode_hw_config="configs/hw/A800.yaml",
        prefill_trace="serve_exp/pd_prefill.json",
        decode_trace="serve_exp/pd_decode.json",
    )

    assert commands["workload_hash"] == artifact.hash
    assert commands["sampling"] == artifact.sampling
    repeated = materialize_workload_commands(workload_path=path, model="/models/dense")
    assert repeated["workload_hash"] == artifact.hash
    assert commands["real_vllm_benchmark"]["workload_path"] == str(path)
    assert commands["colocated_midlayer_prediction"]["workload_path"] == str(path)
    assert commands["disaggregated_midlayer_emulation"]["workload_path"] == str(path)
    assert commands["disaggregated_midlayer_prediction"]["workload_path"] == str(path)
    for name, entry in commands["commands"].items():
        if name == "real_vllm_benchmark":
            assert entry["vllm_dataset_path"] in entry["argv"]
            assert "--max-tokens" not in entry["argv"]
            assert "--prompts-file" not in entry["argv"]
        elif name == "colocated_midlayer_prediction":
            assert entry["workload_path"] == str(path)
            assert any("predict_serve_metric.py" in arg for arg in entry["argv"])
        else:
            assert str(path) in entry["argv"]
            assert "--max-tokens" in entry["argv"]
            assert "8" in entry["argv"]


def test_colocated_materialization_carries_tensor_parallel_size(tmp_path):
    artifact = create_workload_artifact(
        name="shared",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=3,
        tokenizer=_mock_tokenizer,
    )
    path = _write_artifact(tmp_path, artifact)

    commands = materialize_workload_commands(
        workload_path=path,
        model="/models/dense",
        colocated_tp_size=4,
        colocated_trace=tmp_path / "trace.json",
        colocated_prediction_output=tmp_path / "prediction.json",
    )

    emulation = commands["colocated_midlayer_emulation"]
    prediction = commands["colocated_midlayer_prediction"]
    assert "--tensor-parallel-size" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--tensor-parallel-size") + 1] == "4"
    assert emulation["tensor_parallel_size"] == 4
    assert "--tp-size" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--tp-size") + 1] == "4"
    assert prediction["tp_size"] == 4


def test_disaggregated_materialization_carries_engine_args_and_tp_sizes(tmp_path):
    artifact = create_workload_artifact(
        name="shared",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=3,
        tokenizer=_mock_tokenizer,
    )
    path = _write_artifact(tmp_path, artifact)
    prefill_engine_args = {
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.8,
    }
    decode_engine_args = {"tensor_parallel_size": 1}
    expected_prefill = json.dumps(prefill_engine_args, sort_keys=True)
    expected_decode = json.dumps(decode_engine_args, sort_keys=True)
    prediction_output = tmp_path / "prediction.json"

    commands = materialize_workload_commands(
        workload_path=path,
        model="/models/dense",
        deployment="disaggregated",
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        prefill_tp_size=2,
        decode_tp_size=1,
        prefill_kv_port=15001,
        decode_kv_port=15009,
        prefill_trace=tmp_path / "prefill.json",
        decode_trace=tmp_path / "decode.json",
        disaggregated_prediction_output=prediction_output,
    )

    emulation = commands["disaggregated_midlayer_emulation"]
    prediction = commands["disaggregated_midlayer_prediction"]
    assert "--prefill-engine-args" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--prefill-engine-args") + 1] == expected_prefill
    assert "--decode-engine-args" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--decode-engine-args") + 1] == expected_decode
    assert emulation["prefill_engine_args"] == prefill_engine_args
    assert emulation["decode_engine_args"] == decode_engine_args
    assert emulation["prefill_engine_args_json"] == expected_prefill
    assert emulation["decode_engine_args_json"] == expected_decode
    assert "--prefill-kv-port" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--prefill-kv-port") + 1] == "15001"
    assert "--decode-kv-port" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--decode-kv-port") + 1] == "15009"
    assert emulation["prefill_kv_port"] == 15001
    assert emulation["decode_kv_port"] == 15009
    assert "--prefill-tp-size" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--prefill-tp-size") + 1] == "2"
    assert "--decode-tp-size" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--decode-tp-size") + 1] == "1"
    assert "--json" in prediction["argv"]
    assert "--output" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--output") + 1] == str(prediction_output)
    assert prediction["prefill_tp_size"] == 2
    assert prediction["decode_tp_size"] == 1
    assert prediction["outputs"]["prediction"] == str(prediction_output)


def test_real_vllm_benchmark_uses_custom_jsonl_dataset_sidecar(tmp_path):
    prompts = ["alpha beta", "gamma delta"]
    artifact = create_workload_artifact(
        name="bench",
        num_requests=2,
        prompt_len=16,
        output_len=8,
        sampling=_sampling(8),
        seed=3,
        tokenizer=_mock_tokenizer,
        prompts=prompts,
    )
    path = _write_artifact(tmp_path, artifact)

    commands = materialize_workload_commands(workload_path=path, model="/models/dense")

    benchmark = commands["real_vllm_benchmark"]
    dataset_path = tmp_path / "workload.vllm-bench.jsonl"
    assert benchmark["vllm_dataset_path"] == str(dataset_path)
    assert commands["vllm_dataset_path"] == str(dataset_path)
    assert dataset_path.exists()
    prompt_by_id = dict(zip(artifact.request_ids, artifact.prompts))
    assert [
        json.loads(line)
        for line in dataset_path.read_text().splitlines()
    ] == [
        {"prompt": prompt_by_id[request_id]}
        for request_id in benchmark["vllm_sidecar_request_ids"]
    ]
    assert benchmark["workload_path"] == str(path)
    assert benchmark["workload_hash"] == artifact.hash
    assert "--dataset-path" in benchmark["argv"]
    assert str(dataset_path) in benchmark["argv"]
    assert "--custom-output-len" in benchmark["argv"]
    assert "8" in benchmark["argv"]
    assert "--ignore-eos" in benchmark["argv"]
    assert "--custom-skip-chat-template" in benchmark["argv"]
    assert "--seed" in benchmark["argv"]
    assert "3" in benchmark["argv"]
    assert "--temperature" in benchmark["argv"]
    assert "0.0" in benchmark["argv"]
    assert "--prompts-file" not in benchmark["argv"]
    assert "--max-tokens" not in benchmark["argv"]
    assert benchmark["vllm_dataset_seed"] == 3
    assert benchmark["vllm_consumed_request_ids"] == artifact.request_ids
    assert sorted(benchmark["vllm_sidecar_request_ids"]) == sorted(artifact.request_ids)


def test_vllm_sidecar_inverse_shuffle_preserves_canonical_consumed_order(tmp_path):
    prompts = [f"prompt {idx}" for idx in range(6)]
    artifact = create_workload_artifact(
        name="bench",
        num_requests=6,
        prompt_len=2,
        output_len=8,
        sampling=_sampling(8),
        seed=19,
        tokenizer=_mock_tokenizer,
        prompts=prompts,
    )
    path = _write_artifact(tmp_path, artifact)

    commands = materialize_workload_commands(workload_path=path, model="/models/dense")

    benchmark = commands["real_vllm_benchmark"]
    sidecar_prompts = [
        json.loads(line)["prompt"]
        for line in Path(benchmark["vllm_dataset_path"]).read_text().splitlines()
    ]
    shuffled_indices = list(range(len(sidecar_prompts)))
    random.Random(artifact.seed).shuffle(shuffled_indices)
    consumed_prompts = [sidecar_prompts[index] for index in shuffled_indices]

    assert consumed_prompts == artifact.prompts
    assert benchmark["vllm_consumed_request_ids"] == artifact.request_ids
    assert benchmark["vllm_sidecar_request_ids"] != artifact.request_ids


def test_materialization_reuses_existing_vllm_sidecar_without_hash_change(tmp_path):
    artifact = create_workload_artifact(
        name="bench",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=3,
        tokenizer=_mock_tokenizer,
    )
    path = _write_artifact(tmp_path, artifact)

    first = materialize_workload_commands(workload_path=path, model="/models/dense")
    second = materialize_workload_commands(workload_path=path, model="/models/dense")

    assert first["workload_hash"] == second["workload_hash"] == artifact.hash
    assert first["vllm_dataset_path"] == second["vllm_dataset_path"]
    assert Path(first["vllm_dataset_path"]).read_text() == Path(second["vllm_dataset_path"]).read_text()


def test_materialization_rejects_missing_or_divergent_workload_inputs(tmp_path):
    artifact = create_workload_artifact(
        name="shared",
        num_requests=2,
        prompt_len=4,
        output_len=8,
        sampling=_sampling(8),
        seed=3,
        tokenizer=_mock_tokenizer,
    )
    path = _write_artifact(tmp_path, artifact)

    with pytest.raises(ManifestValidationError, match="workload artifact does not exist"):
        materialize_workload_commands(
            workload_path=tmp_path / "missing.json",
            model="/models/dense",
        )

    with pytest.raises(ManifestValidationError, match="divergent output-token controls"):
        materialize_workload_commands(
            workload_path=path,
            model="/models/dense",
            sampling=_sampling(16),
        )

    divergent_sampling = _sampling(8)
    divergent_sampling["temperature"] = 0.7
    with pytest.raises(ManifestValidationError, match="temperature"):
        materialize_workload_commands(
            workload_path=path,
            model="/models/dense",
            sampling=divergent_sampling,
        )

    with pytest.raises(ManifestValidationError, match="divergent workload artifact"):
        materialize_workload_commands(
            workload_path=path,
            model="/models/dense",
            expected_workload_path=tmp_path / "other.json",
        )


@pytest.mark.parametrize(
    "sampling_update, message",
    [
        ({"temperature": 0.7}, "temperature"),
        ({"top_p": 0.9}, "unsupported sampling"),
        ({"top_k": 10}, "unsupported sampling"),
        ({"min_p": 0.1}, "unsupported sampling"),
        ({"presence_penalty": 1.0}, "unsupported sampling"),
    ],
)
def test_workload_artifact_rejects_sampling_not_supported_by_all_paths(
    sampling_update,
    message,
):
    sampling = _sampling(8)
    sampling.update(sampling_update)

    with pytest.raises(ManifestValidationError, match=message):
        create_workload_artifact(
            name="bad-sampling",
            num_requests=1,
            prompt_len=2,
            output_len=8,
            sampling=sampling,
            seed=1,
            tokenizer=_mock_tokenizer,
        )
