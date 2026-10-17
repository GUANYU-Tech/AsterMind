import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from src.kernel.dataset_storage import load_kernel_dataset_document

from dataset_pipeline.kernel_dataset.config import (
    get_profiler_output_dir,
    get_hardware_id,
    load_pipeline_config,
    load_workload_config,
    parse_kernel_list,
    to_float,
)
from dataset_pipeline.kernel_dataset.metax_c500 import (
    _runner_command,
    _to_num,
    build_flash_attn_shapes as build_metax_flash_attn_shapes,
)
from dataset_pipeline.kernel_dataset.ascend_910b import (
    _select_profile_shapes,
    _timeline_name_matches,
    _timeline_steady_state_public_call_measurement,
    _timeline_window_relative_difference_pct,
    build_flash_attn_shapes as build_ascend_flash_attn_shapes,
)
from dataset_pipeline.kernel_dataset.nvidia_a800 import (
    _build_flash_attn_jobs,
    _build_vector_jobs,
    _load_vector_timing_observations,
    _profile_run_output_dir,
    _select_profile_jobs,
    _validate_forced_split_capture,
)
from dataset_pipeline.kernel_dataset.samsung_hbm_pim import (
    _decomp_vector_latency,
    _eager_attn_case_id,
    _enumerate_moe_cases,
)
from dataset_pipeline.conversion.msprof_summary import _parse_mnk
from dataset_pipeline.kernel_dataset.workload_shapes import (
    build_flash_attn_shapes,
    build_gemm_shapes,
    build_vector_shapes,
    flash_attn_request_lengths,
)


def _filenames(shapes):
    return {shape["filename"] for shape in shapes}


def test_a800_vector_timing_parser_requires_exact_serving_kernel_ranges(
        tmp_path):
    case_key = "VECTOR_ROPE_T1_NH32D128_B1"
    path = tmp_path / "timing.csv"
    path.write_text(
        "NVTX Range,Style,Kern Inst,Med (ns),Kernel Name\n"
        f":{case_key},PushPop,10,3328,"
        "void vllm::rotary_embedding_kernel<c10::BFloat16>()\n"
    )
    shape_index = {
        case_key: {"attr": {"op": "rope"}},
    }

    observations = _load_vector_timing_observations(
        str(path), shape_index)

    assert observations == {case_key: 3328.0}


def test_a800_vector_timing_parser_rejects_missing_case(tmp_path):
    path = tmp_path / "timing.csv"
    path.write_text(
        "NVTX Range,Style,Kern Inst,Med (ns),Kernel Name\n")

    with pytest.raises(ValueError, match="do not match"):
        _load_vector_timing_observations(
            str(path),
            {"VECTOR_RMSNORM_T1_H2048_B1": {
                "attr": {"op": "rmsnorm"}}},
        )


def test_shared_workload_shapes_match_cross_hardware_canonical_datasets():
    flash_config = load_workload_config(
        "configs/dataset/kernel/flash_attn.yaml")
    vector_config = load_workload_config(
        "configs/dataset/kernel/vector.yaml")

    a800_flash_names = _filenames(build_flash_attn_shapes(flash_config))
    ascend_flash_names = _filenames(
        build_ascend_flash_attn_shapes(flash_config))
    vector_names = _filenames(build_vector_shapes(vector_config))

    for hardware_id, flash_names in (
            ("ascend_910b", ascend_flash_names),
            ("nvidia_a800", a800_flash_names)):
        flash_dataset = load_kernel_dataset_document(
            f"kernel_dataset/{hardware_id}/flash_attn.json.zst")
        vector_dataset = load_kernel_dataset_document(
            f"kernel_dataset/{hardware_id}/vector.json.zst")
        assert flash_names <= set(flash_dataset)
        serving_cases = set(flash_dataset) - flash_names
        assert len(serving_cases) == 40
        assert all(
            "capture_evidence" in flash_dataset[case_id]
            for case_id in serving_cases)
        assert vector_names == set(vector_dataset)


def test_flash_attn_split_contract_has_100_forced_gpu_cases_and_no_ascend_sp():
    config = load_workload_config(
        "configs/dataset/kernel/flash_attn.yaml", "flash_attn")
    gpu_shapes = build_flash_attn_shapes(config)
    ascend_shapes = build_ascend_flash_attn_shapes(config)

    assert sum(shape["split_policy"] == "forced" for shape in gpu_shapes) == 100
    assert sum(shape["split_policy"] == "serving" for shape in gpu_shapes) == 194
    assert len(ascend_shapes) == 294
    assert all("_SP" not in shape["filename"] for shape in ascend_shapes)
    assert all(shape["attr"]["SP"] is None for shape in ascend_shapes)
    assert all(
        shape["attr"]["SP_REQUESTED"] is None for shape in ascend_shapes)


def test_ascend_steady_timeline_parser_uses_final_exact_public_call(tmp_path):
    summary_dir = (
        tmp_path / "PROF_000001" / "mindstudio_profiler_output")
    summary_dir.mkdir(parents=True)
    summary = summary_dir / "op_summary_000001.csv"
    summary.write_text(
        "Op Name,Task Duration(us),Task Start Time(us),Task ID,Stream ID,"
        "OP Type,Task Type,Block Dim,Mix Block Dim,Input Shapes,"
        "Input Data Types,Input Formats,Output Shapes,Output Data Types,"
        "Output Formats\n"
        "OneTimeSetup,5.0,100.0,1,7,Setup,AI_VECTOR_CORE,1,0,"
        "N/A,N/A,N/A,N/A,N/A,N/A\n"
        "MatMul_123,12.5,110.0,2,7,MatMul,AI_CORE,24,0,"
        "\"1,4096;4096,4096\",DT_BF16;DT_BF16,ND;ND,"
        "\"1,4096\",DT_BF16,ND\n",
        encoding="utf-8",
    )

    measurement = _timeline_steady_state_public_call_measurement(
        str(tmp_path), "gemm", {
            "filename": "GEMM_B1_M1_K4096_N4096",
            "attr": {"B": 1, "M": 1, "K": 4096, "N": 4096},
        })

    assert measurement["runtime_schedule"] == ["MatMul_123"]
    assert measurement["median_op_elapsed_duration_ns"] == 12_500.0
    assert measurement["calls"][0]["tasks"][0]["input_shapes"] == (
        "1,4096;4096,4096")
    assert measurement["retained_call_selection"] == (
        "final_exact_public_call_after_runner_owned_warmup")


def test_ascend_timeline_window_difference_is_symmetric_and_relative():
    expected = 5.0 / 105.0 * 100.0
    assert _timeline_window_relative_difference_pct(100.0, 105.0) == expected
    assert _timeline_window_relative_difference_pct(105.0, 100.0) == expected


def test_ascend_serving_exact_moe_abs_runtime_alias_is_accepted():
    assert _timeline_name_matches(
        "aclnnAbs_AbsAiCore_Abs", "Abs")
    assert _timeline_name_matches(
        "aclnnMoeInitRoutingV3_MoeInitRoutingV3_MoeInitRoutingV3",
        "MoeInitRouting")


@pytest.mark.parametrize("include_cast", [False, True])
def test_ascend_steady_timeline_parser_selects_moe_public_call_boundaries(
        tmp_path, include_cast):
    summary_dir = (
        tmp_path / "PROF_000001" / "mindstudio_profiler_output")
    summary_dir.mkdir(parents=True)
    summary = summary_dir / "op_summary_000001.csv"
    header = (
        "Op Name,Task Duration(us),Task Start Time(us),Task ID,Stream ID,"
        "OP Type,Task Type,Block Dim,Mix Block Dim,Input Shapes,"
        "Input Data Types,Input Formats,Output Shapes,Output Data Types,"
        "Output Formats\n")
    names = [
        "MoeTokenUnpermute_previous_call", "MoeGatingTopK_1",
        "MoeInitRoutingV3_1", "aclnnGroupedMatmulV5_1", "SwiGlu_1",
        "aclnnGroupedMatmulV5_2", "aclnnAbs_AbsAiCore_Abs",
        "MoeTokenUnpermute_1",
    ]
    if include_cast:
        names.insert(3, "aclnnCast_CastAiCore_Cast")
    rows = []
    for index, name in enumerate(names, start=1):
        rows.append(
            f"{name},1.0,{100.0 + 2.0 * index},{index},7,TestOp,"
            "AI_CORE,24,0,N/A,N/A,N/A,N/A,N/A,N/A\n")
    summary.write_text(header + "".join(rows), encoding="utf-8")

    measurement = _timeline_steady_state_public_call_measurement(
        str(tmp_path), "fused_moe", {
            "filename": "FUSEDMOE_TOK1_E8_TOPK2_H2048_INT768",
            "attr": {},
        })

    assert measurement["runtime_schedule"] == names[1:]
    assert measurement["median_op_elapsed_duration_ns"] == (
        8_000.0 if include_cast else 7_000.0)
    assert measurement["calls"][0]["timing_aggregation"] == (
        "same_stream_task_schedule_summed_duration")


def test_flash_attn_request_lengths_are_the_executable_contract():
    uniform_q, uniform_kv = flash_attn_request_lengths(
        16, 1, 1024, "paged_varlen", "uniform")
    assert uniform_q == [1] * 16
    assert uniform_kv[0] == 256
    assert uniform_kv[-1] == 1021
    assert len(uniform_kv) == 16

    skew_q, skew_kv = flash_attn_request_lengths(
        20, 1, 1024, "paged_varlen", "skew")
    assert skew_q == [1] * 20
    assert skew_kv == [256] * 12 + [512] * 5 + [1024] * 3

    prefill_q, prefill_kv = flash_attn_request_lengths(
        8, 1024, 1024, "paged_varlen", "skew")
    assert prefill_q == prefill_kv
    assert prefill_q == [256] * 4 + [512] * 2 + [1024] * 2

    chunk_q, chunk_kv = flash_attn_request_lengths(
        4, 8192, 4096, "chunked_prefill", "skew")
    assert chunk_q == [512, 512, 1024, 2048]
    assert chunk_kv == [1024, 1024, 2048, 4096]
    assert sum(chunk_q) <= 8192

    with pytest.raises(
            ValueError, match="query length cannot exceed its KV length"):
        flash_attn_request_lengths(
            1, 2048, 1024, "chunked_prefill", "skew")


def test_flash_attn_reprofile_selector_is_exactly_the_nine_corrected_cases():
    config = load_workload_config(
        "configs/dataset/kernel/flash_attn.yaml", "flash_attn")
    shapes = build_flash_attn_shapes(config)
    expected = {
        f"FLASHATTN_PAGEDVARLEN_SKEW_PREFILL_B{batch}_S{sequence}"
        "_HQ32_HKV8_D128_C1"
        for batch, sequence in (
            (8, 1024), (8, 2048),
            (16, 1024), (16, 2048),
            (32, 1024), (32, 2048),
            (64, 1024), (64, 2048),
            (128, 2048),
        )
    }

    ascend_selected = _select_profile_shapes(
        "flash_attn", list(shapes), True, None)
    config["nvtx_range_name"] = "profile_region"
    config["script_path"] = "flash_attn_runner.py"
    a800_selected = _select_profile_jobs(
        "flash_attn",
        _build_flash_attn_jobs(config, "ncu", "python"),
        None,
        True,
    )

    assert _filenames(ascend_selected) == expected
    assert {job["filename"] for job in a800_selected} == expected
    assert all(
        shape["attr"]["Q_LENS"] == shape["attr"]["KV_LENS"]
        for shape in ascend_selected)


def test_a800_forced_split_cases_are_the_direct_legacy_replacement():
    config = load_workload_config(
        "configs/dataset/kernel/flash_attn.yaml", "flash_attn")
    shapes = {
        shape["filename"]: shape for shape in build_flash_attn_shapes(config)
    }
    dataset = load_kernel_dataset_document(
        "kernel_dataset/nvidia_a800/flash_attn.json.zst")
    forced = {
        key: shape for key, shape in shapes.items()
        if shape["split_policy"] == "forced"
    }

    assert len(forced) == 100
    for key, shape in forced.items():
        _validate_forced_split_capture(key, shape, dataset[key])

    bad_key, bad_shape = next(
        (key, shape) for key, shape in forced.items()
        if shape["attr"]["SP_REQUESTED"] > 1)
    sp1_entry = next(
        dataset[key] for key, shape in forced.items()
        if shape["attr"]["SP_REQUESTED"] == 1)
    mislabeled_fixed_one = copy.deepcopy(dataset[bad_key])
    mislabeled_fixed_one["kernels"] = copy.deepcopy(sp1_entry["kernels"])
    with pytest.raises(ValueError, match="was not captured with split-KV"):
        _validate_forced_split_capture(
            bad_key, bad_shape, mislabeled_fixed_one)


def test_gemm_shape_records_and_passes_explicit_bf16_dtype():
    config = load_workload_config(
        "configs/dataset/kernel/gemm.yaml", "gemm")

    shape = build_gemm_shapes(config)[0]

    assert shape["attr"]["dtype"] == "bf16"
    dtype_index = shape["script_args"].index("--dtype")
    assert shape["script_args"][dtype_index + 1] == "bf16"


def test_vector_rope_passes_query_and_kv_heads_and_kvcache_has_no_batch_grid():
    config = load_workload_config(
        "configs/dataset/kernel/vector.yaml", "vector")
    shapes = build_vector_shapes(config)
    rope = next(shape for shape in shapes if shape["attr"]["op"] == "rope")
    kvcache = [
        shape for shape in shapes if shape["attr"]["op"] == "kvcache"]

    assert rope["attr"]["num_heads"] > rope["attr"]["num_kv_heads"]
    assert "--num_heads" in rope["script_args"]
    assert "--num_kv_heads" in rope["script_args"]
    assert {shape["attr"]["batch"] for shape in kvcache} == {1}
    assert len(kvcache) == 16


def test_vector_elementwise_operation_is_explicit_in_attr_and_runner_args():
    config = load_workload_config(
        "configs/dataset/kernel/vector.yaml", "vector")
    shapes = [
        shape for shape in build_vector_shapes(config)
        if shape["attr"]["op"] == "elementwise"
    ]

    assert {shape["attr"]["op2"] for shape in shapes} == {"add", "mul"}
    for shape in shapes:
        operation = shape["attr"]["op2"]
        assert shape["filename"].endswith("_" + operation.upper())
        index = shape["script_args"].index("--op2")
        assert shape["script_args"][index + 1] == operation


def test_a800_rope_reprofile_is_exactly_32_jobs_in_named_staging_run():
    config = load_workload_config(
        "configs/dataset/kernel/vector.yaml", "vector")
    config["nvtx_range_name"] = "profile_region"
    config["script_path"] = "vector_runner.py"
    jobs = _build_vector_jobs(config, "ncu", "python")
    rope_jobs = [job for job in jobs if job["vector_op"] == "rope"]
    pipeline = load_pipeline_config("configs/dataset/nvidia_a800.yaml")

    output = _profile_run_output_dir(
        pipeline, "vector", "rope_hq_hkv_20260806")

    assert len(rope_jobs) == 32
    assert output.endswith(
        "profiler_output/kernel_dataset/nvidia_a800/profile_runs/"
        "rope_hq_hkv_20260806/vector")


def test_pim_decomposed_case_ids_are_distinct_from_gpu_source_ids():
    assert _eager_attn_case_id("FLASHATTN_B1_S512") == "EAGERATTN_B1_S512"
    with pytest.raises(ValueError, match="must start with"):
        _eager_attn_case_id("EAGERATTN_B1_S512")

    config = load_workload_config(
        "configs/dataset/kernel/fused_moe.yaml", "fused_moe")
    task_key, source_key, _ = next(iter(_enumerate_moe_cases(config)))
    assert task_key.startswith("UNFUSEDMOE_")
    assert source_key.startswith("FUSEDMOE_")
    assert task_key.removeprefix("UNFUSEDMOE_") == source_key.removeprefix(
        "FUSEDMOE_")


@pytest.mark.parametrize(
    ("operator", "expected_elements", "suffix"),
    (
        ("rope", 4 * (32 + 8) * 128, "rope"),
        ("kvcache", 2 * 1 * 4 * 8 * 128, "kvc"),
    ),
)
def test_pim_vector_decomposition_preserves_multi_tensor_workload(
        operator, expected_elements, suffix):
    calls = []

    def record(workload, sizes, name):
        calls.append((workload, sizes, name))
        return 1.0

    attributes = {
        "op": operator,
        "tok": 4,
        "hidden": 4096,
        "intermediate": 14336,
        "num_heads": 32,
        "head_dim": 128,
        "num_kv_heads": 8,
        "batch": 1,
    }

    latency, stages = _decomp_vector_latency(
        attributes, record, f"test_{operator}")

    assert latency == 1.0
    assert stages == {"elewise": 1.0}
    assert calls == [
        ("elewise", [expected_elements, 1, 1, 1],
         f"test_{operator}_{suffix}")
    ]


@pytest.mark.parametrize(
    "hardware_id",
    ["ascend_910b", "metax_c500", "nvidia_a800", "samsung_hbm_pim"],
)
def test_active_pipeline_configs_pass_runtime_schema(hardware_id):
    config = load_pipeline_config(
        f"configs/dataset/{hardware_id}.yaml")
    assert get_hardware_id(config) == hardware_id


def test_pipeline_config_rejects_unknown_fields_and_legacy_ids(tmp_path):
    source = yaml.safe_load(open("configs/dataset/nvidia_a800.yaml"))
    source["kernel"]["pipeline"]["settings"]["ncu_csv_ouput_dir"] = "typo"
    path = tmp_path / "nvidia_a800.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="unexpected=.*ncu_csv_ouput_dir"):
        load_pipeline_config(path)

    del source["kernel"]["pipeline"]["settings"]["ncu_csv_ouput_dir"]
    source["hardware_id"] = "a800"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="canonical hardware ID"):
        load_pipeline_config(path)


def test_workload_config_rejects_unknown_fields_and_kernel_mismatch(tmp_path):
    source = yaml.safe_load(open(
        "configs/dataset/kernel/gemm.yaml"))
    source["workload_configs"][0]["sequence"] = 1
    path = tmp_path / "invalid_gemm.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="unexpected=.*sequence"):
        load_workload_config(path, "gemm")

    del source["workload_configs"][0]["sequence"]
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="kernel_name mismatch"):
        load_workload_config(path, "flash_attn")


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf])
def test_metax_numeric_parser_rejects_invalid_profiler_scalars(value):
    expected = TypeError if value is True else ValueError
    with pytest.raises(expected):
        _to_num(value)


def test_metax_fused_moe_profiler_executes_runner_without_importing_it(tmp_path):
    script_path = tmp_path / "capture.py"
    shape = {
        "attr": {"TOK": 2, "E": 8, "TOPK": 2, "H": 16, "INT": 32},
        "script_args": ["--num_tokens", "2", "--dtype", "bf16"],
    }
    runner = "scripts/kernel_dataset/metax_c500/fused_moe_runner.py"

    command = _runner_command(
        "fused_moe", shape, str(script_path), "python", 20, runner,
        use_marker=True)

    assert command[:3] == ["python", "-u", str(Path(runner).resolve())]
    assert command[3:] == [
        "--num_tokens", "2",
        "--dtype", "bf16",
        "--tiling_path", str(tmp_path / "capture_tiling.json"),
        "--profiler_wait", "20",
        "--use_marker",
    ]
    assert not script_path.exists()


@pytest.mark.parametrize(
    "value",
    ["1,2,3;4,2", "1,2;4,2,3", "1,2;4,2;8,8"],
)
def test_msprof_matmul_shape_parser_rejects_extra_dimensions(value):
    with pytest.raises(ValueError, match="invalid MatMul input_shapes"):
        _parse_mnk(value)


def test_metax_flash_attn_uses_all_shared_shapes_with_metadata_selected_mla():
    config = load_workload_config(
        "configs/dataset/kernel/flash_attn.yaml")
    shared = build_flash_attn_shapes(config)
    metax = build_metax_flash_attn_shapes(config)

    assert _filenames(metax) == _filenames(shared)
    assert len(metax) == 294
    assert all(
        shape["attr"]["SP_REQUESTED"] is None
        for shape in metax if shape["attr"]["kv_layout"] == "mla")
    canonical = load_kernel_dataset_document(
        "kernel_dataset/metax_c500/flash_attn.json.zst")
    assert _filenames(metax) <= set(canonical)


def test_flash_attn_shape_rejects_unknown_layout_and_non_boolean_causal():
    config = {
        "workload_configs": [{
            "batch": 1,
            "seq_len_cache": 128,
            "num_splits": 1,
            "split_policy": "serving",
            "causal": False,
            "kv_layout": "typo",
            "kv_dist": "skew",
            "seq_len_q": 1,
        }],
        "layers": [{
            "name": "layer",
            "num_heads_q": 8,
            "num_heads_kv": 2,
            "head_dim": 64,
            "dtype": "bf16",
            "override_only": False,
        }],
    }

    with pytest.raises(ValueError, match="unsupported flash_attn kv_layout"):
        build_flash_attn_shapes(config)

    config["workload_configs"][0]["kv_layout"] = "contiguous"
    config["workload_configs"][0]["causal"] = "false"
    with pytest.raises(TypeError, match="causal must be bool"):
        build_flash_attn_shapes(config)


def test_flash_attn_shape_passes_configured_split_to_hardware_runner():
    config = {
        "workload_configs": [{
            "batch": 1,
            "seq_len_cache": 4096,
            "num_splits": 4,
            "split_policy": "forced",
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "skew",
            "seq_len_q": 1,
        }],
        "layers": [{
            "name": "layer",
            "num_heads_q": 8,
            "num_heads_kv": 2,
            "head_dim": 64,
            "dtype": "bf16",
            "override_only": False,
        }],
    }

    config["nvtx_range_name"] = "profile_region"
    config["script_path"] = "flash_attn_runner.py"
    command = _build_flash_attn_jobs(config, "ncu", "python")[0]["cmd"]
    split_index = command.index("--num_splits")
    assert command[split_index + 1] == "4"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("batch", "1", "batch must be an integer"),
        ("kv_layout", 1, "kv_layout must be a string"),
        ("kv_dist", False, "kv_dist must be a string"),
    ],
)
def test_flash_attn_shape_rejects_coercible_config_values(
        field, value, message):
    config = {
        "workload_configs": [{
            "batch": 1,
            "seq_len_cache": 128,
            "num_splits": 1,
            "split_policy": "serving",
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "skew",
            "seq_len_q": 1,
        }],
        "layers": [{
            "name": "layer",
            "num_heads_q": 8,
            "num_heads_kv": 2,
            "head_dim": 64,
            "dtype": "bf16",
            "override_only": False,
        }],
    }
    config["workload_configs"][0][field] = value

    with pytest.raises(TypeError, match=message):
        build_flash_attn_shapes(config)


def test_workload_shape_rejects_unknown_layer_and_vector_operator():
    flash_config = {
        "workload_configs": [{
            "batch": 1,
            "seq_len_cache": 128,
            "num_splits": 1,
            "split_policy": "serving",
            "causal": False,
            "layer": "missing",
        }],
        "layers": [{
            "name": "layer",
            "num_heads_q": 8,
            "num_heads_kv": 2,
            "head_dim": 64,
            "dtype": "bf16",
            "override_only": False,
        }],
    }
    with pytest.raises(ValueError, match="unknown workload layer"):
        build_flash_attn_shapes(flash_config)

    vector_config = {
        "toks": [1],
        "op_groups": [{"op": "unknown", "layers": ["layer"]}],
        "layers": [{
            "name": "layer",
            "hidden": 64,
            "num_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 64,
            "intermediate": 128,
            "dtype": "bf16",
            "override_only": False,
        }],
    }
    with pytest.raises(ValueError, match="unsupported vector operator"):
        build_vector_shapes(vector_config)


def test_workload_shape_rejects_coercible_layer_and_vector_values():
    flash_config = {
        "workload_configs": [{
            "batch": 1,
            "seq_len_cache": 128,
            "num_splits": 1,
            "split_policy": "serving",
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "skew",
            "seq_len_q": 1,
        }],
        "layers": [{
            "name": 7,
            "num_heads_q": 8,
            "num_heads_kv": 2,
            "head_dim": 64,
            "dtype": "bf16",
            "override_only": False,
        }],
    }
    with pytest.raises(TypeError, match="name must be a string"):
        build_flash_attn_shapes(flash_config)

    vector_config = {
        "op_groups": [{
            "op": "rmsnorm",
            "toks": ["1"],
            "layers": ["layer"],
            "op2": "add",
            "batch": 1,
        }],
        "layers": [{
            "name": "layer",
            "hidden": 64,
            "num_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 64,
            "intermediate": 128,
            "dtype": "bf16",
            "override_only": False,
        }],
    }
    with pytest.raises(TypeError, match="tok must be an integer"):
        build_vector_shapes(vector_config)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "nan", "inf"])
def test_dataset_config_to_float_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="finite"):
        to_float(value)


@pytest.mark.parametrize(
    "raw_kernels, message",
    [
        ("gemm,,vector", "empty kernel"),
        ("gemm,unknown", "unsupported kernel"),
        ("gemm,gemm", "duplicate kernel"),
    ],
)
def test_parse_kernel_list_rejects_ambiguous_kernel_sets(
        raw_kernels, message):
    config = {
        "hardware_id": "nvidia_a800",
        "kernel": {"pipeline": {
            "workloads": ["gemm", "vector"],
            "settings": {},
        }},
    }
    with pytest.raises(ValueError, match=message):
        parse_kernel_list(config, raw_kernels)


def test_parse_kernel_list_rejects_non_string_input():
    config = {
        "hardware_id": "nvidia_a800",
        "kernel": {"pipeline": {
            "workloads": ["gemm", "vector"],
            "settings": {},
        }},
    }
    with pytest.raises(TypeError, match="--kernel must be a string"):
        parse_kernel_list(config, ["gemm"])


def test_dataset_config_to_float_rejects_non_measurement_types():
    with pytest.raises(TypeError, match="string or numeric measurement"):
        to_float([1.0])


def test_kernel_profiler_output_requires_canonical_workspace(tmp_path):
    config = load_pipeline_config("configs/dataset/nvidia_a800.yaml")
    assert get_profiler_output_dir(config).endswith(
        "profiler_output/kernel_dataset/nvidia_a800")
