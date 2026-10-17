"""Contract tests for H200 kcampaign candidate assembly (synthetic runs)."""

import hashlib
import json

import pytest

from dataset_pipeline.kernel_dataset import nvidia_h200
from evaluation.kernel.dataset import parse_kernel_evaluation_timing
from src.kernel.dataset_storage import load_kernel_dataset_document


_AFTER_CUTOFF_NS = (
    nvidia_h200.H200_SINGLE_PROCESS_CAPTURE_START_EPOCH_NS + 10 ** 9)


def _plan(family, case_id, runner_args):
    marker = nvidia_h200.h200_timing_marker(family, case_id)
    return {
        "family": family, "case_id": case_id, "marker": marker,
        "runner_relpath": nvidia_h200.H200_CAMPAIGN_RUNNER_RELPATHS[family],
        "runner_args": [*runner_args, "--nvtx_range_name", marker],
    }


def _gemm_plan(case_id, m):
    return _plan("gemm", case_id, [
        "--bs", "1", "--sl", str(m), "--in_feat", "4096",
        "--out_feat", "4096", "--dtype", "bf16"])


def _window(plan, window, duration_ns, *, clocks=(1980, 1980),
            vllm_version="0.30.0+cu129", start_ns=_AFTER_CUTOFF_NS,
            evidence=None):
    telemetry = {
        "clock_before_nvtx": {"sm_clock_mhz": clocks[0]},
        "clock_after_nvtx": {"sm_clock_mhz": clocks[1]},
        "clock_accepted": min(clocks) >= 1980,
        "process_start_epoch_ns": start_ns,
        "device": {"compute_capability": "9.0"},
        "environment": {"vllm_version": vllm_version},
    }
    telemetry.update(evidence or {})
    return {
        "case_id": plan["case_id"], "family": plan["family"],
        "window": window, "marker": plan["marker"],
        "runner_args": plan["runner_args"],
        "launches": [{
            "name": "nvjet_kernel", "grid": [2, 66, 1], "block": [384, 1, 1],
            "duration_ns": duration_ns, "registers_per_thread": 168,
            "static_shared_memory_bytes": 0,
            "dynamic_shared_memory_bytes": 200000,
            "shared_memory_executed_bytes": 233472}],
        "telemetry": telemetry,
    }


def _write_run(root, name, plans, windows_by_case, failures_by_case=()):
    plan_path = root / "plans" / f"{name}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({
        "schema_version": 1, "hardware_id": "nvidia_h200",
        "campaign": name, "case_count": len(plans), "plans": plans}))
    results_root = root / "runs" / name
    results_root.mkdir(parents=True)
    (results_root / "_provenance.json").write_text(json.dumps([{
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}]))
    for case_id, windows in windows_by_case.items():
        case_dir = results_root / "cases" / case_id
        case_dir.mkdir(parents=True)
        for window in windows:
            (case_dir / f"window_{window['window']:02d}.launch.json"
             ).write_text(json.dumps(window))
    for case_id, window in failures_by_case:
        case_dir = results_root / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"window_{window:02d}.failure.json").write_text(
            json.dumps({"case_id": case_id, "window": window}))
    return plan_path, results_root


def test_campaign_assembly_statuses_and_evaluation_timing(tmp_path):
    plans = [_gemm_plan(f"KC_GEMM_M{m}", m) for m in (1, 2, 4, 8, 16)]
    windows = {
        "KC_GEMM_M1": [_window(plans[0], w, d)
                       for w, d in ((1, 1000), (2, 1010), (3, 990))],
        "KC_GEMM_M2": [_window(plans[1], 1, 1000),
                       _window(plans[1], 2, 1000, clocks=(1980, 1755)),
                       _window(plans[1], 3, 1000)],
        "KC_GEMM_M4": [_window(plans[2], w, d)
                       for w, d in ((1, 1000), (2, 1200), (3, 1000))],
        "KC_GEMM_M8": [_window(plans[3], 1, 1000)],
    }
    plan_path, results_root = _write_run(
        tmp_path, "kc_gemm_x__v030", plans, windows,
        failures_by_case=[("KC_GEMM_M8", 2)])
    manifests = nvidia_h200.assemble_campaign_plan(
        plan_path=plan_path, results_root=results_root,
        output_dir=tmp_path / "out")
    manifest = manifests["gemm"]
    statuses = {case["case_id"]: case["status"] for case in manifest["cases"]}
    assert statuses == {
        "KC_GEMM_M1": "pass",
        "KC_GEMM_M2": "insufficient_clock_accepted_windows",
        "KC_GEMM_M4": "reproducibility_exceeds_limit",
        "KC_GEMM_M8": "runner_failure",
        "KC_GEMM_M16": "not_captured",
    }
    assert manifest["vllm_stack"] == "vllm030"
    assert manifest["evaluation_supported"] is True
    assert manifest["complete"] is False
    document = load_kernel_dataset_document(manifest["candidate_path"])
    assert list(document) == ["KC_GEMM_M1"]
    entry = document["KC_GEMM_M1"]
    assert entry["op_info"]["Type"] == "GEMM"
    assert entry["op_info"]["Attr"] == {
        "B": 1, "M": 1, "K": 4096, "N": 4096, "dtype": "bf16"}
    assert entry["h200_capture"]["vllm_version"] == "0.30.0+cu129"
    assert entry["h200_capture"]["selected_window"] == 1
    timing = parse_kernel_evaluation_timing(
        entry, entry["kernels"], "nvidia_h200")
    assert timing.source == "nsys_nvtx_kernel_median"
    assert timing.durations_ns == (1000.0,)
    entry["op_info"]["evaluation"]["timing"]["source"] = "mcTracer"
    with pytest.raises(ValueError, match="nsys_nvtx_kernel_median"):
        parse_kernel_evaluation_timing(entry, entry["kernels"], "nvidia_h200")


def test_campaign_run_mixing_vllm_versions_fails(tmp_path):
    plans = [_gemm_plan("KC_GEMM_M1", 1), _gemm_plan("KC_GEMM_M2", 2)]
    windows = {
        "KC_GEMM_M1": [_window(plans[0], 1, 1000)],
        "KC_GEMM_M2": [_window(plans[1], 1, 1000,
                               vllm_version="0.24.0+cu129")],
    }
    plan_path, results_root = _write_run(tmp_path, "mixed", plans, windows)
    with pytest.raises(ValueError, match="mixes vLLM stacks"):
        nvidia_h200.assemble_campaign_plan(
            plan_path=plan_path, results_root=results_root,
            output_dir=tmp_path / "out")


def test_campaign_plan_hash_must_match_launch_provenance(tmp_path):
    plans = [_gemm_plan("KC_GEMM_M1", 1)]
    plan_path, results_root = _write_run(tmp_path, "p", plans, {})
    plan_path.write_text(plan_path.read_text() + "\n")
    with pytest.raises(ValueError, match="plan sha256"):
        nvidia_h200.assemble_campaign_plan(
            plan_path=plan_path, results_root=results_root,
            output_dir=tmp_path / "out")


def test_selection_rejects_concurrent_capture_and_inconsistent_clock_flag():
    plan = _gemm_plan("KC_GEMM_M1", 1)
    early = nvidia_h200.H200_SINGLE_PROCESS_CAPTURE_START_EPOCH_NS - 1
    windows = [_window(plan, 1, 1000, start_ns=early),
               _window(plan, 2, 1000), _window(plan, 3, 1000)]
    decision = nvidia_h200.select_case_timing("KC_GEMM_M1", windows, [])
    assert decision["status"] == "concurrent_capture_rejected"
    assert decision["concurrent_capture_rejected_windows"] == [1]
    windows[0] = _window(plan, 1, 1000)
    windows[0]["telemetry"]["clock_accepted"] = False
    with pytest.raises(ValueError, match="contradicts clocks"):
        nvidia_h200.select_case_timing("KC_GEMM_M1", windows, [])


def test_vector_and_flash_attn_attributes_come_from_runner_arguments():
    vector = _plan("vector", "KC_ROPE_T8_NH32D128", [
        "--op", "rope", "--tok", "8", "--dtype", "bf16", "--batch", "1",
        "--op2", "add", "--hidden", "4096", "--num_heads", "32",
        "--head_dim", "128", "--intermediate", "14336",
        "--num_kv_heads", "8"])
    op_type, attributes = nvidia_h200.campaign_case_op_info(
        "vector", vector["runner_args"], {"telemetry": {}})
    assert op_type == "ROTARY"
    assert attributes["element_count"] == 8 * 32 * 128
    assert "layer" not in attributes
    flash = _plan("flash_attn", "KC_FA_PAGED_DECODE", [
        "--batch", "2", "--seq_len_cache", "256", "--num_heads_q", "16",
        "--num_heads_kv", "2", "--head_dim", "128", "--dtype", "bf16",
        "--kv_layout", "paged", "--kv_dist", "uniform", "--seq_len_q", "1",
        "--q-lens-json", "[1, 1]", "--kv-lens-json", "[128, 256]",
        "--causal", "0", "--num_splits", "1"])
    op_type, attributes = nvidia_h200.campaign_case_op_info(
        "flash_attn", flash["runner_args"], {"telemetry": {}})
    assert op_type == "FLASHATTN"
    assert attributes["KV_LENS"] == [128, 256]
    assert attributes["split_policy"] == "serving"
    assert "SP" not in attributes
    bad = list(flash["runner_args"])
    bad[bad.index("--kv-lens-json") + 1] = "[128, 128]"
    with pytest.raises(ValueError, match="maxima"):
        nvidia_h200.campaign_case_op_info("flash_attn", bad, {"telemetry": {}})


def test_fused_moe_routing_comes_from_launch_evidence():
    plan = _plan("fused_moe", "KC_MOE", [
        "--num_tokens", "2", "--num_experts", "4", "--top_k", "2",
        "--hidden_size", "64", "--intermediate_size", "32",
        "--dtype", "bf16"])
    projection = {"N": 64, "K": 64, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
                  "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "EM": 64,
                  "grid": 8}
    evidence = {"moe_tiling": {
        "num_tokens": 2, "num_experts": 4, "top_k": 2, "hidden_size": 64,
        "intermediate_size": 32, "tok_per_expert": [1, 1, 2, 0],
        "w1": projection, "w2": projection}}
    op_type, attributes = nvidia_h200.campaign_case_op_info(
        "fused_moe", plan["runner_args"], {"telemetry": evidence})
    assert op_type == "FUSEDMOE"
    assert attributes["tok_per_expert"] == [1, 1, 2, 0]
    assert attributes["moe_tiling"]["w1"]["BM"] == 16
    evidence["moe_tiling"]["tok_per_expert"] = [1, 1, 1, 0]
    with pytest.raises(ValueError, match="tok_per_expert"):
        nvidia_h200.campaign_case_op_info(
            "fused_moe", plan["runner_args"], {"telemetry": evidence})


def test_family_without_evaluation_op_type_is_marked_unsupported(tmp_path):
    plan = _plan("kda", "KC_KDA_DECODE", [
        "--op", "vllm_decode", "--num-heads", "12", "--head-dim", "128",
        "--gate-lower-bound", "-5.0"])
    evidence = {"kda_case": {"chunk_size": 64}}
    windows = {"KC_KDA_DECODE": [
        _window(plan, w, 5000, vllm_version="0.24.0+cu129",
                evidence=evidence) for w in (1, 2, 3)]}
    plan_path, results_root = _write_run(
        tmp_path, "kc_kda_x", [plan], windows)
    manifest = nvidia_h200.assemble_campaign_plan(
        plan_path=plan_path, results_root=results_root,
        output_dir=tmp_path / "out")["kda"]
    assert manifest["evaluation_supported"] is False
    assert manifest["evaluation_unsupported_reason"]
    assert manifest["vllm_stack"] == "vllm024"
    entry = load_kernel_dataset_document(
        manifest["candidate_path"])["KC_KDA_DECODE"]
    assert entry["op_info"]["Type"] == "KDA"
    assert entry["op_info"]["Attr"] == {
        "runner_args": {"op": "vllm_decode", "num-heads": 12,
                        "head-dim": 128, "gate-lower-bound": -5.0},
        "launch_evidence": {"chunk_size": 64}}


def _metadata(batch, *, max_seqlen_q, headdim, page_size, causal, blocks):
    arguments = {
        "batch_size": batch, "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": 1024, "num_heads_q": 32, "num_heads_kv": 8,
        "headdim": headdim, "headdim_v": headdim, "page_size": page_size,
        "causal": causal, "num_splits": 0, "cache_seqlens": [1024] * batch,
        "window_size": [-1, -1], "has_softcap": False}
    rounded = (batch + 3) // 4 * 4
    values = []
    for block in blocks:
        values += block + [0] * (rounded - len(block))
    return {"source": "measured_call_scheduler_metadata",
            "layout_source": "synthetic", "metadata_int32": values + [0],
            "get_scheduler_metadata_arguments": arguments}


def test_fa3_split_attributes_decode_each_metadata_layout():
    decode = nvidia_h200.fa3_split_attributes
    # Paged decode, static bound > 1: prepare_seqlen_q + num_splits_dynamic.
    assert decode(_metadata(3, max_seqlen_q=1, headdim=128, page_size=16,
                            causal=False, blocks=[[4, 4, 4], [2, 1, 3]]),
                  3) == {"SP": 3, "SP_PER_REQUEST": [2, 1, 3],
                         "FA3_SPLIT_COMBINE": True}
    # Causal prefill, static bound 1: prepare_seqlen_q + num_nheads_in_l2.
    assert decode(_metadata(2, max_seqlen_q=512, headdim=128, page_size=None,
                            causal=True, blocks=[[2048, 2048], [8, 8]]),
                  2) == {"SP": 1, "SP_PER_REQUEST": [1, 1],
                         "FA3_SPLIT_COMBINE": False}
    # MLA decode (rope hdim 64): causal is normalized away, so the second
    # block is num_splits_dynamic.
    assert decode(_metadata(2, max_seqlen_q=1, headdim=64, page_size=16,
                            causal=True, blocks=[[64, 64], [1, 1]]),
                  2)["FA3_SPLIT_COMBINE"] is True
    with pytest.raises(ValueError, match="does not match batch"):
        decode(_metadata(2, max_seqlen_q=512, headdim=128, page_size=None,
                         causal=True, blocks=[[1, 1], [1, 1], [1, 1], [1, 1]]),
               2)


def test_flash_attn_case_without_split_evidence_is_excluded(tmp_path):
    plan = _plan("flash_attn", "KC_FA_PAGED", [
        "--batch", "1", "--seq_len_cache", "1024", "--num_heads_q", "32",
        "--num_heads_kv", "8", "--head_dim", "128", "--dtype", "bf16",
        "--kv_layout", "paged", "--kv_dist", "uniform", "--seq_len_q", "1",
        "--q-lens-json", "[1]", "--kv-lens-json", "[1024]",
        "--causal", "0", "--num_splits", "1"])
    evidence = {"fa3_scheduler_metadata": _metadata(
        1, max_seqlen_q=1, headdim=128, page_size=16, causal=False,
        blocks=[[4], [2]])}
    windows = {"KC_FA_PAGED": [_window(plan, 1, 1000),
                               _window(plan, 2, 1000, evidence=evidence),
                               _window(plan, 3, 1000, evidence=evidence)]}
    plan_path, results_root = _write_run(tmp_path, "fa", [plan], windows)
    manifest = nvidia_h200.assemble_campaign_plan(
        plan_path=plan_path, results_root=results_root,
        output_dir=tmp_path / "out")["flash_attn"]
    entry = load_kernel_dataset_document(
        manifest["candidate_path"])["KC_FA_PAGED"]
    assert entry["op_info"]["Attr"]["SP_PER_REQUEST"] == [2]
    for window in windows["KC_FA_PAGED"]:
        if "fa3_scheduler_metadata" in window["telemetry"]:
            del window["telemetry"]["fa3_scheduler_metadata"]
    plan_path, results_root = _write_run(
        tmp_path / "second", "fa", [plan], windows)
    manifest = nvidia_h200.assemble_campaign_plan(
        plan_path=plan_path, results_root=results_root,
        output_dir=tmp_path / "second_out")["flash_attn"]
    assert manifest["case_status_counts"] == {
        "missing_fa3_split_evidence": 1}


def test_mla_verify_case_attributes_and_causal_split_metadata():
    plan = _plan("flash_attn", "DSR1_MLA_DECODE_HQ16_Q3_B2_S7424", [
        "--batch", "2", "--seq_len_cache", "7424", "--num_heads_q", "16",
        "--num_heads_kv", "1", "--head_dim", "512", "--dtype", "bf16",
        "--kv_layout", "mla", "--kv_dist", "uniform", "--seq_len_q", "3",
        "--q-lens-json", "[3, 3]", "--kv-lens-json", "[7424, 7424]",
        "--causal", "1", "--num_splits", "4"])
    op_type, attributes = nvidia_h200.campaign_case_op_info(
        "flash_attn", plan["runner_args"], {"telemetry": {}})
    assert op_type == "FLASHATTN"
    assert (attributes["Q_LEN"], attributes["Q_REQUESTED"],
            attributes["Q_LENS"]) == (3, 3, [3, 3])
    assert (attributes["kv_layout"], attributes["split_policy"]) == (
        "mla", "serving")
    # Q_LEN > 1 keeps FA3 causal: prepare_seqlen_q, num_splits_dynamic and
    # num_nheads_in_l2 blocks.
    metadata = _metadata(2, max_seqlen_q=3, headdim=64, page_size=16,
                         causal=True, blocks=[[48, 48], [3, 2], [8, 8]])
    assert nvidia_h200.fa3_split_attributes(metadata, 2) == {
        "SP": 3, "SP_PER_REQUEST": [3, 2], "FA3_SPLIT_COMBINE": True}
