"""Compare candidate facts against the independently generated V4 reference."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import ast
import gzip
import json
from pathlib import Path
import subprocess

from .v4_flash_reference import build_flash_reference
from .v4_flash_marlin_reference import marlin_scratch_bounds_reference, marlin_tensor_work_reference, marlin_dequant_work_reference, marlin_weight_loads_reference


_PACKED_FP8_ALGORITHMS = {
    "trtllm_fp8_blockscale_swapab_gemm", "deepgemm_sm90_fp8_gemm_1d2d",
    "deepgemm_sm90_fp8_einsum_batched",
}


def _read_json_artifact(path):
    """Read the explicitly named artifact, including lossless gzip exports."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_artifact(path, value, compressed):
    """Keep every reference/check while avoiding expanded duplicate payloads."""
    path = Path(str(path) + ".gz") if compressed else Path(path)
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.write("\n")
    else:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    return path


def _mhc_tensor_work_matches(expected, actual):
    """Compare physical source fields; predictor scope prose is not tensor work."""
    if actual is None or not set(expected).issubset(actual):
        return False
    if set(actual) - set(expected) - {"arithmetic_scope", "byte_scope", "limitations"}:
        return False
    if any(actual[name] != value for name, value in expected.items() if name != "demand"):
        return False
    expected_demand, actual_demand = expected["demand"], actual["demand"]
    if any(name not in actual_demand or actual_demand[name] != value for name, value in expected_demand.items()):
        return False
    # Prenorm's common resource schema explicitly exports an unused vector
    # service. Only its zero is representation-equivalent to no such work.
    return all(name == "compute_vector" and value == 0
               for name, value in actual_demand.items() if name not in expected_demand)


def _fp8_tensor_form_matches(record, calls, sources):
    forms = record["lowering_forms"]
    signature = [(call["kind"], call["variant"]) for call in calls]
    external = signature == [(stage["kind"], stage["variant"]) for stage in forms["external"]]
    internal = signature == [(stage["kind"], stage["variant"]) for stage in forms["in_call"]]
    if not external and not internal:
        return False
    matrix = calls[-1]
    if (matrix["launch"] is None or matrix["dtype"] != "bf16" or matrix["rank"] != record["rank"]
            or any(name not in matrix["shape"] or matrix["shape"][name] != value for name, value in record["shape"].items())
            or "output_dtype" not in matrix["attributes"] or matrix["attributes"]["output_dtype"] not in (None, "bf16")):
        return False
    if internal:
        return matrix["launch"]["algorithm"] == "trtllm_fp8_blockscale_swapab_gemm" and sources["flashinfer-input-quant"] is not None
    quantizer, stage = calls[0], forms["external"][0]
    if (matrix["launch"]["algorithm"] != "deepgemm_sm90_fp8_gemm_1d2d"
            or quantizer["dtype"] != "bf16" or quantizer["rank"] != record["rank"]
            or quantizer["call_id"] not in matrix["dependencies"]
            or any(name not in quantizer["shape"] or quantizer["shape"][name] != value for name, value in stage["required_shape"].items())
            or "logical_work" not in quantizer["resource_input"]):
        return False
    work = quantizer["resource_input"]["logical_work"]
    return all(name in work and work[name] == value for name, value in stage["external_quant_work"].items())


def compare_flash_projections(reference, candidate):
    """Do not convert missing actual facts or unimplemented facets into passes."""
    checks = []

    def check(key, field, expected, actual, source, status=None):
        checks.append({"subject": key, "field": field, "source": source,
                       "applicability": "Flash TP4 PP1 EP-off MTP-off declared source cases",
                       "expected": expected, "actual": actual,
                       "status": ("matched" if expected == actual else "mismatch") if status is None else status})

    phases = defaultdict(list)
    source_alias_actuals = {}
    source_callee_actuals = {}
    runner_links_match = False
    if "compressor_state_source_links" in reference:
        check("full_model", "compressor_state_source_links", reference["compressor_state_source_links"],
              candidate["compressor_state_source_links"] if "compressor_state_source_links" in candidate else None,
              "compressor-state-dataflow")
    if "fp8_runner_source_links" in reference:
        expected_links = reference["fp8_runner_source_links"]
        actual_links = candidate["fp8_runner_source_links"] if "fp8_runner_source_links" in candidate else None
        runner_links_match = expected_links == actual_links
        check("full_model", "fp8_runner_source_links", expected_links, actual_links, "flashinfer-input-quant")
    for name, expected in reference["candidate_requirements"].items():
        check("scenario", name, expected, candidate[name] if name in candidate else None, "declared source profile")
    for name, expected in reference["parallel_requirements"].items():
        check("parallel", name, expected, candidate["deployment"][name], "declared TP4 PP1 EP-off scenario")
    if "embedding_inputs" in reference:
        expected_inputs = reference["embedding_inputs"]
        if expected_inputs is None:
            check("scenario", "independent_embedding_inputs", "externally declared case inputs", None,
                  "independent synthetic workload declaration", "unchecked")
        else:
            check("scenario", "embedding_token_ids_by_phase", expected_inputs,
                  candidate["embedding_token_ids_by_phase"] if "embedding_token_ids_by_phase" in candidate else None,
                  "independent synthetic workload declaration")
    for step in candidate["steps"]:
        phases[step["phase"]].append(step)
    if any("request_ids" in case for case in reference["cases"]):
        if not all("request_ids" in case and len(case["request_ids"]) == len(case["requests"]) for case in reference["cases"]):
            raise ValueError("independent request-identity declaration must cover every phase and row")
        check("scenario", "phase_sequence", [case["phase"] for case in reference["cases"]],
              [step["phase"] for step in candidate["steps"]], "independent consecutive-step declaration")
        prior_ends = {}
        for case in reference["cases"]:
            if len(set(case["request_ids"])) != len(case["request_ids"]):
                raise ValueError("independent phase repeats a request identity")
            for identity, request in zip(case["request_ids"], case["requests"], strict=True):
                if identity in prior_ends:
                    old_end = prior_ends[identity]
                elif "initial_request_prefixes" in reference:
                    if identity not in reference["initial_request_prefixes"]:
                        raise ValueError("independent initial prefix declaration must cover every request")
                    old_end = reference["initial_request_prefixes"][identity]
                else:
                    old_end = 0  # Legacy cold-start reference contract.
                check(case["phase"] + ":request:" + identity, "declared_request_continuity", old_end,
                      request["computed"], "independent consecutive-step declaration")
                prior_ends[identity] = request["computed"] + request["query"]
    expected_phases = {case["phase"] for case in reference["cases"]}
    check("scenario", "phases", sorted(expected_phases), sorted(phases), "external case declaration")
    expected_nodes = {record["node_id"] for record in reference["projections"]}
    for case in reference["cases"]:
        phase = case["phase"]
        if phase not in phases:
            check(phase, "presence", 1, 0, "external case declaration", "missing")
            continue
        if len(phases[phase]) != 1:
            check(phase, "unique_phase", 1, len(phases[phase]), "external case declaration", "duplicate")
            continue
        step = phases[phase][0]
        requests = [{"query": request["input_tokens"] + request["output_tokens"],
                     "computed": request["computed_tokens"]} for request in step["scheduled_requests"]]
        check(phase, "requests", case["requests"], requests, "external case declaration")
        if "request_ids" in case:
            check(phase, "request_identities", case["request_ids"],
                  [request["request_id"] if "request_id" in request else None for request in step["scheduled_requests"]],
                  "independent consecutive-step declaration")
            expected_positions = [list(range(request["computed"], request["computed"] + request["query"])) for request in case["requests"]]
            check(phase, "request_position_ranges", expected_positions,
                  [request["positions"] if "positions" in request else None for request in step["scheduled_requests"]],
                  "independent consecutive-step declaration")
        nodes = defaultdict(list)
        calls = defaultdict(list)
        call_ids = defaultdict(list)
        successors = defaultdict(list)
        for node in step["nodes"]:
            nodes[node["node_id"]].append(node)
            for predecessor in node["predecessors"]:
                successors[predecessor].append(node["node_id"])
            if node["kind"] in ("linear", "router", "lm_head") and node["node_id"] not in expected_nodes:
                check(node["node_id"], "unexpected_projection", False, True, "independent projection inventory", "mismatch")
        for call in step["calls"]:
            if "call_id" in call:
                call_ids[call["call_id"]].append(call)
            for node_id in call["source_node_ids"]:
                calls[node_id].append(call)

        def output_dependency(dependency):
            if len(call_ids[dependency]) != 1:
                return None
            return {field: call_ids[dependency][0][field] for field in
                    ("source_node_ids", "kind", "variant", "rank", "dtype", "shape")}

        def mhc_layer_input_dependency(dependency):
            if len(call_ids[dependency]) != 1 or len(call_ids[dependency][0]["source_node_ids"]) != 1:
                return None
            producer = call_ids[dependency][0]
            source_id = producer["source_node_ids"][0]
            siblings = calls[source_id]
            if any("call_id" not in sibling or len(call_ids[sibling["call_id"]]) != 1 for sibling in siblings):
                return None
            internal_ids = {sibling["call_id"] for sibling in siblings}
            nonterminal = {parent for sibling in siblings for parent in sibling["dependencies"] if parent in internal_ids}
            resource = producer["resource_input"]
            if "mhc_stage_work" not in resource:
                return None
            writes = [operand for operand in resource["mhc_stage_work"]["operands"]
                      if operand["role"] == "layer_input" and operand["access"] == "write"]
            return {"source_node_id": source_id, "rank": producer["rank"],
                    "terminal_source_call": dependency not in nonterminal, "layer_input_writes": writes}

        def terminal_dependency(dependency):
            if len(call_ids[dependency]) != 1 or len(call_ids[dependency][0]["source_node_ids"]) != 1:
                return None
            source_id = call_ids[dependency][0]["source_node_ids"][0]
            siblings = calls[source_id]
            if any("call_id" not in sibling or len(call_ids[sibling["call_id"]]) != 1 for sibling in siblings):
                return None
            internal_ids = {sibling["call_id"] for sibling in siblings}
            nonterminal = {parent for sibling in siblings for parent in sibling["dependencies"] if parent in internal_ids}
            return {"source_node_id": source_id, "terminal_source_call": dependency not in nonterminal}

        def dependencies_reach(required, pending, lowered):
            remaining, queue, seen = set(required), list(pending), set()
            while queue and remaining:
                current = queue.pop()
                if current in seen:
                    continue
                seen.add(current)
                remaining.discard(current)
                if lowered:
                    if len(call_ids[current]) != 1:
                        return False
                    queue.extend(call_ids[current][0]["dependencies"])
                else:
                    if len(nodes[current]) != 1:
                        return False
                    queue.extend(nodes[current][0]["predecessors"])
            return not remaining
        operators = [record for record in reference["operators"] if record["node_id"].startswith(phase + ":")]
        operator_by_id = {record["node_id"]: record for record in operators}
        inventory_ids = {record["node_id"] for record in operators}
        for key in sorted(set(nodes) - inventory_ids):
            check(key, "unexpected_operator", False, True, "independent full logical inventory", "mismatch")
        for record in operators:
            key, source = record["node_id"], record["source"]
            if key not in nodes:
                check(key, "operator_presence", 1, 0, source, "missing")
                continue
            if len(nodes[key]) != 1:
                check(key, "operator_uniqueness", 1, len(nodes[key]), source, "duplicate")
                continue
            node = nodes[key][0]
            for field in ("rank", "kind", "variant", "shape", "dtype"):
                check(key, "operator_" + field, record[field], node[field], source)
            for name, value in record["attributes"].items():
                check(key, "attribute_" + name, value,
                      node["attributes"][name] if name in node["attributes"] else None, source)
            if "partial_state_objects" in record:
                expected = record["partial_state_objects"]
                associated = calls[key]
                valid = (len(associated) == 1 and associated[0]["kind"] == "KV_CACHE"
                    and associated[0]["variant"] == "dsv4_compressor_state_cache" and associated[0]["dtype"] == "fp32"
                    and associated[0]["rank"] == record["rank"])
                check(key, "partial_state_lowered_fp32_writer", True, valid, "save-partial")
                actual = associated[0]["resource_input"] if len(associated) == 1 else {}
                check(key, "partial_state_source_objects", expected,
                      actual["partial_state_objects"] if "partial_state_objects" in actual else None, "save-partial")
                check(key, "partial_state_source_payload", {"read": expected["source_read_bytes"], "write": expected["source_write_bytes"]},
                      actual["modeled_l2_bytes"] if "modeled_l2_bytes" in actual else None, "save-partial")
                check(key, "partial_state_score_ape_fp32_adds", expected["score_ape_fp32_additions"],
                      actual["demand"]["compute_fma"] if "demand" in actual and actual["demand"] is not None else None, "save-partial")
            if "partial_state_frame" in record:
                frame = record["partial_state_frame"]
                parent = frame["projection"]["producer"]
                attributes = node["attributes"]
                check(key, "partial_state_complete_caller_frame", frame,
                      attributes["source_partial_state_frame"] if "source_partial_state_frame" in attributes else None,
                      "compressor-state-dataflow")
                check(key, "partial_state_projection_tensor_input", [frame["projection"]],
                      attributes["source_tensor_inputs"] if "source_tensor_inputs" in attributes else None,
                      "compressor-state-dataflow")
                parent_attributes = nodes[parent][0]["attributes"] if len(nodes[parent]) == 1 else {}
                expected_output = {name: value for name, value in frame["projection"].items()
                                   if name not in ("producer", "output_role")}
                expected_output["aliases_input_or_weight"] = False
                check(key, "partial_state_projection_object", expected_output,
                      parent_attributes["source_dataflow"]["output"] if "source_dataflow" in parent_attributes
                      and "output" in parent_attributes["source_dataflow"] else None, "compressor-state-dataflow")
                same_rank = len(nodes[parent]) == 1 and nodes[parent][0]["rank"] == node["rank"]
                check(key, "partial_state_projection_same_rank", True, same_rank, "compressor-state-dataflow")
                compatible_parent = len(nodes[parent]) == 1 and all(nodes[parent][0][field] == expected
                    for field, expected in (("kind", "linear"), ("variant", "gemm"), ("dtype", "bf16")))
                check(key, "partial_state_projection_typed_form", True,
                      compatible_parent and "weight_dtype" in parent_attributes and parent_attributes["weight_dtype"] == "bf16"
                      and "output_dtype" in parent_attributes and parent_attributes["output_dtype"] == "fp32"
                      and nodes[parent][0]["shape"]["m"] == frame["projection"]["shape"][0]
                      and nodes[parent][0]["shape"]["n"] == frame["projection"]["shape"][1],
                      "compressor-state-dataflow")
                check(key, "partial_state_projection_lowered_readiness", True,
                      bool(calls[key]) and bool(calls[parent]) and dependencies_reach(
                          [item["call_id"] for item in calls[parent]], calls[key][0]["dependencies"], True),
                      "compressor-state-dataflow")
            if "source_partial_state_input" in record["attributes"]:
                binding = record["attributes"]["source_partial_state_input"]
                parent = binding["producer"]
                output = {name: value for name, value in binding.items()
                          if name not in ("producer", "read_only", "global_dtype_cast_allocates", "C128_partial_state_reader_stage")}
                check(key, "finish_same_module_fp32_state_output", output,
                      nodes[parent][0]["attributes"]["source_partial_state_output"]
                      if len(nodes[parent]) == 1 and "source_partial_state_output" in nodes[parent][0]["attributes"] else None,
                      "compressor-state-dataflow")
                check(key, "finish_state_producer_direct_edge", True, parent in node["predecessors"], "compressor-state-dataflow")
                ready = (bool(calls[key]) and bool(calls[parent]) and dependencies_reach(
                    [item["call_id"] for item in calls[parent]], calls[key][0]["dependencies"], True))
                check(key, "finish_state_producer_lowered_readiness", True, ready, "compressor-state-dataflow")
            if "attention_output_region_account" in record:
                account = record["attention_output_region_account"]
                flow = account["declaration"]
                consumer = flow["consumer"]
                associated = calls[key]
                own_ids = [item["call_id"] for item in associated]
                shape = flow["supplied_output"]["shape"]
                requested_output_compatible = len(associated) == 1 and all(
                    item["kind"] == "SPARSE_ATTN" and item["variant"] == record["variant"]
                    and item["rank"] == record["rank"] and item["dtype"] == "bf16"
                    and sum(item["attributes"]["q_lens"]) == shape[0]
                    and item["shape"]["num_query_heads"] == shape[1]
                    and item["shape"]["qk_head_dim"] == shape[2]
                    and item["shape"]["v_head_dim"] == shape[2] for item in associated)
                consumer_ready = bool(own_ids) and bool(calls[consumer]) and all(
                    dependencies_reach(own_ids, item["dependencies"], True) for item in calls[consumer])
                before = account["prefill_predecessors"]
                phase_order = all(calls[parent] and associated and dependencies_reach(
                    [item["call_id"] for item in calls[parent]], associated[0]["dependencies"], True) for parent in before)
                actual = {
                    "declaration": node["attributes"]["source_attention_output_region"]
                        if "source_attention_output_region" in node["attributes"] else None,
                    "query_input_view": node["attributes"]["source_query_input"] if "source_query_input" in node["attributes"] else None,
                    "output_tensor_ports": node["attributes"]["source_tensor_outputs"] if "source_tensor_outputs" in node["attributes"] else None,
                    "consumer_input_ports": [port for port in nodes[consumer][0]["attributes"]["source_tensor_inputs"] if port["producer"] == key]
                        if len(nodes[consumer]) == 1 and "source_tensor_inputs" in nodes[consumer][0]["attributes"] else None,
                    "group_output_tensor_ports": {other: nodes[other][0]["attributes"]["source_tensor_outputs"]
                        if len(nodes[other]) == 1 and "source_tensor_outputs" in nodes[other][0]["attributes"] else None
                        for other in account["group_output_tensor_ports"]},
                    "requested_output_shape_dtype_compatible": bool(requested_output_compatible),
                    "lowered_consumer_reads_after_output": bool(consumer_ready),
                    "prefill_before_decode": bool(phase_order),
                    "prefill_predecessors": sorted(parent for parent in node["predecessors"]
                        if parent in before)}
                check(key, "attention_caller_output_regions", account, actual, "attention-output-regions")
            pending = list(node["predecessors"])
            visited = set()
            required = set(record["required_ancestors"])
            while pending and not required.issubset(visited):
                predecessor = pending.pop()
                if predecessor in visited:
                    continue
                visited.add(predecessor)
                if predecessor not in nodes or len(nodes[predecessor]) != 1:
                    continue  # Missing/duplicated producers have separate failures.
                pending.extend(nodes[predecessor][0]["predecessors"])
            check(key, "required_producer_reachability", sorted(required), sorted(required & visited),
                  record["dependency_source"] if "dependency_source" in record else source)
            if "source_input_binding_account" in record:
                binding = record["source_input_binding_account"]
                external = []
                for call in calls[key]:
                    for dependency in call["dependencies"]:
                        if len(call_ids[dependency]) == 1 and call_ids[dependency][0]["source_node_ids"] == [key]:
                            continue
                        external.append(terminal_dependency(dependency))
                actual_binding = {"direct_input_producers": sorted(node["predecessors"]),
                    "input_tensor_ports": node["attributes"]["source_tensor_inputs"] if "source_tensor_inputs" in node["attributes"] else None,
                    "producer_tensor_outputs": [{"producer": item["producer"],
                        "outputs": nodes[item["producer"]][0]["attributes"]["source_tensor_outputs"]}
                        if len(nodes[item["producer"]]) == 1 and "source_tensor_outputs" in nodes[item["producer"]][0]["attributes"] else None
                        for item in binding["producer_tensor_outputs"]],
                    "external_lowered_producers": sorted(external, key=lambda entry: json.dumps(entry, sort_keys=True))}
                check(key, "source_tensor_return_input_bindings", binding, actual_binding, record["input_binding_source"])
            if "source_alias_account" in record:
                account = record["source_alias_account"]
                if "source_alias_mode" in record:
                    if record["source_alias_mode"] == "embedding_objects":
                        rows, hidden = account["declaration"]["output"]["shape"]
                        consumer = account["declaration"]["consumer"]
                        associated = calls[key]
                        compatible = len(associated) == 1
                        if compatible:
                            call = associated[0]
                            compatible = (call["kind"] == "VECTOR" and call["variant"] == "vocab_parallel_embedding"
                                and call["rank"] == record["rank"] and call["dtype"] == "bf16"
                                and call["source_node_ids"] == [key]
                                and call["shape"] == {"element_count": rows * hidden, "elements": rows * hidden,
                                    "rows": rows, "hidden": hidden, "input_id_bytes": 4, "dtype_bytes": 2}
                                and call["dependencies"] == [])
                        reader = nodes[consumer][0] if len(nodes[consumer]) == 1 else None
                        reader_attrs = None if reader is None else reader["attributes"]
                        own_ids = [call["call_id"] for call in associated]
                        ready = bool(own_ids) and bool(calls[consumer]) and all(
                            dependencies_reach(own_ids, call["dependencies"], True) for call in calls[consumer])
                        source_alias_actuals[key] = {
                            "declaration": node["attributes"]["source_dataflow"] if "source_dataflow" in node["attributes"] else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "direct_consumers": sorted(successors[key]),
                            "output_tensor_roles": node["attributes"]["source_tensor_outputs"] if "source_tensor_outputs" in node["attributes"] else None,
                            "collective_input_producers": None if reader is None else sorted(reader["predecessors"]),
                            "collective_input_ports": reader_attrs["source_tensor_inputs"]
                                if reader_attrs is not None and "source_tensor_inputs" in reader_attrs else None,
                            "collective_consumer": None if reader is None else {"kind": reader["kind"],
                                **{field: reader_attrs[field] if field in reader_attrs else None
                                   for field in ("operation", "reduction", "participants", "bytes")}},
                            "lowered_embedding_compatible": bool(compatible), "collective_reads_after_output": bool(ready)}
                    elif record["source_alias_mode"] == "swa_insert_objects":
                        associated = calls[key]
                        shape = {name: value for name, value in record["shape"].items() if name != "elements"}
                        shape.update(element_count=record["shape"]["elements"], dtype_bytes=2)
                        compatible = len(associated) == 1
                        if compatible:
                            call = associated[0]
                            compatible = (call["kind"] == "KV_CACHE" and call["variant"] == "dsv4_qnorm_rope_kv_insert"
                                and call["dtype"] == "bf16" and call["rank"] == record["rank"] and call["shape"] == shape
                                and call["attributes"] == {"cache_layout": "paged_dsv4_swa", "cache_write_semantics": "append",
                                    "quantization": "fp8_ds_mla", "slot_mapping_all_valid": True}
                                and "swa_insert_objects" in call["resource_input"]
                                and call["resource_input"]["swa_insert_objects"] == record["swa_insert_objects"])
                        reader_ids = account["consumer_query_views"]
                        own_ids = [call["call_id"] for call in associated]
                        ready = bool(own_ids) and all(calls[reader] and all(
                            call["rank"] == record["rank"] and call["dtype"] == "bf16"
                            and call["kind"] == "SPARSE_ATTN" and call["variant"] in ("dsv4_sparse_decode", "dsv4_sparse_prefill")
                            and dependencies_reach(own_ids, call["dependencies"], True) for call in calls[reader]) for reader in reader_ids)
                        source_alias_actuals[key] = {
                            "declaration": node["attributes"]["source_dataflow"] if "source_dataflow" in node["attributes"] else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "input_tensor_ports": node["attributes"]["source_tensor_inputs"] if "source_tensor_inputs" in node["attributes"] else None,
                            "output_tensor_roles": node["attributes"]["source_tensor_outputs"] if "source_tensor_outputs" in node["attributes"] else None,
                            "producer_outputs": [{"node_id": port["node_id"], **{field: nodes[port["node_id"]][0][field]
                                for field in ("rank", "kind", "shape", "dtype")}} if len(nodes[port["node_id"]]) == 1 else None
                                for port in account["producer_outputs"]],
                            "direct_consumers": sorted(successors[key]),
                            "consumer_query_views": {reader: nodes[reader][0]["attributes"]["source_query_input"]
                                if len(nodes[reader]) == 1 and "source_query_input" in nodes[reader][0]["attributes"] else None for reader in reader_ids},
                            "all_readers_after_insert": ready, "lowered_objects_compatible": bool(compatible),
                            "lowered_readiness": sorted([terminal_dependency(parent) for call in associated for parent in call["dependencies"]],
                                                       key=lambda item: json.dumps(item, sort_keys=True))}
                    elif record["source_alias_mode"] == "v4_route_objects":
                        objects = record["route_tensor_objects"]
                        rows, experts = objects["logits"]["shape"]
                        associated = calls[key]
                        compatible = len(associated) == 1
                        if compatible:
                            call = associated[0]
                            compatible = (call["kind"] == "VECTOR" and call["variant"] == "dsv4_moe_route"
                                and call["rank"] == record["rank"] and call["dtype"] == "fp32"
                                and call["shape"] == {"element_count": rows * experts, "elements": rows * experts,
                                    "rows": rows, "experts": experts, "topk": 6, "dtype_bytes": 4}
                                and "routing_source" in call["attributes"]
                                and call["attributes"]["routing_source"] == account["declaration"]["routing_source"]
                                and "route_tensor_objects" in call["resource_input"] and call["resource_input"]["route_tensor_objects"] == objects)
                        own_ids = [item["call_id"] for item in associated]
                        consumers = account["direct_consumers"]
                        ready = bool(own_ids) and all(calls[reader] and all(
                            dependencies_reach(own_ids, item["dependencies"], True) for item in calls[reader]) for reader in consumers)
                        typed = True
                        for reader in consumers:
                            if len(calls[reader]) != 1:
                                typed = False
                                continue
                            expert = calls[reader][0]
                            resource = expert["resource_input"]
                            if (expert["kind"] != "FUSED_MOE" or expert["variant"] != "routed_experts" or expert["dtype"] != "bf16"
                                    or expert["rank"] != record["rank"] or expert["shape"]["tokens"] != rows or expert["shape"]["topk"] != 6
                                    or "logical_work" not in resource):
                                typed = False
                                continue
                            work = resource["logical_work"]
                            weights = [item for item in work["operands"] if item["role"] == "selected_weights" and item["access"] == "read"]
                            typed = typed and len(weights) == 1 and weights[0]["dtype"] == "fp32" and weights[0]["elements"] == rows * 6
                            typed = typed and work["routing_alignment"]["index_dtype"] == "int32"
                        parent = account["declaration"]["logits"]["producer"]
                        source_alias_actuals[key] = {
                            "declaration": node["attributes"]["source_dataflow"] if "source_dataflow" in node["attributes"] else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "logits_producer_outputs": nodes[parent][0]["attributes"]["source_tensor_outputs"]
                                if len(nodes[parent]) == 1 and "source_tensor_outputs" in nodes[parent][0]["attributes"] else None,
                            "input_tensor_ports": node["attributes"]["source_tensor_inputs"] if "source_tensor_inputs" in node["attributes"] else None,
                            "output_tensor_roles": node["attributes"]["source_tensor_outputs"] if "source_tensor_outputs" in node["attributes"] else None,
                            "direct_consumers": sorted(successors[key]),
                            "consumer_routing_inputs": {reader: nodes[reader][0]["attributes"]["source_routing_inputs"]
                                if len(nodes[reader]) == 1 and "source_routing_inputs" in nodes[reader][0]["attributes"] else None for reader in consumers},
                            "model_input_ids": [item["attributes"]["embedding_token_ids"]
                                for records in nodes.values() for item in records if item["kind"] == "gather" and item["variant"] == "embedding"
                                and item["rank"] == record["rank"] and "embedding_token_ids" in item["attributes"]],
                            "lowered_tensor_form_compatible": compatible,
                            "lowered_readiness": sorted([terminal_dependency(parent_id) for item in associated for parent_id in item["dependencies"]],
                                                       key=lambda item: json.dumps(item, sort_keys=True)),
                            "expert_reads_after_route": ready, "expert_routing_types_compatible": typed}
                    elif record["source_alias_mode"] == "short_context_topk_buffer":
                        work = record["short_fill_source_work"]
                        rows, width = work["output"]["shape"]
                        associated = calls[key]
                        compatible = len(associated) == 1
                        if compatible:
                            call = associated[0]
                            compatible = (call["kind"] == "VECTOR" and call["variant"] == "dsv4_short_context_topk"
                                and call["rank"] == record["rank"] and call["dtype"] == "int32"
                                and call["shape"] == {"element_count": rows * width, "elements": rows * width, "seq_len": rows,
                                    "topk_capacity": width, "compress_ratio": 4, "dtype_bytes": 4}
                                and "position_bytes" in call["attributes"] and call["attributes"]["position_bytes"] == 8
                                and "logical_work" in call["resource_input"] and call["resource_input"]["logical_work"] == work)
                        prior = account["declaration"]["previous_layer_readers"]
                        prior_calls = [item["call_id"] for reader in prior for item in calls[reader]]
                        prior_present = all(calls[reader] for reader in prior)
                        own_ids = [item["call_id"] for item in associated]
                        consumer_ready = bool(own_ids) and all(calls[reader] and all(
                            dependencies_reach(own_ids, item["dependencies"], True) for item in calls[reader])
                            for reader in account["direct_consumers"])
                        previous_ready = (prior_present and dependencies_reach(prior, node["predecessors"], False)
                            and bool(associated) and all(dependencies_reach(prior_calls, item["dependencies"], True) for item in associated))
                        parent = account["cache_readiness_producer"]["node_id"]
                        source_alias_actuals[key] = {"declaration": node["attributes"]["source_dataflow"]
                                if "source_dataflow" in node["attributes"] else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "cache_readiness_producer": {"node_id": parent, **{field: nodes[parent][0][field]
                                for field in ("rank", "kind", "shape", "dtype")}} if len(nodes[parent]) == 1 else None,
                            "direct_consumers": sorted(successors[key]),
                            "consumer_input_views": {reader: nodes[reader][0]["attributes"]["source_topk_input"]
                                if len(nodes[reader]) == 1 and "source_topk_input" in nodes[reader][0]["attributes"] else None
                                for reader in account["consumer_input_views"]},
                            "model_position_inputs": [item["attributes"]["source_model_input_order"]["model_positions"]
                                for records in nodes.values() for item in records
                                if item["kind"] == "gather" and item["variant"] == "embedding" and item["rank"] == record["rank"]
                                and "source_model_input_order" in item["attributes"]],
                            "lowered_tensor_form_compatible": compatible,
                            "lowered_readiness": sorted([terminal_dependency(dependency) for item in associated for dependency in item["dependencies"]],
                                                       key=lambda item: json.dumps(item, sort_keys=True)),
                            "consumer_reads_after_fill": consumer_ready, "previous_reads_before_reuse": previous_ready}
                    elif record["source_alias_mode"] == "finish_caller_objects":
                        attrs = node["attributes"]
                        frame = record["attributes"]["source_finish_frame"]
                        parent = frame["state_input"]["producer"]
                        associated = calls[key]
                        forms = record["lowering_account"]
                        compatible = len(associated) == len(forms) and all(
                            all(call[field] == form[field] for field in ("kind", "variant", "dtype"))
                            and call["rank"] == record["rank"]
                            and call["attributes"]["finish_stage"] == (form["finish_stage"] if "finish_stage" in form else None)
                            for call, form in zip(associated, forms))
                        programs = []
                        for call, expected in zip(associated, account["source_stage_programs"]):
                            resource = call["resource_input"]
                            work = resource["finish_work"] if "finish_work" in resource else {}
                            programs.append({field: work[field] if field in work else None for field in expected})
                        source_alias_actuals[key] = {
                            "frame": attrs["source_finish_frame"] if "source_finish_frame" in attrs else None,
                            "output_cache": attrs["source_compressed_cache_output"] if "source_compressed_cache_output" in attrs else None,
                            "partial_state_output": nodes[parent][0]["attributes"]["source_partial_state_output"]
                                if len(nodes[parent]) == 1 and "source_partial_state_output" in nodes[parent][0]["attributes"] else None,
                            "direct_input_producers": sorted(node["predecessors"]), "direct_consumers": sorted(successors[key]),
                            "consumer_cache_inputs": {reader: nodes[reader][0]["attributes"]["source_compressed_cache_inputs"]
                                if len(nodes[reader]) == 1 and "source_compressed_cache_inputs" in nodes[reader][0]["attributes"] else None
                                for reader in account["consumer_cache_inputs"]},
                            "source_stage_programs": programs, "lowered_stage_forms_compatible": compatible,
                            "state_ready_before_finish": bool(associated) and bool(calls[parent]) and dependencies_reach(
                                [call["call_id"] for call in calls[parent]], associated[0]["dependencies"], True),
                            "store_reads_compress_result": len(associated) == 1 or len(associated) == 2 and dependencies_reach(
                                [associated[0]["call_id"]], associated[1]["dependencies"], True),
                            "cache_readers_follow_final_store": bool(associated) and all(calls[reader] and all(dependencies_reach(
                                [associated[-1]["call_id"]], call["dependencies"], True) for call in calls[reader])
                                for reader in account["consumer_cache_inputs"])}
                    elif record["source_alias_mode"] == "partial_state_writer_objects":
                        attrs = node["attributes"]
                        frame = record["partial_state_frame"]
                        parent = frame["projection"]["producer"]
                        parent_attrs = nodes[parent][0]["attributes"] if len(nodes[parent]) == 1 else {}
                        lowered = calls[key]
                        compatible = len(lowered) == 1 and all(lowered[0][field] == expected
                            for field, expected in (("kind", "KV_CACHE"), ("variant", "dsv4_compressor_state_cache"),
                                                    ("dtype", "fp32"), ("rank", record["rank"])))
                        if compatible:
                            compatible = all(name in lowered[0]["shape"] and lowered[0]["shape"][name] == record["shape"][name]
                                             for name in ("seq_len", "state_bytes", "dtype_bytes"))
                        source_alias_actuals[key] = {
                            "frame": attrs["source_partial_state_frame"] if "source_partial_state_frame" in attrs else None,
                            "projection_input": attrs["source_tensor_inputs"] if "source_tensor_inputs" in attrs else None,
                            "projection_output": parent_attrs["source_dataflow"]["output"] if "source_dataflow" in parent_attrs
                                and "output" in parent_attrs["source_dataflow"] else None,
                            "state_output": attrs["source_partial_state_output"] if "source_partial_state_output" in attrs else None,
                            "source_slot_page_program": attrs["source_dataflow"]["slot_mapping_source"]
                                if "source_dataflow" in attrs and "slot_mapping_source" in attrs["source_dataflow"] else None,
                            "direct_input_producers": sorted(node["predecessors"]), "direct_consumers": sorted(successors[key]),
                            "finish_state_inputs": {reader: nodes[reader][0]["attributes"]["source_partial_state_input"]
                                if len(nodes[reader]) == 1 and "source_partial_state_input" in nodes[reader][0]["attributes"] else None
                                for reader in account["finish_state_inputs"]},
                            "lowered_writer_form_compatible": compatible,
                            "projection_ready_before_write": bool(lowered) and bool(calls[parent]) and dependencies_reach(
                                [item["call_id"] for item in calls[parent]], lowered[0]["dependencies"], True),
                            "input_readiness": {source: bool(lowered) and bool(calls[source]) and dependencies_reach(
                                [item["call_id"] for item in calls[source]], lowered[0]["dependencies"], True)
                                for source in account["input_readiness"]},
                            "scope": "source views and in-place writes for symbolic backing/strides; actual page allocation and cross-step reuse remain separate obligations"}
                    elif record["source_alias_mode"] == "modular_marlin_objects":
                        attributes = node["attributes"]
                        associated = calls[key]
                        compatible = len(associated) == 1
                        work = scratch = None
                        if compatible:
                            call, = associated
                            compatible = (call["kind"] == "FUSED_MOE" and call["variant"] == "routed_experts"
                                and call["dtype"] == "bf16" and call["rank"] == record["rank"]
                                and all(name in call["shape"] and call["shape"][name] == value for name, value in record["shape"].items())
                                and call["launch"] is not None and call["launch"]["algorithm"] == "vllm_marlin_moe_mxfp4_aggregate"
                                and call["attributes"]["routed_weight_dtype"] == "mxfp4"
                                and call["attributes"]["expert_parallel_size"] == 1)
                            resource = call["resource_input"]
                            work = resource["logical_work"] if "logical_work" in resource else None
                            scratch = resource["marlin_scratch"] if "marlin_scratch" in resource else None
                        external = [terminal_dependency(dependency) for call in associated for dependency in call["dependencies"]]
                        programs = None if work is None else {name: work[name] if name in work else None for name in account["source_programs"]}
                        axis_preserved = False
                        if len(associated) == 1:
                            lowered = associated[0]["attributes"]
                            vectors = ("tokens_per_expert", "local_tokens_per_expert", "owned_experts")
                            if all(name in attributes and name in lowered for name in vectors):
                                counts = tuple(attributes["tokens_per_expert"])
                                axis_preserved = (len(counts) == record["shape"]["experts"]
                                    and all(type(value) is int for entry in (attributes, lowered)
                                            for name in vectors for value in entry[name])
                                    and all(type(count) is int and 0 <= count <= record["shape"]["tokens"] for count in counts)
                                    and sum(counts) == record["shape"]["tokens"] * record["shape"]["topk"]
                                    and tuple(attributes["local_tokens_per_expert"]) == counts
                                    and tuple(attributes["owned_experts"]) == tuple(range(record["shape"]["experts"]))
                                    and all(tuple(attributes[name]) == tuple(lowered[name]) for name in vectors)
                                    and all("expert_rank" in entry and type(entry["expert_rank"]) is int
                                            and entry["expert_rank"] == 0 and "expert_parallel_size" in entry
                                            and type(entry["expert_parallel_size"]) is int and entry["expert_parallel_size"] == 1
                                            for entry in (attributes, lowered)))
                        source_alias_actuals[key] = {
                            "declaration": attributes["source_moe_io"] if "source_moe_io" in attributes else None,
                            "parameter_module": attributes["module_path"] if "module_path" in attributes else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "producer_outputs": {parent: nodes[parent][0]["attributes"]["source_tensor_outputs"]
                                if len(nodes[parent]) == 1 and "source_tensor_outputs" in nodes[parent][0]["attributes"] else None
                                for parent in record["required_ancestors"]},
                            "input_tensor_ports": attributes["source_tensor_inputs"] if "source_tensor_inputs" in attributes else None,
                            "routing_tensor_ports": attributes["source_routing_inputs"] if "source_routing_inputs" in attributes else None,
                            "output_tensor_roles": attributes["source_tensor_outputs"] if "source_tensor_outputs" in attributes else None,
                            "direct_consumers": sorted(successors[key]),
                            "consumer_tensor_inputs": {consumer: nodes[consumer][0]["attributes"]["source_tensor_inputs"]
                                if len(nodes[consumer]) == 1 and "source_tensor_inputs" in nodes[consumer][0]["attributes"] else None
                                for consumer in account["consumer_tensor_inputs"]},
                            "source_programs": programs,
                            "scratch_requests": (None if scratch is None else
                                {name: scratch[name] if name in scratch else None for name in account["scratch_requests"]}),
                            "scratch_runtime_claims": (None if scratch is None else
                                {name: scratch[name] if name in scratch else None for name in account["scratch_runtime_claims"]}),
                            "declared_expert_axis_preserved": bool(axis_preserved),
                            "lowered_tensor_form_compatible": compatible,
                            "lowered_input_readiness": sorted(external, key=lambda item: json.dumps(item, sort_keys=True)),
                            "consumer_reads_after_output": all(any(
                                terminal_dependency(dependency) == {"source_node_id": key, "terminal_source_call": True}
                                for call in calls[consumer] for dependency in call["dependencies"])
                                for consumer in account["consumer_tensor_inputs"])}
                    elif record["source_alias_mode"] == "bf16_lm_head_objects":
                        attributes = node["attributes"]
                        parent = account["direct_input_producers"][0]
                        consumer = account["direct_consumers"][0]
                        parent_node = nodes[parent][0] if len(nodes[parent]) == 1 else None
                        gather_node = nodes[consumer][0] if len(nodes[consumer]) == 1 else None
                        associated = calls[key]
                        compatible = len(associated) == 1
                        if compatible:
                            call = associated[0]
                            compatible = (call["kind"] == "GEMM" and call["variant"] == record["variant"]
                                and call["dtype"] == "bf16" and call["rank"] == record["rank"]
                                and call["source_node_ids"] == [key]
                                and call["shape"] == {**record["shape"], "dtype_bytes": 2}
                                # The canonical plain GEMM schema is homogeneous
                                # and unbiased; these fields belong to other
                                # variants/mixed output and are forbidden here.
                                and not {"weight_dtype", "weight_group_size", "has_bias"}.intersection(call["attributes"])
                                and "output_dtype" in call["attributes"] and call["attributes"]["output_dtype"] in (None, "bf16")
                                and (call["launch"] is None or call["launch"]["algorithm"] not in _PACKED_FP8_ALGORITHMS))
                        head_ids = [item["call_id"] for item in associated]
                        input_ids = [item["call_id"] for item in calls[parent]]
                        head_ready = (len(input_ids) == 1 and len(associated) == 1
                                      and associated[0]["dependencies"] == input_ids)
                        all_head_ids = [item["call_id"] for head in account["gather_input_producers"] for item in calls[head]]
                        gather_ready = (bool(head_ids) and len(all_head_ids) == len(account["gather_input_producers"])
                            and bool(calls[consumer]) and all(dependencies_reach(all_head_ids, item["dependencies"], True)
                                                            for item in calls[consumer]))
                        source_alias_actuals[key] = {
                            "declaration": attributes["source_lm_head_io"] if "source_lm_head_io" in attributes else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "input_tensor_ports": attributes["source_tensor_inputs"] if "source_tensor_inputs" in attributes else None,
                            "producer_output_ports": parent_node["attributes"]["source_tensor_outputs"]
                                if parent_node is not None and "source_tensor_outputs" in parent_node["attributes"] else None,
                            "output_tensor_roles": attributes["source_tensor_outputs"] if "source_tensor_outputs" in attributes else None,
                            "direct_consumers": sorted(successors[key]),
                            "gather_input_producers": sorted(gather_node["predecessors"]) if gather_node is not None else None,
                            "gather_input_ports": gather_node["attributes"]["source_tensor_inputs"]
                                if gather_node is not None and "source_tensor_inputs" in gather_node["attributes"] else None,
                            "lowered_head_compatible": bool(compatible), "head_reads_after_row_selection": bool(head_ready),
                            "gather_reads_after_all_heads": bool(gather_ready)}
                    elif record["source_alias_mode"] == "fp8_linear_objects":
                        attributes = node["attributes"]
                        external = {}
                        for call in calls[key]:
                            for dependency in call["dependencies"]:
                                if len(call_ids[dependency]) == 1 and call_ids[dependency][0]["source_node_ids"] == [key]:
                                    continue
                                entry = terminal_dependency(dependency)
                                external[json.dumps(entry, sort_keys=True)] = entry
                        matrices = [call for call in calls[key] if call["kind"] == "GEMM"]
                        wrapper = None
                        if len(matrices) == 1 and "logical_work" in matrices[0]["resource_input"]:
                            work = matrices[0]["resource_input"]["logical_work"]
                            if "linear_wrapper_source_program" in work:
                                wrapper = work["linear_wrapper_source_program"]
                        source_alias_actuals[key] = {
                            "declaration": attributes["source_linear_io"] if "source_linear_io" in attributes else None,
                            "parameter_module": attributes["module_path"] if "module_path" in attributes else None,
                            "input_tensor_ports": attributes["source_tensor_inputs"] if "source_tensor_inputs" in attributes else None,
                            "output_tensor_roles": attributes["source_tensor_outputs"] if "source_tensor_outputs" in attributes else None,
                            "direct_input_producers": sorted(node["predecessors"]),
                            "producer_outputs": [{"node_id": parent, **{field: nodes[parent][0][field] for field in ("rank", "kind", "shape", "dtype")}}
                                if len(nodes[parent]) == 1 else None for parent in sorted(record["required_ancestors"])],
                            "direct_consumers": sorted(successors[key]),
                            "consumer_tensor_inputs": {consumer: nodes[consumer][0]["attributes"]["source_tensor_inputs"]
                                if len(nodes[consumer]) == 1 and "source_tensor_inputs" in nodes[consumer][0]["attributes"] else None
                                for consumer in account["consumer_tensor_inputs"]},
                            "external_lowered_producers": [external[name] for name in sorted(external)],
                            "lowered_tensor_form_compatible": _fp8_tensor_form_matches(record, calls[key], reference["sources"]),
                            "wrapper_source_program": wrapper}
                        if ("source_callee_account" in record and len(matrices) == 1
                                and matrices[0]["launch"] is not None
                                and matrices[0]["launch"]["algorithm"] == "trtllm_fp8_blockscale_swapab_gemm"):
                            work = matrices[0]["resource_input"]["logical_work"] if "logical_work" in matrices[0]["resource_input"] else None
                            workspace = work["flashinfer_workspace_source_program"] if work is not None and "flashinfer_workspace_source_program" in work else None
                            source_callee_actuals[key] = {
                                "source_form": "installed_SDK_SM90_in_call_quantization",
                                "caller_source_contract_matched": source_alias_actuals[key] == record["source_alias_account"],
                                "runner_source_links_matched": runner_links_match,
                                "runner_history": attributes["source_linear_io"]["conditional_sdk_history"]
                                    if "source_linear_io" in attributes and "conditional_sdk_history" in attributes["source_linear_io"] else None,
                                "workspace_source_program": workspace,
                                "quantizer_then_matrix": work["source_substage_order"] if work is not None and "source_substage_order" in work else None,
                                "workspace_source_lifetime": workspace["source_lifetime"] if workspace is not None and "source_lifetime" in workspace else None,
                                "numeric_initial_H_provider_spills_and_allocator_accepted": False}
                    elif record["source_alias_mode"] != "mhc_wrapper_objects" or "source_input_binding_account" not in record:
                        raise ValueError("unknown or unbound wrapper alias mode")
                    else:
                        tensor_stages = [{**{field: call[field] for field in ("kind", "variant", "dtype", "rank")},
                            "shape": call["shape"],
                            "mhc_stage_work": call["resource_input"]["mhc_stage_work"]
                                if "mhc_stage_work" in call["resource_input"] else None} for call in calls[key]]
                        compatible_form = any(len(form) == len(tensor_stages) and all(
                            all(actual[field] == stage[field] for field in ("kind", "variant", "dtype", "rank"))
                            and _mhc_tensor_work_matches(stage["mhc_stage_work"], actual["mhc_stage_work"])
                            and all(name in actual["shape"] and actual["shape"][name] == value for name, value in stage["required_shape"].items())
                            for actual, stage in zip(tensor_stages, form, strict=True)) for form in record["source_alias_tensor_forms"])
                        source_alias_actuals[key] = {
                            "declaration": node["attributes"]["source_dataflow"] if "source_dataflow" in node["attributes"] else None,
                            "source_stream_bindings": actual_binding, "direct_consumers": sorted(successors[key]),
                            "consumer_tensor_inputs": {consumer: nodes[consumer][0]["attributes"]["source_tensor_inputs"]
                                if len(nodes[consumer]) == 1 and "source_tensor_inputs" in nodes[consumer][0]["attributes"] else None
                                for consumer in account["consumer_tensor_inputs"]},
                            "lowered_tensor_form_compatible": compatible_form}
                        # Source form compatibility does not select a backend.
                else:
                    if "source_alias_dependency_mode" in record:
                        mode = record["source_alias_dependency_mode"]
                        if mode not in ("mhc_layer_input_completion", "source_completion"):
                            raise ValueError("unknown source alias dependency mode")
                        dependency_reference = mhc_layer_input_dependency if mode == "mhc_layer_input_completion" else terminal_dependency
                    else:
                        dependency_reference = output_dependency
                    source_alias_actuals[key] = {
                        "declaration": node["attributes"]["source_dataflow"]
                            if "source_dataflow" in node["attributes"] else None,
                        "direct_input_producers": sorted(node["predecessors"]),
                        "producer_outputs": [{"node_id": operand["node_id"], **{field: nodes[operand["node_id"]][0][field]
                            for field in ("rank", "kind", "shape", "dtype")}}
                            if len(nodes[operand["node_id"]]) == 1 else None
                            for operand in account["producer_outputs"]],
                        "direct_consumers": sorted(successors[key]),
                        "lowered_calls": [{**{field: call[field] for field in
                            ("kind", "variant", "rank", "dtype", "shape", "attributes")},
                            "dependencies": sorted([dependency_reference(dependency) for dependency in call["dependencies"]],
                                                   key=lambda entry: json.dumps(entry, sort_keys=True))}
                            for call in calls[key]]}
                    if "consumer_tensor_inputs" in account:
                        source_alias_actuals[key]["consumer_tensor_inputs"] = {
                            consumer: nodes[consumer][0]["attributes"]["source_tensor_inputs"]
                            if len(nodes[consumer]) == 1 and "source_tensor_inputs" in nodes[consumer][0]["attributes"] else None
                            for consumer in account["consumer_tensor_inputs"]}
                    for field, attribute in (("input_tensor_ports", "source_tensor_inputs"), ("output_tensor_roles", "source_tensor_outputs")):
                        if field in account:
                            source_alias_actuals[key][field] = node["attributes"][attribute] if attribute in node["attributes"] else None
            stages = record["lowering_account"]
            stage_source = record["lowering_source"] if "lowering_source" in record else source
            if "mhc_nonfused_forms" in record:
                conditional = record["mhc_nonfused_forms"]
                compatible = []
                for form in conditional["forms"]:
                    if conditional["selected_backend"] is not None and form["backend"] != conditional["selected_backend"]:
                        continue
                    expected = form["stages"]
                    if len(calls[key]) != len(expected):
                        continue
                    if all(call["kind"] == stage["kind"] and call["variant"] == stage["variant"]
                           and all(field in call["shape"] and call["shape"][field] == value
                                   for field, value in stage["required_shape"].items())
                           and (conditional["selected_backend"] is None
                                or stage["variant"] != "mhc_prenorm"
                                or ("source_prenorm_backend" in call["attributes"]
                                    and call["attributes"]["source_prenorm_backend"] == conditional["selected_backend"]))
                           for call, stage in zip(calls[key], expected, strict=True)):
                        compatible.append(form)
                check(key, "mhc_nonfused_source_form", True, bool(compatible), stage_source)
                # Matching a source form checks its conditional arithmetic
                # and payload; it never supplies the missing runtime inputs.
                # In particular, H200 does not imply has_deep_gemm(). Keep
                # this required condition explicitly unverified even when
                # both forms happen to use the same split allocation.
                if conditional["selected_backend"] is None:
                    check(key, "mhc_nonfused_backend_condition",
                          [form["backend_condition"] for form in conditional["forms"]],
                          None, stage_source, "unchecked")
                else:
                    # The independent env declaration selects a source
                    # program. A candidate echo cannot supply that input,
                    # and a shape match never establishes a GPU provider.
                    check(key, "mhc_nonfused_backend_condition",
                          {"source_use_deep_gemm": False, "source_backend": "tilelang"},
                          {"source_use_deep_gemm": node["attributes"]["source_use_deep_gemm"]
                              if "source_use_deep_gemm" in node["attributes"] else None,
                           "source_backend": "tilelang" if compatible else None}, stage_source)
                if compatible:
                    stages = compatible[0]["stages"]
            if stages is None and "lowering_forms" in record:
                forms = record["lowering_forms"]
                signature = [(call["kind"], call["variant"]) for call in calls[key]]
                external_signature = [(stage["kind"], stage["variant"]) for stage in forms["external"]]
                internal_signature = [(stage["kind"], stage["variant"]) for stage in forms["in_call"]]
                if signature == external_signature:
                    stages, stage_source = forms["external"], "fp8-stage-forms"
                elif (signature == internal_signature and calls[key][0]["launch"] is not None
                      and calls[key][0]["launch"]["algorithm"] == "trtllm_fp8_blockscale_swapab_gemm"):
                    if reference["sources"]["flashinfer-input-quant"] is not None:
                        stages, stage_source = forms["in_call"], "flashinfer-input-quant"
                    else:
                        check(key, "conditional_fp8_sdk_source", "genuine SDK quantizer-to-matmul source", None,
                              "flashinfer-input-quant", "unchecked")
                else:
                    check(key, "conditional_fp8_call_packaging", [external_signature, internal_signature], signature,
                          "fp8-stage-forms", "mismatch")
            if stages is None:
                expected_stages = ({"required": "source-derived logical stage sequence",
                                    "unbound_condition": record["lowering_source_condition"]}
                                   if "lowering_source_condition" in record else "source-derived logical stage sequence")
                check(key, "logical_lowering_stages", expected_stages, None, stage_source, "unchecked")
            elif len(calls[key]) != len(stages):
                check(key, "logical_lowering_stage_count", len(stages), len(calls[key]), stage_source,
                      "missing" if len(calls[key]) < len(stages) else "duplicate")
            else:
                for index, (stage, call) in enumerate(zip(stages, calls[key], strict=True)):
                    subject = key + f":source-stage-{index}"
                    for field in ("kind", "variant", "dtype"):
                        check(subject, "stage_" + field, stage[field], call[field], stage_source)
                    check(subject, "stage_rank", record["rank"], call["rank"], stage_source)
                    if index:
                        check(subject, "previous_stage_dependency", True,
                              calls[key][index - 1]["call_id"] in call["dependencies"], stage_source)
                    for field in ("element_count", "dtype_bytes"):
                        if field in stage:
                            check(subject, "stage_" + field, stage[field], call["shape"][field], stage_source)
                    if "external_quant_work" in stage:
                        expected_work, resource = stage["external_quant_work"], call["resource_input"]
                        if "logical_work" not in resource:
                            check(subject, "external_fp8_quantizer_source_work", expected_work, None, stage_source, "unchecked")
                        else:
                            for field, expected in expected_work.items():
                                check(subject, "external_fp8_" + field, expected,
                                      resource["logical_work"][field] if field in resource["logical_work"] else None, stage_source)
                        check(subject, "quantizer_native_binding", ["unsupported", None],
                              [call["execution_support"], call["launch"]], stage_source)
                        check(subject, "conditional_quantizer_common_baseline_proof", True,
                              resource["all_source_programs_memory_dominated"] if "all_source_programs_memory_dominated" in resource else None,
                              stage_source)
                        for field in ("logical_bytes", "source_read_write_bytes"):
                            check(subject, "external_fp8_resource_" + field, expected_work[field],
                                  resource[field] if field in resource else None, stage_source)
                    if "required_shape" in stage:
                        for field, expected in stage["required_shape"].items():
                            check(subject, "stage_" + field, expected,
                                  call["shape"][field] if field in call["shape"] else None, stage_source)
                    if "required_attributes" in stage:
                        for field, expected in stage["required_attributes"].items():
                            check(subject, "stage_attribute_" + field, expected,
                                  call["attributes"][field] if field in call["attributes"] else None, stage_source)
                    if "unbound_library_call" in stage and stage["unbound_library_call"]:
                        check(subject, "library_launch_unbound", ["unsupported", None],
                              [call["execution_support"] if "execution_support" in call else None, call["launch"]], stage_source)
                    if "mhc_stage_work" in stage:
                        resource = call["resource_input"]
                        expected_work = stage["mhc_stage_work"]
                        if "mhc_stage_work" not in resource:
                            check(subject, "mhc_source_stage_account", expected_work, None, stage_source, "unchecked")
                        else:
                            work = resource["mhc_stage_work"]
                            for field in ("stage", "operands", "read_bytes", "write_bytes", "views", "semantic_fields"):
                                check(subject, "mhc_" + field, expected_work[field], work[field], stage_source)
                            for field in ("unpriced_typed_actions", "nonlinear_function_counts"):
                                check(subject, "mhc_" + field, expected_work[field],
                                      work[field] if field in work else None, stage_source)
                            for role, expected in expected_work["demand"].items():
                                check(subject, "mhc_work_" + role, expected, work["demand"][role], stage_source)
                                check(subject, "mhc_predictor_" + role, expected,
                                      resource["demand"][role], stage_source)
                            if "modeled_l2_bytes" in resource:
                                check(subject, "mhc_unique_operand_l2_scope",
                                      {"read": expected_work["read_bytes"], "write": expected_work["write_bytes"]},
                                      resource["modeled_l2_bytes"], stage_source)
                            else:
                                check(subject, "mhc_unique_operand_total", expected_work["read_bytes"] + expected_work["write_bytes"],
                                      resource["logical_bytes"], stage_source)
                    if "finish_stage" in stage:
                        check(subject, "finish_stage", stage["finish_stage"], call["attributes"]["finish_stage"], stage_source)
                    if "finish_work" in stage:
                        expected_work = stage["finish_work"]
                        resource = call["resource_input"]
                        if "finish_work" not in resource:
                            check(subject, "finish_source_work", expected_work, None, stage_source, "unchecked")
                        else:
                            work = resource["finish_work"]
                            for field, expected in expected_work.items():
                                check(subject, "finish_" + field, expected,
                                      work[field] if field in work else None, stage_source)
                            if "partial_state_windows" in record["attributes"]:
                                flow = record["attributes"]["partial_state_windows"]
                                windows = [request["completed_windows"] for request in flow["request_windows"]]
                                rows = sum(end - begin for sequence in windows for row in sequence
                                           for begin, end in [row["positions"]])
                                check(subject, "finish_state_window_source_rows", rows,
                                      work["valid_window_rows"] if "valid_window_rows" in work else None, "compressor-state-dataflow")
                                check(subject, "finish_state_window_source_boundaries",
                                      [[row["boundary_end_exclusive"] for row in sequence] for sequence in windows],
                                      work["boundary_ends_by_request"] if "boundary_ends_by_request" in work else None,
                                      "compressor-state-dataflow")
                                check(subject, "finish_state_window_source_dimension", flow["head_dim"],
                                      work["state_dimension"] if "state_dimension" in work else None,
                                      "compressor-state-dataflow")
                        for role, count in expected_work["demand"].items():
                            check(subject, "finish_predictor_" + role, count, resource["demand"][role], stage_source)
                    if "partial_state_input" in stage:
                        resource = call["resource_input"]
                        check(subject, "finish_partial_state_fp32_input", stage["partial_state_input"],
                              resource["partial_state_input"] if "partial_state_input" in resource else "missing input account",
                              "compressor-state-dataflow")
                    if "exp_calls" in stage:
                        check(subject, "source_exp_calls", stage["exp_calls"],
                              call["resource_input"]["demand"]["compute_sfu"], stage_source)
                    if "source_sfu_calls" in stage:
                        check(subject, "source_sfu_calls", stage["source_sfu_calls"],
                              call["resource_input"]["demand"]["compute_sfu"], stage["sfu_source"])
                    if "quantization_actions" in stage:
                        resource = call["resource_input"]
                        actions = (resource["finish_work"]["unpriced_typed_actions"]
                            if "finish_work" in resource and "unpriced_typed_actions" in resource["finish_work"] else {})
                        for name, expected in stage["quantization_actions"].items():
                            check(subject, "source_typed_" + name, expected,
                                actions[name] if name in actions else None, stage_source)
                    if "source_payload_bytes" in stage:
                        resource = call["resource_input"]
                        if "modeled_l2_bytes" not in resource:
                            check(subject, "source_stage_read_write", stage["source_payload_bytes"], None,
                                  stage_source, "unchecked")
                        else:
                            check(subject, "source_stage_read_write", stage["source_payload_bytes"],
                                  resource["modeled_l2_bytes"], stage_source)
                    if "output_write_bytes" in stage:
                        resource = call["resource_input"]
                        if "modeled_l2_bytes" in resource:
                            check(subject, "source_output_write_bytes", stage["output_write_bytes"],
                                  resource["modeled_l2_bytes"]["write"], stage_source)
                        else:
                            check(subject, "source_output_write_bytes", stage["output_write_bytes"], None,
                                  stage_source, "unchecked")
            if "indexer_baseline" in record:
                expected_work = record["indexer_baseline"]
                if len(calls[key]) != 1 or "indexer_logical_work" not in calls[key][0]["resource_input"]:
                    check(key, "indexer_useful_source_account", expected_work, None, "indexer-baseline", "unchecked")
                else:
                    call = calls[key][0]
                    check(key, "indexer_lowered_context_parallel_size", expected_work["source_semantics"]["context_parallel_size"],
                          call["attributes"]["context_parallel_size"] if "attributes" in call and "context_parallel_size" in call["attributes"] else None,
                          "indexer-baseline")
                    for field, expected in expected_work.items():
                        work = call["resource_input"]["indexer_logical_work"]
                        check(key, "indexer_" + field, expected, work[field] if field in work else None, "indexer-baseline")
                    check(key, "indexer_baseline_scope", expected_work["scope"], call["prediction"]["scope"], "indexer-baseline")
                    check(key, "indexer_useful_payload_total", expected_work["logical_bytes"],
                          call["resource_input"]["logical_bytes"], "indexer-baseline")
                # Useful-work agreement does not certify the omitted logical
                # gather/metadata/scratch or value-dependent selector work.
                check(key, "indexer_complete_source_services", expected_work["remaining_source_services"],
                      None, "indexer-baseline", "unchecked")
            if "route_source_work" in record:
                expected_work = record["route_source_work"]
                if len(calls[key]) != 1 or "route_source_work" not in calls[key][0]["resource_input"]:
                    check(key, "routing_source_account", expected_work, None, "v4-route-source", "unchecked")
                else:
                    call = calls[key][0]
                    resource = call["resource_input"]
                    actual_work = resource["route_source_work"]
                    for field, expected in expected_work.items():
                        check(key, "route_" + field, expected,
                              actual_work[field] if field in actual_work else None, "v4-route-source")
                    check(key, "route_declared_source", expected_work["source"],
                          call["attributes"]["routing_source"], "v4-route-source")
                    for field in ("expert_id_selection", "routed_scaling_factor"):
                        if field in record["attributes"]:
                            check(key, "route_lowered_" + field, record["attributes"][field],
                                  call["attributes"][field], "v4-route-source")
                    check(key, "route_baseline_extent", expected_work["logical_bytes"],
                          resource["logical_bytes"], "v4-route-source")
                    check(key, "route_baseline_byte_projection", expected_work["baseline_read_write_bytes"],
                          resource["modeled_dram_bytes"] if "modeled_dram_bytes" in resource else None, "v4-route-source")
            if "decode_index_materialization" in record:
                expected_indices = record["decode_index_materialization"]
                if len(calls[key]) != 1 or "decode_index_materialization" not in calls[key][0]["resource_input"]:
                    check(key, "decode_c4_index_materialization", expected_indices, None, "decode-c4-indices", "unchecked")
                else:
                    check(key, "decode_c4_index_materialization", expected_indices,
                          calls[key][0]["resource_input"]["decode_index_materialization"], "decode-c4-indices")
            if "attention_inputs" in record:
                expected_inputs = record["attention_inputs"]
                if len(calls[key]) != 1 or "attention_inputs" not in calls[key][0]["resource_input"]:
                    check(key, "attention_source_inputs", expected_inputs, None, "attention-sink", "unchecked")
                else:
                    check(key, "attention_source_inputs", expected_inputs,
                          calls[key][0]["resource_input"]["attention_inputs"], "attention-sink")
            if "decode_attention_index_inputs" in record:
                expected = record["decode_attention_index_inputs"]
                actual = (calls[key][0]["resource_input"]["decode_attention_index_inputs"]
                          if len(calls[key]) == 1 and "decode_attention_index_inputs" in calls[key][0]["resource_input"] else None)
                check(key, "decode_attention_index_inputs", expected, actual,
                      "decode-attention-index-inputs", "unchecked" if actual is None else None)
            if "source_work" in record:
                associated = calls[key]
                if len(associated) != 1:
                    check(key, "source_work_single_call", 1, len(associated), source, "unchecked")
                elif record["variant"] == "dsv4_indexer_q_rope_quant":
                    actual_work = associated[0]["resource_input"]["logical_work"]
                    for field in ("mathematical_primitives", "mathematical_primitives_condition", "backend_alternatives",
                                  "selected_backend", "numerical_equivalence_asserted", "q_scale_has_separate_buffer"):
                        if field not in actual_work:
                            check(key, field, record["source_work"][field], None, source, "unchecked")
                        else:
                            check(key, field, record["source_work"][field], actual_work[field], source)
                    check(key, "typed_payload_bytes", record["source_work"]["typed_payload_bytes"],
                          {operand["role"]: operand["bytes"] for operand in actual_work["operands"]}, source)
                    if "roofline_baseline" in record:
                        resource = associated[0]["resource_input"]
                        expected = record["roofline_baseline"]
                        if "roofline_source_programs" not in resource:
                            check(key, "indexer_q_arithmetic_byte_baseline", expected, None, source, "unchecked")
                        else:
                            check(key, "indexer_q_baseline_logical_bytes", expected["logical_bytes"], resource["logical_bytes"], source)
                            check(key, "indexer_q_conditional_roofline_programs", expected["source_programs"],
                                  resource["roofline_source_programs"], source)
                            prediction = associated[0]["prediction"]
                            check(key, "indexer_q_price_scope", expected["scope"], prediction["scope"], source)
                            check(key, "indexer_q_price_kind", ["roofline", "priced"],
                                  [prediction["method"], prediction["status"]], source)
                elif record["variant"] in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
                    check(key, "active_qk_tensor_flops", record["source_work"]["active_qk_tensor_flops"],
                          associated[0]["resource_input"]["demand"]["compute_tensor"], source)
                elif record["kind"] == "fused_moe" and record["variant"] == "full":
                    resource = associated[0]["resource_input"]
                    expected = record["source_work"]
                    moe_source = record["source_work_source"]
                    groups = expected["activation_resource_groups"]
                    primitives = expected["mathematical_primitives"]
                    scalar = {"compute_fma": groups["compute_fma"] + primitives["source_topk_fp32_add"],
                        "compute_sfu": groups["compute_sfu"],
                        "compute_alu": groups["compute_alu"]}
                    for role, count in scalar.items():
                        check(key, "marlin_live_scalar_" + role, count, resource["demand"][role], moe_source)
                    if "logical_work" not in resource:
                        check(key, "marlin_source_work", expected, None, moe_source, "unchecked")
                    else:
                        work = resource["logical_work"]
                        for field in expected:
                            if "source_work_unchecked_fields" in record and field in record["source_work_unchecked_fields"]:
                                check(key, "marlin_" + field, expected[field],
                                      work[field] if field in work else None, moe_source, "unchecked")
                            elif field not in work:
                                check(key, "marlin_" + field, expected[field], None, moe_source, "unchecked")
                            else:
                                check(key, "marlin_" + field, expected[field], work[field], moe_source)
                    templates = expected["roofline_stage_templates"]
                    attributes = associated[0]["attributes"]
                    tensor_work = None
                    if "local_tokens_per_expert" in attributes:
                        histogram = attributes["local_tokens_per_expert"]
                        if (all(type(count) is int and count >= 0 for count in histogram)
                                and sum(histogram) == expected["routed_rows"]):
                            tensor_work = marlin_tensor_work_reference(expected, histogram)
                            padded_demands = {stage["stage"]: stage["source_tensor_flops"]
                                              for stage in tensor_work["stages"]}
                            templates = [{**stage, "demand": {**stage["demand"],
                                **({"compute_tensor": padded_demands[stage["stage"]]}
                                   if stage["stage"] in padded_demands else {})}} for stage in templates]
                    for role in templates[0]["demand"]:
                        check(key, "marlin_source_baseline_" + role,
                              sum(stage["demand"][role] for stage in templates), resource["demand"][role], moe_source)
                    if "marlin_roofline" not in resource:
                        check(key, "marlin_serial_baseline", templates, None, moe_source, "unchecked")
                    else:
                        baseline = resource["marlin_roofline"]
                        if tensor_work is not None:
                            check(key, "marlin_padded_tensor_work", tensor_work,
                                  baseline["tensor_work"] if "tensor_work" in baseline else None, moe_source)
                            check(key, "marlin_typed_register_dequant_work", marlin_dequant_work_reference(expected, histogram),
                                  baseline["dequant_work"] if "dequant_work" in baseline else None, moe_source)
                            check(key, "marlin_source_weight_loads", marlin_weight_loads_reference(expected, histogram),
                                  baseline["weight_load_work"] if "weight_load_work" in baseline else None, moe_source)
                        else:
                            check(key, "marlin_padded_tensor_work", "valid explicit expert histogram",
                                  None, moe_source, "unchecked")
                        check(key, "marlin_baseline_stage_templates", templates,
                            [{field: stage[field] if field in stage else None for field in templates[0]}
                             for stage in baseline["stages"]], moe_source)
                        check(key, "marlin_baseline_composition", "sum_serial_stage_rooflines",
                              baseline["price_composition"], moe_source)
                        check(key, "marlin_baseline_not_checkpoint_routing", False,
                              baseline["routing_is_checkpoint_assignment"], moe_source)
                        if "gemm_metadata" in expected:
                            attributes = associated[0]["attributes"]
                            if "local_tokens_per_expert" not in attributes:
                                check(key, "marlin_declared_metadata_input", "explicit nonnegative expert histogram",
                                      None, moe_source, "unchecked")
                            else:
                                histogram = attributes["local_tokens_per_expert"]
                                valid = all(type(count) is int and count >= 0 for count in histogram)
                                check(key, "marlin_declared_metadata_histogram_valid", True, valid, moe_source)
                                if valid:
                                    check(key, "marlin_declared_metadata_route_count", expected["routed_rows"], sum(histogram), moe_source)
                                    block = expected["routing_alignment"]["block_size_m"]
                                    blocks = sum((count + block - 1) // block for count in histogram)
                                    active = sum(count > 0 for count in histogram)
                                    check(key, "marlin_declared_active_m_blocks", blocks,
                                          baseline["declared_active_m_blocks"] if "declared_active_m_blocks" in baseline else None, moe_source)
                                    check(key, "marlin_declared_active_experts", active,
                                          baseline["declared_active_experts"] if "declared_active_experts" in baseline else None, moe_source)
                                    matrix_templates = [stage for stage in templates if stage["stage"] in ("w13", "w2_with_router_weight")]
                                    reads = [{"stage": stage["stage"], "read": stage["nonweight_read_write_bytes"]["read"]
                                        + blocks * sum(stage["weight_payload_per_active_expert_bytes"].values())
                                        + blocks * stage["metadata_read_bytes_per_active_m_block"] + stage["metadata_read_bytes_per_stage"]}
                                        for stage in matrix_templates]
                                    actual_reads = [{"stage": stage["stage"], "read": stage["read_write_bytes"]["read"]
                                        if "read_write_bytes" in stage else None}
                                        for stage in baseline["stages"] if stage["stage"] in ("w13", "w2_with_router_weight")]
                                    check(key, "marlin_declared_matrix_read_bytes", reads, actual_reads, moe_source)
                        if "workspace_zero_bytes" in record:
                            zero_stages = [stage for stage in baseline["stages"] if stage["stage"] == "workspace_zero"]
                            check(key, "marlin_workspace_zero_read_write", [{"read": 0, "write": record["workspace_zero_bytes"]}],
                                  [stage["read_write_bytes"] if "read_write_bytes" in stage else None for stage in zero_stages], moe_source)
                    if "scratch_allocation" in record:
                        check(key, "marlin_scratch_allocation", record["scratch_allocation"],
                              {field: resource["marlin_scratch"][field] if field in resource["marlin_scratch"] else None
                               for field in record["scratch_allocation"]} if "marlin_scratch" in resource else None, moe_source,
                              None if "marlin_scratch" in resource else "unchecked")
                        attributes = associated[0]["attributes"]
                        has_bounds = ("marlin_scratch" in resource and "conditional_access_bounds" in resource["marlin_scratch"])
                        if "local_tokens_per_expert" in attributes and has_bounds:
                            expected_bounds = marlin_scratch_bounds_reference(expected,
                                attributes["local_tokens_per_expert"], record["scratch_allocation"]["sm_count"])
                            check(key, "marlin_conditional_scratch_access_bounds", expected_bounds,
                                  resource["marlin_scratch"]["conditional_access_bounds"]
                                  if has_bounds else None, moe_source)
                        else:
                            check(key, "marlin_conditional_scratch_access_bounds",
                                  "conditional source census under an explicit valid histogram, not point traffic",
                                  None, moe_source, "unchecked")
                elif record["variant"] == "dsv4_sparse_prefill":
                    resource = associated[0]["resource_input"]
                    expected = record["source_work"]
                    gather_source = record["source_work_source"]
                    check(key, "active_qk_pv_tensor_flops", expected["active_qk_pv_tensor_flops"],
                          resource["demand"]["compute_tensor"], gather_source)
                    if "logical_work" not in resource:
                        check(key, "prefill_gather_source_account", expected, None, gather_source, "unchecked")
                    else:
                        work = resource["logical_work"]
                        for field in ("gathered_rows", "pair_capacities", "common_useful_fp_multiply_values", "workspace",
                                      "backend_alternatives", "source_operation_order", "selected_backend", "numerical_equivalence_asserted",
                                      "index_materialization"):
                            field_source = "prefill-combine-indices" if field == "index_materialization" else gather_source
                            if field not in work:
                                check(key, "prefill_" + field, expected[field], None, field_source, "unchecked")
                            else:
                                check(key, "prefill_" + field, expected[field], work[field], field_source)
                        check(key, "prefill_baseline_logical_byte_projection", expected["baseline_logical_bytes"],
                              resource["logical_bytes"], "prefill-combine-indices")
                        check(key, "prefill_useful_dequant_charged", expected["common_useful_fp_multiply_values"],
                              resource["demand"]["compute_fma"] - work["non_gather_scalar_fma_group"], gather_source)
                else:
                    raise ValueError("source_work has no independent comparator")
            if record["kind"] not in ("linear", "router", "lm_head"):
                account_source = record["account_source"] if "account_source" in record else source
                associated = calls[key]
                math = record["math_account"]
                byte = record["byte_account"]
                if math is not None and "mode" in math and math["mode"] == "source_stages":
                    # Every arithmetic/byte stage was checked above. Stage
                    # count failures remain failures, never an aggregate pass.
                    continue
                if math is None:
                    check(key, "math_and_comparisons", "complete source arithmetic/comparison account", None,
                          account_source, "unchecked")
                elif len(associated) != 1:
                    check(key, "math_single_call", 1, len(associated), account_source, "unchecked")
                elif "demand" in math:
                    actual_resource = associated[0]["resource_input"]
                    actual_demand = actual_resource["demand"]
                    if actual_demand is None:
                        check(key, "source_arithmetic", math["demand"], None, account_source, "unchecked")
                    else:
                        check(key, "source_arithmetic", math["demand"],
                              {name: actual_demand[name] for name in math["demand"]}, account_source)
                    if "semantic_fields" in math:
                        for field, expected in math["semantic_fields"].items():
                            if "logical_work" not in actual_resource:
                                check(key, field, expected, None, account_source, "unchecked")
                            else:
                                check(key, field, expected, actual_resource["logical_work"][field], account_source)
                    if "argmax_pair" in math:
                        work = actual_resource["logical_work"]
                        if reference["sources"]["argmax-pair"] is None:
                            check(key, "argmax_source_header", "Torch 2.11 GreaterOrNan", None, "argmax-pair", "unchecked")
                        else:
                            program = work["logical_pair_program"]
                            for field, expected in math["argmax_pair"].items():
                                check(key, "argmax_" + field, expected, program[field], "argmax-pair")
                            check(key, "mathematical_value_comparisons", math["argmax_pair"]["logical_pair_merges"],
                                  work["mathematical_value_comparisons"], "argmax-pair")
                elif "mode" in math and math["mode"] == "collective_semantics":
                    transfer = associated[0]["transfer"] if "transfer" in associated[0] else None
                    if transfer is None:
                        check(key, "collective_math_contract", math, None, account_source, "unchecked")
                    else:
                        for field in ("operation", "reduction", "participants"):
                            check(key, "collective_" + field, math[field], transfer[field], account_source)
                        for predecessor in record["required_ancestors"]:
                            actual_dtype = (nodes[predecessor][0]["dtype"]
                                            if predecessor in nodes and len(nodes[predecessor]) == 1 else None)
                            check(key, "collective_input_dtype:" + predecessor, math["input_dtype"], actual_dtype, account_source)
                        # Algebra of the exported operation contract, never
                        # a prediction of library instruction/traffic work.
                        adds = ((len(transfer["participants"]) - 1) * (transfer["bytes"] // 2)
                                if transfer["operation"] == "all_reduce" and transfer["reduction"] == "sum" else 0)
                        check(key, "collective_minimum_global_sum_adds", math["minimum_global_sum_adds"], adds, account_source)
                elif "mode" in math and math["mode"] == "logical_copy_program":
                    resource = associated[0]["resource_input"]
                    for field, expected in math["semantic_fields"].items():
                        if "logical_work" not in resource or field not in resource["logical_work"]:
                            check(key, "copy_" + field, expected, None, account_source, "unchecked")
                        else:
                            check(key, "copy_" + field, expected, resource["logical_work"][field], account_source)
                elif "primitive_fields" in math:
                    resource = associated[0]["resource_input"]
                    if "logical_work" not in resource:
                        check(key, "mathematical_primitives", math["primitive_fields"], None, account_source, "unchecked")
                    elif "primitive_condition" in math:
                        work = resource["logical_work"]
                        if "mathematical_primitives_condition" not in work:
                            check(key, "conditional_mathematical_primitives", math, None, account_source, "unchecked")
                        else:
                            check(key, "conditional_mathematical_primitives", math,
                                {"primitive_fields": work["mathematical_primitives"],
                                 "primitive_condition": work["mathematical_primitives_condition"]}, account_source)
                    else:
                        work = resource["logical_work"]
                        check(key, "mathematical_primitives", math["primitive_fields"],
                              work["mathematical_primitives"], account_source)
                    if "semantic_fields" in math:
                        for field, expected in math["semantic_fields"].items():
                            if "logical_work" not in resource or field not in resource["logical_work"]:
                                check(key, field, expected, None, account_source, "unchecked")
                            else:
                                check(key, field, expected, resource["logical_work"][field], account_source)
                else:
                    raise ValueError("independent math account has no comparator")
                if byte is None:
                    check(key, "read_write_bytes", "complete source typed read/write account", None, account_source, "unchecked")
                elif byte["mode"] == "source_stages":
                    if stages is None or not stages or any("source_payload_bytes" not in stage for stage in stages):
                        raise ValueError("stage byte account requires an explicit ledger for every source stage")
                elif len(associated) != 1:
                    check(key, "bytes_single_call", 1, len(associated), account_source, "unchecked")
                elif byte["mode"] == "collective_payload":
                    transfer = associated[0]["transfer"] if "transfer" in associated[0] else None
                    if transfer is None:
                        check(key, "collective_payload_contract", byte, None, account_source, "unchecked")
                    else:
                        check(key, "collective_bytes_semantics", byte["bytes_semantics"], transfer["bytes_semantics"], account_source)
                        check(key, "collective_per_rank_input_bytes", byte["per_rank_input_bytes"], transfer["bytes"], account_source)
                        output_bytes = (transfer["bytes"] * len(transfer["participants"])
                                        if transfer["operation"] == "all_gather" else transfer["bytes"])
                        check(key, "collective_per_rank_output_bytes", byte["per_rank_output_bytes"], output_bytes, account_source)
                elif byte["mode"] in ("modeled_dram_bytes", "modeled_l2_bytes"):
                    actual_resource = associated[0]["resource_input"]
                    expected = {name: byte[name] for name in ("read", "write")}
                    mode = byte["mode"]
                    if mode not in actual_resource:
                        check(key, mode + "_read_write", expected, None, account_source, "unchecked")
                    else:
                        check(key, mode + "_read_write", expected, actual_resource[mode], account_source)
                elif byte["mode"] == "logical_operands":
                    resource = associated[0]["resource_input"]
                    if "logical_work" not in resource:
                        check(key, "logical_typed_operands", byte["operands"], None, account_source, "unchecked")
                    else:
                        actual_operands = resource["logical_work"]["operands"]
                        fields = ("role", "dtype", "elements", "bytes", "access")
                        actual = [{name: operand[name] for name in fields} for operand in actual_operands]
                        check(key, "logical_typed_operands", byte["operands"], actual, account_source)
                        if "logical_bytes" in byte:
                            check(key, "source_materialization_logical_bytes", byte["logical_bytes"],
                                  resource["logical_bytes"] if "logical_bytes" in resource else None, account_source)
                        if math is not None and "mode" in math and math["mode"] == "logical_copy_program":
                            check(key, "embedding_baseline_logical_bytes", sum(item["bytes"] for item in byte["operands"]),
                                  resource["logical_bytes"] if "logical_bytes" in resource else None, account_source)
                else:
                    raise ValueError("independent byte account has no comparator")
        for record in reference["projections"]:
            key = record["node_id"]
            if not key.startswith(phase + ":"):
                continue
            # Resolve this key through source_locations/reference.sources;
            # do not duplicate whole constructor ASTs for every field/rank.
            source = record["source"]
            if key not in nodes:
                check(key, "presence", 1, 0, source, "missing")
                continue
            if len(nodes[key]) != 1:
                check(key, "unique_node", 1, len(nodes[key]), source, "duplicate")
                continue
            node = nodes[key][0]
            for field, actual in (("shape", node["shape"]), ("rank", node["rank"]),
                                  ("input_dtype", node["dtype"]),
                                  ("weight_dtype", node["attributes"]["weight_dtype"]),
                                  ("output_dtype", node["attributes"]["output_dtype"]
                                   if "output_dtype" in node["attributes"] else node["dtype"])):
                check(key, field, record[field], actual, source)
            lowered = [call for call in calls[key] if call["kind"] == "GEMM"]
            if len(lowered) != 1:
                check(key, "unique_lowered_projection", 1, len(lowered), source,
                      "missing" if not lowered else "duplicate")
                continue
            call = lowered[0]
            check(key, "lowered_shape", record["shape"],
                  {name: call["shape"][name] for name in ("b", "m", "n", "k")}, source)
            packed = call["launch"] is not None and call["launch"]["algorithm"] in _PACKED_FP8_ALGORITHMS
            actual_weight = "fp8" if packed else (
                call["attributes"]["weight_dtype"] if "weight_dtype" in call["attributes"] else call["dtype"])
            check(key, "lowered_weight_dtype", record["weight_dtype"], actual_weight, source)
            check(key, "lowered_output_dtype", record["output_dtype"],
                  call["attributes"]["output_dtype"] if call["attributes"]["output_dtype"] is not None else call["dtype"], source)
            demand = call["resource_input"]["demand"]
            if packed and "logical_work" in call["resource_input"]:
                work = call["resource_input"]["logical_work"]
                if "linear_wrapper_source_program" in record:
                    check(key, "linear_wrapper_source_program_exported", True,
                          "linear_wrapper_source_program" in work, "fp8-linear-input-quant")
                    if "linear_wrapper_source_program" in work:
                        check(key, "linear_wrapper_source_program", record["linear_wrapper_source_program"],
                              work["linear_wrapper_source_program"], "fp8-linear-input-quant")
                if "flashinfer_workspace_source_program" in record:
                    workspace_source = "flashinfer-input-quant"
                    expected_workspace = record["flashinfer_workspace_source_program"]
                    if reference["sources"][workspace_source] is None:
                        check(key, "flashinfer_workspace_source_program", expected_workspace, None,
                              workspace_source, "unchecked")
                    elif "flashinfer_workspace_source_program" not in work:
                        check(key, "flashinfer_workspace_source_program", expected_workspace, None,
                              workspace_source, "missing")
                    else:
                        check(key, "flashinfer_workspace_source_program", expected_workspace,
                              work["flashinfer_workspace_source_program"], workspace_source)
                if "lowering_forms" in operator_by_id[key]:
                    expected_order = (["quantize_bf16_and_write_scales", "packed_fp8_matmul"]
                                      if call["launch"]["algorithm"] == "trtllm_fp8_blockscale_swapab_gemm" else ["packed_fp8_matmul"])
                    check(key, "fp8_source_substage_order", expected_order,
                          work["source_substage_order"] if "source_substage_order" in work else None,
                          "fp8-stage-forms", "unchecked" if "source_substage_order" not in work else None)
                check(key, "mathematical_flops", record["mathematical_flops"], work["mathematical_flops"], source)
                check(key, "typed_operand_payload_bytes", record["operand_payload_bytes"], work["operand_payload_bytes"], source)
                shape = record["shape"]
                kblocks, nblocks = (shape["k"] + 127) // 128, (shape["n"] + 127) // 128
                expected_scales = {"activation_scales": 4 * shape["b"] * shape["m"] * kblocks,
                                   "weight_scales": 4 * shape["b"] * nblocks * kblocks}
                check(key, "FP32_template_scale_read_bytes", expected_scales, work["scale_bytes"], source)
                if "input_quantizer_operands" in record:
                    expected_quant = record["input_quantizer_operands"]
                    quantizers = [item for item in calls[key] if item["kind"] == "VECTOR" and item["variant"] == "fp8_group_quant"]
                    if len(quantizers) > 1:
                        check(key, "unique_input_quantizer", 1, len(quantizers), "fp8-linear-input-quant", "duplicate")
                    elif quantizers:
                        quant = quantizers[0]
                        for field, expected, actual in (
                            ("input_dtype", expected_quant["input_dtype"], quant["dtype"]),
                            ("elements", expected_quant["fp8_output_write_bytes"], quant["shape"]["element_count"]),
                            ("group_size", expected_quant["group_size"], quant["attributes"]["group_size"]),
                            ("producer_dependency", True, quant["call_id"] in call["dependencies"])):
                            check(key, "quantizer_" + field, expected, actual, "fp8-linear-input-quant")
                        expected_bytes = {"read": expected_quant["bf16_input_read_bytes"],
                                          "write": expected_quant["fp8_output_write_bytes"] + expected_quant["scale_output_write_bytes"]}
                        resource = quant["resource_input"]
                        if "source_read_write_bytes" in resource:
                            check(key, "quantizer_source_read_write", expected_bytes, resource["source_read_write_bytes"], "fp8-stage-forms")
                        elif "modeled_l2_bytes" not in resource:
                            check(key, "quantizer_read_write", expected_bytes, None, "fp8-linear-input-quant", "unchecked")
                        else:
                            check(key, "quantizer_read_write", expected_bytes, resource["modeled_l2_bytes"], "fp8-linear-input-quant")
                        if "activation_quantizer" not in work:
                            check(key, "matmul_quantizer_account", None, "not exported", "fp8-linear-input-quant", "unchecked")
                        else:
                            check(key, "separate_quantizer_not_repeated_in_matmul", None, work["activation_quantizer"], "fp8-linear-input-quant")
                            check(key, "matmul_only_operand_role_bytes",
                                  sum(record["operand_payload_bytes"].values()) + sum(expected_scales.values()),
                                  work["logical_bytes"], "fp8-linear-input-quant")
                    elif call["launch"]["algorithm"] == "trtllm_fp8_blockscale_swapab_gemm":
                        quant_source = "flashinfer-input-quant"
                        if reference["sources"][quant_source] is None:
                            check(key, "in_call_quantizer_source_binding", "genuine scale_1x128 SDK source", None, quant_source, "unchecked")
                        else:
                            expected_total = (sum(record["operand_payload_bytes"].values()) + sum(expected_scales.values())
                                + expected_quant["bf16_input_read_bytes"] + expected_quant["fp8_output_write_bytes"]
                                + expected_quant["scale_output_write_bytes"])
                            check(key, "whole_call_operand_role_bytes", expected_total, work["logical_bytes"], quant_source)
                            if "activation_quantizer" not in work or work["activation_quantizer"] is None:
                                check(key, "in_call_quantizer_account", expected_quant, None, quant_source, "missing")
                            else:
                                quant = work["activation_quantizer"]
                                for field, expected in expected_quant.items():
                                    check(key, "in_call_quantizer_" + field, expected, quant[field], quant_source)
                                check(key, "quantizer_output_read_by_matmul", True, quant["matmul_reads_quantized_output"], quant_source)
                                if "input_quantizer_source_program" in record:
                                    expected_program = record["input_quantizer_source_program"]
                                    check(key, "in_call_baseline_multiplies", expected_program["source_expression_counts"]["fp32_value_multiply"],
                                          quant["common_fp_multiply_values"], quant_source)
                                    if "source_program" not in quant:
                                        check(key, "in_call_quantizer_source_program", expected_program, None,
                                              quant_source, "unchecked")
                                    else:
                                        for field, expected in expected_program.items():
                                            check(key, "quantizer_program_" + field, expected,
                                                  quant["source_program"][field] if field in quant["source_program"] else None,
                                                  quant_source)
                    else:
                        check(key, "input_quantizer_placement", "source branch binding required", None, "fp8-linear-input-quant", "unchecked")
                    check(key, "FP32_scale_producer_consumer_extent", expected_quant["scale_output_write_bytes"],
                          work["scale_bytes"]["activation_scales"], "fp8-linear-input-quant")
                elif record["role"] == "wo-a":
                    if "activation_quantizer" not in work:
                        check(key, "prequantized_einsum_producer_account", None, "not exported", source, "unchecked")
                    else:
                        check(key, "prequantized_einsum_does_not_repeat_quantization", None, work["activation_quantizer"], source)
            elif demand is None:
                check(key, "mathematical_flops", record["mathematical_flops"], None, source, "unchecked")
                check(key, "typed_operand_payload_bytes", record["operand_payload_bytes"], None, source, "unchecked")
            else:
                check(key, "mathematical_flops", record["mathematical_flops"],
                      demand["compute_tensor"] + demand["compute_fma"], source)
                check(key, "unique_operand_payload_total_bytes", sum(record["operand_payload_bytes"].values()),
                      call["resource_input"]["logical_bytes"], source)
    if "required_dataflow_obligations" in reference:
        obligations = reference["required_dataflow_obligations"]
        facets = ("tensor_alias_and_temporary_materialization", "persistent_cache_and_cross_step_state", "fp8_callee_materialization")
        if any(obligation["facet"] not in facets for obligation in obligations):
            raise ValueError("unknown required dataflow obligation facet")
        for facet in facets:
            required = [record["node_id"] for record in reference["operators"]
                        if (facet == facets[0] or (facet == facets[1] and record["kind"] in ("kv_update", "eager_attn"))
                            or (facet == facets[2] and "source_alias_mode" in record and record["source_alias_mode"] == "fp8_linear_objects"))]
            declared = [obligation["subject"] for obligation in obligations if obligation["facet"] == facet]
            check("full_model", "required_dataflow_inventory:" + facet, sorted(required), sorted(declared),
                  "independent logical operator inventory")
        for obligation in obligations:
            if "source_alias_account" in obligation:
                check(obligation["subject"], "required_dataflow:" + obligation["facet"],
                      obligation["source_alias_account"], source_alias_actuals[obligation["subject"]]
                      if obligation["subject"] in source_alias_actuals else None,
                      obligation["source"])
            elif "source_callee_account" in obligation and obligation["subject"] in source_callee_actuals:
                check(obligation["subject"], "required_dataflow:" + obligation["facet"],
                      obligation["source_callee_account"], source_callee_actuals[obligation["subject"]], obligation["source"])
            else:
                check(obligation["subject"], "required_dataflow:" + obligation["facet"],
                      {"requirement": obligation["expected"], "missing": obligation["missing"]}, None,
                      obligation["source"], "unchecked")
    for facet in reference["remaining_required_facets"]:
        check("full_model", facet, "independent derivation and comparison required", None,
              "user alignment scope", "unchecked")
    counts = Counter(record["status"] for record in checks)
    summary = {name: counts[name] for name in ("matched", "missing", "duplicate", "mismatch", "unchecked", "not_applicable")}
    summary["total"] = len(checks)
    return {"scope": reference["scope"], "candidate_snapshot": candidate["snapshot"],
            "source_locations": reference["sources"],
            "summary": summary, "alignment_passed": all(summary[name] == 0 for name in
                ("missing", "duplicate", "mismatch", "unchecked")),
            "checks": checks, "limitations": reference["limitations"]}


def _compare_flash_case(arguments):
    config_path, source_root, prefill_tokens, path, torch_include, flashinfer_source_root, input_path, source_max_model_len, source_use_deep_gemm = arguments
    inputs = None if input_path is None else json.loads(Path(input_path).read_text(encoding="utf-8"))
    reference = build_flash_reference(config_path, source_root, prefill_tokens=prefill_tokens,
                                      torch_include=torch_include, flashinfer_source_root=flashinfer_source_root,
                                      embedding_token_ids=inputs, v4_source_max_model_len=source_max_model_len,
                                      v4_source_use_deep_gemm=source_use_deep_gemm)
    candidate = _read_json_artifact(path)
    phase_pricing = []
    for step in candidate["steps"]:
        unpriced = [call for call in step["calls"] if call["prediction"]["status"] not in ("priced", "logical_alias")]
        phase_pricing.append({"phase": step["phase"], "calls": len(step["calls"]),
            "unpriced_calls": len(unpriced), "unpriced_variants": dict(Counter(call["variant"] for call in unpriced)),
            "logical_timeline_present": step["logical_roofline_timeline"] is not None})
    return (reference, {"candidate_path": str(path), "prefill_tokens": prefill_tokens,
                        "independent_case_input_path": None if input_path is None else str(Path(input_path).resolve()),
                        "phase_pricing": phase_pricing, **compare_flash_projections(reference, candidate)},
            {"prefill_tokens": prefill_tokens, "token_ids": reference["embedding_inputs"]})


def write_flash_alignment(config_path, source_root, candidates, output, *, torch_include=None, flashinfer_source_root=None,
                          case_inputs=(), v4_source_max_model_len=None, v4_source_use_deep_gemm=None,
                          compress_artifacts=False):
    """Compare independent source cases concurrently, then publish one report."""
    if not candidates:
        raise ValueError("Flash alignment requires at least one declared source case")
    if type(compress_artifacts) is not bool:
        raise TypeError("compress_artifacts must be boolean")
    tokens_set = {tokens for tokens, _ in candidates}
    if len(tokens_set) != len(candidates):
        raise ValueError("duplicate Flash candidate case")
    input_paths = dict(case_inputs)
    if len(input_paths) != len(case_inputs) or not set(input_paths).issubset(tokens_set):
        raise ValueError("independent case inputs must be unique and match a declared candidate case")
    arguments = [(config_path, source_root, tokens, path, torch_include, flashinfer_source_root,
                  input_paths[tokens] if tokens in input_paths else None, v4_source_max_model_len, v4_source_use_deep_gemm)
                 for tokens, path in candidates]
    if len(arguments) == 1:
        results = [_compare_flash_case(arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(arguments)) as pool:
            results = list(pool.map(_compare_flash_case, arguments))
    references, comparisons, reproduction_inputs = (list(values) for values in zip(*results, strict=True))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = _write_json_artifact(output / "reference.json", {"references": references}, compress_artifacts)
    totals = {name: sum(result["summary"][name] for result in comparisons)
              for name in ("total", "matched", "missing", "duplicate", "mismatch", "unchecked", "not_applicable")}
    passed = bool(comparisons) and all(result["alignment_passed"] for result in comparisons)
    root = Path(__file__).resolve().parents[3]
    generator_snapshot = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                 capture_output=True, text=True).stdout.strip(),
        "uncommitted_changes": subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                              capture_output=True, text=True).stdout.splitlines()}
    result = {"summary": totals, "alignment_passed": passed, "comparisons": comparisons,
              "reference_generator_snapshot": generator_snapshot}
    result["coverage"] = {"expected_logical_operators": sum(len(ref["operators"]) for ref in references),
                          "expected_dense_projections": sum(len(ref["projections"]) for ref in references),
                          "expanded_dataflow_obligations": sum(len(ref["required_dataflow_obligations"]) for ref in references),
                          "expanded_alias_obligations": sum(item["facet"] == "tensor_alias_and_temporary_materialization"
                              for ref in references for item in ref["required_dataflow_obligations"]),
                          "expanded_cache_state_obligations": sum(item["facet"] == "persistent_cache_and_cross_step_state"
                              for ref in references for item in ref["required_dataflow_obligations"]),
                          "expanded_fp8_callee_obligations": sum(item["facet"] == "fp8_callee_materialization"
                              for ref in references for item in ref["required_dataflow_obligations"]),
                          "source_alias_accounts": sum("source_alias_account" in item
                              for ref in references for item in ref["required_dataflow_obligations"]),
                          "matched_conditional_fp8_callee_accounts": sum(
                              check["field"] == "required_dataflow:fp8_callee_materialization" and check["status"] == "matched"
                              for comparison in comparisons for check in comparison["checks"]),
                          "full_required_facet_denominator_complete": False}
    families = {}
    for reference, comparison in zip(references, comparisons, strict=True):
        by_node = {record["node_id"]: record for record in reference["operators"]}
        for operator in reference["operators"]:
            family = operator["kind"] + "/" + str(operator["variant"])
            if family not in families:
                families[family] = {"instances": 0, "arithmetic": set(), "bytes": set(), "lowering": set(), "dataflow": set()}
            families[family]["instances"] += 1
        for check in comparison["checks"]:
            if check["status"] != "unchecked":
                continue
            subject = check["subject"].split(":source-stage-", 1)[0]
            if subject not in by_node:
                continue
            operator = by_node[subject]
            family = operator["kind"] + "/" + str(operator["variant"])
            identity = (comparison["candidate_path"], subject)
            field = check["field"]
            if field.startswith("required_dataflow:"):
                families[family]["dataflow"].add(identity)
            elif field == "finish_source_work":
                families[family]["arithmetic"].add(identity)
            elif field == "mhc_source_stage_account":
                families[family]["arithmetic"].add(identity)
                families[family]["bytes"].add(identity)
            elif field == "routing_source_account":
                families[family]["arithmetic"].add(identity)
                families[family]["bytes"].add(identity)
            elif field in ("indexer_useful_source_account", "indexer_complete_source_services"):
                families[family]["arithmetic"].add(identity)
                families[family]["bytes"].add(identity)
                families[family]["lowering"].add(identity)
            elif field == "marlin_source_work":
                families[family]["arithmetic"].add(identity)
                families[family]["bytes"].add(identity)
            elif field == "indexer_q_arithmetic_byte_baseline":
                families[family]["arithmetic"].add(identity)
                families[family]["bytes"].add(identity)
            elif field in ("math_and_comparisons", "source_arithmetic", "argmax_source_header", "math_single_call",
                           "mathematical_primitives", "conditional_mathematical_primitives",
                           "mathematical_primitives_condition", "backend_alternatives", "source_programs",
                           "selected_implementation", "selected_backend", "numerical_equivalence_asserted",
                           "collective_math_contract") or field.startswith("copy_"):
                families[family]["arithmetic"].add(identity)
            elif field in ("read_write_bytes", "bytes_single_call", "modeled_l2_bytes_read_write",
                           "modeled_dram_bytes_read_write", "source_output_write_bytes", "source_stage_read_write",
                           "logical_typed_operands", "collective_payload_contract"):
                families[family]["bytes"].add(identity)
            elif field in ("logical_lowering_stages", "mhc_nonfused_backend_condition"):
                families[family]["lowering"].add(identity)
    result["family_coverage"] = [{"family": family, "instances": values["instances"],
        "arithmetic_unchecked_instances": len(values["arithmetic"]),
        "bytes_unchecked_instances": len(values["bytes"]),
        "lowering_unchecked_instances": len(values["lowering"]),
        "dataflow_unchecked_instances": len(values["dataflow"])} for family, values in sorted(families.items())]
    comparison_path = _write_json_artifact(output / "comparison.json", result, compress_artifacts)
    _write_reproduction(config_path, source_root, reproduction_inputs, output, torch_include, flashinfer_source_root,
                        v4_source_max_model_len, v4_source_use_deep_gemm, compress_artifacts)
    phase_count = sum(len(comparison["phase_pricing"]) for comparison in comparisons)
    priced_phase_count = sum(phase["unpriced_calls"] == 0 and phase["logical_timeline_present"]
        for comparison in comparisons for phase in comparison["phase_pricing"])
    unpriced_calls = sum(phase["unpriced_calls"] for comparison in comparisons for phase in comparison["phase_pricing"])
    rows = ["# DeepSeek-V4 Flash 源码静态对齐", "",
            "状态：" + ("声明范围对齐通过，待主 agent 独立复核。" if passed else "未完成。已比较声明 case 的完整逻辑图及投影，包含短上下文/完整 indexer 分支；数据 alias、非投影运算/字节等剩余要求仍显式未检查。"), "",
            "| 交付项 | 当前结果 |", "|---|---|",
            "| 全模型算子接入验收 | 未完成；已有全层清单，仍须核对剩余数学、量化及数据流 |",
            f"| 独立源码对齐 | {'待独立复核' if passed else '未完成'}；遗漏 {totals['missing']}、重复 {totals['duplicate']}、失配 {totals['mismatch']}、未检查 {totals['unchecked']} |",
            f"| 当前 case 的算术/字节基线时间线 | 已生成 {priced_phase_count}/{phase_count} 个阶段；未计价调用 {unpriced_calls}；精细指令成本另列 |",
            "| 完整 Roofline 静态支持验收 | " + ("待独立复核" if passed and priced_phase_count == phase_count else "未完成；还须核对剩余算子工作量及数据流，确认是否漏算") + " |",
            "| GPU/provider、性能精度、Serving、MTP | 本阶段不验收 |", "",
            "| prompt tokens | 阶段 | 调用 | 未计价 | 已生成逻辑时间线 | 未计价算子 |",
            "|---:|---|---:|---:|---|---|"]
    rows += [f"| {comparison['prefill_tokens']} | {phase['phase']} | {phase['calls']} | {phase['unpriced_calls']} | {'是' if phase['logical_timeline_present'] else '否'} | {phase['unpriced_variants']} |"
             for comparison in comparisons for phase in comparison["phase_pricing"]]
    rows += ["",
            "下表统计已实现检查，不能换算为完整任务完成百分比。`unchecked` 中的条目目前仍包含尚未展开的要求类别，完整细项分母尚未建立。", "",
            "| 检查结果 | 数量 |", "|---|---:|"]
    rows += [f"| {name} | {value} |" for name, value in totals.items()]
    rows += ["", "预期由原始 config 和上游源码独立生成；未导入 Midlayer builder、lowering 或 predictor。每个投影均含源码位置、适用条件、shape、dtype、数学 FLOPs 和逻辑 payload；每项比较保留预期与实际。", "",
             f"数据流覆盖清单已展开为 {result['coverage']['expanded_alias_obligations']} 个逐节点 alias/临时物化要求、{result['coverage']['expanded_cache_state_obligations']} 个 cache/跨步状态要求及 {result['coverage']['expanded_fp8_callee_obligations']} 个 FP8 callee 临时物化要求；其中 {result['coverage']['source_alias_accounts']} 个提供单操作调用方逻辑 tensor-object 检查，callee workspace、cache 等未闭合项分别保留为未检查。数学、量化和分支的其余细项继续补齐，完整分母仍未闭合。", "",
             f"普通FP8的条件SDK callee有 {result['coverage']['matched_conditional_fp8_callee_accounts']} 项匹配：源码与实际图同时证明一个rank/model-call的行数一致时，以未知入口P导出每个in-call的H=max(P,align4(M))，并逐rank连接连续步骤的exit/entry port，不选择冷启动数值。A预留和scale偏移使用该H，scale stride仍为本次4行对齐M；empty和padding不增加初始化流量。匹配还必须满足独立caller对象/typed IO、量化后矩阵的顺序、全部workspace和矩阵source storage程序。缺SDK source、外部DeepGEMM及非一致行数的内部顺序继续未检查。该结果只接受声明的条件源码形态，不证明目标backend选择、真实初始P、compiled spill、allocator安全或正常容量，也不关闭全模型FP8量化/分支facet。", "",
             "C4 decode新增完整索引转换producer账本：整个INT32 top-k行读取并写出（含-1）、每token的INT32 request映射/bool有效标志、INT32长度写出及attention长度读取，均不能用有效候选数替代整行。短上下文的非负项数由源码确定并计block-table读取；scored路径保留未知项数/每项4B读取系数，未把候选上界当真实mask。C128/SWA-only不重复加此每层helper。原attention容量/partial投影及完整callee服务仍未验收。", "",
             "短上下文topk fill新增完整source对象检查：每rank的model.topk_indices_buffer在各层复用，完整当前前缀含sentinel均被覆盖；positions与声明的model行顺序相连，不读旧buffer，不逐层分配或复制。decode取起始行，prefill跳过decode行后取自身窗口；核对typed view、实际lowered fill源码账本、cache就绪依赖、reader在fill之后及下一层覆盖在上一层读完之后。该简单fill的源码临时物化可独立闭合，不能据此关闭attention内部workspace、真实allocator/容量或GPU服务。", "",
             "路由新增source返回对象及专家输入检查：FP32 weights/INT32 IDs均为独立fresh [T,6]，直接传给同rank专家；实际lowered专家账本须以FP32读取weights并保留INT32索引接口。CUDA hash包装的第三个empty INT32临时量没有传给所选fast kernel，无初始化/读写；普通Triton只分配两个返回tensor。INT32模型IDs不因hash包装产生全局cast，normal路径IDs参数不被kernel读取，bias.data不复制。当前声明正常路径无外加capture/replay或fused-shared列；真实checkpoint专家分配、compiled spill/provider和allocator仍不由该对象检查接受。", "",
             "共享专家合并按当前 modular 分支 MoERunner.forward 的 shared_output + fused_output 核对，结果为独立 tensor，不得套用 mega-MoE 分支的原地加法。逐实例比较输入 producer 的类型/shape/rank、直接边、独立结果、无额外源码 tensor temporary、TP collective consumer 和 lowering 的两读一写及依赖；producer 内部 alias、实际分配地址和库内部 scratch 继续由各自范围验收。", "",
             "贪心输出链逐实例核对最终 hidden 行索引、BF16→FP32 cast、FP32 argmax→INT64及INT64→INT32转换的source tensor-object关系。高级行索引即使选择全部行也复制；argmax的view(-1)、已为INT64的long()和INT32转换后的unsqueeze均不复制。分别核对输入/输出shape与dtype、rank-local producer/consumer和最终输出调用依赖。实际库scratch、分配地址及runtime不在这些source tensor-object检查中。", "",
             "Q/KV RMSNorm的输入是同一BF16投影的两段split views，保留完整投影行stride与KV列offset；split不复制。归一化分别分配两份dense BF16输出，FP32归一化和gamma乘法完成后一次写出；Q输出供主Q投影及适用长上下文indexer投影读取，KV输出供融合Qnorm/RoPE/cache-insert读取。逐实例核对消费者的typed output role和直接数据边，不能用经Q投影的可达性代替KV输入绑定。原始配置rms_norm_eps经TargetModelSemantics导出并校验，不猜默认值。", "",
             "普通block-FP8 linear包装新增条件source tensor程序：input.view不复制；外部量化的FP8值与FP32 scales由matmul消费；FlashInfer包装则直接转交BF16输入，As参数不传给实际operator。input_scale缺省时的new_empty(1)是BF16、2B逻辑对象且未初始化/未消费，不能当作内部FP32 scale，也不添加payload读写。相同BF16 dtype的返回to和最终view不复制。program枚举不选择实际wrapper/provider，不包含完整callee scratch、packed-scale runtime或正常容量，原全算子alias要求仍未通过。", "",
             f"独立清单包含 {result['coverage']['expected_logical_operators']} 个逻辑节点和 {result['coverage']['expected_dense_projections']} 个投影实例。覆盖 embedding、43层、head、greedy 到 INT32，比较节点、shape/dtype、部分属性及必要 producer 可达性。可达性不能证明 tensor alias 或精确读写。数据 alias、临时物化、缓存跨步副作用、非投影运算/字节、量化 scale 和更完整的长上下文/fusion矩阵仍需继续补齐。", "",
             "argmax 的 FLOPs/字节基线不包含 predicate/index/reduction 指令成本。GPU launch/provider/runtime、性能精度、Serving 容量和 MTP 不在本次验收内。", "",
             "连续步骤按独立声明的 request/new-request 身份、prefill→decode1→decode2→mixed 顺序、每行位置区间及同一请求前缀连续性比较。请求长度相同不能替代请求身份或位置一致；该检查证明声明场景的逻辑连续性，实际 cache 页/slot 和状态窗口仍另列。", "",
             "compressor 的 FP32 投影拆成 KV/score 两个 view，沿用同一行 stride 与不同 storage offset，没有 split copy；C4 为重叠双宽、C128 为单宽。save-partial 先检查负 slot，再读取 projection/position/APE 并写入同一持久状态对象；score 半部写入 score+APE[position%ratio]，finish 消费同一 layer/component/rank 的状态，源调用关闭 PDL 并保持先写后读。实际 state-page alias、读写窗口及跨步有效域仍需继续核对，这些源码子程序通过不替代完整缓存数据流验收。", "",
             "finish 的有效状态窗口已按请求/边界展开，并与 lowering 的边界、维度和有效行数相连：C4读取最近8行并屏蔽负位置，首边界4只读窗口行4..7、选择half1；C128读取128行、half0。half由窗口相对行决定，不能按绝对位置取模。状态页索引/页内偏移按绝对位置和源码block size推导，前一步已有行与本步写入行完整分割窗口。页号/stride仍为参数；CPU源码寻址检查不等于实际state/output共享页、allocator或runtime验收。", "",
             "CP1 slot constructor已核对普通与分块路径的等价寻址：table index为position//kernel_block_size，INT64 slot为page*kernel_block_size+页内offset。compressor直接转交同一slot tensor，并对common block table原地clamp(min=0)；只在被访问页号原本非负的域中保持一致，不能无条件宣称等价。cache groups来自同一zeroed int8 backing，dtype/view/squeeze均不复制，组间可覆盖相同bytes；活跃page归属、选定layout与覆盖顺序仍未闭合。元数据准备在model forward之外，其table/tail extent与traffic保持未知，未当成零或已验收。", "",
             "页的 source 生命周期新增核对：null block不可分配，只有非null/ref_cnt=1/无cached hash才可写；touch/free依引用计数操作队列，cache身份包含group ID。unpin的0引用内容可读到复用通知发生，通知不证明GPU fence。对旧committed前缀C，滑动窗口W和页大小B在extra retention非负时，释放prefix的上界为floor(max(0,C-W+1)/B)*B；每个声明窗口所需prior行都在保证保留区间内。actual eviction、live IDs、COW与物理alias仍未验收，不把保留界用于容量或点预测。", "",
             "2051-token case 独立检查包含长上下文 indexer Q 的 trailing RoPE、BF16 rounding、FP8 输出和 FP32 folded weights，以及每行已完成 C4 状态的 floor 候选数。V4 专用 RoPE cache 已由源码确定为 FP32；Q 准备步骤仍不伪造实际 launch 或指令速率。", "",
             "indexer Q 的 CuTe/Triton 分支分别核对：CuTe 使用 BF16x2 abs/max、整数 exponent bits 和 FP32 乘法，Triton 使用 FP32 abs/max、div_rn 和 log2/ceil/exp2。权重 fold 的乘法次序及 host 合并常量也分别保留；旧的 Triton 计数只在其源码条件下有效，不据此宣称实际 backend 或有限精度等价。", "",
             "indexer Q 新增基础数学 Roofline：FP32 加/减/乘/除各按1个算术 FLOP，log2/exp2 按现有通用 SFU 组；比较、转换、舍入、整数位构造与地址指令不属于该算术/字节计价，源码工作仍保留。对准确 Flash 的64 index heads，两条源码分支的理想计算项都小于相同逻辑 payload 的带宽项，因此可在不选择 backend 的情况下给出共同基线；任一分支不能证明 memory dominated 时直接拒绝。这个结论只属于声明的理想模型，不是实际 div.rn 吞吐、硬件时延边界或完整 native 算子计价。", "",
             "最终 RMSNorm 新增共同数学、唯一 BF16 输入/gamma/输出 payload 核对。IR reference 在 gamma 乘法前舍入到 BF16，vllm_c CUDA 在 FP32 中乘 gamma；两种源码程序分别记录，不宣称有限精度等价或选中实际 provider。数学求和使用最小 H-1 次加法，CUB 树/零初始化累加和通用 predictor 的行服务归一化仍与数学清单分开。", "",
             "routed experts 新增独立模块化 Marlin W4A16、EP-off 源码核对：W13/activation/W2/求和的有用数学、BF16 中间物化、FP32 router weights 读入后转 BF16、FP32 求和与 BF16 输出，以及 W13/W2 对同一 intermediate_cache13 的分阶段 view。CUDA clamped SiLU 保留 alpha 乘法、beta 加法和 SiLU 后的 BF16 舍入，已修正原少计标量组。MXFP4 scale 虽在 post-load 中短暂 upcast，Marlin 常驻输入仍转回 E8M0，按每组1B列出；权重读取只给每 active expert 的源码系数，不冒称已知 checkpoint 专家分配。GEMM padding、align/sort、FP32 scratch、完整字节和实际 backend 仍不算整算子验收通过。", "",
             "routed experts 的当前基础 Roofline 已按 W13→activation→W2/router-weight→top-k reduce 分阶段取计算/带宽最大项再求和；不再用一次 aggregate max 将串行 producer/consumer 重叠。useful tensor FLOPs 使用真实 routed rows，不使用 launch padded rows；非权重字节来自 typed role uses，权重按显式 histogram 的 active experts 乘源码系数。移除旧的未绑定 routing-ID 读取假设，排序/metadata、padding、FP32 scratch 与解量化工作仍是待核实项。该价格依赖声明的 routing assumption，不证明真实 checkpoint 分配或完整 MoE 服务；其他 INT4/EP 路径没有改变。", "",
             "MoE alignment 新增 source allocation extent 和 count/fill→sort 前序的逻辑字节。sorted IDs 的全容量 sentinel 初始化与有效 route 覆盖不是同一次写，expert IDs 包括 inactive 尾部写出；general 程序还包含全局 INT32 cumsum 与逐 route atomic read/modify/write。Flash 的256 experts 不满足 small-batch-expert 分支，即使单 token 也不能省掉 general sort。两步先于 W13，整数/atomic 服务及 GEMM metadata consumer 的 CTA/split 重读仍另列，未声明 allocator 或实际 DRAM 绑定。", "",
             "W13/W2 的 metadata 唯一输入已补入：每次分别读取有效 M-block 的完整 INT32 sorted-ID prefix、expert-ID prefix 及一个 post-pad scalar，不读取 allocation tail。独立参考核对每 block 字节系数、声明 histogram 的 prefix 数和实际基础字节投影；histogram 仅为声明输入，不证明 checkpoint 路由。W13 以 route_id//topk 取 token 行，W2 以 route_id 取展开行，两者都 scatter 到 route_id，并排除 sentinel。CTA/split 重读、lock 和整数服务仍未闭合，不能据此宣称完整 MoE 对齐。", "",
             "MoE FP32 scratch 新增独立源码核对：W13/W2 各自按 min(N*sorted_capacity,SM*4*block_m*256) 请求 FP32 空间，m8 再翻倍；空分配不计初始化写，声明 H200132 SM 不当作 runtime 观测。非 first 读/加、非 last 写及 fragment/lane-tail 条件分别记录。参数化的每 partial 值数不选择实际 thread_n 或 slice 数；全调用 scratch traffic/附加加法仍未知，不能写0、按申请大小计流量或把两个申请相加当容量。已有基础 Roofline 数值不因新增空间账本改变，完整 MoE 数学/字节/alias 仍需后续核对。", "",
             "默认无 LoRA workspace 另有源码明确的 INT32 零初始化：alignment 后、W13 前写 SM*4 个槽位，H200132 SM 为每 MoE 调用2112B，W13/W2 共享一次初始化。现纳入独立串行 workspace_zero 阶段并核对实际 read/write 投影。该写入不是 FP32 empty scratch 初始化，也未代替后续 lock acquire/release、同步服务或正常 allocator 验收。", "",
             "本轮修正：head 按未传 quant_config 的上游构造器保留 BF16；补齐完整 indexer 的 Q 准备及缩放权重 producer；评分改为读取 FP8 Q，并以 floor 计已完成状态。各候选是否具备这些修复，以其原 snapshot 的逐项结果为准。", "",
             "新增修正：indexer cos/sin cache 按 V4 专用类确定为 FP32；共享 clamped SiLU 保留 CUDA 源码的 alpha/beta 操作及 SiLU 结果转回 BF16 后再乘 up 的阶段。新增逐项核对覆盖 Q/KV RMSNorm、FP32 save-partial/APE、short-topk、共享激活/add、row-select、logits cast、argmax/INT32，以及 C128 两阶段顺序、FP32 scratch 写出和 exp 次数。", "",
             "mHC 独立阶段核对包含首层 broadcast、短 token 融合及大 token 非融合。非融合按 post→prenorm→norm 分别推导 DeepGEMM split 与 TileLang 单partial两种条件程序，核对 BF16 residual 物化/flatten view、FP32 projection/sqrsum partial及消费阶段。独立声明首次cached dispatch前固定VLLM_USE_DEEP_GEMM=False时，后续非融合source prenorm确定为TileLang单partial；True或未声明时，backend条件仍逐实例未检查，H200不能证明has_deep_gemm()。首层broadcast始终强制DeepGEMM，源码分支选择不证明actual provider；条件形式的点价格也不是两种程序共同价格。", "",
             "routing 按显式声明的纯文本 modular CUDA 场景分别核对 hash CUDA 与普通 Triton top6：hash只读六个选中logit，普通分支计算整个power-of-two块；stable softplus与threshold softplus分别保留，NaN、并列ID、归一化分母及原始无bias权重的语义不混同。源码标量组直接参与predictor计数；逐行bias读与单个bias对象extent分开，不把对象extent说成已观测DRAM或缓存命中。checkpoint hash表内容、实际选择结果、compiler/native指令成本及真实provider仍未绑定。", "",
             "完整indexer已补有用数学和typed阶段账本：QK、ReLU、head权重乘法、H-1求和及KV scale乘法；INT32 top-k缓冲区先全capacity写-1、再写选择结果，两次写保留。显式CP1的prefill每请求完整压缩上下文只gather一次，query子块复用；packed缓存读、FP8 values和4B UINT8 scale backing写均计入，scale的FP32 view不产生转换拷贝。清零→gather→FP32 score物化→选择按串行阶段做基础Roofline。未绑定CP信息不默认CP1；撤去无源码依据的log2(topk)比较轮数，ReLU比较和整数top-k指令成本另列。官方数学参考不证明浮点顺序或选中provider。workspace分组/峰值、矩形score backing、metadata及selector scratch/值依赖工作仍逐实例未检查，完整indexer与native计价没有通过。", "",
             "评分输入的INT32 metadata已与selector读取分开：CP1非空prefill context每行两个bound，普通未padding的单token decode每行一个effective context length。未知CP、空context分组或多token decode保持未绑定。score wrapper传clean_logits=False，并将同一FP32 logits交给选择器；矩形逻辑shape不等于inactive区域写入或物理backing。单请求prefill全部query子块的逻辑输出面积固定为Q*floor(L/4)，多请求只保留分组面积界，均不替代流量或容量。decode列数仍依赖构造期runtime indexer max_model_len，不能用active context冒充。", "",
             "mixed源码调用顺序已核对：indexer_op内部先prefill后decode，整个调用返回后才进入FlashMLA；FlashMLA再先prefill后decode。图中保留这两个阶段顺序及全部indexer→attention的join，不再允许prefill attention越过尚未完成的decode indexer。此处是源码调用依赖，不冒称实际compiler/stream/provider绑定。", "",
             "prefill top-k另按sampler.cu的rowLen<=topK捷径核对：直接输出自然顺序索引及-1，不读score，NaN和并列值不影响此分支。只对非捷径行保留有用score读取，并加入每行两个INT32 bound的逻辑读取；额外refinement/scratch仍待核对。全捷径行的QK/FP32 score producer仍保留，不做源码没有的消除。", "",
             "逆 RoPE/FP8 量化按源码核对 live rotation/quantization 数学组、BF16 输入、INT64 positions、FP32 cos/sin 和 FP8 输出；FP32 scale backing 按 token 数向4对齐，包含源码对 padding 行的零写。唯一 payload 不包含 partner/per-head 重读或实际 DRAM 事务。", "",
             "QNorm/RoPE/cache insert 新增独立 source-group 数学及唯一 payload 核对。C4/C128 finish 的 FP32 状态窗口、metadata、gamma/trig 和缓存写出按阶段比较；main CuTe UE8M0 scale 是整数位构造，indexer Triton 则使用 log2/ceil/exp2。abs+两次clamp下界曾检出单个 C128 store 的471次不足；随后展开完整 active-value 账本，区分 main 的raw-scale除法/下界比较和 indexer 的raw-scale乘法，当前 main store 为1360次ALU，仍满足1344次独立下界。", "",
             "finish 的压缩、规范化、RoPE、量化与转换值数均按源码分阶段核对，并验证 predictor 使用同一数学账本。main 的 scale 下界是 max(1e-4,absmax/448)，indexer 则是 max(absmax,1e-4)/448；两者不假设数值等价。最小 norm/quant 树、active indexer 槽位和显式 C128 merge 属于当前基线；CuTe 的源级重复表达式、Triton masked/identity vector 域及类型转换另列，不冒充 compiler issue 或完整 native 服务。", "",
             "C128 源码临时对象为 FP32 [T,512] 的 torch.empty；只有边界 token 对应的原始行写入并被store读取，不是紧凑的 [完成状态数,512]。逐项检查该源码形状、读写行、dtype及两阶段依赖；source_alias表示源码传递同一张量，实际 allocator/KernelPlan buffer 绑定仍未验证，不据此通过全部 alias 或正常容量验收。", "",
             "attention 的 prefill/decode wrapper 均传入 padded-head FP32 attn_sink 参数。本场景64个 padded heads 对应每层每TP rank的256B唯一参数输入，现补入读账本；不把参数对象按 query 数复制，也不宣称实际 DRAM reload 次数。只核对初始化/输入角色，checkpoint 的 live sink 值和 FlashMLA callee 的 sink 数值程序仍未绑定，不能把这256B输入检查当成完整 attention 数学或 native/provider 验收。", "",
             "稀疏 prefill 新增独立因果容量和 gather 工作核对：scored top-k 也受每个 query 的已完成压缩前缀限制，不能把最终512候选乘到每行；BF16 workspace 对 K/V 是同一份物化，共同有用的448值/行反量化乘法已计入基础数学组。CuTe 的 BF16 multiply/8-scale-byte 路径与 Triton 的 FP32 multiply/7-scale-byte/exp2 路径分别保留，未选择实际 backend，也未断言两者有限精度等价。branch-specific exp2、可能被覆盖的尾部工作、metadata/helper/native 服务仍不作为已验收的完整算子计价。", "",
             "SWA/cache 记录的存储宽度现按源码保持584B，不能把 QK/PV 两种数学用途写成两份存储。decode 的逐 query packed payload 投影相应修正；实际 kernel 重读次数、缓存命中和完整 FlashMLA split/combine 访存仍需另外证明。C4 的独立候选数及 KernelSpec/evaluation normalization 均按 floor(kv/4) 限定，部分压缩窗口不能多出一个有效状态。", "",
             "prefill 索引物化新增源码核对和基础字节计价：完整对齐 INT32 缓冲区 fill(-1)、combine 的有效索引/长度写出、attention 的索引/长度读取分别保留。fill 与覆盖写共享同一缓冲区，两次写都必须计入；256 workers/request 在循环前读取的5个 INT32 metadata 按源码 role uses 记录，不当成唯一 footprint、已观测 DRAM 或 compiler 指令。该新增检查还核对整个已声明 baseline 字节投影的数值；并不补齐 gather metadata、FlashMLA 内部临时量或整数控制服务的全部验收。", "",
             "未量化投影与 wo_a 已预量化输入的逻辑 lowering 核对唯一矩阵调用、rank、输入类型和几何。普通 block-FP8 linear 独立列出两种源码包装：VECTOR量化后GEMM，或TRTLLM调用内量化后矩阵乘法；二者归一化为一条量化→matmul逻辑图，并检查完整调用清单、类型、shape、rank和边。此检查只证明声明模板符合源码条件形式，不据候选路径反推已选运行时class/provider；VLLM_BATCH_INVARIANT、contiguity、scale rounding、TMA对齐和compiler/launch条件仍分别保留。", "",
             "独立量化按vLLM0.30 CUDA/Triton源码记录逐值FP32除法、abs、max/clamp、类型转换和共享/SSA复用，Hopper的普通/UE8M0取整scale都保持FP32存储。两条backend的raw-scale运算和取整下界不同，不能假定有限精度等价。FP32加减乘除各按1个理想FLOP，log2/exp2按通用SFU；比较/abs/ceil和lane重复只记录源码数量，不在缺ISA映射时塞入integer ALU速率。仅所有源码条件分支均证明相同payload的带宽项主导时返回共同基线，否则拒绝点预测。该source KernelSpec无launch且native unsupported。", "",
             "TRTLLM模板同调用的scale_1x128保留amax零/非零分支；源码先量化再矩阵乘法，理想Roofline按两阶段串行求和，不把它们重叠成一次max。独立量化的GEMM不再计内部producer；FP32 scale live写入、对齐stride/地址跨度及consumer读取分别检查，padding不算写入或实际allocator容量。SDK源码绑定不等于vLLM0.30实际provider/版本兼容或native服务验收。", "",
             "embedding 的预期 token IDs 来自显式 --case-input 外部合成输入，candidate 回显只作为被检查事实。独立计算每个 TP rank 的词表区间与归属行数，检查 INT32 输入、仅归属权重行的读取、所有行的 BF16 输出写入（包含非归属零值），以及单 kernel 内部标量 widening；不虚构全局 INT64 buffer。缺少独立输入仍为未检查，不能从 candidate 自动补齐。", "",
             "FlashInfer scale_1x128 局部源码程序继续核对 BF16 abs/local max、FP32 warp max 后的 BF16 舍入、共同乘法及 FP32 inverse scale。amax 是否为零的除法分支保留未知条件，不假定真实输入非零。scale 的逻辑 [K/128,M] 视图使用 [ceil(M/4)*4,1] stride；live 写入、地址跨度和 padded extent 分开，padding 不计为 quantizer 写入，也不冒称实际 workspace 容量。source expression 次数不等于 compiler issue 次数，新增描述不改变 HWSpec 或基线定价。", "",
             "TP collective 检查导出的 KernelSpec.transfer，而非只看 workload 属性：operation、SUM、participants、per-rank input bytes 与 BF16 producer 类型逐项比较。源码数学记录全组 SUM 的最少加法和 gather 无算术，逻辑输出按操作契约确定。该检查不证明 reduction order、有限精度等价、实际通信算法、内部 scratch/local buffer 访问或 network traffic；既有通信预测保持原状。", "",
             "字节比较逐项声明口径：有些是唯一 tensor payload，有些是每 active token 的参数行使用量。L2 与 DRAM 读写分别保留；例如 shared-add 的唯一 payload 不被当成实际 DRAM 读取量。逻辑 lowering stage 检查不等同实际 GPU launch/provider 绑定。", "",
             "## 具体差异", ""]
    differences = Counter((check["field"], str(check["expected"]), str(check["actual"]))
                          for comparison in comparisons for check in comparison["checks"]
                          if check["status"] == "mismatch")
    rows += [f"- {count} 项 `{field}`：预期 `{expected}`，实际 `{actual}`。"
             for (field, expected, actual), count in differences.items()]
    if not differences:
        rows += ["已实现检查范围内没有不匹配；未检查项继续导致验收失败。"]
    rows += ["", "## 逐算子剩余检查", "",
             "下表为逐实例的未检查数，0仅指该列已实现的口径。tensor IO/alias 已逐逻辑节点展开，cache/page/slot 与跨步状态已逐相关算子展开；已核对的源码子程序不替代完整数据流证明。其余运算/量化/分支细项仍待展开，完整分母未闭合。", "",
             "| 算子族 | 实例 | 算术未检查 | 字节未检查 | lowering未检查 | 数据流未检查 |",
             "|---|---:|---:|---:|---:|---:|"]
    rows += [f"| {item['family']} | {item['instances']} | {item['arithmetic_unchecked_instances']} | {item['bytes_unchecked_instances']} | {item['lowering_unchecked_instances']} | {item['dataflow_unchecked_instances']} |"
             for item in result["family_coverage"]]
    rows += ["", "## 候选与复现", ""]
    rows += [f"完整参考：[打开]({reference_path.name})；逐项比较：[打开]({comparison_path.name})。压缩只改变存储，不删除检查项或修改验收口径。", ""]
    rows += [f"比较器 snapshot：`{generator_snapshot}`。", ""]
    rows += ["完整重建（并行生成独立 case，再运行比较）：`python " + str(output / "reproduce.py") + "`。脚本携带本轮明确声明的合成 token IDs，不下载配置或权重，不使用 GPU；新结果写入 `reproduced/`，保留原候选 snapshot。最终比较启用 `--require-aligned`：仍有未检查项时预期非零退出，产物生成成功不等于验收通过。", ""]
    rows += [f"- `{item['candidate_path']}`：snapshot `{item['candidate_snapshot']}`。" for item in comparisons]
    if flashinfer_source_root is not None:
        sdk = references[0]["sources"]["flashinfer-input-quant"]
        rows += [f"- FlashInfer SDK source：`{sdk['package_root']}`，version `{sdk['version']}`，只读取源码及 build metadata；实际 provider 未绑定。"]
    rows += ["", "```bash", "python scripts/evaluation/serving/report_v4_flash_alignment.py \\",
             f"  --model {config_path} \\", f"  --source-root {source_root} \\"]
    rows += [f"  --candidate {tokens} {path} \\" for tokens, path in candidates]
    rows += [f"  --case-input {tokens} {path} \\" for tokens, path in case_inputs]
    if torch_include is not None:
        rows += [f"  --torch-include {torch_include} \\"]
    if flashinfer_source_root is not None:
        rows += [f"  --flashinfer-source-root {flashinfer_source_root} \\"]
    if v4_source_max_model_len is not None:
        rows += [f"  --v4-source-max-model-len {v4_source_max_model_len} \\"]
    if v4_source_use_deep_gemm is not None:
        flag = "--v4-source-use-deep-gemm" if v4_source_use_deep_gemm else "--no-v4-source-use-deep-gemm"
        rows += ["  " + flag + " " + chr(92)]
    if compress_artifacts:
        rows += ["  --compress-artifacts " + chr(92)]
    rows += [f"  --output {output}", "```", ""]
    (output / "report.md").write_text("\n".join(rows), encoding="utf-8")
    return result


def _write_reproduction(config_path, source_root, inputs, output, torch_include, flashinfer_source_root,
                        source_max_model_len=None, source_use_deep_gemm=None, compress_artifacts=False):
    """Generated thin CLI orchestration and explicit inputs, not a new oracle."""
    root = Path(__file__).resolve().parents[3]
    settings = {"root": str(root), "config": str(Path(config_path).resolve()),
                "source": str(Path(source_root).resolve()), "inputs": inputs, "source_max_model_len": source_max_model_len,
                "source_use_deep_gemm": source_use_deep_gemm,
                "compress_artifacts": compress_artifacts,
                "torch_include": None if torch_include is None else str(Path(torch_include).resolve()),
                "flashinfer_source_root": None if flashinfer_source_root is None else str(Path(flashinfer_source_root).resolve())}
    body = '''"""Regenerate declared synthetic Flash cases and independent comparisons."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

settings = json.loads(SETTINGS_LITERAL)
root = Path(settings["root"])
output = Path(__file__).resolve().parent / "reproduced"
output.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ)
environment["CUDA_VISIBLE_DEVICES"] = ""

def generate(case):
    tokens = case["prefill_tokens"]
    if case["token_ids"] is None:
        raise ValueError("producer reproduction requires independently declared --case-input, never candidate echoes")
    ids = output / f"input_t{tokens}.json"
    ids.write_text(json.dumps(case["token_ids"]) + "\\n")
    probe = output / f"probe_t{tokens}.json"
    command = [sys.executable, "scripts/evaluation/serving/report_model_operator_probe.py",
        "--model", settings["config"], "--tensor-parallel-size", "4", "--mtp-mode", "off",
        "--v4-eager-metadata", "--v4-routing-source", "vllm_0_30_modular_cuda_top6_pure_text",
        "--target-sampling-source", "vllm_0_30_plain_greedy_bf16", "--prefill-tokens", str(tokens),
        "--embedding-token-ids", str(ids), "--require-logical-roofline-forward", "--output", str(probe)]
    if settings["source_max_model_len"] is not None:
        command.extend(["--v4-source-max-model-len", str(settings["source_max_model_len"])])
    if settings["source_use_deep_gemm"] is not None:
        command.append("--v4-source-use-deep-gemm" if settings["source_use_deep_gemm"] else "--no-v4-source-use-deep-gemm")
    with (output / f"probe_t{tokens}.log").open("w") as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    return tokens, probe, ids

with ThreadPoolExecutor(max_workers=len(settings["inputs"])) as pool:
    results = list(pool.map(generate, settings["inputs"]))
command = [sys.executable, "scripts/evaluation/serving/report_v4_flash_alignment.py",
    "--model", settings["config"], "--source-root", settings["source"], "--output", str(output), "--require-aligned"]
for tokens, probe, ids in results:
    command.extend(["--candidate", str(tokens), str(probe)])
    command.extend(["--case-input", str(tokens), str(ids)])
if settings["torch_include"] is not None:
    command.extend(["--torch-include", settings["torch_include"]])
if settings["flashinfer_source_root"] is not None:
    command.extend(["--flashinfer-source-root", settings["flashinfer_source_root"]])
if settings["source_max_model_len"] is not None:
    command.extend(["--v4-source-max-model-len", str(settings["source_max_model_len"])])
if settings["source_use_deep_gemm"] is not None:
    command.append("--v4-source-use-deep-gemm" if settings["source_use_deep_gemm"] else "--no-v4-source-use-deep-gemm")
if settings["compress_artifacts"]:
    command.append("--compress-artifacts")
subprocess.run(command, cwd=root, env=environment, check=True)
'''
    source = body.replace("SETTINGS_LITERAL", repr(json.dumps(settings)))
    ast.parse(source)
    (output / "reproduce.py").write_text(source, encoding="utf-8")
