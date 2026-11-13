"""Self-contained viewer for the existing Serving execution artifact.

This module displays predictions; it never prices work or infers missing events.
"""
from __future__ import annotations

from evaluation.reporting.serving_viewer_ui import with_serving_viewer_ui, report_experiment_configuration, _configuration_metadata

import base64
import gzip
import json
import math
import re
from pathlib import Path
import time


def _operation_display(call: dict) -> tuple[str, int | None]:
    """Keep source-derived display labels, without retaining internal node IDs."""
    if "display_category" in call:
        return call["display_category"], call["layer"]
    spec = call["kernel_spec"]
    kind = spec["kind"].upper()
    source = " ".join(call["source_node_ids"]).lower()
    variant = str(spec["variant"]).lower()
    match = re.search(r"layer-(\d+)", source)
    layer = int(match[1]) if match else None
    if kind == "COMMUNICATION":
        operation = spec["transfer"]["operation"] if spec["transfer"] is not None else None
        names = {"all_reduce": "AllReduce", "all_gather": "AllGather",
                 "all_to_all": "AllToAll", "send_recv": "点对点交接"}
        category = names[operation] if operation in names else "其他通信"
    elif kind in ("MEMCPY", "KV_CACHE", "KV_OFFLOAD"):
        category = "KV 更新 / 搬运"
    elif kind == "SPARSE_ATTN_INDEXER":
        category = "稀疏索引"
    elif "ATTN" in kind:
        category = "Attention"
    elif "MOE" in kind:
        category = "MoE 计算"
    elif kind == "VECTOR" and ("router" in source or "routing" in source or "routing" in variant):
        category = "专家路由"
    elif kind == "GEMM" and any(value in source for value in ("qkv", "q-proj", "kv-proj")):
        category = "QKV 投影"
    elif kind == "GEMM" and any(value in source for value in ("o-proj", "out-proj")):
        category = "输出投影"
    elif kind == "GEMM" and any(value in source for value in ("gate-up", ":down", "ffn")):
        category = "FFN 投影"
    elif kind == "VECTOR" and any(value in variant for value in ("norm", "rotary", "silu", "activation")):
        category = "归一化 / 激活"
    else:
        category = "采样 / 其他"
    return category, layer


def _safe_plan(plan: dict) -> dict:
    calls = []
    for call in plan["calls"]:
        category, layer = _operation_display(call)
        calls.append({**{name: call[name] for name in ("call_id", "pool_name", "rank", "resource")},
            "display_category": category, "layer": layer,
            "kernel_spec": {name: call["kernel_spec"][name] for name in (
                "kind", "hardware_id", "shape", "dtype", "variant", "transfer")}})
    return {"calls": calls}


def _safe_predictions(predictions: list | tuple) -> list:
    return [{**{name: prediction[name] for name in (
        "call_id", "t_total_s", "t_launch_s", "bottleneck_group")},
        "t_per_resource_s": {name: value for name, value in prediction["t_per_resource_s"].items()
            if value != 0}} for prediction in predictions]


def _safe_timeline(timeline: dict) -> dict:
    # Absolute endpoints are retained exactly, avoiding float round-trip drift.
    def interval(row):
        return {name: row[name] for name in ("absolute_start_time_s", "absolute_end_time_s")}
    result = {"intervals": [{"call_id": row["call_id"], **interval(row)}
        for row in timeline["intervals"]],
        "host_events": [{**{name: row[name] for name in (
            "event_kind", "pool_name", "rank", "call_id", "call_occurrence_index",
            "runtime_graph_mode", "graph_segment_id", "exposed_time_s", "overlapped_time_s")},
            **interval(row)} for row in timeline["host_events"]]}
    if "block_repeats" in timeline:
        result["block_repeats"] = [{"block_id": block["block_id"], "repeat_count": block["repeat_count"],
            "call_intervals": [{"call_id": call["call_id"],
                "intervals": [interval(row) for row in call["intervals"]]}
                for call in block["call_intervals"]]} for block in timeline["block_repeats"]]
    return result


_ID_MAPS = frozenset(("visualization_resources", "resources", "by_call_id", "missing_by_call_id"))


def _pack_projection(payload: dict) -> dict:
    """Lossless report-only tables; this does not change execution contracts."""
    tables = {"version": 1, "call_ids": [], "kernel_specs": [], "placements": []}
    lookups = {name: {} for name in ("call_ids", "kernel_specs", "placements")}

    def intern(name, value):
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key not in lookups[name]:
            lookups[name][key] = len(tables[name])
            tables[name].append(value)
        return lookups[name][key]

    def pack(value, context=""):
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        if not isinstance(value, dict):
            return value
        if context in _ID_MAPS:
            return {str(intern("call_ids", key)): pack(item) for key, item in value.items()}
        result = {}
        placement = {name: value[name] for name in ("pool_name", "rank", "resource") if name in value}
        has_placement = "pool_name" in placement and "rank" in placement
        if has_placement:
            result["placement_ref"] = intern("placements", placement)
        for key, item in value.items():
            if has_placement and key in placement:
                continue
            if key == "kernel_spec":
                result["kernel_spec_ref"] = intern("kernel_specs", item)
            elif key == "call_id" and item is not None:
                result[key] = intern("call_ids", item)
            else:
                result[key] = pack(item, key)
        return result

    packed = pack(payload)
    packed["visualization_storage"] = tables
    return packed


def _safe_report_metadata(value):
    """Apply the owner's privacy rules, borrowing unchanged JSON subtrees.

    Finalization only reads the result. Copy a branch when filtering or type
    normalization changes it instead of duplicating every recorded dictionary.
    """
    allowed_fields = {}
    scalar_types = (str, int, float, bool)
    def normalize(item):
        kind = type(item)
        if item is None or kind in scalar_types:
            return item
        if kind is tuple:
            return [child if child is None or type(child) in scalar_types else normalize(child)
                for child in item]
        if kind is list:
            result = None
            for index, child in enumerate(item):
                normalized = child if child is None or type(child) in scalar_types else normalize(child)
                if result is None and normalized is not child:
                    result = item[:index]
                if result is not None:
                    result.append(normalized)
            return item if result is None else result
        if kind is dict:
            keys = tuple(item)
            if keys not in allowed_fields:
                if len(allowed_fields) >= 256:
                    allowed_fields.clear()
                # Validate names and privacy through the one existing owner;
                # values are normalized below without repeated dataclass checks.
                permitted = frozenset(_configuration_metadata(dict.fromkeys(keys)))
                allowed_fields[keys] = None if len(permitted) == len(keys) else permitted
            permitted = allowed_fields[keys]
            if permitted is not None:
                return {key: child if child is None or type(child) in scalar_types else normalize(child)
                    for key, child in item.items() if key in permitted}
            result = None
            for key, child in item.items():
                normalized = child if child is None or type(child) in scalar_types else normalize(child)
                if normalized is not child:
                    if result is None:
                        result = item.copy()
                    result[key] = normalized
            return item if result is None else result
        return _configuration_metadata(item)
    return normalize(value)


_BULK_REPORT_FIELDS = frozenset({"visualization_summary", "visualization_schedule",
    "visualization_iterations", "scheduler_progress", "host_timing", "source_activation_memory"})


def _report_json_bytes(fields, serialized_values):
    """Encode one object, reusing pinned values within read-only finalization.

    Only the caller's few shared bulk values are retained. Sanitized copies
    have different identities and are encoded afresh; unused entries are
    released before assembling the next object. No cache escapes the render.
    """
    identities = {id(value) for value in fields.values()}
    for identity in tuple(serialized_values):
        if identity not in identities:
            del serialized_values[identity]
    parts = [b"{"]
    for index, (name, value) in enumerate(fields.items()):
        if not isinstance(name, str):
            raise TypeError("report object keys must be strings")
        if index:
            parts.append(b",")
        parts.extend((json.dumps(name, allow_nan=False).encode(), b":"))
        identity = id(value)
        if identity in serialized_values:
            retained, encoded = serialized_values[identity]
            if retained is not value:
                raise RuntimeError("report serialization crossed retained value identity")
            if encoded is None:
                encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
                serialized_values[identity] = (value, encoded)
            else:
                del serialized_values[identity]
        else:
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        parts.append(encoded)
    parts.append(b"}")
    return b"".join(parts)


def _pack_summary_projection(payload: dict, *, serialized_values=None) -> dict:
    """Losslessly compress large report fields; semantic fields remain unchanged on decode."""
    import hashlib
    fields = {key: value for key, value in payload.items() if key in _BULK_REPORT_FIELDS}
    encoded = (json.dumps(fields, separators=(",", ":"), allow_nan=False).encode()
        if serialized_values is None else _report_json_bytes(fields, serialized_values))
    if len(encoded) < 1024 * 1024:
        return payload
    compressed = gzip.compress(encoded, compresslevel=1, mtime=0)
    return {**{key: value for key, value in payload.items() if key not in fields},
        "visualization_storage": {"version": 2, "encoding": "gzip-json-base64",
            "fields": list(fields), "sha256": hashlib.sha256(compressed).hexdigest(),
            "data": base64.b64encode(compressed).decode("ascii")}}


def _unpack_projection(payload: dict) -> dict:
    """Accept legacy projections; reject invalid table references in new ones."""
    if "visualization_storage" not in payload:
        return payload
    tables = payload["visualization_storage"]
    if type(tables["version"]) is int and tables["version"] == 2:
        import hashlib
        if set(tables) != {"version", "encoding", "fields", "sha256", "data"} or tables["encoding"] != "gzip-json-base64":
            raise ValueError("invalid compressed visualization storage")
        names = tables["fields"]
        if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                or len(names) != len(set(names)) or not names
                or not set(names) <= _BULK_REPORT_FIELDS or set(names) & set(payload)):
            raise ValueError("invalid compressed visualization fields")
        compressed = base64.b64decode(tables["data"], validate=True)
        if hashlib.sha256(compressed).hexdigest() != tables["sha256"]:
            raise ValueError("compressed visualization checksum differs")
        decoded = json.loads(gzip.decompress(compressed))
        if not isinstance(decoded, dict) or set(decoded) != set(names):
            raise ValueError("compressed visualization fields differ from index")
        return {**{key: value for key, value in payload.items() if key != "visualization_storage"}, **decoded}
    if type(tables["version"]) is not int or tables["version"] != 1:
        raise ValueError("unsupported visualization storage version")
    for name in ("call_ids", "kernel_specs", "placements"):
        if not isinstance(tables[name], list):
            raise TypeError("visualization storage tables must be lists")
    if any(not isinstance(value, str) or not value for value in tables["call_ids"]):
        raise ValueError("visualization call ID table is invalid")
    if len(set(tables["call_ids"])) != len(tables["call_ids"]):
        raise ValueError("visualization call ID table has duplicates")
    for placement in tables["placements"]:
        if (not isinstance(placement, dict) or "pool_name" not in placement or "rank" not in placement
                or set(placement) - {"pool_name", "rank", "resource"}):
            raise ValueError("visualization placement table is invalid")

    def resolve(name, index):
        if type(index) is not int or not 0 <= index < len(tables[name]):
            raise ValueError("invalid visualization table reference")
        return tables[name][index]

    def unpack(value, context=""):
        if isinstance(value, list):
            return [unpack(item) for item in value]
        if not isinstance(value, dict):
            return value
        if context in _ID_MAPS:
            if any(not key.isdecimal() or str(int(key)) != key for key in value):
                raise ValueError("invalid visualization call ID reference")
            return {resolve("call_ids", int(key)): unpack(item) for key, item in value.items()}
        result = {}
        for key, item in value.items():
            if key == "visualization_storage":
                continue
            if key == "placement_ref":
                if set(value).intersection(("pool_name", "rank", "resource")):
                    raise ValueError("visualization placement duplicates inline fields")
                result.update(resolve("placements", item))
            elif key == "kernel_spec_ref":
                if "kernel_spec" in value:
                    raise ValueError("visualization kernel spec duplicates inline field")
                result["kernel_spec"] = resolve("kernel_specs", item)
            elif key == "call_id" and item is not None:
                result[key] = resolve("call_ids", item)
            else:
                result[key] = unpack(item, key)
        return result

    return unpack(payload)


def _validate_diagnostics(diagnostics: object, summaries: object) -> None:
    """Check references in the optional bounded predictor-diagnostics projection."""
    if diagnostics is None:
        if summaries is not None:
            for row in summaries:
                if "model_bottleneck_groups" in row:
                    for group in row["model_bottleneck_groups"].values():
                        if "critical_paths" in group and group["critical_paths"]:
                            raise ValueError("visualization diagnostic path catalog is missing")
        return
    if not isinstance(diagnostics, dict):
        raise TypeError("visualization diagnostics must be a dict or null")
    paths = diagnostics["critical_paths"]
    by_call_id = diagnostics["by_call_id"]
    missing_by_call_id = diagnostics["missing_by_call_id"]
    if (not isinstance(paths, list) or not isinstance(by_call_id, dict)
            or not isinstance(missing_by_call_id, dict)):
        raise TypeError("visualization diagnostic catalog has invalid collections")
    if set(by_call_id).intersection(missing_by_call_id):
        raise ValueError("visualization diagnostic call cannot be both present and missing")
    for call_id, reason in missing_by_call_id.items():
        if not isinstance(call_id, str) or not call_id or not isinstance(reason, str) or not reason:
            raise ValueError("visualization missing diagnostic requires a call ID and reason")
    for name in ("catalog_limit", "per_group_step_limit"):
        if type(diagnostics[name]) is not int or diagnostics[name] < 1:
            raise ValueError(f"visualization {name} must be a positive integer")
    if len(paths) > diagnostics["catalog_limit"]:
        raise ValueError("visualization diagnostic catalog exceeds its limit")
    for entry in paths:
        if (not isinstance(entry, dict) or not isinstance(entry["critical_path"], list)
                or not entry["critical_path"]
                or any(not isinstance(item, str) or not item for item in entry["critical_path"])
                or entry["support_level"] not in ("complete", "partial", "aggregate_only", "compatibility")
                or not isinstance(entry["limitations"], list)
                or any(not isinstance(item, str) or not item for item in entry["limitations"])):
            raise ValueError("visualization diagnostic path is invalid")
    def validate_path_id(value):
        if not isinstance(value, str) or not value.isdecimal() or int(value) >= len(paths):
            raise ValueError("visualization diagnostic path reference is invalid")
    for call_id, value in by_call_id.items():
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("visualization diagnostic call ID must be non-empty")
        validate_path_id(value)
    if summaries is not None:
        for row in summaries:
            if "model_bottleneck_groups" in row:
                for group in row["model_bottleneck_groups"].values():
                    if "critical_paths" in group:
                        if (not isinstance(group["critical_paths"], dict)
                                or len(group["critical_paths"]) > diagnostics["per_group_step_limit"]):
                            raise ValueError("visualization per-step diagnostic paths exceed their limit")
                        count = group["critical_path_unavailable_or_omitted"]
                        if type(count) is not int or count < 0:
                            raise ValueError("invalid missing diagnostic count")
                        for value, item in group["critical_paths"].items():
                            validate_path_id(value)
                            if type(item["operation_count"]) is not int or item["operation_count"] < 0:
                                raise ValueError("invalid diagnostic path count")
                            count += item["operation_count"]
                        if count != group["operation_count"]:
                            raise ValueError("diagnostic path counts do not match group")


def write_interactive_report(payload: dict, output: str | Path) -> dict:
    """Embed an existing execution artifact without external dependencies."""
    started = time.perf_counter()
    payload = _unpack_projection(payload)
    artifact = payload["execution_artifact"]
    if "steps" not in artifact:
        raise ValueError("interactive report requires retained execution steps")
    # Closed allowlist: request text, token IDs and backend inputs stay out.
    steps = []
    for step in artifact["steps"]:
        row = {name: step[name] for name in (
            "step_id", "pool_name", "schedule_interval", "timing")}
        row["execution_timeline"] = _safe_timeline(step["execution_timeline"])
        row["kernel_plan"] = _safe_plan(step["kernel_plan"])
        row["requests"] = [{name: request[name] for name in (
            "request_id", "phase", "input_tokens", "output_tokens")}
            for request in step["backend_step"]["schedule"]["requests"]]
        # Earlier saved projections did not retain prediction diagnostics.
        # Their absence is explicit; never reconstruct model timing from work.
        row["predictions"] = _safe_predictions(step["predictions"]) if "predictions" in step else []
        steps.append(row)
    safe_steps = [{**step, "backend_step": {"schedule": {"requests": step["requests"]}}}
        for step in steps]
    safe_transfers = []
    for transfer in artifact["transfers"]:
        safe_transfer = {name: transfer[name] for name in (
            "request_id", "source_engine_id", "destination_engine_id", "trace_transfer")}
        safe_transfer["execution_timeline"] = _safe_timeline(transfer["execution_timeline"])
        safe_transfer["kernel_plan"] = _safe_plan(transfer["kernel_plan"])
        safe_transfer["predictions"] = _safe_predictions(transfer["predictions"]) if "predictions" in transfer else []
        safe_transfers.append(safe_transfer)
        steps.append({"step_id": transfer["trace_transfer"]["transfer_id"],
            "pool_name": transfer["source_engine_id"],
            "kernel_plan": safe_transfer["kernel_plan"],
            "predictions": safe_transfer["predictions"],
            "execution_timeline": safe_transfer["execution_timeline"],
            "requests": [{"request_id": transfer["request_id"], "phase": "kv_handoff"}],
            "transfer": transfer["trace_transfer"]})
    data = {
        "steps": steps,
        "experiment_configuration": report_experiment_configuration(payload),
        "routing": artifact["schedule_trace"]["route_decisions"],
        "resource_summary": payload["visualization_summary"] if "visualization_summary" in payload else None,
        "coverage": payload["visualization_coverage"] if "visualization_coverage" in payload else None,
        "links": payload["visualization_links"] if "visualization_links" in payload else [],
        "schedule": payload["visualization_schedule"] if "visualization_schedule" in payload else [],
        "hardware": payload["visualization_hardware"] if "visualization_hardware" in payload else None,
        "capacity": payload["kv_capacity"] if "kv_capacity" in payload else None,
        "metrics": {name: payload["metrics"][name] for name in (
            "requests", "request_throughput_per_s")},
        "prediction_method": payload["prediction_method"] if "prediction_method" in payload else None,
        "launch_policy": payload["launch_policy"] if "launch_policy" in payload else None,
        "simulation_wall_time_s": payload["simulation_wall_time_s"],
        "schedule_end_time_s": payload["schedule_end_time_s"],
        "resources": payload["visualization_resources"] if "visualization_resources" in payload else None,
        "prediction_diagnostics": payload["visualization_diagnostics"] if "visualization_diagnostics" in payload else None,
        "source_activation_memory": payload["source_activation_memory"] if "source_activation_memory" in payload else None,
    }
    _validate_diagnostics(data["prediction_diagnostics"], data["resource_summary"])
    if data["coverage"] is not None:
        if "detail_intervals_s" in data["coverage"]:
            intervals = data["coverage"]["detail_intervals_s"]
            previous_end = -math.inf
            for interval in intervals:
                if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval)
                        or interval[0] < previous_end or interval[1] < interval[0]):
                    raise ValueError("invalid visualization detail coverage intervals")
                previous_end = interval[1]
            for name in ("requests", "steps"):
                detailed, total = data["coverage"]["detailed_" + name], data["coverage"]["total_" + name]
                if type(detailed) is not int or type(total) is not int or not 0 <= detailed <= total:
                    raise ValueError("invalid visualization detail coverage counts")
            detail_interval = [intervals[0][0], intervals[-1][1]] if intervals else None
        else:
            detail_interval = ([min(step["schedule_interval"]["start_time_s"] for step in safe_steps),
                max(step["schedule_interval"]["end_time_s"] for step in safe_steps)] if safe_steps else None)
        data["coverage"] = {**data["coverage"], "detail_interval_s": detail_interval,
            "aggregate_interval_s": [0.0, data["schedule_end_time_s"]],
            "omitted_detail_events": data["coverage"]["total_events"] - data["coverage"]["detail_events"]}
    encoded = json.dumps(_pack_projection(data), separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
    embedded = base64.b64encode(compressed).decode("ascii")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(with_serving_viewer_ui(_HTML).replace("__DATA__", embedded))
    safe_payload = {
        "execution_artifact": {"steps": safe_steps, "transfers": safe_transfers,
            "schedule_trace": {"route_decisions": data["routing"]}},
        "metrics": data["metrics"], "simulation_wall_time_s": data["simulation_wall_time_s"],
        "prediction_method": data["prediction_method"], "launch_policy": data["launch_policy"],
        "schedule_end_time_s": data["schedule_end_time_s"],
        "visualization_resources": data["resources"],
        "visualization_diagnostics": data["prediction_diagnostics"],
        "source_activation_memory": data["source_activation_memory"],
        "visualization_summary": data["resource_summary"],
        "visualization_coverage": data["coverage"],
        "visualization_links": data["links"],
        "visualization_schedule": data["schedule"],
        "visualization_hardware": data["hardware"],
        "kv_capacity": data["capacity"],
        "visualization_experiment_configuration": data["experiment_configuration"],
    }
    data_path = output.with_suffix(".visualization.json")
    safe_payload["visualization_projection"] = {
        "resource_times": "nonzero entries; omitted resource times are zero",
        "omitted": ["t_per_group_s", "source_node_ids", "host_dependency_edges", "covered_call_occurrences"],
        "time_basis": "absolute simulation seconds; exact original endpoints"}
    data_path.write_text(json.dumps(_pack_projection(safe_payload), separators=(",", ":"), allow_nan=False))
    return {"page_generation_s": time.perf_counter() - started,
            "html_bytes": output.stat().st_size, "data_bytes": data_path.stat().st_size,
            "embedded_uncompressed_bytes": len(encoded), "embedded_gzip_bytes": len(compressed),
            "data_path": str(data_path), "path": str(output)}



_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Midlayer Serving timeline</title>
<style>
:root{color-scheme:light}body{font:14px system-ui,-apple-system,sans-serif;margin:0;color:#253747;background:#f5f7f9}.page{max-width:1600px;margin:auto;padding:24px 28px 56px}h1{font-size:25px;margin:0 0 6px}h2{font-size:19px;margin:0 0 12px}p{color:#536675;line-height:1.55;margin:8px 0}.subtitle{margin-bottom:18px}.card{background:white;border:1px solid #d8e1e7;border-radius:10px;padding:18px;margin:16px 0;box-shadow:0 2px 10px #1632470a}.metrics{display:flex;gap:30px;flex-wrap:wrap}.metric strong{font-size:21px;display:block;color:#17324a}.metric span{color:#607889;font-size:12px}.controls{display:flex;gap:8px 16px;flex-wrap:wrap;align-items:end;margin:12px 0}label{display:inline-flex;flex-direction:column;gap:4px;color:#526b79;font-size:12px}button,input,select{padding:7px 9px;color:#203846;background:white;border:1px solid #b9c8d1;border-radius:5px;font:inherit}button{cursor:pointer;background:#f2f7fa}button:hover{background:#dcecf4}input[type=number]{width:130px}input[type=range]{width:190px}canvas{background:#fff;width:100%;display:block}.chart-wrap{overflow-x:auto;border:1px solid #d8e1e7;border-radius:7px}.legend{display:flex;gap:7px 16px;flex-wrap:wrap;margin:10px 0;color:#465d69;font-size:12px}.legend span{display:inline-flex;align-items:center;gap:5px}.swatch{width:13px;height:13px;display:inline-block;border-radius:2px}.notice{background:#eef6fc;border-left:3px solid #438fbb;padding:9px 12px;font-size:13px}.warning{background:#fff8eb;border-left-color:#d19b43}.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px 20px}.detail-item{border-bottom:1px solid #edf1f3;padding:7px 0}.detail-item strong{display:block;font-size:14px;overflow-wrap:anywhere}.detail-item span{display:block;color:#687e89;font-size:12px}pre{white-space:pre-wrap;max-height:420px;overflow:auto;font-size:12px}.table-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid #e3e9ec;border-radius:6px}table.scroll-table{min-width:640px}table{border-collapse:collapse;width:100%;margin:10px 0}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #e3e9ec}th{color:#617985;font-size:12px;white-space:nowrap}summary{cursor:pointer;padding:10px 0;color:#315c76}.resource-ranks{display:flex;flex-wrap:wrap;gap:8px}.rank-card{flex:1;min-width:210px;background:#f6f9fb;padding:10px;border-radius:6px}.rank-card b{display:block}.rank-card span{font-size:12px;color:#617985}#coverage{font-size:12px}
#timeline,#detail-timeline,#diagnostic-timeline{cursor:grab;touch-action:none}#timeline:active{cursor:grabbing}.bucket-layout{display:flex;align-items:center;gap:28px;flex-wrap:wrap}.bucket-layout svg{width:260px;height:260px;flex-shrink:0}.bucket-layout table{flex:1;min-width:340px}.bucket-layout td:first-child{white-space:nowrap}</style><div class="page"><h1>推理执行时间线</h1><p class="subtitle">同一模拟时间轴：按实例和设备查看计算、通信、KV 搬运与 Host 提交。所有时间均为模型预测。</p>
<div class="card"><div id="summary" class="metrics"></div><p id="method"></p></div>
<section class="card" id="card-overview"><h2>1 · 物理卡时间线</h2><p class="notice">一张卡一行，颜色表示操作类型；点卡名看细分，滚轮平移，Alt+滚轮缩放。</p>
<div class="controls"><label>实例<select id="pool"></select></label><label>设备 rank<select id="rank"></select></label><label>操作类型<select id="kind"></select></label><label>请求（筛选参与批次）<select id="request"></select></label><label>调度批次<select id="step"></select></label><button id="show-breakdown">细分时间线</button><button id="first-layer" title="定位一个 decode step、一个 microbatch 的首层执行；保留窗口内的其他并发活动。">首层细节</button><button id="zoom">放大 2 倍</button><button id="reset">保存窗口</button></div>
<div class="controls"><label>开始（秒）<input id="start" type="number" step="any" value="0"></label><label>结束（秒）<input id="end" type="number" step="any"></label><label>平移时间范围<input id="range" type="range" min="0" max="1000" value="0"></label></div>
<div class="legend" id="legend"></div><p id="coverage"></p><div class="chart-wrap"><canvas id="timeline" height="360"></canvas></div>
<h2 style="margin-top:18px">选中的操作</h2><div id="detail-card" class="detail-grid"><p>点击时间线中的色块。</p></div><details id="operation-diagnostics"><summary>展开此操作的瓶颈与资源诊断</summary><div id="detail-diagnostics" class="detail-grid"></div><pre id="detail">点击时间线中的色块。</pre></details></section>
<section class="card" id="device-detail-section"><details id="detail-lanes"><summary><strong>2 · Timeline Breakdown · 展开细分泳道（默认折叠）</strong></summary><p>FPGA同卡双芯片的计算、DDR与通信分带显示，保留实际重叠与起止时间。点击色块查看操作；拖动时间线可平移。</p><div class="controls"><label>物理卡<select id="detail-card-select"><option value="All">全部卡</option></select></label><span>FPGA按芯片分别显示计算、DDR与通信；Host与等待明细保留。</span></div><div class="chart-wrap"><canvas id="detail-timeline" height="200"></canvas></div></details></section>
<section class="card" id="time-buckets"><h2>3 · 时间分桶</h2><div class="controls"><label>统计口径<select id="bucket-channel"><option value="cumulative_buckets">累计操作时间</option><option value="gpu_compute_buckets">GPU计算通道（含空闲）</option><option value="fpga_compute_buckets">FPGA计算通道（含空闲）</option><option value="fpga_ddr_buckets">FPGA DDR通道（含空闲）</option></select></label></div><p id="bucket-scope"></p><div id="bucket-chart" class="bucket-layout"></div></section>
<section class="card" id="diagnostic-section"><details id="bottleneck-diagnostics"><summary><strong>4 · 瓶颈、逻辑HBM与资源诊断（默认折叠）</strong></summary><p class="notice warning">逻辑HBM需求与资源下界不是实测带宽或端到端关键路径；诊断数据保留，未记录字段保持未知。</p><div class="chart-wrap"><canvas id="diagnostic-timeline" height="200"></canvas></div><div id="window-insight"></div><div id="resource-table"></div><label>资源随时间变化<select id="resource-metric"><option value="busy">设备忙碌占比</option><option value="compute">最大算力资源需求</option><option value="hbm">HBM 逻辑需求下界</option></select></label><div class="chart-wrap"><canvas id="utilization" height="220"></canvas></div><details><summary>路由与请求排队</summary><div id="routing"></div></details><details><summary>单位、分母、容量与原始证据</summary><pre id="resources"></pre></details></details></section></div>
<script id="data" type="application/octet-stream">__DATA__</script>
<p id="load-status" role="status">正在解压并加载时间线…</p><script type="text/plain" id="viewer-code">
'use strict';
const d=window.midlayerReportData,events=[];if(d.experiment_configuration)window.renderExperimentConfiguration(d.experiment_configuration);
const scenario=d.experiment_configuration?.deployment?.actual_scenario_configuration||null;const isFlashFp4=!!(scenario&&['hygon_bw1100','nvidia_rtx_pro_5500'].includes(scenario.hardware_id)&&['tp8_ep8','dp8_ep8','dp8_fpga'].includes(scenario.scheme));
const BUCKETS=[['attention','Attention / KV','#377EB8'],['indexer','稀疏索引','#8C6BB1'],['routed','路由专家','#E58606'],['shared','共享专家','#E6AB02'],['route_merge','路由 / 合并 / 本地归约','#4DAF4A'],['communication','通信','#D95F76'],['staging','DDR搬运','#17A2A4'],['other','其他 / 归一化 / 采样','#8D99A6'],['dense_ffn','Dense FFN / MLP','#59A14F'],['projections','QKV / output projections','#4E79A7'],['lm_head','LM head','#B07AA1']];
function operationBucket(e){const source=String(e.call_id||'')+' '+String(e.category||'')+' '+String(e.name||'');if(e.kind==='COMMUNICATION')return BUCKETS[5];if(e.kind==='MEMCPY'&&e.pool_name==='experts')return BUCKETS[6];if(/local-sum|本地归约/.test(source))return BUCKETS[4];if(/expert-group|MoE GEMM|MoE 计算/.test(source))return BUCKETS[2];if(/:shared(?=:|$)|共享专家/.test(source))return BUCKETS[3];if(e.kind==='SPARSE_ATTN_INDEXER'||/indexer|稀疏索引/.test(source))return BUCKETS[1];if(/router|routing|route|merge|dispatch-pack|expert-lane|shared-add|pack-experts|moe_sum|专家路由/.test(source))return BUCKETS[4];if(/(?:^|:)lm-head(?=:|$)/.test(String(e.call_id||'')))return BUCKETS[10];if(/FFN 投影|FFN projection/.test(source)||/(?:^|:)(?:gate[-_]up|down|ffn|silu)(?=:|$)/.test(String(e.call_id||'')))return BUCKETS[8];if(/QKV|输出投影|Output projection/.test(source)||/(?:^|:)(?:qkv|q-proj|kv-proj|o-proj|out-proj)(?=:|$)/.test(String(e.call_id||'')))return BUCKETS[9];if(/ATTN/.test(e.kind)||['KV_CACHE','KV_OFFLOAD'].includes(e.kind)||/:attn|Attention|KV 更新/.test(source))return BUCKETS[0];return BUCKETS[7];}
function physicalCard(e){const rank=Number(e.rank);if(isFlashFp4&&Number.isInteger(rank)){if(rank>=0&&rank<8)return 'GPU '+rank;if(scenario.scheme==='dp8_fpga'&&rank>=8&&rank<40)return 'FPGA '+Math.floor((rank-8)/2);return null;}return e.pool_name+' · rank '+e.rank;}
function communicationPlacements(c,s){const t=c.kernel_spec.transfer;if(!t||c.kernel_spec.kind!=='COMMUNICATION')return [{pool_name:c.pool_name,rank:c.rank}];return t.participants.flatMap((rank,i)=>{if(t.traffic_matrix_bytes?.length&&!t.traffic_matrix_bytes[i].some(x=>x>0)&&!t.traffic_matrix_bytes.some(row=>row[i]>0))return [];if(isFlashFp4&&t.link==='pcie'&&rank>=8)return [];const pool=isFlashFp4?(rank<8?'decode_gpu':'experts'):(s.transfer?(rank===t.source_rank?s.transfer.source_pool_id:s.transfer.destination_pool_id):c.pool_name);return [{pool_name:pool,rank}];});}
const linksByCall=new Map(d.links.map(link=>[link.call_id,link]));
// These names only group source operations for display; they do not price work.
const palette=Object.fromEntries(BUCKETS.map(x=>[x[1],x[2]]));
function modelGroup(group){
 const known={'launch':['设备启动 / 分派','#c9a05a'],'pipeline':['多资源流水线','#8a6aa4'],'analytical_execution_floor':['解析执行下限（资源未细分）','#9eabb2'],'analytical_sampler_memory':['采样 · HBM 逻辑需求','#477dae'],'dram':['HBM / DRAM','#477dae'],'l2':['L2 缓存','#748bb7'],'l1_cache':['L1 缓存','#748bb7'],'smem':['共享内存','#748bb7'],'xbar':['片上互连','#748bb7'],'tensor_pipe':['Tensor Core','#b56e3f'],'fma_sfu_pipe':['FMA / SFU','#b56e3f'],'alu_pipe':['ALU','#b56e3f'],'warp_issue':['Warp 调度','#b56e3f'],'communication':['通信','#52897b'],'roofline':['Roofline（未细分）','#9eabb2']};
 if(group&&Object.prototype.hasOwnProperty.call(known,group))return known[group];if(group&&group.startsWith('dependency:'))return ['依赖路径 · '+group.slice(11),'#8a6aa4'];return [group?'模型组：'+group:'未保留','#9eabb2'];
}
function resourceBounds(usage){
 if(!usage||usage.availability!=='available')return {status:'unavailable',reason:usage&&usage.reason?usage.reason:'没有保留此操作的资源输入'};
 const demands=usage.compute.map(c=>({role:c.role,time_s:c.operations/c.capacity_operations_per_s}));
 const dominant=demands.reduce((a,b)=>a.time_s>=b.time_s?a:b,{role:null,time_s:0});
 const memory=usage.logical_bytes/usage.hbm_capacity_bytes_per_s;
 if(memory===0&&dominant.time_s===0)return {status:'unavailable',reason:'没有正的计算或逻辑字节需求'};
 return {status:'available',dominant:memory>dominant.time_s?'HBM':memory<dominant.time_s?'compute':'tie',compute_role:dominant.role,compute_lower_bound_s:dominant.time_s,logical_hbm_lower_bound_s:memory};
}
for(const s of d.steps){
 const calls=new Map(s.kernel_plan.calls.map(c=>[c.call_id,c]));
 const predictions=new Map(s.predictions.map(p=>[p.call_id,p]));
 const t=s.execution_timeline,repeated=new Set();
 function device(i){const c=calls.get(i.call_id);const transfer=c.kernel_spec.transfer;const placements=communicationPlacements(c,s);const evidence=d.resources&&Object.prototype.hasOwnProperty.call(d.resources,c.call_id)?d.resources[c.call_id]:null;for(const placement of placements)events.push({...i,...placement,resource:c.resource,shared_call_id:c.call_id,name:c.kernel_spec.kind+(c.kernel_spec.variant?' / '+c.kernel_spec.variant:''),category:c.display_category,layer:c.layer,kind:c.kernel_spec.kind,domain:'Device',step:s.step_id,requests:s.requests,phase:[...new Set(s.requests.map(r=>r.phase))].join('+'),shape:c.kernel_spec.shape,dtype:c.kernel_spec.dtype,transfer:c.kernel_spec.transfer,handoff:s.transfer||null,link_evidence:linksByCall.has(c.call_id)?linksByCall.get(c.call_id):null,prediction:predictions.has(c.call_id)?predictions.get(c.call_id):null,resource_evidence:evidence,resource_bounds:resourceBounds(evidence),scope:'scheduled batch operation; shared across its requests'});}
 if(t.block_repeats)for(const b of t.block_repeats)for(const c of b.call_intervals){repeated.add(c.call_id);for(const i of c.intervals)device({...i,call_id:c.call_id,...Object.fromEntries(['pool_name','rank','resource'].map(k=>[k,calls.get(c.call_id)[k]])),repeat_block:b.block_id,repeat_count:b.repeat_count});}
 for(const i of t.intervals)if(!repeated.has(i.call_id))device(i);
 for(const h of t.host_events)events.push({...h,name:h.event_kind,category:'Host 提交',kind:h.event_kind,domain:'Host',step:s.step_id,requests:s.requests,phase:[...new Set(s.requests.map(r=>r.phase))].join('+')});
}
function linkStats(a,b){
 const rows={};for(const l of d.links){if(l.availability!=='available')continue;
 if($('#pool').value!=='All'&&$('#pool').value!==l.pool_name)continue;
 for(const rank of l.participants){if($('#rank').value!=='All'&&$('#rank').value!==String(rank))continue;
 const key=l.link+' / '+String(l.resource_id)+' / endpoint '+rank;
 if(!rows[key])rows[key]={wire_bytes:0,capacity_bytes_per_s:l.capacity_bytes_per_s,basis:l.basis};
 for(const [st,en] of l.intervals)if(en>st)rows[key].wire_bytes+=l.wire_bytes_per_endpoint*Math.max(0,Math.min(b,en)-Math.max(a,st))/(en-st);
 }}for(const r of Object.values(rows))r.utilization=r.wire_bytes/(r.capacity_bytes_per_s*(b-a));return rows;
}
function drawRouting(){
 const root=$('#routing');root.replaceChildren();const table=document.createElement('table');table.className='scroll-table';const header=document.createElement('tr');for(const label of ['请求 / 阶段','来源 → 选中实例','候选实例排队 token','排队时长 / 派发时刻（秒）']){const cell=document.createElement('th');cell.textContent=label;header.append(cell);}table.append(header);
 let count=0;for(const r of d.routing){if($('#pool').value!=='All'&&r.selected_engine_id!==$('#pool').value&&r.source_engine_id!==$('#pool').value)continue;const row=document.createElement('tr');for(const text of [r.request_id+' / '+r.stage,(r.source_engine_id||'ingress')+' → '+r.selected_engine_id,JSON.stringify(r.candidate_queued_tokens),(r.queue_exit_time_s-r.queue_enter_time_s).toPrecision(4)+' / '+r.dispatch_time_s.toPrecision(5)]){const td=document.createElement('td');td.textContent=text;row.append(td);}table.append(row);count++;}root.append(count?table:document.createTextNode('所选实例没有记录到路由选择。'));
}
function drawResourceTable(a,b,selected){
 const target=$('#resource-table');target.replaceChildren();const table=document.createElement('table');table.className='scroll-table';
 const header=document.createElement('tr');for(const text of ['实例 / rank','设备忙碌','最大算力需求','HBM 逻辑需求下界','资源证据']){const th=document.createElement('th');th.textContent=text;header.append(th);}table.append(header);
 const rows=resourceSummary(selected,a,b),percent=x=>x===null?'不可用':(x*100).toFixed(2)+'%';
 for(const [name,r] of Object.entries(rows)){const values=Object.values(r.compute).map(c=>c.utilization),tr=document.createElement('tr');for(const text of [name,percent(r.device_busy_fraction),percent(values.length?Math.max(...values):null),percent(r.logical_hbm_demand_lower_bound_fraction),r.available+' 个有依据 / '+r.unavailable+' 个不可用']){const td=document.createElement('td');td.textContent=text;tr.append(td);}table.append(tr);}target.append(table);
 if(d.hardware){for(const [name,h] of Object.entries(d.hardware)){const line=document.createElement('p');line.textContent=name+' · 每张设备 HBM 容量 '+(h.hbm_capacity_bytes/2**30).toFixed(2)+' GiB · 预测模型配置带宽 '+h.hbm_bandwidth_bytes_per_s.toExponential(3)+' B/s';target.append(line);}}
 if(d.source_activation_memory){for(const [pool,p] of Object.entries(d.source_activation_memory.pools))for(const [rank,r] of Object.entries(p.ranks)){const line=document.createElement('p');line.textContent=pool+' / rank '+rank+' · 已知 activation 峰值 '+(r.known_activation_peak_bytes/2**20).toFixed(3)+' MiB（不完整）；正常容量验收 = '+(d.source_activation_memory.normal_capacity_accepted?'是':'否');target.append(line);}}else{const line=document.createElement('p');line.textContent='已知 activation 峰值：不可用（没有来源分配记录）';target.append(line);}
 const links=linkStats(a,b);for(const [name,r] of Object.entries(links)){const line=document.createElement('p');line.textContent=name+': '+percent(r.utilization)+' · '+r.wire_bytes.toFixed(0)+' B 算法传输量 · '+r.capacity_bytes_per_s.toExponential(3)+' B/s / 端点';target.append(line);}
}
function drawHistory(a,b){
 const canvas=$('#utilization'),c=canvas.getContext('2d'),w=canvas.width=canvas.clientWidth*devicePixelRatio,h=canvas.height=220*devicePixelRatio,left=220*devicePixelRatio;canvas.style.height='220px';
 const n=48,series={};let max=1;
 for(let i=0;i<n;i++){const rows=d.resource_summary?aggregateSummary(a+(b-a)*i/n,a+(b-a)*(i+1)/n):resourceSummary(events,a+(b-a)*i/n,a+(b-a)*(i+1)/n);
 for(const [key,r] of Object.entries(rows)){if(!series[key])series[key]=Array(n).fill(null);const metric=$('#resource-metric').value;const value=metric==='busy'?r.device_busy_fraction:metric==='hbm'?r.logical_hbm_demand_lower_bound_fraction:(Object.keys(r.compute).length?Math.max(...Object.values(r.compute).map(c=>c.utilization)):null);series[key][i]=value;if(value!==null)max=Math.max(max,value);}}
 c.font=`${12*devicePixelRatio}px system-ui`;c.fillStyle='#526b79';c.fillText('0 – '+(max*100).toFixed(1)+'% · 48 个时间窗',8*devicePixelRatio,20*devicePixelRatio);
 const colors=['#267f9f','#bb753b','#7555a3','#b35160','#60963f'];Object.entries(series).forEach(([key,values],j)=>{c.strokeStyle=c.fillStyle=colors[j%colors.length];c.fillText(key,8*devicePixelRatio,(42+j*19)*devicePixelRatio);c.beginPath();let connected=false;values.forEach((v,i)=>{if(v===null){connected=false;return;}const x=left+(w-left)*(i+.5)/n,y=h-20*devicePixelRatio-v/max*(h-40*devicePixelRatio);if(connected)c.lineTo(x,y);else c.moveTo(x,y);connected=true;});c.stroke();});
}
function aggregateSummary(a,b){
 const rows={};
 for(const s of d.resource_summary){
 if($('#pool').value!=='All'&&$('#pool').value!==s.pool_name)continue;
 if($('#rank').value!=='All'&&$('#rank').value!==String(s.rank))continue;
 const start=Math.max(a,s.start_time_s),end=Math.min(b,s.end_time_s);if(end<=start)continue;
 const key=s.pool_name+' / '+s.rank;if(!rows[key])rows[key]={busy:[],busy_unknown:false,compute:{},logical_bytes:0,hbm_capacity:null,available:0,unavailable:0};const r=rows[key],f=(end-start)/(s.end_time_s-s.start_time_s);
 if(s.busy_intervals===null)r.busy_unknown=true;else for(const [st,en] of s.busy_intervals)if(en>a&&st<b)r.busy.push([Math.max(a,st),Math.min(b,en)]);
 r.logical_bytes+=s.logical_bytes*f;if(s.hbm_capacity!==null)r.hbm_capacity=s.hbm_capacity;r.available+=s.available;r.unavailable+=s.unavailable;
 for(const [k,c] of Object.entries(s.compute)){if(!r.compute[k])r.compute[k]={operations:0,capacity_operations_per_s:c.capacity_operations_per_s};r.compute[k].operations+=c.operations*f;}
 }
 for(const r of Object.values(rows)){r.busy.sort((x,y)=>x[0]-y[0]);let end=-Infinity,busy=0;for(const [s,e] of r.busy){busy+=Math.max(0,e-Math.max(s,end));end=Math.max(end,e);}r.device_busy_fraction=r.busy_unknown?null:busy/(b-a);delete r.busy;for(const c of Object.values(r.compute))c.utilization=c.operations/(c.capacity_operations_per_s*(b-a));r.logical_hbm_demand_lower_bound_fraction=r.hbm_capacity===null?null:r.logical_bytes/(r.hbm_capacity*(b-a));}
 return rows;
}
function resourceSummary(selected,a,b){
 if(d.resource_summary)return aggregateSummary(a,b);
 const rows={};
 for(const e of selected){if(e.domain!=='Device')continue;const st=Math.max(a,e.absolute_start_time_s),en=Math.min(b,e.absolute_end_time_s);if(en<=st)continue;
 const key=e.pool_name+' / '+e.rank;if(!rows[key])rows[key]={busy:[],busy_unknown:false,compute:{},logical_bytes:0,hbm_capacity:null,available:0,unavailable:0};const r=rows[key];r.busy.push([st,en]);
 const u=d.resources&&d.resources[e.call_id];if(!u||u.availability!=='available'){r.unavailable++;continue;}r.available++;
 const fraction=(en-st)/(e.absolute_end_time_s-e.absolute_start_time_s);r.logical_bytes+=u.logical_bytes*fraction;r.hbm_capacity=u.hbm_capacity_bytes_per_s;
 for(const c of u.compute){const k=u.dtype+' / '+c.role;if(!r.compute[k])r.compute[k]={operations:0,capacity_operations_per_s:c.capacity_operations_per_s};r.compute[k].operations+=c.operations*fraction;}
 }
 for(const r of Object.values(rows)){r.busy.sort((x,y)=>x[0]-y[0]);let end=-Infinity,busy=0;for(const [s,e] of r.busy){busy+=Math.max(0,e-Math.max(s,end));end=Math.max(end,e);}r.device_busy_fraction=r.busy_unknown?null:busy/(b-a);delete r.busy;for(const c of Object.values(r.compute))c.utilization=c.operations/(c.capacity_operations_per_s*(b-a));r.logical_hbm_demand_lower_bound_fraction=r.hbm_capacity===null?null:r.logical_bytes/(r.hbm_capacity*(b-a));}
 return rows;
}
if(d.resource_summary){const detailed=new Set(d.steps.map(s=>s.step_id));for(const row of d.resource_summary)if(!detailed.has(row.step_id)){
 events.push({name:'Model step · aggregated envelope',kind:'aggregated_step',domain:'Device',pool_name:row.pool_name,rank:row.rank,absolute_start_time_s:row.start_time_s,absolute_end_time_s:row.end_time_s,step:row.step_id,aggregate_count:row.device_events,operations:row.operation_counts||'unavailable in this saved projection',model_bottleneck_groups:row.model_bottleneck_groups||null,phase:row.phases||'unavailable in this saved projection',scope:row.busy_intervals===null?'activity intervals unavailable; Forward envelope is not exact busy time':'envelope includes gaps; exact busy union and all resource demand remain in resource summaries; operation details omitted'});
 if(row.host_intervals&&row.host_intervals.length)events.push({name:'Host submissions · aggregated envelope',kind:'aggregated_host',domain:'Host',pool_name:row.pool_name,rank:row.rank,absolute_start_time_s:Math.min(...row.host_intervals.map(x=>x[0])),absolute_end_time_s:Math.max(...row.host_intervals.map(x=>x[1])),step:row.step_id,aggregate_count:row.host_intervals.length,scope:'Host event envelope, includes gaps; not continuous CPU busy'});
}}
for(const s of d.schedule){
 let cursor=s.start_time_s;
 const parts=[['scheduler_plan',s.scheduler_plan_cpu_s],['device_queue_wait',s.device_queue_wait_s],['execution',s.execution_wall_s],['completion_queue_wait',s.completion_queue_wait_s],['scheduler_update',s.scheduler_update_cpu_s]];
 for(const [name,duration] of parts){if(name!=='execution'&&duration>0)events.push({name,kind:name,domain:name.includes('wait')?'Wait':'Host',pool_name:s.pool_name,rank:'shared',absolute_start_time_s:cursor,absolute_end_time_s:cursor+duration,step:s.step_id,phase:'schedule',source:'ScheduleTraceEntry'});cursor+=duration;}
}
for(const r of d.routing){if(r.queue_exit_time_s>r.queue_enter_time_s)events.push({name:'request route queue',kind:'route_queue_wait',domain:'Wait',pool_name:r.selected_engine_id,rank:'shared',absolute_start_time_s:r.queue_enter_time_s,absolute_end_time_s:r.queue_exit_time_s,request_id:r.request_id,phase:r.stage,source:'RouteDecision'});}
for(const e of events)if(!e.category)e.category=e.domain==='Wait'?'等待':e.domain==='Host'?'Host 提交':'聚合区间';
const $=s=>document.querySelector(s),total=d.schedule_end_time_s;
function phaseName(phase){const names={'prefill':'预填充','decode':'逐 token 解码','kv_handoff':'KV 交接','mixed':'混合'};return Object.prototype.hasOwnProperty.call(names,phase)?names[phase]:phase;}
function clock(s){if(Math.abs(s)>=1)return s.toFixed(6)+' s';if(Math.abs(s)>=0.001)return (s*1000).toFixed(3)+' ms';return (s*1e6).toFixed(2)+' µs';}
function duration(s){return clock(s);}
function roleName(role){const roles={'compute_tensor':'Tensor Core','compute_vector':'向量单元','compute_fma':'FMA','compute_alu':'ALU','compute_reduction':'归约单元','compute_sfu':'特殊函数单元','tensor_core_bf16':'Tensor Core BF16','tensor_core_fp16':'Tensor Core（fp16 资源组）','tensor_core_fp8':'Tensor Core FP8','shared_mem':'共享内存','block_reduction_dependency':'归约依赖','tensor_pipe':'Tensor Core 管线','fma_pipe':'FMA 管线','sfu_pipe':'SFU 管线','alu_pipe':'ALU 管线','dram':'HBM / DRAM','l2':'L2 缓存','l1_cache':'L1 缓存','smem':'共享内存','xbar':'片上互连'};return role&&Object.prototype.hasOwnProperty.call(roles,role)?roles[role]:role||'未提供';}
function addMetric(label,value){const node=document.createElement('div');node.className='metric';const a=document.createElement('span'),b=document.createElement('strong');a.textContent=label;b.textContent=value;node.append(a,b);$('#summary').append(node);}
const requestMetrics=Object.values(d.metrics.requests);const mean=key=>{const values=requestMetrics.map(r=>r[key]).filter(x=>typeof x==='number');return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;};
$('#method').textContent='预测方法：'+(d.prediction_method||'旧投影未记录')+'；发射策略：'+(d.launch_policy||'旧投影未记录')+'。页面数值来自解析仿真，非实机观测。';
const steady=d.experiment_configuration&&d.experiment_configuration.steady_metrics;if(steady){addMetric('稳态输出吞吐',steady.steady_output_tokens_per_s.toFixed(2)+' token / 秒');addMetric('稳态 TPOT',duration(steady.steady_tpot_mean_s));addMetric('总并发',String(steady.concurrency));}else{addMetric('完成请求',String(requestMetrics.length));addMetric('平均首字延迟 TTFT',mean('ttft_s')===null?'不可用':duration(mean('ttft_s')));addMetric('平均每 token 延迟 TPOT',mean('tpot_s')===null?'不可用':duration(mean('tpot_s')));addMetric('请求吞吐',typeof d.metrics.request_throughput_per_s==='number'&&Number.isFinite(d.metrics.request_throughput_per_s)?d.metrics.request_throughput_per_s.toFixed(2)+' 请求 / 秒':'未统计');}addMetric('仿真程序耗时',d.simulation_wall_time_s.toFixed(2)+' 秒');
for(const [id,key] of [['pool','pool_name'],['rank','rank'],['kind','category']]){const el=$('#'+id);for(const v of ['All',...new Set(events.map(e=>String(e[key])))]){const o=document.createElement('option');o.value=v;o.textContent=v==='All'?'全部':v;if(id==='pool'&&v!=='All')o.setAttribute('data-no-translate','');el.append(o);}el.onchange=draw;}
const stepSelect=$('#step'),requestSelect=$('#request');
if(steady&&requestMetrics.length===0){requestSelect.closest('label').style.display='none';stepSelect.closest('label').style.display='none';}
for(const id of ['All',...[...new Set(d.steps.flatMap(step=>step.requests.map(r=>r.request_id)))].sort()]){const o=document.createElement('option');o.value=id;o.textContent=id==='All'?'全部已保留明细的请求':id;if(id!=='All')o.setAttribute('data-no-translate','');requestSelect.append(o);}
function matchingSteps(){return d.steps.filter(step=>requestSelect.value==='All'||step.requests.some(r=>r.request_id===requestSelect.value));}
function defaultStep(){const steps=matchingSteps();return steps.length?steps.reduce((a,b)=>b.requests.length>a.requests.length?b:a).step_id:'All';}
function fillSteps(){stepSelect.replaceChildren();const all=document.createElement('option');all.value='All';all.textContent='全部批次（全程汇总，明细按覆盖范围）';stepSelect.append(all);matchingSteps().forEach((step,index)=>{const o=document.createElement('option');o.value=step.step_id;const phases=[...new Set(step.requests.map(r=>phaseName(r.phase)))].join(' + ');o.textContent=`${index+1}. ${step.pool_name} · ${phases} · ${step.requests.length} 个请求 · ${step.step_id}`;stepSelect.append(o);});stepSelect.value=defaultStep();}
fillSteps();requestSelect.onchange=()=>{fillSteps();firstLayer();};stepSelect.onchange=()=>{if(stepSelect.value==='All')setRange(0,total);else firstLayer();};
for(const [key,name,color] of BUCKETS){if(!events.some(e=>operationBucket(e)[0]===key))continue;const item=document.createElement('span'),swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=color;item.append(swatch,document.createTextNode(name));$('#legend').append(item);}
$('#end').value=total;
const cv=$('#timeline'),ctx=cv.getContext('2d');let hits=[],layerFocus=null;
function bounds(){return [Number($('#start').value),Number($('#end').value)];}
function rangeDomain(){if(isFlashFp4&&d.coverage?.detail_interval_s)return d.coverage.detail_interval_s;if(stepSelect.value==='All')return [0,total];const step=d.steps.find(s=>s.step_id===stepSelect.value);if(!step)return [0,total];const span=step.schedule_interval||step.transfer;if(!span||span.end_time_s<=span.start_time_s)return [0,total];return [span.start_time_s,span.end_time_s];}
function setRange(a,b,focus=null){const [low,high]=rangeDomain(),length=high-low,width=Math.min(length,Math.max(Number.EPSILON*Math.max(1,total),b-a));a=Math.max(low,Math.min(high-width,a));$('#start').value=a;$('#end').value=a+width;$('#range').value=length>width?Math.round(1000*(a-low)/(length-width)):0;layerFocus=focus;draw();}
function firstLayer(){
 if(stepSelect.value==='All'&&d.steps.length)stepSelect.value=defaultStep();
 const candidates=events.filter(e=>e.domain==='Device'&&e.step===stepSelect.value&&e.layer!==null&&e.layer!==undefined&&($('#pool').value==='All'||$('#pool').value===e.pool_name)&&($('#rank').value==='All'||$('#rank').value===String(e.rank)));
 if(!candidates.length){const [start,end]=rangeDomain();setRange(start,end);return;}
 let first=Infinity,seed=null;
 for(const e of candidates)if(e.layer<first||(e.layer===first&&e.absolute_start_time_s<seed.absolute_start_time_s)){first=e.layer;seed=e;}
 const invocation=e=>/(?:^|:)mb-(\d+):(?:rank-\d+:)?step-(\d+):/.exec(e.call_id||''),unit=isFlashFp4?invocation(seed):null;
 if(isFlashFp4&&unit===null){$('#coverage').textContent='无法定位单次首层：保存的事件缺少 decode step / microbatch 标识。';return;}
 let start=Infinity,end=-Infinity;
 for(const e of candidates){if(e.layer!==first)continue;if(unit!==null){const other=invocation(e);if(other===null||other[1]!==unit[1]||other[2]!==unit[2])continue;}start=Math.min(start,e.absolute_start_time_s);end=Math.max(end,e.absolute_end_time_s);}
 const pad=Math.max((end-start)*.06,1e-12),focus=(unit===null?'':'Decode step '+unit[2]+' · MB '+unit[1]+' · ')+'Layer '+first;
 setRange(Math.max(0,start-pad),Math.min(total,end+pad),focus);
}
function laneType(e){if(e.aggregate_count!==undefined&&e.domain==='Device')return '聚合（各类别混合）';if(e.domain==='Host')return 'Host 提交';if(e.domain==='Wait')return '排队 / 等待';if(e.kind==='COMMUNICATION')return '通信';if(['MEMCPY','KV_CACHE','KV_OFFLOAD'].includes(e.kind))return 'KV 更新 / 搬运';return '计算';}
function laneKey(pool,rank,type){return `${pool} · rank ${rank} · ${type}`;}
function pathLabel(path){const stages=path.critical_path.map(x=>x.split('/').pop());return stages.slice(0,12).join(' → ')+(stages.length>12?' → …（完整路径见下方原始数据）':'');}
function diagnosticFor(call){const diag=d.prediction_diagnostics;if(!diag)return {reason:'旧投影未保留'};if(Object.prototype.hasOwnProperty.call(diag.by_call_id,call))return {path:diag.critical_paths[Number(diag.by_call_id[call])]};return {reason:Object.prototype.hasOwnProperty.call(diag.missing_by_call_id,call)?diag.missing_by_call_id[call]:'此操作未保留诊断'};}
function showDetail(e,view){
 let box=$('#detail-card');box.replaceChildren();$('#detail-diagnostics').replaceChildren();$('#operation-diagnostics').open=false;
 const add=(label,value)=>{const item=document.createElement('div'),title=document.createElement('span'),content=document.createElement('strong');item.className='detail-item';title.textContent=label;content.textContent=value;item.append(title,content);box.append(item);};
 add('操作名称',e.category+' · '+e.name);add('操作类型',operationBucket(e)[1]);if(e.layer!==null&&e.layer!==undefined)add('模型层','Layer '+e.layer);add('起点 → 终点',clock(e.absolute_start_time_s)+' → '+clock(e.absolute_end_time_s));add('预测持续时间',duration(e.absolute_end_time_s-e.absolute_start_time_s));add('实例 / rank',e.pool_name+' / '+e.rank);add('所属步骤 / 阶段',(e.step||'不可用')+' / '+(e.phase?(Array.isArray(e.phase)?e.phase:e.phase.split('+')).map(phaseName).join(' + '):'不可用'));
 if(e.shape)add('Shape / dtype',Object.entries(e.shape).map(([k,v])=>k+'='+v).join('，')+' / '+e.dtype);
 if(e.requests)add('同一批次请求',e.requests.map(r=>r.request_id+' ('+phaseName(r.phase)+')').join(', ')||'未提供');
 box=$('#detail-diagnostics');if(e.resource)add('调度资源',e.resource==='compute'?'设备计算资源（调度口径）':e.resource);
 if(e.handoff)add('KV 交接',e.handoff.source_pool_id+' → '+e.handoff.destination_pool_id+' · '+e.handoff.byte_count+' B');
 if(e.link_evidence){if(e.link_evidence.availability==='available')add('互联资源',e.link_evidence.link+' · 每端 '+e.link_evidence.wire_bytes_per_endpoint+' B / '+e.link_evidence.capacity_bytes_per_s.toExponential(3)+' B/s（算法字节）');else add('互联资源','不可用：'+e.link_evidence.reason);}
 const bound=e.resource_bounds;
 if(bound&&bound.status==='available'){
  const dominant=bound.dominant==='HBM'?'HBM 逻辑字节 / 带宽下界较大':bound.dominant==='compute'?roleName(bound.compute_role)+' 算力下界较大':'两项下界相等';
  add('工作量下界比较',dominant);add('下界数值','算力 '+duration(bound.compute_lower_bound_s)+' · 逻辑 HBM '+duration(bound.logical_hbm_lower_bound_s));
 }else if(e.domain==='Device')add('资源下界','不可用：'+(bound&&bound.reason?bound.reason:'聚合区间或未保留资源输入'));
 if(e.prediction){add('模型瓶颈组',modelGroup(e.prediction.bottleneck_group)[0]+'（原模型输出）');const diag=diagnosticFor(e.call_id);if(diag.path){add('算子模型关键路径',pathLabel(diag.path)+'（单算子内部模型路径，非请求级关键路径）');add('路径支持级别 / 限制',diag.path.support_level+' · '+(diag.path.limitations.join('；')||'模型未列出其他限制'));}else add('算子模型关键路径','不可用：'+diag.reason+'；资源服务时间不能直接等同关键路径');if(e.prediction.t_launch_s>0)add('模型启动 / 分派时间',duration(e.prediction.t_launch_s));const times=Object.entries(e.prediction.t_per_resource_s).filter(([k,v])=>v>0).sort((a,b)=>b[1]-a[1]);if(times.length)add('模型资源服务时间',times.slice(0,4).map(([k,v])=>roleName(k)+' '+duration(v)).join(' · ')+'；各资源时间不能直接相加为关键路径');}
 else if(e.domain==='Device')add('模型瓶颈字段','未保留；不可从最终时延反推');
 if(e.aggregate_count!==undefined)add('聚合范围',String(e.aggregate_count)+' 个事件；区间包含空隙，具体操作未保留');
 if(e.model_bottleneck_groups){const groups=Object.entries(e.model_bottleneck_groups).sort((a,b)=>b[1].predicted_operation_time_s-a[1].predicted_operation_time_s);add('模型瓶颈组（整步汇总）',groups.map(([name,v])=>modelGroup(name)[0]+' '+v.operation_count+' 次 / '+duration(v.predicted_operation_time_s)).join(' · ')+'；仅为累加，不提供组别顺序');const paths=[];let missing=0;for(const [,g] of groups){if(!g.critical_paths)continue;missing+=g.critical_path_unavailable_or_omitted;for(const [id,v] of Object.entries(g.critical_paths))paths.push([id,v]);}paths.sort((a,b)=>b[1].operation_count-a[1].operation_count);if(paths.length&&d.prediction_diagnostics)add('常见算子模型路径（整步）',paths.slice(0,3).map(([id,v])=>pathLabel(d.prediction_diagnostics.critical_paths[Number(id)])+'：'+v.operation_count+' 次').join('；')+'。缺失或超预算 '+missing+' 次；非时间顺序、非请求级关键路径');}
 if(view==='memory')add('图上这一行','逻辑字节需求按操作区间均匀分摊，非实际 HBM 事务');
 if(view==='bound')add('图上这一行','工作量/峰值下界的比较，不包含启动和同步成本，不能代表实际关键路径');if(view==='model')add('图上这一行','每个操作的预测器瓶颈组；不表示跨操作的端到端关键路径');if(view==='model_summary')add('图上这一行','整步各组预测操作时长的汇总；细边颜色标记累计最多的组，不能表示组别发生顺序、持续占用或端到端关键路径');
 const evidence=e.resource_evidence||(e.domain==='Device'?{availability:'unavailable',reason:'此投影未保留逐操作计算/HBM 数据'}:{availability:'not_applicable',reason:'Host/排队事件不提供设备资源工作量'});
 $('#detail').textContent=JSON.stringify({...e,model_diagnostic:e.prediction?diagnosticFor(e.call_id):null,duration_s:e.absolute_end_time_s-e.absolute_start_time_s,resource_evidence:evidence},null,2);
}
function drawInsight(selected,a,b){
 const byCategory=new Map(),byBound=new Map();let unknown=0,aggregated=0;
 for(const e of selected){if(e.domain!=='Device')continue;const overlap=Math.max(0,Math.min(b,e.absolute_end_time_s)-Math.max(a,e.absolute_start_time_s));if(overlap<=0)continue;
  if(e.aggregate_count!==undefined){aggregated++;continue;}
  byCategory.set(e.category,(byCategory.has(e.category)?byCategory.get(e.category):0)+overlap);
  if(e.resource_bounds&&e.resource_bounds.status==='available'){const key=e.resource_bounds.dominant;byBound.set(key,(byBound.has(key)?byBound.get(key):0)+overlap);}else unknown++;
 }
 const node=$('#window-insight');node.replaceChildren();const lead=document.createElement('p');lead.className='notice';
 const ranked=[...byCategory.entries()].sort((a,b)=>b[1]-a[1]);
 lead.textContent=ranked.length?'当前窗口已保留明细中，累计操作时长最多：'+ranked.slice(0,3).map(([name,t])=>name+' '+duration(t)).join(' · ')+'。这是各操作时长之和，重叠和多 rank 会重复计数，不能作为端到端关键路径。':'当前窗口没有保留具体操作明细；聚合条只表示范围，不能据此判定瓶颈。';node.append(lead);
 const note=document.createElement('p');note.textContent='下界比较中，HBM 较大的操作其预测时长累计 '+duration(byBound.has('HBM')?byBound.get('HBM'):0)+'；算力下界较大的操作累计 '+duration(byBound.has('compute')?byBound.get('compute'):0)+'。'+unknown+' 个操作没有可比较的资源证据，'+aggregated+' 个聚合区间不作逐操作判断；该比较不含启动、同步及物理内存事务。';node.append(note);
 const requestSteps=new Set(matchingSteps().map(step=>step.step_id));const totals=new Map();let summarized=0,missing=0;for(const row of (d.resource_summary||[])){if(requestSelect.value!=='All'&&!requestSteps.has(row.step_id))continue;if($('#pool').value!=='All'&&$('#pool').value!==row.pool_name)continue;if($('#rank').value!=='All'&&$('#rank').value!==String(row.rank))continue;if(stepSelect.value!=='All'&&stepSelect.value!==row.step_id)continue;if(row.end_time_s<=a||row.start_time_s>=b)continue;if(!row.model_bottleneck_groups){missing++;continue;}summarized++;for(const [name,item] of Object.entries(row.model_bottleneck_groups)){if(!totals.has(name))totals.set(name,{count:0,time:0});const total=totals.get(name);total.count+=item.operation_count;total.time+=item.predicted_operation_time_s;}}
 const model=document.createElement('p');const rankedGroups=[...totals.entries()].sort((a,b)=>b[1].time-a[1].time);model.textContent=rankedGroups.length?'相交步骤的模型瓶颈组（整步累计，含重叠）：'+rankedGroups.slice(0,5).map(([name,item])=>modelGroup(name)[0]+' '+item.count+' 次 / '+duration(item.time)).join(' · ')+'。整步数值不按当前缩放窗口裁剪，不受操作类型筛选影响，也不是端到端关键路径。按请求筛选时，只汇总该请求参与且保留明细的批次，仍包含同批次其他请求的工作。':'当前数据没有保留整步的模型瓶颈组；灰色条表示未知。';if(missing)model.textContent+=' '+missing+' 个相交步骤未记录该字段。';node.append(model);
}
function drawDiagnostics(){
 const cv=$('#diagnostic-timeline'),ctx=cv.getContext('2d');
 const [a,b]=bounds();if(!(b>a))return;cv.width=cv.clientWidth*devicePixelRatio;const w=cv.width;
 const selected=events.filter(e=>[['pool','pool_name'],['rank','rank'],['kind','category']].every(([id,key])=>$('#'+id).value==='All'||$('#'+id).value===String(e[key]))&&(stepSelect.value==='All'||stepSelect.value===e.step)&&(requestSelect.value==='All'||e.request_id===requestSelect.value||(e.requests&&e.requests.some(r=>r.request_id===requestSelect.value))));
 const groups=[...new Set(selected.map(e=>JSON.stringify([e.pool_name,e.rank])))].map(x=>JSON.parse(x)).sort((x,y)=>String(x[0]).localeCompare(String(y[0]))||String(x[1]).localeCompare(String(y[1]),undefined,{numeric:true}));
 const lanes=[];for(const [pool,rank] of groups)if(selected.some(e=>e.domain==='Device'&&e.pool_name===pool&&String(e.rank)===String(rank)))for(const type of ['HBM 逻辑需求','模型瓶颈组','下界较大项'])lanes.push(laneKey(pool,rank,type));
 const height=Math.max(150,lanes.length*31+67);cv.style.height=height+'px';cv.height=height*devicePixelRatio;ctx.font=`${12*devicePixelRatio}px system-ui`;ctx.clearRect(0,0,w,cv.height);diagnosticHits=[];const left=220*devicePixelRatio,scale=(w-left)/(b-a),rowHeight=31*devicePixelRatio,top=55*devicePixelRatio;
 lanes.forEach((l,n)=>{const y=top+n*rowHeight;ctx.fillStyle=n%2?'#f7fafb':'#fff';ctx.fillRect(0,y-3*devicePixelRatio,w,rowHeight);ctx.fillStyle='#415a68';ctx.fillText(l,8*devicePixelRatio,y+16*devicePixelRatio);});
 for(let i=0;i<=5;i++){const x=left+(w-left)*i/5;ctx.fillStyle='#687e89';ctx.fillText(clock(a+(b-a)*i/5),Math.min(x,w-75*devicePixelRatio),13*devicePixelRatio);ctx.strokeStyle='#e4ebef';ctx.beginPath();ctx.moveTo(x,top-7*devicePixelRatio);ctx.lineTo(x,cv.height);ctx.stroke();}
 if(stepSelect.value!=='All'){const layers=new Map();for(const e of selected){if(e.domain!=='Device'||e.layer===null||e.layer===undefined||e.absolute_end_time_s<=a||e.absolute_start_time_s>=b)continue;const k=e.layer;if(!layers.has(k))layers.set(k,[e.absolute_start_time_s,e.absolute_end_time_s]);else{const span=layers.get(k);span[0]=Math.min(span[0],e.absolute_start_time_s);span[1]=Math.max(span[1],e.absolute_end_time_s);}}for(const [layer,[start,end]] of layers){const x=left+(Math.max(a,start)-a)*scale,right=left+(Math.min(b,end)-a)*scale;if(right-x>32*devicePixelRatio){ctx.fillStyle='#536b79';ctx.fillText('Layer '+layer,x+3*devicePixelRatio,37*devicePixelRatio);}ctx.strokeStyle='#bccbd4';ctx.beginPath();ctx.moveTo(x,22*devicePixelRatio);ctx.lineTo(x,top-7*devicePixelRatio);ctx.stroke();}}
 function bar(e,type,color,view){const st=e.absolute_start_time_s,en=e.absolute_end_time_s;if(en<=a||st>=b)return;const index=lanes.indexOf(laneKey(e.pool_name,e.rank,type));if(index<0)return;const y=top+index*rowHeight,x=left+(Math.max(a,st)-a)*scale,ew=Math.max(1,(Math.min(b,en)-Math.max(a,st))*scale),eh=21*devicePixelRatio;ctx.fillStyle=view==='model_summary'?'#eaf0f2':color;ctx.fillRect(x,y,ew,eh);if(view==='model_summary'){ctx.fillStyle=color;ctx.fillRect(x,y,ew,3*devicePixelRatio);ctx.strokeStyle='#a7b8c1';ctx.setLineDash([3*devicePixelRatio,3*devicePixelRatio]);ctx.strokeRect(x,y,ew,eh);ctx.setLineDash([]);}diagnosticHits.push({x,y,w:ew,h:eh,e,view});if(ew>56*devicePixelRatio){ctx.save();ctx.beginPath();ctx.rect(x,y,ew,eh);ctx.clip();ctx.fillStyle=view==='model_summary'?'#415a68':'#fff';ctx.fillText(view==='model_summary'?'整步汇总':view==='bound'?(!e.resource_bounds||e.resource_bounds.status!=='available'?'不可用':e.resource_bounds.dominant==='HBM'?'HBM 下界':e.resource_bounds.dominant==='compute'?roleName(e.resource_bounds.compute_role):'相等'):view==='model'?modelGroup(e.prediction?e.prediction.bottleneck_group:null)[0]:e.category,x+3*devicePixelRatio,y+15*devicePixelRatio);ctx.restore();}}
 for(const e of selected){const kind=laneType(e),color=operationBucket(e)[2];if(e.domain==='Device'&&e.resource_evidence&&e.resource_evidence.availability==='available'&&e.resource_evidence.logical_bytes>0)bar(e,'HBM 逻辑需求','#69a8ba','memory');if(e.domain==='Device'&&e.aggregate_count===undefined)bar(e,'模型瓶颈组',modelGroup(e.prediction?e.prediction.bottleneck_group:null)[1],'model');if(e.domain==='Device'&&e.aggregate_count!==undefined&&e.model_bottleneck_groups){const groups=Object.entries(e.model_bottleneck_groups).sort((a,b)=>b[1].predicted_operation_time_s-a[1].predicted_operation_time_s);const winner=groups.length&&!(groups.length>1&&groups[0][1].predicted_operation_time_s===groups[1][1].predicted_operation_time_s)?groups[0][0]:null;bar(e,'模型瓶颈组',modelGroup(winner)[1],'model_summary');}if(e.domain==='Device'&&e.aggregate_count===undefined)bar(e,'下界较大项',!e.resource_bounds||e.resource_bounds.status!=='available'?'#a9b7bd':e.resource_bounds.dominant==='HBM'?'#487eac':e.resource_bounds.dominant==='compute'?'#ca8752':'#84969e','bound');}
 drawInsight(selected,a,b);
 drawRouting();drawResourceTable(a,b,selected);drawHistory(a,b);
$('#resources').textContent=JSON.stringify({window_s:[a,b],links:linkStats(a,b),capacity_budget:d.capacity,hardware_capacity:d.hardware,schedule_queues:d.schedule,lanes:resourceSummary(selected,a,b),resource_scope:d.resource_summary?'All operations, selected instances and ranks; step demand prorated, busy union exact':'Selected operation detail',coverage:d.coverage,known_activation_memory:d.source_activation_memory,routing:d.routing},null,2);
$('#coverage').textContent=`窗口 ${clock(a)} – ${clock(b)}；保留的具体操作 ${d.coverage?d.coverage.detail_events:events.length} / ${d.coverage?d.coverage.total_events:events.length} 个；具体明细外包络（非连续保证） ${d.coverage&&d.coverage.detail_interval_s?d.coverage.detail_interval_s.map(clock).join(' – '):'完整区间'}；其余 ${d.coverage?d.coverage.omitted_detail_events:0} 个事件的细节未保留。聚合条只表示范围，包含空隙；若需更多逐操作细节，需要提高记录上限并重新仿真。`;if(d.coverage&&d.coverage.detailed_requests!==undefined)$('#coverage').textContent+=` 明细覆盖 ${d.coverage.detailed_requests} / ${d.coverage.total_requests} 个请求、${d.coverage.detailed_steps} / ${d.coverage.total_steps} 个模型批次，分布在 ${d.coverage.detail_intervals_s.length} 段采样区间。请求被覆盖表示至少一个参与批次有明细，不代表其全部生成过程均有明细。`;
}
let detailHits=[],diagnosticHits=[],cardLabelHits=[];
function filteredEvents(){return events.filter(e=>[['pool','pool_name'],['rank','rank'],['kind','category']].every(([id,key])=>$('#'+id).value==='All'||$('#'+id).value===String(e[key]))&&(stepSelect.value==='All'||stepSelect.value===e.step)&&(requestSelect.value==='All'||e.request_id===requestSelect.value||(e.requests&&e.requests.some(r=>r.request_id===requestSelect.value))));}
function physicalCards(selected){if(isFlashFp4&&$('#pool').value==='All'&&$('#rank').value==='All')return Array.from({length:scenario.scheme==='dp8_fpga'?24:8},(_,i)=>i<8?'GPU '+i:'FPGA '+(i-8));return [...new Set(selected.filter(e=>e.domain==='Device').map(physicalCard).filter(Boolean))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
function paintTimeline(canvas,rows,items,rowFor,overview=false){const [a,b]=bounds(),ratio=devicePixelRatio,ctx=canvas.getContext('2d'),rowHeight=(overview?34:27)*ratio,top=35*ratio,left=205*ratio;canvas.width=Math.max(400,canvas.clientWidth)*ratio;canvas.height=(45+rows.length*(overview?34:27))*ratio;canvas.style.height=canvas.height/ratio+'px';const w=canvas.width,scale=(w-left)/(b-a),result=[];ctx.font=`${11*ratio}px system-ui`;ctx.clearRect(0,0,w,canvas.height);rows.forEach((row,i)=>{const y=top+i*rowHeight;ctx.fillStyle=i%2?'#f4f7fa':'#fff';ctx.fillRect(0,y-2*ratio,w,rowHeight);ctx.fillStyle='#30485b';ctx.fillText(row,8*ratio,y+14*ratio);if(overview)cardLabelHits.push({card:row,y,h:rowHeight});});for(let i=0;i<=5;i++){const x=left+(w-left)*i/5;ctx.fillStyle='#607483';ctx.fillText(clock(a+(b-a)*i/5),Math.min(x,w-85*ratio),17*ratio);ctx.strokeStyle='#e1e8ed';ctx.beginPath();ctx.moveTo(x,top-5*ratio);ctx.lineTo(x,canvas.height);ctx.stroke();}const indices=new Map(rows.map((row,i)=>[row,i]));for(const e of items){const row=rowFor(e);if(!indices.has(row)||e.absolute_end_time_s<=a||e.absolute_start_time_s>=b)continue;const x=left+(Math.max(a,e.absolute_start_time_s)-a)*scale,ew=Math.max(1,(Math.min(b,e.absolute_end_time_s)-Math.max(a,e.absolute_start_time_s))*scale);let offset=0,eh=18*ratio;if(overview){const channel=e.kind==='COMMUNICATION'?1:e.kind==='MEMCPY'?2:0;if(isFlashFp4&&e.pool_name==='experts'){const chip=(Number(e.rank)-8)%2;offset=(chip*15+[0,7,11][channel])*ratio;eh=[6,3,3][channel]*ratio;}else{offset=[0,13,19][channel]*ratio;eh=[12,5,4][channel]*ratio;}}const y=top+indices.get(row)*rowHeight+offset;ctx.fillStyle=operationBucket(e)[2];ctx.fillRect(x,y,ew,eh);result.push({x,y,w:ew,h:eh,e,view:'operation'});if(!overview&&ew>90*ratio){ctx.save();ctx.beginPath();ctx.rect(x,y,ew,eh);ctx.clip();ctx.fillStyle='#fff';ctx.fillText(e.category,x+3*ratio,y+13*ratio);ctx.restore();}}return result;}
function drawBucketChart(){const saved=d.experiment_configuration?.timeline_buckets,select=$('#bucket-channel');for(const option of select.options)option.disabled=option.value!=='cumulative_buckets'&&!(saved&&Array.isArray(saved[option.value]));if(select.selectedOptions[0].disabled)select.value='cumulative_buckets';let buckets=saved&&Array.isArray(saved[select.value])?saved[select.value]:null;let span=d.coverage?.detail_interval_s||[0,total];if(saved)span=[saved.window_start_s,saved.window_end_s];if(!buckets){const sums=new Map(BUCKETS.map(x=>[x[0],0])),seen=new Set();for(const e of events){if(e.domain!=='Device'||e.aggregate_count!==undefined)continue;const key=JSON.stringify([e.shared_call_id||e.call_id,e.absolute_start_time_s,e.absolute_end_time_s]);if(seen.has(key))continue;seen.add(key);const elapsed=Math.max(0,Math.min(span[1],e.absolute_end_time_s)-Math.max(span[0],e.absolute_start_time_s));sums.set(operationBucket(e)[0],sums.get(operationBucket(e)[0])+elapsed);}const sum=[...sums.values()].reduce((a,b)=>a+b,0);buckets=BUCKETS.map(([key,label,color])=>({key,label,color,duration_s:sums.get(key),fraction:sum?sums.get(key)/sum:0}));}$('#bucket-scope').textContent='保存窗口 '+clock(span[0])+' – '+clock(span[1])+'；'+(select.value==='cumulative_buckets'?'跨设备累计算子时长，重叠活动可重复累计，不是wall-time分解。':'通道累计时长包含空闲；占比以GPU的8×窗口、FPGA各通道的32×窗口归一化，累计时长不是wall-time。')+' 饼图统计固定保存窗口，不随上方缩放改变。';const box=$('#bucket-chart');box.replaceChildren();const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 260 260');svg.setAttribute('role','img');svg.setAttribute('aria-label','时间分桶饼图');let angle=-Math.PI/2;const positive=buckets.filter(row=>row.fraction>0);for(const row of positive){const end=angle+row.fraction*Math.PI*2,path=document.createElementNS(ns,'path');const point=t=>[130+110*Math.cos(t),130+110*Math.sin(t)];let command;if(row.fraction>=1-1e-12)command='M130 20 A110 110 0 1 1 129.999 20 Z';else{const p=point(angle),q=point(end);command=`M130 130 L${p[0]} ${p[1]} A110 110 0 ${end-angle>Math.PI?1:0} 1 ${q[0]} ${q[1]} Z`;}path.setAttribute('d',command);path.setAttribute('fill',row.color);const title=document.createElementNS(ns,'title');title.textContent=row.label+' '+(row.fraction*100).toFixed(2)+'% · '+duration(row.duration_s);path.append(title);svg.append(path);angle=end;}if(!positive.length){const t=document.createElementNS(ns,'text');t.setAttribute('x','40');t.setAttribute('y','130');t.textContent='此保存窗口没有操作明细';svg.append(t);}box.append(svg);const table=document.createElement('table');table.className='scroll-table';table.innerHTML='<thead><tr><th>类型</th><th>'+(select.value==='cumulative_buckets'?'累计操作时长':'通道累计时长')+'</th><th>占比</th></tr></thead>';const body=document.createElement('tbody');for(const row of buckets){const tr=document.createElement('tr');for(const value of [row.label,duration(row.duration_s),(row.fraction*100).toFixed(2)+'%']){const td=document.createElement('td');td.textContent=value;tr.append(td);}tr.firstChild.style.borderLeft='5px solid '+row.color;body.append(tr);}table.append(body);box.append(table);}
function draw(){const [a,b]=bounds();if(!(b>a))return;const selected=filteredEvents(),cards=physicalCards(selected);cv.dataset.cardCount=String(cards.length);cv.dataset.cardLabels=JSON.stringify(cards);cv.setAttribute('role','img');cv.setAttribute('aria-label','Physical card timeline: '+cards.join(', '));cardLabelHits=[];hits=paintTimeline(cv,cards,selected.filter(e=>e.domain==='Device'),physicalCard,true);const choice=$('#detail-card-select'),before=choice.value;if(choice.options.length!==cards.length+1||[...choice.options].slice(1).some((option,i)=>option.value!==cards[i])){choice.replaceChildren();for(const value of ['All',...cards]){const option=document.createElement('option');option.value=value;option.textContent=value==='All'?'全部卡':value;choice.append(option);}if(cards.includes(before))choice.value=before;}if($('#detail-lanes').open){const detail=selected.filter(e=>choice.value==='All'||physicalCard(e)===choice.value),rowFor=e=>physicalCard(e)+(isFlashFp4&&e.pool_name==='experts'?' · chip '+((Number(e.rank)-8)%2):'')+' · '+laneType(e),rows=[...new Set(detail.map(rowFor))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));detailHits=paintTimeline($('#detail-timeline'),rows,detail,rowFor);}if($('#bottleneck-diagnostics').open)drawDiagnostics();$('#coverage').textContent='当前窗口 '+clock(a)+' – '+clock(b)+' · '+cards.length+' 张物理卡；明细范围 '+(d.coverage?.detail_interval_s?d.coverage.detail_interval_s.map(clock).join(' – '):'完整记录')+'。'+(layerFocus===null?'':' 单次首层：'+layerFocus+'；其他并发活动仍按实际时间显示。');drawBucketChart();}

function openCardDetails(card){if(card!==null)$('#detail-card-select').value=card;$('#detail-lanes').open=true;draw();$('#device-detail-section').scrollIntoView({behavior:'smooth',block:'start'});}
$('#show-breakdown').onclick=()=>openCardDetails(null);
cv.onclick=ev=>{if(ev.offsetX<205){const label=cardLabelHits.find(row=>ev.offsetY*devicePixelRatio>=row.y&&ev.offsetY*devicePixelRatio<row.y+row.h);if(label)openCardDetails(label.card);return;}if(drag&&Math.abs(ev.clientX-drag.x)>3)return;const r=cv.getBoundingClientRect(),x=(ev.clientX-r.left)*devicePixelRatio,y=(ev.clientY-r.top)*devicePixelRatio;const h=hits.findLast(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);if(h)showDetail(h.e,h.view);};
cv.onmousemove=ev=>{if(drag)return;const r=cv.getBoundingClientRect(),x=(ev.clientX-r.left)*devicePixelRatio,y=(ev.clientY-r.top)*devicePixelRatio,h=hits.findLast(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);cv.title=h?(h.e.category+' · '+h.e.name+' · '+duration(h.e.absolute_end_time_s-h.e.absolute_start_time_s)+'；点击查看依据'):'';};
cv.onwheel=e=>{e.preventDefault();const [a,b]=bounds(),r=cv.getBoundingClientRect(),plot=Math.max(1,r.width-220),d=Math.abs(e.deltaX)>Math.abs(e.deltaY)?e.deltaX:e.deltaY;if(!d)return;if(!e.altKey){const pixels=d*(e.deltaMode===1?16:e.deltaMode===2?plot:1),dt=pixels/plot*(b-a);setRange(a+dt,b+dt);return;}const f=Math.max(0,Math.min(1,(e.clientX-r.left-220)/plot)),t=a+(b-a)*f,width=(b-a)*(d>0?1.5:1/1.5);setRange(t-width*f,t+width*(1-f));};
let drag=null;cv.onpointerdown=e=>{if(e.button!==0)return;drag={x:e.clientX,range:bounds()};cv.setPointerCapture(e.pointerId);};cv.onpointermove=e=>{if(drag){const [a,b]=drag.range,delta=(drag.x-e.clientX)*(b-a)/Math.max(1,cv.clientWidth-220);setRange(a+delta,b+delta);}};cv.onpointerup=()=>{setTimeout(()=>drag=null,0);};cv.onpointercancel=()=>drag=null;
$('#first-layer').onclick=firstLayer;$('#zoom').onclick=()=>{const [a,b]=bounds();setRange(a,a+(b-a)/2);};$('#reset').onclick=()=>{requestSelect.value='All';fillSteps();stepSelect.value='All';const span=d.coverage?.detail_interval_s||[0,total];setRange(span[0],span[1]);};$('#range').oninput=e=>{const [a,b]=bounds(),[low,high]=rangeDomain(),start=low+(high-low-(b-a))*e.target.value/1000;setRange(start,start+b-a);};$('#resource-metric').onchange=draw;$('#start').onchange=()=>{const [a,b]=bounds();if(b>a)setRange(a,b);};$('#end').onchange=()=>{const [a,b]=bounds();if(b>a)setRange(a,b);};window.onresize=draw;$('#detail-lanes').ontoggle=draw;$('#bottleneck-diagnostics').ontoggle=draw;$('#detail-card-select').onchange=draw;$('#bucket-channel').onchange=drawBucketChart;for(const [selector,readHits] of [['#detail-timeline',()=>detailHits],['#diagnostic-timeline',()=>diagnosticHits]]){const canvas=$(selector);canvas.onclick=event=>{if(drag&&Math.abs(event.clientX-drag.x)>3)return;const r=canvas.getBoundingClientRect(),x=(event.clientX-r.left)*devicePixelRatio,y=(event.clientY-r.top)*devicePixelRatio,hit=readHits().findLast(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);if(hit)showDetail(hit.e,hit.view);};canvas.onmousemove=event=>{const r=canvas.getBoundingClientRect(),x=(event.clientX-r.left)*devicePixelRatio,y=(event.clientY-r.top)*devicePixelRatio,hit=readHits().findLast(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);canvas.title=hit?hit.e.category+' · '+duration(hit.e.absolute_end_time_s-hit.e.absolute_start_time_s):'';};canvas.onwheel=cv.onwheel;canvas.onpointerdown=event=>{if(event.button!==0)return;drag={x:event.clientX,range:bounds()};canvas.setPointerCapture(event.pointerId);};canvas.onpointermove=event=>{if(drag){const [a,b]=drag.range,delta=(drag.x-event.clientX)*(b-a)/Math.max(1,canvas.clientWidth-205);setRange(a+delta,b+delta);}};canvas.onpointerup=()=>setTimeout(()=>drag=null,0);canvas.onpointercancel=()=>drag=null;}const initial=d.coverage?.detail_interval_s;if(initial&&initial[1]>initial[0]){stepSelect.value='All';setRange(initial[0],initial[1]);}else if(d.steps.length)firstLayer();else draw();
</script><script>
'use strict';
async function loadReport(){
 const status=document.querySelector('#load-status');
 const controls=[...document.querySelectorAll('button,input,select')];controls.forEach(x=>x.disabled=true);
 try{
  if(typeof DecompressionStream!=='function')throw new Error('此浏览器不支持自动解压，请使用较新版本的 Chrome、Edge、Firefox 或 Safari');
  const source=document.querySelector('#data');
  const bytes=Uint8Array.from(atob(source.textContent.trim()),c=>c.charCodeAt(0));source.remove();
  const text=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  const data=JSON.parse(text),tables=data.visualization_storage;
  if(tables&&tables.version!==1)throw new Error('不支持的报告存储版本');
  const idMaps=new Set(['resources','visualization_resources','by_call_id','missing_by_call_id']);
  function ref(name,index){if(!Number.isInteger(index)||index<0||index>=tables[name].length)throw new Error('报告包含无效的表引用');return tables[name][index];}
  function expand(value,context=''){
   if(Array.isArray(value))return value.map(x=>expand(x));
   if(value===null||typeof value!=='object')return value;
   if(idMaps.has(context))return Object.fromEntries(Object.entries(value).map(([key,item])=>[ref('call_ids',Number(key)),expand(item)]));
   const result={};for(const [key,item] of Object.entries(value)){
    if(key==='visualization_storage')continue;
    if(key==='placement_ref')Object.assign(result,ref('placements',item));
    else if(key==='kernel_spec_ref')result.kernel_spec=ref('kernel_specs',item);
    else if(key==='call_id'&&item!==null)result[key]=ref('call_ids',item);
    else result[key]=expand(item,key);
   }return result;
  }
  window.midlayerReportData=tables?expand(data):data;
  const code=document.querySelector('#viewer-code'),script=document.createElement('script');script.textContent=code.textContent;code.remove();
  let viewerError=null;const onError=event=>{viewerError=event.error||new Error(event.message);};window.addEventListener('error',onError);
  try{document.body.append(script);}finally{window.removeEventListener('error',onError);}
  if(viewerError)throw viewerError;
  controls.forEach(x=>x.disabled=false);status.textContent='';window.midlayerReportReady=true;
 }catch(error){status.textContent='报告加载失败：'+error.message;status.style.color='#a13232';window.midlayerReportError=String(error);}
}
loadReport();
</script></html>'''


def collect_resource_evidence(result, hardware_by_id: dict) -> dict:
    """Read production demand contracts once per distinct immutable spec."""
    from src.kernel.resource_usage import kernel_resource_usage

    by_spec = {}
    by_call = {}
    for step in (*result.steps, *result.transfers):
        for call in step.kernel_plan.calls:
            spec = call.kernel_spec
            # KernelSpec contains mappings; its identity is not a hash contract.
            # Only this bounded in-process cache uses repr, never provenance.
            key = repr(spec)
            if key not in by_spec:
                by_spec[key] = kernel_resource_usage(spec, hardware_by_id[spec.hardware_id])
            by_call[call.call_id] = by_spec[key]
    return by_call



def default_report_path(result_output, explicit_report=None):
    """Derive one run's destination without mutating reusable CLI arguments."""
    if explicit_report is not None:
        path = Path(explicit_report)
        if result_output is not None and path.resolve() == Path(result_output).resolve():
            raise ValueError("report and result output paths must differ")
        return path
    if result_output is None:
        return Path("experiment_logs") / "serving_reports" / str(time.time_ns()) / "report.html"
    path = Path(result_output)
    return (path.with_name(path.stem + ".report.html") if path.suffix == ".html"
        else path.with_suffix(".html"))

def write_status_report(payload: dict, output: str | Path) -> dict:
    """Write a truthful run overview when no completed Serving timeline exists."""
    import html
    from evaluation.reporting.serving_viewer_ui import _configuration_metadata
    started = time.perf_counter()
    status = payload["main_chain_status"]
    if not isinstance(status["status"], str) or not status["status"]:
        raise ValueError("run status must be a nonempty recorded string")
    from evaluation.reporting.serving_viewer_ui import report_run_context
    configuration = report_experiment_configuration(payload)
    costs = report_run_context(payload, {})["report_costs"]
    safe = {"visualization_view": "status", "main_chain_status": _configuration_metadata(status),
        "visualization_experiment_configuration": configuration,
        **({"visualization_run_costs": costs} if costs is not None else {})}
    links = payload["report_links"] if "report_links" in payload else []
    safe["report_links"] = [{"label": str(link["label"]), "path": str(Path(link["path"]).resolve())} for link in links]
    import base64
    link_parts = []
    chart_html = []
    for link in safe["report_links"]:
        label = html.escape(link["label"])
        target = Path(link["path"])
        if target.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg") and target.is_file():
            mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}[target.suffix.lower()]
            encoded = base64.b64encode(target.read_bytes()).decode("ascii")
            chart_html.append('<figure style="margin:14px 0"><img alt="' + label +
                '" src="data:' + mime_type + ';base64,' + encoded +
                '" style="max-width:100%;border:1px solid #dce4ed;border-radius:8px"><figcaption>' + label + '</figcaption></figure>')
        link_parts.append('<p><a href="' + html.escape(target.as_uri(), quote=True) + '">' + label + '</a></p>')
    link_html = "".join(link_parts)
    chart_block = "".join(chart_html)
    facts = payload["report_facts"] if "report_facts" in payload else []
    safe["report_facts"] = _configuration_metadata(facts)
    fact_rows = []
    deferred_groups = []
    for fact in safe["report_facts"]:
        value = fact["value"]
        label = fact["label"]
        if isinstance(value, list) and value and all(isinstance(item, dict) and "label" in item and "value" in item for item in value):
            # Nested per-case/per-bucket groups render as a searchable,
            # collapsible list so the first screen stays on the summary.
            rows = "".join('<tr data-case-row data-search="' + html.escape(str(item["label"]).lower()) + '"><th style="text-align:left;padding:6px 16px 6px 0">' + html.escape(str(item["label"])) +
                '</th><td>' + html.escape(str(item["value"])) + '</td></tr>' for item in value)
            rows = '<tbody data-search-group>' + rows + '</tbody>'
            deferred_groups.append(
                '<details><summary>' + html.escape(label) + '（' + str(len(value)) + '）</summary>'
                '<input type="search" placeholder="按 case ID 筛选" oninput="const q=this.value.trim().toLowerCase();'
                'for(const tr of this.parentElement.querySelectorAll(\'[data-case-row]\'))tr.hidden=!!q&&!tr.dataset.search.includes(q);">'
                '<table>' + rows + '</table></details>')
            continue
        text = "不可用" if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if type(value) is float and ("（秒）" in label or "(seconds)" in label):
            text = f"{value * 1000:.3f}"
            label = label.replace("（秒）", "（毫秒）").replace("(seconds)", "(ms)")
        fact_rows.append('<tr><th style="text-align:left;padding:6px 16px 6px 0">' + html.escape(label) +
            '</th><td>' + html.escape(text) + '</td></tr>')
    summary_html = '<table>' + ''.join(fact_rows) + '</table>' if fact_rows else ''
    fact_html = ''.join(deferred_groups)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_path = output.with_suffix(".visualization.json")
    data_path.write_text(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
    encoded = json.dumps(configuration, ensure_ascii=True).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Experiment status</title>
<style>body{margin:24px;background:#f3f6fa;color:#24364b;font:14px system-ui}.page{max-width:1200px;margin:auto}.card{background:white;padding:20px;border:1px solid #dce4ed;border-radius:10px;margin:16px 0}h1{font-size:23px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<div class="page"><h1>实验状态与配置</h1><div class="card"><h2>__STATUS__</h2><p>__REASON__</p>__SUMMARY____CHARTS__<details><summary>Case 明细（点击展开与搜索）</summary>__FACTS__</details><p>本次没有可用的完整时间线；没有生成活动区间或利用率数值。</p>__LINKS__</div><details class="card"><summary>运行状态与成本</summary><p id="status-cost-summary">未记录</p><pre id="status-costs" data-no-translate></pre></details></div>
<script>window.renderExperimentConfiguration(__CONFIG__);
window.addEventListener('DOMContentLoaded',()=>{const recorded=document.querySelector('#recorded-report-costs'),costs=recorded?JSON.parse(recorded.textContent):__COSTS__;if(costs){document.querySelector('#status-costs').textContent=JSON.stringify(costs,null,2);if(costs.total_wall_through_result_write_s!=null)document.querySelector('#status-cost-summary').textContent='总实验耗时（截至结果写入）'+': '+costs.total_wall_through_result_write_s.toFixed(3)+' s';}});
</script></html>'''
    page = page.replace("__STATUS__", html.escape({"complete": "完成", "partial": "部分完成", "unsupported": "不支持", "failed": "失败"}[status["status"]] if status["status"] in {"complete", "partial", "unsupported", "failed"} else status["status"])).replace("__REASON__",
        html.escape(status["reason"])).replace("__CONFIG__", encoded).replace("__LINKS__", link_html).replace("__CHARTS__", chart_block).replace("__FACTS__", fact_html).replace("__SUMMARY__", summary_html).replace("__COSTS__", json.dumps(costs, allow_nan=False).replace("<", "\\u003c"))
    output.write_text(with_serving_viewer_ui(page))
    return {"status": status["status"], "path": str(output), "data_path": str(data_path),
        "html_bytes": output.stat().st_size, "data_bytes": data_path.stat().st_size,
        "page_generation_s": time.perf_counter() - started}


def regenerate_report(payload, output):
    """Dispatch the existing saved report views; old projections are breakdown."""
    view = payload["visualization_view"] if "visualization_view" in payload else "breakdown"
    if view == "swimlanes":
        from evaluation.reporting.serving_swimlanes import write_swimlane_report
        return write_swimlane_report(payload, output)
    if view == "status":
        return write_status_report(payload, output)
    if view == "breakdown":
        return write_interactive_report(payload, output)
    raise ValueError("unknown saved Serving visualization view")
