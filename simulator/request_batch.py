"""Atomic workload editing: normalize input requests, never run a simulation."""

import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import io
import json
import random
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from network_sim.model import Contract
from .config import Migration, Request, SimulationConfig

MAX_REQUESTS = 100_000
MAX_CONTENT_BYTES = 8_000_000
MAX_SCENE_BYTES = 9_900_000
ALIASES = {
    "request_id": "id",
    "arrival_time_s": "arrival_s",
    "ingress_id": "user_device",
    "cell_id": "cell",
    "prompt_tokens": "input_tokens",
    "请求id": "id",
    "到达时间(s)": "arrival_s",
    "地面接入": "user_device",
    "输入token": "input_tokens",
    "输出token": "output_tokens",
    "目标cell": "cell",
}
INTEGER_FIELDS = {
    "arrival_ns",
    "input_tokens",
    "output_tokens",
    "input_bytes",
    "output_bytes_per_token",
    "prefix_hit_tokens",
}


class Generation(Contract):
    count: Annotated[int, Field(strict=True, ge=1, le=MAX_REQUESTS)] = 1000
    window_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60
    start_s: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    distribution: Literal["uniform", "spaced", "simultaneous"] = "uniform"
    seed: Annotated[int, Field(strict=True, ge=0)] = 42
    input_tokens: Annotated[int, Field(strict=True, ge=1)] = 32
    output_tokens: Annotated[int, Field(strict=True, ge=1)] = 4
    user_device: str | None = None
    cell: str | None = None


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"字段重复：{key}")
        result[key] = value
    return result


def _json(text):
    def constant(value):
        raise ValueError(f"不允许非有限数值：{value}")

    return json.loads(text, object_pairs_hook=_pairs, parse_constant=constant)


def parse_rows(content, format):
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise ValueError("请求文件须为 UTF-8 文本且不超过 8 MB")
    content = content.lstrip("\ufeff")
    if not content.strip():
        raise ValueError("请求内容为空")
    if format == "csv":
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        try:
            header = next(reader)
            if len(set(header)) != len(header):
                raise ValueError("CSV 表头存在重复列")
            rows = []
            for row in reader:
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != len(header):
                    raise ValueError(f"CSV 第 {reader.line_num} 行列数与表头不一致")
                rows.append((reader.line_num, dict(zip(header, row))))
                if len(rows) > MAX_REQUESTS:
                    raise ValueError(f"一次最多导入 {MAX_REQUESTS} 条请求")
        except (csv.Error, StopIteration) as error:
            raise ValueError(f"CSV 格式错误：{error}") from error
    elif format == "json":
        value = _json(content)
        if isinstance(value, dict) and set(value) == {"requests"}:
            value = value["requests"]
        if not isinstance(value, list):
            raise ValueError("JSON 应为请求数组或包含 requests 数组的对象")
        rows = list(enumerate(value, 1))
    elif format == "jsonl":
        rows = [
            (i, _json(line))
            for i, line in enumerate(content.splitlines(), 1)
            if line.strip()
        ]
    else:
        raise ValueError("请选择 CSV、JSON 或 JSONL 格式")
    if not rows or len(rows) > MAX_REQUESTS:
        raise ValueError(f"请求数量须在 1–{MAX_REQUESTS} 之间")
    return rows


def ground_devices(config):
    owners = set(config.ground_owners)
    return {
        row["id"]: row["satellite"]
        for row in config.physical["devices"]
        if row["satellite"] in owners and row["kind"] == "terminal"
    }


def resolve_ground(value, devices):
    if value in devices:
        return value
    matches = [identity for identity, owner in devices.items() if owner == value]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"地面接入节点不存在或不唯一：{value}")


def _integer(value):
    if isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation as error:
            raise ValueError("必须是整数") from error
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("必须是有限整数")
        return int(number)
    return value


def seconds_ns(value):
    if isinstance(value, bool):
        raise ValueError("到达时间必须是非负有限数值")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("到达时间必须是数值") from error
    if not number.is_finite() or number < 0:
        raise ValueError("到达时间必须是非负有限数值")
    return int((number * 10**9).to_integral_value(rounding=ROUND_CEILING))


def normalize_request(row, devices):
    if not isinstance(row, dict):
        raise ValueError("请求必须是对象")
    fields = {}
    allowed = set(Request.model_fields) | {"arrival_s"}
    for key, value in row.items():
        key = key.strip().lower()
        key = ALIASES.get(key, key)
        if key not in allowed or key in fields:
            raise ValueError(f"未知或重复字段：{key}")
        fields[key] = value.strip() if isinstance(value, str) else value
    for key in tuple(fields):
        if fields[key] == "" and key not in {"id", "user_device", "input_tokens"}:
            del fields[key]
    if "arrival_s" in fields:
        if "arrival_ns" in fields:
            raise ValueError("arrival_s 与 arrival_ns 只能填写一种")
        fields["arrival_ns"] = seconds_ns(fields.pop("arrival_s"))
    for key in INTEGER_FIELDS & fields.keys():
        fields[key] = _integer(fields[key])
    if "user_device" in fields:
        fields["user_device"] = resolve_ground(fields["user_device"], devices)
    if "input_bytes" not in fields and type(fields.get("input_tokens")) is int:
        fields["input_bytes"] = fields["input_tokens"] * 4
    return Request.model_validate(fields)


def validate_requests(config, requests):
    if len(requests) > MAX_REQUESTS:
        raise ValueError(f"业务请求最多 {MAX_REQUESTS} 条")
    devices = ground_devices(config)
    seen, histories = set(), {}
    by_cell = {cell.id: cell for cell in config.cells}
    for request in requests:
        if request.id in seen:
            raise ValueError(f"请求 ID 重复：{request.id}")
        seen.add(request.id)
        if request.user_device not in devices:
            raise ValueError(f"请求 {request.id} 的地面接入节点不在当前物理场景中")
        if request.arrival_ns >= config.horizon_ns:
            raise ValueError(f"请求 {request.id} 的到达时间必须在仿真时域内")
        if request.cell is not None and request.cell not in by_cell:
            raise ValueError(f"请求 {request.id} 的目标 Cell 不存在")
        targets = [by_cell[request.cell]] if request.cell else config.cells
        if any(
            request.input_tokens + request.output_tokens > cell.max_model_len
            for cell in targets
        ):
            raise ValueError(
                f"请求 {request.id} 的输入与输出 token 总数超过目标 Cell 上下文上限"
            )
        if request.prefix_hit_tokens and not request.session_id:
            raise ValueError(f"请求 {request.id} 的前缀复用需要 session_id")
    for request in sorted(requests, key=lambda row: row.arrival_ns):
        if request.session_id:
            identity = request.session_id, request.context_id or request.id
            if identity in histories:
                raise ValueError(f"请求 {request.id} 的会话 context_id 重复")
            if request.parent_context_id:
                parent = histories.get((request.session_id, request.parent_context_id))
                if (
                    parent is None
                    or request.input_tokens < parent.input_tokens + parent.output_tokens
                ):
                    raise ValueError(f"请求 {request.id} 必须引用更早的完整会话历史")
            histories[identity] = request
    updated = SimulationConfig.model_validate(
        {**config.model_dump(), "requests": [row.model_dump() for row in requests]}
    )
    by_id = {row.id: row for row in updated.requests}
    for migration in updated.migrations:
        trigger = by_id[migration.after_request]
        if (trigger.cell, trigger.session_id, trigger.context_id or trigger.id) != (
            migration.source_cell,
            migration.session_id,
            migration.context_id,
        ):
            raise ValueError(f"KV 迁移 {migration.id} 与替换后的触发请求不一致")
    return updated.requests


def generate_requests(config, options, existing_ids=()):
    options = Generation.model_validate(options)
    start = seconds_ns(options.start_s)
    window = seconds_ns(options.window_s)
    if start >= config.horizon_ns or (
        options.distribution != "simultaneous" and start + window > config.horizon_ns
    ):
        raise ValueError("生成请求的时间窗口超出仿真时域")
    devices = ground_devices(config)
    endpoints = (
        [resolve_ground(options.user_device, devices)]
        if options.user_device
        else sorted(devices)
    )
    if not endpoints:
        raise ValueError("当前场景没有地面接入节点")
    rng = random.Random(options.seed)
    offsets = (
        sorted(rng.randrange(window) for _ in range(options.count))
        if options.distribution == "uniform"
        else [i * window // options.count for i in range(options.count)]
        if options.distribution == "spaced"
        else [0] * options.count
    )
    used, sequence, rows = set(existing_ids), 1, []
    for i, offset in enumerate(offsets):
        while f"request-{sequence}" in used:
            sequence += 1
        identity = f"request-{sequence}"
        used.add(identity)
        rows.append(
            Request(
                id=identity,
                arrival_ns=start + offset,
                user_device=endpoints[i % len(endpoints)],
                input_tokens=options.input_tokens,
                output_tokens=options.output_tokens,
                input_bytes=options.input_tokens * 4,
                cell=options.cell,
            )
        )
    return rows


def prepare_batch(value):
    if not isinstance(value, dict) or set(value) - {
        "experiment",
        "kind",
        "format",
        "content",
        "options",
        "mode",
    }:
        raise ValueError("批量请求参数包含未知字段")
    mode = value.get("mode", "append")
    if mode not in {"append", "replace"}:
        raise ValueError("导入方式须为追加或替换")
    original = value["experiment"]
    config = SimulationConfig.model_validate(
        {**original, "requests": [], "migrations": []}
    )
    config = config.model_copy(
        update={
            "requests": tuple(
                Request.model_validate(row) for row in original.get("requests", [])
            )
            if mode == "append"
            else (),
            "migrations": tuple(
                Migration.model_validate(row) for row in original.get("migrations", [])
            ),
        }
    )
    retained = list(config.requests) if mode == "append" else []
    if value.get("kind", "import") == "generate":
        incoming = generate_requests(
            config, value.get("options", {}), (row.id for row in retained)
        )
    elif value.get("kind", "import") == "import":
        incoming, errors = [], []
        devices = ground_devices(config)
        for number, row in parse_rows(value.get("content"), value.get("format", "csv")):
            try:
                incoming.append(normalize_request(row, devices))
            except (ValueError, TypeError, ValidationError) as error:
                errors.append(f"第 {number} 行：{error}")
                if len(errors) >= 20:
                    break
        if errors:
            raise ValueError("\n".join(errors))
    else:
        raise ValueError("未知批量请求操作")
    requests = validate_requests(config, retained + incoming)
    serialized = [row.model_dump(mode="json") for row in requests]
    scene = {**original, "requests": serialized}
    resources = {
        key: scene[key]
        for key in (
            "physical",
            "physical_link_bindings",
            "ground_owners",
            "satellite_roles",
        )
        if key in scene
    }
    editable = {key: item for key, item in scene.items() if key not in resources}
    # Match the editor's pretty JSON inside its preflight envelope. A batch
    # must remain savable through the existing 10 MB HTTP transport contract.
    preflight = dict(
        experiment_json=json.dumps(editable, ensure_ascii=False, indent=2),
        resource_context=resources,
    )
    if len(json.dumps(preflight, ensure_ascii=False).encode("utf-8")) > MAX_SCENE_BYTES:
        raise ValueError("请求数量使场景超过 10 MB 保存上限，请减少请求或拆分实验")
    return dict(
        requests=serialized,
        imported_count=len(incoming),
        total_count=len(requests),
        mode=mode,
    )
