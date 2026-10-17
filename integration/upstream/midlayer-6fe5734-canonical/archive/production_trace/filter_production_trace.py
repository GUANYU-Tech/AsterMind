from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable

try:
    from openpyxl import load_workbook
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("Missing dependency: openpyxl. Install it with: pip install openpyxl") from exc


SCRIPT_VERSION = "2.3.0"
SEED_NAMESPACE = "production-trace-order-preserving-uniform-v2"
SOURCE_PATTERN = "glm5.2_kimi-k2.6_*_workload_*.xlsx"
SOURCE_NAME_RE = re.compile(
    r"^glm5\.2_kimi-k2\.6_(high|medium|low)_workload_\d{12}_\d+min\.xlsx$"
)
MODEL_TO_PREFIX = {"glm-5.2": "glm-5.2", "kimi-k2.6": "kimi-k2.6"}
DEBUG_WORKLOAD = "medium"

REMOVAL_REASONS = {
    "case1_status1_failure_or_rejection": (
        "Case 1: Status=1; admission/concurrency/HTTP/request failure; no model tokens recorded"
    ),
    "case2_canceled_before_first_output": (
        "Case 2: canceled/canceled_partial_body and Out Token=0; canceled before first output token"
    ),
    "case5_unexpected_state": (
        "Case 5: Finish Reason=unexpected_state; execution state is abnormal"
    ),
    "case6_zero_token_missing_finish": (
        "Case 6: Status=0, Error/Finish blank, Total Token=0; no recorded compute load"
    ),
}

SOURCE_HEADERS = [
    "Query Time Stamp",
    "Model Name",
    "In Token",
    "Out Token",
    "Normal Input Tokens",
    "Cached hit",
    "Total Token",
    "Finish Reason",
    "Inference Time",
    "Error Classify",
    "Status",
    "Ttft",
    "gen_speed",
]

DROP_COLUMNS = {"Model Name", "Finish Reason", "Error Classify", "Status"}
HEADER_RENAMES = {
    "Query Time Stamp": "Query Time Stamp [ms]",
    "In Token": "In Token [tokens]",
    "Out Token": "Out Token [tokens]",
    "Normal Input Tokens": "Normal Input Tokens [tokens]",
    "Cached hit": "Cached hit [tokens]",
    "Total Token": "Total Token [tokens]",
    "Inference Time": "Inference Time [ms]",
    "Ttft": "Ttft [ms]",
    "gen_speed": "gen_speed [token/s]",
}
OUTPUT_HEADERS = [
    HEADER_RENAMES.get(name, name) for name in SOURCE_HEADERS if name not in DROP_COLUMNS
]


@dataclass(frozen=True)
class TraceRow:
    timestamp: datetime
    source_ordinal: int
    values: tuple[str, ...]


def log(message: str = "") -> None:
    print(message, flush=True)


def count_text(value: int) -> str:
    return f"{value:,}"


def log_removal_counts(counts: Counter[str] | dict[str, int], indent: str = "  ") -> None:
    total = sum(counts.values())
    if total == 0:
        log(f"{indent}REMOVED: 0")
        return
    log(f"{indent}REMOVED: {count_text(total)}")
    for reason, description in REMOVAL_REASONS.items():
        amount = int(counts.get(reason, 0))
        if amount:
            log(f"{indent}  - {count_text(amount)} requests | {description}")


def parse_args() -> argparse.Namespace:
    default_source = Path.home() / "Desktop" / "production_trace"
    parser = argparse.ArgumentParser(
        description=(
            "Build simulator-ready CSV traces from the three production XLSX files. "
            "The source workbooks are read-only and never modified."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help=f"Directory containing the three source XLSX files (default: {default_source})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated CSV files and the JSON report (default: script directory)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(value)
    return str(value)


def parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    text = serialize_cell(value).strip()
    if not text:
        raise ValueError("empty Query Time Stamp")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"unsupported Query Time Stamp: {text!r}") from exc


def numeric(value: str) -> float:
    text = value.strip()
    if not text:
        return 0.0
    result = float(text)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def floor_second(value: datetime) -> datetime:
    return value.replace(microsecond=0)


def classify_for_removal(values: tuple[str, ...], indexes: dict[str, int]) -> str | None:
    status = numeric(values[indexes["Status"]])
    error = values[indexes["Error Classify"]].strip()
    finish = values[indexes["Finish Reason"]].strip()
    out_tokens = numeric(values[indexes["Out Token"]])
    total_tokens = numeric(values[indexes["Total Token"]])

    if status == 1:
        return "case1_status1_failure_or_rejection"
    if finish == "unexpected_state":
        return "case5_unexpected_state"
    if error in {"canceled_partial_body", "canceled"} and out_tokens == 0:
        return "case2_canceled_before_first_output"
    if status == 0 and not error and not finish and total_tokens == 0:
        return "case6_zero_token_missing_finish"
    return None


def project_non_time_values(values: tuple[str, ...], indexes: dict[str, int]) -> list[str]:
    projected: list[str] = []
    for name in SOURCE_HEADERS[1:]:
        if name not in DROP_COLUMNS:
            projected.append(values[indexes[name]])
    return projected


def seed_for(workload: str, model: str, second: datetime) -> int:
    payload = (
        f"{SEED_NAMESPACE}|{workload}|{model}|{second.isoformat(sep=' ')}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def open_uniform(rng: random.Random) -> float:
    return (rng.getrandbits(53) + 1) / ((1 << 53) + 1)


def ordered_uniform_offsets(
    workload: str,
    model: str,
    second: datetime,
    count: int,
    is_first_second: bool,
) -> list[float]:
    rng = random.Random(seed_for(workload, model, second))
    if is_first_second:
        return [0.0, *sorted(open_uniform(rng) for _ in range(count - 1))]
    return sorted(open_uniform(rng) for _ in range(count))


def discover_sources(source_dir: Path) -> list[tuple[Path, str]]:
    sources = sorted(source_dir.glob(SOURCE_PATTERN))
    if len(sources) != 3:
        raise RuntimeError(f"expected exactly 3 source XLSX files, found {len(sources)} in {source_dir}")
    discovered: list[tuple[Path, str]] = []
    seen_workloads: set[str] = set()
    for source in sources:
        match = SOURCE_NAME_RE.match(source.name)
        if not match:
            raise RuntimeError(f"unexpected source filename: {source.name}")
        workload = match.group(1)
        if workload in seen_workloads:
            raise RuntimeError(f"duplicate workload source: {workload}")
        seen_workloads.add(workload)
        discovered.append((source, workload))
    if seen_workloads != {"high", "medium", "low"}:
        raise RuntimeError(f"expected high/medium/low sources, found {sorted(seen_workloads)}")
    return discovered


def load_and_filter_source(
    source: Path,
    workload: str,
) -> tuple[dict[str, list[TraceRow]], dict[str, object]]:
    warnings.filterwarnings("ignore", message="Workbook contains no default style")
    workbook = load_workbook(source, read_only=True, data_only=False)
    if len(workbook.worksheets) != 1:
        workbook.close()
        raise RuntimeError(f"expected one worksheet in {source.name}")
    worksheet = workbook.worksheets[0]
    iterator = worksheet.iter_rows(values_only=True)
    try:
        raw_header = next(iterator)
    except StopIteration as exc:
        workbook.close()
        raise RuntimeError(f"empty workbook: {source.name}") from exc
    headers = [serialize_cell(value) for value in raw_header]
    if headers != SOURCE_HEADERS:
        workbook.close()
        raise RuntimeError(
            f"unexpected header in {source.name}\nexpected={SOURCE_HEADERS}\nactual={headers}"
        )
    indexes = {name: index for index, name in enumerate(headers)}
    groups: dict[str, list[TraceRow]] = {model: [] for model in MODEL_TO_PREFIX}
    original_by_model = Counter()
    kept_by_model = Counter()
    removed_by_reason = Counter()
    removed_by_model: dict[str, Counter[str]] = {
        model: Counter() for model in MODEL_TO_PREFIX
    }

    for ordinal, raw_values in enumerate(iterator):
        values = tuple(serialize_cell(value) for value in raw_values)
        if len(values) != len(SOURCE_HEADERS):
            workbook.close()
            raise RuntimeError(f"column count mismatch at row {ordinal + 2} in {source.name}")
        model = values[indexes["Model Name"]].strip()
        if model not in groups:
            workbook.close()
            raise RuntimeError(f"unexpected model {model!r} at row {ordinal + 2} in {source.name}")
        original_by_model[model] += 1
        removal_reason = classify_for_removal(values, indexes)
        if removal_reason is not None:
            removed_by_reason[removal_reason] += 1
            removed_by_model[model][removal_reason] += 1
            continue
        timestamp = parse_timestamp(raw_values[indexes["Query Time Stamp"]])
        groups[model].append(TraceRow(timestamp, ordinal, values))
        kept_by_model[model] += 1
    workbook.close()

    for model, rows in groups.items():
        if not rows:
            raise RuntimeError(f"no retained rows for {model} in {source.name}")
        rows.sort(key=lambda row: (row.timestamp, row.source_ordinal))

    original_rows = sum(original_by_model.values())
    kept_rows = sum(kept_by_model.values())
    removed_rows = sum(removed_by_reason.values())
    if original_rows != kept_rows + removed_rows:
        raise AssertionError(f"row accounting mismatch in {source.name}")

    return groups, {
        "workload": workload,
        "worksheet": worksheet.title,
        "headers": headers,
        "original_rows": original_rows,
        "kept_rows": kept_rows,
        "removed_rows": removed_rows,
        "original_rows_by_model": dict(original_by_model),
        "kept_rows_by_model": dict(kept_by_model),
        "removed_by_reason": dict(removed_by_reason),
        "removed_by_model": {
            model: dict(counts) for model, counts in removed_by_model.items()
        },
    }


def duration_minutes(rows: list[TraceRow]) -> int:
    start_minute = floor_second(rows[0].timestamp).replace(second=0)
    end_second = floor_second(rows[-1].timestamp)
    return int((end_second - start_minute).total_seconds() // 60) + 1


def output_filename(workload: str, model: str, rows: list[TraceRow]) -> str:
    duration = duration_minutes(rows)
    return f"{MODEL_TO_PREFIX[model]}/{workload}_workload_{duration}min.csv"


def write_main_csv(
    path: Path,
    workload: str,
    model: str,
    rows: list[TraceRow],
    indexes: dict[str, int],
) -> dict[str, object]:
    first_second = floor_second(rows[0].timestamp)
    last_relative_ms = 0.0
    first_second_requests = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_HEADERS)
        grouped = itertools.groupby(rows, key=lambda row: floor_second(row.timestamp))
        for group_number, (second, grouped_rows) in enumerate(grouped):
            same_second_rows = list(grouped_rows)
            if group_number == 0:
                first_second_requests = len(same_second_rows)
            base_ms = (second - first_second).total_seconds() * 1000.0
            offsets = ordered_uniform_offsets(
                workload, model, second, len(same_second_rows), group_number == 0
            )
            for row, offset in zip(same_second_rows, offsets):
                relative_ms = base_ms + offset * 1000.0
                writer.writerow(
                    [f"{relative_ms:.6f}", *project_non_time_values(row.values, indexes)]
                )
                last_relative_ms = relative_ms
    return {
        "rows": len(rows),
        "original_time_min": rows[0].timestamp.isoformat(sep=" "),
        "original_time_max": rows[-1].timestamp.isoformat(sep=" "),
        "duration_minutes": duration_minutes(rows),
        "first_second_requests": first_second_requests,
        "first_relative_ms": 0.0,
        "last_relative_ms": last_relative_ms,
    }


def verify_main_csv(
    path: Path,
    rows: list[TraceRow],
    indexes: dict[str, int],
) -> dict[str, object]:
    first_second = floor_second(rows[0].timestamp)
    previous_time: float | None = None
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != OUTPUT_HEADERS:
            raise AssertionError(f"header mismatch in {path.name}")
        for expected, actual in itertools.zip_longest(rows, reader):
            if expected is None or actual is None:
                raise AssertionError(f"row count mismatch in {path.name}")
            count += 1
            if len(actual) != len(OUTPUT_HEADERS):
                raise AssertionError(f"column count mismatch at row {count + 1} in {path.name}")
            if actual[1:] != project_non_time_values(expected.values, indexes):
                raise AssertionError(f"non-time data changed at row {count + 1} in {path.name}")
            relative_ms = float(actual[0])
            if count == 1 and relative_ms != 0.0:
                raise AssertionError(f"first timestamp is not zero in {path.name}")
            if previous_time is not None and relative_ms <= previous_time:
                raise AssertionError(f"timestamps are not strictly increasing in {path.name}")
            expected_bucket = int((floor_second(expected.timestamp) - first_second).total_seconds())
            if math.floor(relative_ms / 1000.0) != expected_bucket:
                raise AssertionError(f"original second bucket changed at row {count + 1} in {path.name}")
            previous_time = relative_ms

    if count != len(rows):
        raise AssertionError(f"row count mismatch in {path.name}")
    if any(name in header for name in DROP_COLUMNS):
        raise AssertionError(f"dropped column remains in {path.name}")
    return {
        "header_exact": True,
        "row_count_exact": True,
        "non_time_fields_exact": True,
        "strictly_increasing": True,
        "original_second_buckets_preserved": True,
        "first_relative_ms_zero": True,
        "dropped_columns_absent": True,
        "sha256": sha256_file(path),
    }


def create_debug_csv(main_path: Path, debug_path: Path) -> dict[str, object]:
    selected: list[list[str]] = []
    with main_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.reader(source_handle)
        header = next(reader)
        if header != OUTPUT_HEADERS:
            raise AssertionError(f"main header mismatch in {main_path.name}")
        for row in reader:
            if float(row[0]) >= 1000.0:
                break
            selected.append(row)
    if not selected:
        raise AssertionError(f"no first-second rows in {main_path.name}")

    with debug_path.open("w", encoding="utf-8-sig", newline="") as target_handle:
        writer = csv.writer(target_handle, lineterminator="\n")
        writer.writerow(OUTPUT_HEADERS)
        writer.writerows(selected)

    with debug_path.open("r", encoding="utf-8-sig", newline="") as check_handle:
        check_rows = list(csv.reader(check_handle))
    if check_rows != [OUTPUT_HEADERS, *selected]:
        raise AssertionError(f"debug verification failed in {debug_path.name}")
    if float(selected[0][0]) != 0.0 or any(float(row[0]) >= 1000.0 for row in selected):
        raise AssertionError(f"debug time window mismatch in {debug_path.name}")
    return {
        "rows": len(selected),
        "exact_main_first_second_prefix": True,
        "first_relative_ms": float(selected[0][0]),
        "last_relative_ms": float(selected[-1][0]),
        "sha256": sha256_file(debug_path),
    }


def safe_stale_names(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    generated = report.get("generated_csv_files", [])
    if not isinstance(generated, list):
        return set()
    safe_names: set[str] = set()
    for name in generated:
        if not isinstance(name, str) or not name.endswith(".csv"):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        safe_names.add(relative.as_posix())
    return safe_names


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(source_dir)
    source_hashes_before = {source.name: sha256_file(source) for source, _ in sources}
    report_path = output_dir / "trace_processing_report.json"
    previous_generated = safe_stale_names(report_path)

    log("=" * 76)
    log(f"Production trace processor v{SCRIPT_VERSION}")
    log(f"Source directory: {source_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Source files found: {len(sources)}")
    log("Removal policy:")
    for description in REMOVAL_REASONS.values():
        log(f"  - {description}")
    log("Retention policy: keep all other requests to replay observed In/Out Token compute load")
    log("Columns removed from output: Model Name, Finish Reason, Error Classify, Status")
    log("=" * 76)

    global_removed = Counter()
    global_original = 0
    global_kept = 0
    report: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "seed_namespace": SEED_NAMESPACE,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "output_headers": OUTPUT_HEADERS,
        "dropped_columns": sorted(DROP_COLUMNS),
        "removal_policy": REMOVAL_REASONS,
        "sources": {},
        "main_files": {},
        "debug_files": {},
    }
    generated_names: list[str] = []

    with tempfile.TemporaryDirectory(prefix=".trace-build-", dir=output_dir) as temp_name:
        staging_dir = Path(temp_name)
        indexes = {name: index for index, name in enumerate(SOURCE_HEADERS)}
        medium_main_names: list[str] = []

        for source, workload in sources:
            log()
            log(f"[{workload.upper()}] Reading: {source.name}")
            groups, source_report = load_and_filter_source(source, workload)
            source_report["sha256_before"] = source_hashes_before[source.name]
            report["sources"][source.name] = source_report
            global_original += int(source_report["original_rows"])
            global_kept += int(source_report["kept_rows"])
            global_removed.update(source_report["removed_by_reason"])

            log(
                f"[{workload.upper()}] ORIGINAL {count_text(int(source_report['original_rows']))} | "
                f"KEPT {count_text(int(source_report['kept_rows']))} | "
                f"REMOVED {count_text(int(source_report['removed_rows']))}"
            )
            log_removal_counts(source_report["removed_by_reason"])

            original_by_model = source_report["original_rows_by_model"]
            kept_by_model = source_report["kept_rows_by_model"]
            removed_by_model = source_report["removed_by_model"]
            for model in MODEL_TO_PREFIX:
                model_removed = removed_by_model[model]
                log(
                    f"  MODEL {model}: ORIGINAL {count_text(int(original_by_model[model]))} | "
                    f"KEPT {count_text(int(kept_by_model[model]))} | "
                    f"REMOVED {count_text(sum(model_removed.values()))}"
                )
                log_removal_counts(model_removed, indent="    ")

            for model, rows in groups.items():
                filename = output_filename(workload, model, rows)
                if filename in generated_names:
                    raise RuntimeError(f"duplicate output filename: {filename}")
                staged_path = staging_dir / filename
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                transform = write_main_csv(staged_path, workload, model, rows, indexes)
                validation = verify_main_csv(staged_path, rows, indexes)
                report["main_files"][filename] = {
                    "source": source.name,
                    "workload": workload,
                    "model_removed_from_output": model,
                    "transform": transform,
                    "validation": validation,
                }
                generated_names.append(filename)
                log(
                    f"  OUTPUT {filename}: {count_text(len(rows))} requests | "
                    f"first second {count_text(int(transform['first_second_requests']))} | "
                    f"source time {transform['original_time_min']} to {transform['original_time_max']}"
                )
                if workload == DEBUG_WORKLOAD:
                    medium_main_names.append(filename)

        if len(generated_names) != 6 or len(medium_main_names) != 2:
            raise AssertionError("expected 6 main files and 2 medium main files")
        if global_original != global_kept + sum(global_removed.values()):
            raise AssertionError("global row accounting mismatch")

        for main_name in sorted(medium_main_names):
            main_relative = Path(main_name)
            debug_name = (main_relative.parent / "debug_workload_1s.csv").as_posix()
            staged_main = staging_dir / main_name
            staged_debug = staging_dir / debug_name
            staged_debug.parent.mkdir(parents=True, exist_ok=True)
            report["debug_files"][debug_name] = {
                "source_main": main_name,
                "validation": create_debug_csv(staged_main, staged_debug),
            }
            debug_rows = int(report["debug_files"][debug_name]["validation"]["rows"])
            log(f"  DEBUG {debug_name}: {count_text(debug_rows)} requests (main-file first second)")
            generated_names.append(debug_name)

        if len(generated_names) != 8:
            raise AssertionError(f"expected 8 generated CSV files, got {len(generated_names)}")

        source_hashes_after = {source.name: sha256_file(source) for source, _ in sources}
        if source_hashes_before != source_hashes_after:
            raise AssertionError("one or more source XLSX files changed")

        report["global_validation"] = {
            "source_file_count": 3,
            "main_file_count": 6,
            "debug_file_count": 2,
            "source_rows": global_original,
            "retained_main_rows": global_kept,
            "removed_rows": sum(global_removed.values()),
            "removed_by_reason": dict(global_removed),
            "rows_accounted_for": global_original == global_kept + sum(global_removed.values()),
            "source_sha256_unchanged": True,
            "all_status1_removed": True,
            "all_zero_output_cancellations_removed": True,
            "all_unexpected_state_removed": True,
            "all_zero_token_missing_finish_removed": True,
            "dropped_columns_absent": True,
        }
        report["generated_csv_files"] = sorted(generated_names)

        log()
        log("[GLOBAL] REMOVAL SUMMARY")
        log_removal_counts(global_removed)
        log(
            f"[GLOBAL] ORIGINAL {count_text(global_original)} | "
            f"KEPT {count_text(global_kept)} | "
            f"REMOVED {count_text(sum(global_removed.values()))}"
        )
        log(
            "[VERIFY] PASS: row accounting, source hashes, dropped columns, strict time order, "
            "original second buckets, and debug first-second prefix"
        )

        staged_report = staging_dir / report_path.name
        staged_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for name in generated_names:
            target_path = output_dir / name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_dir / name, target_path)
        os.replace(staged_report, report_path)

    stale_names = previous_generated - set(generated_names)
    for name in sorted(stale_names):
        stale_path = output_dir / name
        if stale_path.is_file():
            stale_path.unlink()
            parent = stale_path.parent
            while parent != output_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

    log(f"[DONE] Generated 6 main CSVs and 2 debug CSVs; report: {report_path}")
    log("[RESULT_JSON]")
    print(json.dumps({
        "source_rows": global_original,
        "retained_main_rows": global_kept,
        "removed_rows": sum(global_removed.values()),
        "removed_by_reason": dict(global_removed),
        "generated_csv_files": sorted(generated_names),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
