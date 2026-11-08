#!/usr/bin/env python3
"""Select, resume, and aggregate Serving experiment cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from evaluation.serving.experiments.experiment_plan import (
    git_code_identity,
    load_model_tokenizer,
    materialize_experiment_plan,
    validate_vllm_experiment_environment,
    validate_vllm_scheduler_environment,
)
from evaluation.serving.experiments.experiment_stages import (
    execute_experiment_stages,
    validate_production_trace_outputs,
)
from evaluation.serving.experiments.experiment_manifest import (
    ExperimentManifest,
    ManifestValidationError,
    ProductionTraceSweepCell,
    load_experiment_manifest,
)

from evaluation.json_shape import write_json_file as _write_json


def load_existing_experiment_cell(cell_root: Path) -> dict[str, Any] | None:
    path = cell_root / "command_manifest.json"
    if not path.exists():
        return None
    return _read_json(path)


def experiment_cell_output_exists(cell_root: Path) -> bool:
    return cell_root.exists() and any(cell_root.iterdir())


def experiment_cell_status(command_manifest: dict[str, Any]) -> str:
    manifest_status = command_manifest["status"]
    if not isinstance(manifest_status, str):
        raise ManifestValidationError("command manifest status must be a string")
    status_path = Path(command_manifest["outputs"]["status"])
    if status_path.exists():
        status = _read_json(status_path)
        if "status" not in status or not isinstance(status["status"], str):
            raise ManifestValidationError(
                f"status artifact has no string status: {status_path}")
        artifact_status = status["status"]
        if artifact_status != manifest_status:
            raise ManifestValidationError(
                "command manifest and status artifact disagree: "
                f"{manifest_status!r} != {artifact_status!r}")
        if (artifact_status == "success"
                and "experiment_type" in command_manifest):
            experiment_type = command_manifest["experiment_type"]
            if experiment_type == "production_trace":
                validate_production_trace_outputs(command_manifest)
            else:
                raise ManifestValidationError(
                    f"unknown experiment type: {experiment_type!r}")
        return artifact_status
    if manifest_status in {"dry_run", "running"}:
        return manifest_status
    elif manifest_status in {"success", "failed", "blocked"}:
        raise ManifestValidationError(
            f"terminal experiment state {manifest_status!r} is missing "
            f"status artifact: {status_path}")
    else:
        raise ManifestValidationError(
            f"unknown command manifest status: {manifest_status!r}")


def experiment_resume_decision(status: str | None, *, dry_run: bool) -> str:
    if status == "success":
        return "skip"
    elif dry_run and status == "dry_run":
        return "skip"
    elif status in {"failed", "blocked"}:
        return "rerun"
    elif status is None or status == "running" or (
            status == "dry_run" and not dry_run):
        return "reject"
    else:
        raise ManifestValidationError(f"unknown experiment status: {status!r}")


def validate_resume_manifest_identity(
    command_manifest: dict[str, Any],
    manifest: ExperimentManifest,
    *,
    expected_plan: dict[str, Any] | None = None,
) -> None:
    """Reject resume when an existing cell belongs to another manifest."""
    if manifest.version == 1:
        if ("experiment_manifest" in command_manifest
                or "experiment_type" in command_manifest):
            raise ManifestValidationError(
                "existing cell belongs to a different experiment manifest "
                "version; use --force to replace it")
        return
    elif manifest.version in (2, 3, 4, 5, 6, 7, 8):
        if expected_plan is None:
            raise ManifestValidationError(
                "Production Trace resume requires a plan rematerialized from "
                "the current experiment manifest")
        if "experiment_manifest" not in command_manifest:
            raise ManifestValidationError(
                "existing Production Trace cell has no experiment manifest "
                "identity; use --force to replace it")
        existing_identity = command_manifest["experiment_manifest"]
        if not isinstance(existing_identity, dict):
            raise ManifestValidationError(
                "existing Production Trace cell experiment manifest identity "
                "must be an object")
        required_fields = {"version", "path", "source_sha256"}
        if set(existing_identity) != required_fields:
            raise ManifestValidationError(
                "existing Production Trace cell experiment manifest identity "
                "fields must be "
                f"{sorted(required_fields)}")
        if existing_identity["version"] != manifest.version:
            raise ManifestValidationError(
                "existing cell experiment manifest version disagrees with the "
                "current manifest; use --force to replace it")
        if existing_identity["source_sha256"] != manifest.source_sha256:
            raise ManifestValidationError(
                "existing cell was produced by a different experiment manifest; "
                "use --force to replace it")
        if "provenance" not in command_manifest:
            raise ManifestValidationError(
                "existing Production Trace cell has no code provenance; use "
                "--force to replace it")
        existing_provenance = command_manifest["provenance"]
        if not isinstance(existing_provenance, dict):
            raise ManifestValidationError(
                "existing Production Trace cell code provenance must be an "
                "object")
        required_provenance = {"git_commit", "git_diff_hash"}
        if not required_provenance.issubset(existing_provenance):
            raise ManifestValidationError(
                "existing Production Trace cell code provenance is missing "
                "required fields: "
                f"{sorted(required_provenance - set(existing_provenance))}")
        current_code_identity = git_code_identity(ROOT)
        if existing_provenance["git_commit"] != current_code_identity[
                "git_commit"]:
            raise ManifestValidationError(
                "existing cell was produced by a different code commit; "
                "use --force to replace it")
        if existing_provenance["git_diff_hash"] != current_code_identity[
                "git_diff_hash"]:
            raise ManifestValidationError(
                "existing cell was produced by different worktree contents; "
                "use --force to replace it")
        _validate_resume_static_plan(command_manifest, expected_plan)
        return
    else:
        raise ManifestValidationError(
            f"unsupported experiment manifest version: {manifest.version}")


def _validate_resume_static_plan(
    command_manifest: dict[str, Any],
    expected_plan: dict[str, Any],
) -> None:
    """Bind a resumable Production Trace cell to its canonical plan."""
    static_fields = {
        "version",
        "experiment_type",
        "workflow",
        "cell_id",
        "output_root",
        "input",
        "model",
        "simulation",
        "analysis",
        "environment",
        "validation_status",
        "outputs",
        "commands",
        "command_manifest_path",
    }
    runtime_fields = {
        "experiment_manifest",
        "provenance",
        "status",
        "runtime",
        "failed_stage",
    }
    expected_fields = static_fields | {"provenance", "experiment_manifest"}
    if set(expected_plan) != expected_fields:
        raise AssertionError(
            "rematerialized Production Trace plan fields disagree with the "
            "runner contract: "
            f"expected {sorted(expected_fields)}, got {sorted(expected_plan)}")
    unknown_fields = set(command_manifest) - static_fields - runtime_fields
    if unknown_fields:
        raise ManifestValidationError(
            "existing Production Trace command manifest has unknown fields: "
            f"{sorted(unknown_fields)}")
    missing_fields = static_fields - set(command_manifest)
    if missing_fields:
        raise ManifestValidationError(
            "existing Production Trace command manifest is missing static "
            "plan fields: "
            f"{sorted(missing_fields)}")
    for field in sorted(static_fields):
        existing_value = json.dumps(
            command_manifest[field],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected_value = json.dumps(
            expected_plan[field],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if existing_value != expected_value:
            raise ManifestValidationError(
                "existing Production Trace command manifest static plan "
                "disagrees with the "
                f"current experiment manifest at field {field!r}; use --force "
                "to replace it")


def summarize_experiment_cell(
    command_manifest: dict[str, Any],
    status: str,
    *,
    resumed: bool = False,
) -> dict[str, Any]:
    common = {
        "cell_id": command_manifest["cell_id"],
        "status": status,
        "output_root": command_manifest["output_root"],
        "command_manifest": command_manifest["command_manifest_path"],
    }
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type != "production_trace":
            raise ManifestValidationError(
                f"unknown experiment type: {experiment_type!r}")
        summary = {
            **common,
            "input_kind": command_manifest["input"]["kind"],
            "trace_source_sha256": command_manifest["input"]["trace"][
                "source_sha256"],
            "sweep": command_manifest["outputs"]["sweep"],
            "report_markdown": command_manifest["outputs"][
                "report_markdown"],
        }
    else:
        summary = {
            **common,
            "case_id": command_manifest["case_id"],
            "workload_hash": command_manifest["workload"]["hash"],
        }
    if resumed:
        summary["resumed"] = True
    return summary


def run_experiments(
    argv: list[str] | None = None,
    *,
    subprocess_run=subprocess.run,
    subprocess_popen=subprocess.Popen,
    sleep=time.sleep,
    readiness_probe=None,
    tokenizer_loader=load_model_tokenizer,
) -> dict[str, Any]:
    invocation = _experiment_invocation(argv)
    args = _parse_args(argv)
    mode = _run_mode(args)
    project_root = Path(args.project_root).resolve(strict=False)
    manifest = load_experiment_manifest(args.manifest, project_root=project_root)
    selected = [
        cell for cell in manifest.cells
        if args.only is None or cell.id == args.only
    ]
    if args.only is not None and not selected:
        raise ManifestValidationError(f"unknown cell id for --only: {args.only}")
    if mode == "real_run":
        if manifest.version == 1:
            validate_vllm_experiment_environment()
        elif manifest.version in (2, 3, 4, 5, 6, 7, 8):
            validate_vllm_scheduler_environment()
        else:
            raise ManifestValidationError(
                f"unsupported experiment manifest version: {manifest.version}")

    run_dir = Path(args.run_dir).resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=True)

    cell_summaries = []
    tokenizers: dict[str, Any] = {}
    for cell in selected:
        cell_root = run_dir / cell.id
        experiment_plan: dict[str, Any] | None = None
        existing_cell = load_existing_experiment_cell(cell_root)
        if existing_cell is not None and not args.force:
            if args.resume:
                if manifest.version in (2, 3, 4, 5, 6, 7, 8):
                    if not isinstance(cell, ProductionTraceSweepCell):
                        raise AssertionError(
                            "Production Trace manifest produced a "
                            "non-Production-Trace cell")
                    experiment_plan = materialize_experiment_plan(
                        cell,
                        cell_root,
                        project_root=project_root,
                        invocation=invocation,
                        tokenizer=None,
                    )
                    experiment_plan["experiment_manifest"] = {
                        "version": manifest.version,
                        "path": str(manifest.path.resolve(strict=False)),
                        "source_sha256": manifest.source_sha256,
                    }
                validate_resume_manifest_identity(
                    existing_cell,
                    manifest,
                    expected_plan=experiment_plan,
                )
            existing_status = experiment_cell_status(existing_cell)
            if args.resume:
                decision = experiment_resume_decision(
                    existing_status,
                    dry_run=args.dry_run,
                )
                if decision == "skip":
                    cell_summaries.append(summarize_experiment_cell(
                        existing_cell,
                        "skipped",
                        resumed=True,
                    ))
                    continue
                elif decision == "reject":
                    raise ManifestValidationError(
                        "unresumable existing cell state "
                        f"{existing_status!r} for {cell_root}; use --force or inspect "
                        "and clean up the cell output manually")
                elif decision == "rerun":
                    pass
                else:
                    raise ManifestValidationError(
                        f"unknown resume decision: {decision!r}")
            if not args.resume:
                raise ManifestValidationError(
                    f"cell output already exists: {cell_root}; use --resume or --force")
        elif (
            existing_cell is None
            and experiment_cell_output_exists(cell_root)
            and not args.force
        ):
            raise ManifestValidationError(
                f"cell output already exists: {cell_root}; use --resume or --force")

        if experiment_plan is None:
            if isinstance(cell, ProductionTraceSweepCell):
                tokenizer = None
            else:
                model_key = str(cell.model)
                if model_key not in tokenizers:
                    tokenizers[model_key] = tokenizer_loader(cell.model)
                tokenizer = tokenizers[model_key]
            experiment_plan = materialize_experiment_plan(
                cell,
                cell_root,
                project_root=project_root,
                invocation=invocation,
                tokenizer=tokenizer,
            )
            if manifest.version in (2, 3, 4, 5, 6, 7, 8):
                experiment_plan["experiment_manifest"] = {
                    "version": manifest.version,
                    "path": str(manifest.path.resolve(strict=False)),
                    "source_sha256": manifest.source_sha256,
                }
        result = execute_experiment_stages(
            experiment_plan,
            mode=mode,
            subprocess_run=subprocess_run,
            subprocess_popen=subprocess_popen,
            server_wait_seconds=args.server_wait_seconds,
            readiness_poll_interval_seconds=args.readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
        cell_summaries.append(summarize_experiment_cell(
            result,
            result["status"],
        ))

    summary = {
        "version": manifest.version,
        "mode": mode,
        "manifest": str(Path(args.manifest)),
        "run_dir": str(run_dir),
        "status": _aggregate_status([cell["status"] for cell in cell_summaries]),
        "cells": cell_summaries,
    }
    if manifest.version == 1:
        if manifest.case_manifest_path is None:
            raise AssertionError("v1 experiment manifest lost its case manifest")
        summary["case_manifest"] = str(manifest.case_manifest_path)
    elif manifest.version not in (2, 3, 4, 5, 6, 7, 8):
        raise ManifestValidationError(
            f"unsupported experiment manifest version: {manifest.version}")
    else:
        summary["manifest_sha256"] = manifest.source_sha256
    _write_json(run_dir / "run_summary.json", summary)
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manifest-driven real vLLM E2E experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--only", default=None)
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--resume", action="store_true")
    rerun.add_argument("--force", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--mock-run", action="store_true")
    mode.add_argument("--real-run", action="store_true")
    parser.add_argument(
        "--server-wait-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for vLLM HTTP readiness in real-run mode.",
    )
    parser.add_argument(
        "--readiness-poll-interval-seconds",
        type=float,
        default=1.0,
        help="Seconds between vLLM readiness probes in real-run mode.",
    )
    args = parser.parse_args(argv)
    if args.readiness_poll_interval_seconds <= 0:
        raise ManifestValidationError(
            "--readiness-poll-interval-seconds must be positive")
    return args


def _run_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    elif args.mock_run:
        return "mock_run"
    elif args.real_run:
        return "real_run"
    else:
        raise ManifestValidationError(
            "runner requires --dry-run, --mock-run, or --real-run")


def _experiment_invocation(argv: list[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    return ["scripts/experiments/run_serving_experiments.py", *args]


def _aggregate_status(statuses: list[str]) -> str:
    allowed = {"dry_run", "success", "skipped", "blocked", "failed"}
    unknown = sorted(set(statuses) - allowed)
    if unknown:
        raise ManifestValidationError(
            f"cannot aggregate unknown experiment statuses: {unknown}")
    if not statuses:
        return "empty"
    elif all(status == "dry_run" for status in statuses):
        return "dry_run"
    elif all(status in {"success", "skipped"} for status in statuses):
        return "success"
    elif any(status == "blocked" for status in statuses):
        return "blocked"
    elif any(status == "failed" for status in statuses):
        return "failed"
    else:
        raise ManifestValidationError(
            f"unhandled experiment status combination: {statuses}")




def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    run_experiments(argv)
    return 0
