"""Repeat a predeclared independent state grid through the working native SDK.

No model/GPU or production timing. Existing capture CLI measures only native
schedule/update inside its CPU process. Raw retains source, host and phases.
Mixed service is not measured here; any future common decode/mixed profile
must explicitly retain that approximation rather than label it observed.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "profiler_output/microbench/nvidia_a800/serving_accuracy_scheduler_v030_20261107"
PYTHON = "/share_data/yangxinhao/miniconda_yxh/envs/r2r_qwen3_next/bin/python3.10"
SDK = "/share_data/yangxinhao/chip_modeling/midlayer/.venv-vllm030/bin/python"
SOURCE = "/share_data/yangxinhao/chip_modeling/upstream/vllm/v0.30.0"
COUNTS = (1, 8, 32)
LENGTHS = (64, 256)
OUTPUT_COUNTS = (1, 2, 3)
SAMPLES = 240


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "sdk_grid"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    records = []
    for length in LENGTHS:
        for count in COUNTS:
            for output_count in OUTPUT_COUNTS:
                stem = f"n{count}_l{length}_o{output_count}"
                path = target / f"{stem}.json"
                command = [PYTHON, str(ROOT / "scripts/measure_vllm_scheduler_service.py"),
                    "--hardware-id", "nvidia_h200", "--vllm-version", "0.30.0",
                    "--vllm030-python", SDK, "--vllm030-source", SOURCE,
                    "--async-scheduling", "--enable-prefix-caching", "--include-step-records",
                    "--model-config", str(ROOT / "configs/model/qwen3-4b.json"),
                    "--output", str(path), "--sample-count", str(SAMPLES),
                    "--max-batched-tokens", "8192", "--max-requests", str(count),
                    "--prompt-tokens", str(length), "--output-tokens", str(output_count)]
                started = time.monotonic()
                with (target / f"{stem}.log").open("w") as log:
                    code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                record = {"request_count": count, "prompt_tokens": length, "output_tokens": output_count,
                    "command": command, "exit_code": code, "wall_s": time.monotonic() - started}
                records.append(record)
                if code:
                    (target / "failure.json").write_text(json.dumps(records, indent=2) + "\n")
                    raise RuntimeError(f"native SDK cell failed: {stem}; see retained log")
                raw = json.loads(path.read_text())
                assert raw["scheduler"]["measurement_source"]["scheduler_class"] == "AsyncScheduler"
                assert raw["scheduler"]["vllm_version"] == "0.30.0"
                assert raw["measurement"]["control_workload"]["enable_prefix_caching"] is True
                assert all(step["scheduled_request_count"] == count for step in raw["steps"])
                assert [step["phase"] for step in raw["steps"]] == [
                    "prefill" if index % output_count == 0 else "decode" for index in range(SAMPLES)]
                record.update(raw_path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                print(json.dumps(record), flush=True)
    (OUT / "sdk_grid_manifest.json").write_text(json.dumps({"kind": "independent_native_vllm030_scheduler_cpu_grid",
        "production_duration_consumed": False, "model_or_gpu_executed": False,
        "grid": {"counts": COUNTS, "prompt_lengths": LENGTHS, "output_counts": OUTPUT_COUNTS, "samples": SAMPLES},
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
