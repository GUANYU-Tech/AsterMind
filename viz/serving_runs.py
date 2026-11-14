"""Full shell-serving experiments on the original local console."""

import gzip
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import threading

from paths import WORKSPACE as ROOT
FOLDER = ROOT / "local/results/full-serving"
JOBS = {}
LOCK = threading.Lock()


def identity(users, parallel, backend):
    if (
        users not in ("single", "multi")
        or parallel not in ("tp_pp", "ep")
        or backend not in ("flowsim", "ns3")
    ):
        raise ValueError("unknown serving experiment")
    return users + "-" + parallel + "-" + backend


@lru_cache(maxsize=1)
def physical(parallel):
    if parallel not in ("tp_pp", "ep"):
        raise ValueError("unknown deployment")
    data = json.loads(
        (FOLDER / ("physical-v1-" + parallel + ".json")).read_text(encoding="utf8")
    )
    data["time"] = data["trace"]["trace_manifest"]["time"]
    data.pop("trace")
    return data


def index(parallel):
    data = physical(parallel)
    return {
        **{
            k: data[k]
            for k in (
                "network",
                "time",
                "sun_positions",
                "ranks_per_replica",
                "shell_satellites",
                "replicas",
            )
        },
        "frame_times": [f["time_s"] for f in data["frames"]],
    }


def frame(parallel, number):
    frames = physical(parallel)["frames"]
    if not 0 <= number < len(frames):
        raise ValueError("frame outside horizon")
    return gzip.compress(
        json.dumps(frames[number], separators=(",", ":")).encode(), compresslevel=1
    )


def result(users, parallel, backend):
    path = FOLDER / (identity(users, parallel, backend) + ".json")
    if not path.is_file():
        raise ValueError("该推理场景尚未运行，请运行或切换已有结果")
    return json.loads(path.read_text(encoding="utf8"))


def status(users, parallel, backend):
    key = identity(users, parallel, backend)
    with LOCK:
        value = dict(JOBS.get(key, {}))
    return value or {
        "state": "completed" if (FOLDER / (key + ".json")).is_file() else "missing"
    }


def compare(users, parallel):
    rows = {}
    for backend in ("flowsim", "ns3"):
        try:
            data = result(users, parallel, backend)
        except ValueError:
            continue
        timings = [r["timing"] for r in data["records"]]
        rows[backend] = {
            "completed": data["summary"]["completed_request_count"],
            "mean_e2e_s": sum(r["e2e_s"] for r in timings) / len(timings),
            "mean_ttft_s": sum(r["ttft_s"] for r in timings) / len(timings),
            "payload_hop_bytes": data["payload_hop_bytes"],
            "native_counters": data["provenance"].get("native_counters"),
            "primitive_count": data["primitive_count"],
            "replicas_used": data["replicas_used"],
        }
    return rows


def start(users, parallel, backend):
    raise ValueError('此历史预设依赖旧 FlowSim/ns-3 或旧部署契约，V1 不支持等价运行。请在业务与部署页面生成 V1 原生场景。')
    key = identity(users, parallel, backend)
    with LOCK:
        if JOBS.get(key, {}).get("state") == "running":
            raise ValueError("该场景正在运行")
        JOBS[key] = {"state": "running"}

    def work():
        try:
            FOLDER.mkdir(parents=True, exist_ok=True)
            temporary = FOLDER / (key + ".pending.json")
            from native_environment import inference_runtime
            python, environment = inference_runtime()
            process = subprocess.run(
                [
                    python,
                    "-B",
                    str(ROOT / "examples/astermind_joint/full_serving.py"),
                    "--users",
                    users,
                    "--parallel",
                    parallel,
                    "--backend",
                    backend,
                    "--output",
                    str(temporary),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            if process.returncode:
                raise ValueError((process.stderr or process.stdout)[-4000:])
            temporary.replace(FOLDER / (key + ".json"))
            with LOCK:
                JOBS[key] = {"state": "completed"}
        except Exception as error:
            with LOCK:
                JOBS[key] = {"state": "failed", "error": str(error)}

    threading.Thread(target=work, daemon=True).start()
    return {"state": "running"}
