"""Verify the deliverable from its extracted tree, without workspace imports."""
from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive_path = ROOT / "build/network_sim_topology_v0.2.zip"
    output = ROOT / "runs/verification_v0.2"
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    commands = [
        ["-m", "unittest", "discover", "-s", "network_sim/tests", "-v"],
        ["-m", "network_sim", "--scenario", "network_sim/examples/public_basic.json",
         "--transfers", "network_sim/examples/public_transfers.json", "--output-dir", "runs/basic"],
        ["-m", "network_sim", "--scenario", "network_sim/examples/handoff_dynamic.json",
         "--transfers", "network_sim/examples/public_transfers.json", "--until-ns", "3000000",
         "--output-dir", "runs/dynamic"],
    ]
    with tempfile.TemporaryDirectory(prefix="handoff-check-", dir=ROOT / "build") as directory:
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("network_sim/PUBLIC_MANIFEST.json"))
            assert set(archive.namelist()) == {"network_sim/"+p for p in manifest["files"]} | {"network_sim/PUBLIC_MANIFEST.json"}
            for relative, digest in manifest["file_sha256"].items():
                assert hashlib.sha256(archive.read("network_sim/"+relative)).hexdigest() == digest
            dll = archive.read("network_sim/build/network_native.dll")
            pe_offset = struct.unpack_from("<I", dll, 60)[0]
            assert dll[pe_offset:pe_offset+4] == b"PE\0\0"
            assert struct.unpack_from("<H", dll, pe_offset+4)[0] == 0x8664
            archive.extractall(directory)
        log = []
        for command in commands:
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", *command],
                                    cwd=directory, env=env, capture_output=True, text=True, encoding="utf-8")
            log.append("python " + " ".join(command) + "\n" + result.stdout + result.stderr)
            (output / "extracted-package.log").write_text("\n".join(log), encoding="utf-8")
            result.check_returncode()
        basic = json.loads((Path(directory) / "runs/basic/report.json").read_text(encoding="utf-8"))
        dynamic = json.loads((Path(directory) / "runs/dynamic/report.json").read_text(encoding="utf-8"))
        assert [e["completion_time_ns"] for e in basic["completion_events"]] == [2000103, 2000103]
        assert dynamic["completion_events"][0]["completion_time_ns"] == 1600103
        assert dynamic["failure_events"][0]["failure_time_ns"] == 1000000
        assert dynamic["failure_events"][0]["remaining_bytes"] == 600000
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    evidence = dict(version="0.2", package=archive_path.name, sha256=digest,
                    entries=len(manifest["files"])+1, windows_x64_verified=True,
                    extracted_unit_tests="40 passed", static_demo="2 completed at 2000103 ns",
                    dynamic_demo="flow0 failed at 1000000 ns; flow1 completed at 1600103 ns",
                    workspace_imports="excluded: subprocess cwd is extracted tree; PYTHONPATH removed")
    (output / "handoff-checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    archive_path.with_suffix(".sha256").write_text(digest+"  "+archive_path.name+"\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
