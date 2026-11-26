"""Register the portable Case Study III inputs in the local web workspace."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory
import uuid


BUNDLED_INPUTS = Path(__file__).resolve().parents[1] / "examples/case-study-iii/inputs.tar.gz"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def verify(bundle):
    manifest = read(bundle / "manifest.json")
    if manifest["schema"] != "orbitserve.case_study_iii.input_bundle.v1":
        raise ValueError("Unsupported Case Study III input bundle")
    files = manifest["sha256"]
    required = {"physical.json", "scene.json", "requests.jsonl", "topology/physical_dataset.json"}
    if not required <= files.keys():
        raise ValueError("Incomplete Case Study III input manifest")
    if "device-routes/index.json" in files and "orbital-routes/index-24h.json" not in files:
        raise ValueError("Prepared device routes require their matching orbital route frames")
    for name, digest in files.items():
        source = (bundle / name).resolve()
        if not source.is_relative_to(bundle) or not source.is_file():
            raise ValueError(f"Missing bundle file: {name}")
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


def install(bundle, *, archive_sha256=None):
    manifest = verify(bundle)
    files = manifest["sha256"]
    physical = read(bundle / "physical.json")
    experiment = read(bundle / "scene.json")
    experiment["requests"] = [
        json.loads(line)
        for line in (bundle / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "viz"), str(root)]
    from runtime_environment import activate

    activate()
    from astermind.builder.physical.physical_dataset import load_dataset
    from paths import WORKSPACE
    from studio import save

    # Imported assets are owned by this workspace. A run never needs the
    # original download folder, which may live on a different machine.
    store = WORKSPACE / "local/results/network"
    store.mkdir(parents=True, exist_ok=True)
    run_id = "physical-case-study-iii-" + uuid.uuid4().hex[:12]
    destination = store / run_id
    with TemporaryDirectory(prefix=".case-study-iii-", dir=store) as temp:
        stage = Path(temp)
        for name in files:
            path = Path(name)
            if path.parts[0] not in ("topology", "orbital-routes", "device-routes"):
                continue
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / path, target)
        load_dataset(stage / "topology", verify_exports=False)
        model = save(WORKSPACE, "physical", physical)
        write(stage / "status.json", dict(id=run_id, state="completed", mode="physical",
                                         name=model["name"], saved_at=model["saved_at"]))
        # Existing asset discovery looks under network/ for an offline dataset.
        (stage / "topology").rename(stage / "network")
        stage.rename(destination)

    write(WORKSPACE / "local/inputs/physical-data" / (model["id"] + ".json"),
          dict(run_id=run_id, physical_model_id=model["id"]))
    experiment["physical_dataset"] = str(destination / "network")
    if "orbital-routes/index-24h.json" in files:
        experiment["fast_orbital_route_frames"] = str(destination / "orbital-routes")
    if "device-routes/index.json" in files:
        experiment["fast_device_route_frames"] = str(destination / "device-routes")
    scene = save(WORKSPACE, "scenes", dict(name=physical["name"],
                                         physical_model_id=model["id"], experiment=experiment))
    if manifest.get("arrivals") == 600:
        write(WORKSPACE / "local/inputs/examples/case-study-iii.json",
              dict(physical_model_id=model["id"], scene_id=scene["id"], archive_sha256=archive_sha256))
    return "/scene?id=" + scene["id"]


def install_archive(archive, *, ensure):
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if ensure:
        root = BUNDLED_INPUTS.parents[2]
        sys.path.insert(0, str(root / "viz"))
        from paths import WORKSPACE
        from studio import physical_catalog

        registration = WORKSPACE / "local/inputs/examples/case-study-iii.json"
        if registration.is_file() and read(registration).get("archive_sha256") == digest:
            preset = next((row for row in physical_catalog(WORKSPACE) if row.get("case_study_iii_scene_id")), None)
            if preset:
                return "/scene?id=" + preset["case_study_iii_scene_id"]
    with TemporaryDirectory(prefix="orbitserve-case-study-iii-") as folder:
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(folder, filter="data")
        bundle = Path(folder)
        if not (bundle / "manifest.json").is_file():
            manifests = list(bundle.glob("*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError("Expected one Case Study III input manifest")
            bundle = manifests[0].parent
        return install(bundle, archive_sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=BUNDLED_INPUTS,
                        help="Input archive or unpacked directory")
    parser.add_argument("--ensure", action="store_true", help="Reuse the registered archive if unchanged")
    parser.add_argument("--base-url", default="http://127.0.0.1:8766",
                        help="URL of the local web service, used in the printed link")
    options = parser.parse_args()
    bundle = options.bundle.resolve()
    route = install(bundle) if bundle.is_dir() else install_archive(bundle, ensure=options.ensure)
    print(options.base_url.rstrip("/") + route)


if __name__ == "__main__":
    main()
