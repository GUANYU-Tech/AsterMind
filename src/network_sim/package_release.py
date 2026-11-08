"""Create a local preview from an explicit allowlist, never from recursive globbing."""

from pathlib import Path
import argparse
import json
import hashlib
import zipfile


PUBLIC_FILES = (
    "__init__.py", "__main__.py", "api.py", "model.py", "topology.py", "runtime.py", "link_policy.py",
    "physical/__init__.py", "physical/model.py", "physical/topology.py", "physical/link_policy.py",
    "comm/__init__.py", "comm/model.py", "comm/runtime.py",
    "build_native.py", "package_release.py", "requirements.txt", "README.md", ".gitignore",
    "physical/_vendor/__init__.py", "physical/_vendor/astermind_interfaces.py",
    "physical/_vendor/orbitserve_link_selection.py",
    "comm/_vendor/flowsim/include/flowsim/flowsim.hh",
    "comm/_vendor/flowsim/src/topology.cc", "comm/_vendor/flowsim/src/event_simulator.cc",
    "comm/_vendor/flowsim/src/simulator.cc", "comm/_vendor/flowsim/tests/test_flowsim.cc",
    "comm/native/bridge.cc", "tests/test_network.py", "tests/test_release.py",
    "tests/test_link_policy.py",
    "examples/public_basic.json", "examples/public_transfers.json",
    "docs/architecture.md", "docs/architecture.html", "docs/interfaces.md",
    "docs/architecture.png", "docs/devices.png", "docs/sharing.png",
    "docs/release.md", "docs/reuse.md", "docs/link-policy-reference.md",
    "docs/vendor_changes.patch", "docs/source_manifest.json",
    "docs/compute-handoff.md", "docs/link_policy.png",
)

HANDOFF_FILES = ("examples/handoff_dynamic.json", "build/network_native.dll")


def package(output: Path, handoff=False):
    root = Path(__file__).resolve().parent
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate every input before opening the destination.
    files = PUBLIC_FILES + (HANDOFF_FILES if handoff else ())
    for relative in files:
        if not (root / relative).is_file():
            raise FileNotFoundError(relative)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(root / relative, f"network_sim/{relative}")
        archive.writestr("network_sim/PUBLIC_MANIFEST.json", json.dumps({
            "version": "0.2", "kind": "compute_handoff" if handoff else "local_release_preview",
            "files": list(files),
            "file_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
            "native_binary_platform": "Windows x64" if handoff else None,
            "example_scope": "synthetic minimal example only",
            "license_confirmation": "required before actual publication",
        }, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handoff", action="store_true", help="include synthetic dynamic example and Windows x64 DLL")
    args = parser.parse_args()
    print(package(args.output, handoff=args.handoff))


if __name__ == "__main__":
    main()
