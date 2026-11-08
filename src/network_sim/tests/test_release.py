import tempfile
from pathlib import Path
import unittest
import zipfile

from network_sim.package_release import PUBLIC_FILES, HANDOFF_FILES, package


class ReleaseTests(unittest.TestCase):
    def test_public_allowlist_contains_only_basic_configuration(self):
        configs = {p for p in PUBLIC_FILES if p.startswith("examples/")}
        self.assertEqual(configs, {"examples/public_basic.json", "examples/public_transfers.json"})
        self.assertFalse(any(p.startswith(("private/", "experimental/", "runs/", "build/")) for p in PUBLIC_FILES))

    def test_release_archive_matches_allowlist(self):
        # Keep scratch files in the known writable build directory.
        build = Path(__file__).resolve().parents[1] / "build"
        build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build) as directory:
            output = package(Path(directory) / "public.zip")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {f"network_sim/{p}" for p in PUBLIC_FILES} |
                                 {"network_sim/PUBLIC_MANIFEST.json"})

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "build/network_native.dll").is_file(),
                         "Windows handoff archive requires a locally built Windows DLL")
    def test_handoff_only_adds_synthetic_example_and_windows_binary(self):
        self.assertEqual(set(HANDOFF_FILES), {"examples/handoff_dynamic.json", "build/network_native.dll"})
        build = Path(__file__).resolve().parents[1] / "build"
        with tempfile.TemporaryDirectory(dir=build) as directory:
            with zipfile.ZipFile(package(Path(directory) / "handoff.zip", handoff=True)) as archive:
                self.assertEqual(set(archive.namelist()), {f"network_sim/{p}" for p in PUBLIC_FILES + HANDOFF_FILES} |
                                 {"network_sim/PUBLIC_MANIFEST.json"})
