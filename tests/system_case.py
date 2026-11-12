"""Independent physical/cost fixture inputs; no experiment execution wrapper."""
import json
from pathlib import Path
from astermind.builder.physical.network_inputs import snapshot_network_input
ROOT = Path(__file__).resolve().parents[1]
def physical_case():
    path = ROOT / "tests/fixtures/manifests/coupled_physical.json"
    value = json.loads(path.read_text())
    network = value["physical_network"]
    base = path.parent
    if isinstance(network, str):
        source = base / network
        base = source.parent
        network = json.loads(source.read_text())
    value["physical_network"] = snapshot_network_input(network, base)
    return value
