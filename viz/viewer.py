"""Self-contained diagnostic view of backend network snapshots and route facts."""

import json
from pathlib import Path


def write_network_viewer(replay: dict, destination: str | Path, *, runtime_path=None) -> None:
    replay = dict(replay)

    data = json.dumps(replay, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Never allow an input ID or provenance string to terminate the JSON script.
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
    Path(destination).write_text(template.replace("__REPLAY_JSON__", data), encoding="utf-8")
