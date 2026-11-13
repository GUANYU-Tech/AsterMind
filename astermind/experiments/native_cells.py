"""Native device/Cell execution through the standard experiment CLI."""

import json
from pathlib import Path


def run_native_cell_manifest(args):
    from astermind.__main__ import json_default
    from astermind.engine.audit import audit
    from astermind.builder.config import SimulationConfig
    from astermind.engine.simulation import Simulation

    config = SimulationConfig.load(args.manifest)
    artifact = args.artifact_dir or (
        (args.out.parent / (args.out.stem + ".artifacts"))
        if args.out
        else Path("results/experiments/native-cells")
    )
    with Simulation(
        config,
        midlayer_root=args.midlayer_root,
        physical_output_dir=artifact / "physical",
    ) as simulation:
        result = simulation.run()
        result["audit"] = (
            audit(result) if result["status"] == "completed" else {"passed": False}
        )
    payload = result
    if not args.full and args.output_mode != "full":
        artifact.mkdir(parents=True, exist_ok=True)
        complete = artifact / "native_result.json"
        complete.write_text(
            json.dumps(
                result, indent=args.indent, ensure_ascii=False, default=json_default
            )
            + "\n",
            encoding="utf8",
        )
        payload = {
            key: result[key]
            for key in (
                "schema_version",
                "status",
                "requests",
                "measurement",
                "provenance",
                "audit",
            )
        }
        payload["artifacts"] = {"native_result": str(complete.resolve())}
    encoded = (
        json.dumps(
            payload, indent=args.indent, ensure_ascii=False, default=json_default
        )
        + "\n"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf8")
    else:
        print(encoded, end="")
    return int(result["status"] != "completed")
