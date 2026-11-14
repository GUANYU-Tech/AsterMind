"""Run one unchanged V1 example and keep evidence inside viz/.runtime."""
from pathlib import Path
import json
import time
import traceback

def main():
    from runtime_environment import activate
    from paths import RUNTIME
    root = activate()
    from astermind.builder.config import SimulationConfig
    from astermind.engine.simulation import Simulation
    from astermind.__main__ import json_default
    destination = RUNTIME / 'verification/backend'
    destination.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig.load(root / 'examples/native/tp2_two_satellites.json')
    start = time.monotonic()
    try:
        with Simulation(config, physical_output_dir=destination / 'physical',
                        startup_progress=lambda p: print(json.dumps(p, default=json_default), flush=True)) as simulation:
            result = simulation.run()
        (destination / 'result.json').write_text(json.dumps(result, default=json_default))
        status = {'status': result['status'], 'elapsed_s': time.monotonic() - start,
                  'request_count': len(result['requests']), 'schema_version': result['schema_version']}
    except Exception as error:
        (destination / 'error.log').write_text(traceback.format_exc())
        status = {'status': 'failed', 'error': str(error), 'elapsed_s': time.monotonic() - start}
    (destination / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)

if __name__ == '__main__':
    main()
